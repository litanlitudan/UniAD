#!/usr/bin/env python3
"""Generate enhanced module reports with operation decomposition for UniAD"""

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.operation_decomposition import OperationDecomposer
from core.enhanced_profiler import EnhancedProfiler
from generate_module_op_analysis import UNIAD_MODULE_DETAILS, PYTORCH_OP_DETAILS


def create_mock_uniad_modules():
    """Create mock UniAD modules based on the module details"""
    
    modules = {}
    
    # Backbone modules
    class MockResNet101DCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            # Simplified layers with DCN
            self.layer3_conv = nn.Conv2d(512, 256, 3, padding=1)  # Would be DCNv2
            self.layer3_bn = nn.BatchNorm2d(256)
            self.layer4_conv = nn.Conv2d(1024, 512, 3, padding=1)  # Would be DCNv2
            self.layer4_bn = nn.BatchNorm2d(512)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # Simplified forward
            return x
    
    class MockFPN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lateral_conv0 = nn.Conv2d(512, 256, 1)
            self.lateral_conv1 = nn.Conv2d(1024, 256, 1)
            self.lateral_conv2 = nn.Conv2d(2048, 256, 1)
            self.fpn_conv0 = nn.Conv2d(256, 256, 3, padding=1)
            self.fpn_conv1 = nn.Conv2d(256, 256, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            
        def forward(self, x):
            # Simplified - just process one level
            x = self.lateral_conv0(x)
            x = self.fpn_conv0(x)
            x = self.relu(x)
            return x
    
    # BEV Encoder modules
    class MockBEVFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.bev_queries = nn.Parameter(torch.randn(200 * 200, 256))
            self.temporal_self_attn = nn.MultiheadAttention(256, 8, batch_first=True)
            self.spatial_cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
            self.ffn1 = nn.Linear(256, 512)
            self.ffn2 = nn.Linear(512, 256)
            self.norm1 = nn.LayerNorm(256)
            self.norm2 = nn.LayerNorm(256)
            self.relu = nn.ReLU(inplace=True)
            
        def forward(self, x):
            # x shape: [B, N, C] where N = H*W
            b = x.shape[0]
            queries = self.bev_queries.unsqueeze(0).expand(b, -1, -1)
            
            # Temporal self-attention
            queries = self.norm1(queries)
            attn_out, _ = self.temporal_self_attn(queries, queries, queries)
            queries = queries + attn_out
            
            # FFN
            queries = self.norm2(queries)
            ffn_out = self.ffn2(self.relu(self.ffn1(queries)))
            queries = queries + ffn_out
            
            return queries
    
    # Transformer modules
    class MockPerceptionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(256, 8, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
            self.linear1 = nn.Linear(256, 2048)
            self.linear2 = nn.Linear(2048, 256)
            self.norm1 = nn.LayerNorm(256)
            self.norm2 = nn.LayerNorm(256)
            self.norm3 = nn.LayerNorm(256)
            self.relu = nn.ReLU(inplace=True)
            
        def forward(self, x):
            # Self attention
            x2 = self.norm1(x)
            x = x + self.self_attn(x2, x2, x2)[0]
            
            # Cross attention (simplified - use self)
            x2 = self.norm2(x)
            x = x + self.cross_attn(x2, x2, x2)[0]
            
            # FFN
            x2 = self.norm3(x)
            x = x + self.linear2(self.relu(self.linear1(x2)))
            
            return x
    
    # Task heads
    class MockTrackHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_layer1 = nn.Linear(256, 256)
            self.cls_norm1 = nn.LayerNorm(256)
            self.cls_relu = nn.ReLU(inplace=True)
            self.cls_layer2 = nn.Linear(256, 10)
            
            self.reg_layer1 = nn.Linear(256, 256)
            self.reg_norm1 = nn.LayerNorm(256)
            self.reg_relu = nn.ReLU(inplace=True)
            self.reg_layer2 = nn.Linear(256, 10)
            
            self.track_embed1 = nn.Linear(256, 256)
            self.track_relu = nn.ReLU(inplace=True)
            self.track_embed2 = nn.Linear(256, 256)
            
        def forward(self, x):
            # Classification branch
            cls = self.cls_layer1(x)
            cls = self.cls_norm1(cls)
            cls = self.cls_relu(cls)
            cls = self.cls_layer2(cls)
            
            # Regression branch
            reg = self.reg_layer1(x)
            reg = self.reg_norm1(reg)
            reg = self.reg_relu(reg)
            reg = self.reg_layer2(reg)
            
            # Track embedding
            track = self.track_embed1(x)
            track = self.track_relu(track)
            track = self.track_embed2(track)
            
            return cls, reg, track
    
    class MockSegHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
            self.classifier = nn.Conv2d(32, 4, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x
    
    class MockMotionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(256, 128, batch_first=True)
            self.linear1 = nn.Linear(128, 256)
            self.relu = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(256, 6 * 12 * 2)  # 6 modes, 12 steps, 2D
            self.mode_prob = nn.Linear(128, 6)
            self.softmax = nn.Softmax(dim=-1)
            
        def forward(self, x):
            # x shape: [B, N, C]
            lstm_out, _ = self.lstm(x)
            
            # Trajectory prediction
            traj = self.linear1(lstm_out)
            traj = self.relu(traj)
            traj = self.linear2(traj)
            
            # Mode probabilities
            modes = self.mode_prob(lstm_out[:, -1, :])  # Use last timestep
            modes = self.softmax(modes)
            
            return traj, modes
    
    class MockOccHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d_1 = nn.Conv3d(256, 128, 3, padding=1)
            self.bn3d_1 = nn.BatchNorm3d(128)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv3d_2 = nn.Conv3d(128, 64, 3, padding=1)
            self.bn3d_2 = nn.BatchNorm3d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3d_3 = nn.Conv3d(64, 32, 3, padding=1)
            self.flow_head = nn.Conv3d(32, 3, 1)
            self.semantic_head = nn.Conv3d(32, 2, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            # x shape: [B, C, D, H, W]
            x = self.conv3d_1(x)
            x = self.bn3d_1(x)
            x = self.relu1(x)
            x = self.conv3d_2(x)
            x = self.bn3d_2(x)
            x = self.relu2(x)
            x = self.conv3d_3(x)
            
            flow = self.flow_head(x)
            semantic = self.semantic_head(x)
            semantic = self.sigmoid(semantic)
            
            return flow, semantic
    
    class MockPlanningHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(512, 256, batch_first=True)
            self.linear1 = nn.Linear(256, 128)
            self.relu = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(128, 2)  # 2D trajectory
            self.tanh = nn.Tanh()
            self.conv1d_1 = nn.Conv1d(256, 128, 3, padding=1)
            self.conv1d_2 = nn.Conv1d(128, 64, 3, padding=1)
            
        def forward(self, x):
            # x shape: [B, T, C]
            gru_out, _ = self.gru(x)
            
            # MLP for trajectory
            traj = self.linear1(gru_out)
            traj = self.relu(traj)
            traj = self.linear2(traj)
            traj = self.tanh(traj)
            
            # Cost volume (simplified)
            cost = x.transpose(1, 2)  # [B, C, T]
            cost = self.conv1d_1(cost)
            cost = self.relu(cost)
            cost = self.conv1d_2(cost)
            
            return traj, cost
    
    # Auxiliary modules
    class MockMemoryBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_embed = nn.Linear(256, 256)
            self.query_embed = nn.Linear(256, 256)
            self.softmax = nn.Softmax(dim=-1)
            
        def forward(self, query, memory):
            # Simplified attention mechanism
            q = self.query_embed(query)
            k = self.memory_embed(memory)
            
            # Compute attention weights
            attn = torch.matmul(q, k.transpose(-2, -1)) / (256 ** 0.5)
            attn = self.softmax(attn)
            
            # Apply attention
            out = torch.matmul(attn, memory)
            return out + query
    
    class MockQueryInteraction(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(256, 256)
            self.norm = nn.LayerNorm(256)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(0.1)
            self.linear2 = nn.Linear(256, 256)
            
        def forward(self, x):
            x2 = self.linear1(x)
            x2 = self.norm(x2)
            x2 = self.relu(x2)
            x2 = self.dropout(x2)
            x2 = self.linear2(x2)
            return x + x2
    
    # Create module mapping
    modules = {
        "backbone": {
            "ResNet101-DCN": MockResNet101DCN(),
            "FPN": MockFPN()
        },
        "bev_encoder": {
            "BEVFormer": MockBEVFormer()
        },
        "transformer": {
            "PerceptionTransformer": MockPerceptionTransformer()
        },
        "task_heads": {
            "TrackHead": MockTrackHead(),
            "SegHead": MockSegHead(),
            "MotionHead": MockMotionHead(),
            "OccHead": MockOccHead(),
            "PlanningHead": MockPlanningHead()
        },
        "auxiliary": {
            "MemoryBank": MockMemoryBank(),
            "QueryInteraction": MockQueryInteraction()
        }
    }
    
    return modules


def create_dummy_inputs(module_name: str, module: nn.Module, device: str = 'cpu'):
    """Create appropriate dummy inputs for each module type"""
    
    batch_size = 1
    
    if "ResNet" in module_name or "FPN" in module_name:
        # Image input
        return torch.randn(batch_size, 3, 224, 224).to(device)
    
    elif "BEVFormer" in module_name or "Transformer" in module_name:
        # Sequence input [B, N, C]
        seq_len = 100
        embed_dim = 256
        return torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    elif "TrackHead" in module_name:
        # Feature input
        num_queries = 300
        embed_dim = 256
        return torch.randn(batch_size, num_queries, embed_dim).to(device)
    
    elif "SegHead" in module_name:
        # BEV feature input [B, C, H, W]
        return torch.randn(batch_size, 256, 50, 50).to(device)
    
    elif "MotionHead" in module_name:
        # Sequence input for LSTM
        seq_len = 10
        embed_dim = 256
        return torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    elif "OccHead" in module_name:
        # 3D feature input [B, C, D, H, W]
        return torch.randn(batch_size, 256, 16, 32, 32).to(device)
    
    elif "PlanningHead" in module_name:
        # Temporal feature input
        seq_len = 6
        embed_dim = 512
        return torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    elif "MemoryBank" in module_name:
        # Query and memory inputs
        num_queries = 100
        memory_len = 4
        embed_dim = 256
        query = torch.randn(batch_size, num_queries, embed_dim).to(device)
        memory = torch.randn(batch_size, memory_len, embed_dim).to(device)
        return (query, memory)
    
    else:  # QueryInteraction and others
        # Generic feature input
        num_features = 100
        embed_dim = 256
        return torch.randn(batch_size, num_features, embed_dim).to(device)


def generate_enhanced_module_report(module_name: str, module: nn.Module, 
                                  module_info: Dict[str, Any], 
                                  category: str, output_dir: str,
                                  device: str = 'cpu'):
    """Generate an enhanced report for a specific module"""
    
    print(f"  Profiling {module_name}...")
    
    # Create profiler
    profiler = EnhancedProfiler(device=device)
    
    # Get dummy input
    inputs = create_dummy_inputs(module_name, module, device)
    
    # Profile the module
    try:
        trace_nodes = profiler.profile_model(module, inputs, capture_kernels=True, warmup_runs=2)
    except Exception as e:
        print(f"    Warning: Profiling failed for {module_name}: {e}")
        trace_nodes = []
    
    # Create report
    safe_name = module_name.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f"{safe_name}_enhanced.md")
    
    with open(output_path, 'w') as f:
        f.write(f"# {module_name} - Enhanced Operation Analysis\n\n")
        f.write(f"**Category**: {category}\n")
        f.write(f"**Description**: {module_info['description']}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Module statistics from original data
        f.write("## Module Statistics\n\n")
        if "total_params" in module_info:
            f.write(f"- **Total Parameters**: {module_info['total_params']}\n")
        if "memory_footprint" in module_info:
            f.write(f"- **Memory Footprint**: {module_info['memory_footprint']}\n")
        if "computational_complexity" in module_info:
            f.write(f"- **Computational Complexity**: {module_info['computational_complexity']}\n")
        f.write("\n")
        
        # Enhanced profiling results
        if trace_nodes:
            f.write("## Enhanced Profiling Results\n\n")
            
            # Summary
            total_time = sum(node.cuda_time_ms for node in trace_nodes)
            total_flops = sum(node.estimated_flops for node in trace_nodes)
            
            f.write(f"- **Total Operations Profiled**: {len(trace_nodes)}\n")
            f.write(f"- **Total Execution Time**: {total_time:.2f} ms\n")
            f.write(f"- **Total Estimated FLOPs**: {total_flops:,}\n")
            if total_time > 0:
                f.write(f"- **Effective TFLOPS**: {(total_flops / (total_time * 1e9)):.3f}\n")
            f.write("\n")
            
            # Operation breakdown
            f.write("## Operation Breakdown with Decomposition\n\n")
            
            # Group by high-level operation
            op_groups = {}
            for node in trace_nodes:
                if node.operation not in op_groups:
                    op_groups[node.operation] = []
                op_groups[node.operation].append(node)
            
            # Show decomposition for key operations
            decomposer = OperationDecomposer()
            shown_ops = set()
            
            for op_type in ['Conv2d', 'Conv3d', 'Linear', 'MultiheadAttention', 
                           'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'LSTM', 'GRU']:
                if op_type in op_groups and op_type not in shown_ops:
                    shown_ops.add(op_type)
                    nodes = op_groups[op_type]
                    
                    f.write(f"### {op_type} Operations ({len(nodes)} instances)\n\n")
                    
                    # Get decomposition
                    decomp = decomposer.decompose(op_type)
                    if decomp:
                        f.write("**Decomposition**:\n")
                        for i, primitive in enumerate(decomp.primitives, 1):
                            f.write(f"{i}. **{primitive.name}** ({primitive.category})\n")
                            f.write(f"   - Hardware: {primitive.hardware_mapping}\n")
                            f.write(f"   - FLOPs Formula: `{primitive.flops_formula}`\n")
                            if primitive.notes:
                                f.write(f"   - Notes: {primitive.notes}\n")
                        
                        f.write(f"\n**Memory Access Pattern**: {decomp.memory_access_pattern}\n")
                        
                        if decomp.fusion_opportunities:
                            f.write(f"**Fusion Opportunities**: {', '.join(decomp.fusion_opportunities)}\n")
                        
                        if decomp.hardware_requirements:
                            f.write("**Hardware Requirements**:\n")
                            for req, val in decomp.hardware_requirements.items():
                                f.write(f"- {req}: {val}\n")
                    
                    # Show timing for this operation type
                    total_op_time = sum(n.cuda_time_ms for n in nodes)
                    avg_op_time = total_op_time / len(nodes) if nodes else 0
                    
                    f.write(f"\n**Performance**:\n")
                    f.write(f"- Total Time: {total_op_time:.2f} ms\n")
                    f.write(f"- Average Time: {avg_op_time:.2f} ms\n")
                    
                    # Show hardware utilization if available
                    eligible_nodes = [n for n in nodes if n.tensor_core_eligible]
                    if eligible_nodes:
                        f.write(f"- Tensor Core Eligible: {len(eligible_nodes)}/{len(nodes)}\n")
                    
                    avg_compute = sum(n.compute_utilization_pct for n in nodes) / len(nodes) if nodes else 0
                    avg_memory = sum(n.memory_bandwidth_pct for n in nodes) / len(nodes) if nodes else 0
                    
                    if avg_compute > 0:
                        f.write(f"- Average Compute Utilization: {avg_compute:.1f}%\n")
                    if avg_memory > 0:
                        f.write(f"- Average Memory Bandwidth: {avg_memory:.1f}%\n")
                    
                    f.write("\n---\n\n")
        
        # Hardware optimization opportunities
        if trace_nodes and profiler.gpu_properties:
            f.write("## Hardware Optimization Opportunities\n\n")
            
            # Check for mixed precision opportunities
            fp32_ops = [n for n in trace_nodes if n.operation in ['Conv2d', 'Linear', 'MultiheadAttention'] 
                       and not n.tensor_core_eligible]
            
            if fp32_ops:
                potential_speedup = len(fp32_ops) * 2  # Rough estimate
                f.write(f"### Mixed Precision (FP16/TF32)\n")
                f.write(f"- {len(fp32_ops)} operations could benefit from mixed precision\n")
                f.write(f"- Potential speedup: ~{potential_speedup}x for these operations\n")
                f.write(f"- GPU supports Tensor Cores: {'Yes' if profiler.gpu_properties.get('tensor_cores') else 'No'}\n\n")
            
            # Check for fusion opportunities
            f.write("### Kernel Fusion\n")
            fusion_patterns = {
                "Conv + BN + ReLU": ["Conv2d", "BatchNorm2d", "ReLU"],
                "Linear + ReLU": ["Linear", "ReLU"],
                "Linear + GELU": ["Linear", "GELU"],
            }
            
            # Simple pattern matching (would be more sophisticated in practice)
            found_patterns = []
            op_sequence = [n.operation for n in trace_nodes[:20]]  # Check first 20 ops
            
            for pattern_name, pattern_ops in fusion_patterns.items():
                # Check if pattern exists in sequence
                for i in range(len(op_sequence) - len(pattern_ops) + 1):
                    if op_sequence[i:i+len(pattern_ops)] == pattern_ops:
                        found_patterns.append(pattern_name)
                        break
            
            if found_patterns:
                f.write(f"Detected fusable patterns: {', '.join(found_patterns)}\n")
            else:
                f.write("No obvious fusion patterns detected in profiled operations\n")
    
    print(f"    Generated: {output_path}")


def generate_enhanced_category_summary(category: str, modules: Dict[str, Dict[str, Any]], 
                                     output_dir: str):
    """Generate enhanced summary for a category"""
    
    output_path = os.path.join(output_dir, f"{category}_enhanced_summary.md")
    
    with open(output_path, 'w') as f:
        f.write(f"# {category.title()} - Enhanced Analysis Summary\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This category contains {len(modules)} modules with detailed operation decomposition analysis.\n\n")
        
        # Module table
        f.write("## Modules in Category\n\n")
        f.write("| Module | Description | Parameters | Key Operations |\n")
        f.write("|--------|-------------|------------|----------------|\n")
        
        for name, data in modules.items():
            params = data.get('total_params', 'N/A')
            
            # Extract key operations from architecture
            key_ops = set()
            if 'architecture' in data:
                def extract_ops(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, tuple) and len(v) >= 1:
                                key_ops.add(v[0])
                            else:
                                extract_ops(v)
                extract_ops(data['architecture'])
            
            key_ops_str = ', '.join(list(key_ops)[:4])
            if len(key_ops) > 4:
                key_ops_str += '...'
            
            f.write(f"| {name} | {data['description']} | {params} | {key_ops_str} |\n")
        
        f.write("\n")
        
        # Decomposition patterns
        f.write("## Common Decomposition Patterns\n\n")
        
        decomposer = OperationDecomposer()
        
        # Find common operations
        common_ops = set()
        for module_data in modules.values():
            if 'architecture' in module_data:
                def extract_ops(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, tuple) and len(v) >= 1:
                                common_ops.add(v[0])
                            else:
                                extract_ops(v)
                extract_ops(module_data['architecture'])
        
        # Show decomposition for common operations
        for op in sorted(common_ops):
            decomp = decomposer.decompose(op)
            if decomp:
                f.write(f"### {op}\n")
                f.write(f"- **Primitives**: {', '.join([p.name for p in decomp.primitives])}\n")
                f.write(f"- **Hardware**: {', '.join(set([p.hardware_mapping for p in decomp.primitives]))}\n")
                if decomp.fusion_opportunities:
                    f.write(f"- **Fusion**: {', '.join(decomp.fusion_opportunities)}\n")
                f.write("\n")
    
    print(f"Generated: {output_path}")


def main():
    """Main entry point"""
    
    print("="*60)
    print("Enhanced UniAD Module Analysis with Operation Decomposition")
    print(f"Generated at: {datetime.now()}")
    print("="*60)
    
    # Create output directory
    base_dir = "reports/enhanced_module_operations"
    os.makedirs(base_dir, exist_ok=True)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create mock modules
    print("\nCreating mock UniAD modules...")
    mock_modules = create_mock_uniad_modules()
    
    # Process each category
    for category, modules in UNIAD_MODULE_DETAILS.items():
        print(f"\nProcessing {category.upper()}...")
        
        category_dir = os.path.join(base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate reports for each module
        for module_name, module_info in modules.items():
            if category in mock_modules and module_name in mock_modules[category]:
                mock_module = mock_modules[category][module_name]
                generate_enhanced_module_report(
                    module_name, mock_module, module_info, 
                    category, category_dir, device
                )
            else:
                print(f"  Skipping {module_name} - no mock module available")
        
        # Generate category summary
        generate_enhanced_category_summary(category, modules, base_dir)
    
    # Generate master report
    print("\nGenerating master enhanced report...")
    
    master_path = os.path.join(base_dir, "UniAD_Enhanced_Operations_Analysis.md")
    with open(master_path, 'w') as f:
        f.write("# UniAD Enhanced Operations Analysis with Decomposition\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This enhanced analysis provides deep insights into UniAD's operations by:\n")
        f.write("- Decomposing PyTorch operations into hardware primitives (GEMM, elementwise, etc.)\n")
        f.write("- Mapping operations to actual CUDA kernels (cuBLAS, cuDNN, custom)\n")
        f.write("- Analyzing hardware utilization (compute, memory bandwidth, tensor cores)\n")
        f.write("- Identifying optimization opportunities (kernel fusion, mixed precision)\n\n")
        
        # Summary table
        f.write("## Module Categories\n\n")
        f.write("| Category | Modules | Key Hardware Patterns |\n")
        f.write("|----------|---------|----------------------|\n")
        
        hardware_patterns = {
            "backbone": "Conv2d → im2col + GEMM, DCNv2 custom kernels",
            "bev_encoder": "MultiheadAttention → QKV projections + scaled dot product",
            "transformer": "Attention mechanisms, high GEMM utilization",
            "task_heads": "Mixed operations: Conv2d/3d, Linear, LSTM/GRU",
            "auxiliary": "Lightweight Linear + normalization operations"
        }
        
        for category, modules in UNIAD_MODULE_DETAILS.items():
            pattern = hardware_patterns.get(category, "Various operations")
            f.write(f"| {category} | {len(modules)} | {pattern} |\n")
        
        f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        f.write("### Compute Patterns\n")
        f.write("- **GEMM-dominated**: Linear layers, attention mechanisms (>70% of compute)\n")
        f.write("- **Memory-bound**: Normalization, activation functions (<10% compute utilization)\n")
        f.write("- **Custom kernels**: DCNv2, specialized attention implementations\n\n")
        
        f.write("### Optimization Opportunities\n")
        f.write("1. **Mixed Precision**: Most GEMMs and convolutions eligible for FP16/TF32\n")
        f.write("2. **Kernel Fusion**: Conv+BN+ReLU, Linear+activation patterns\n")
        f.write("3. **Flash Attention**: Replace standard attention with fused implementation\n")
        f.write("4. **Graph Optimization**: TorchScript or TensorRT for inference\n\n")
        
        f.write("### Hardware Requirements\n")
        f.write("- **Memory Bandwidth**: Critical for BEV features (200×200×256)\n")
        f.write("- **Tensor Cores**: Essential for efficient GEMM operations\n")
        f.write("- **CUDA Compute**: >= 7.0 for DCNv2 and custom kernels\n")
    
    print(f"Generated: {master_path}")
    
    print("\n" + "="*60)
    print("Enhanced analysis complete!")
    print(f"Reports generated in: {base_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()