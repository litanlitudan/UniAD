#!/usr/bin/env python3
"""Profile UniAD models with operation decomposition and hardware analysis"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.operation_decomposition import OperationDecomposer
from core.enhanced_profiler import EnhancedProfiler, EnhancedTraceNode


def create_report_markdown(trace_nodes: List[EnhancedTraceNode], 
                          gpu_properties: Dict[str, Any],
                          output_path: str):
    """Create a detailed markdown report with decomposition analysis"""
    
    with open(output_path, 'w') as f:
        f.write("# UniAD Enhanced Profiling Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # GPU Properties
        if gpu_properties:
            f.write("## GPU Properties\n\n")
            f.write(f"- **Device**: {gpu_properties.get('name', 'Unknown')}\n")
            f.write(f"- **Compute Capability**: {gpu_properties.get('compute_capability', 'Unknown')}\n")
            f.write(f"- **Memory Bandwidth**: {gpu_properties.get('memory_bandwidth_gb_s', 0):.1f} GB/s\n")
            f.write(f"- **FP32 Performance**: {gpu_properties.get('fp32_tflops', 0):.1f} TFLOPS\n")
            f.write(f"- **FP16 Performance**: {gpu_properties.get('fp16_tflops', 0):.1f} TFLOPS\n")
            f.write(f"- **Tensor Cores**: {'Yes' if gpu_properties.get('tensor_cores') else 'No'}\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        total_time = sum(node.cuda_time_ms for node in trace_nodes)
        total_flops = sum(node.estimated_flops for node in trace_nodes)
        
        f.write(f"- **Total Operations**: {len(trace_nodes)}\n")
        f.write(f"- **Total CUDA Time**: {total_time:.2f} ms\n")
        f.write(f"- **Total Estimated FLOPs**: {total_flops:,}\n")
        f.write(f"- **Effective TFLOPS**: {(total_flops / (total_time * 1e9)):.2f}\n\n")
        
        # Operation Breakdown
        f.write("## Operation Breakdown\n\n")
        
        # Group by operation type
        op_groups = {}
        for node in trace_nodes:
            if node.operation not in op_groups:
                op_groups[node.operation] = {
                    'count': 0,
                    'total_time': 0,
                    'total_flops': 0,
                    'nodes': []
                }
            op_groups[node.operation]['count'] += 1
            op_groups[node.operation]['total_time'] += node.cuda_time_ms
            op_groups[node.operation]['total_flops'] += node.estimated_flops
            op_groups[node.operation]['nodes'].append(node)
        
        f.write("| Operation | Count | Total Time (ms) | Total FLOPs | Avg Time (ms) |\n")
        f.write("|-----------|-------|-----------------|-------------|---------------|\n")
        
        for op, stats in sorted(op_groups.items(), key=lambda x: x[1]['total_time'], reverse=True):
            avg_time = stats['total_time'] / stats['count']
            f.write(f"| {op} | {stats['count']} | {stats['total_time']:.2f} | "
                   f"{stats['total_flops']:,} | {avg_time:.2f} |\n")
        
        f.write("\n")
        
        # Detailed Decomposition Analysis
        f.write("## Detailed Operation Decomposition\n\n")
        
        # Show top 10 most time-consuming operations
        sorted_nodes = sorted(trace_nodes, key=lambda x: x.cuda_time_ms, reverse=True)[:10]
        
        for i, node in enumerate(sorted_nodes, 1):
            f.write(f"### {i}. {node.operation} - {node.module_name}\n\n")
            f.write(f"**Timing**: CPU: {node.cpu_time_ms:.2f}ms, CUDA: {node.cuda_time_ms:.2f}ms\n")
            f.write(f"**Memory**: {node.memory_allocated_mb:.1f}MB allocated\n")
            f.write(f"**Shapes**: Input: {node.input_shapes}, Output: {node.output_shapes}\n\n")
            
            if node.decomposed_ops:
                f.write("**Decomposition**:\n")
                for j, primitive in enumerate(node.decomposed_ops.primitives, 1):
                    f.write(f"{j}. **{primitive.name}** ({primitive.category})\n")
                    f.write(f"   - Hardware: {primitive.hardware_mapping}\n")
                    f.write(f"   - FLOPs: {primitive.flops_formula}\n")
                    if primitive.notes:
                        f.write(f"   - Notes: {primitive.notes}\n")
                
                f.write(f"\n**Total FLOPs Formula**: `{node.decomposed_ops.total_flops_formula}`\n")
                f.write(f"**Estimated FLOPs**: {node.estimated_flops:,}\n")
                
                if node.decomposed_ops.fusion_opportunities:
                    f.write(f"**Fusion Opportunities**: {', '.join(node.decomposed_ops.fusion_opportunities)}\n")
                
                f.write("\n**Hardware Utilization**:\n")
                f.write(f"- Tensor Core Eligible: {'Yes' if node.tensor_core_eligible else 'No'}\n")
                f.write(f"- Compute Utilization: {node.compute_utilization_pct:.1f}%\n")
                f.write(f"- Memory Bandwidth: {node.memory_bandwidth_pct:.1f}%\n")
            
            if node.cuda_kernels:
                f.write("\n**CUDA Kernels**:\n")
                for kernel in node.cuda_kernels[:5]:  # Show top 5 kernels
                    f.write(f"- {kernel.name} ({kernel.kernel_type}): {kernel.duration_us:.1f}μs\n")
            
            f.write("\n---\n\n")
        
        # Hardware Bottleneck Analysis
        f.write("## Hardware Bottleneck Analysis\n\n")
        
        compute_bound = [n for n in trace_nodes if n.compute_utilization_pct > 70]
        memory_bound = [n for n in trace_nodes if n.memory_bandwidth_pct > 70]
        
        f.write(f"- **Compute-Bound Operations**: {len(compute_bound)} ({len(compute_bound)/len(trace_nodes)*100:.1f}%)\n")
        f.write(f"- **Memory-Bound Operations**: {len(memory_bound)} ({len(memory_bound)/len(trace_nodes)*100:.1f}%)\n\n")
        
        # Optimization Opportunities
        f.write("## Optimization Opportunities\n\n")
        
        # Check for fusion opportunities
        fusion_candidates = []
        for i in range(len(trace_nodes) - 1):
            curr = trace_nodes[i]
            next_node = trace_nodes[i + 1]
            
            if curr.decomposed_ops and next_node.decomposed_ops:
                for fusion in curr.decomposed_ops.fusion_opportunities:
                    if next_node.operation.lower() in fusion.lower():
                        fusion_candidates.append((curr.operation, next_node.operation, fusion))
        
        if fusion_candidates:
            f.write("### Kernel Fusion Opportunities\n")
            for curr_op, next_op, fusion in fusion_candidates[:10]:
                f.write(f"- {curr_op} → {next_op}: {fusion}\n")
        
        # Tensor Core opportunities
        tc_eligible = [n for n in trace_nodes if not n.tensor_core_eligible and 
                      n.operation in ['Conv2d', 'Linear', 'MultiheadAttention']]
        
        if tc_eligible:
            f.write("\n### Tensor Core Optimization\n")
            f.write(f"{len(tc_eligible)} operations could benefit from FP16/TF32 for tensor cores:\n")
            for node in tc_eligible[:5]:
                f.write(f"- {node.operation} in {node.module_name}\n")


def profile_mock_model():
    """Profile a mock model to demonstrate the system"""
    
    # Create a simple model
    class MockUniADModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
            self.linear = nn.Linear(128, 256)
            
        def forward(self, x):
            # Conv block
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            
            # Reshape for attention
            b, c, h, w = x.shape
            x = x.view(b, c, -1).transpose(1, 2)  # [B, H*W, C]
            
            # Attention
            x, _ = self.attention(x, x, x)
            
            # Final projection
            x = self.linear(x)
            
            return x
    
    # Create dummy input
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Create model
    model = MockUniADModule().to(device)
    
    # Profile
    profiler = EnhancedProfiler(device=device)
    trace_nodes = profiler.profile_model(model, dummy_input, capture_kernels=True)
    
    # Generate report
    output_path = "reports/enhanced_profiling_demo.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    create_report_markdown(
        trace_nodes, 
        profiler.gpu_properties,
        output_path
    )
    
    # Also save JSON report
    json_path = output_path.replace('.md', '.json')
    profiler.generate_report(json_path)
    
    print(f"Reports generated:")
    print(f"  - Markdown: {output_path}")
    print(f"  - JSON: {json_path}")
    
    return trace_nodes


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced profiling with operation decomposition')
    parser.add_argument('--model', type=str, help='Model to profile (default: mock)')
    parser.add_argument('--output', type=str, default='reports/enhanced_profile.md',
                        help='Output report path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Enhanced UniAD Profiling with Operation Decomposition")
    print(f"Generated at: {datetime.now()}")
    print("="*60)
    
    # For now, just run the mock model demo
    print("\nProfiling mock model...")
    trace_nodes = profile_mock_model()
    
    print(f"\nProfiled {len(trace_nodes)} operations")
    print("\nTop 5 operations by time:")
    sorted_nodes = sorted(trace_nodes, key=lambda x: x.cuda_time_ms, reverse=True)[:5]
    for i, node in enumerate(sorted_nodes, 1):
        print(f"{i}. {node.operation}: {node.cuda_time_ms:.2f}ms")
    
    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)


if __name__ == '__main__':
    main()