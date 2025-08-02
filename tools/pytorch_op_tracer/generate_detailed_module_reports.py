#!/usr/bin/env python3
"""Generate detailed reports for all UniAD modules down to PyTorch operation level"""

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# UniAD Module Hierarchy
UNIAD_MODULES = {
    "backbone": {
        "ResNet101-DCN": {
            "description": "Image backbone with deformable convolutions",
            "submodules": [
                "stem", "layer1", "layer2", "layer3", "layer4",
                "dcn_layers", "norm_layers"
            ],
            "key_ops": ["Conv2d", "BatchNorm2d", "ReLU", "DCNv2", "MaxPool2d"]
        },
        "FPN": {
            "description": "Feature Pyramid Network for multi-scale features",
            "submodules": [
                "lateral_convs", "fpn_convs", "extra_convs"
            ],
            "key_ops": ["Conv2d", "Upsample", "Add", "ReLU"]
        }
    },
    "bev_encoder": {
        "BEVFormer": {
            "description": "Bird's Eye View transformer encoder",
            "submodules": [
                "bev_queries", "spatial_cross_attention", "temporal_self_attention",
                "ffn", "layer_norm"
            ],
            "key_ops": ["Linear", "MultiheadAttention", "LayerNorm", "Dropout"]
        },
        "PositionalEncoding": {
            "description": "Learnable positional encoding for BEV grid",
            "submodules": ["row_embed", "col_embed"],
            "key_ops": ["Embedding", "Add"]
        }
    },
    "transformer": {
        "PerceptionTransformer": {
            "description": "Main transformer for perception tasks",
            "submodules": [
                "encoder", "decoder", "embed_dims_projection"
            ],
            "key_ops": ["Linear", "MultiheadAttention", "LayerNorm", "GELU"]
        },
        "MSDeformableAttention3D": {
            "description": "Multi-scale deformable attention in 3D",
            "submodules": [
                "sampling_offsets", "attention_weights", "value_proj", "output_proj"
            ],
            "key_ops": ["Linear", "Conv2d", "Softmax", "MatMul"]
        }
    },
    "task_heads": {
        "TrackHead": {
            "description": "3D object detection and tracking head",
            "submodules": [
                "cls_branch", "reg_branch", "track_branch", "query_embedding"
            ],
            "key_ops": ["Linear", "Conv1d", "Sigmoid", "ReLU"]
        },
        "SegHead": {
            "description": "BEV segmentation head for lanes and map",
            "submodules": [
                "seg_conv", "seg_classifier", "panoptic_fusion"
            ],
            "key_ops": ["Conv2d", "BatchNorm2d", "ReLU", "Softmax"]
        },
        "MotionHead": {
            "description": "Multi-modal motion prediction head",
            "submodules": [
                "motion_decoder", "traj_refine", "mode_prob"
            ],
            "key_ops": ["Linear", "LSTM", "Softmax", "Tanh"]
        },
        "OccHead": {
            "description": "Occupancy flow prediction head",
            "submodules": [
                "occ_encoder", "flow_head", "semantic_head"
            ],
            "key_ops": ["Conv3d", "ConvTranspose3d", "BatchNorm3d", "ReLU"]
        },
        "PlanningHead": {
            "description": "Ego vehicle trajectory planning head",
            "submodules": [
                "planning_decoder", "cost_volume", "trajectory_optimizer"
            ],
            "key_ops": ["Linear", "Conv1d", "GRU", "Tanh"]
        }
    },
    "auxiliary": {
        "MemoryBank": {
            "description": "Temporal memory bank for tracking",
            "submodules": ["memory_embed", "memory_update"],
            "key_ops": ["Linear", "Softmax", "MatMul"]
        },
        "QueryInteraction": {
            "description": "Query interaction module for tracking",
            "submodules": ["query_update", "query_fusion"],
            "key_ops": ["Linear", "LayerNorm", "Dropout"]
        },
        "GridMask": {
            "description": "Data augmentation with grid masking",
            "submodules": ["mask_generator"],
            "key_ops": ["Bernoulli", "Mul"]
        }
    }
}

# PyTorch operation categories
PYTORCH_OP_CATEGORIES = {
    "convolution": ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "DCNv2"],
    "pooling": ["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool2d"],
    "normalization": ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm2d"],
    "activation": ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax"],
    "linear": ["Linear", "Bilinear"],
    "dropout": ["Dropout", "Dropout2d", "Dropout3d"],
    "attention": ["MultiheadAttention", "ScaledDotProductAttention"],
    "recurrent": ["LSTM", "GRU", "RNN"],
    "embedding": ["Embedding", "EmbeddingBag"],
    "loss": ["CrossEntropyLoss", "MSELoss", "L1Loss", "SmoothL1Loss", "BCELoss"],
    "tensor_ops": ["Add", "Mul", "MatMul", "Concat", "Split", "Reshape", "Transpose", "Permute"]
}


def create_mock_module(module_info: Dict[str, Any]) -> nn.Module:
    """Create a mock module based on module info"""
    class MockModule(nn.Module):
        def __init__(self, name: str, ops: List[str]):
            super().__init__()
            self.name = name
            
            # Add mock operations based on key_ops
            for i, op in enumerate(ops[:5]):  # Limit to 5 ops for simplicity
                if op == "Conv2d":
                    setattr(self, f"{op.lower()}_{i}", nn.Conv2d(256, 256, 3, padding=1))
                elif op == "Conv1d":
                    setattr(self, f"{op.lower()}_{i}", nn.Conv1d(256, 256, 3, padding=1))
                elif op == "Conv3d":
                    setattr(self, f"{op.lower()}_{i}", nn.Conv3d(64, 64, 3, padding=1))
                elif op == "Linear":
                    setattr(self, f"{op.lower()}_{i}", nn.Linear(256, 256))
                elif op == "BatchNorm2d":
                    setattr(self, f"{op.lower()}_{i}", nn.BatchNorm2d(256))
                elif op == "BatchNorm3d":
                    setattr(self, f"{op.lower()}_{i}", nn.BatchNorm3d(64))
                elif op == "LayerNorm":
                    setattr(self, f"{op.lower()}_{i}", nn.LayerNorm(256))
                elif op == "ReLU":
                    setattr(self, f"{op.lower()}_{i}", nn.ReLU())
                elif op == "GELU":
                    setattr(self, f"{op.lower()}_{i}", nn.GELU())
                elif op == "Sigmoid":
                    setattr(self, f"{op.lower()}_{i}", nn.Sigmoid())
                elif op == "Softmax":
                    setattr(self, f"{op.lower()}_{i}", nn.Softmax(dim=-1))
                elif op == "Dropout":
                    setattr(self, f"{op.lower()}_{i}", nn.Dropout(0.1))
                elif op == "LSTM":
                    setattr(self, f"{op.lower()}_{i}", nn.LSTM(256, 128, batch_first=True))
                elif op == "GRU":
                    setattr(self, f"{op.lower()}_{i}", nn.GRU(256, 128, batch_first=True))
                elif op == "MultiheadAttention":
                    setattr(self, f"{op.lower()}_{i}", nn.MultiheadAttention(256, 8))
                elif op == "MaxPool2d":
                    setattr(self, f"{op.lower()}_{i}", nn.MaxPool2d(2))
                elif op == "Embedding":
                    setattr(self, f"{op.lower()}_{i}", nn.Embedding(1000, 256))
                else:
                    # For ops we don't have specific implementations, use Identity
                    setattr(self, f"{op.lower()}_{i}", nn.Identity())
        
        def forward(self, x):
            # Simple forward pass through all operations
            for name, module in self.named_children():
                if isinstance(module, (nn.LSTM, nn.GRU)):
                    if x.dim() == 2:
                        x = x.unsqueeze(1)  # Add sequence dimension
                    x, _ = module(x)
                    x = x.squeeze(1)  # Remove sequence dimension
                elif isinstance(module, nn.MultiheadAttention):
                    if x.dim() == 2:
                        x = x.unsqueeze(0)  # Add sequence dimension
                    x, _ = module(x, x, x)
                    x = x.squeeze(0)
                else:
                    x = module(x)
            return x
    
    return MockModule


def trace_module(module: nn.Module, module_name: str, device: str = 'cpu') -> Dict[str, Any]:
    """Trace a module and collect operation information"""
    from core import OperationTracer
    from analyzers import TraceAnalyzer
    
    # Create appropriate dummy input based on module type
    if 'Conv2d' in str(module):
        dummy_input = torch.randn(1, 256, 64, 64).to(device)
    elif 'Conv1d' in str(module):
        dummy_input = torch.randn(1, 256, 100).to(device)
    elif 'Conv3d' in str(module):
        dummy_input = torch.randn(1, 64, 16, 16, 16).to(device)
    else:
        dummy_input = torch.randn(1, 256).to(device)
    
    # Create tracer
    tracer = OperationTracer(module, stage=2)
    
    # Perform tracing
    try:
        outputs = tracer.trace(dummy_input)
        trace_nodes = tracer.get_trace_data()
        
        # Analyze trace
        analyzer = TraceAnalyzer()
        analysis = analyzer.analyze(trace_nodes, stage=2)
        
        # Collect operation statistics
        op_stats = {}
        for node in trace_nodes:
            op_type = node.operation
            if op_type not in op_stats:
                op_stats[op_type] = {
                    'count': 0,
                    'total_memory_mb': 0,
                    'total_compute_ms': 0,
                    'shapes': []
                }
            op_stats[op_type]['count'] += 1
            op_stats[op_type]['total_memory_mb'] += node.memory_usage
            op_stats[op_type]['total_compute_ms'] += node.compute_time
            if node.output_shapes:
                op_stats[op_type]['shapes'].append(str(node.output_shapes[0]))
        
        return {
            'module_name': module_name,
            'total_operations': len(trace_nodes),
            'unique_operations': len(op_stats),
            'total_memory_mb': sum(node.memory_usage for node in trace_nodes),
            'total_compute_ms': sum(node.compute_time for node in trace_nodes),
            'operation_stats': op_stats,
            'summary': analysis.get('summary', {}),
            'trace_nodes': trace_nodes
        }
    except Exception as e:
        print(f"Error tracing {module_name}: {e}")
        return {
            'module_name': module_name,
            'error': str(e),
            'total_operations': 0,
            'unique_operations': 0,
            'operation_stats': {}
        }


def generate_module_report(module_info: Dict[str, Any], category: str, output_dir: str):
    """Generate a detailed report for a specific module"""
    from visualizers import DataflowVisualizer
    
    module_name = module_info['module_name']
    safe_name = module_name.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f"{safe_name}_report.md")
    
    with open(output_path, 'w') as f:
        f.write(f"# {module_name} - Detailed Operation Report\n\n")
        f.write(f"**Category**: {category}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'error' in module_info:
            f.write(f"## Error\n\n")
            f.write(f"Failed to trace module: {module_info['error']}\n\n")
            return
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- Total Operations: {module_info['total_operations']}\n")
        f.write(f"- Unique Operation Types: {module_info['unique_operations']}\n")
        f.write(f"- Total Memory Usage: {module_info['total_memory_mb']:.2f} MB\n")
        f.write(f"- Total Compute Time: {module_info['total_compute_ms']:.2f} ms\n\n")
        
        # Operation breakdown
        f.write("## Operation Breakdown\n\n")
        f.write("| Operation | Count | Memory (MB) | Compute (ms) | Example Shape |\n")
        f.write("|-----------|-------|-------------|--------------|---------------|\n")
        
        op_stats = module_info.get('operation_stats', {})
        for op_type, stats in sorted(op_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            example_shape = stats['shapes'][0] if stats['shapes'] else 'N/A'
            f.write(f"| {op_type} | {stats['count']} | "
                   f"{stats['total_memory_mb']:.2f} | "
                   f"{stats['total_compute_ms']:.2f} | "
                   f"{example_shape} |\n")
        
        f.write("\n")
        
        # PyTorch operation categories
        f.write("## Operation Categories\n\n")
        categorized_ops = {}
        for op_type in op_stats:
            found = False
            for cat, ops in PYTORCH_OP_CATEGORIES.items():
                if op_type in ops:
                    if cat not in categorized_ops:
                        categorized_ops[cat] = []
                    categorized_ops[cat].append(op_type)
                    found = True
                    break
            if not found:
                if 'other' not in categorized_ops:
                    categorized_ops['other'] = []
                categorized_ops['other'].append(op_type)
        
        for cat, ops in sorted(categorized_ops.items()):
            f.write(f"### {cat.title()}\n")
            f.write(f"- Operations: {', '.join(ops)}\n")
            total_count = sum(op_stats[op]['count'] for op in ops if op in op_stats)
            f.write(f"- Total Count: {total_count}\n\n")
        
        # Visualizations
        if 'trace_nodes' in module_info and module_info['trace_nodes']:
            f.write("## Dataflow Visualization\n\n")
            visualizer = DataflowVisualizer(
                show_shapes=True,
                shape_format='compact',
                visualization_mode='full',
                max_nodes=30
            )
            try:
                mermaid_diagram = visualizer.generate_mermaid(
                    module_info['trace_nodes'][:30],  # Limit to 30 nodes
                    {},
                    {'operation_memory': op_stats}
                )
                f.write(mermaid_diagram)
                f.write("\n\n")
            except Exception as e:
                f.write(f"Could not generate visualization: {e}\n\n")
    
    print(f"Generated report: {output_path}")


def generate_category_summary(category: str, modules: Dict[str, Dict[str, Any]], 
                             module_results: List[Dict[str, Any]], output_dir: str):
    """Generate a summary report for a category of modules"""
    output_path = os.path.join(output_dir, f"{category}_summary.md")
    
    with open(output_path, 'w') as f:
        f.write(f"# {category.title()} Modules - Summary Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"Total modules analyzed: {len(modules)}\n\n")
        
        # Module table
        f.write("## Module Summary\n\n")
        f.write("| Module | Description | Key Operations | Total Ops | Memory (MB) |\n")
        f.write("|--------|-------------|----------------|-----------|-------------|\n")
        
        for module_name, module_data in modules.items():
            # Find corresponding result
            result = next((r for r in module_results if r['module_name'] == module_name), None)
            if result and 'error' not in result:
                total_ops = result['total_operations']
                memory = result['total_memory_mb']
            else:
                total_ops = 'N/A'
                memory = 'N/A'
            
            key_ops = ', '.join(module_data['key_ops'][:3]) + ('...' if len(module_data['key_ops']) > 3 else '')
            f.write(f"| {module_name} | {module_data['description']} | "
                   f"{key_ops} | {total_ops} | "
                   f"{memory:.2f if isinstance(memory, (int, float)) else memory} |\n")
        
        f.write("\n")
        
        # Aggregate statistics
        f.write("## Aggregate Statistics\n\n")
        
        total_memory = sum(r['total_memory_mb'] for r in module_results if 'error' not in r)
        total_ops = sum(r['total_operations'] for r in module_results if 'error' not in r)
        
        f.write(f"- Total Memory Usage: {total_memory:.2f} MB\n")
        f.write(f"- Total Operations: {total_ops}\n")
        
        # Operation frequency across category
        f.write("\n## Most Common Operations\n\n")
        op_frequency = {}
        for result in module_results:
            if 'error' not in result:
                for op, stats in result.get('operation_stats', {}).items():
                    if op not in op_frequency:
                        op_frequency[op] = 0
                    op_frequency[op] += stats['count']
        
        f.write("| Operation | Total Count | Modules Using |\n")
        f.write("|-----------|-------------|---------------|\n")
        
        for op, count in sorted(op_frequency.items(), key=lambda x: x[1], reverse=True)[:10]:
            modules_using = sum(1 for r in module_results 
                              if 'error' not in r and op in r.get('operation_stats', {}))
            f.write(f"| {op} | {count} | {modules_using} |\n")
    
    print(f"Generated category summary: {output_path}")


def generate_master_summary(all_results: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Generate a master summary of all modules"""
    output_path = os.path.join(output_dir, "UniAD_Complete_Module_Analysis.md")
    
    with open(output_path, 'w') as f:
        f.write("# UniAD Complete Module Analysis\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        total_modules = sum(len(results) for results in all_results.values())
        total_memory = sum(r['total_memory_mb'] for results in all_results.values() 
                          for r in results if 'error' not in r)
        total_ops = sum(r['total_operations'] for results in all_results.values() 
                       for r in results if 'error' not in r)
        
        f.write(f"- Total Modules Analyzed: {total_modules}\n")
        f.write(f"- Total Operations Traced: {total_ops}\n")
        f.write(f"- Total Memory Usage: {total_memory:.2f} MB\n\n")
        
        # Category breakdown
        f.write("## Category Breakdown\n\n")
        f.write("| Category | Modules | Operations | Memory (MB) |\n")
        f.write("|----------|---------|------------|-------------|\n")
        
        for category, results in all_results.items():
            cat_ops = sum(r['total_operations'] for r in results if 'error' not in r)
            cat_memory = sum(r['total_memory_mb'] for r in results if 'error' not in r)
            f.write(f"| {category.title()} | {len(results)} | {cat_ops} | {cat_memory:.2f} |\n")
        
        f.write("\n")
        
        # PyTorch operation distribution
        f.write("## PyTorch Operation Distribution\n\n")
        
        global_op_stats = {}
        for results in all_results.values():
            for result in results:
                if 'error' not in result:
                    for op, stats in result.get('operation_stats', {}).items():
                        if op not in global_op_stats:
                            global_op_stats[op] = {
                                'count': 0,
                                'memory_mb': 0,
                                'modules_using': set()
                            }
                        global_op_stats[op]['count'] += stats['count']
                        global_op_stats[op]['memory_mb'] += stats['total_memory_mb']
                        global_op_stats[op]['modules_using'].add(result['module_name'])
        
        f.write("### Top 20 Most Used Operations\n\n")
        f.write("| Operation | Total Count | Memory (MB) | Modules Using |\n")
        f.write("|-----------|-------------|-------------|---------------|\n")
        
        sorted_ops = sorted(global_op_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
        for op, stats in sorted_ops:
            f.write(f"| {op} | {stats['count']} | {stats['memory_mb']:.2f} | "
                   f"{len(stats['modules_using'])} |\n")
        
        f.write("\n")
        
        # Operation category analysis
        f.write("## Operation Category Analysis\n\n")
        
        category_stats = {}
        for op, stats in global_op_stats.items():
            found = False
            for cat, ops in PYTORCH_OP_CATEGORIES.items():
                if op in ops:
                    if cat not in category_stats:
                        category_stats[cat] = {'count': 0, 'memory_mb': 0, 'ops': set()}
                    category_stats[cat]['count'] += stats['count']
                    category_stats[cat]['memory_mb'] += stats['memory_mb']
                    category_stats[cat]['ops'].add(op)
                    found = True
                    break
            if not found:
                if 'other' not in category_stats:
                    category_stats['other'] = {'count': 0, 'memory_mb': 0, 'ops': set()}
                category_stats['other']['count'] += stats['count']
                category_stats['other']['memory_mb'] += stats['memory_mb']
                category_stats['other']['ops'].add(op)
        
        f.write("| Category | Operation Count | Memory (MB) | Unique Ops |\n")
        f.write("|----------|-----------------|-------------|------------|\n")
        
        for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            f.write(f"| {cat.title()} | {stats['count']} | {stats['memory_mb']:.2f} | "
                   f"{len(stats['ops'])} |\n")
        
        f.write("\n")
        
        # Module complexity ranking
        f.write("## Module Complexity Ranking\n\n")
        f.write("### Top 10 Most Complex Modules (by operation count)\n\n")
        f.write("| Module | Category | Operations | Memory (MB) |\n")
        f.write("|--------|----------|------------|-------------|\n")
        
        all_modules = []
        for category, results in all_results.items():
            for result in results:
                if 'error' not in result:
                    all_modules.append((result['module_name'], category, 
                                      result['total_operations'], result['total_memory_mb']))
        
        for module, cat, ops, mem in sorted(all_modules, key=lambda x: x[2], reverse=True)[:10]:
            f.write(f"| {module} | {cat} | {ops} | {mem:.2f} |\n")
        
        f.write("\n")
        
        # Report index
        f.write("## Generated Reports Index\n\n")
        for category in all_results:
            f.write(f"### {category.title()}\n")
            f.write(f"- `{category}_summary.md` - Category summary\n")
            f.write(f"- `{category}/` - Individual module reports\n\n")
    
    print(f"Generated master summary: {output_path}")


def main():
    """Main entry point"""
    print("="*60)
    print("UniAD Detailed Module Analysis")
    print(f"Generated at: {datetime.now()}")
    print("="*60)
    
    # Create output directory structure
    base_output_dir = "reports/detailed_modules"
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_results = {}
    
    # Process each category
    for category, modules in UNIAD_MODULES.items():
        print(f"\n{'='*40}")
        print(f"Processing {category.upper()} modules")
        print(f"{'='*40}")
        
        category_dir = os.path.join(base_output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        category_results = []
        
        for module_name, module_info in modules.items():
            print(f"\nAnalyzing {module_name}...")
            
            # Create mock module
            mock_module = MockModule(module_name, module_info['key_ops'])
            
            # Trace module
            result = trace_module(mock_module, module_name, device='cpu')
            category_results.append(result)
            
            # Generate individual report
            generate_module_report(result, category, category_dir)
        
        # Generate category summary
        generate_category_summary(category, modules, category_results, base_output_dir)
        
        all_results[category] = category_results
    
    # Generate master summary
    print(f"\n{'='*40}")
    print("Generating master summary...")
    print(f"{'='*40}")
    generate_master_summary(all_results, base_output_dir)
    
    # Save raw results as JSON
    json_path = os.path.join(base_output_dir, "complete_analysis.json")
    with open(json_path, 'w') as f:
        # Convert sets to lists for JSON serialization
        json_data = {}
        for cat, results in all_results.items():
            json_data[cat] = []
            for result in results:
                clean_result = {k: v for k, v in result.items() if k != 'trace_nodes'}
                json_data[cat].append(clean_result)
        json.dump(json_data, f, indent=2)
    
    print(f"\nRaw analysis data saved to: {json_path}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Reports generated in: {base_output_dir}")
    print("="*60)
    
    # Print summary statistics
    total_modules = sum(len(modules) for modules in UNIAD_MODULES.values())
    print(f"\nTotal modules analyzed: {total_modules}")
    print(f"Categories: {len(UNIAD_MODULES)}")
    print("\nReport structure:")
    print(f"  {base_output_dir}/")
    print(f"    ├── UniAD_Complete_Module_Analysis.md")
    print(f"    ├── complete_analysis.json")
    for category in UNIAD_MODULES:
        print(f"    ├── {category}_summary.md")
        print(f"    └── {category}/")
        print(f"        └── [individual module reports]")


def MockModule(name: str, ops: List[str]) -> nn.Module:
    """Create a mock module with specified operations"""
    class _MockModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = name
            
            # Add operations based on key_ops
            self.ops = nn.ModuleList()
            for i, op in enumerate(ops[:5]):  # Limit to 5 ops
                if op == "Conv2d":
                    self.ops.append(nn.Conv2d(256, 256, 3, padding=1))
                elif op == "Conv1d":
                    self.ops.append(nn.Conv1d(256, 256, 3, padding=1))
                elif op == "Conv3d":
                    self.ops.append(nn.Conv3d(64, 64, 3, padding=1))
                elif op == "Linear":
                    self.ops.append(nn.Linear(256, 256))
                elif op == "BatchNorm2d":
                    self.ops.append(nn.BatchNorm2d(256))
                elif op == "BatchNorm3d":
                    self.ops.append(nn.BatchNorm3d(64))
                elif op == "LayerNorm":
                    self.ops.append(nn.LayerNorm(256))
                elif op == "ReLU":
                    self.ops.append(nn.ReLU())
                elif op == "GELU":
                    self.ops.append(nn.GELU())
                elif op == "Sigmoid":
                    self.ops.append(nn.Sigmoid())
                elif op == "Softmax":
                    self.ops.append(nn.Softmax(dim=-1))
                elif op == "Dropout":
                    self.ops.append(nn.Dropout(0.1))
                elif op == "MultiheadAttention":
                    self.ops.append(nn.MultiheadAttention(256, 8))
                else:
                    self.ops.append(nn.Identity())
        
        def forward(self, x):
            for op in self.ops:
                if isinstance(op, nn.MultiheadAttention):
                    # Handle attention specially
                    if x.dim() == 2:
                        x = x.unsqueeze(0)
                    x, _ = op(x, x, x)
                    x = x.squeeze(0)
                else:
                    x = op(x)
            return x
    
    return _MockModule()


if __name__ == '__main__':
    main()