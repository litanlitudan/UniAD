#!/usr/bin/env python3
"""
Main CLI for PyTorch Operation Tracer

Usage:
    python trace_ops.py --config CONFIG_PATH --checkpoint CHECKPOINT_PATH [options]
"""

import argparse
import json
import os
import sys
import time

# Add package to path if running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import OperationTracer
from analyzers import TraceAnalyzer
from visualizers import DataflowVisualizer
from utils import create_dummy_input, load_uniad_model, load_uniad_model_from_text


def main():
    parser = argparse.ArgumentParser(description='PyTorch Operation Tracer for UniAD')
    
    # Model configuration
    parser.add_argument('--config', type=str, help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2], 
                        help='UniAD stage (1: perception, 2: end-to-end)')
    
    # Tracing options
    parser.add_argument('--task-heads', type=str, nargs='+', 
                        default=['track', 'seg', 'motion', 'occ', 'planning'],
                        help='Task heads to trace')
    parser.add_argument('--temporal-frames', type=int, default=3, 
                        help='Number of temporal frames')
    parser.add_argument('--trace-backward', action='store_true', 
                        help='Trace backward pass')
    parser.add_argument('--filter-ops', type=str, nargs='+', 
                        help='Filter specific operations')
    parser.add_argument('--max-nodes', type=int, default=50, 
                        help='Maximum nodes to visualize')
    
    # Visualization options
    parser.add_argument('--visualization-mode', type=str, default='top-level',
                        choices=['top-level', 'expanded', 'full'],
                        help='Visualization mode')
    parser.add_argument('--expand-modules', type=str, nargs='+',
                        help='Modules to expand (e.g., BEVFormer,TrackHead)')
    parser.add_argument('--expand-pattern', type=str,
                        help='Pattern for module expansion (e.g., "*Head", "BEV*")')
    parser.add_argument('--expand-depth', type=int, default=2,
                        help='Depth to expand modules')
    parser.add_argument('--expand-heavy-modules', action='store_true',
                        help='Auto-expand modules using >1GB memory')
    parser.add_argument('--memory-threshold', type=float, default=1000.0,
                        help='Memory threshold in MB for auto-expansion')
    
    # Shape display options
    parser.add_argument('--show-shapes', action='store_true', default=True,
                        help='Show tensor shapes (default: True)')
    parser.add_argument('--no-shapes', dest='show_shapes', action='store_false',
                        help='Disable tensor shape display')
    parser.add_argument('--shape-format', type=str, default='full',
                        choices=['full', 'compact', 'semantic'],
                        help='Tensor shape display format')
    parser.add_argument('--track-shape-changes', action='store_true',
                        help='Track and highlight shape transformations')
    parser.add_argument('--highlight-reshapes', action='store_true',
                        help='Highlight reshape operations')
    parser.add_argument('--annotate-memory-per-element', action='store_true',
                        help='Show memory usage per tensor element')
    
    # Data type (dtype) options
    parser.add_argument('--show-dtype', action='store_true', default=True,
                        help='Show tensor data types (default: True)')
    parser.add_argument('--no-dtype', dest='show_dtype', action='store_false',
                        help='Disable data type display')
    parser.add_argument('--track-dtype', action='store_true',
                        help='Track data type conversions and mixed precision')
    parser.add_argument('--mixed-precision', type=str, choices=['fp16', 'bf16', 'int8'],
                        help='Simulate mixed precision execution')
    parser.add_argument('--dtype-memory-analysis', action='store_true',
                        help='Analyze memory impact of different data types')
    
    # Analysis options
    parser.add_argument('--memory-profile', action='store_true', 
                        help='Enable memory profiling')
    parser.add_argument('--bev-focus', action='store_true', 
                        help='Focus on BEV operations')
    parser.add_argument('--visualize-temporal', action='store_true', 
                        help='Visualize temporal flow')
    
    # Output options
    parser.add_argument('--output', type=str, default='trace_output.md', 
                        help='Output file path')
    parser.add_argument('--export-json', action='store_true', 
                        help='Export trace data as JSON')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to run on (cuda/cpu)')
    
    # Testing mode
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with dummy model')
    parser.add_argument('--text-load', action='store_true',
                        help='Use text-based model loading instead of mmdet3d registry')
    
    args = parser.parse_args()
    
    # Check config requirement
    if not args.test_mode and not args.config:
        parser.error("--config is required unless --test-mode is specified")
    
    # Initialize model
    if args.test_mode:
        print("Running in test mode with dummy model")
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ).to(args.device)
        cfg = None
    else:
        try:
            if args.text_load:
                print(f"Loading UniAD model using text-based parser from {args.config}")
                model, cfg = load_uniad_model_from_text(args.config, args.checkpoint, args.device)
            else:
                print(f"Loading UniAD model from {args.config}")
                model, cfg = load_uniad_model(args.config, args.checkpoint, args.device)
        except ImportError as e:
            print(f"Error: {e}")
            print("Running in fallback mode. Install mmdet3d for full functionality.")
            return 1
    
    # Create tracer
    tracer = OperationTracer(
        model, 
        trace_backward=args.trace_backward,
        filter_ops=args.filter_ops,
        stage=args.stage,
        task_heads=args.task_heads
    )
    
    # Create dummy input
    print("Creating dummy input...")
    dummy_input = create_dummy_input(cfg if not args.test_mode else None, args.device)
    
    # Perform tracing
    print("Tracing model operations...")
    start_time = time.time()
    outputs = tracer.trace(dummy_input)
    trace_time = time.time() - start_time
    
    # Get trace data
    trace_nodes = tracer.get_trace_data()
    print(f"Traced {len(trace_nodes)} operations in {trace_time:.2f} seconds")
    
    # Analyze trace
    print("Analyzing trace data...")
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes, stage=args.stage)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Process module expansion patterns
    expand_modules = args.expand_modules or []
    if args.expand_pattern and model:
        # Use hierarchy analyzer to find matching modules
        from core import ModuleHierarchyAnalyzer
        hierarchy_analyzer = ModuleHierarchyAnalyzer(model)
        pattern_matches = hierarchy_analyzer.get_modules_by_pattern(args.expand_pattern)
        expand_modules.extend(pattern_matches)
        print(f"Pattern '{args.expand_pattern}' matched {len(pattern_matches)} modules")
    
    visualizer = DataflowVisualizer(
        max_nodes=args.max_nodes,
        visualization_mode=args.visualization_mode,
        expand_modules=expand_modules,
        expand_heavy_modules=args.expand_heavy_modules,
        show_shapes=args.show_shapes,
        shape_format=args.shape_format,
        track_shape_changes=args.track_shape_changes,
        model=model if not args.test_mode else None,
        memory_threshold=args.memory_threshold
    )
    
    # Update visualization state with dtype options
    visualizer.state.show_dtype = args.show_dtype
    mermaid_diagram = visualizer.generate_mermaid(
        trace_nodes, 
        analysis['head_analysis'],
        analysis['memory_profile']
    )
    memory_heatmap = visualizer.generate_memory_heatmap(analysis['memory_profile'])
    
    # Write output
    print(f"Writing output to {args.output}")
    with open(args.output, 'w') as f:
        f.write(f"# UniAD PyTorch Operation Trace Report\n\n")
        f.write(f"**Model**: {args.config}\n")
        f.write(f"**Stage**: {args.stage}\n")
        f.write(f"**Trace Time**: {trace_time:.2f} seconds\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        summary = analysis['summary']
        f.write(f"- Total Operations: {summary['total_operations']}\n")
        f.write(f"- Unique Operations: {summary['unique_operations']}\n")
        f.write(f"- Total Memory: {summary['total_memory_mb']:.1f} MB\n")
        f.write(f"- Total Compute Time: {summary['total_compute_ms']:.1f} ms\n\n")
        
        # Task Head Analysis
        f.write("## Task Head Analysis\n\n")
        head_memory = analysis['head_analysis']['memory_by_head']
        head_compute = analysis['head_analysis']['compute_by_head']
        for head in ['track', 'seg', 'motion', 'occ', 'planning']:
            if head in head_memory:
                f.write(f"- **{head.capitalize()}**: {head_memory[head]:.1f} MB, {head_compute[head]:.1f} ms\n")
        f.write("\n")
        
        # BEV Analysis
        if args.bev_focus:
            f.write("## BEV Analysis\n\n")
            bev = analysis['bev_analysis']
            f.write(f"- Total BEV Operations: {bev['total_bev_ops']}\n")
            f.write(f"- BEV Memory: {bev['bev_memory_mb']:.1f} MB\n")
            f.write(f"- Grid Sizes: {bev['grid_sizes']}\n")
            f.write(f"- Frozen Encoder: {bev['frozen_encoder']}\n\n")
        
        # Dataflow Diagram
        f.write("## Dataflow Visualization\n\n")
        f.write(mermaid_diagram)
        f.write("\n\n")
        
        # Memory Heatmap
        if args.memory_profile:
            f.write("## Memory Profile\n\n")
            f.write(memory_heatmap)
            f.write("\n\n")
        
        # Shape Transformation Analysis
        if args.track_shape_changes or args.highlight_reshapes:
            f.write("## Shape Transformation Analysis\n\n")
            shape_report = visualizer.generate_shape_transformation_report(trace_nodes)
            f.write(shape_report)
            f.write("\n\n")
        
        # Memory-based Auto-expansion View
        if args.expand_heavy_modules:
            f.write("## Memory-Based Module Expansion\n\n")
            memory_view = visualizer.generate_memory_based_view(trace_nodes, args.memory_threshold)
            f.write(memory_view)
            f.write("\n\n")
        
        # Data Type Memory Analysis
        if args.dtype_memory_analysis:
            f.write("## Data Type Memory Impact Analysis\n\n")
            f.write("This section analyzes how different data types affect memory usage.\n\n")
            
            # Calculate memory impact for different precision levels
            from core.data_structures import PYTORCH_DTYPES, UNIAD_DTYPE_CONFIGS
            
            total_memory_fp32 = sum(node.memory_usage for node in trace_nodes)
            
            f.write("### Memory Usage by Precision\n\n")
            f.write("| Precision | Total Memory | Reduction | Notes |\n")
            f.write("|-----------|-------------|-----------|-------|\n")
            f.write(f"| FP32 (baseline) | {total_memory_fp32:.1f} MB | 0% | Full precision |\n")
            f.write(f"| FP16 | {total_memory_fp32 * 0.5:.1f} MB | 50% | Half precision |\n")
            f.write(f"| BF16 | {total_memory_fp32 * 0.5:.1f} MB | 50% | Brain float |\n")
            f.write(f"| INT8 | {total_memory_fp32 * 0.25:.1f} MB | 75% | Quantized |\n")
            f.write("\n\n")
            
            # Show mixed precision configurations
            f.write("### Mixed Precision Configurations\n\n")
            for config_name, config in UNIAD_DTYPE_CONFIGS.items():
                f.write(f"**{config_name}**:\n")
                for component, dtype in config.items():
                    f.write(f"- {component}: {dtype}\n")
                f.write("\n")
            
            # Dtype tracking if enabled
            if args.track_dtype:
                f.write("### Data Type Conversions\n\n")
                f.write("Tracking of data type conversions throughout the model:\n\n")
                dtype_conversions = []
                for node in trace_nodes:
                    if node.input_shapes and node.output_shapes:
                        in_dtype = node.input_shapes[0].dtype if node.input_shapes else "unknown"
                        out_dtype = node.output_shapes[0].dtype if node.output_shapes else "unknown"
                        if in_dtype != out_dtype:
                            dtype_conversions.append({
                                'module': node.module_path,
                                'operation': node.operation,
                                'from': in_dtype,
                                'to': out_dtype
                            })
                
                if dtype_conversions:
                    f.write("| Module | Operation | From | To |\n")
                    f.write("|--------|-----------|------|----|\n")
                    for conv in dtype_conversions[:20]:  # Limit to top 20
                        f.write(f"| {conv['module'][:40]} | {conv['operation'][:20]} | {conv['from']} | {conv['to']} |\n")
                else:
                    f.write("No data type conversions detected.\n")
                f.write("\n\n")
    
    # Export JSON if requested
    if args.export_json:
        json_path = args.output.replace('.md', '.json')
        print(f"Exporting trace data to {json_path}")
        trace_data = {
            'config': args.config,
            'stage': args.stage,
            'analysis': analysis,
            'nodes': [node.to_dict() for node in trace_nodes]
        }
        with open(json_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())