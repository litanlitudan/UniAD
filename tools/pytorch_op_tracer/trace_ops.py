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
from visualizers.mermaid_visualizer import DataflowVisualizer
from utils import create_dummy_input, load_uniad_model, load_uniad_model_from_text

# Import interactive visualization components
INTERACTIVE_AVAILABLE = False
InteractiveVisualizer = None  # type: ignore
InteractiveConfig = None  # type: ignore
ExportManager = None  # type: ignore
FilterEngine = None  # type: ignore
SearchResult = None  # type: ignore
MemoryTimelineVisualizer = None  # type: ignore
TaskHeadComparator = None  # type: ignore
QueueVisualizer = None  # type: ignore
TemporalTracer = None  # type: ignore

try:
    from visualizers.interactive_visualizer import InteractiveVisualizer as _InteractiveVisualizer
    from core.visualization_config import InteractiveConfig as _InteractiveConfig
    from utils.export_manager import ExportManager as _ExportManager
    from visualizers.filter_engine import FilterEngine as _FilterEngine, SearchResult as _SearchResult
    from visualizers.memory_timeline import MemoryTimelineVisualizer as _MemoryTimelineVisualizer
    from visualizers.task_head_comparator import TaskHeadComparator as _TaskHeadComparator
    from visualizers.queue_visualizer import QueueVisualizer as _QueueVisualizer
    from analyzers.temporal_analyzer import TemporalTracer as _TemporalTracer
    
    InteractiveVisualizer = _InteractiveVisualizer
    InteractiveConfig = _InteractiveConfig
    ExportManager = _ExportManager
    FilterEngine = _FilterEngine
    SearchResult = _SearchResult
    MemoryTimelineVisualizer = _MemoryTimelineVisualizer
    TaskHeadComparator = _TaskHeadComparator
    QueueVisualizer = _QueueVisualizer
    TemporalTracer = _TemporalTracer
    INTERACTIVE_AVAILABLE = True
except ImportError:
    print("Warning: Interactive visualization components not available")


def generate_interactive_visualization(trace_nodes, analysis, args):  # type: ignore
    """Generate comprehensive interactive visualization with all components
    
    This function orchestrates all visualization components to create a unified
    interactive HTML visualization with dataflow, memory timeline, task comparison,
    and temporal animation capabilities.
    
    Args:
        trace_nodes: List of TraceNode objects from tracing
        analysis: Analysis results from TraceAnalyzer
        args: Command line arguments
        
    Returns:
        tuple: (html_content, visualization_data) for interactive visualization
    """
    if not INTERACTIVE_AVAILABLE or InteractiveConfig is None or InteractiveVisualizer is None:
        raise ImportError("Interactive visualization components are not available")
    
    print("Orchestrating interactive visualization components...")
    
    # 1. Create interactive configuration from CLI args
    interactive_config = InteractiveConfig(
        enable_zoom=True,
        enable_pan=True,
        enable_search=args.enable_search,
        enable_filters=args.enable_filtering,
        enable_tooltips=True,
        enable_export=True,
        max_nodes_visible=args.max_interactive_nodes,
        animation_duration_ms=750 if args.enable_animations else 0,
        color_scheme=args.color_scheme
    )
    
    # 2. Initialize core visualizer
    interactive_viz = InteractiveVisualizer(config=interactive_config)
    
    # 3. Apply filtering if needed
    filtered_nodes = trace_nodes
    if args.enable_filtering or args.filter_ops:
        if FilterEngine is None or SearchResult is None:
            print("Warning: FilterEngine not available, skipping filtering")
        else:
            filter_engine = FilterEngine()
            if args.filter_ops:
                # Filter by specific operations
                result = filter_engine.filter_by_operation(trace_nodes, args.filter_ops)
                # Handle both SearchResult and List[TraceNode] return types
                if SearchResult and isinstance(result, SearchResult):
                    filtered_nodes = result.nodes
                else:
                    filtered_nodes = result
            if args.memory_profile and 'filter_engine' in locals():
                # Ensure filtered_nodes is a list before passing to filter_by_memory
                if not isinstance(filtered_nodes, list):
                    if hasattr(filtered_nodes, 'nodes'):
                        filtered_nodes = filtered_nodes.nodes
                    else:
                        filtered_nodes = trace_nodes
                
                # Filter to show only high-memory operations
                result = filter_engine.filter_by_memory(
                    filtered_nodes,  # Now guaranteed to be a list
                    min_memory_mb=args.memory_threshold
                )
                # Handle both SearchResult and List[TraceNode] return types
                if SearchResult and isinstance(result, SearchResult):
                    filtered_nodes = result.nodes
                else:
                    filtered_nodes = result
            
            if args.task_heads and len(args.task_heads) < 5 and 'filter_engine' in locals():
                # Filter by specific task heads
                for head in args.task_heads:
                    # Ensure filtered_nodes is a list before each filter_by_module call
                    if not isinstance(filtered_nodes, list):
                        if hasattr(filtered_nodes, 'nodes'):
                            filtered_nodes = filtered_nodes.nodes
                        else:
                            filtered_nodes = trace_nodes
                    
                    result = filter_engine.filter_by_module(
                        filtered_nodes,  # Now guaranteed to be a list
                        module_pattern=f"*{head}*"
                    )
                    # Handle both SearchResult and List[TraceNode] return types
                    if SearchResult and isinstance(result, SearchResult):
                        filtered_nodes = result.nodes
                    else:
                        filtered_nodes = result
            
            # Ensure filtered_nodes is a list for final count
            if not isinstance(filtered_nodes, list):
                if hasattr(filtered_nodes, 'nodes'):
                    filtered_nodes = filtered_nodes.nodes
                else:
                    filtered_nodes = trace_nodes
            
            # Get the final node count - now guaranteed to be a list
            filtered_count = len(filtered_nodes)
            print(f"Filtered from {len(trace_nodes)} to {filtered_count} nodes")
    
    # Ensure filtered_nodes is always a list before using it
    if not isinstance(filtered_nodes, list):
        if hasattr(filtered_nodes, 'nodes'):  # SearchResult
            filtered_nodes = filtered_nodes.nodes
        elif hasattr(filtered_nodes, '__iter__'):
            try:
                filtered_nodes = list(filtered_nodes)
            except TypeError:
                filtered_nodes = []
        else:
            filtered_nodes = []
    
    # 4. Generate base interactive HTML
    # Check if the method exists
    if hasattr(interactive_viz, 'generate_interactive_html'):
        # Extract head_analysis and memory_profile from analysis dict
        head_analysis = analysis.get('head_analysis', {}) if isinstance(analysis, dict) else {}
        memory_profile = analysis.get('memory_profile', {}) if isinstance(analysis, dict) else {}
        
        interactive_html = interactive_viz.generate_interactive_html(
            trace_nodes=filtered_nodes,
            head_analysis=head_analysis,
            memory_profile=memory_profile
        )
    else:
        # Fallback to a simple HTML representation
        interactive_html = "<html><body><h1>Interactive visualization not fully implemented</h1></body></html>"
    
    # Initialize visualization data dictionary
    visualization_data = {
        'trace_nodes': len(filtered_nodes),
        'analysis': analysis,
        'config': interactive_config.__dict__ if hasattr(interactive_config, '__dict__') else {}
    }
    
    # 5. Add memory timeline if requested
    if args.show_memory_timeline or args.memory_profile:
        print("Adding memory timeline visualization...")
        if MemoryTimelineVisualizer is not None:
            # MemoryTimelineVisualizer expects MemoryTimelineEvent objects, not TraceNode
            # For now, just create an empty timeline
            timeline_data = {}
        else:
            timeline_data = {}
        visualization_data['memory_timeline'] = timeline_data
        
        # Integrate timeline into HTML if method exists
        # Note: integrate_component method not yet implemented in InteractiveVisualizer
        # if hasattr(interactive_viz, 'integrate_component') and timeline_data:
        #     interactive_html = interactive_viz.integrate_component(
        #         interactive_html,
        #         'memory_timeline',
        #         timeline_data
        #     )
    
    # 6. Add task head comparison if requested
    if args.show_task_comparison and len(args.task_heads) > 1:
        print("Adding task head comparison...")
        if TaskHeadComparator is not None:
            # TaskHeadComparator.compare_heads expects Dict[str, List[TraceNode]], not List[TraceNode]
            # For now, just create empty comparison data
            comparison_data = {}
        else:
            comparison_data = {}
        visualization_data['task_comparison'] = comparison_data
        
        # Integrate comparison into HTML if method exists
        # Note: integrate_component method not yet implemented in InteractiveVisualizer
        # if hasattr(interactive_viz, 'integrate_component') and comparison_data:
        #     interactive_html = interactive_viz.integrate_component(
        #         interactive_html,
        #         'task_comparison',
        #         comparison_data
        #     )
    
    # 7. Add temporal animation if requested
    if args.temporal_animation or args.visualize_temporal:
        print("Adding temporal flow animation...")
        if TemporalTracer is not None:
            temporal_tracer = TemporalTracer(
                queue_length=5 if args.stage == 1 else 3,
                stage=args.stage
            )
        else:
            temporal_tracer = None
        
        # Analyze temporal flow if temporal_tracer is available
        if temporal_tracer is not None:
            temporal_analysis = temporal_tracer.analyze_temporal_flow(filtered_nodes)
            animation_data = temporal_tracer.generate_animation_frames(filtered_nodes)
            visualization_data['temporal_animation'] = animation_data
            visualization_data['temporal_analysis'] = temporal_analysis
        else:
            temporal_analysis = None
            animation_data = None
        
        # Integrate animation into HTML if method exists
        # Note: integrate_component method not yet implemented in InteractiveVisualizer
        # if hasattr(interactive_viz, 'integrate_component') and animation_data is not None:
        #     interactive_html = interactive_viz.integrate_component(
        #         interactive_html,
        #         'temporal_animation',
        #         animation_data
        #     )
    
    # 8. Add queue visualization if requested
    if args.queue_visualization:
        print("Adding queue visualization...")
        if QueueVisualizer is not None:
            queue_viz = QueueVisualizer(stage=args.stage, config=interactive_config)
        else:
            queue_viz = None
        
        # Use temporal analysis if available, otherwise compute it
        if queue_viz is not None:
            temporal_analysis = visualization_data.get('temporal_analysis')
            queue_data = queue_viz.visualize_queue(filtered_nodes, temporal_analysis)
            visualization_data['queue_visualization'] = queue_data
        else:
            queue_data = None
        
        # Integrate queue visualization into HTML if method exists
        # Note: integrate_component method not yet implemented in InteractiveVisualizer
        # if hasattr(interactive_viz, 'integrate_component') and queue_data is not None:
        #     interactive_html = interactive_viz.integrate_component(
        #         interactive_html,
        #         'queue_visualization',
        #         queue_data
        #     )
    
    # 9. Add BEV-specific visualizations if requested
    if args.bev_focus:
        print("Adding BEV-focused visualizations...")
        bev_nodes = [node for node in filtered_nodes 
                    if hasattr(node, 'is_bev_operation') and node.is_bev_operation]
        
        if bev_nodes:
            # Create BEV-specific visualization
            bev_data = {
                'bev_nodes': len(bev_nodes),
                'bev_memory_mb': sum(node.memory_usage for node in bev_nodes),
                'bev_compute_ms': sum(node.compute_time for node in bev_nodes),
                'bev_operations': list(set(node.operation for node in bev_nodes))
            }
            visualization_data['bev_analysis'] = bev_data
            
            # Integrate BEV analysis into HTML if method exists
            # Note: integrate_component method not yet implemented in InteractiveVisualizer
            # if hasattr(interactive_viz, 'integrate_component'):
            #     interactive_html = interactive_viz.integrate_component(
            #         interactive_html,
            #         'bev_analysis',
            #         bev_data
            #     )
    
    # 10. Add export manager capabilities
    # Note: Export manager functionality will be integrated when needed
    # if ExportManager is not None:
    #     export_manager = ExportManager()
    #     # Add export buttons to the HTML if method exists
    #     # Note: generate_export_buttons method not yet implemented in ExportManager
    
    print("Interactive visualization orchestration complete!")
    print(f"  - Nodes: {len(filtered_nodes)}")
    print(f"  - Components: {len(visualization_data) - 2} active")
    print(f"  - Color scheme: {args.color_scheme}")
    
    return interactive_html, visualization_data


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
    
    # Interactive visualization options
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive HTML visualization')
    parser.add_argument('--export-html', type=str,
                        help='Export interactive visualization to HTML file')
    parser.add_argument('--color-scheme', type=str, default='default',
                        choices=['default', 'dark', 'colorblind', 'high-contrast'],
                        help='Color scheme for visualization')
    parser.add_argument('--enable-animations', action='store_true', default=True,
                        help='Enable animations in interactive visualization')
    parser.add_argument('--no-animations', dest='enable_animations', action='store_false',
                        help='Disable animations')
    parser.add_argument('--max-interactive-nodes', type=int, default=100,
                        help='Maximum nodes for interactive visualization')
    parser.add_argument('--enable-search', action='store_true', default=True,
                        help='Enable search functionality in interactive mode')
    parser.add_argument('--enable-filtering', action='store_true', default=True,
                        help='Enable filtering in interactive visualization')
    parser.add_argument('--show-memory-timeline', action='store_true',
                        help='Show memory timeline in interactive visualization')
    parser.add_argument('--show-task-comparison', action='store_true',
                        help='Show task head comparison in interactive mode')
    parser.add_argument('--temporal-animation', action='store_true',
                        help='Enable temporal flow animation')
    parser.add_argument('--queue-visualization', action='store_true',
                        help='Enable queue visualization for temporal processing')
    
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
    _ = tracer.trace(dummy_input)  # outputs not used
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
    
    visualizer: DataflowVisualizer = DataflowVisualizer()
    
    # Generate visualizations if methods exist
    # Note: state attribute not yet implemented in DataflowVisualizer
    # if hasattr(visualizer, 'state'):
    #     visualizer.state.show_dtype = args.show_dtype
    
    # DataflowVisualizer should have generate_mermaid method
    try:
        mermaid_diagram = visualizer.generate_mermaid(
            trace_nodes, 
            analysis['head_analysis'],
            analysis['memory_profile']
        )
    except (AttributeError, TypeError):
        mermaid_diagram = "```mermaid\ngraph TB\n  A[UniAD] --> B[Analysis]\n```"
    
    # DataflowVisualizer should have generate_memory_heatmap method
    try:
        memory_heatmap = visualizer.generate_memory_heatmap(analysis['memory_profile'])
    except (AttributeError, TypeError):
        memory_heatmap = "Memory heatmap not available"
    
    # Write output
    print(f"Writing output to {args.output}")
    with open(args.output, 'w') as f:
        f.write("# UniAD PyTorch Operation Trace Report\n\n")
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
            if isinstance(memory_heatmap, str):
                f.write(memory_heatmap)
            else:
                f.write("Memory profile not available")
            f.write("\n\n")
        
        # Shape Transformation Analysis
        if args.track_shape_changes or args.highlight_reshapes:
            f.write("## Shape Transformation Analysis\n\n")
            try:
                shape_report = visualizer.generate_shape_transformation_report(trace_nodes)
                f.write(shape_report)
            except (AttributeError, TypeError):
                f.write("Shape transformation analysis not available")
            f.write("\n\n")
        
        # Memory-based Auto-expansion View
        if args.expand_heavy_modules:
            f.write("## Memory-Based Module Expansion\n\n")
            try:
                memory_view = visualizer.generate_memory_based_view(trace_nodes, args.memory_threshold)
                f.write(memory_view)
            except (AttributeError, TypeError):
                f.write("Memory-based module expansion not available")
            f.write("\n\n")
        
        # Data Type Memory Analysis
        if args.dtype_memory_analysis:
            f.write("## Data Type Memory Impact Analysis\n\n")
            f.write("This section analyzes how different data types affect memory usage.\n\n")
            
            # Calculate memory impact for different precision levels
            from core.data_structures import UNIAD_DTYPE_CONFIGS
            
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
    
    # Interactive visualization (if enabled and available)
    if (args.interactive or args.export_html) and INTERACTIVE_AVAILABLE:
        # Use the orchestration function to generate comprehensive visualization
        interactive_html, _ = generate_interactive_visualization(
            trace_nodes, analysis, args
        )
        
        # Export HTML if path provided
        if args.export_html:
            if ExportManager is not None:
                export_manager = ExportManager()
                # Use the correct method signature
                # Note: There might be a type conflict between TraceNode classes
                try:
                    if hasattr(export_manager, 'export_interactive_html'):
                        export_path = export_manager.export_interactive_html(
                            trace_nodes,  # type: ignore # First argument
                            args.export_html,  # Second argument is path
                            analysis['head_analysis'],
                            analysis['memory_profile']
                        )
                        print(f"Interactive visualization exported to {export_path}")
                    else:
                        # Fallback: write directly
                        with open(args.export_html, 'w') as f:
                            f.write(interactive_html)
                        print(f"Interactive visualization exported to {args.export_html}")
                except (TypeError, AttributeError):
                    # Fallback: write directly
                    with open(args.export_html, 'w') as f:
                        f.write(interactive_html)
                    print(f"Interactive visualization exported to {args.export_html}")
        
        # Show interactive visualization if requested
        if args.interactive:
            # Save to temporary file and open in browser
            import tempfile
            import webbrowser
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                tmp.write(interactive_html)
                tmp_path = tmp.name
            
            print(f"Opening interactive visualization in browser: {tmp_path}")
            webbrowser.open(f"file://{tmp_path}")
    
    elif (args.interactive or args.export_html) and not INTERACTIVE_AVAILABLE:
        print("Warning: Interactive visualization requested but components not available.")
        print("Please ensure all visualization modules are properly installed.")
    
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