#!/usr/bin/env python3
"""Generate reports for UniAD models with both standard and text-based loading approaches"""

import os
import subprocess
import sys
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Report configurations for full UniAD configs
UNIAD_REPORTS = [
    {
        'name': 'stage1_track_map',
        'config': 'projects/configs/stage1_track_map/base_track_map.py',
        'stage': 1,
        'checkpoint': None,  # Add if available
        'options': ['--memory-profile', '--bev-focus', '--show-shapes', '--show-dtype']
    },
    {
        'name': 'stage2_e2e',
        'config': 'projects/configs/stage2_e2e/base_e2e.py',
        'stage': 2,
        'checkpoint': None,  # Add if available
        'options': ['--memory-profile', '--bev-focus', '--show-shapes', '--show-dtype', '--task-heads', 'track', 'motion', 'planning']
    },
    {
        'name': 'stage2_e2e_full',
        'config': 'projects/configs/stage2_e2e/base_e2e.py',
        'stage': 2,
        'checkpoint': None,
        'options': ['--memory-profile', '--bev-focus', '--show-shapes', '--show-dtype', '--expand-modules', 'BEVFormer', '--visualization-mode', 'expanded']
    },
    {
        'name': 'stage2_memory_analysis',
        'config': 'projects/configs/stage2_e2e/base_e2e.py',
        'stage': 2,
        'checkpoint': None,
        'options': ['--memory-profile', '--dtype-memory-analysis', '--track-dtype', '--mixed-precision', 'fp16']
    },
    {
        'name': 'stage2_temporal',
        'config': 'projects/configs/stage2_e2e/base_e2e.py',
        'stage': 2,
        'checkpoint': None,
        'options': ['--visualize-temporal', '--temporal-frames', '3', '--show-shapes']
    }
]

# Mock model configurations for testing
MOCK_CONFIGS = {
    'mock_uniad_track': """
model = dict(
    type='UniADTrack',
    num_classes=10,
    embed_dims=256,
    num_query=900,
    queue_length=3,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
)
""",
    'mock_uniad_full': """
model = dict(
    type='UniAD',
    num_classes=10,
    embed_dims=256,
    num_query=900,
    queue_length=3,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    task_loss_weight=dict(
        track=1.0,
        map=1.0,
        motion=1.0,
        occ=1.0,
        planning=1.0
    ),
    seg_head=dict(type='PansegformerHead'),
    motion_head=dict(type='MotionHead'),
    occ_head=dict(type='OccHead'),
    planning_head=dict(type='PlanningHeadSingleMode'),
)
"""
}


def setup_environment():
    """Setup the environment and paths"""
    # Change to the tracer directory
    tracer_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tracer_dir)
    
    # Add tracer directory to Python path
    if tracer_dir not in sys.path:
        sys.path.insert(0, tracer_dir)
    
    return tracer_dir


def run_trace_command(config: str, output: str, stage: int, checkpoint: Optional[str] = None, 
                     options: Optional[List[str]] = None, text_load: bool = False) -> bool:
    """Run the trace_ops.py script with given parameters"""
    cmd = [
        sys.executable,
        'trace_ops.py',
        '--config', config,
        '--output', output,
        '--stage', str(stage),
        '--device', 'cpu'  # Use CPU to avoid GPU memory issues
    ]
    
    if checkpoint:
        cmd.extend(['--checkpoint', checkpoint])
    
    if text_load:
        cmd.append('--text-load')
    
    if options:
        cmd.extend(options)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running trace: {result.stderr}")
        return False
    
    print(f"Successfully generated: {output}")
    return True


def trace_model_direct(model: Any, device: str = 'cpu', stage: int = 2) -> Tuple[List[Any], Dict[str, Any]]:
    """Directly trace a model and return analysis"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from utils import create_dummy_input
    from core.tracer import OperationTracer
    from analyzers.trace_analyzer import TraceAnalyzer
    
    # Create dummy input
    if hasattr(model, 'cfg'):
        dummy_input = create_dummy_input(model.cfg, device)
    else:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Create tracer
    tracer = OperationTracer(model, stage=stage)
    
    # Perform tracing
    print("Tracing model...")
    outputs = tracer.trace(dummy_input)
    trace_nodes = tracer.get_trace_data()
    print(f"Traced {len(trace_nodes)} operations")
    
    # Analyze trace
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes, stage=stage)
    
    return trace_nodes, analysis


def generate_trace_report(trace_nodes: List[Any], analysis: Dict[str, Any], 
                         output_path: str, title: str, approach: str):
    """Generate a trace report from analysis data"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from visualizers.mermaid_visualizer import MermaidVisualizer as DataflowVisualizer
    
    visualizer = DataflowVisualizer(
        show_shapes=True,
        shape_format='full',
        visualization_mode='top-level'
    )
    
    # Generate visualizations
    mermaid_diagram = visualizer.generate_mermaid(
        trace_nodes, 
        analysis.get('head_analysis', {}),  # Provide empty dict as default
        analysis.get('memory_profile', {})  # Also handle memory_profile safely
    )
    memory_heatmap = visualizer.generate_memory_heatmap(analysis.get('memory_profile', {}))
    
    with open(output_path, 'w') as f:
        f.write(f"# {title} - {approach} Approach\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        summary = analysis['summary']
        f.write(f"- Total Operations: {summary['total_operations']}\n")
        f.write(f"- Unique Operations: {summary['unique_operations']}\n")
        f.write(f"- Total Memory: {summary['total_memory_mb']:.1f} MB\n")
        f.write(f"- Total Compute Time: {summary['total_compute_ms']:.1f} ms\n\n")
        
        # Dataflow
        f.write("## Dataflow Visualization\n\n")
        f.write(mermaid_diagram)
        f.write("\n\n")
        
        # Memory Profile
        f.write("## Memory Profile\n\n")
        f.write(memory_heatmap)
        f.write("\n\n")
        
        # Operation Distribution
        f.write("## Operation Distribution\n\n")
        op_dist = analysis.get('operation_distribution', {})
        for op, count in sorted(op_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"- {op}: {count}\n")
        f.write("\n")


def generate_mock_model_reports():
    """Generate reports for mock models using direct tracing"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from utils.model_utils import load_uniad_model_from_text
    
    print("\n=== Generating Mock Model Reports ===\n")
    
    # Create output directory
    os.makedirs('reports/comparison/mock_models', exist_ok=True)
    
    for name, config_content in MOCK_CONFIGS.items():
        # Write config to temporary file
        config_file = f'temp_{name}_config.py'
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        try:
            # Load model with text-based approach
            print(f"\nLoading {name} model...")
            model, cfg = load_uniad_model_from_text(config_file, device='cpu')
            
            # Determine stage based on model type
            stage = 1 if 'Track' in name else 2
            
            # Trace the model
            trace_nodes, analysis = trace_model_direct(model, device='cpu', stage=stage)
            
            # Generate report
            output_path = f'reports/comparison/mock_models/{name}_report.md'
            generate_trace_report(trace_nodes, analysis, output_path, 
                                name.replace('_', ' ').title(), "Text-based Loading")
            print(f"Report generated: {output_path}")
            
        finally:
            # Clean up
            if os.path.exists(config_file):
                os.unlink(config_file)


def generate_uniad_reports():
    """Generate reports for actual UniAD configs using both approaches"""
    print("\n=== Generating UniAD Reports ===\n")
    
    # Create output directories
    output_dir = 'reports/comparison'
    mmdet3d_dir = os.path.join(output_dir, 'mmdet3d')
    text_load_dir = os.path.join(output_dir, 'text_load')
    
    os.makedirs(mmdet3d_dir, exist_ok=True)
    os.makedirs(text_load_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Track results
    results = {
        'mmdet3d': {'success': 0, 'failed': 0},
        'text_load': {'success': 0, 'failed': 0}
    }
    
    # First, try with standard mmdet3d approach
    print("\n--- Generating reports with standard mmdet3d approach ---\n")
    for report in UNIAD_REPORTS:
        output_path = os.path.join(mmdet3d_dir, f"{report['name']}_{timestamp}.md")
        success = run_trace_command(
            config=report['config'],
            output=output_path,
            stage=report['stage'],
            checkpoint=report.get('checkpoint'),
            options=report.get('options'),
            text_load=False
        )
        if success:
            results['mmdet3d']['success'] += 1
        else:
            results['mmdet3d']['failed'] += 1
    
    # Then, try with text-based loading
    print("\n--- Generating reports with text-based loading approach ---\n")
    for report in UNIAD_REPORTS:
        output_path = os.path.join(text_load_dir, f"{report['name']}_{timestamp}.md")
        success = run_trace_command(
            config=report['config'],
            output=output_path,
            stage=report['stage'],
            checkpoint=report.get('checkpoint'),
            options=report.get('options'),
            text_load=True
        )
        if success:
            results['text_load']['success'] += 1
        else:
            results['text_load']['failed'] += 1
    
    return results, timestamp


def generate_summary_report(results: Dict[str, Dict[str, int]], timestamp: str):
    """Generate a summary report comparing both approaches"""
    print("\n=== Generating Summary Report ===\n")
    
    summary_path = f'reports/comparison/summary_{timestamp}.md'
    with open(summary_path, 'w') as f:
        f.write("# Report Generation Summary\n\n")
        f.write(f"Generated at: {timestamp}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares the standard mmdet3d model loading approach ")
        f.write("with the new text-based loading approach.\n\n")
        
        f.write("## Results\n\n")
        f.write("### Standard mmdet3d Approach\n")
        f.write(f"- Successfully generated: {results['mmdet3d']['success']}\n")
        f.write(f"- Failed: {results['mmdet3d']['failed']}\n\n")
        
        f.write("### Text-based Loading Approach\n")
        f.write(f"- Successfully generated: {results['text_load']['success']}\n")
        f.write(f"- Failed: {results['text_load']['failed']}\n\n")
        
        f.write("## Key Differences\n\n")
        f.write("### 1. Model Construction\n")
        f.write("- **mmdet3d approach**: Uses global registry system with `@DETECTORS.register_module()`\n")
        f.write("- **Text-based approach**: Directly parses config and constructs model hierarchy\n\n")
        
        f.write("### 2. Dependencies\n")
        f.write("- **mmdet3d approach**: Requires full mmdet3d installation and all dependencies\n")
        f.write("- **Text-based approach**: Can work with mock models when dependencies are missing\n\n")
        
        f.write("### 3. Flexibility\n")
        f.write("- **mmdet3d approach**: Fixed to registered components\n")
        f.write("- **Text-based approach**: Can be easily modified or extended\n\n")
        
        f.write("### 4. Use Cases\n")
        f.write("- **mmdet3d approach**: Production training and inference\n")
        f.write("- **Text-based approach**: Architecture analysis, debugging, and prototyping\n\n")
        
        f.write("## Report Files\n\n")
        f.write("### UniAD Config Reports\n")
        f.write("#### mmdet3d Reports\n")
        for report in UNIAD_REPORTS:
            f.write(f"- `{report['name']}_{timestamp}.md`\n")
        
        f.write("\n#### Text-load Reports\n")
        for report in UNIAD_REPORTS:
            f.write(f"- `{report['name']}_{timestamp}.md`\n")
        
        f.write("\n### Mock Model Reports\n")
        for name in MOCK_CONFIGS.keys():
            f.write(f"- `{name}_report.md`\n")
        
        f.write("\n## Configuration Details\n\n")
        for report in UNIAD_REPORTS:
            f.write(f"### {report['name']}\n")
            f.write(f"- Config: `{report['config']}`\n")
            f.write(f"- Stage: {report['stage']}\n")
            f.write(f"- Options: `{' '.join(report.get('options', []))}`\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The text-based loading approach provides a lightweight alternative for model analysis ")
        f.write("and debugging, especially useful when:\n")
        f.write("- Full dependencies are not available\n")
        f.write("- Quick architecture analysis is needed\n")
        f.write("- Custom modifications to model construction are required\n")
        f.write("- Understanding the model structure from config files\n")
    
    print(f"Summary report written to: {summary_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate UniAD model reports')
    parser.add_argument('--mock-only', action='store_true', 
                        help='Generate only mock model reports')
    parser.add_argument('--uniad-only', action='store_true',
                        help='Generate only UniAD config reports')
    parser.add_argument('--skip-summary', action='store_true',
                        help='Skip summary report generation')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    print("="*60)
    print(f"UniAD Report Generation")
    print(f"Generated at: {datetime.now()}")
    print("="*60)
    
    # Default: generate both if neither flag is set
    generate_mock = not args.uniad_only
    generate_uniad = not args.mock_only
    
    results = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Generate mock model reports
        if generate_mock:
            generate_mock_model_reports()
        
        # Generate UniAD reports
        if generate_uniad:
            results, timestamp = generate_uniad_reports()
        
        # Generate summary report
        if not args.skip_summary and results:
            generate_summary_report(results, timestamp)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        if results:
            print(f"Standard mmdet3d: {results['mmdet3d']['success']} success, {results['mmdet3d']['failed']} failed")
            print(f"Text-based load:  {results['text_load']['success']} success, {results['text_load']['failed']} failed")
        if generate_mock:
            print(f"Mock models: {len(MOCK_CONFIGS)} reports generated")
        print("="*60 + "\n")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nNote: Some functionality requires proper Python path setup.")
        print("Try running from the pytorch_op_tracer directory or installing as a package.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())