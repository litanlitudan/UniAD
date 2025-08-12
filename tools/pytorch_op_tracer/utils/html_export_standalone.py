#!/usr/bin/env python3
"""
Standalone example of using the ExportManager with HTMLTemplateEngine

This file can be run directly from the command line:
    python html_export_standalone.py
"""

import sys
from pathlib import Path
from typing import Optional, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup (E402 is expected for this pattern)
from utils.export_manager import ExportManager  # noqa: E402
from core.data_structures import TraceNode  # noqa: E402

# Import TemplateConfig through the export_manager module to ensure type consistency
try:
    # Try to import from the same module that ExportManager uses
    from utils.export_manager import TemplateConfig  # noqa: E402, F401
except ImportError:
    # Fallback to direct import if not re-exported
    from web.template_engine import TemplateConfig  # noqa: E402


def create_sample_data():
    """Create sample data for demonstration"""
    # Create sample trace nodes
    trace_nodes = []
    
    # BEV operations
    for i in range(3):
        node = TraceNode(
            operation=f"BEVEncoder_layer_{i}",
            module_path=f"bev_encoder.layers.{i}",
            input_shapes=[],
            output_shapes=[],
            memory_usage=1500.0 + i * 100,
            compute_time=20.0 + i * 2,
            temporal_index=None
        )
        node.task_head = "bev"
        node.is_bev_operation = True
        trace_nodes.append(node)
    
    # Track head operations
    for i in range(2):
        node = TraceNode(
            operation=f"TrackHead_op_{i}",
            module_path=f"track_head.decoder.{i}",
            input_shapes=[],
            output_shapes=[],
            memory_usage=800.0 + i * 50,
            compute_time=10.0 + i,
            temporal_index=None
        )
        node.task_head = "track"
        trace_nodes.append(node)
    
    # Motion head operations
    for i in range(2):
        node = TraceNode(
            operation=f"MotionHead_op_{i}",
            module_path=f"motion_head.predictor.{i}",
            input_shapes=[],
            output_shapes=[],
            memory_usage=600.0 + i * 40,
            compute_time=8.0 + i,
            temporal_index=None
        )
        node.task_head = "motion"
        trace_nodes.append(node)
    
    # Planning head operations
    node = TraceNode(
        operation="PlanningHead_trajectory",
        module_path="planning_head.decoder",
        input_shapes=[],
        output_shapes=[],
        memory_usage=400.0,
        compute_time=5.0,
        temporal_index=None
    )
    node.task_head = "planning"
    trace_nodes.append(node)
    
    # Create head analysis
    head_analysis = {
        'memory_by_head': {
            'bev': 4800.0,
            'track': 1700.0,
            'motion': 1280.0,
            'planning': 400.0
        },
        'compute_by_head': {
            'bev': 66.0,
            'track': 21.0,
            'motion': 17.0,
            'planning': 5.0
        }
    }
    
    # Create memory profile
    memory_profile = {
        'total_memory_mb': 8180.0,
        'peak_memory_mb': 12000.0,
        'average_memory_mb': 6500.0,
        'top_consumers': [
            {'operation': 'BEVEncoder_layer_2', 'memory_mb': 1700.0, 'percentage': 20.8},
            {'operation': 'BEVEncoder_layer_1', 'memory_mb': 1600.0, 'percentage': 19.6},
            {'operation': 'BEVEncoder_layer_0', 'memory_mb': 1500.0, 'percentage': 18.3},
            {'operation': 'TrackHead_op_1', 'memory_mb': 850.0, 'percentage': 10.4},
            {'operation': 'TrackHead_op_0', 'memory_mb': 800.0, 'percentage': 9.8}
        ]
    }
    
    return trace_nodes, head_analysis, memory_profile


def main():
    """Main function demonstrating HTML export functionality"""
    print("HTMLTemplateEngine Export Example")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./html_exports")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create template configuration
    template_config = TemplateConfig(
        enable_csp=True,
        enable_caching=True,
        theme="uniad",
        include_d3js=True,
        include_chartjs=True,
        compression_level="moderate"
    )
    print("✓ Template configuration created")
    
    # Initialize ExportManager
    export_manager = ExportManager(
        output_dir=str(output_dir),
        stage=2,
        template_config=template_config
    )
    print("✓ ExportManager initialized")
    
    # Create sample data
    trace_nodes, head_analysis, memory_profile = create_sample_data()
    print(f"✓ Created {len(trace_nodes)} sample trace nodes")
    
    # Export interactive dataflow visualization
    print("\nExporting visualizations...")
    print("-" * 30)
    
    try:
        # 1. Interactive dataflow
        dataflow_path = export_manager.export_html_dataflow(
            trace_nodes=trace_nodes,
            path="uniad_dataflow.html",
            head_analysis=head_analysis,
            memory_profile=memory_profile,
            title="UniAD Stage 2 Dataflow Analysis"
        )
        print(f"✓ Dataflow visualization: {dataflow_path}")
    except Exception as e:
        print(f"✗ Dataflow export failed: {e}")
    
    try:
        # 2. Dashboard
        dashboard_path = export_manager.export_html_dashboard(
            trace_nodes=trace_nodes,
            path="uniad_dashboard.html",
            head_analysis=head_analysis,
            memory_profile=memory_profile,
            title="UniAD Analysis Dashboard"
        )
        print(f"✓ Dashboard: {dashboard_path}")
    except Exception as e:
        print(f"✗ Dashboard export failed: {e}")
    
    try:
        # 3. Batch export
        analysis_data = {
            'trace_nodes': trace_nodes,
            'head_analysis': head_analysis,
            'memory_profile': memory_profile
        }
        
        results = export_manager.export_batch(
            analysis_data=analysis_data,
            base_name="uniad_analysis",
            formats=['html', 'mermaid', 'md']
        )
        
        print("✓ Batch export completed:")
        for format_type, file_path in results.items():
            print(f"  - {format_type}: {file_path}")
    except Exception as e:
        print(f"✗ Batch export failed: {e}")
    
    print("\n" + "=" * 50)
    print("Export completed successfully!")
    print(f"View the HTML files in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()