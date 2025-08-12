"""
Example usage of the enhanced ExportManager with HTMLTemplateEngine

This example demonstrates how to use the new HTML export methods that leverage
the HTMLTemplateEngine for creating standalone interactive visualizations.
"""

# NOTE: This is an example file for documentation purposes.
# Due to relative import issues, it cannot be run directly as a script.
# Instead, it should be used as part of the pytorch_op_tracer package.

def example_html_export():
    """Example of how to use the enhanced HTML export functionality"""
    
    # This would be the typical usage when running from within the package context
    try:
        from ..utils.export_manager import ExportManager
        from ..web.template_engine import TemplateConfig
    except (ImportError, ValueError):
        # Fallback for different import contexts
        from utils.export_manager import ExportManager
        from web.template_engine import TemplateConfig
    
    # Create custom template configuration
    template_config = TemplateConfig(
        enable_csp=True,
        enable_caching=True,
        theme="uniad",  # Options: default, dark, light, uniad
        include_d3js=True,
        include_chartjs=True,
        compression_level="moderate"  # Options: none, moderate, aggressive
    )
    
    # Initialize ExportManager with enhanced HTML support
    export_manager = ExportManager(
        output_dir="./exports",
        stage=2,  # UniAD stage (1 or 2)
        template_config=template_config
    )
    
    # Example data (normally would come from actual tracing)
    trace_nodes = []  # List[TraceNode] from actual tracing
    head_analysis = {}  # From actual analysis
    memory_profile = {}  # From actual memory profiling
    
    # Export different visualization types:
    
    # 1. Interactive dataflow visualization
    export_manager.export_html_dataflow(
        trace_nodes=trace_nodes,
        path="uniad_dataflow.html",
        head_analysis=head_analysis,
        memory_profile=memory_profile,
        title="UniAD Stage 2 Dataflow Analysis"
    )
    
    # 2. Memory timeline visualization
    # memory_timeline = MemoryTimelineData(...)  # From actual memory tracing
    # export_manager.export_html_memory_timeline(
    #     memory_timeline=memory_timeline,
    #     path="uniad_memory_timeline.html",
    #     title="UniAD Memory Usage Timeline"
    # )
    
    # 3. Multi-view dashboard
    export_manager.export_html_dashboard(
        trace_nodes=trace_nodes,
        path="uniad_dashboard.html",
        head_analysis=head_analysis,
        memory_profile=memory_profile,
        title="UniAD Comprehensive Analysis Dashboard"
    )
    
    # 4. Model comparison
    models_data = [
        {
            'model_name': 'UniAD Stage 1',
            'trace_nodes': [],  # trace_nodes_stage1
            'head_analysis': {},  # head_analysis_stage1
            'memory_profile': {}  # memory_profile_stage1
        },
        {
            'model_name': 'UniAD Stage 2',
            'trace_nodes': trace_nodes,
            'head_analysis': head_analysis,
            'memory_profile': memory_profile
        }
    ]
    
    export_manager.export_html_comparison(
        models_data=models_data,
        path="uniad_comparison.html",
        title="UniAD Stage 1 vs Stage 2 Comparison"
    )
    
    # 5. Batch export with multiple formats
    analysis_data = {
        'trace_nodes': trace_nodes,
        'head_analysis': head_analysis,
        'memory_profile': memory_profile
    }
    
    results = export_manager.export_batch(
        analysis_data=analysis_data,
        base_name="uniad_analysis",
        formats=['html', 'mermaid', 'md']  # Will create multiple files
    )
    
    print("Export completed. Generated files:")
    for format_type, file_path in results.items():
        print(f"  {format_type}: {file_path}")


def html_export_features():
    """
    Features provided by the enhanced HTML export functionality:
    
    1. Multiple Visualization Types:
       - Dataflow: Interactive node-link diagrams with task head coloring
       - Memory Timeline: Chart.js-based memory usage over time
       - Dashboard: Multi-view interface with tabs for different analyses
       - Comparison: Side-by-side model comparison charts
    
    2. Interactive Features:
       - Zoom and pan with D3.js
       - Filtering by task head, memory threshold, search terms
       - Node expansion/collapse for hierarchical data
       - Hover tooltips with detailed information
       - Connection highlighting and path tracing
       - Export to PNG functionality
    
    3. Standalone Operation:
       - All JavaScript and CSS embedded inline
       - No external dependencies required
       - Works offline without CDN access
       - Content Security Policy support
    
    4. UniAD-Specific Features:
       - Task head color coding (track, seg, motion, occ, planning, bev)
       - Stage-specific layouts and analysis
       - Memory optimization recommendations
       - UniAD architectural understanding for connections
    
    5. Security & Performance:
       - Path sanitization to prevent directory traversal
       - HTML escaping and XSS prevention
       - Template caching for performance
       - Optional HTML compression
       - CSP headers for enhanced security
    
    6. Customization:
       - Multiple themes (default, dark, light, UniAD)
       - Custom CSS and JavaScript injection
       - Configurable node limits and display options
       - Custom titles and metadata
    """
    pass


if __name__ == "__main__":
    print("This is an example file showing how to use the enhanced ExportManager.")
    print("Run this code as part of the pytorch_op_tracer package to avoid import issues.")
    print("See the example_html_export() function for usage patterns.")