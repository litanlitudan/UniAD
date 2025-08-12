"""Web components for UniAD dataflow visualization

This package provides HTML template engines and web-based visualization
components for the UniAD PyTorch operation tracer.
"""

# Handle import issues when running as standalone module
try:
    from .template_engine import (
        HTMLTemplateEngine,
        TemplateConfig,
        VisualizationData,
        create_dataflow_visualization,
        create_memory_visualization
    )
except (ImportError, ValueError):
    # Handle case when run as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from template_engine import (
        HTMLTemplateEngine,
        TemplateConfig,
        VisualizationData,
        create_dataflow_visualization,
        create_memory_visualization
    )

__all__ = [
    'HTMLTemplateEngine',
    'TemplateConfig', 
    'VisualizationData',
    'create_dataflow_visualization',
    'create_memory_visualization'
]