# Web Module - Interactive HTML Visualizations

This module provides HTML template engines for creating interactive web-based visualizations of UniAD dataflow analysis.

## Overview

The web module extends the PyTorch operation tracer with interactive HTML visualizations that can be viewed in any modern web browser. It supports multiple visualization types and includes comprehensive interactivity features.

## Key Features

- **Interactive Dataflow Graphs**: Pan, zoom, filter, and explore model operations
- **Memory Timeline Visualization**: Track memory usage patterns over time  
- **Multiple Export Formats**: HTML, PNG, and data export capabilities
- **Theme Support**: Default, dark, and UniAD-specific themes
- **Security Features**: Content Security Policy and input sanitization
- **Template Caching**: Performance optimization for repeated use
- **Responsive Design**: Works on desktop and mobile devices

## Components

### HTMLTemplateEngine

The main template engine class that handles HTML generation, data injection, and JavaScript library management.

```python
from web.template_engine import HTMLTemplateEngine, TemplateConfig

# Configure engine
config = TemplateConfig(
    theme="uniad",
    enable_caching=True,
    compression_level="moderate"
)

# Create engine
engine = HTMLTemplateEngine(config)
```

### Template Types

1. **Dataflow Template**: Interactive node-edge graph with filtering
2. **Memory Template**: Timeline visualization with Chart.js
3. **Comparison Template**: Side-by-side model comparison
4. **Dashboard Template**: Multi-view dashboard layout

### Configuration Options

```python
@dataclass
class TemplateConfig:
    enable_csp: bool = True              # Content Security Policy
    enable_caching: bool = True          # Template caching
    compression_level: str = "moderate"   # HTML compression
    theme: str = "uniad"                 # Visual theme
    include_d3js: bool = True            # D3.js for interactivity
    include_chartjs: bool = True         # Chart.js for timelines
    include_plotlyjs: bool = False       # Plotly for advanced charts
```

## Usage Examples

### Basic Dataflow Visualization

```python
from web.template_engine import create_dataflow_visualization

# Sample node and edge data
nodes = [
    {
        'id': 'node1',
        'display_name': 'BEVEncoder',
        'task_head': 'bev',
        'memory_mb': 2048.5,
        'color': '#3498db',
        'size': 25
    }
]

edges = [
    {'source': 'node1', 'target': 'node2'}
]

# Create visualization
html_content = create_dataflow_visualization(
    nodes, 
    edges, 
    title="UniAD Dataflow Analysis"
)

# Save to file
with open('visualization.html', 'w') as f:
    f.write(html_content)
```

### Memory Timeline Visualization

```python
from web.template_engine import create_memory_visualization

# Timeline data as (timestamp, memory_mb) tuples
timeline_points = [
    (0.0, 1024.0),
    (100.0, 2048.0),
    (200.0, 1536.0)
]

html_content = create_memory_visualization(
    timeline_points,
    "UniAD Memory Usage Timeline"
)
```

### Integration with Existing Tracers

```python
from web.integration_example import create_interactive_visualization_from_trace

# Convert existing trace nodes to interactive HTML
html_content = create_interactive_visualization_from_trace(
    trace_nodes,
    title="UniAD Interactive Analysis",
    output_path="analysis.html"
)
```

## Interactive Features

### Dataflow Visualization

- **Pan and Zoom**: Navigate large graphs with mouse or touch
- **Node Filtering**: Filter by task head, memory threshold, or search query
- **Expandable Nodes**: Click to expand/collapse hierarchical modules
- **Hover Tooltips**: Detailed information on hover
- **Export Options**: PNG export and data download

### Controls

- **Task Head Filters**: Toggle visibility of different task heads
- **Memory Threshold**: Hide nodes below memory threshold  
- **Search**: Find specific operations by name
- **View Controls**: Expand all, collapse all, reset zoom
- **Export**: Download visualization as PNG

### Responsive Design

- **Desktop**: Full sidebar with detailed controls
- **Mobile**: Collapsible sidebar and touch-friendly interface
- **Print**: Optimized for printing and PDF export

## Themes

### UniAD Theme (Default)
- Clean, professional appearance
- UniAD brand colors and typography
- Optimized for technical presentations

### Dark Theme
- Dark background with light text
- Reduced eye strain for long analysis sessions
- High contrast for better visibility

### Default Theme
- Standard web colors
- Good for general use and embedding

## Security Features

- **Content Security Policy**: Prevents XSS attacks
- **Input Sanitization**: All user data is escaped
- **Safe Template Injection**: Protected against code injection
- **Validation**: Input validation and type checking

## Performance Optimization

- **Template Caching**: Reuse compiled templates
- **HTML Compression**: Reduce file size
- **CDN Libraries**: Fast loading of JavaScript dependencies
- **Lazy Loading**: Load large datasets progressively

## Browser Compatibility

- **Modern Browsers**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **JavaScript**: ES6+ features used
- **Dependencies**: D3.js v7, Chart.js v4

## Files

- `template_engine.py`: Main template engine implementation
- `integration_example.py`: Integration with existing tracers
- `__init__.py`: Module exports and imports
- `README.md`: This documentation

## Integration with Existing Code

The web module is designed to work seamlessly with existing PyTorch operation tracers. Use the integration functions to convert trace data:

```python
# Convert trace nodes to HTML format
html_nodes = convert_trace_nodes_to_html_nodes(trace_nodes)

# Create dependency edges  
edges = create_edges_from_dependencies(html_nodes)

# Generate interactive visualization
html_content = create_dataflow_visualization(html_nodes, edges)
```

This preserves all existing functionality while adding interactive web capabilities.