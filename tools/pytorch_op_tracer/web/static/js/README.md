# Interactive Connection Highlighting for UniAD Dataflow Visualization

This module provides advanced connection highlighting functionality for UniAD dataflow visualizations, allowing users to interactively explore data paths, transformations, and bottlenecks in the neural network architecture.

## Features

### Core Functionality
- **Connection Click Highlighting**: Click on any edge/connection to highlight the complete data path
- **Path Visualization**: Visual highlighting of entire data flows from source to destination
- **Transformation Details**: Detailed information about shape changes, memory usage, and operations
- **Multi-Path Support**: Highlight multiple paths simultaneously for comparison
- **Animation**: Smooth animated traversal showing data flow direction

### User Interactions
- **Edge Clicks**: Primary interaction - click edges to highlight data paths
- **Node Exploration**: Ctrl+Click on nodes to explore all connected paths
- **Context Menus**: Right-click for advanced options and path analysis
- **Keyboard Shortcuts**: ESC to clear, Ctrl+H for modes, Ctrl+A for animation

### Visual Features
- **Color-Coded Paths**: Different colors for multiple highlighted paths
- **Path Indices**: Numbered nodes showing sequence in data flow
- **Animation Effects**: Traveling dots and pulse animations
- **Memory Visualization**: Visual indicators of memory consumption along paths
- **Bottleneck Detection**: Automatic identification of memory/compute bottlenecks

## File Structure

```
web/static/js/
├── interactive.js          # Main connection highlighting implementation
├── test_interactive.html   # Test page demonstrating functionality
└── README.md              # This documentation file
```

## Integration

### Basic Integration

```javascript
// Initialize after D3.js visualization is ready
const highlighter = initializeConnectionHighlighter(
    svg,           // D3.js SVG selection
    nodeElements,  // D3.js node selection
    edgeElements,  // D3.js edge selection
    data,          // Graph data {nodes: [], edges: []}
    config         // Optional configuration
);
```

### Configuration Options

```javascript
const config = {
    animationSpeed: 750,              // Animation duration in ms
    enablePathTraversal: true,        // Enable path traversal animation
    enableMultipleHighlights: true,   // Allow multiple simultaneous highlights
    highlightColors: {                // Custom color scheme
        primary: '#ff6b35',
        secondary: '#4ecdc4',
        tertiary: '#45b7d1'
    }
};
```

### Template Integration

The module is automatically included in the base template and can be used in any dataflow visualization:

```html
<!-- In template head -->
<script src="static/js/interactive.js"></script>

<!-- In template body -->
<script>
// After creating D3.js visualization
const highlighter = initializeConnectionHighlighter(svg, nodes, edges, data);
</script>
```

## API Reference

### ConnectionHighlighter Class

#### Constructor
```javascript
new ConnectionHighlighter(svg, nodeElements, edgeElements, data, config)
```

#### Methods

##### Path Highlighting
- `highlightConnection(connectionId, edgeData)` - Highlight a specific connection
- `highlightDataPath(pathInfo, colorType)` - Highlight complete data path
- `clearAllHighlights()` - Clear all active highlights
- `clearConnectionHighlight(connectionId)` - Clear specific connection highlight

##### Animation
- `animateConnectionHighlight(edgeData)` - Animate connection with traveling dot
- `animateActivePaths()` - Animate all currently highlighted paths
- `animatePathTraversal(nodeIds)` - Animate traversal of specific path

##### Information Display
- `showTransformationDetails(edgeData, pathInfo)` - Show detailed transformation info
- `showConnectionPreview(event, edgeData)` - Show hover preview
- `hideTransformationDetails()` - Hide detail panel

##### Utility
- `findDataPath(sourceId, targetId)` - Find path between nodes
- `getTransformationInfo(edgeData)` - Get transformation details
- `exportPathData()` - Export highlighted paths as JSON

### Global Functions

These functions are available globally for template integration:

```javascript
clearHighlights()    // Clear all highlights
animatePaths()       // Animate active paths  
exportPathData()     // Export path data
```

## Data Requirements

### Node Data Structure
```javascript
{
    id: "unique_node_id",
    display_name: "Node Display Name", 
    task_head: "track|seg|motion|occ|planning|bev",
    memory_mb: 123.45,
    compute_ms: 67.89,
    size: 25,
    color: "#ff6b35",
    shape_info: {
        input_shape: [1, 256, 200, 200],
        output_shape: [1, 256, 200, 200]
    }
}
```

### Edge Data Structure
```javascript
{
    source: "source_node_id",  // or node object
    target: "target_node_id",  // or node object
    weight: 1.0,
    transformations: {
        shape_change: { from: [...], to: [...] },
        memory_change: 123.45
    }
}
```

## UniAD-Specific Features

### Task Head Integration
- Automatic task head detection (track, seg, motion, occ, planning, bev)
- Color coding based on UniAD architecture
- Task-specific path analysis

### BEV Operation Support
- Special handling for BEV encoder/decoder operations
- Spatial attention visualization
- Temporal frame analysis

### Memory Profiling
- Memory consumption tracking along paths
- Bottleneck identification
- Performance impact analysis

### Architecture Awareness
- UniAD-specific connection patterns
- Multi-task learning path visualization
- Stage 1 vs Stage 2 model differences

## Usage Examples

### Basic Usage
```javascript
// Initialize highlighter
const highlighter = initializeConnectionHighlighter(svg, nodes, edges, data);

// Highlight a specific path
highlighter.highlightConnection('bev_encoder-to-track_head', edgeData);

// Clear all highlights
highlighter.clearAllHighlights();
```

### Advanced Path Analysis
```javascript
// Find path between specific nodes
const pathInfo = highlighter.findDataPath('backbone', 'planning_head');

if (pathInfo) {
    // Highlight the complete path
    highlighter.highlightDataPath(pathInfo, 'primary');
    
    // Show transformation details
    highlighter.showTransformationDetails(edgeData, pathInfo);
    
    // Animate path traversal
    highlighter.animatePathTraversal(pathInfo.path_nodes);
}
```

### Multiple Path Comparison
```javascript
// Highlight multiple paths for comparison
const path1 = highlighter.findDataPath('bev_encoder', 'motion_head');
const path2 = highlighter.findDataPath('bev_encoder', 'occ_head');

highlighter.highlightDataPath(path1, 'primary');
highlighter.highlightDataPath(path2, 'secondary');
```

## Styling

### CSS Classes

The module adds several CSS classes that can be styled:

```css
/* Highlighted connections */
.highlighted {
    animation: pulse 2s infinite;
}

.path-highlighted {
    filter: drop-shadow(0 0 8px rgba(255, 107, 53, 0.6));
}

/* Path index labels */
.path-index {
    background: white;
    border: 1px solid #ccc;
    border-radius: 3px;
    padding: 2px 4px;
}

/* Detail panel */
.transformation-detail-panel {
    position: fixed;
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 20px;
    z-index: 1001;
}
```

### Custom Animations

```css
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

.connection-tracer {
    animation: travel 1s linear forwards;
}
```

## Testing

Open `test_interactive.html` in a web browser to see the functionality in action:

```bash
cd /home/tanl/UniAD/tools/pytorch_op_tracer/web/static/js
# Open test_interactive.html in browser
```

The test page includes:
- Sample UniAD-like graph structure
- Interactive controls
- Usage instructions
- Status feedback

## Browser Compatibility

- **Modern Browsers**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **Dependencies**: D3.js v7+ required
- **Features Used**: ES6 classes, arrow functions, async/await, Map/Set

## Performance Considerations

### Optimization Features
- **Efficient Path Finding**: BFS algorithm with caching
- **DOM Manipulation**: Minimal DOM updates during animation
- **Memory Management**: Automatic cleanup of temporary elements
- **Event Delegation**: Efficient event handling for large graphs

### Limitations
- **Large Graphs**: Performance may degrade with 1000+ nodes
- **Memory Usage**: Path caching may consume memory with complex graphs
- **Animation**: Multiple simultaneous animations may impact performance

## Troubleshooting

### Common Issues

1. **Highlighter not initializing**
   - Ensure D3.js is loaded before interactive.js
   - Check that nodeElements and edgeElements are valid D3 selections
   - Verify data structure matches expected format

2. **Paths not highlighting**
   - Check console for JavaScript errors
   - Verify edge data has correct source/target references
   - Ensure nodes have valid IDs

3. **Animations not working**
   - Check browser supports CSS animations and transforms
   - Verify animationSpeed configuration is valid
   - Check for CSS conflicts

### Debug Mode

Enable debug logging:

```javascript
const highlighter = new ConnectionHighlighter(svg, nodes, edges, data, {
    debug: true,
    animationSpeed: 1000
});
```

## Future Enhancements

### Planned Features
- **Path Comparison**: Side-by-side path analysis
- **Memory Heatmaps**: Visual memory consumption maps  
- **Export Formats**: SVG, PNG, PDF export of highlighted paths
- **Filter Integration**: Integration with existing filter systems
- **Touch Support**: Mobile device gesture support

### Extension Points
- Custom path finding algorithms
- Pluggable animation systems
- Custom visualization styles
- Integration with external analytics

## Contributing

When extending this module:

1. Follow the existing code style and patterns
2. Add JSDoc comments for new methods
3. Update tests for new functionality
4. Maintain browser compatibility
5. Document breaking changes

## License

This module is part of the UniAD project and follows the same license terms.