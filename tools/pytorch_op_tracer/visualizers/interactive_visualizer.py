"""Interactive HTML visualizer for dataflow with collapsible hierarchies and hover tooltips"""

import json
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from ..core.data_structures import TraceNode, TensorInfo
    from ..core.visualization_config import InteractiveConfig
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode, TensorInfo
    from core.visualization_config import InteractiveConfig

# Import base visualizer with proper fallback handling
# Use type ignore to avoid type conflicts between different DataflowVisualizer implementations
try:
    from .mermaid_visualizer import DataflowVisualizer  # type: ignore
except (ImportError, ValueError):
    try:
        # Fallback to package-style import
        from visualizers.mermaid_visualizer import DataflowVisualizer  # type: ignore
    except (ImportError, ValueError):
        # If mermaid_visualizer has import issues, create a minimal fallback
        class DataflowVisualizer:  # type: ignore
            def __init__(self, max_nodes=50, show_shapes=True, shape_format='semantic'):
                self.max_nodes = max_nodes
                self.show_shapes = show_shapes
                self.shape_format = shape_format


class InteractiveVisualizer:
    """Generates interactive HTML visualizations from trace data with hierarchical navigation"""
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        """Initialize interactive visualizer
        
        Args:
            config: Interactive configuration options
        """
        self.config = config or InteractiveConfig.for_uniad_analysis()
        
        # Initialize base dataflow visualizer for data processing
        self.dataflow_visualizer = DataflowVisualizer(
            max_nodes=self.config.max_nodes_visible,
            show_shapes=True,
            shape_format='semantic'
        )
    
    def generate_interactive_html(self, trace_nodes: List[TraceNode], 
                                head_analysis: Dict[str, Any],
                                memory_profile: Dict[str, Any]) -> str:
        """Generate complete interactive HTML visualization
        
        Args:
            trace_nodes: List of traced operations
            head_analysis: Analysis results by task head
            memory_profile: Memory usage profile
            
        Returns:
            Complete HTML string with embedded JavaScript for interactivity
        """
        # Create visualization data
        viz_data = self.create_visualization_data(trace_nodes, head_analysis, memory_profile)
        
        # Generate HTML template with embedded data and controls
        html_template = self._get_html_template()
        
        # Embed data and configuration
        html_content = html_template.format(
            title="UniAD Dataflow Visualization",
            config_json=json.dumps(self.config.to_dict(), indent=2),
            data_json=json.dumps(viz_data, indent=2),
            css_styles=self._get_css_styles(),
            javascript_code=self._get_javascript_code()
        )
        
        return html_content
    
    def create_visualization_data(self, trace_nodes: List[TraceNode],
                                head_analysis: Dict[str, Any],
                                memory_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured visualization data from trace nodes
        
        Args:
            trace_nodes: List of traced operations
            head_analysis: Analysis results by task head
            memory_profile: Memory usage profile
            
        Returns:
            Structured data dictionary for visualization
        """
        # Build module hierarchy
        hierarchy = self._build_module_hierarchy(trace_nodes)
        
        # Create nodes for visualization
        nodes = []
        edges = []
        
        # Process hierarchy to create interactive nodes
        for module_name, module_nodes in hierarchy.items():
            node_data = self._create_node_data(module_name, module_nodes, head_analysis)
            nodes.append(node_data)
            
            # Create child nodes if expanded
            if self._should_show_children(module_name, module_nodes):
                children = self._create_child_nodes(module_name, module_nodes)
                nodes.extend(children)
        
        # Create edges between nodes
        edges = self._create_edges(trace_nodes, hierarchy)
        
        # Create summary statistics
        summary = self._create_summary_stats(trace_nodes, head_analysis, memory_profile)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'hierarchy': hierarchy,
            'summary': summary,
            'task_heads': self._extract_task_heads(head_analysis),
            'memory_profile': memory_profile,
            'config': self.config.to_dict()
        }
    
    def apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to visualization data
        
        Args:
            data: Original visualization data
            filters: Filter criteria
                - task_heads: List of task heads to show
                - memory_threshold: Minimum memory usage (MB)
                - search_query: Search string for node names
                - operation_types: List of operation types to show
                
        Returns:
            Filtered data dictionary
        """
        filtered_data = data.copy()
        nodes = data['nodes'].copy()
        
        # Apply task head filter
        if 'task_heads' in filters and filters['task_heads']:
            nodes = [n for n in nodes if n.get('task_head') in filters['task_heads']]
        
        # Apply memory threshold filter
        if 'memory_threshold' in filters and filters['memory_threshold'] > 0:
            nodes = [n for n in nodes if n.get('memory_mb', 0) >= filters['memory_threshold']]
        
        # Apply search query filter
        if 'search_query' in filters and filters['search_query']:
            query = filters['search_query'].lower()
            nodes = [n for n in nodes if query in n.get('display_name', '').lower() 
                    or query in n.get('operation', '').lower()]
        
        # Apply operation type filter
        if 'operation_types' in filters and filters['operation_types']:
            nodes = [n for n in nodes if n.get('operation_type') in filters['operation_types']]
        
        # Update edges to only include filtered nodes
        node_ids = {n['id'] for n in nodes}
        edges = [e for e in data['edges'] if e['source'] in node_ids and e['target'] in node_ids]
        
        filtered_data.update({
            'nodes': nodes,
            'edges': edges,
            'filtered_count': len(nodes),
            'total_count': len(data['nodes'])
        })
        
        return filtered_data
    
    def _build_module_hierarchy(self, trace_nodes: List[TraceNode]) -> Dict[str, List[TraceNode]]:
        """Build hierarchical structure from trace nodes"""
        hierarchy = defaultdict(list)
        
        for node in trace_nodes[:self.config.max_nodes_visible]:
            # Group by top-level module for hierarchical display
            parts = node.module_path.split('.')
            if parts:
                top_module = parts[0]
                hierarchy[top_module].append(node)
        
        return hierarchy
    
    def _create_node_data(self, module_name: str, module_nodes: List[TraceNode],
                         head_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create node data for a module"""
        # Calculate aggregate metrics
        total_memory = sum(n.memory_usage for n in module_nodes)
        total_compute = sum(n.compute_time for n in module_nodes)
        operation_count = len(module_nodes)
        
        # Determine task head
        task_head = self._determine_task_head(module_name, module_nodes)
        
        # Create position (will be calculated by layout algorithm)
        position = self._calculate_node_position(module_name, task_head)
        
        # Determine node color based on configuration
        color = self._get_node_color(module_name, task_head, total_memory)
        
        # Create tooltip data
        tooltip_data = self._create_tooltip_data(module_name, module_nodes, total_memory, total_compute)
        
        # Determine if node should be expanded by default
        expanded = self._should_expand_module(module_name, module_nodes)
        
        return {
            'id': self._safe_id(module_name),
            'display_name': module_name,
            'type': 'module',
            'task_head': task_head,
            'operation_count': operation_count,
            'memory_mb': total_memory,
            'compute_ms': total_compute,
            'position': position,
            'color': color,
            'size': self._calculate_node_size(total_memory, operation_count),
            'expanded': expanded,
            'tooltip_data': tooltip_data,
            'has_children': operation_count > 1,
            'children_ids': [self._safe_id(f"{module_name}.{node.operation}") 
                           for node in module_nodes] if expanded else []
        }
    
    def _create_child_nodes(self, parent_module: str, module_nodes: List[TraceNode]) -> List[Dict[str, Any]]:
        """Create child nodes for expanded modules"""
        children = []
        
        for i, node in enumerate(module_nodes[:10]):  # Limit children for performance
            child_id = f"{parent_module}.{node.operation}.{i}"
            
            # Create tooltip data for individual operation
            tooltip_data = {
                'operation': node.operation,
                'module_path': node.module_path,
                'memory_mb': node.memory_usage,
                'compute_ms': node.compute_time,
                'input_shapes': [self._format_tensor_info(t) for t in node.input_shapes],
                'output_shapes': [self._format_tensor_info(t) for t in node.output_shapes],
                'shape_transform': node.shape_transform,
                'is_frozen': node.is_frozen,
                'temporal_index': node.temporal_index
            }
            
            child_data = {
                'id': self._safe_id(child_id),
                'display_name': node.operation,
                'type': 'operation',
                'parent_id': self._safe_id(parent_module),
                'task_head': node.task_head,
                'operation': node.operation,
                'memory_mb': node.memory_usage,
                'compute_ms': node.compute_time,
                'position': self._calculate_child_position(parent_module, i),
                'color': self._get_operation_color(node),
                'size': self._calculate_node_size(node.memory_usage, 1),
                'tooltip_data': tooltip_data,
                'has_children': False,
                'shape_info': self._get_shape_transformation_info(node)
            }
            
            children.append(child_data)
        
        return children
    
    def _create_edges(self, trace_nodes: List[TraceNode], 
                     hierarchy: Dict[str, List[TraceNode]]) -> List[Dict[str, Any]]:
        """Create edges between nodes based on dependencies"""
        edges = []
        
        # Create high-level module connections
        module_connections = self._get_module_connections(hierarchy)
        
        for source, target in module_connections:
            edge_data = {
                'id': f"{source}_to_{target}",
                'source': self._safe_id(source),
                'target': self._safe_id(target),
                'type': 'module_connection',
                'weight': 1.0
            }
            edges.append(edge_data)
        
        return edges
    
    def _get_module_connections(self, hierarchy: Dict[str, List[TraceNode]]) -> List[Tuple[str, str]]:
        """Determine connections between modules based on UniAD architecture"""
        connections = []
        
        # UniAD-specific connections based on architectural knowledge
        module_names = list(hierarchy.keys())
        
        # Check for common UniAD components and their dependencies
        bev_modules = [m for m in module_names if 'bev' in m.lower() or 'encoder' in m.lower()]
        track_modules = [m for m in module_names if 'track' in m.lower()]
        seg_modules = [m for m in module_names if 'seg' in m.lower() or 'panseg' in m.lower()]
        motion_modules = [m for m in module_names if 'motion' in m.lower()]
        occ_modules = [m for m in module_names if 'occ' in m.lower()]
        planning_modules = [m for m in module_names if 'planning' in m.lower()]
        
        # BEV encoder feeds into track and segmentation
        for bev_mod in bev_modules:
            for track_mod in track_modules:
                connections.append((bev_mod, track_mod))
            for seg_mod in seg_modules:
                connections.append((bev_mod, seg_mod))
        
        # Track head feeds into motion and occupancy
        for track_mod in track_modules:
            for motion_mod in motion_modules:
                connections.append((track_mod, motion_mod))
            for occ_mod in occ_modules:
                connections.append((track_mod, occ_mod))
        
        # Motion and occupancy feed into planning
        for motion_mod in motion_modules:
            for planning_mod in planning_modules:
                connections.append((motion_mod, planning_mod))
        
        for occ_mod in occ_modules:
            for planning_mod in planning_modules:
                connections.append((occ_mod, planning_mod))
        
        return connections
    
    def _create_summary_stats(self, trace_nodes: List[TraceNode],
                            head_analysis: Dict[str, Any],
                            memory_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for the visualization"""
        return {
            'total_nodes': len(trace_nodes),
            'visible_nodes': min(len(trace_nodes), self.config.max_nodes_visible),
            'total_memory_mb': memory_profile.get('total_memory_mb', 0),
            'total_compute_ms': sum(n.compute_time for n in trace_nodes),
            'task_head_count': len(head_analysis.get('memory_by_head', {})),
            'memory_by_head': head_analysis.get('memory_by_head', {}),
            'compute_by_head': head_analysis.get('compute_by_head', {})
        }
    
    def _extract_task_heads(self, head_analysis: Dict[str, Any]) -> List[str]:
        """Extract available task heads from analysis"""
        return list(head_analysis.get('memory_by_head', {}).keys())
    
    def _should_show_children(self, module_name: str, module_nodes: List[TraceNode]) -> bool:
        """Determine if children should be shown for a module"""
        return len(module_nodes) > 1 and len(module_nodes) <= 20  # Show children for small modules
    
    def _should_expand_module(self, module_name: str, module_nodes: List[TraceNode]) -> bool:
        """Determine if module should be expanded by default"""
        # Expand high-memory modules or modules with interesting operations
        total_memory = sum(n.memory_usage for n in module_nodes)
        return total_memory > 1000 or any('attention' in n.operation.lower() for n in module_nodes)
    
    def _determine_task_head(self, module_name: str, module_nodes: List[TraceNode]) -> Optional[str]:
        """Determine task head for a module"""
        # Check explicit task head assignments
        task_heads = [n.task_head for n in module_nodes if n.task_head]
        if task_heads:
            return task_heads[0]
        
        # Infer from module name
        module_lower = module_name.lower()
        if 'track' in module_lower:
            return 'track'
        elif 'seg' in module_lower or 'panseg' in module_lower:
            return 'seg'
        elif 'motion' in module_lower:
            return 'motion'
        elif 'occ' in module_lower:
            return 'occ'
        elif 'planning' in module_lower:
            return 'planning'
        elif 'bev' in module_lower:
            return 'bev'
        
        return None
    
    def _calculate_node_position(self, module_name: str, task_head: Optional[str]) -> Tuple[float, float]:
        """Calculate initial node position based on task head and hierarchy"""
        # Position nodes in a rough layout based on UniAD architecture
        base_positions = {
            'bev': (400, 100),
            'track': (200, 300),
            'seg': (600, 300),
            'motion': (200, 500),
            'occ': (400, 500),
            'planning': (400, 700)
        }
        
        if task_head and task_head in base_positions:
            return base_positions[task_head]
        
        # Default position with some randomization based on module name hash
        hash_val = hash(module_name) % 1000
        return (300 + hash_val % 400, 300 + hash_val % 300)
    
    def _calculate_child_position(self, parent_module: str, child_index: int) -> Tuple[float, float]:
        """Calculate position for child nodes relative to parent"""
        parent_pos = self._calculate_node_position(parent_module, None)
        offset_x = (child_index % 3) * 60 - 60  # Grid layout
        offset_y = (child_index // 3) * 40 + 50
        return (parent_pos[0] + offset_x, parent_pos[1] + offset_y)
    
    def _get_node_color(self, module_name: str, task_head: Optional[str], memory_mb: float) -> str:
        """Get node color based on configuration scheme"""
        if self.config.color_scheme == "memory":
            # Get palette or use default colors
            palette = self.config.get_color_palette() if hasattr(self.config, 'get_color_palette') else {}
            if memory_mb > 5000:
                return palette.get("high_memory", "#ff4444") if isinstance(palette, dict) else "#ff4444"
            elif memory_mb > 1000:
                return palette.get("medium_memory", "#ffaa00") if isinstance(palette, dict) else "#ffaa00"
            elif memory_mb > 100:
                return palette.get("low_memory", "#44ff44") if isinstance(palette, dict) else "#44ff44"
            else:
                return palette.get("no_memory", "#cccccc") if isinstance(palette, dict) else "#cccccc"
        
        elif self.config.color_scheme == "operation":
            # Use task-based coloring
            task_colors = {
                'track': '#ff9999',
                'seg': '#99ccff',
                'motion': '#9999ff',
                'occ': '#ffcc99',
                'planning': '#99ff99',
                'bev': '#ffccff'
            }
            return task_colors.get(task_head or 'default', "#95a5a6")
        
        return "#34495e"  # Default
    
    def _get_operation_color(self, node: TraceNode) -> str:
        """Get color for individual operation nodes"""
        if self.config.color_scheme == "operation":
            palette = self.config.get_color_palette() if hasattr(self.config, 'get_color_palette') else {}
            op_lower = node.operation.lower()
            
            if 'conv' in op_lower:
                return palette.get("conv", "#ff6b6b") if isinstance(palette, dict) else "#ff6b6b"
            elif 'linear' in op_lower or 'matmul' in op_lower:
                return palette.get("linear", "#4ecdc4") if isinstance(palette, dict) else "#4ecdc4"
            elif 'attention' in op_lower:
                return palette.get("attention", "#45b7d1") if isinstance(palette, dict) else "#45b7d1"
            elif 'relu' in op_lower or 'gelu' in op_lower:
                return palette.get("activation", "#96ceb4") if isinstance(palette, dict) else "#96ceb4"
            elif 'norm' in op_lower:
                return palette.get("normalization", "#feca57") if isinstance(palette, dict) else "#feca57"
            elif 'pool' in op_lower:
                return palette.get("pooling", "#ff9ff3") if isinstance(palette, dict) else "#ff9ff3"
            else:
                return palette.get("other", "#95a5a6") if isinstance(palette, dict) else "#95a5a6"
        
        # Default memory-based coloring
        return self._get_node_color("", node.task_head, node.memory_usage)
    
    def _calculate_node_size(self, memory_mb: float, operation_count: int) -> float:
        """Calculate node size based on memory and operation count"""
        # Size between 10 and 100 based on memory usage
        memory_factor = min(memory_mb / 1000, 10)  # Normalize to 0-10 range
        count_factor = min(operation_count / 10, 5)  # Normalize to 0-5 range
        return max(15, 15 + memory_factor * 5 + count_factor * 3)
    
    def _create_tooltip_data(self, module_name: str, module_nodes: List[TraceNode],
                           memory_mb: float, compute_ms: float) -> Dict[str, Any]:
        """Create comprehensive tooltip data for a module"""
        # Aggregate shape information
        input_shapes = []
        output_shapes = []
        
        for node in module_nodes[:5]:  # Sample first 5 nodes
            if node.input_shapes:
                input_shapes.extend([self._format_tensor_info(t) for t in node.input_shapes[:2]])
            if node.output_shapes:
                output_shapes.extend([self._format_tensor_info(t) for t in node.output_shapes[:2]])
        
        # Get unique operations
        operations = list(set(node.operation for node in module_nodes))[:10]
        
        return {
            'module_name': module_name,
            'operation_count': len(module_nodes),
            'total_memory_mb': memory_mb,
            'total_compute_ms': compute_ms,
            'operations': operations,
            'sample_input_shapes': input_shapes[:5],
            'sample_output_shapes': output_shapes[:5],
            'frozen_operations': sum(1 for n in module_nodes if n.is_frozen),
            'bev_operations': sum(1 for n in module_nodes if n.is_bev_operation),
            'shape_transforms': [n.shape_transform for n in module_nodes if n.shape_transform][:5]
        }
    
    def _format_tensor_info(self, tensor_info: TensorInfo) -> str:
        """Format tensor information for display"""
        if not tensor_info:
            return ""
        
        shape_str = f"[{','.join(map(str, tensor_info.shape))}]"
        dtype_str = tensor_info.dtype.replace('float', 'fp').replace('bfloat', 'bf')
        return f"{shape_str}@{dtype_str}"
    
    def _get_shape_transformation_info(self, node: TraceNode) -> Optional[Dict[str, Any]]:
        """Get shape transformation information for a node"""
        if not (node.input_shapes and node.output_shapes and node.shape_transform):
            return None
        
        in_shape = node.input_shapes[0]
        out_shape = node.output_shapes[0]
        
        return {
            'type': node.shape_transform,
            'input_shape': self._format_tensor_info(in_shape),
            'output_shape': self._format_tensor_info(out_shape),
            'memory_change_mb': (out_shape.memory_size() - in_shape.memory_size()) / (1024 * 1024)
        }
    
    def _safe_id(self, name: str) -> str:
        """Create safe ID for HTML/JavaScript"""
        return name.replace('.', '_').replace('/', '_').replace(' ', '_')
    
    def _get_html_template(self) -> str:
        """Get the base HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div id="visualization-container">
        <div id="controls-panel">
            <div class="control-group">
                <h3>Filters</h3>
                <div class="filter-controls">
                    <label>Task Heads:</label>
                    <div id="task-head-filters"></div>
                    <label>Memory Threshold (MB):</label>
                    <input type="range" id="memory-threshold" min="0" max="10000" value="0">
                    <span id="memory-threshold-value">0 MB</span>
                    <label>Search:</label>
                    <input type="text" id="search-input" placeholder="Search nodes...">
                </div>
            </div>
            <div class="control-group">
                <h3>View Options</h3>
                <div class="view-controls">
                    <button id="expand-all-btn">Expand All</button>
                    <button id="collapse-all-btn">Collapse All</button>
                    <button id="reset-zoom-btn">Reset Zoom</button>
                    <button id="export-btn">Export PNG</button>
                </div>
            </div>
            <div class="control-group">
                <h3>Summary</h3>
                <div id="summary-stats"></div>
            </div>
        </div>
        <div id="visualization-main">
            <svg id="graph-svg"></svg>
            <div id="tooltip"></div>
        </div>
    </div>

    <script>
        // Embed configuration and data
        const CONFIG = {config_json};
        const DATA = {data_json};
        
        {javascript_code}
    </script>
</body>
</html>'''
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the interactive visualization"""
        return '''
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }

        #visualization-container {
            display: flex;
            height: 100vh;
        }

        #controls-panel {
            width: 300px;
            background-color: #ffffff;
            border-right: 1px solid #dee2e6;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 4px rgba(0,0,0,0.1);
        }

        #visualization-main {
            flex: 1;
            position: relative;
            overflow: hidden;
        }

        #graph-svg {
            width: 100%;
            height: 100%;
            background-color: #ffffff;
        }

        .control-group {
            margin-bottom: 30px;
        }

        .control-group h3 {
            margin: 0 0 15px 0;
            color: #495057;
            font-size: 16px;
            font-weight: 600;
        }

        .filter-controls, .view-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .filter-controls label {
            font-weight: 500;
            color: #6c757d;
            margin-top: 10px;
        }

        .filter-controls input[type="range"] {
            width: 100%;
        }

        .filter-controls input[type="text"] {
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }

        #task-head-filters {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .task-filter-checkbox {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
        }

        .view-controls button {
            padding: 8px 12px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }

        .view-controls button:hover {
            background-color: #0056b3;
        }

        #summary-stats {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }

        .summary-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .summary-label {
            color: #6c757d;
            font-weight: 500;
        }

        .summary-value {
            color: #495057;
            font-weight: 600;
        }

        /* Graph nodes and edges */
        .node {
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .node:hover {
            stroke-width: 3px;
            filter: brightness(1.1);
        }

        .node.collapsed {
            opacity: 0.7;
        }

        .node.highlighted {
            stroke: #ff6b35;
            stroke-width: 4px;
        }

        .edge {
            fill: none;
            stroke: #adb5bd;
            stroke-width: 2px;
            marker-end: url(#arrowhead);
        }

        .edge.highlighted {
            stroke: #ff6b35;
            stroke-width: 3px;
        }

        .node-label {
            font-size: 12px;
            font-weight: 600;
            text-anchor: middle;
            pointer-events: none;
            fill: #212529;
        }

        .child-node {
            opacity: 0.9;
        }

        .child-node .node-label {
            font-size: 10px;
            font-weight: 400;
        }

        /* Tooltip */
        #tooltip {
            position: absolute;
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            max-width: 300px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }

        #tooltip.visible {
            opacity: 1;
        }

        .tooltip-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #ffc107;
        }

        .tooltip-section {
            margin-bottom: 6px;
        }

        .tooltip-label {
            font-weight: 500;
            color: #adb5bd;
        }

        .tooltip-value {
            color: #ffffff;
        }

        /* Zoom controls */
        .zoom-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .zoom-btn {
            width: 40px;
            height: 40px;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #dee2e6;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            color: #495057;
            transition: all 0.2s;
        }

        .zoom-btn:hover {
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Loading indicator */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            font-size: 18px;
            color: #6c757d;
        }
        '''
    
    def _get_javascript_code(self) -> str:
        """Get JavaScript code for interactivity"""
        return '''
        // Interactive visualization implementation
        class InteractiveDataflowVisualization {
            constructor(config, data) {
                this.config = config;
                this.data = data;
                this.filteredData = data;
                this.currentTransform = d3.zoomIdentity;
                
                this.init();
            }
            
            init() {
                this.setupSVG();
                this.setupTooltip();
                this.setupControls();
                this.setupSummary();
                this.renderVisualization();
            }
            
            setupSVG() {
                this.svg = d3.select("#graph-svg");
                const container = d3.select("#visualization-main");
                
                this.width = container.node().offsetWidth;
                this.height = container.node().offsetHeight;
                
                // Create zoom behavior
                this.zoom = d3.zoom()
                    .scaleExtent([0.1, 3])
                    .on("zoom", (event) => {
                        this.currentTransform = event.transform;
                        this.mainGroup.attr("transform", event.transform);
                    });
                
                this.svg.call(this.zoom);
                
                // Main group for all elements
                this.mainGroup = this.svg.append("g");
                
                // Define arrowhead marker
                this.svg.append("defs").append("marker")
                    .attr("id", "arrowhead")
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 8)
                    .attr("refY", 0)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .attr("fill", "#adb5bd");
                    
                // Add zoom controls
                this.addZoomControls();
            }
            
            setupTooltip() {
                this.tooltip = d3.select("#tooltip");
            }
            
            setupControls() {
                // Task head filters
                const taskHeadContainer = d3.select("#task-head-filters");
                this.data.task_heads.forEach(taskHead => {
                    const checkbox = taskHeadContainer.append("div")
                        .attr("class", "task-filter-checkbox");
                    
                    checkbox.append("input")
                        .attr("type", "checkbox")
                        .attr("id", `task-${taskHead}`)
                        .attr("checked", true)
                        .on("change", () => this.applyFilters());
                    
                    checkbox.append("label")
                        .attr("for", `task-${taskHead}`)
                        .text(taskHead);
                });
                
                // Memory threshold
                d3.select("#memory-threshold")
                    .on("input", (event) => {
                        const value = event.target.value;
                        d3.select("#memory-threshold-value").text(`${value} MB`);
                        this.applyFilters();
                    });
                
                // Search input
                d3.select("#search-input")
                    .on("input", () => this.applyFilters());
                
                // Control buttons
                d3.select("#expand-all-btn").on("click", () => this.expandAll());
                d3.select("#collapse-all-btn").on("click", () => this.collapseAll());
                d3.select("#reset-zoom-btn").on("click", () => this.resetZoom());
                d3.select("#export-btn").on("click", () => this.exportPNG());
            }
            
            setupSummary() {
                const summary = this.data.summary;
                const summaryContainer = d3.select("#summary-stats");
                
                const items = [
                    { label: "Total Nodes", value: summary.total_nodes.toLocaleString() },
                    { label: "Visible Nodes", value: summary.visible_nodes.toLocaleString() },
                    { label: "Total Memory", value: `${summary.total_memory_mb.toFixed(1)} MB` },
                    { label: "Total Compute", value: `${summary.total_compute_ms.toFixed(1)} ms` },
                    { label: "Task Heads", value: summary.task_head_count }
                ];
                
                items.forEach(item => {
                    const div = summaryContainer.append("div")
                        .attr("class", "summary-item");
                    
                    div.append("span")
                        .attr("class", "summary-label")
                        .text(item.label);
                        
                    div.append("span")
                        .attr("class", "summary-value")
                        .text(item.value);
                });
            }
            
            applyFilters() {
                // Collect filter values
                const filters = {
                    task_heads: [],
                    memory_threshold: +d3.select("#memory-threshold").node().value,
                    search_query: d3.select("#search-input").node().value
                };
                
                // Get selected task heads
                this.data.task_heads.forEach(taskHead => {
                    if (d3.select(`#task-${taskHead}`).node().checked) {
                        filters.task_heads.push(taskHead);
                    }
                });
                
                // Apply filters using the existing method
                this.filteredData = this.applyDataFilters(this.data, filters);
                
                // Re-render
                this.renderVisualization();
            }
            
            applyDataFilters(data, filters) {
                let nodes = [...data.nodes];
                
                // Apply task head filter
                if (filters.task_heads.length > 0) {
                    nodes = nodes.filter(n => 
                        !n.task_head || filters.task_heads.includes(n.task_head));
                }
                
                // Apply memory threshold
                if (filters.memory_threshold > 0) {
                    nodes = nodes.filter(n => n.memory_mb >= filters.memory_threshold);
                }
                
                // Apply search query
                if (filters.search_query) {
                    const query = filters.search_query.toLowerCase();
                    nodes = nodes.filter(n =>
                        n.display_name.toLowerCase().includes(query) ||
                        (n.operation && n.operation.toLowerCase().includes(query))
                    );
                }
                
                // Filter edges
                const nodeIds = new Set(nodes.map(n => n.id));
                const edges = data.edges.filter(e =>
                    nodeIds.has(e.source) && nodeIds.has(e.target));
                
                return {
                    ...data,
                    nodes,
                    edges
                };
            }
            
            renderVisualization() {
                // Clear existing elements
                this.mainGroup.selectAll("*").remove();
                
                if (this.filteredData.nodes.length === 0) {
                    this.showMessage("No nodes match current filters");
                    return;
                }
                
                // Create force simulation
                this.simulation = d3.forceSimulation(this.filteredData.nodes)
                    .force("link", d3.forceLink(this.filteredData.edges)
                        .id(d => d.id)
                        .distance(150))
                    .force("charge", d3.forceManyBody().strength(-800))
                    .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                    .force("collision", d3.forceCollide().radius(d => d.size + 5));
                
                this.renderEdges();
                this.renderNodes();
                
                // Start simulation
                this.simulation.on("tick", () => this.updatePositions());
            }
            
            renderEdges() {
                this.edgeGroup = this.mainGroup.append("g").attr("class", "edges");
                
                this.edges = this.edgeGroup.selectAll(".edge")
                    .data(this.filteredData.edges)
                    .enter().append("line")
                    .attr("class", "edge")
                    .attr("stroke-width", d => d.weight * 2);
            }
            
            renderNodes() {
                this.nodeGroup = this.mainGroup.append("g").attr("class", "nodes");
                
                // Create node groups
                this.nodeElements = this.nodeGroup.selectAll(".node-group")
                    .data(this.filteredData.nodes)
                    .enter().append("g")
                    .attr("class", "node-group")
                    .call(this.createDragBehavior());
                
                // Add circles
                this.nodeElements.append("circle")
                    .attr("class", "node")
                    .attr("r", d => d.size)
                    .attr("fill", d => d.color)
                    .attr("stroke", "#ffffff")
                    .attr("stroke-width", 2)
                    .on("click", (event, d) => this.handleNodeClick(event, d))
                    .on("mouseover", (event, d) => this.showTooltip(event, d))
                    .on("mouseout", () => this.hideTooltip());
                
                // Add labels
                this.nodeElements.append("text")
                    .attr("class", "node-label")
                    .attr("dy", "0.3em")
                    .text(d => this.truncateLabel(d.display_name))
                    .style("font-size", d => `${Math.min(12, d.size / 3)}px`);
                
                // Add child nodes if expanded
                this.renderChildNodes();
            }
            
            renderChildNodes() {
                const expandedNodes = this.filteredData.nodes.filter(d => d.expanded && d.has_children);
                
                expandedNodes.forEach(parentNode => {
                    const children = this.filteredData.nodes.filter(d => d.parent_id === parentNode.id);
                    
                    children.forEach((child, i) => {
                        const childGroup = this.nodeElements.filter(d => d.id === child.id);
                        
                        childGroup.select("circle")
                            .attr("class", "node child-node")
                            .attr("r", child.size * 0.8);
                        
                        childGroup.select("text")
                            .attr("class", "node-label child-node")
                            .style("font-size", "10px");
                    });
                });
            }
            
            updatePositions() {
                this.edges
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                this.nodeElements
                    .attr("transform", d => `translate(${d.x},${d.y})`);
            }
            
            createDragBehavior() {
                return d3.drag()
                    .on("start", (event, d) => {
                        if (!event.active) this.simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    })
                    .on("drag", (event, d) => {
                        d.fx = event.x;
                        d.fy = event.y;
                    })
                    .on("end", (event, d) => {
                        if (!event.active) this.simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    });
            }
            
            handleNodeClick(event, d) {
                if (d.has_children) {
                    d.expanded = !d.expanded;
                    this.renderVisualization();
                }
            }
            
            showTooltip(event, d) {
                const tooltip = d3.select("#tooltip");
                const tooltipData = d.tooltip_data;
                
                let content = `<div class="tooltip-title">${d.display_name}</div>`;
                
                if (tooltipData.operation_count) {
                    content += `<div class="tooltip-section">
                        <span class="tooltip-label">Operations:</span> 
                        <span class="tooltip-value">${tooltipData.operation_count}</span>
                    </div>`;
                }
                
                content += `<div class="tooltip-section">
                    <span class="tooltip-label">Memory:</span> 
                    <span class="tooltip-value">${tooltipData.total_memory_mb?.toFixed(1) || d.memory_mb.toFixed(1)} MB</span>
                </div>`;
                
                content += `<div class="tooltip-section">
                    <span class="tooltip-label">Compute:</span> 
                    <span class="tooltip-value">${tooltipData.total_compute_ms?.toFixed(1) || d.compute_ms.toFixed(1)} ms</span>
                </div>`;
                
                if (tooltipData.operations && tooltipData.operations.length > 0) {
                    content += `<div class="tooltip-section">
                        <span class="tooltip-label">Sample Operations:</span><br>
                        <span class="tooltip-value">${tooltipData.operations.slice(0, 3).join(', ')}</span>
                    </div>`;
                }
                
                if (tooltipData.sample_input_shapes && tooltipData.sample_input_shapes.length > 0) {
                    content += `<div class="tooltip-section">
                        <span class="tooltip-label">Input Shapes:</span><br>
                        <span class="tooltip-value">${tooltipData.sample_input_shapes.slice(0, 2).join(', ')}</span>
                    </div>`;
                }
                
                tooltip.html(content)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px")
                    .classed("visible", true);
            }
            
            hideTooltip() {
                d3.select("#tooltip").classed("visible", false);
            }
            
            truncateLabel(text, maxLength = 15) {
                if (text.length <= maxLength) return text;
                return text.substring(0, maxLength - 3) + "...";
            }
            
            expandAll() {
                this.filteredData.nodes.forEach(d => {
                    if (d.has_children) d.expanded = true;
                });
                this.renderVisualization();
            }
            
            collapseAll() {
                this.filteredData.nodes.forEach(d => {
                    if (d.has_children) d.expanded = false;
                });
                this.renderVisualization();
            }
            
            resetZoom() {
                this.svg.transition()
                    .duration(750)
                    .call(this.zoom.transform, d3.zoomIdentity);
            }
            
            addZoomControls() {
                const controls = d3.select("#visualization-main")
                    .append("div")
                    .attr("class", "zoom-controls");
                
                controls.append("div")
                    .attr("class", "zoom-btn")
                    .text("+")
                    .on("click", () => {
                        this.svg.transition().call(
                            this.zoom.scaleBy, 1.5
                        );
                    });
                
                controls.append("div")
                    .attr("class", "zoom-btn")
                    .text("−")
                    .on("click", () => {
                        this.svg.transition().call(
                            this.zoom.scaleBy, 1 / 1.5
                        );
                    });
            }
            
            exportPNG() {
                // Simple export implementation
                const svgElement = document.getElementById("graph-svg");
                const svgData = new XMLSerializer().serializeToString(svgElement);
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                const img = new Image();
                
                canvas.width = this.width;
                canvas.height = this.height;
                
                img.onload = () => {
                    ctx.fillStyle = "white";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                    
                    const link = document.createElement("a");
                    link.download = "uniad-dataflow.png";
                    link.href = canvas.toDataURL();
                    link.click();
                };
                
                img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
            }
            
            showMessage(message) {
                this.mainGroup.append("text")
                    .attr("x", this.width / 2)
                    .attr("y", this.height / 2)
                    .attr("text-anchor", "middle")
                    .style("font-size", "18px")
                    .style("fill", "#6c757d")
                    .text(message);
            }
        }
        
        // Initialize visualization when DOM is loaded
        document.addEventListener("DOMContentLoaded", () => {
            new InteractiveDataflowVisualization(CONFIG, DATA);
        });
        '''