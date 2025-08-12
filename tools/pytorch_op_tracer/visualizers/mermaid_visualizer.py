"""Mermaid diagram visualizer for dataflow"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
import torch.nn as nn

try:
    from ..core.data_structures import TraceNode
    from ..core.hierarchy_analyzer import ModuleHierarchyAnalyzer
    from ..core.visualization_state import VisualizationState
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode
    from core.hierarchy_analyzer import ModuleHierarchyAnalyzer
    from core.visualization_state import VisualizationState


class DataflowVisualizer:
    """Generates Mermaid diagrams from trace data with hierarchical visualization"""
    
    def __init__(self, max_nodes: int = 50, visualization_mode: str = 'top-level',
                 expand_modules: Optional[List[str]] = None, expand_heavy_modules: bool = False,
                 show_shapes: bool = True, shape_format: str = 'full',
                 track_shape_changes: bool = False, model: Optional[nn.Module] = None,
                 memory_threshold: Optional[float] = None):
        self.max_nodes = max_nodes
        
        # Initialize visualization state
        self.state = VisualizationState()
        self.state.visualization_mode = visualization_mode
        self.state.show_shapes = show_shapes
        self.state.shape_format = shape_format
        self.state.highlight_shape_changes = track_shape_changes
        
        # Set up module expansion
        if expand_modules:
            self.state.expanded_modules = set(expand_modules)
        if expand_heavy_modules and memory_threshold:
            self.state.memory_threshold = memory_threshold
        elif expand_heavy_modules:
            self.state.memory_threshold = 1000.0  # Default 1GB
        
        # Initialize hierarchy analyzer if model is provided
        self.hierarchy_analyzer = ModuleHierarchyAnalyzer(model) if model else None
    
    def _format_shape(self, tensor_info) -> str:
        """Format tensor shape based on display format"""
        if not self.state.show_shapes or not tensor_info:
            return ""
            
        shape = tensor_info.shape if hasattr(tensor_info, 'shape') else tensor_info
        dtype = ""
        
        # Extract dtype if available and show_dtype is enabled
        if self.state.show_dtype and hasattr(tensor_info, 'dtype'):
            dtype_str = tensor_info.dtype.replace('float', 'fp').replace('bfloat', 'bf')
            dtype = f"@{dtype_str}"
        
        if self.state.shape_format == 'compact':
            # Compact format: (B,C,H,W)@dtype
            dims = ','.join(map(str, shape))
            return f"({dims}){dtype}"
        elif self.state.shape_format == 'semantic' and hasattr(tensor_info, 'semantic_dims') and tensor_info.semantic_dims:
            # Semantic format with dimension labels
            dims = []
            for i, dim in enumerate(shape):
                label = tensor_info.semantic_dims.get(i, '')
                if label:
                    dims.append(f"{label}={dim}")
                else:
                    dims.append(str(dim))
            return f"[{', '.join(dims)}]{dtype}"
        else:
            # Full format
            return f"[{' × '.join(map(str, shape))}]{dtype}"
    
    def _build_module_hierarchy(self, trace_nodes: List[TraceNode]) -> Dict[str, List[TraceNode]]:
        """Build module hierarchy from trace nodes"""
        hierarchy = defaultdict(list)
        
        for node in trace_nodes:
            # Extract module path components
            parts = node.module_path.split('.')
            
            # Group by top-level module
            if parts:
                top_module = parts[0]
                hierarchy[top_module].append(node)
                
                # Also track sub-modules if expanded
                if self.state.visualization_mode == 'expanded' or top_module in self.state.expanded_modules:
                    if len(parts) > 1:
                        sub_module = '.'.join(parts[:2])
                        hierarchy[sub_module].append(node)
        
        return hierarchy
    
    def _should_expand_module(self, module_name: str, module_nodes: List[TraceNode]) -> bool:
        """Determine if a module should be expanded"""
        # Calculate module info
        total_memory = sum(node.memory_usage for node in module_nodes)
        module_info = {
            'memory': total_memory,
            'operations': len(module_nodes),
            'has_children': len(module_nodes) > 1
        }
        
        # Use visualization state to determine expansion
        return self.state.should_expand_module(module_name, module_info)
    
    def generate_mermaid(self, trace_nodes: List[TraceNode], 
                        head_analysis: Dict[str, Any],
                        memory_profile: Dict[str, Any]) -> str:
        """Generate hierarchical Mermaid diagram from trace data"""
        lines = ["```mermaid", "graph TB"]
        
        # Build module hierarchy
        hierarchy = self._build_module_hierarchy(trace_nodes[:self.max_nodes])
        
        if self.state.visualization_mode == 'top-level' and not self.state.expanded_modules:
            # Top-level view
            self._generate_top_level_view(lines, hierarchy, head_analysis, memory_profile)
        else:
            # Expanded or full view
            self._generate_expanded_view(lines, hierarchy, trace_nodes[:self.max_nodes])
        
        # Add shape transformation highlights if enabled
        if self.state.highlight_shape_changes:
            self._add_shape_transformations(lines, trace_nodes[:self.max_nodes])
        
        lines.append("```")
        
        return '\n'.join(lines)
    
    def _generate_top_level_view(self, lines: List[str], hierarchy: Dict[str, List[TraceNode]], 
                                head_analysis: Dict[str, Any], memory_profile: Dict[str, Any]):
        """Generate top-level module view"""
        # Group by major components
        components = {
            'BEVFormer': [],
            'TrackHead': [],
            'SegHead': [],
            'MotionHead': [],
            'OccHead': [],
            'PlanningHead': []
        }
        
        # Aggregate nodes by component
        for module, nodes in hierarchy.items():
            for comp in components:
                if comp.lower() in module.lower():
                    components[comp].extend(nodes)
                    break
        
        # Add input node
        if self.state.show_shapes:
            lines.append('    Input["Multi-Camera Input<br/>[6, 3, 928, 1600]"]')
        else:
            lines.append('    Input["Multi-Camera Input"]')
        
        # Add BEVFormer
        if components['BEVFormer']:
            memory = sum(n.memory_usage for n in components['BEVFormer'])
            compute = sum(n.compute_time for n in components['BEVFormer'])
            label = f"BEVFormer Encoder<br/>Memory: {memory:.1f}MB<br/>Time: {compute:.1f}ms"
            if self.state.show_shapes:
                label += "<br/>Output: [1, 256, 200, 200]"
            lines.append(f'    BEVFormer["{label}"]')
            lines.append('    Input --> BEVFormer')
        
        # Add task heads
        head_info = head_analysis.get('memory_by_head', {})
        compute_info = head_analysis.get('compute_by_head', {})
        
        task_heads = [
            ('TrackHead', 'track', 'Track'),
            ('SegHead', 'seg', 'Segmentation'),
            ('MotionHead', 'motion', 'Motion'),
            ('OccHead', 'occ', 'Occupancy'),
            ('PlanningHead', 'planning', 'Planning')
        ]
        
        for comp_key, head_key, display_name in task_heads:
            if components[comp_key] or head_key in head_info:
                memory = head_info.get(head_key, 0)
                compute = compute_info.get(head_key, 0)
                label = f"{display_name} Head<br/>Memory: {memory:.1f}MB<br/>Time: {compute:.1f}ms"
                
                # Add output shapes based on task type
                if self.state.show_shapes:
                    if head_key == 'track':
                        label += "<br/>Output: [300, 10] boxes"
                    elif head_key == 'seg':
                        label += "<br/>Output: [200, 200] segmap"
                    elif head_key == 'motion':
                        label += "<br/>Output: [N, 6, 12, 2] trajectories"
                    elif head_key == 'occ':
                        label += "<br/>Output: [200, 200, 5] occupancy"
                    elif head_key == 'planning':
                        label += "<br/>Output: [1, 6, 2] ego trajectory"
                
                lines.append(f'    {comp_key}["{label}"]')
        
        # Add dependencies
        lines.append('')
        lines.append('    BEVFormer --> TrackHead')
        lines.append('    BEVFormer --> SegHead')
        lines.append('    TrackHead --> MotionHead')
        lines.append('    TrackHead --> OccHead')
        lines.append('    MotionHead --> PlanningHead')
        lines.append('    OccHead --> PlanningHead')
        
        # Add styling
        lines.append('')
        lines.append('    style TrackHead fill:#f9f,stroke:#333,stroke-width:2px')
        lines.append('    style MotionHead fill:#bbf,stroke:#333,stroke-width:2px')
        lines.append('    style PlanningHead fill:#bfb,stroke:#333,stroke-width:2px')
        lines.append('    style BEVFormer fill:#ffd,stroke:#333,stroke-width:2px')
    
    def _generate_expanded_view(self, lines: List[str], hierarchy: Dict[str, List[TraceNode]], 
                               trace_nodes: List[TraceNode]):
        """Generate expanded module view"""
        # Track node IDs for connections
        node_map = {}
        
        # Process each module
        for module_name, module_nodes in sorted(hierarchy.items()):
            if not module_nodes:
                continue
                
            # Check if this module should be expanded
            if self._should_expand_module(module_name, module_nodes):
                # Create subgraph for expanded module
                safe_name = module_name.replace('.', '_')
                lines.append(f'    subgraph "{module_name}"')
                
                # Add individual operations
                for i, node in enumerate(module_nodes[:10]):  # Limit to 10 ops per module
                    node_id = f"{safe_name}_op{i}"
                    node_map[node.node_id] = node_id
                    
                    # Create label with operation and shapes
                    label = node.operation
                    if self.state.show_shapes and node.input_shapes and node.output_shapes:
                        in_shape = self._format_shape(node.input_shapes[0])
                        out_shape = self._format_shape(node.output_shapes[0])
                        label += f"<br/>{in_shape} → {out_shape}"
                    
                    if node.memory_usage > 100:  # Show memory for heavy ops
                        label += f"<br/>Mem: {node.memory_usage:.0f}MB"
                    
                    lines.append(f'        {node_id}["{label}"]')
                
                lines.append('    end')
                lines.append('')
            else:
                # Create aggregated node
                safe_name = module_name.replace('.', '_')
                total_memory = sum(n.memory_usage for n in module_nodes)
                
                label = f"{module_name}<br/>Ops: {len(module_nodes)}<br/>Memory: {total_memory:.1f}MB"
                if self.state.show_shapes and module_nodes:
                    # Show aggregate shape transformation
                    first_input = module_nodes[0].input_shapes[0] if module_nodes[0].input_shapes else None
                    last_output = module_nodes[-1].output_shapes[0] if module_nodes[-1].output_shapes else None
                    if first_input and last_output:
                        in_shape = self._format_shape(first_input)
                        out_shape = self._format_shape(last_output)
                        label += f"<br/>{in_shape} → {out_shape}"
                
                lines.append(f'    {safe_name}["{label}"]')
                
                # Map all nodes in this module to the aggregated node
                for node in module_nodes:
                    node_map[node.node_id] = safe_name
        
        # Add connections based on dependencies
        connections = set()
        for node in trace_nodes:
            if node.node_id in node_map:
                src = node_map[node.node_id]
                for dep in node.feeds_into:
                    if dep in node_map:
                        dst = node_map[dep]
                        if src != dst:  # Avoid self-loops
                            connections.add((src, dst))
        
        lines.append('')
        for src, dst in sorted(connections):
            lines.append(f'    {src} --> {dst}')
    
    def _add_shape_transformations(self, lines: List[str], trace_nodes: List[TraceNode]):
        """Add highlights for shape transformations"""
        lines.append('')
        lines.append('    %% Shape transformations')
        
        # Track different types of transformations
        transformations = {
            'reshape': [],
            'flatten': [],
            'permute': [],
            'squeeze': [],
            'unsqueeze': [],
            'view': []
        }
        
        for node in trace_nodes:
            if node.shape_transform:
                transform_type = node.shape_transform.lower()
                if transform_type in transformations:
                    transformations[transform_type].append(node)
        
        # Apply different styles for different transformations
        style_map = {
            'reshape': 'fill:#faa,stroke:#f00,stroke-width:3px',
            'flatten': 'fill:#ffa,stroke:#ff0,stroke-width:3px',
            'permute': 'fill:#aff,stroke:#0ff,stroke-width:3px',
            'squeeze': 'fill:#faf,stroke:#f0f,stroke-width:3px',
            'unsqueeze': 'fill:#afa,stroke:#0f0,stroke-width:3px',
            'view': 'fill:#aaf,stroke:#00f,stroke-width:3px'
        }
        
        total_transforms = 0
        for transform_type, nodes in transformations.items():
            if nodes:
                lines.append(f'    %% {transform_type}: {len(nodes)} operations')
                style = style_map.get(transform_type, 'fill:#ccc,stroke:#666')
                for node in nodes:
                    safe_id = node.module_path.replace('.', '_')
                    lines.append(f'    style {safe_id} {style}')
                total_transforms += len(nodes)
        
        if total_transforms > 0:
            lines.append(f'    %% Total shape transformations: {total_transforms}')
            
    def generate_shape_transformation_report(self, trace_nodes: List[TraceNode]) -> str:
        """Generate detailed shape transformation report"""
        lines = ["### Shape Transformation Analysis", ""]
        
        # Collect all shape transformations
        transformations = []
        for node in trace_nodes:
            if node.shape_transform and node.input_shapes and node.output_shapes:
                in_shape = node.input_shapes[0].shape if node.input_shapes else None
                out_shape = node.output_shapes[0].shape if node.output_shapes else None
                
                if in_shape and out_shape:
                    # Calculate shape change metrics
                    in_elements = 1
                    for dim in in_shape:
                        in_elements *= dim
                    out_elements = 1
                    for dim in out_shape:
                        out_elements *= dim
                    
                    # Check if significant shape change
                    dim_change = len(out_shape) - len(in_shape)
                    shape_preserved = (in_elements == out_elements)
                    
                    transformations.append({
                        'node': node,
                        'type': node.shape_transform,
                        'in_shape': in_shape,
                        'out_shape': out_shape,
                        'dim_change': dim_change,
                        'shape_preserved': shape_preserved,
                        'memory_impact': node.memory_usage
                    })
        
        if not transformations:
            lines.append("No shape transformations detected.")
            return '\n'.join(lines)
        
        # Summary
        lines.append(f"**Total Transformations**: {len(transformations)}")
        lines.append("")
        
        # Group by transformation type
        by_type = {}
        for t in transformations:
            t_type = t['type']
            if t_type not in by_type:
                by_type[t_type] = []
            by_type[t_type].append(t)
        
        lines.append("**By Type**:")
        for t_type, items in sorted(by_type.items()):
            lines.append(f"- {t_type}: {len(items)} operations")
        lines.append("")
        
        # Detailed transformation list
        lines.append("**Transformation Details**:")
        lines.append("```")
        lines.append(f"{'Operation':<40} {'Type':<10} {'Input Shape':<20} {'Output Shape':<20} {'Memory (MB)':<10}")
        lines.append("-" * 105)
        
        for t in sorted(transformations, key=lambda x: x['memory_impact'], reverse=True)[:20]:
            node = t['node']
            op_name = f"{node.module_path}.{node.operation}"[:39]
            in_shape_str = str(t['in_shape'])[:19]
            out_shape_str = str(t['out_shape'])[:19]
            
            lines.append(f"{op_name:<40} {t['type']:<10} {in_shape_str:<20} {out_shape_str:<20} {t['memory_impact']:<10.1f}")
        
        lines.append("```")
        
        # High-impact transformations
        high_impact = [t for t in transformations if t['memory_impact'] > 100]
        if high_impact:
            lines.append("")
            lines.append("**High Memory Impact Transformations** (>100MB):")
            for t in sorted(high_impact, key=lambda x: x['memory_impact'], reverse=True):
                node = t['node']
                lines.append(f"- {node.module_path}: {t['in_shape']} → {t['out_shape']} ({t['memory_impact']:.1f}MB)")
        
        return '\n'.join(lines)
    
    def generate_memory_based_view(self, trace_nodes: List[TraceNode], 
                                  memory_threshold: float = 5000.0) -> str:
        """Generate view with auto-expanded memory-heavy modules
        
        Args:
            trace_nodes: Traced nodes
            memory_threshold: Memory threshold in MB for auto-expansion
            
        Returns:
            Mermaid diagram string
        """
        # Temporarily set memory threshold
        old_threshold = self.state.memory_threshold
        self.state.memory_threshold = memory_threshold
        
        # Generate diagram
        lines = ["```mermaid", "graph TB"]
        lines.append(f"    %% Memory-based auto-expansion (threshold: {memory_threshold}MB)")
        
        # Build hierarchy and expand heavy modules
        hierarchy = self._build_module_hierarchy(trace_nodes[:self.max_nodes])
        
        # Find and expand memory-heavy modules
        heavy_modules = []
        for module_name, module_nodes in hierarchy.items():
            total_memory = sum(n.memory_usage for n in module_nodes)
            if total_memory > memory_threshold:
                self.state.expanded_modules.add(module_name)
                heavy_modules.append((module_name, total_memory))
        
        # Generate expanded view
        self._generate_expanded_view(lines, hierarchy, trace_nodes[:self.max_nodes])
        
        # Add summary
        lines.append("")
        lines.append(f"    %% Auto-expanded {len(heavy_modules)} modules exceeding {memory_threshold}MB")
        for module, memory in sorted(heavy_modules, key=lambda x: x[1], reverse=True):
            lines.append(f"    %% {module}: {memory:.1f}MB")
        
        lines.append("```")
        
        # Restore original threshold
        self.state.memory_threshold = old_threshold
        
        return '\n'.join(lines)
    
    def generate_memory_heatmap(self, memory_profile: Dict[str, Any]) -> str:
        """Generate memory usage heatmap"""
        lines = ["### Memory Usage Heatmap", ""]
        lines.append("```")
        lines.append(f"{'Operation':<30} | {'Memory (MB)':<12} | {'Percentage':<10} | {'Visual':<40}")
        lines.append("-" * 95)
        
        total_memory = memory_profile['total_memory_mb']
        for consumer in memory_profile['top_consumers']:
            op = consumer['operation'][:30]
            mem = consumer['memory_mb']
            pct = consumer['percentage']
            bar_length = int(pct / 100 * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            
            lines.append(f"{op:<30} | {mem:<12.1f} | {pct:<10.1f} | {bar}")
        
        lines.append("-" * 95)
        lines.append(f"{'Total':<30} | {total_memory:<12.1f} | {'100.0':<10} | {'█' * 40}")
        lines.append("```")
        
        return '\n'.join(lines)