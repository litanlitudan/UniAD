"""
Mermaid Visualizer for UniAD Model Analyzer.

This module generates Mermaid diagrams for visualizing UniAD model architecture,
data flow, and analysis results.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

from core.data_structures import TraceNode, TaskHead, AnalysisResult


@dataclass
class MermaidNode:
    """Represents a node in the Mermaid diagram."""
    
    id: str
    label: str
    node_type: str  # 'operation', 'data', 'decision', 'process'
    style: Optional[str] = None
    
    def to_mermaid(self) -> str:
        """Convert to Mermaid syntax."""
        if self.node_type == 'operation':
            return f"{self.id}[{self.label}]"
        elif self.node_type == 'data':
            return f"{self.id}[({self.label})]"
        elif self.node_type == 'decision':
            return f"{self.id}{{{self.label}}}"
        elif self.node_type == 'process':
            return f"{self.id}[/{self.label}/]"
        else:
            return f"{self.id}[{self.label}]"


@dataclass
class MermaidEdge:
    """Represents an edge in the Mermaid diagram."""
    
    from_node: str
    to_node: str
    label: Optional[str] = None
    edge_type: str = 'normal'  # 'normal', 'dotted', 'thick'
    
    def to_mermaid(self) -> str:
        """Convert to Mermaid syntax."""
        arrow = "-->"
        if self.edge_type == 'dotted':
            arrow = "-.->"
        elif self.edge_type == 'thick':
            arrow = "==>"
        
        if self.label:
            return f"{self.from_node} {arrow}|{self.label}| {self.to_node}"
        else:
            return f"{self.from_node} {arrow} {self.to_node}"


class MermaidVisualizer:
    """
    Generates Mermaid diagrams for UniAD model visualization.
    
    Supports multiple diagram types:
    - Architecture flow diagrams
    - Task head dependency graphs
    - Memory usage charts
    - Temporal processing flows
    - BEV transformation diagrams
    """
    
    def __init__(self, theme: str = 'default'):
        """
        Initialize the Mermaid visualizer.
        
        Args:
            theme: Mermaid theme ('default', 'dark', 'forest', 'neutral')
        """
        self.theme = theme
        self.nodes: List[MermaidNode] = []
        self.edges: List[MermaidEdge] = []
        self.subgraphs: Dict[str, List[str]] = defaultdict(list)
        
        # Style definitions
        self.styles = {
            'backbone': 'fill:#f9f,stroke:#333,stroke-width:2px',
            'bev': 'fill:#bbf,stroke:#333,stroke-width:2px',
            'temporal': 'fill:#bfb,stroke:#333,stroke-width:2px',
            'task_head': 'fill:#fbb,stroke:#333,stroke-width:2px',
            'critical': 'fill:#f66,stroke:#333,stroke-width:4px',
            'optimized': 'fill:#6f6,stroke:#333,stroke-width:2px',
        }
    
    def generate_architecture_diagram(self, 
                                     trace_data: List[TraceNode],
                                     analysis_result: Optional[AnalysisResult] = None) -> str:
        """
        Generate architecture flow diagram from trace data.
        
        Args:
            trace_data: List of trace nodes
            analysis_result: Optional analysis results for enhancement
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = []
        self.edges = []
        self.subgraphs = defaultdict(list)
        
        # Group nodes by module hierarchy
        module_hierarchy = self._build_module_hierarchy(trace_data)
        
        # Create nodes and edges
        self._create_architecture_nodes(module_hierarchy)
        self._create_architecture_edges(trace_data)
        
        # Apply analysis results if available
        if analysis_result:
            self._enhance_with_analysis(analysis_result)
        
        # Generate Mermaid code
        return self._generate_mermaid_code('flowchart TD')
    
    def generate_task_head_diagram(self, task_head_analysis: Dict[str, Any]) -> str:
        """
        Generate task head dependency diagram.
        
        Args:
            task_head_analysis: Task head analysis from MultiHeadAnalyzer
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = []
        self.edges = []
        
        # Create nodes for each task head
        for task_head in TaskHead:
            if task_head != TaskHead.NONE:
                node = MermaidNode(
                    id=f"head_{task_head.value}",
                    label=f"{task_head.value.upper()} Head",
                    node_type='process'
                )
                self.nodes.append(node)
        
        # Add shared backbone
        backbone_node = MermaidNode(
            id="backbone",
            label="Shared Backbone",
            node_type='operation'
        )
        self.nodes.append(backbone_node)
        
        # Add BEV encoder
        bev_node = MermaidNode(
            id="bev_encoder",
            label="BEV Encoder",
            node_type='operation'
        )
        self.nodes.append(bev_node)
        
        # Create edges based on dependencies
        if 'dependency_graph' in task_head_analysis:
            for head, deps in task_head_analysis['dependency_graph'].items():
                for dep in deps:
                    edge = MermaidEdge(
                        from_node=f"head_{dep}",
                        to_node=f"head_{head}",
                        edge_type='normal'
                    )
                    self.edges.append(edge)
        
        # Connect backbone to BEV to heads
        self.edges.append(MermaidEdge("backbone", "bev_encoder"))
        for task_head in TaskHead:
            if task_head != TaskHead.NONE:
                self.edges.append(MermaidEdge("bev_encoder", f"head_{task_head.value}"))
        
        return self._generate_mermaid_code('graph LR')
    
    def generate_memory_flow_diagram(self, memory_analysis: Dict[str, Any]) -> str:
        """
        Generate memory usage flow diagram.
        
        Args:
            memory_analysis: Memory analysis from MemoryProfiler
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = []
        self.edges = []
        
        # Parse memory timeline
        if 'memory_timeline' in memory_analysis:
            timeline = memory_analysis['memory_timeline']
            
            # Create nodes for memory states
            for i, point in enumerate(timeline[:10]):  # Limit to 10 points
                node = MermaidNode(
                    id=f"mem_{i}",
                    label=f"{point['allocated_mb']:.1f}MB",
                    node_type='data'
                )
                self.nodes.append(node)
                
                # Add edge to next state
                if i > 0:
                    delta = point['delta_mb']
                    edge_label = f"+{delta:.1f}MB" if delta > 0 else f"{delta:.1f}MB"
                    edge = MermaidEdge(
                        from_node=f"mem_{i-1}",
                        to_node=f"mem_{i}",
                        label=edge_label,
                        edge_type='thick' if abs(delta) > 10 else 'normal'
                    )
                    self.edges.append(edge)
        
        return self._generate_mermaid_code('graph LR')
    
    def generate_temporal_flow_diagram(self, temporal_analysis: Dict[str, Any]) -> str:
        """
        Generate temporal processing flow diagram.
        
        Args:
            temporal_analysis: Temporal analysis from TemporalAnalyzer
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = []
        self.edges = []
        
        num_frames = temporal_analysis.get('num_frames', 3)
        
        # Create nodes for each temporal frame
        for i in range(num_frames):
            node = MermaidNode(
                id=f"frame_{i}",
                label=f"Frame t-{num_frames-1-i}",
                node_type='data'
            )
            self.nodes.append(node)
        
        # Add temporal aggregation node
        agg_node = MermaidNode(
            id="temporal_agg",
            label="Temporal Aggregation",
            node_type='process'
        )
        self.nodes.append(agg_node)
        
        # Add attention nodes if present
        if 'attention_analysis' in temporal_analysis:
            attn_node = MermaidNode(
                id="temporal_attn",
                label="Temporal Attention",
                node_type='operation'
            )
            self.nodes.append(attn_node)
            
            # Connect frames to attention
            for i in range(num_frames):
                self.edges.append(MermaidEdge(f"frame_{i}", "temporal_attn"))
            
            # Connect attention to aggregation
            self.edges.append(MermaidEdge("temporal_attn", "temporal_agg"))
        else:
            # Direct connection to aggregation
            for i in range(num_frames):
                self.edges.append(MermaidEdge(f"frame_{i}", "temporal_agg"))
        
        # Add output
        output_node = MermaidNode(
            id="temporal_output",
            label="Temporal Features",
            node_type='data'
        )
        self.nodes.append(output_node)
        self.edges.append(MermaidEdge("temporal_agg", "temporal_output"))
        
        return self._generate_mermaid_code('graph TD')
    
    def generate_bev_transformation_diagram(self, bev_analysis: Dict[str, Any]) -> str:
        """
        Generate BEV transformation flow diagram.
        
        Args:
            bev_analysis: BEV analysis from BEVAnalyzer
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = []
        self.edges = []
        self.subgraphs = defaultdict(list)
        
        # Camera inputs
        for i in range(6):
            node = MermaidNode(
                id=f"cam_{i}",
                label=f"Camera {i}",
                node_type='data'
            )
            self.nodes.append(node)
            self.subgraphs['Cameras'].append(f"cam_{i}")
        
        # Projection operations
        proj_node = MermaidNode(
            id="projection",
            label="Camera→BEV Projection",
            node_type='operation'
        )
        self.nodes.append(proj_node)
        
        # BEV grid
        grid_config = bev_analysis.get('bev_grid_config', {})
        grid_res = grid_config.get('grid_resolution', (200, 200))
        bev_node = MermaidNode(
            id="bev_grid",
            label=f"BEV Grid {grid_res[0]}x{grid_res[1]}",
            node_type='process'
        )
        self.nodes.append(bev_node)
        
        # Connect cameras to projection
        for i in range(6):
            self.edges.append(MermaidEdge(f"cam_{i}", "projection"))
        
        # Connect projection to BEV
        self.edges.append(MermaidEdge("projection", "bev_grid"))
        
        # Add transformations if present
        if 'spatial_transformations' in bev_analysis:
            for i, trans in enumerate(bev_analysis['spatial_transformations'][:3]):
                trans_node = MermaidNode(
                    id=f"trans_{i}",
                    label=f"{trans['type']}\n{trans['from_resolution']}→{trans['to_resolution']}",
                    node_type='operation'
                )
                self.nodes.append(trans_node)
                self.edges.append(MermaidEdge("bev_grid", f"trans_{i}"))
        
        return self._generate_mermaid_code('graph TD')
    
    def generate_optimization_diagram(self, 
                                    memory_suggestions: List[Dict[str, Any]],
                                    dtype_opportunities: List[Dict[str, Any]]) -> str:
        """
        Generate optimization opportunity diagram.
        
        Args:
            memory_suggestions: Memory optimization suggestions
            dtype_opportunities: DType optimization opportunities
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = []
        self.edges = []
        
        # Current state
        current_node = MermaidNode(
            id="current",
            label="Current Model",
            node_type='operation'
        )
        self.nodes.append(current_node)
        
        # Memory optimizations
        if memory_suggestions:
            for i, suggestion in enumerate(memory_suggestions[:3]):
                opt_node = MermaidNode(
                    id=f"mem_opt_{i}",
                    label=f"{suggestion['category']}\n-{suggestion['expected_savings_mb']:.0f}MB",
                    node_type='decision'
                )
                self.nodes.append(opt_node)
                self.edges.append(MermaidEdge(
                    "current", 
                    f"mem_opt_{i}",
                    label=suggestion['severity']
                ))
        
        # DType optimizations
        if dtype_opportunities:
            for i, opp in enumerate(dtype_opportunities[:3]):
                dtype_node = MermaidNode(
                    id=f"dtype_opt_{i}",
                    label=f"FP32→{opp['target_dtype']}\n-{opp['memory_savings_mb']:.0f}MB",
                    node_type='decision'
                )
                self.nodes.append(dtype_node)
                self.edges.append(MermaidEdge(
                    "current",
                    f"dtype_opt_{i}",
                    label=f"x{opp['speedup_factor']:.1f}"
                ))
        
        # Optimized state
        optimized_node = MermaidNode(
            id="optimized",
            label="Optimized Model",
            node_type='process'
        )
        self.nodes.append(optimized_node)
        
        # Connect optimizations to optimized state
        for node in self.nodes:
            if node.id.startswith('mem_opt_') or node.id.startswith('dtype_opt_'):
                self.edges.append(MermaidEdge(node.id, "optimized", edge_type='dotted'))
        
        return self._generate_mermaid_code('graph TD')
    
    def _build_module_hierarchy(self, trace_data: List[TraceNode]) -> Dict[str, Set[str]]:
        """Build module hierarchy from trace data."""
        hierarchy = defaultdict(set)
        
        for node in trace_data:
            parts = node.module_path.split('.')
            for i in range(len(parts)):
                if i > 0:
                    parent = '.'.join(parts[:i])
                    child = '.'.join(parts[:i+1])
                    hierarchy[parent].add(child)
        
        return hierarchy
    
    def _create_architecture_nodes(self, hierarchy: Dict[str, Set[str]]) -> None:
        """Create nodes for architecture diagram."""
        created_nodes = set()
        
        for parent, children in hierarchy.items():
            # Create parent node if not exists
            if parent not in created_nodes:
                node_type = self._determine_node_type(parent)
                node = MermaidNode(
                    id=self._sanitize_id(parent),
                    label=parent.split('.')[-1],
                    node_type=node_type
                )
                self.nodes.append(node)
                created_nodes.add(parent)
            
            # Create child nodes
            for child in children:
                if child not in created_nodes:
                    node_type = self._determine_node_type(child)
                    node = MermaidNode(
                        id=self._sanitize_id(child),
                        label=child.split('.')[-1],
                        node_type=node_type
                    )
                    self.nodes.append(node)
                    created_nodes.add(child)
    
    def _create_architecture_edges(self, trace_data: List[TraceNode]) -> None:
        """Create edges for architecture diagram."""
        # Create edges based on execution order
        prev_node = None
        for node in trace_data[:20]:  # Limit to first 20 for clarity
            node_id = self._sanitize_id(node.module_path)
            
            if prev_node and prev_node != node_id:
                # Check if edge already exists
                edge_exists = any(
                    e.from_node == prev_node and e.to_node == node_id
                    for e in self.edges
                )
                
                if not edge_exists:
                    edge = MermaidEdge(
                        from_node=prev_node,
                        to_node=node_id
                    )
                    self.edges.append(edge)
            
            prev_node = node_id
    
    def _enhance_with_analysis(self, analysis_result: AnalysisResult) -> None:
        """Enhance diagram with analysis results."""
        # Add memory info to nodes
        if analysis_result.memory_profile:
            peak_mb = analysis_result.memory_profile.peak_allocated / (1024 * 1024)
            peak_node = MermaidNode(
                id="peak_memory",
                label=f"Peak: {peak_mb:.1f}MB",
                node_type='data'
            )
            self.nodes.append(peak_node)
        
        # Add timing info
        if analysis_result.total_duration_ms > 0:
            time_node = MermaidNode(
                id="total_time",
                label=f"Total: {analysis_result.total_duration_ms:.1f}ms",
                node_type='data'
            )
            self.nodes.append(time_node)
    
    def _determine_node_type(self, module_path: str) -> str:
        """Determine node type based on module path."""
        path_lower = module_path.lower()
        
        if 'head' in path_lower:
            return 'process'
        elif 'bev' in path_lower:
            return 'operation'
        elif 'temporal' in path_lower:
            return 'operation'
        elif 'backbone' in path_lower:
            return 'operation'
        else:
            return 'operation'
    
    def _sanitize_id(self, text: str) -> str:
        """Sanitize text for use as Mermaid node ID."""
        return text.replace('.', '_').replace('-', '_').replace(' ', '_')
    
    def _generate_mermaid_code(self, diagram_type: str = 'graph TD') -> str:
        """Generate the complete Mermaid diagram code."""
        lines = [f"```mermaid", diagram_type]
        
        # Add subgraphs
        for subgraph_name, node_ids in self.subgraphs.items():
            lines.append(f"    subgraph {subgraph_name}")
            for node_id in node_ids:
                node = next((n for n in self.nodes if n.id == node_id), None)
                if node:
                    lines.append(f"        {node.to_mermaid()}")
            lines.append("    end")
        
        # Add standalone nodes
        subgraph_nodes = set()
        for node_ids in self.subgraphs.values():
            subgraph_nodes.update(node_ids)
        
        for node in self.nodes:
            if node.id not in subgraph_nodes:
                lines.append(f"    {node.to_mermaid()}")
        
        # Add edges
        for edge in self.edges:
            lines.append(f"    {edge.to_mermaid()}")
        
        # Add styles
        for style_name, style_def in self.styles.items():
            matching_nodes = [n for n in self.nodes if style_name in n.id.lower()]
            if matching_nodes:
                node_ids = ','.join([n.id for n in matching_nodes])
                lines.append(f"    style {node_ids} {style_def}")
        
        lines.append("```")
        
        return '\n'.join(lines)
    
    def export_to_file(self, diagram: str, filepath: str) -> None:
        """
        Export Mermaid diagram to file.
        
        Args:
            diagram: Mermaid diagram string
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            f.write(diagram)
    
    def export_to_html(self, diagram: str, filepath: str, title: str = "UniAD Model Analysis") -> None:
        """
        Export Mermaid diagram as standalone HTML.
        
        Args:
            diagram: Mermaid diagram string
            filepath: Output HTML file path
            title: HTML page title
        """
        # Remove markdown code block markers if present
        diagram_code = diagram.replace('```mermaid\n', '').replace('\n```', '')
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: '{self.theme}',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true
            }}
        }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .mermaid {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="mermaid">
{diagram_code}
    </div>
</body>
</html>"""
        
        with open(filepath, 'w') as f:
            f.write(html_template)