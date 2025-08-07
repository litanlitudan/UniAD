#!/usr/bin/env python3
"""Integration example showing how to use HTMLTemplateEngine with existing visualizers

This example demonstrates how to convert existing Mermaid-based visualizations
to interactive HTML format using the HTMLTemplateEngine.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import web components
sys.path.insert(0, str(Path(__file__).parent))
from template_engine import (
    HTMLTemplateEngine, 
    TemplateConfig,
    create_dataflow_visualization
)

def convert_trace_nodes_to_html_nodes(trace_nodes) -> List[Dict[str, Any]]:
    """
    Convert TraceNode objects to HTML visualization node format.
    
    This function bridges the existing tracer output with the HTML template engine.
    """
    html_nodes = []
    
    for i, node in enumerate(trace_nodes):
        # Safely get attributes with fallbacks for compatibility
        html_node = {
            'id': f'node_{i}',
            'display_name': getattr(node, 'operation', f'Operation_{i}'),
            'type': 'operation',
            'task_head': getattr(node, 'task_head', None),
            'operation': getattr(node, 'operation', 'unknown'),
            'memory_mb': getattr(node, 'memory_usage', 0.0),
            'compute_ms': getattr(node, 'compute_time', 0.0),
            'module_path': getattr(node, 'module_path', ''),
            'position': [i * 100, 100],  # Simple positioning
            'color': _get_task_color(getattr(node, 'task_head', None) or 'unknown'),
            'size': _calculate_node_size(getattr(node, 'memory_usage', 0.0)),
            'tooltip_data': {
                'operation': getattr(node, 'operation', 'unknown'),
                'module_path': getattr(node, 'module_path', ''),
                'memory_mb': getattr(node, 'memory_usage', 0.0),
                'compute_ms': getattr(node, 'compute_time', 0.0),
                'task_head': getattr(node, 'task_head', None),
                'is_frozen': getattr(node, 'is_frozen', False),
                'input_shapes': _format_shapes(getattr(node, 'input_shapes', [])),
                'output_shapes': _format_shapes(getattr(node, 'output_shapes', []))
            }
        }
        html_nodes.append(html_node)
    
    return html_nodes

def _get_task_color(task_head: Optional[str] = None) -> str:
    """Get color for task head"""
    colors = {
        'track': '#ff6b6b',      # Red
        'seg': '#4ecdc4',        # Teal  
        'motion': '#45b7d1',     # Blue
        'occ': '#96ceb4',        # Green
        'planning': '#feca57',   # Yellow
        'bev': '#ff9ff3',        # Pink
    }
    return colors.get(task_head or 'unknown', '#95a5a6')  # Gray default

def _calculate_node_size(memory_mb: float) -> float:
    """Calculate node size based on memory usage"""
    return max(15, min(50, 15 + (memory_mb / 100) * 10))

def _format_shapes(shapes) -> List[str]:
    """Format tensor shape information"""
    if not shapes:
        return []
    
    formatted = []
    for shape in shapes[:3]:  # Limit to first 3 shapes
        if hasattr(shape, 'shape') and hasattr(shape, 'dtype'):
            shape_str = f"[{','.join(map(str, shape.shape))}]@{shape.dtype}"
        else:
            shape_str = str(shape)
        formatted.append(shape_str)
    
    return formatted

def create_edges_from_dependencies(nodes: List[Dict], dependency_rules: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    """
    Create edges between nodes based on dependency rules or simple sequence.
    
    Args:
        nodes: List of node dictionaries
        dependency_rules: Optional rules for task head dependencies
    """
    edges = []
    
    # Default UniAD dependency rules
    if dependency_rules is None:
        dependency_rules = {
            'bev': ['track', 'seg'],          # BEV feeds into track and segmentation
            'track': ['motion', 'occ'],       # Track feeds into motion and occupancy
            'motion': ['planning'],           # Motion feeds into planning
            'occ': ['planning'],              # Occupancy feeds into planning
        }
    
    # Create task head to node mapping
    task_nodes = {}
    for node in nodes:
        task_head = node.get('task_head')
        if task_head:
            if task_head not in task_nodes:
                task_nodes[task_head] = []
            task_nodes[task_head].append(node)
    
    # Create edges based on dependency rules
    edge_id = 0
    for source_task, target_tasks in dependency_rules.items():
        if source_task in task_nodes:
            source_nodes = task_nodes[source_task]
            
            for target_task in target_tasks:
                if target_task in task_nodes:
                    target_nodes = task_nodes[target_task]
                    
                    # Connect first node of source task to first node of target task
                    if source_nodes and target_nodes:
                        edge = {
                            'id': f'edge_{edge_id}',
                            'source': source_nodes[0]['id'],
                            'target': target_nodes[0]['id'],
                            'type': 'task_dependency',
                            'weight': 2.0
                        }
                        edges.append(edge)
                        edge_id += 1
    
    # Add sequential edges within same task head
    for task_head, task_nodes_list in task_nodes.items():
        if len(task_nodes_list) > 1:
            for i in range(len(task_nodes_list) - 1):
                edge = {
                    'id': f'edge_{edge_id}',
                    'source': task_nodes_list[i]['id'],
                    'target': task_nodes_list[i + 1]['id'],
                    'type': 'sequential',
                    'weight': 1.0
                }
                edges.append(edge)
                edge_id += 1
    
    return edges

def create_interactive_visualization_from_trace(trace_nodes, 
                                              title: str = "UniAD Interactive Visualization",
                                              output_path: Optional[str] = None) -> str:
    """
    Create interactive HTML visualization from trace nodes.
    
    This is the main integration function that bridges existing trace data
    with the new HTML template engine.
    """
    print(f"Converting {len(trace_nodes)} trace nodes to interactive HTML...")
    
    # Convert trace nodes to HTML format
    html_nodes = convert_trace_nodes_to_html_nodes(trace_nodes)
    print(f"✓ Converted to {len(html_nodes)} HTML nodes")
    
    # Create edges based on dependencies
    edges = create_edges_from_dependencies(html_nodes)
    print(f"✓ Created {len(edges)} edges")
    
    # Configure template engine for UniAD
    config = TemplateConfig(
        theme="uniad",
        enable_caching=True,
        compression_level="moderate",
        include_d3js=True,
        include_chartjs=True
    )
    
    # Create HTML visualization
    html_content = create_dataflow_visualization(
        html_nodes, 
        edges, 
        title,
        config
    )
    
    print(f"✓ Generated HTML visualization ({len(html_content)} characters)")
    
    # Save to file if path provided
    if output_path:
        engine = HTMLTemplateEngine(config)
        saved_path = engine.export_html(html_content, output_path)
        print(f"✓ Saved to: {saved_path}")
    
    return html_content

def demo_with_sample_data():
    """Demo the HTML visualization with sample UniAD-like data"""
    
    # Create sample trace nodes that mimic real UniAD operations
    class MockTraceNode:
        def __init__(self, operation, task_head, memory_usage, compute_time, module_path):
            self.operation = operation
            self.task_head = task_head
            self.memory_usage = memory_usage
            self.compute_time = compute_time
            self.module_path = module_path
            self.is_frozen = False
            self.input_shapes = []
            self.output_shapes = []
    
    sample_nodes = [
        MockTraceNode("ImageBackbone", "bev", 2048.5, 25.4, "backbone.resnet101"),
        MockTraceNode("BEVEncoder", "bev", 1536.2, 18.7, "bev_encoder.transformer"),
        MockTraceNode("TrackHead_forward", "track", 1024.8, 12.3, "task_heads.track_head"),
        MockTraceNode("TrackDecoder", "track", 768.4, 9.1, "task_heads.track_head.decoder"),
        MockTraceNode("SegmentationHead", "seg", 896.7, 14.2, "task_heads.panseg_head"),
        MockTraceNode("MotionPredictor", "motion", 512.3, 8.9, "task_heads.motion_head"),
        MockTraceNode("MotionDecoder", "motion", 384.1, 6.7, "task_heads.motion_head.decoder"),
        MockTraceNode("OccupancyHead", "occ", 643.2, 11.4, "task_heads.occ_head"),
        MockTraceNode("PlanningHead", "planning", 256.8, 5.2, "task_heads.planning_head"),
        MockTraceNode("TrajectoryDecoder", "planning", 192.4, 3.8, "task_heads.planning_head.decoder")
    ]
    
    # Create interactive visualization
    output_path = project_root / "demo_interactive_visualization.html"
    
    html_content = create_interactive_visualization_from_trace(
        sample_nodes,
        title="UniAD Interactive Dataflow Demo",
        output_path=str(output_path)
    )
    
    print("\n✅ Demo complete!")
    print(f"📁 Open {output_path} in your web browser to view the interactive visualization")
    
    return html_content

if __name__ == "__main__":
    print("HTMLTemplateEngine Integration Example")
    print("=" * 50)
    
    # Run demo
    demo_with_sample_data()
    
    print("\nIntegration functions available:")
    print("- convert_trace_nodes_to_html_nodes()")
    print("- create_edges_from_dependencies()")  
    print("- create_interactive_visualization_from_trace()")
    print("\nThese functions can be used to convert existing trace data to interactive HTML.")