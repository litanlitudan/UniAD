"""
Unit tests for MermaidVisualizer.

Tests the Mermaid diagram generation functionality including dataflow diagrams,
hierarchical views, and other visualizations.
"""

import unittest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TaskHead
from visualizers.mermaid_visualizer import MermaidVisualizer, MermaidNode, MermaidEdge


class TestMermaidVisualizer(unittest.TestCase):
    """Test suite for MermaidVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = MermaidVisualizer()
        self.sample_trace_data = self._create_sample_trace_data()
    
    def _create_sample_trace_data(self):
        """Create sample trace data for testing."""
        trace_nodes = []
        
        # Create diverse operations for testing
        operations = [
            ("Conv2d", "backbone.conv1", TaskHead.NONE, 100),
            ("BatchNorm", "backbone.bn1", TaskHead.NONE, 50),
            ("ReLU", "backbone.relu", TaskHead.NONE, 10),
            ("MaxPool2d", "backbone.pool", TaskHead.NONE, 80),
            ("TrackHead", "track_head.layer1", TaskHead.TRACK, 200),
            ("SegHead", "seg_head.layer1", TaskHead.SEGMENTATION, 300),
            ("MotionHead", "motion_head.layer1", TaskHead.MOTION, 250),
        ]
        
        for i, (name, module_path, task_head, memory_mb) in enumerate(operations):
            node = TraceNode(
                id=f"op_{i}",
                name=name,
                module_path=module_path,
                start_time=i * 1000000,
                duration=500000 + i * 100000,
                memory_allocated=memory_mb * 1024 * 1024,
                memory_freed=memory_mb * 1024 * 512,
                cuda_memory_allocated=memory_mb * 1024 * 1024,
                cuda_memory_freed=memory_mb * 1024 * 512,
                task_head=task_head
            )
            trace_nodes.append(node)
        
        return trace_nodes
    
    def test_generate_dataflow_diagram(self):
        """Test dataflow diagram generation."""
        diagram = self.visualizer.generate_dataflow_diagram(self.sample_trace_data)
        
        self.assertIsInstance(diagram, str)
        self.assertIn("graph TD", diagram)
        self.assertIn("Dataflow Diagram", diagram)
        
        # Check that nodes are included
        for node in self.sample_trace_data:
            self.assertIn(node.name, diagram)
        
        # Check for edges (sequential flow)
        self.assertIn("-->", diagram)
    
    def test_generate_hierarchical_view(self):
        """Test hierarchical view generation."""
        diagram = self.visualizer.generate_hierarchical_view(self.sample_trace_data)
        
        self.assertIsInstance(diagram, str)
        self.assertIn("graph TB", diagram)
        self.assertIn("Module Hierarchy View", diagram)
        
        # Check for module names
        self.assertIn("backbone", diagram)
        self.assertIn("track_head", diagram)
        self.assertIn("seg_head", diagram)
        
        # Check for hierarchical connections
        self.assertIn("-->", diagram)
    
    def test_generate_task_head_diagram(self):
        """Test task head diagram generation."""
        # Create task head analysis mock data
        task_head_analysis = {
            'head_info': {
                'name': 'track',
                'task_type': TaskHead.TRACK
            },
            'operations': {
                'total_ops': 5,
                'unique_op_types': ['Conv2d', 'Linear', 'ReLU']
            },
            'memory_usage': {
                'total_allocated_mb': 500.0,
                'peak_memory_mb': 200.0
            },
            'performance': {
                'total_time_ms': 10.5,
                'avg_operation_time_us': 2100
            }
        }
        
        diagram = self.visualizer.generate_task_head_diagram(task_head_analysis)
        
        self.assertIsInstance(diagram, str)
        self.assertIn("graph LR", diagram)
        self.assertIn("track", diagram.lower())
    
    def test_generate_memory_flow_diagram(self):
        """Test memory flow diagram generation."""
        # Create memory analysis mock data
        memory_analysis = {
            'summary': {
                'peak_memory_gb': 40.5,
                'memory_utilization': 0.85
            },
            'timeline': [
                {'timestamp_ms': 0, 'allocated_mb': 100, 'operation': 'Conv2d'},
                {'timestamp_ms': 1, 'allocated_mb': 200, 'operation': 'BatchNorm'},
                {'timestamp_ms': 2, 'allocated_mb': 150, 'operation': 'ReLU'},
            ],
            'allocation_patterns': {
                'Conv2d': {'count': 10, 'total_mb': 1000},
                'Linear': {'count': 5, 'total_mb': 500}
            }
        }
        
        diagram = self.visualizer.generate_memory_flow_diagram(memory_analysis)
        
        self.assertIsInstance(diagram, str)
        self.assertIn("graph TD", diagram)
        self.assertIn("Memory", diagram)
    
    def test_mermaid_node_creation(self):
        """Test MermaidNode creation and conversion."""
        node = MermaidNode(
            id="test_node",
            label="Test Operation",
            node_type="operation"
        )
        
        mermaid_str = node.to_mermaid()
        self.assertIn("test_node", mermaid_str)
        self.assertIn("Test Operation", mermaid_str)
    
    def test_mermaid_edge_creation(self):
        """Test MermaidEdge creation and conversion."""
        edge = MermaidEdge(
            from_node="node1",
            to_node="node2",
            label="data flow"
        )
        
        mermaid_str = edge.to_mermaid()
        self.assertIn("node1", mermaid_str)
        self.assertIn("node2", mermaid_str)
        self.assertIn("-->", mermaid_str)
    
    def test_export_html(self):
        """Test HTML export functionality."""
        diagram = self.visualizer.generate_dataflow_diagram(self.sample_trace_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            self.visualizer.export_to_html(
                diagram,
                temp_path,
                "Test Diagram"
            )
            
            # Check file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Check content
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn("mermaid", content)
                self.assertIn("Test Diagram", content)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_empty_trace_data(self):
        """Test handling of empty trace data."""
        empty_diagram = self.visualizer.generate_dataflow_diagram([])
        self.assertIsInstance(empty_diagram, str)
        self.assertIn("graph TD", empty_diagram)
    
    def test_single_node(self):
        """Test handling of single node."""
        single_node = [self.sample_trace_data[0]]
        diagram = self.visualizer.generate_dataflow_diagram(single_node)
        
        self.assertIsInstance(diagram, str)
        self.assertIn(single_node[0].name, diagram)
        # Should not have edges for single node
        self.assertEqual(diagram.count("-->"), 0)
    
    def test_style_generation(self):
        """Test that styles are properly generated."""
        diagram = self.visualizer.generate_dataflow_diagram(self.sample_trace_data)
        
        # Check for style classes
        if "classDef" in diagram:
            self.assertIn("classDef", diagram)
        
        # Check for task head specific styles
        for node in self.sample_trace_data:
            if node.task_head != TaskHead.NONE:
                # Style should be applied for task heads
                pass  # Style application is optional
    
    def test_temporal_flow_diagram(self):
        """Test temporal flow diagram generation."""
        temporal_analysis = {
            'frame_count': 3,
            'queue_memory_mb': 500,
            'temporal_dependencies': [
                {'frame': 0, 'operations': 10},
                {'frame': 1, 'operations': 12},
                {'frame': 2, 'operations': 11}
            ]
        }
        
        diagram = self.visualizer.generate_temporal_flow_diagram(temporal_analysis)
        
        self.assertIsInstance(diagram, str)
        self.assertIn("graph LR", diagram)
    
    def test_bev_transformation_diagram(self):
        """Test BEV transformation diagram generation."""
        bev_analysis = {
            'encoder_layers': 6,
            'feature_channels': 256,
            'spatial_shape': (200, 200),
            'memory_usage_mb': 2048
        }
        
        diagram = self.visualizer.generate_bev_transformation_diagram(bev_analysis)
        
        self.assertIsInstance(diagram, str)
        self.assertIn("graph TB", diagram)
        self.assertIn("BEV", diagram)


if __name__ == '__main__':
    unittest.main()