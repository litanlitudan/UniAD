#!/usr/bin/env python3
"""Test the hierarchical visualization features"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import OperationTracer
from core.data_structures import TraceNode, TensorInfo
from core.hierarchy_analyzer import ModuleHierarchyAnalyzer
from core.visualization_state import VisualizationState
from visualizers import DataflowVisualizer
from analyzers import TraceAnalyzer


def create_dummy_model():
    """Create a dummy model that mimics UniAD structure"""
    class DummyBEVFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
            self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
            self.bn = nn.BatchNorm2d(256)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn(x)
            x = self.conv2(x)
            return x
    
    class DummyTrackHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(256, 128, 1)
            self.fc = nn.Linear(128 * 200 * 200, 300 * 10)
            
        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x.view(-1, 300, 10)
    
    class DummyUniAD(nn.Module):
        def __init__(self):
            super().__init__()
            self.bevformer = DummyBEVFormer()
            self.track_head = DummyTrackHead()
            
        def forward(self, x):
            bev_feat = self.bevformer(x)
            track_out = self.track_head(bev_feat)
            return track_out
    
    return DummyUniAD()


def test_visualization_modes():
    """Test different visualization modes"""
    # Create model and dummy input
    model = create_dummy_model()
    dummy_input = torch.randn(1, 256, 200, 200)
    
    # Create tracer
    tracer = OperationTracer(model, stage=2)
    
    # Perform tracing
    print("Tracing model...")
    outputs = tracer.trace(dummy_input)
    trace_nodes = tracer.get_trace_data()
    print(f"Traced {len(trace_nodes)} operations")
    
    # Analyze trace
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes, stage=2)
    
    # Test different visualization modes
    print("\n=== Testing Top-Level View ===")
    visualizer = DataflowVisualizer(
        visualization_mode='top-level',
        show_shapes=True,
        shape_format='full'
    )
    diagram = visualizer.generate_mermaid(trace_nodes, analysis['head_analysis'], analysis['memory_profile'])
    print(diagram)
    
    print("\n=== Testing Expanded View (BEVFormer) ===")
    visualizer = DataflowVisualizer(
        visualization_mode='expanded',
        expand_modules=['bevformer'],
        show_shapes=True,
        shape_format='compact'
    )
    diagram = visualizer.generate_mermaid(trace_nodes, analysis['head_analysis'], analysis['memory_profile'])
    print(diagram)
    
    print("\n=== Testing Full View ===")
    visualizer = DataflowVisualizer(
        visualization_mode='full',
        show_shapes=True,
        shape_format='full',
        track_shape_changes=True
    )
    diagram = visualizer.generate_mermaid(trace_nodes, analysis['head_analysis'], analysis['memory_profile'])
    print(diagram)
    
    # Test shape format variations
    print("\n=== Testing Semantic Shape Format ===")
    visualizer = DataflowVisualizer(
        visualization_mode='top-level',
        show_shapes=True,
        shape_format='semantic'
    )
    diagram = visualizer.generate_mermaid(trace_nodes, analysis['head_analysis'], analysis['memory_profile'])
    print(diagram)
    
    # Test memory heatmap
    print("\n=== Memory Heatmap ===")
    heatmap = visualizer.generate_memory_heatmap(analysis['memory_profile'])
    print(heatmap)


def test_new_features():
    """Test new visualization features"""
    # Create model and dummy input
    model = create_dummy_model()
    dummy_input = torch.randn(1, 256, 200, 200)
    
    # Create tracer
    tracer = OperationTracer(model, stage=2)
    
    # Perform tracing
    print("Tracing model...")
    outputs = tracer.trace(dummy_input)
    trace_nodes = tracer.get_trace_data()
    
    # Analyze trace
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes, stage=2)
    
    print("\n=== Testing Module Hierarchy Analyzer ===")
    hierarchy_analyzer = ModuleHierarchyAnalyzer(model)
    
    # Test top-level modules
    top_modules = hierarchy_analyzer.get_top_level_modules()
    print(f"Top-level modules: {top_modules}")
    
    # Test module info
    for module in top_modules:
        info = hierarchy_analyzer.get_module_info(module)
        print(f"{module}: {info}")
    
    # Test pattern matching
    print("\nModules matching '*head': ", hierarchy_analyzer.get_modules_by_pattern('*head'))
    print("Modules matching 'bev*': ", hierarchy_analyzer.get_modules_by_pattern('bev*'))
    
    # Test memory threshold
    print("\nModules using >0.1MB: ", hierarchy_analyzer.get_modules_by_memory_threshold(0.1))
    
    print("\n=== Testing Visualization State ===")
    state = VisualizationState()
    
    # Test state management
    state.visualization_mode = 'expanded'
    state.expanded_modules.add('bevformer')
    state.memory_threshold = 1.0
    
    # Test should_expand_module
    module_info = {'memory': 2.0, 'operations': 5, 'has_children': True}
    print(f"Should expand module with 2MB: {state.should_expand_module('test_module', module_info)}")
    
    # Test serialization
    state_dict = state.to_dict()
    print(f"State dict keys: {list(state_dict.keys())}")
    
    # Test deserialization
    new_state = VisualizationState.from_dict(state_dict)
    print(f"Restored state mode: {new_state.visualization_mode}")
    
    print("\n=== Testing Memory-Based Auto-Expansion ===")
    visualizer = DataflowVisualizer(
        visualization_mode='top-level',
        show_shapes=True,
        shape_format='full',
        model=model
    )
    memory_view = visualizer.generate_memory_based_view(trace_nodes, memory_threshold=0.1)
    print(memory_view)
    
    print("\n=== Testing Shape Transformation Report ===")
    # Add some dummy shape transformations to trace nodes
    if len(trace_nodes) > 2:
        trace_nodes[1].shape_transform = 'flatten'
        trace_nodes[2].shape_transform = 'reshape'
    
    shape_report = visualizer.generate_shape_transformation_report(trace_nodes)
    print(shape_report)
    
    print("\n=== Testing Enhanced Temporal Analysis ===")
    from analyzers.temporal_analyzer import TemporalTracer
    temporal_tracer = TemporalTracer(queue_length=3, stage=2)
    
    # Mark some nodes as temporal
    for i, node in enumerate(trace_nodes[:3]):
        node.temporal_index = i % 3
    
    temporal_analysis = temporal_tracer.analyze_temporal_flow(trace_nodes)
    print(f"Temporal analysis keys: {list(temporal_analysis.keys())}")
    print(f"Queue length: {temporal_analysis['queue_length']}")
    print(f"Temporal overhead: {temporal_analysis['temporal_overhead_percent']:.1f}%")
    
    # Test temporal visualization
    temporal_diagram = temporal_tracer.visualize_temporal_flow(trace_nodes)
    print("\n=== Temporal Flow Diagram ===")
    print(temporal_diagram)


if __name__ == '__main__':
    print("Testing original visualization modes...")
    test_visualization_modes()
    
    print("\n\n" + "="*60 + "\n")
    print("Testing new features...")
    test_new_features()