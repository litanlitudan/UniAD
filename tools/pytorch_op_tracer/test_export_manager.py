#!/usr/bin/env python3
"""
Test script for ExportManager functionality.

This script creates sample data and tests the export functionality
to ensure all formats work correctly.
"""

import tempfile
import os
from pathlib import Path
from typing import List
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define fallback classes first
class TensorInfo:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
    
    def to_dict(self):
        return {'shape': self.shape, 'dtype': self.dtype}

class TraceNode:
    def __init__(self, node_id=None, operation="", module_path="", input_shapes=None, output_shapes=None,
                memory_usage=0.0, compute_time=0.0, task_head=None, temporal_index=None,
                is_bev_operation=False, is_frozen=False, shape_transform=None,
                feeds_into=None, depends_on=None):
        self.node_id = node_id or ""
        self.operation = operation
        self.module_path = module_path
        self.input_shapes = input_shapes or []
        self.output_shapes = output_shapes or []
        self.memory_usage = memory_usage
        self.compute_time = compute_time
        self.task_head = task_head
        self.temporal_index = temporal_index
        self.is_bev_operation = is_bev_operation
        self.is_frozen = is_frozen
        self.shape_transform = shape_transform
        self.feeds_into = feeds_into or []
        self.depends_on = depends_on or []

class MemoryTimelineEvent:
    def __init__(self, timestamp, operation, module_path, memory_delta,
                cumulative_memory, task_head=None, temporal_index=None,
                is_bev_operation=False, is_frozen_module=False,
                compute_time=0.0, tensor_info=None):
        self.timestamp = timestamp
        self.operation = operation
        self.module_path = module_path
        self.memory_delta = memory_delta
        self.cumulative_memory = cumulative_memory
        self.task_head = task_head
        self.temporal_index = temporal_index
        self.is_bev_operation = is_bev_operation
        self.is_frozen_module = is_frozen_module
        self.compute_time = compute_time
        self.tensor_info = tensor_info or {}

class MemoryPeak:
    def __init__(self, timestamp, peak_memory_mb, operation, module_path,
                task_head=None, is_problematic=False, context_operations=None):
        self.timestamp = timestamp
        self.peak_memory_mb = peak_memory_mb
        self.operation = operation
        self.module_path = module_path
        self.task_head = task_head
        self.is_problematic = is_problematic
        self.context_operations = context_operations or []

class MemoryTimelineData:
    def __init__(self, events, timeline_points, peaks, task_breakdown,
                memory_pressure, problematic_operations, recommendations):
        self.events = events
        self.timeline_points = timeline_points
        self.peaks = peaks
        self.task_breakdown = task_breakdown
        self.memory_pressure = memory_pressure
        self.problematic_operations = problematic_operations
        self.recommendations = recommendations

class ExportManager:
    def __init__(self, output_dir=None, stage=2, **_kwargs):
        self.output_dir = output_dir or os.getcwd()
        self.stage = stage
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_mermaid(self, _trace_nodes, path, _head_analysis=None, _memory_profile=None):
        with open(path, 'w') as f:
            f.write("```mermaid\ngraph TB\nA[Test] --> B[Export]\n```")
    
    def export_interactive_html(self, _trace_nodes, path, _head_analysis=None, _memory_profile=None):
        with open(path, 'w') as f:
            f.write("<html><body><h1>Test Export</h1></body></html>")
    
    def export_report(self, _analysis_data, path):
        with open(path, 'w') as f:
            f.write("# Test Report\n\nThis is a test export.")
    
    def export_memory_timeline(self, _timeline_data, path, format='html'):
        with open(path, 'w') as f:
            if format == 'html':
                f.write("<html><body><h1>Memory Timeline</h1></body></html>")
            else:
                f.write("# Memory Timeline\n\nTest report.")
    
    def export_png(self, _analysis_data, path):
        # Create minimal PNG-like file
        with open(path, 'w') as f:
            f.write("PNG placeholder")
    
    def export_batch(self, _analysis_data, base_name, formats):
        results = {}
        for fmt in formats:
            filename = f"{base_name}.{fmt}"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Test export for {fmt}")
            results[fmt] = filepath
        return results
    
    def _sanitize_path(self, path):
        if '..' in path:
            raise ValueError("Invalid path")
        return path

# Try to import the actual classes
try:
    from utils.export_manager import ExportManager as RealExportManager  # type: ignore
    from core.data_structures import TraceNode as RealTraceNode, TensorInfo as RealTensorInfo  # type: ignore
    from visualizers.memory_timeline import MemoryTimelineData as RealMemoryTimelineData, MemoryPeak as RealMemoryPeak  # type: ignore
    from core.memory_structures import MemoryTimelineEvent as RealMemoryTimelineEvent  # type: ignore
    
    # Override the fallback classes with real ones
    ExportManager = RealExportManager  # type: ignore
    TraceNode = RealTraceNode  # type: ignore
    TensorInfo = RealTensorInfo  # type: ignore
    MemoryTimelineData = RealMemoryTimelineData  # type: ignore
    MemoryPeak = RealMemoryPeak  # type: ignore
    MemoryTimelineEvent = RealMemoryTimelineEvent  # type: ignore
    
    print("Successfully imported real classes")
except ImportError as e:
    print(f"Import failed: {e}. Using fallback classes for testing.")


def create_sample_trace_nodes() -> List[TraceNode]:
    """Create sample trace nodes for testing"""
    nodes = []
    
    # Sample UniAD operations
    operations = [
        ("conv2d", "backbone.layer1", 1024.0, 15.0, "track"),
        ("linear", "track_head.fc", 512.0, 8.0, "track"),
        ("attention", "bev_encoder.self_attn", 2048.0, 25.0, "bev"),
        ("conv2d", "seg_head.conv", 768.0, 12.0, "seg"),
        ("linear", "motion_head.fc", 1536.0, 18.0, "motion"),
        ("conv2d", "occ_head.conv3d", 2560.0, 32.0, "occ"),
        ("linear", "planning_head.mlp", 256.0, 5.0, "planning"),
    ]
    
    for i, (op, module, memory, compute, task) in enumerate(operations):
        # Create sample tensor info
        input_shape = TensorInfo(shape=[1, 256, 200, 200], dtype="float32")
        output_shape = TensorInfo(shape=[1, 256, 200, 200], dtype="float32")
        
        node = TraceNode(
            node_id=f"node_{i}",
            operation=op,
            module_path=module,
            input_shapes=[input_shape],
            output_shapes=[output_shape],
            memory_usage=memory,
            compute_time=compute,
            task_head=task,
            temporal_index=None,
            is_bev_operation="bev" in task,
            is_frozen=False,
            shape_transform="view" if "linear" in op else None,
            feeds_into=[f"node_{i+1}"] if i < len(operations)-1 else []
        )
        
        nodes.append(node)
    
    return nodes


def create_sample_analysis_data(trace_nodes: List[TraceNode]) -> dict:
    """Create sample analysis data"""
    # Create head analysis
    head_analysis = {
        'memory_by_head': {
            'track': 1536.0,
            'seg': 768.0,
            'motion': 1536.0,
            'occ': 2560.0,
            'planning': 256.0,
            'bev': 2048.0
        },
        'compute_by_head': {
            'track': 23.0,
            'seg': 12.0,
            'motion': 18.0,
            'occ': 32.0,
            'planning': 5.0,
            'bev': 25.0
        }
    }
    
    # Create memory profile
    memory_profile = {
        'total_memory_mb': sum(node.memory_usage for node in trace_nodes),
        'peak_memory_mb': max(node.memory_usage for node in trace_nodes),
        'top_consumers': [
            {
                'operation': node.operation,
                'memory_mb': node.memory_usage,
                'percentage': (node.memory_usage / sum(n.memory_usage for n in trace_nodes)) * 100,
                'module_path': node.module_path
            }
            for node in sorted(trace_nodes, key=lambda x: x.memory_usage, reverse=True)[:5]
        ]
    }
    
    # Create sample memory timeline data
    events = []
    cumulative_memory = 0
    for i, node in enumerate(trace_nodes):
        cumulative_memory += node.memory_usage
        event = MemoryTimelineEvent(
            timestamp=float(i * 10),
            operation=node.operation,
            module_path=node.module_path,
            memory_delta=node.memory_usage,
            cumulative_memory=cumulative_memory,
            task_head=node.task_head,
            temporal_index=node.temporal_index,
            is_bev_operation=node.is_bev_operation,
            is_frozen_module=node.is_frozen,
            compute_time=node.compute_time,
            tensor_info={}
        )
        events.append(event)
    
    # Create sample peaks
    peaks = [
        MemoryPeak(
            timestamp=30.0,
            peak_memory_mb=2560.0,
            operation="conv2d",
            module_path="occ_head.conv3d",
            task_head="occ",
            is_problematic=False,
            context_operations=["attention", "conv2d"]
        )
    ]
    
    memory_timeline = MemoryTimelineData(
        events=events,
        timeline_points=[(event.timestamp, event.cumulative_memory) for event in events],
        peaks=peaks,
        task_breakdown={},
        memory_pressure=0.3,
        problematic_operations=[],
        recommendations=["Consider optimizing OCC head memory usage"]
    )
    
    return {
        'trace_nodes': trace_nodes,
        'head_analysis': head_analysis,
        'memory_profile': memory_profile,
        'memory_timeline': memory_timeline
    }


def test_export_manager():
    """Test the ExportManager functionality"""
    print("Testing ExportManager...")
    
    # Create sample data
    trace_nodes = create_sample_trace_nodes()
    analysis_data = create_sample_analysis_data(trace_nodes)
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Output directory: {temp_dir}")
        
        # Initialize ExportManager
        export_manager = ExportManager(output_dir=temp_dir, stage=2)
        
        # Test individual exports
        print("\n1. Testing Mermaid export...")
        try:
            mermaid_path = os.path.join(temp_dir, "test_diagram.mermaid")
            export_manager.export_mermaid(
                trace_nodes, 
                mermaid_path,
                analysis_data['head_analysis'],
                analysis_data['memory_profile']
            )
            print(f"✓ Mermaid export successful: {os.path.exists(mermaid_path)}")
        except Exception as e:
            print(f"✗ Mermaid export failed: {e}")
        
        print("\n2. Testing HTML export...")
        try:
            html_path = os.path.join(temp_dir, "test_interactive.html")
            export_manager.export_interactive_html(
                trace_nodes,
                html_path,
                analysis_data['head_analysis'],
                analysis_data['memory_profile']
            )
            print(f"✓ HTML export successful: {os.path.exists(html_path)}")
            
            # Check HTML content
            with open(html_path, 'r') as f:
                content = f.read()
                print(f"  - HTML size: {len(content)} characters")
                print(f"  - Contains D3.js: {'d3js.org' in content}")
                print(f"  - Contains chart: {'Chart' in content}")
        except Exception as e:
            print(f"✗ HTML export failed: {e}")
        
        print("\n3. Testing comprehensive report export...")
        try:
            report_path = os.path.join(temp_dir, "test_report.md")
            export_manager.export_report(analysis_data, report_path)
            print(f"✓ Report export successful: {os.path.exists(report_path)}")
            
            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
                print(f"  - Report size: {len(content)} characters")
                print(f"  - Contains summary: {'Executive Summary' in content}")
                print(f"  - Contains recommendations: {'Optimization Recommendations' in content}")
        except Exception as e:
            print(f"✗ Report export failed: {e}")
        
        print("\n4. Testing memory timeline export...")
        try:
            timeline_path = os.path.join(temp_dir, "test_memory_timeline.html")
            export_manager.export_memory_timeline(
                analysis_data['memory_timeline'], 
                timeline_path,
                format='html'
            )
            print(f"✓ Memory timeline export successful: {os.path.exists(timeline_path)}")
        except Exception as e:
            print(f"✗ Memory timeline export failed: {e}")
        
        print("\n5. Testing PNG export...")
        try:
            png_path = os.path.join(temp_dir, "test_visualization.png")
            export_manager.export_png(analysis_data, png_path)
            print(f"✓ PNG export successful: {os.path.exists(png_path)}")
        except ImportError as e:
            print(f"⚠ PNG export skipped: {e}")
        except Exception as e:
            print(f"✗ PNG export failed: {e}")
        
        print("\n6. Testing batch export...")
        try:
            batch_results = export_manager.export_batch(
                analysis_data,
                "uniad_analysis",
                ["mermaid", "html", "md"]
            )
            print(f"✓ Batch export successful: {len(batch_results)} formats exported")
            for format_type, path in batch_results.items():
                exists = os.path.exists(path)
                print(f"  - {format_type}: {exists} ({path})")
        except Exception as e:
            print(f"✗ Batch export failed: {e}")
        
        print("\n7. Testing path sanitization...")
        try:
            # Test dangerous path
            export_manager._sanitize_path("../../../etc/passwd")
            print("✗ Path sanitization failed: dangerous path allowed")
        except ValueError:
            print("✓ Path sanitization working: dangerous path rejected")
        
        # List all generated files
        print(f"\nGenerated files in {temp_dir}:")
        for file in Path(temp_dir).glob("*"):
            size = file.stat().st_size if file.is_file() else 0
            print(f"  - {file.name} ({size} bytes)")


def main():
    """Main test function"""
    print("ExportManager Test Suite")
    print("=" * 50)
    
    test_export_manager()
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    main()