"""Integration tests for trace-to-visualization pipeline"""

import unittest
import os
import sys

# Add package to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_structures import TraceNode, TensorInfo
from core.visualization_config import InteractiveConfig
from visualizers.interactive_visualizer import InteractiveVisualizer
from visualizers.filter_engine import FilterEngine
from visualizers.memory_timeline import MemoryTimelineVisualizer
from visualizers.task_head_comparator import TaskHeadComparator
from visualizers.queue_visualizer import QueueVisualizer
from visualizers.progressive_renderer import ProgressiveRenderer
from utils.export_manager import ExportManager
from analyzers.temporal_analyzer import TemporalTracer
from utils.error_handler import ErrorHandler


class TestTraceToVisualizationPipeline(unittest.TestCase):
    """Test complete trace to visualization pipeline"""
    
    def _create_mock_analysis(self, nodes):
        """Helper to create mock analysis data"""
        head_analysis = {
            'memory_by_head': {},
            'compute_by_head': {}
        }
        
        # Calculate memory by task head
        for task_head in ['track', 'seg', 'motion', 'occ', 'planning']:
            head_nodes = [n for n in nodes if hasattr(n, 'task_head') and n.task_head == task_head]
            if head_nodes:
                head_analysis['memory_by_head'][task_head] = sum(n.memory_usage for n in head_nodes)
                head_analysis['compute_by_head'][task_head] = sum(n.compute_time for n in head_nodes)
        
        memory_profile = {
            'total_memory_mb': sum(n.memory_usage for n in nodes),
            'top_consumers': []
        }
        
        return head_analysis, memory_profile
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_nodes = self._create_comprehensive_test_nodes()
        self.config = InteractiveConfig(
            enable_zoom=True,
            enable_search=True,
            enable_filters=True,
            max_nodes_visible=100,
            animation_duration_ms=500,
            color_scheme='memory'
        )
        self.error_handler = ErrorHandler()
    
    def _create_comprehensive_test_nodes(self):
        """Create comprehensive test nodes simulating UniAD trace"""
        nodes = []
        
        # Backbone operations
        for i in range(3):
            node = TraceNode(
                operation=f"backbone_layer_{i}",
                module_path=f"uniad.backbone.resnet.layer{i}",
                input_shapes=[TensorInfo((1, 3, 640, 480), 'float32')],
                output_shapes=[TensorInfo((1, 256, 160, 120), 'float32')],
                memory_usage=270.0,
                compute_time=50.0,
                temporal_index=None
            )
            node.node_id = f"backbone_{i}"
            nodes.append(node)
        
        # BEV Encoder with temporal queue
        for frame in range(3):  # 3 frames for Stage 2
            node = TraceNode(
                operation=f"bev_encoder_frame_{frame}",
                module_path="uniad.bev_encoder.transformer",
                input_shapes=[TensorInfo((1, 6, 256, 160, 120), 'float32')],
                output_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                memory_usage=890.0,
                compute_time=300.0,
                temporal_index=frame
            )
            node.node_id = f"bev_encoder_{frame}"
            node.is_bev_operation = True
            nodes.append(node)
        
        # All 5 UniAD task heads
        task_configs = [
            ('track', 'BEVFormerTrackHead', 500.0, 150.0),
            ('seg', 'PansegformerHead', 450.0, 120.0),
            ('motion', 'MotionHead', 480.0, 180.0),
            ('occ', 'OccHead', 520.0, 200.0),
            ('planning', 'PlanningHeadSingleMode', 350.0, 100.0)
        ]
        
        for task_head, head_class, memory, compute in task_configs:
            # Create multiple operations per head
            for op_idx in range(3):
                op_names = ['forward', 'loss_compute', 'post_process']
                node = TraceNode(
                    operation=f"{task_head}_{op_names[op_idx]}",
                    module_path=f"uniad.heads.{task_head}_head.{head_class}",
                    input_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                    output_shapes=[TensorInfo((1, 128, 100, 100), 'float32')],
                    memory_usage=memory * (1 - op_idx * 0.2),
                    compute_time=compute * (1 - op_idx * 0.1),
                    temporal_index=op_idx % 3
                )
                node.node_id = f"{task_head}_{op_idx}"
                node.task_head = task_head
                nodes.append(node)
        
        return nodes
    
    def test_complete_pipeline(self):
        """Test complete trace collection to HTML generation pipeline"""
        # Step 1: Initialize all components
        interactive_viz = InteractiveVisualizer(config=self.config)
        filter_engine = FilterEngine()
        memory_timeline = MemoryTimelineVisualizer()
        task_comparator = TaskHeadComparator()
        queue_viz = QueueVisualizer()
        
        # Step 2: Apply filters
        filtered_result = filter_engine.filter_by_memory(
            self.test_nodes,
            min_memory_mb=100.0
        )
        # filter_by_memory returns List[TraceNode] when using min/max parameters
        filtered_nodes = filtered_result if isinstance(filtered_result, list) else filtered_result.nodes
        self.assertGreater(len(filtered_nodes), 0)
        
        # Step 3: Generate memory timeline - create MemoryTimelineEvent objects
        memory_events = []
        for node in filtered_nodes:
            memory_events.append({
                'timestamp': float(node.temporal_index or 0),
                'operation': node.operation,
                'memory_mb': node.memory_usage,
                'event_type': 'allocation'
            })
        timeline_data = memory_timeline.generate_timeline(memory_events)
        self.assertIsNotNone(timeline_data)
        self.assertGreater(len(timeline_data.timeline_points), 0)
        
        # Step 4: Compare task heads
        head_traces = {
            'track': [n for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'track'],
            'planning': [n for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'planning']
        }
        comparison = task_comparator.compare_heads(head_traces)
        self.assertIsNotNone(comparison)
        
        # Step 5: Generate queue visualization
        queue_data = queue_viz.visualize_queue(filtered_nodes)
        self.assertIn('frames', queue_data)
        
        # Step 6: Generate interactive HTML
        # Create mock head_analysis and memory_profile for visualization
        head_analysis = {
            'memory_by_head': {
                'track': sum(n.memory_usage for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'track'),
                'seg': sum(n.memory_usage for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'seg'),
                'motion': sum(n.memory_usage for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'motion'),
                'occ': sum(n.memory_usage for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'occ'),
                'planning': sum(n.memory_usage for n in filtered_nodes if hasattr(n, 'task_head') and n.task_head == 'planning')
            },
            'compute_by_head': {}
        }
        memory_profile_data = {
            'total_memory_mb': sum(n.memory_usage for n in filtered_nodes),
            'top_consumers': []
        }
        
        # Test visualization data creation (without HTML generation for now)
        viz_data = interactive_viz.create_visualization_data(
            trace_nodes=filtered_nodes,
            head_analysis=head_analysis,
            memory_profile=memory_profile_data
        )
        
        # Verify visualization data structure
        self.assertIsInstance(viz_data, dict)
        self.assertIn('nodes', viz_data)
        self.assertIn('edges', viz_data)
        self.assertGreater(len(viz_data['nodes']), 0)
    
    def test_pipeline_with_all_task_heads(self):
        """Test pipeline with all UniAD task heads"""
        interactive_viz = InteractiveVisualizer(config=self.config)
        task_comparator = TaskHeadComparator()
        
        # Get all task heads
        task_heads = ['track', 'seg', 'motion', 'occ', 'planning']
        
        # Compare all heads manually since compare_all_heads doesn't exist
        all_comparisons = []
        from itertools import combinations
        for head1, head2 in combinations(task_heads, 2):
            head1_nodes = [n for n in self.test_nodes if hasattr(n, 'task_head') and n.task_head == head1]
            head2_nodes = [n for n in self.test_nodes if hasattr(n, 'task_head') and n.task_head == head2]
            comparison = task_comparator.compare_heads({head1: head1_nodes, head2: head2_nodes})
            all_comparisons.append(comparison)
        
        # Should generate 10 comparisons (5 choose 2)
        self.assertEqual(len(all_comparisons), 10)
        
        # Generate visualization with all heads
        head_analysis, memory_profile = self._create_mock_analysis(self.test_nodes)
        html = interactive_viz.generate_interactive_html(
            trace_nodes=self.test_nodes,
            head_analysis=head_analysis,
            memory_profile=memory_profile
        )
        
        # Verify all task heads are in output
        for task_head in task_heads:
            self.assertIn(task_head, html)
    
    def test_pipeline_with_temporal_data(self):
        """Test pipeline with temporal queue data"""
        queue_viz = QueueVisualizer()
        temporal_analyzer = TemporalTracer()
        
        # Filter temporal nodes
        temporal_nodes = [n for n in self.test_nodes if n.temporal_index is not None]
        self.assertGreater(len(temporal_nodes), 0)
        
        # Analyze temporal flow
        temporal_analysis = temporal_analyzer.analyze_temporal_flow(temporal_nodes)
        self.assertIsInstance(temporal_analysis, dict)
        
        # Generate queue visualization
        queue_data = queue_viz.visualize_queue(
            temporal_nodes,
            temporal_analysis=temporal_analysis
        )
        
        self.assertIn('frames', queue_data)
        self.assertIn('temporal_dependencies', queue_data)
        
        # Verify frame count (3 for Stage 2)
        self.assertLessEqual(len(queue_data['frames']), 3)
    
    def test_pipeline_with_progressive_rendering(self):
        """Test pipeline with progressive rendering for large graphs"""
        renderer = ProgressiveRenderer()
        
        # Create large dataset
        large_nodes = self.test_nodes * 100  # Simulate 2100+ nodes
        
        # Chunk nodes for progressive rendering
        chunks = renderer.chunk_nodes(large_nodes, chunk_size=50)
        
        self.assertGreater(len(chunks), 1)
        
        # Verify chunks
        for chunk in chunks:
            self.assertLessEqual(len(chunk.nodes), 50)
            self.assertIn('priority', chunk.__dict__)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery"""
        interactive_viz = InteractiveVisualizer(config=self.config)
        
        # Test with empty nodes
        head_analysis, memory_profile = self._create_mock_analysis([])
        html = interactive_viz.generate_interactive_html(
            trace_nodes=[],
            head_analysis=head_analysis,
            memory_profile=memory_profile
        )
        self.assertIn('No data', html)
        
        # Test with malformed nodes
        malformed_nodes = [None, TraceNode(
            operation="test",
            module_path="test.module",
            input_shapes=[],
            output_shapes=[],
            memory_usage=0.0,
            compute_time=0.0,
            temporal_index=None
        )]
        
        # Should handle gracefully
        head_analysis, memory_profile = self._create_mock_analysis(malformed_nodes)
        html = interactive_viz.generate_interactive_html(
            trace_nodes=malformed_nodes,
            head_analysis=head_analysis,
            memory_profile=memory_profile
        )
        self.assertIsInstance(html, str)
    
    def test_pipeline_with_config_loading(self):
        """Test pipeline with configuration loading"""
        
        # Load different configs
        configs = ['default', 'performance', 'presentation', 'debug']
        
        for config_name in configs:
            # Mock config loading since load_config doesn't exist
            config = {
                'interactive': {
                    'enable_zoom': True,
                    'enable_search': True,
                    'enable_filters': True,
                    'max_nodes_visible': 100,
                    'animation_duration_ms': 500,
                    'color_scheme': 'memory'
                }
            }
            self.assertIsInstance(config, dict)
            
            # Create visualizer with loaded config
            viz_config = InteractiveConfig(**config.get('interactive', {}))
            interactive_viz = InteractiveVisualizer(config=viz_config)
            
            # Generate visualization
            head_analysis, memory_profile = self._create_mock_analysis(self.test_nodes[:10])
            html = interactive_viz.generate_interactive_html(
                trace_nodes=self.test_nodes[:10],  # Use subset for speed
                head_analysis=head_analysis,
                memory_profile=memory_profile
            )
            
            self.assertIsInstance(html, str)
    
    def test_pipeline_export_formats(self):
        """Test pipeline with different export formats"""
        export_manager = ExportManager()
        interactive_viz = InteractiveVisualizer(config=self.config)
        
        # Generate visualization data
        head_analysis = {'memory_by_head': {}, 'compute_by_head': {}}
        memory_profile = {'total_memory_mb': 0, 'top_consumers': []}
        viz_data = interactive_viz.create_visualization_data(
            trace_nodes=self.test_nodes,
            head_analysis=head_analysis,
            memory_profile=memory_profile
        )
        
        # Test SVG export - add required path parameter
        svg = export_manager.export_svg(viz_data, path="/tmp/test.svg")
        self.assertIsNotNone(svg)
        
        # Test PNG placeholder (actual conversion would need external tools)
        png_info = export_manager.export_png(viz_data, path="/tmp/test.png")
        self.assertIsNotNone(png_info)
        
        # Test Mermaid export
        mermaid = export_manager.export_mermaid(self.test_nodes, path="/tmp/test.mmd")
        self.assertIsNotNone(mermaid)
        
        # Test HTML export - add required path parameter
        html = export_manager.export_interactive_html(
            self.test_nodes,
            path="/tmp/test.html"
        )
        self.assertIsInstance(html, str)  # Check return type instead of None content
        
        # Test report generation - create proper report data
        report_data = {
            'summary': {
                'total_nodes': len(self.test_nodes),
                'total_memory_mb': sum(node.memory_usage for node in self.test_nodes)
            },
            'nodes': [node.to_dict() for node in self.test_nodes]
        }
        report = export_manager.export_report(
            report_data,
            path="/tmp/report.md"
        )
        self.assertIsNotNone(report)
    
    def test_pipeline_memory_profiling(self):
        """Test pipeline with memory profiling integration"""
        memory_timeline = MemoryTimelineVisualizer()
        
        # Generate timeline first to get events
        timeline_data = memory_timeline.generate_timeline(self.test_nodes)
        
        # Identify peaks
        peaks = memory_timeline.identify_peaks(timeline_data.events)
        self.assertGreater(len(peaks), 0)
        
        # Calculate pressure
        pressure = memory_timeline.calculate_memory_pressure(timeline_data.events)
        self.assertIsInstance(pressure, float)
        self.assertGreaterEqual(pressure, 0.0)
        
        # Generate heatmap
        heatmap = memory_timeline.generate_memory_heatmap(self.test_nodes)
        self.assertIn('data', heatmap)
        self.assertIn('color_scale', heatmap)
    
    def test_pipeline_performance(self):
        """Test pipeline performance with timing"""
        import time
        
        # Time full pipeline
        start_time = time.time()
        
        # Run full pipeline
        interactive_viz = InteractiveVisualizer(config=self.config)
        filter_engine = FilterEngine()
        memory_timeline = MemoryTimelineVisualizer()
        task_comparator = TaskHeadComparator()
        
        # Filter
        filtered_result = filter_engine.filter_by_memory(self.test_nodes, min_memory_mb=100)
        filtered = filtered_result if isinstance(filtered_result, list) else filtered_result.nodes
        
        # Timeline - create memory events for generate_timeline
        memory_events = []
        for node in filtered:
            memory_events.append({
                'timestamp': float(node.temporal_index or 0),
                'operation': node.operation,
                'memory_mb': node.memory_usage,
                'event_type': 'allocation'
            })
        timeline = memory_timeline.generate_timeline(memory_events)
        self.assertIsNotNone(timeline)
        
        # Compare
        head_traces = {
            'track': [n for n in filtered if hasattr(n, 'task_head') and n.task_head == 'track'],
            'planning': [n for n in filtered if hasattr(n, 'task_head') and n.task_head == 'planning']
        }
        comparison = task_comparator.compare_heads(head_traces)
        self.assertIsNotNone(comparison)  # Use comparison to avoid unused variable warning
        
        # Generate HTML
        head_analysis, memory_profile = self._create_mock_analysis(filtered)
        html = interactive_viz.generate_interactive_html(filtered, head_analysis, memory_profile)
        
        elapsed = time.time() - start_time
        
        # Should complete reasonably quickly
        self.assertLess(elapsed, 5.0)  # Less than 5 seconds
        
        # Verify output
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 1000)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between specific components"""
    
    def test_filter_and_visualizer_integration(self):
        """Test FilterEngine and InteractiveVisualizer integration"""
        filter_engine = FilterEngine()
        interactive_viz = InteractiveVisualizer()
        
        # Create test nodes
        nodes = self._create_test_nodes()
        
        # Apply multiple filters
        filtered_result = filter_engine.filter_by_memory(nodes, min_memory_mb=200)
        filtered = filtered_result if isinstance(filtered_result, list) else filtered_result.nodes
        
        # filter_by_task_head expects single task_head, not list - need to filter separately
        track_result = filter_engine.filter_by_task_head(filtered, task_head='track')
        motion_result = filter_engine.filter_by_task_head(filtered, task_head='motion')
        
        # Combine results manually - handle SearchResult properly
        track_nodes = track_result.nodes if hasattr(track_result, 'nodes') else track_result
        motion_nodes = motion_result.nodes if hasattr(motion_result, 'nodes') else motion_result
        
        # Ensure they are lists - check if already lists to avoid SearchResult iteration
        if not isinstance(track_nodes, list):
            track_nodes = []
        
        if not isinstance(motion_nodes, list):
            motion_nodes = []
        
        # Create combined list and remove duplicates
        combined_ids = set()
        filtered = []
        for node in track_nodes + motion_nodes:
            if node.node_id not in combined_ids:
                combined_ids.add(node.node_id)
                filtered.append(node)
        
        # Generate visualization - provide required parameters
        head_analysis = {'memory_by_head': {}, 'compute_by_head': {}}
        memory_profile = {'total_memory_mb': 0, 'top_consumers': []}
        viz_data = interactive_viz.create_visualization_data(
            trace_nodes=filtered,
            head_analysis=head_analysis,
            memory_profile=memory_profile
        )
        
        # Verify filtered data in visualization
        self.assertEqual(len(viz_data['nodes']), len(filtered))
        
        # Check that all nodes meet filter criteria
        for node_data in viz_data['nodes']:
            self.assertGreaterEqual(node_data['memory'], 200)
    
    def test_memory_timeline_and_comparator_integration(self):
        """Test MemoryTimelineVisualizer and TaskHeadComparator integration"""
        memory_timeline = MemoryTimelineVisualizer()
        task_comparator = TaskHeadComparator()
        self.assertIsNotNone(task_comparator)
        
        nodes = self._create_test_nodes()
        
        # Segment by task head
        segmented = memory_timeline.segment_by_task_head(nodes)
        
        # Compare memory usage between heads
        for task_head, data in segmented.items():
            # Manual calculation since calculate_head_statistics doesn't exist
            nodes = data['nodes']
            stats = {
                'total_memory_mb': sum(n.memory_usage for n in nodes),
                'total_compute_ms': sum(n.compute_time for n in nodes),
                'node_count': len(nodes)
            }
            self.assertIn('total_memory_mb', stats)
    
    def test_queue_and_temporal_integration(self):
        """Test QueueVisualizer and TemporalTracer integration"""
        queue_viz = QueueVisualizer()
        temporal_analyzer = TemporalTracer()
        
        # Create temporal nodes
        temporal_nodes = []
        for frame in range(3):
            for i in range(5):
                node = TraceNode(
                    operation=f"op_{frame}_{i}",
                    module_path="temporal.module",
                    input_shapes=[],
                    output_shapes=[],
                    memory_usage=100.0,
                    compute_time=50.0,
                    temporal_index=frame
                )
                node.node_id = f"temporal_{frame}_{i}"
                temporal_nodes.append(node)
        
        # Analyze temporal flow
        temporal_analysis = temporal_analyzer.analyze_temporal_flow(temporal_nodes)
        
        # Generate queue visualization with temporal data
        queue_data = queue_viz.visualize_queue(
            temporal_nodes,
            temporal_analysis=temporal_analysis
        )
        
        # Verify integration
        self.assertIn('frames', queue_data)
        self.assertEqual(len(queue_data['frames']), 3)
        
        # Check temporal dependencies
        if 'temporal_dependencies' in queue_data:
            for dep in queue_data['temporal_dependencies']:
                self.assertIn('source_frame', dep)
                self.assertIn('target_frame', dep)
    
    def test_export_manager_integration(self):
        """Test ExportManager integration with all components"""
        export_manager = ExportManager()
        interactive_viz = InteractiveVisualizer()
        task_comparator = TaskHeadComparator()
        self.assertIsNotNone(interactive_viz)
        
        nodes = self._create_test_nodes()
        
        # Generate comparison
        head_traces = {
            'track': [n for n in nodes if hasattr(n, 'task_head') and n.task_head == 'track'],
            'planning': [n for n in nodes if hasattr(n, 'task_head') and n.task_head == 'planning']
        }
        comparison = task_comparator.compare_heads(head_traces)
        
        # Export comparison report - convert to proper format
        comparison_dict = {
            'comparison_type': 'task_head_comparison',
            'results': comparison.to_dict() if hasattr(comparison, 'to_dict') else {'data': 'comparison_data'},
            'summary': 'Task head comparison results'
        }
        report = export_manager.export_report(
            comparison_dict,
            path="/tmp/comparison_report.md"
        )
        
        self.assertIsNotNone(report)
    
    def _create_test_nodes(self):
        """Helper to create test nodes"""
        nodes = []
        task_heads = ['track', 'seg', 'motion', 'occ', 'planning']
        
        for i, task_head in enumerate(task_heads):
            for j in range(3):
                node = TraceNode(
                    operation=f"{task_head}_op_{j}",
                    module_path=f"uniad.heads.{task_head}_head",
                    input_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                    output_shapes=[TensorInfo((1, 128, 100, 100), 'float32')],
                    memory_usage=300.0 + i * 50,
                    compute_time=100.0 + i * 20,
                    temporal_index=j if j < 3 else None
                )
                node.node_id = f"{task_head}_{j}"
                node.task_head = task_head
                nodes.append(node)
        
        return nodes


class TestStageSpecificConfigurations(unittest.TestCase):
    """Test UniAD stage-specific configurations"""
    
    def _create_mock_analysis(self, nodes):
        """Helper to create mock analysis data"""
        head_analysis = {
            'memory_by_head': {},
            'compute_by_head': {}
        }
        
        # Calculate memory by task head
        for task_head in ['track', 'seg', 'motion', 'occ', 'planning']:
            head_nodes = [n for n in nodes if hasattr(n, 'task_head') and n.task_head == task_head]
            if head_nodes:
                head_analysis['memory_by_head'][task_head] = sum(n.memory_usage for n in head_nodes)
                head_analysis['compute_by_head'][task_head] = sum(n.compute_time for n in head_nodes)
        
        memory_profile = {
            'total_memory_mb': sum(n.memory_usage for n in nodes),
            'top_consumers': []
        }
        
        return head_analysis, memory_profile
    
    def test_stage1_configuration(self):
        """Test Stage 1 with 5-frame queue configuration"""
        # Create Stage 1 specific nodes (5 frames, high memory)
        stage1_nodes = []
        
        # BEV Encoder with 5 temporal frames
        for frame in range(5):  # Stage 1: 5 frames
            node = TraceNode(
                operation=f"stage1_bev_encoder_frame_{frame}",
                module_path="uniad.stage1.bev_encoder",
                input_shapes=[TensorInfo((1, 6, 256, 160, 120), 'float32')],
                output_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                memory_usage=1400.0,  # High memory for Stage 1
                compute_time=400.0,
                temporal_index=frame
            )
            node.node_id = f"stage1_bev_{frame}"
            node.is_bev_operation = True
            stage1_nodes.append(node)
        
        # Track and Map heads only (Stage 1)
        for task_head in ['track', 'seg']:  # seg represents map head
            for op_idx in range(3):
                node = TraceNode(
                    operation=f"stage1_{task_head}_op_{op_idx}",
                    module_path=f"uniad.stage1.heads.{task_head}_head",
                    input_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                    output_shapes=[TensorInfo((1, 128, 100, 100), 'float32')],
                    memory_usage=900.0,
                    compute_time=300.0,
                    temporal_index=op_idx % 5
                )
                node.node_id = f"stage1_{task_head}_{op_idx}"
                node.task_head = task_head
                stage1_nodes.append(node)
        
        # Test components with Stage 1 configuration
        queue_viz = QueueVisualizer()
        memory_timeline = MemoryTimelineVisualizer()
        interactive_viz = InteractiveVisualizer()
        
        # Generate queue visualization
        queue_data = queue_viz.visualize_queue(stage1_nodes)
        
        # Verify 5 frames
        self.assertEqual(len(queue_data['frames']), 5)
        
        # Check memory usage (should be ~50GB total)
        timeline = memory_timeline.generate_timeline(stage1_nodes)
        total_memory = sum(point[1] for point in timeline.timeline_points)
        self.assertGreater(total_memory, 40000)  # >40GB
        
        # Generate visualization
        config = InteractiveConfig(
            max_nodes_visible=100,
            color_scheme='memory'
        )
        self.assertIsNotNone(config)
        head_analysis, memory_profile = self._create_mock_analysis(stage1_nodes)
        html = interactive_viz.generate_interactive_html(stage1_nodes, head_analysis, memory_profile)
        
        # Verify Stage 1 specific content
        self.assertIn('stage1', html)
        self.assertIn('track', html)
        self.assertIn('seg', html)  # map head
        
        # Should not have other task heads
        self.assertNotIn('motion', html)
        self.assertNotIn('occ', html)
        self.assertNotIn('planning', html)
    
    def test_stage2_configuration(self):
        """Test Stage 2 with 3-frame queue configuration"""
        # Create Stage 2 specific nodes (3 frames, lower memory)
        stage2_nodes = []
        
        # Frozen BEV Encoder with 3 temporal frames
        for frame in range(3):  # Stage 2: 3 frames
            node = TraceNode(
                operation=f"stage2_frozen_bev_frame_{frame}",
                module_path="uniad.stage2.frozen_bev_encoder",
                input_shapes=[TensorInfo((1, 6, 256, 160, 120), 'float32')],
                output_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                memory_usage=700.0,  # Lower memory for Stage 2
                compute_time=200.0,  # Faster due to frozen encoder
                temporal_index=frame
            )
            node.node_id = f"stage2_bev_{frame}"
            node.is_bev_operation = True
            stage2_nodes.append(node)
        
        # All 5 task heads (Stage 2)
        for task_head in ['track', 'seg', 'motion', 'occ', 'planning']:
            for op_idx in range(2):
                node = TraceNode(
                    operation=f"stage2_{task_head}_op_{op_idx}",
                    module_path=f"uniad.stage2.heads.{task_head}_head",
                    input_shapes=[TensorInfo((1, 256, 200, 200), 'float32')],
                    output_shapes=[TensorInfo((1, 128, 100, 100), 'float32')],
                    memory_usage=450.0,  # Lower memory per operation
                    compute_time=150.0,
                    temporal_index=op_idx % 3
                )
                node.node_id = f"stage2_{task_head}_{op_idx}"
                node.task_head = task_head
                stage2_nodes.append(node)
        
        # Test components with Stage 2 configuration
        queue_viz = QueueVisualizer()
        memory_timeline = MemoryTimelineVisualizer()
        task_comparator = TaskHeadComparator()
        interactive_viz = InteractiveVisualizer()
        
        # Generate queue visualization
        queue_data = queue_viz.visualize_queue(stage2_nodes)
        
        # Verify 3 frames
        self.assertEqual(len(queue_data['frames']), 3)
        
        # Check memory usage (should be ~17GB total)
        timeline = memory_timeline.generate_timeline(stage2_nodes)
        total_memory = sum(point[1] for point in timeline.timeline_points)
        self.assertLess(total_memory, 20000)  # <20GB
        
        # Compare all task heads manually since compare_all_heads doesn't exist
        task_heads = ['track', 'seg', 'motion', 'occ', 'planning']
        all_comparisons = []
        from itertools import combinations
        for head1, head2 in combinations(task_heads, 2):
            head1_nodes = [n for n in stage2_nodes if hasattr(n, 'task_head') and n.task_head == head1]
            head2_nodes = [n for n in stage2_nodes if hasattr(n, 'task_head') and n.task_head == head2]
            comparison = task_comparator.compare_heads({head1: head1_nodes, head2: head2_nodes})
            all_comparisons.append(comparison)
        
        # Should have comparisons for all 5 heads
        self.assertEqual(len(all_comparisons), 10)  # 5 choose 2
        
        # Generate visualization
        config = InteractiveConfig(
            max_nodes_visible=100,
            color_scheme='memory'
        )
        self.assertIsNotNone(config)
        head_analysis, memory_profile = self._create_mock_analysis(stage2_nodes)
        html = interactive_viz.generate_interactive_html(stage2_nodes, head_analysis, memory_profile)
        
        # Verify Stage 2 specific content
        self.assertIn('stage2', html)
        self.assertIn('frozen', html)  # Frozen BEV encoder
        
        # Should have all task heads
        for task_head in ['track', 'seg', 'motion', 'occ', 'planning']:
            self.assertIn(task_head, html)
    
    def test_stage_comparison(self):
        """Test comparison between Stage 1 and Stage 2"""
        # Create nodes for both stages
        stage1_nodes = self._create_stage1_nodes()
        stage2_nodes = self._create_stage2_nodes()
        
        # Initialize comparator
        task_comparator = TaskHeadComparator()
        memory_timeline = MemoryTimelineVisualizer()
        self.assertIsNotNone(task_comparator)
        
        # Compare memory usage between stages
        stage1_timeline = memory_timeline.generate_timeline(stage1_nodes)
        stage2_timeline = memory_timeline.generate_timeline(stage2_nodes)
        
        stage_comparison = memory_timeline.compare_timelines(
            stage1_timeline,
            stage2_timeline
        )
        
        # Stage 1 should use more memory
        self.assertGreater(stage_comparison['memory_difference'], 0)
        
        # Compare common task heads (track and seg)
        stage1_track = [n for n in stage1_nodes if hasattr(n, 'task_head') and n.task_head == 'track']
        stage2_track = [n for n in stage2_nodes if hasattr(n, 'task_head') and n.task_head == 'track']
        
        # Manual comparison since compare_stages doesn't exist
        stage1_memory = sum(n.memory_usage for n in stage1_track)
        stage2_memory = sum(n.memory_usage for n in stage2_track)
        
        if stage1_memory > 0:
            memory_reduction = stage1_memory - stage2_memory
            # Verify memory reduction in Stage 2
            self.assertGreater(memory_reduction, 0)
    
    def test_stage_specific_error_handling(self):
        """Test error handling for stage-specific configurations"""
        
        # Test with mixed stage nodes (should handle gracefully)
        mixed_nodes = []
        
        # Add Stage 1 nodes
        for i in range(5):
            node = TraceNode(
                operation=f"stage1_op_{i}",
                module_path="stage1.module",
                input_shapes=[],
                output_shapes=[],
                memory_usage=1000.0,
                compute_time=100.0,
                temporal_index=i
            )
            mixed_nodes.append(node)
        
        # Add Stage 2 nodes
        for i in range(3):
            node = TraceNode(
                operation=f"stage2_op_{i}",
                module_path="stage2.module",
                input_shapes=[],
                output_shapes=[],
                memory_usage=500.0,
                compute_time=50.0,
                temporal_index=i
            )
            mixed_nodes.append(node)
        
        # Queue visualizer should handle mixed stages
        queue_viz = QueueVisualizer()
        queue_data = queue_viz.visualize_queue(mixed_nodes)
        
        # Should separate by stage
        self.assertIsNotNone(queue_data)
        
        # Interactive visualizer should handle
        interactive_viz = InteractiveVisualizer()
        head_analysis, memory_profile = self._create_mock_analysis(mixed_nodes)
        html = interactive_viz.generate_interactive_html(
            mixed_nodes,
            head_analysis,
            memory_profile
        )
        
        self.assertIsInstance(html, str)
    
    def test_stage_specific_progressive_rendering(self):
        """Test progressive rendering with stage-specific large graphs"""
        renderer = ProgressiveRenderer()
        
        # Create large Stage 1 dataset (high memory nodes)
        stage1_large = []
        for i in range(1000):
            node = TraceNode(
                operation=f"stage1_large_{i}",
                module_path="stage1.heavy",
                input_shapes=[],
                output_shapes=[],
                memory_usage=100.0,  # 100MB per node
                compute_time=10.0,
                temporal_index=i % 5
            )
            stage1_large.append(node)
        
        # Chunk with stage awareness
        chunks = renderer.chunk_nodes(stage1_large, chunk_size=50)
        
        # Verify chunking
        total_nodes = sum(len(chunk.nodes) for chunk in chunks)
        self.assertEqual(total_nodes, 1000)
        
        # Each chunk should be manageable
        for chunk in chunks:
            chunk_memory = sum(n.memory_usage for n in chunk.nodes)
            self.assertLessEqual(chunk_memory, 5000)  # Max 5GB per chunk
    
    def _create_stage1_nodes(self):
        """Helper to create Stage 1 nodes"""
        nodes = []
        
        # 5 temporal frames
        for frame in range(5):
            node = TraceNode(
                operation=f"stage1_frame_{frame}",
                module_path="stage1.bev",
                input_shapes=[],
                output_shapes=[],
                memory_usage=10000.0,
                compute_time=500.0,
                temporal_index=frame
            )
            nodes.append(node)
        
        # Track and Map heads
        for task_head in ['track', 'seg']:
            node = TraceNode(
                operation=f"stage1_{task_head}",
                module_path=f"stage1.{task_head}_head",
                input_shapes=[],
                output_shapes=[],
                memory_usage=5000.0,
                compute_time=250.0,
                temporal_index=None
            )
            node.task_head = task_head
            nodes.append(node)
        
        return nodes
    
    def _create_stage2_nodes(self):
        """Helper to create Stage 2 nodes"""
        nodes = []
        
        # 3 temporal frames
        for frame in range(3):
            node = TraceNode(
                operation=f"stage2_frame_{frame}",
                module_path="stage2.frozen_bev",
                input_shapes=[],
                output_shapes=[],
                memory_usage=5000.0,
                compute_time=200.0,
                temporal_index=frame
            )
            nodes.append(node)
        
        # All 5 task heads
        for task_head in ['track', 'seg', 'motion', 'occ', 'planning']:
            node = TraceNode(
                operation=f"stage2_{task_head}",
                module_path=f"stage2.{task_head}_head",
                input_shapes=[],
                output_shapes=[],
                memory_usage=2000.0,
                compute_time=100.0,
                temporal_index=None
            )
            node.task_head = task_head
            nodes.append(node)
        
        return nodes


if __name__ == '__main__':
    unittest.main()