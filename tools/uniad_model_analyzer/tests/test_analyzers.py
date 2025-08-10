"""
Unit tests for analyzer modules.
"""

import unittest
import torch
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TaskHead, TensorShape
from analyzers.multi_head_analyzer import MultiHeadAnalyzer, TaskHeadProfile
from analyzers.temporal_analyzer import TemporalAnalyzer, TemporalFrameProfile
from analyzers.bev_analyzer import BEVAnalyzer, BEVGridProfile
from analyzers.memory_profiler import MemoryProfiler, MemorySnapshot
from analyzers.dtype_analyzer import DTypeAnalyzer, DTypeProfile


def create_tensor_shape(shape, dtype=torch.float32, device='cuda', requires_grad=False):
    """Helper to create TensorShape with calculated memory."""
    import numpy as np
    elements = np.prod(shape) if shape else 0
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int64: 8,
        torch.int8: 1,
        torch.bool: 1,
    }
    memory_bytes = elements * dtype_sizes.get(dtype, 4)
    return TensorShape(
        shape=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        memory_bytes=memory_bytes
    )


class TestMultiHeadAnalyzer(unittest.TestCase):
    """Test cases for MultiHeadAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiHeadAnalyzer()
        self.trace_nodes = self._create_multi_head_trace()
    
    def _create_multi_head_trace(self) -> List[TraceNode]:
        """Create trace nodes simulating multi-head architecture."""
        nodes = []
        
        # Shared backbone
        for i in range(5):
            node = TraceNode(
                id=f"backbone_{i}",
                name="Conv2d",
                module_path=f"backbone.layer{i}",
                task_head=TaskHead.NONE,
                start_time=i * 1000,
                duration=1000,
                memory_allocated=0,
                memory_freed=0,
            )
            node.cuda_memory_allocated = 1024 * 1024  # 1MB
            nodes.append(node)
        
        # Task-specific heads
        for head in [TaskHead.TRACK, TaskHead.SEGMENTATION, TaskHead.MOTION]:
            for i in range(3):
                node = TraceNode(
                    id=f"{head.value}_{i}",
                    name="Linear",
                    module_path=f"{head.value}_head.layer{i}",
                    task_head=head,
                    start_time=(10 + i) * 1000,
                        duration=1000,
                    memory_allocated=0,
                    memory_freed=0,
                )
                node.cuda_memory_allocated = 512 * 1024  # 512KB
                node.parameters = 10000
                nodes.append(node)
        
        return nodes
    
    def test_analyze_task_head(self):
        """Test task head analysis."""
        profile = self.analyzer.analyze_task_head(TaskHead.TRACK, self.trace_nodes)
        
        self.assertIsInstance(profile, TaskHeadProfile)
        self.assertEqual(profile.task_head, TaskHead.TRACK)
        self.assertGreater(profile.operation_count, 0)
        self.assertGreater(profile.total_memory_bytes, 0)
    
    def test_analyze_head_interactions(self):
        """Test head interaction analysis."""
        analysis = self.analyzer.analyze_head_interactions(self.trace_nodes)
        
        self.assertIn('shared_operation_count', analysis)
        self.assertIn('dependency_graph', analysis)
        self.assertIn('resource_distribution', analysis)
        self.assertIsInstance(analysis['shared_operation_count'], int)
    
    def test_compute_head_metrics(self):
        """Test head metrics computation."""
        metrics = self.analyzer.compute_head_metrics(self.trace_nodes)
        
        self.assertIsInstance(metrics, dict)
        # Should have metrics for heads with operations
        for head in [TaskHead.TRACK, TaskHead.SEGMENTATION, TaskHead.MOTION]:
            if head in metrics:
                self.assertIn('operation_count', metrics[head])
                self.assertIn('memory_mb', metrics[head])
                self.assertIn('efficiency_score', metrics[head])
    
    def test_generate_head_comparison(self):
        """Test head comparison generation."""
        # First analyze heads
        for head in TaskHead:
            if head != TaskHead.NONE:
                self.analyzer.analyze_task_head(head, self.trace_nodes)
        
        comparison = self.analyzer.generate_head_comparison()
        
        self.assertIn('head_count', comparison)
        self.assertIn('total_operations', comparison)
        self.assertIn('head_rankings', comparison)


class TestTemporalAnalyzer(unittest.TestCase):
    """Test cases for TemporalAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TemporalAnalyzer(num_frames=3)
        self.trace_nodes = self._create_temporal_trace()
    
    def _create_temporal_trace(self) -> List[TraceNode]:
        """Create trace nodes simulating temporal processing."""
        nodes = []
        
        # Create nodes for 3 temporal frames
        for frame in range(3):
            for i in range(5):
                node = TraceNode(
                    id=f"frame{frame}_op{i}",
                    name="TemporalConv",
                    module_path=f"temporal.frame{frame}.layer{i}",
                    temporal_frame=frame,
                    start_time=(frame * 10 + i) * 1000,
                    duration=1000,
                    memory_allocated=0,
                    memory_freed=0,
                )
                node.cuda_memory_allocated = 2 * 1024 * 1024  # 2MB
                
                # Add some attention operations
                if i % 2 == 0:
                    node.module_path = f"temporal.attention.frame{frame}"
                    node.name = "SelfAttention"
                
                nodes.append(node)
        
        return nodes
    
    def test_analyze_temporal_flow(self):
        """Test temporal flow analysis."""
        analysis = self.analyzer.analyze_temporal_flow(self.trace_nodes)
        
        self.assertIn('num_frames', analysis)
        self.assertIn('frame_profiles', analysis)
        self.assertIn('temporal_patterns', analysis)
        self.assertIn('temporal_efficiency', analysis)
        
        self.assertEqual(analysis['num_frames'], 3)
        self.assertIsInstance(analysis['frame_profiles'], list)
        self.assertEqual(len(analysis['frame_profiles']), 3)
    
    def test_analyze_queue_memory(self):
        """Test queue memory analysis."""
        analysis_3 = self.analyzer.analyze_queue_memory(3)
        analysis_5 = self.analyzer.analyze_queue_memory(5)
        
        self.assertIn('total_memory', analysis_3)
        self.assertIn('optimal_for', analysis_3)
        self.assertIn('optimization_suggestions', analysis_3)
        
        # 5-frame queue should use more memory
        self.assertGreater(analysis_5['total_memory'], analysis_3['total_memory'])
    
    def test_visualize_temporal_attention(self):
        """Test temporal attention visualization."""
        attention_data = self.analyzer.visualize_temporal_attention()
        
        self.assertIn('attention_matrices', attention_data)
        self.assertIn('frame_connections', attention_data)
        self.assertIsInstance(attention_data['attention_matrices'], list)
    
    def test_frame_profile_creation(self):
        """Test frame profile creation."""
        self.analyzer.analyze_temporal_flow(self.trace_nodes)
        
        # Check frame profiles
        for frame_idx, profile in self.analyzer.frame_profiles.items():
            self.assertIsInstance(profile, TemporalFrameProfile)
            self.assertEqual(profile.frame_index, frame_idx)
            self.assertGreater(profile.operation_count, 0)


class TestBEVAnalyzer(unittest.TestCase):
    """Test cases for BEVAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BEVAnalyzer(
            grid_size=(200, 200),
            spatial_extent=(-51.2, 51.2, -51.2, 51.2)
        )
        self.trace_nodes = self._create_bev_trace()
    
    def _create_bev_trace(self) -> List[TraceNode]:
        """Create trace nodes simulating BEV processing."""
        nodes = []
        
        # Camera to BEV projection
        for cam in range(6):  # 6 cameras
            node = TraceNode(
                id=f"cam{cam}_project",
                name="ProjectionConv",
                module_path=f"bev_encoder.camera{cam}.project",
                start_time=cam * 1000,
                duration=1000,
                memory_allocated=0,
                memory_freed=0,
            )
            node.cuda_memory_allocated = 10 * 1024 * 1024  # 10MB
            
            # Add shapes
            node.input_shapes = [create_tensor_shape((1, 256, 32, 88))]
            node.output_shapes = [create_tensor_shape((1, 256, 200, 200))]
            nodes.append(node)
        
        # BEV aggregation
        node = TraceNode(
            id="bev_aggregate",
            name="BEVAggregate",
            module_path="bev_encoder.aggregate",
            bev_operation=True,
            start_time=7000,
            duration=1000,
            memory_allocated=0,
            memory_freed=0,
        )
        node.cuda_memory_allocated = 50 * 1024 * 1024  # 50MB
        node.output_shapes = [create_tensor_shape((1, 256, 200, 200))]
        nodes.append(node)
        
        return nodes
    
    def test_analyze_bev_encoder(self):
        """Test BEV encoder analysis."""
        analysis = self.analyzer.analyze_bev_encoder(self.trace_nodes)
        
        self.assertIn('bev_grid_config', analysis)
        self.assertIn('bev_operations', analysis)
        self.assertIn('spatial_transformations', analysis)
        self.assertIn('memory_analysis', analysis)
        self.assertIn('optimization_opportunities', analysis)
    
    def test_analyze_lift_splat_shoot(self):
        """Test Lift-Splat-Shoot analysis."""
        analysis = self.analyzer.analyze_lift_splat_shoot(self.trace_nodes)
        
        self.assertIn('lift_stage', analysis)
        self.assertIn('splat_stage', analysis)
        self.assertIn('shoot_stage', analysis)
        self.assertIn('total_lss_operations', analysis)
    
    def test_profile_bev_grid(self):
        """Test BEV grid profiling."""
        grid_shape = (1, 256, 200, 200)
        operations = self.trace_nodes[-1:]  # BEV aggregate operation
        
        profile = self.analyzer.profile_bev_grid(grid_shape, operations)
        
        self.assertIsInstance(profile, BEVGridProfile)
        self.assertEqual(profile.grid_resolution, (200, 200))
        self.assertEqual(profile.feature_channels, 256)
        self.assertGreater(profile.memory_bytes, 0)
    
    def test_grid_configuration(self):
        """Test grid configuration details."""
        self.analyzer.analyze_bev_encoder(self.trace_nodes)
        config = self.analyzer._get_grid_configuration()
        
        self.assertIn('grid_resolution', config)
        self.assertIn('spatial_extent_m', config)
        self.assertIn('meters_per_pixel', config)
        self.assertEqual(config['grid_resolution'], (200, 200))


class TestMemoryProfiler(unittest.TestCase):
    """Test cases for MemoryProfiler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler(target_memory_gb=40.0)
        self.trace_nodes = self._create_memory_trace()
    
    def _create_memory_trace(self) -> List[TraceNode]:
        """Create trace nodes with memory patterns."""
        nodes = []
        
        # Simulate memory allocation pattern
        memory_pattern = [10, 20, 50, 100, 80, 60, 30, 10]  # MB
        
        for i, mem_mb in enumerate(memory_pattern):
            node = TraceNode(
                id=f"op_{i}",
                name=f"Operation{i}",
                module_path=f"model.layer{i}",
                start_time=i * 1000,
                duration=1000,
                memory_allocated=0,
                memory_freed=0,
            )
            node.cuda_memory_allocated = mem_mb * 1024 * 1024
            node.cuda_memory_freed = (mem_mb // 2) * 1024 * 1024 if i > 4 else 0
            nodes.append(node)
        
        return nodes
    
    def test_profile_memory(self):
        """Test memory profiling."""
        analysis = self.profiler.profile_memory(self.trace_nodes)
        
        self.assertIn('summary', analysis)
        self.assertIn('peak_analysis', analysis)
        self.assertIn('memory_timeline', analysis)
        self.assertIn('optimization_suggestions', analysis)
        self.assertIn('memory_efficiency', analysis)
    
    def test_analyze_memory_lifecycle(self):
        """Test memory lifecycle analysis."""
        lifecycle = self.profiler.analyze_memory_lifecycle(self.trace_nodes)
        
        self.assertIn('allocation_frequency', lifecycle)
        self.assertIn('memory_leaks', lifecycle)
        self.assertIsInstance(lifecycle['memory_leaks'], list)
    
    def test_suggest_gradient_checkpointing(self):
        """Test gradient checkpointing suggestions."""
        # Add gradient info to some nodes
        for node in self.trace_nodes[:3]:
            node.gradient_info = {"grad_input": (1024,), "grad_output": (1024,)}
            node.cuda_memory_allocated = 200 * 1024 * 1024  # 200MB
        
        suggestions = self.profiler.suggest_gradient_checkpointing(self.trace_nodes)
        
        self.assertIsInstance(suggestions, list)
        if suggestions:
            self.assertIn('module', suggestions[0])
            self.assertIn('expected_savings_mb', suggestions[0])
    
    def test_analyze_mixed_precision_opportunities(self):
        """Test mixed precision analysis."""
        # Add tensor shapes with dtypes
        for node in self.trace_nodes:
            node.output_shapes = [create_tensor_shape((32, 256, 14, 14))]
        
        analysis = self.profiler.analyze_mixed_precision_opportunities(self.trace_nodes)
        
        self.assertIn('fp32_operations', analysis)
        self.assertIn('fp16_compatible', analysis)
        self.assertIn('potential_savings_gb', analysis)
        self.assertIn('recommendations', analysis)
    
    def test_memory_snapshots(self):
        """Test memory snapshot creation."""
        self.profiler.profile_memory(self.trace_nodes)
        
        self.assertGreater(len(self.profiler.snapshots), 0)
        
        for snapshot in self.profiler.snapshots:
            self.assertIsInstance(snapshot, MemorySnapshot)
            self.assertGreaterEqual(snapshot.allocated, 0)
            self.assertGreaterEqual(snapshot.reserved, 0)


class TestDTypeAnalyzer(unittest.TestCase):
    """Test cases for DTypeAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DTypeAnalyzer()
        self.trace_nodes = self._create_dtype_trace()
    
    def _create_dtype_trace(self) -> List[TraceNode]:
        """Create trace nodes with various dtypes."""
        nodes = []
        
        dtypes = [torch.float32, torch.float16, torch.int64, torch.bool]
        operations = ['Conv2d', 'Linear', 'ReLU', 'Softmax']
        
        for i, (dtype, op) in enumerate(zip(dtypes * 2, operations * 2)):
            node = TraceNode(
                id=f"op_{i}",
                name=op,
                module_path=f"model.{op.lower()}_{i}",
                start_time=i * 1000,
                duration=1000,
                memory_allocated=0,
                memory_freed=0,
            )
            
            # Add tensor shapes with specific dtypes
            shape = (32, 256, 14, 14) if 'Conv' in op else (32, 256)
            node.input_shapes = [create_tensor_shape(shape, dtype=dtype)]
            node.output_shapes = [create_tensor_shape(shape, dtype=dtype)]
            node.cuda_memory_allocated = 10 * 1024 * 1024  # 10MB
            
            nodes.append(node)
        
        return nodes
    
    def test_analyze_dtypes(self):
        """Test dtype analysis."""
        analysis = self.analyzer.analyze_dtypes(self.trace_nodes)
        
        self.assertIn('summary', analysis)
        self.assertIn('dtype_distribution', analysis)
        self.assertIn('memory_breakdown', analysis)
        self.assertIn('mixed_precision_opportunities', analysis)
        self.assertIn('recommendations', analysis)
    
    def test_analyze_precision_requirements(self):
        """Test precision requirement analysis."""
        analysis = self.analyzer.analyze_precision_requirements(self.trace_nodes)
        
        self.assertIn('high_precision_required', analysis)
        self.assertIn('low_precision_suitable', analysis)
        self.assertIn('optimization_ratio', analysis)
        self.assertIsInstance(analysis['optimization_ratio'], (int, float))
    
    def test_suggest_dtype_strategy(self):
        """Test dtype strategy suggestion."""
        strategy = self.analyzer.suggest_dtype_strategy(self.trace_nodes)
        
        self.assertIn('recommended_approach', strategy)
        self.assertIn('expected_memory_savings', strategy)
        self.assertIn('implementation_steps', strategy)
        self.assertIn('risk_assessment', strategy)
    
    def test_dtype_profiles(self):
        """Test dtype profile creation."""
        self.analyzer.analyze_dtypes(self.trace_nodes)
        
        self.assertGreater(len(self.analyzer.dtype_profiles), 0)
        
        for dtype, profile in self.analyzer.dtype_profiles.items():
            self.assertIsInstance(profile, DTypeProfile)
            self.assertEqual(profile.dtype, dtype)
            self.assertGreater(profile.operation_count, 0)
    
    def test_mixed_precision_opportunities(self):
        """Test mixed precision opportunity identification."""
        # Make all operations FP32 for better opportunity detection
        for node in self.trace_nodes:
            node.input_shapes = [create_tensor_shape((32, 256))]
            node.output_shapes = [create_tensor_shape((32, 256))]
        
        self.analyzer.analyze_dtypes(self.trace_nodes)
        
        # Should identify some opportunities
        self.assertIsInstance(self.analyzer.mixed_precision_opportunities, list)
    
    def test_quantization_candidates(self):
        """Test quantization candidate identification."""
        self.analyzer.analyze_dtypes(self.trace_nodes)
        
        # Should identify some candidates
        self.assertIsInstance(self.analyzer.quantization_candidates, list)


class IntegrationTest(unittest.TestCase):
    """Integration tests for multiple analyzers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trace_nodes = self._create_comprehensive_trace()
    
    def _create_comprehensive_trace(self) -> List[TraceNode]:
        """Create a comprehensive trace for integration testing."""
        nodes = []
        
        # Backbone operations
        for i in range(10):
            node = TraceNode(
                id=f"backbone_{i}",
                name="Conv2d",
                module_path=f"backbone.layer{i}",
                task_head=TaskHead.NONE,
                start_time=i * 1000,
                duration=1000,
                memory_allocated=0,
                memory_freed=0,
            )
            node.cuda_memory_allocated = (i + 1) * 1024 * 1024
            node.input_shapes = [create_tensor_shape((2, 256, 50, 50))]
            node.output_shapes = [create_tensor_shape((2, 256, 50, 50))]
            nodes.append(node)
        
        # BEV operations
        for i in range(5):
            node = TraceNode(
                id=f"bev_{i}",
                name="BEVEncoder",
                module_path=f"bev_encoder.layer{i}",
                bev_operation=True,
                start_time=(10 + i) * 1000,
                duration=1000,
                memory_allocated=0,
                memory_freed=0,
            )
            node.cuda_memory_allocated = 20 * 1024 * 1024
            node.output_shapes = [create_tensor_shape((2, 256, 200, 200))]
            nodes.append(node)
        
        # Temporal operations
        for frame in range(3):
            for i in range(3):
                node = TraceNode(
                    id=f"temporal_f{frame}_op{i}",
                    name="TemporalAttention",
                    module_path=f"temporal.frame{frame}.attn{i}",
                    temporal_frame=frame,
                    start_time=(20 + frame * 3 + i) * 1000,
                    duration=1000,
                    memory_allocated=0,
                    memory_freed=0,
                )
                node.cuda_memory_allocated = 15 * 1024 * 1024
                nodes.append(node)
        
        # Task heads
        for head in [TaskHead.TRACK, TaskHead.SEGMENTATION, TaskHead.PLANNING]:
            for i in range(3):
                node = TraceNode(
                    id=f"{head.value}_{i}",
                    name="TaskHead",
                    module_path=f"{head.value}_head.layer{i}",
                    task_head=head,
                    start_time=(30 + i) * 1000,
                    duration=1000,
                    memory_allocated=0,
                    memory_freed=0,
                )
                node.cuda_memory_allocated = 8 * 1024 * 1024
                node.parameters = 100000
                nodes.append(node)
        
        return nodes
    
    def test_multi_analyzer_integration(self):
        """Test multiple analyzers on the same trace."""
        # Run all analyzers
        multi_head = MultiHeadAnalyzer()
        temporal = TemporalAnalyzer(num_frames=3)
        bev = BEVAnalyzer()
        memory = MemoryProfiler(target_memory_gb=40.0)
        dtype = DTypeAnalyzer()
        
        # Analyze with each
        multi_head_analysis = multi_head.analyze_head_interactions(self.trace_nodes)
        temporal_analysis = temporal.analyze_temporal_flow(self.trace_nodes)
        bev_analysis = bev.analyze_bev_encoder(self.trace_nodes)
        memory_analysis = memory.profile_memory(self.trace_nodes)
        dtype_analysis = dtype.analyze_dtypes(self.trace_nodes)
        
        # Verify all analyses completed
        self.assertIsNotNone(multi_head_analysis)
        self.assertIsNotNone(temporal_analysis)
        self.assertIsNotNone(bev_analysis)
        self.assertIsNotNone(memory_analysis)
        self.assertIsNotNone(dtype_analysis)
        
        # Check for expected keys
        self.assertIn('shared_operation_count', multi_head_analysis)
        self.assertIn('temporal_efficiency', temporal_analysis)
        self.assertIn('bev_grid_config', bev_analysis)
        self.assertIn('peak_analysis', memory_analysis)
        self.assertIn('dtype_distribution', dtype_analysis)
    
    def test_export_functionality(self):
        """Test export functionality of all analyzers."""
        analyzers = [
            MultiHeadAnalyzer(),
            TemporalAnalyzer(),
            BEVAnalyzer(),
            MemoryProfiler(),
            DTypeAnalyzer(),
        ]
        
        for analyzer in analyzers:
            # Run analysis if needed
            if isinstance(analyzer, MultiHeadAnalyzer):
                analyzer.analyze_head_interactions(self.trace_nodes)
            elif isinstance(analyzer, TemporalAnalyzer):
                analyzer.analyze_temporal_flow(self.trace_nodes)
            elif isinstance(analyzer, BEVAnalyzer):
                analyzer.analyze_bev_encoder(self.trace_nodes)
            elif isinstance(analyzer, MemoryProfiler):
                analyzer.profile_memory(self.trace_nodes)
            elif isinstance(analyzer, DTypeAnalyzer):
                analyzer.analyze_dtypes(self.trace_nodes)
            
            # Test export
            export_data = analyzer.export_analysis()
            self.assertIsInstance(export_data, dict)
            self.assertGreater(len(export_data), 0)


if __name__ == '__main__':
    unittest.main()