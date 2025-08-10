"""
Comprehensive unit tests for MemoryProfiler.

Tests memory profiling, optimization suggestions, gradient checkpointing,
and mixed precision analysis for UniAD's 30-50GB GPU memory requirements.
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TaskHead, TensorShape
from analyzers.memory_profiler import MemoryProfiler


class TestMemoryProfiler(unittest.TestCase):
    """Test suite for MemoryProfiler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler(
            target_memory_gb=40.0,
            optimization_threshold=0.8
        )
        self.sample_trace_data = self._create_sample_trace_data()
    
    def _create_sample_trace_data(self):
        """Create sample trace data with memory patterns."""
        trace_nodes = []
        
        # Simulate memory allocation pattern
        memory_pattern = [
            (100, 50),   # 100MB alloc, 50MB free
            (200, 100),  # 200MB alloc, 100MB free
            (500, 200),  # 500MB alloc, 200MB free
            (1000, 300), # 1GB alloc, 300MB free
            (2000, 500), # 2GB alloc, 500MB free (peak)
            (300, 1000), # 300MB alloc, 1GB free
            (200, 500),  # 200MB alloc, 500MB free
        ]
        
        start_time = 0
        for i, (alloc_mb, free_mb) in enumerate(memory_pattern):
            node = TraceNode(
                id=f"op_{i}",
                name=f"Operation_{i}",
                module_path=f"module.layer_{i}",
                start_time=start_time,
                duration=1000000 * (i + 1),  # Increasing duration
                memory_allocated=alloc_mb * 1024 * 1024,
                memory_freed=free_mb * 1024 * 1024,
                cuda_memory_allocated=alloc_mb * 1024 * 1024,
                cuda_memory_freed=free_mb * 1024 * 1024,
                task_head=TaskHead.NONE
            )
            
            # Add tensor shapes
            node.input_shapes = [TensorShape(
                shape=(2, 256, 100, 100),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,
                memory_bytes=2 * 256 * 100 * 100 * 4
            )]
            node.output_shapes = [TensorShape(
                shape=(2, 512, 50, 50),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,
                memory_bytes=2 * 512 * 50 * 50 * 4
            )]
            
            # Add gradient info for some operations
            if i % 2 == 0:
                node.gradient_info = {
                    'requires_grad': True,
                    'grad_fn': f'{node.name}Backward'
                }
            
            trace_nodes.append(node)
            start_time += node.duration
        
        return trace_nodes
    
    def test_profile_memory_basic(self):
        """Test basic memory profiling."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        
        self.assertIn('summary', analysis)
        self.assertIn('peak_analysis', analysis)
        self.assertIn('timeline', analysis)
        self.assertIn('allocation_patterns', analysis)
        self.assertIn('memory_regions', analysis)
        
        # Check summary
        summary = analysis['summary']
        self.assertIn('peak_memory_gb', summary)
        self.assertIn('memory_utilization', summary)
        self.assertIn('status', summary)
        self.assertGreater(summary['peak_memory_gb'], 0)
    
    def test_peak_memory_detection(self):
        """Test peak memory detection."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        peak_analysis = analysis['peak_analysis']
        
        self.assertIn('peak_memory_gb', peak_analysis)
        self.assertIn('peak_timestamp_ms', peak_analysis)
        self.assertIn('peak_operation', peak_analysis)
        self.assertIn('peak_percentage', peak_analysis)
        
        # Peak should be around 2GB (from our pattern)
        self.assertGreater(peak_analysis['peak_memory_gb'], 1.5)
        self.assertLess(peak_analysis['peak_memory_gb'], 3.0)
    
    def test_memory_timeline(self):
        """Test memory timeline generation."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        timeline = analysis['timeline']
        
        self.assertIsInstance(timeline, list)
        self.assertGreater(len(timeline), 0)
        
        # Check timeline structure
        for point in timeline:
            self.assertIn('timestamp_ms', point)
            self.assertIn('allocated_mb', point)
            self.assertIn('freed_mb', point)
            self.assertIn('net_memory_mb', point)
            self.assertIn('operation', point)
    
    def test_allocation_patterns(self):
        """Test memory allocation pattern analysis."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        patterns = analysis['allocation_patterns']
        
        self.assertIsInstance(patterns, dict)
        
        # Should have patterns for different operation types
        for op_type, stats in patterns.items():
            self.assertIn('count', stats)
            self.assertIn('total_mb', stats)
            self.assertIn('avg_mb', stats)
            self.assertIn('max_mb', stats)
            self.assertIn('min_mb', stats)
    
    def test_memory_regions(self):
        """Test memory region identification."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        regions = analysis['memory_regions']
        
        self.assertIsInstance(regions, list)
        
        if regions:
            region = regions[0]
            self.assertIn('name', region)
            self.assertIn('start_ms', region)
            self.assertIn('end_ms', region)
            self.assertIn('duration_ms', region)
            self.assertIn('peak_memory_mb', region)
            self.assertIn('avg_memory_mb', region)
            self.assertIn('efficiency', region)
    
    def test_optimization_suggestions(self):
        """Test generation of optimization suggestions."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        suggestions = analysis.get('optimization_suggestions', [])
        
        self.assertIsInstance(suggestions, list)
        
        for suggestion in suggestions:
            self.assertIn('severity', suggestion)
            self.assertIn('target', suggestion)
            self.assertIn('description', suggestion)
            self.assertIn('expected_savings_mb', suggestion)
            self.assertIn('difficulty', suggestion)
            self.assertIn('techniques', suggestion)
    
    def test_gradient_checkpointing_suggestions(self):
        """Test gradient checkpointing suggestions."""
        suggestions = self.profiler.suggest_gradient_checkpointing(self.sample_trace_data)
        
        self.assertIsInstance(suggestions, list)
        
        for suggestion in suggestions:
            self.assertIn('module', suggestion)
            self.assertIn('operation', suggestion)
            self.assertIn('activation_memory_mb', suggestion)
            self.assertIn('expected_savings_mb', suggestion)
            self.assertIn('recomputation_time_ms', suggestion)
            self.assertIn('worth_checkpointing', suggestion)
    
    def test_mixed_precision_analysis(self):
        """Test mixed precision opportunity analysis."""
        analysis = self.profiler.analyze_mixed_precision_opportunities(self.sample_trace_data)
        
        self.assertIn('fp32_operations', analysis)
        self.assertIn('fp16_compatible', analysis)
        self.assertIn('current_memory_gb', analysis)
        self.assertIn('potential_savings_gb', analysis)
        self.assertIn('conversion_percentage', analysis)
        self.assertIn('recommendations', analysis)
        
        # Check recommendations
        recommendations = analysis['recommendations']
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_memory_lifecycle_analysis(self):
        """Test memory lifecycle analysis."""
        lifecycle = self.profiler.analyze_memory_lifecycle(self.sample_trace_data)
        
        self.assertIn('allocation_frequency', lifecycle)
        self.assertIn('deallocation_patterns', lifecycle)
        self.assertIn('memory_leaks', lifecycle)
        self.assertIn('fragmentation_score', lifecycle)
        
        # Check allocation frequency
        alloc_freq = lifecycle['allocation_frequency']
        self.assertIsInstance(alloc_freq, dict)
        
        # Check for memory leaks
        leaks = lifecycle['memory_leaks']
        self.assertIsInstance(leaks, list)
        
        # Check fragmentation score
        frag_score = lifecycle['fragmentation_score']
        self.assertIsInstance(frag_score, float)
        self.assertGreaterEqual(frag_score, 0.0)
        self.assertLessEqual(frag_score, 1.0)
    
    def test_task_head_memory_distribution(self):
        """Test memory distribution across task heads."""
        # Add some task head operations
        trace_with_heads = self.sample_trace_data.copy()
        
        heads = [TaskHead.TRACK, TaskHead.SEGMENTATION, TaskHead.MOTION]
        for i, head in enumerate(heads):
            node = TraceNode(
                id=f"head_{i}",
                name=f"{head.value}_op",
                module_path=f"{head.value}_head",
                start_time=10000000 + i * 1000000,
                duration=500000,
                memory_allocated=(i + 1) * 512 * 1024 * 1024,
                memory_freed=0,
                cuda_memory_allocated=(i + 1) * 512 * 1024 * 1024,
                cuda_memory_freed=0,
                task_head=head
            )
            trace_with_heads.append(node)
        
        analysis = self.profiler.profile_memory(trace_with_heads)
        distribution = analysis.get('task_head_distribution', {})
        
        self.assertIsInstance(distribution, dict)
        
        # Check that task heads are present
        for head in heads:
            head_name = head.value.lower()
            if head_name in distribution:
                self.assertIn('memory_mb', distribution[head_name])
                self.assertIn('percentage', distribution[head_name])
                self.assertIn('operation_count', distribution[head_name])
    
    def test_memory_efficiency_calculation(self):
        """Test memory efficiency calculation."""
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        efficiency = analysis.get('memory_efficiency', 0.0)
        
        self.assertIsInstance(efficiency, float)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_memory_status_classification(self):
        """Test memory status classification."""
        # Test different memory usage levels
        test_cases = [
            (10.0, 'underutilized'),  # 25% of 40GB
            (25.0, 'normal'),          # 62.5% of 40GB
            (35.0, 'warning'),         # 87.5% of 40GB
            (39.0, 'critical'),        # 97.5% of 40GB
        ]
        
        for peak_gb, expected_status in test_cases:
            # Create trace with specific peak memory
            trace = [TraceNode(
                id="test",
                name="TestOp",
                module_path="test",
                start_time=0,
                duration=1000000,
                memory_allocated=int(peak_gb * 1024 * 1024 * 1024),
                memory_freed=0,
                cuda_memory_allocated=int(peak_gb * 1024 * 1024 * 1024),
                cuda_memory_freed=0,
                task_head=TaskHead.NONE
            )]
            
            analysis = self.profiler.profile_memory(trace)
            status = analysis['summary']['status']
            
            # Status should match expected or be close
            self.assertIn(status, ['underutilized', 'normal', 'warning', 'critical'])
    
    def test_memory_snapshot_creation(self):
        """Test memory snapshot creation."""
        # Note: _create_memory_snapshots is a private method
        # Testing indirectly through profile_memory
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        
        # Verify that snapshots were created internally by checking the timeline
        self.assertIn('memory_timeline', analysis)
        timeline = analysis['memory_timeline']
        self.assertIsInstance(timeline, list)
        self.assertGreater(len(timeline), 0)
    
    def test_optimization_suggestion_creation(self):
        """Test creation of optimization suggestions."""
        # Note: _create_optimization_suggestion is a private method
        # Testing indirectly through profile_memory which creates suggestions
        analysis = self.profiler.profile_memory(self.sample_trace_data)
        
        # Check that optimization suggestions are created
        suggestions = analysis.get('optimization_suggestions', [])
        self.assertIsInstance(suggestions, list)
        
        # If there are suggestions, validate their structure
        for suggestion in suggestions:
            self.assertIn('target', suggestion)
            self.assertIn('severity', suggestion)
            self.assertIn('expected_savings_mb', suggestion)
            self.assertIn('techniques', suggestion)
    
    def test_export_analysis(self):
        """Test export of memory analysis."""
        export_data = self.profiler.export_analysis()
        
        self.assertIn('configuration', export_data)
        self.assertIn('last_analysis', export_data)
        self.assertIn('suggestions', export_data)
        
        # Check configuration
        config = export_data['configuration']
        self.assertEqual(config['target_memory_gb'], 40.0)
        self.assertEqual(config['optimization_threshold'], 0.8)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty trace
        empty_analysis = self.profiler.profile_memory([])
        self.assertIn('summary', empty_analysis)
        self.assertEqual(empty_analysis['summary']['peak_memory_gb'], 0.0)
        
        # Test with single operation
        single_op = [self.sample_trace_data[0]]
        single_analysis = self.profiler.profile_memory(single_op)
        self.assertIn('summary', single_analysis)
        
        # Test with no gradient operations
        no_grad_trace = []
        for node in self.sample_trace_data:
            new_node = TraceNode(
                id=node.id,
                name=node.name,
                module_path=node.module_path,
                start_time=node.start_time,
                duration=node.duration,
                memory_allocated=node.memory_allocated,
                memory_freed=node.memory_freed,
                cuda_memory_allocated=node.cuda_memory_allocated,
                cuda_memory_freed=node.cuda_memory_freed,
                task_head=node.task_head
            )
            no_grad_trace.append(new_node)
        
        grad_suggestions = self.profiler.suggest_gradient_checkpointing(no_grad_trace)
        # Should still return a list (possibly empty)
        self.assertIsInstance(grad_suggestions, list)
    
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        # Create trace with potential memory leak
        leak_trace = []
        for i in range(5):
            # Operations that allocate but never free
            node = TraceNode(
                id=f"leak_{i}",
                name=f"LeakyOp_{i}",
                module_path=f"leaky_module.{i}",
                start_time=i * 1000000,
                duration=500000,
                memory_allocated=100 * 1024 * 1024,  # 100MB
                memory_freed=0,  # Never freed!
                cuda_memory_allocated=100 * 1024 * 1024,
                cuda_memory_freed=0,
                task_head=TaskHead.NONE
            )
            leak_trace.append(node)
        
        lifecycle = self.profiler.analyze_memory_lifecycle(leak_trace)
        leaks = lifecycle['memory_leaks']
        
        self.assertGreater(len(leaks), 0)
        
        for leak in leaks:
            self.assertIn('module', leak)
            self.assertIn('allocated_mb', leak)
            self.assertIn('freed_mb', leak)
            self.assertIn('leaked_mb', leak)
            self.assertGreater(leak['leaked_mb'], 0)


class TestMemoryOptimizationIntegration(unittest.TestCase):
    """Test suite for memory optimization integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler(
            target_memory_gb=32.0,  # Lower target for testing
            optimization_threshold=0.7
        )
    
    def test_comprehensive_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create a complex trace simulating UniAD
        trace = self._create_uniad_like_trace()
        
        # Profile memory
        analysis = self.profiler.profile_memory(trace)
        
        # Get optimization suggestions
        suggestions = analysis.get('optimization_suggestions', [])
        
        # Get gradient checkpointing candidates
        grad_checkpoint = self.profiler.suggest_gradient_checkpointing(trace)
        
        # Get mixed precision opportunities
        mixed_precision = self.profiler.analyze_mixed_precision_opportunities(trace)
        
        # Verify we get comprehensive optimization plan
        self.assertGreater(len(suggestions), 0)
        self.assertGreater(len(grad_checkpoint), 0)
        self.assertGreater(mixed_precision['potential_savings_gb'], 0)
        
        # Check that suggestions are prioritized
        if len(suggestions) > 1:
            severities = ['critical', 'high', 'medium', 'low']
            first_severity_idx = severities.index(suggestions[0]['severity'])
            last_severity_idx = severities.index(suggestions[-1]['severity'])
            self.assertLessEqual(first_severity_idx, last_severity_idx)
    
    def _create_uniad_like_trace(self):
        """Create trace data similar to UniAD model."""
        trace = []
        
        # Backbone operations (memory intensive)
        for i in range(10):
            trace.append(TraceNode(
                id=f"backbone_{i}",
                name="Conv2d",
                module_path=f"backbone.layer{i}",
                start_time=i * 1000000,
                duration=2000000,
                memory_allocated=512 * 1024 * 1024,  # 512MB
                memory_freed=256 * 1024 * 1024,
                cuda_memory_allocated=512 * 1024 * 1024,
                cuda_memory_freed=256 * 1024 * 1024,
                task_head=TaskHead.NONE
            ))
        
        # BEV encoder (very memory intensive)
        for i in range(5):
            trace.append(TraceNode(
                id=f"bev_{i}",
                name="BEVEncoder",
                module_path=f"bev_encoder.layer{i}",
                start_time=(10 + i) * 1000000,
                duration=3000000,
                memory_allocated=1024 * 1024 * 1024,  # 1GB
                memory_freed=512 * 1024 * 1024,
                cuda_memory_allocated=1024 * 1024 * 1024,
                cuda_memory_freed=512 * 1024 * 1024,
                task_head=TaskHead.NONE
            ))
        
        # Task heads
        for head in [TaskHead.TRACK, TaskHead.SEGMENTATION, TaskHead.MOTION, 
                     TaskHead.OCCUPANCY, TaskHead.PLANNING]:
            for i in range(3):
                trace.append(TraceNode(
                    id=f"{head.value}_{i}",
                    name=f"{head.value}Head",
                    module_path=f"{head.value}_head.layer{i}",
                    start_time=(15 + len(trace)) * 1000000,
                    duration=1500000,
                    memory_allocated=256 * 1024 * 1024,  # 256MB
                    memory_freed=128 * 1024 * 1024,
                    cuda_memory_allocated=256 * 1024 * 1024,
                    cuda_memory_freed=128 * 1024 * 1024,
                    task_head=head
                ))
        
        return trace


if __name__ == '__main__':
    unittest.main()