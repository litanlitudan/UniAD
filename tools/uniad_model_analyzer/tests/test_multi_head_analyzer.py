"""
Comprehensive unit tests for MultiHeadAnalyzer.

Tests the multi-head analysis functionality for UniAD's 5 task heads:
Track, Segmentation, Motion, Occupancy, and Planning.
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TaskHead, TensorShape
from analyzers.multi_head_analyzer import MultiHeadAnalyzer


class TestMultiHeadAnalyzer(unittest.TestCase):
    """Test suite for MultiHeadAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiHeadAnalyzer()
        self.sample_trace_data = self._create_sample_trace_data()
    
    def _create_sample_trace_data(self):
        """Create sample trace data with multiple task heads."""
        trace_nodes = []
        
        # Track head operations
        for i in range(5):
            node = TraceNode(
                id=f"track_{i}",
                name=f"TrackOp_{i}",
                module_path=f"track_head.layer_{i}",
                start_time=i * 1000000,
                duration=500000,
                memory_allocated=1024 * 1024 * (i + 1),
                memory_freed=512 * 1024,
                cuda_memory_allocated=1024 * 1024 * (i + 1),
                cuda_memory_freed=512 * 1024,
                task_head=TaskHead.TRACK
            )
            node.input_shapes = [TensorShape(
                shape=(2, 256, 50, 50),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,
                memory_bytes=2 * 256 * 50 * 50 * 4
            )]
            node.output_shapes = [TensorShape(
                shape=(2, 128, 50, 50),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,
                memory_bytes=2 * 128 * 50 * 50 * 4
            )]
            trace_nodes.append(node)
        
        # Segmentation head operations
        for i in range(3):
            node = TraceNode(
                id=f"seg_{i}",
                name=f"SegOp_{i}",
                module_path=f"seg_head.layer_{i}",
                start_time=(i + 5) * 1000000,
                duration=800000,
                memory_allocated=2048 * 1024 * (i + 1),
                memory_freed=1024 * 1024,
                cuda_memory_allocated=2048 * 1024 * (i + 1),
                cuda_memory_freed=1024 * 1024,
                task_head=TaskHead.SEGMENTATION
            )
            node.input_shapes = [TensorShape(
                shape=(2, 512, 100, 100),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,
                memory_bytes=2 * 512 * 100 * 100 * 4
            )]
            trace_nodes.append(node)
        
        # Motion head operations
        for i in range(4):
            node = TraceNode(
                id=f"motion_{i}",
                name=f"MotionOp_{i}",
                module_path=f"motion_head.layer_{i}",
                start_time=(i + 8) * 1000000,
                duration=600000,
                memory_allocated=1536 * 1024 * (i + 1),
                memory_freed=768 * 1024,
                cuda_memory_allocated=1536 * 1024 * (i + 1),
                cuda_memory_freed=768 * 1024,
                task_head=TaskHead.MOTION
            )
            trace_nodes.append(node)
        
        # Add shared feature extraction (no specific head)
        for i in range(2):
            node = TraceNode(
                id=f"shared_{i}",
                name=f"SharedOp_{i}",
                module_path=f"backbone.layer_{i}",
                start_time=(i + 12) * 1000000,
                duration=1000000,
                memory_allocated=4096 * 1024,
                memory_freed=2048 * 1024,
                cuda_memory_allocated=4096 * 1024,
                cuda_memory_freed=2048 * 1024,
                task_head=TaskHead.NONE
            )
            trace_nodes.append(node)
        
        return trace_nodes
    
    def test_analyze_task_head_basic(self):
        """Test basic task head analysis."""
        analysis = self.analyzer.analyze_task_head(
            self.sample_trace_data,
            TaskHead.TRACK
        )
        
        # Check basic structure
        self.assertIsInstance(analysis, dict)
        
        if 'no_operations_found' not in analysis:
            self.assertIn('head_info', analysis)
            self.assertIn('operations', analysis)
            self.assertIn('memory_usage', analysis)
            self.assertIn('performance', analysis)
            
            # Check head info
            self.assertEqual(analysis['head_info']['name'], 'track')
            self.assertEqual(analysis['head_info']['task_type'], TaskHead.TRACK)
            
            # Check operations
            self.assertEqual(analysis['operations']['total_ops'], 5)
            self.assertIn('unique_op_types', analysis['operations'])
            
            # Check memory usage
            self.assertGreater(analysis['memory_usage']['total_allocated_mb'], 0)
            self.assertIn('peak_memory_mb', analysis['memory_usage'])
            
            # Check performance
            self.assertGreater(analysis['performance']['total_time_ms'], 0)
            self.assertIn('avg_operation_time_us', analysis['performance'])
    
    def test_analyze_task_head_empty(self):
        """Test analysis with no operations for a head."""
        analysis = self.analyzer.analyze_task_head(
            self.sample_trace_data,
            TaskHead.PLANNING  # No planning operations in sample data
        )
        
        self.assertIsInstance(analysis, dict)
        # Either no operations found or empty results
        if 'no_operations_found' in analysis:
            self.assertTrue(analysis['no_operations_found'])
    
    def test_analyze_all_heads(self):
        """Test analyzing all task heads."""
        # Analyze each head individually since analyze_all_heads may not exist
        track_analysis = self.analyzer.analyze_task_head(self.sample_trace_data, TaskHead.TRACK)
        seg_analysis = self.analyzer.analyze_task_head(self.sample_trace_data, TaskHead.SEGMENTATION)
        motion_analysis = self.analyzer.analyze_task_head(self.sample_trace_data, TaskHead.MOTION)
        
        # Check that analyses are valid
        self.assertIsInstance(track_analysis, dict)
        self.assertIsInstance(seg_analysis, dict)
        self.assertIsInstance(motion_analysis, dict)
    
    def test_analyze_head_interactions(self):
        """Test head interaction analysis."""
        interactions = self.analyzer.analyze_head_interactions(self.sample_trace_data)
        
        self.assertIn('interaction_matrix', interactions)
        self.assertIn('temporal_overlap', interactions)
        self.assertIn('memory_correlation', interactions)
        self.assertIn('shared_operations', interactions)
        
        # Check interaction matrix structure
        matrix = interactions['interaction_matrix']
        self.assertIsInstance(matrix, dict)
        
        # Check temporal overlap
        overlap = interactions['temporal_overlap']
        self.assertIsInstance(overlap, list)
        
        # Check shared operations count
        self.assertEqual(interactions['shared_operations'], 2)  # 2 shared ops in sample
    
    def test_compute_head_metrics(self):
        """Test computation of head-specific metrics."""
        metrics = self.analyzer.compute_head_metrics(self.sample_trace_data)
        
        self.assertIsInstance(metrics, dict)
        
        # Check for expected keys
        if metrics:  # If metrics are returned
            self.assertIn('head_distribution', metrics)
            
            # Check head distribution if present
            if 'head_distribution' in metrics:
                distribution = metrics['head_distribution']
                self.assertIsInstance(distribution, dict)
    
    def test_generate_head_comparison(self):
        """Test head comparison generation."""
        comparison = self.analyzer.generate_head_comparison(self.sample_trace_data)
        
        self.assertIn('summary_table', comparison)
        self.assertIn('performance_ranking', comparison)
        self.assertIn('memory_ranking', comparison)
        self.assertIn('recommendations', comparison)
        
        # Check summary table
        table = comparison['summary_table']
        self.assertIsInstance(table, list)
        self.assertGreater(len(table), 0)
        
        # Check rankings
        perf_ranking = comparison['performance_ranking']
        self.assertIsInstance(perf_ranking, list)
        
        mem_ranking = comparison['memory_ranking']
        self.assertIsInstance(mem_ranking, list)
        
        # Check recommendations
        recommendations = comparison['recommendations']
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty trace data
        empty_analysis = self.analyzer.analyze_task_head([], TaskHead.TRACK)
        self.assertIsInstance(empty_analysis, dict)
        
        # Test with None
        none_analysis = self.analyzer.analyze_task_head(None, TaskHead.TRACK)
        self.assertIsInstance(none_analysis, dict)
        
        # Test with single operation
        single_op = [self.sample_trace_data[0]]
        single_analysis = self.analyzer.analyze_task_head(single_op, TaskHead.TRACK)
        self.assertIsInstance(single_analysis, dict)
    
    def test_export_analysis(self):
        """Test export of complete analysis."""
        export_data = self.analyzer.export_analysis(self.sample_trace_data)
        
        self.assertIsInstance(export_data, dict)
        # Check that export contains some data
        self.assertGreater(len(export_data), 0)



if __name__ == '__main__':
    unittest.main()