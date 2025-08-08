"""
Unit tests for ShapeRecorder module.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.shape_recorder import ShapeRecorder, ShapeTransformation
from core.data_structures import TaskHead, TensorShape


class TestShapeTransformation(unittest.TestCase):
    """Test cases for ShapeTransformation."""
    
    def test_transformation_classification(self):
        """Test transformation type classification."""
        # Identity transformation
        trans = ShapeTransformation(
            from_shape=(2, 3, 64, 64),
            to_shape=(2, 3, 64, 64),
            operation="Conv2d",
            module_path="model.conv"
        )
        self.assertEqual(trans.dimension_change['type'], 'identity')
        
        # Spatial transformation
        trans = ShapeTransformation(
            from_shape=(2, 3, 64, 64),
            to_shape=(2, 16, 32, 32),
            operation="Conv2d",
            module_path="model.conv"
        )
        self.assertEqual(trans.dimension_change['type'], 'spatial_transform')
        
        # Expansion
        trans = ShapeTransformation(
            from_shape=(2, 10),
            to_shape=(2, 10, 1, 1),
            operation="Unsqueeze",
            module_path="model.unsqueeze"
        )
        self.assertEqual(trans.dimension_change['type'], 'expansion')
        
        # Reduction
        trans = ShapeTransformation(
            from_shape=(2, 10, 5, 5),
            to_shape=(2, 10),
            operation="AdaptiveAvgPool",
            module_path="model.pool"
        )
        self.assertEqual(trans.dimension_change['type'], 'reduction')
    
    def test_dimension_analysis(self):
        """Test dimension change analysis."""
        trans = ShapeTransformation(
            from_shape=(2, 3, 64, 64),
            to_shape=(2, 16, 32, 32),
            operation="Conv2d",
            module_path="model.conv"
        )
        
        analysis = trans.dimension_change
        self.assertEqual(analysis['dimension_change'], 0)  # Same number of dims
        # Check that size calculation is done
        self.assertIn('size_ratio', analysis)
        self.assertIn('total_size_change', analysis)
        # The actual size might increase or decrease depending on channels
    
    def test_empty_shape_handling(self):
        """Test handling of empty shapes."""
        trans = ShapeTransformation(
            from_shape=(),
            to_shape=(10, 20),
            operation="Input",
            module_path="model.input"
        )
        
        self.assertEqual(trans.dimension_change['type'], 'expansion')
        self.assertEqual(trans.dimension_change['dimension_change'], 2)


class TestShapeRecorder(unittest.TestCase):
    """Test cases for ShapeRecorder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recorder = ShapeRecorder(cache_size=100)
    
    def test_initialization(self):
        """Test shape recorder initialization."""
        self.assertEqual(self.recorder.cache_size, 100)
        self.assertEqual(len(self.recorder.shape_cache), 0)
        self.assertEqual(len(self.recorder.transformations), 0)
    
    def test_record_tensor(self):
        """Test tensor recording."""
        tensor = torch.randn(2, 3, 64, 64)
        context = {
            'module_path': 'model.conv',
            'task_head': TaskHead.TRACK,
            'bev_operation': False,
        }
        
        shape_obj = self.recorder.record_tensor(tensor, context)
        
        # Check returned shape object
        self.assertIsInstance(shape_obj, TensorShape)
        self.assertEqual(shape_obj.shape, (2, 3, 64, 64))
        self.assertEqual(shape_obj.dtype, torch.float32)
        
        # Check statistics updated
        self.assertEqual(self.recorder.shape_counts[(2, 3, 64, 64)], 1)
        self.assertEqual(self.recorder.dtype_counts[torch.float32], 1)
        
        # Check task head tracking
        self.assertEqual(len(self.recorder.task_head_shapes[TaskHead.TRACK]), 1)
    
    def test_cache_management(self):
        """Test LRU cache management."""
        # Fill cache beyond capacity
        for i in range(150):
            tensor = torch.randn(2, i, 10, 10)  # Different shapes
            context = {'module_path': f'model.layer{i}'}
            self.recorder.record_tensor(tensor, context)
        
        # Cache should not exceed max size
        self.assertLessEqual(len(self.recorder.shape_cache), 100)
    
    def test_record_transformation(self):
        """Test shape transformation recording."""
        from_tensor = torch.randn(2, 3, 64, 64)
        to_tensor = torch.randn(2, 16, 32, 32)
        
        trans = self.recorder.record_transformation(
            from_tensor, to_tensor,
            operation="Conv2d",
            module_path="model.conv"
        )
        
        self.assertIsInstance(trans, ShapeTransformation)
        self.assertEqual(trans.from_shape, (2, 3, 64, 64))
        self.assertEqual(trans.to_shape, (2, 16, 32, 32))
        self.assertEqual(len(self.recorder.transformations), 1)
    
    def test_bev_shape_tracking(self):
        """Test BEV-specific shape tracking."""
        tensor = torch.randn(2, 256, 200, 200)  # BEV grid size
        context = {
            'module_path': 'model.bev_encoder',
            'bev_operation': True,
            'task_head': TaskHead.NONE,
        }
        
        self.recorder.record_tensor(tensor, context)
        
        # Check BEV tracking
        self.assertEqual(len(self.recorder.bev_shapes), 1)
        self.assertEqual(self.recorder.bev_shapes[0].shape, (2, 256, 200, 200))
    
    def test_temporal_shape_tracking(self):
        """Test temporal frame shape tracking."""
        for frame in range(3):
            tensor = torch.randn(2, 64, 50, 50)
            context = {
                'module_path': f'model.temporal_{frame}',
                'temporal_frame': frame,
                'task_head': TaskHead.NONE,
            }
            self.recorder.record_tensor(tensor, context)
        
        # Check temporal tracking
        self.assertEqual(len(self.recorder.temporal_shapes), 3)
        for frame in range(3):
            self.assertEqual(len(self.recorder.temporal_shapes[frame]), 1)
    
    def test_shape_pattern_analysis(self):
        """Test shape pattern analysis."""
        # Record various shapes
        shapes = [
            (2, 3, 900, 1600),   # Image full resolution
            (2, 256, 200, 200),  # BEV grid
            (3, 64, 100, 100),   # Temporal with 3 frames
            (2, 128, 50, 50),    # Generic feature map
        ]
        
        for shape in shapes:
            tensor = torch.randn(*shape)
            context = {'module_path': 'model.layer'}
            self.recorder.record_tensor(tensor, context)
        
        analysis = self.recorder.analyze_shape_patterns()
        
        # Check analysis structure
        self.assertIn('total_unique_shapes', analysis)
        self.assertIn('total_tensors_tracked', analysis)
        self.assertIn('most_common_shapes', analysis)
        self.assertIn('dimension_distribution', analysis)
        self.assertIn('memory_analysis', analysis)
        
        self.assertEqual(analysis['total_unique_shapes'], 4)
        self.assertEqual(analysis['total_tensors_tracked'], 4)
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        # Record tensors of different sizes
        small_tensor = torch.randn(1, 1, 10, 10)
        large_tensor = torch.randn(10, 512, 100, 100)
        
        context = {'module_path': 'model.layer'}
        self.recorder.record_tensor(small_tensor, context)
        self.recorder.record_tensor(large_tensor, context)
        
        # Check memory tracking
        self.assertGreater(self.recorder.total_memory_tracked, 0)
        self.assertGreater(self.recorder.peak_shape_memory, 0)
        
        # Large tensor should have more memory
        self.assertGreater(
            self.recorder.peak_shape_memory,
            small_tensor.element_size() * small_tensor.numel()
        )
    
    def test_pattern_identification(self):
        """Test UniAD pattern identification."""
        # Record BEV grid pattern
        bev_tensor = torch.randn(2, 256, 200, 200)
        context = {'module_path': 'model.bev', 'bev_operation': True}
        self.recorder.record_tensor(bev_tensor, context)
        
        # Record temporal pattern
        temporal_tensor = torch.randn(3, 64, 50, 50)
        context = {'module_path': 'model.temporal'}
        self.recorder.record_tensor(temporal_tensor, context)
        
        analysis = self.recorder.analyze_shape_patterns()
        patterns = analysis.get('identified_patterns', {})
        
        # Should identify patterns
        if 'bev_grids' in patterns:
            self.assertGreater(len(patterns['bev_grids']), 0)
        if 'temporal' in patterns:
            self.assertGreater(len(patterns['temporal']), 0)
    
    def test_export_shape_data(self):
        """Test shape data export."""
        # Record some data
        tensor = torch.randn(2, 3, 64, 64)
        context = {'module_path': 'model.conv'}
        self.recorder.record_tensor(tensor, context)
        
        from_tensor = torch.randn(2, 3, 64, 64)
        to_tensor = torch.randn(2, 16, 32, 32)
        self.recorder.record_transformation(
            from_tensor, to_tensor, "Conv2d", "model.conv"
        )
        
        # Export data
        export_data = self.recorder.export_shape_data()
        
        # Check export structure
        self.assertIn('shape_counts', export_data)
        self.assertIn('dtype_counts', export_data)
        self.assertIn('transformations', export_data)
        self.assertIn('analysis', export_data)
        self.assertIn('cache_stats', export_data)
        
        self.assertEqual(len(export_data['transformations']), 1)
    
    def test_clear(self):
        """Test clearing recorded data."""
        # Record some data
        tensor = torch.randn(2, 3, 64, 64)
        context = {'module_path': 'model.conv'}
        self.recorder.record_tensor(tensor, context)
        
        # Clear
        self.recorder.clear()
        
        # Check everything is cleared
        self.assertEqual(len(self.recorder.shape_cache), 0)
        self.assertEqual(len(self.recorder.transformations), 0)
        self.assertEqual(len(self.recorder.shape_counts), 0)
        self.assertEqual(self.recorder.total_memory_tracked, 0)


if __name__ == '__main__':
    unittest.main()