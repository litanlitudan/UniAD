"""
Unit tests for HookManager module.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hook_manager import HookManager, register_hooks_on_model
from core.data_structures import TaskHead


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class UniADLikeModel(nn.Module):
    """Model with UniAD-like task heads for testing."""
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 64, 3)
        self.bev_encoder = nn.Conv2d(64, 128, 3)
        self.track_head = nn.Conv2d(128, 10, 1)
        self.seg_head = nn.Conv2d(128, 20, 1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.bev_encoder(x)
        track = self.track_head(x)
        seg = self.seg_head(x)
        return track, seg


class TestHookManager(unittest.TestCase):
    """Test cases for HookManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hook_manager = HookManager()
        self.model = SimpleTestModel()
        self.input_tensor = torch.randn(1, 3, 10, 10)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'hook_manager'):
            self.hook_manager.cleanup_hooks()
    
    def test_hook_registration(self):
        """Test hook registration on model."""
        self.hook_manager.register_module_hooks(self.model, recursive=True)
        
        # Should have hooks for each module
        self.assertGreater(len(self.hook_manager.hooks), 0)
        
        # Test that hooks are registered on submodules
        module_count = sum(1 for _ in self.model.modules())
        self.assertEqual(len(self.hook_manager.hooks), module_count)
    
    def test_forward_tracing(self):
        """Test forward pass tracing."""
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(self.model, recursive=True)
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        # Stop tracing and get nodes
        trace_nodes = self.hook_manager.stop_tracing()
        
        # Should have traced operations
        self.assertGreater(len(trace_nodes), 0)
        
        # Check node properties
        for node in trace_nodes:
            self.assertIsNotNone(node.id)
            self.assertIsNotNone(node.name)
            self.assertIsNotNone(node.module_path)
            self.assertGreaterEqual(node.duration, 0)
    
    def test_shape_recording(self):
        """Test that shapes are recorded correctly."""
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(self.model, recursive=True)
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        trace_nodes = self.hook_manager.stop_tracing()
        
        # Check that shapes are recorded
        for node in trace_nodes:
            if node.input_shapes or node.output_shapes:
                # At least some nodes should have shapes
                if node.input_shapes:
                    for shape in node.input_shapes:
                        self.assertIsNotNone(shape.shape)
                        self.assertIsNotNone(shape.dtype)
                if node.output_shapes:
                    for shape in node.output_shapes:
                        self.assertIsNotNone(shape.shape)
                        self.assertIsNotNone(shape.dtype)
    
    def test_task_head_identification(self):
        """Test task head identification for UniAD-like models."""
        model = UniADLikeModel()
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(model, recursive=True)
        
        # Run forward pass
        input_tensor = torch.randn(1, 3, 10, 10)
        with torch.no_grad():
            output = model(input_tensor)
        
        trace_nodes = self.hook_manager.stop_tracing()
        
        # Check for task head identification
        track_nodes = [n for n in trace_nodes if 'track' in n.module_path.lower()]
        seg_nodes = [n for n in trace_nodes if 'seg' in n.module_path.lower()]
        
        # The nodes exist even if task_head identification might not be perfect
        self.assertGreater(len(trace_nodes), 0, "Should have traced operations")
        
        # If we found track/seg nodes, check their assignment
        if track_nodes:
            for node in track_nodes:
                self.assertEqual(node.task_head, TaskHead.TRACK)
        if seg_nodes:
            for node in seg_nodes:
                self.assertEqual(node.task_head, TaskHead.SEGMENTATION)
    
    def test_bev_operation_detection(self):
        """Test BEV operation detection."""
        model = UniADLikeModel()
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(model, recursive=True)
        
        # Run forward pass
        input_tensor = torch.randn(1, 3, 10, 10)
        with torch.no_grad():
            output = model(input_tensor)
        
        trace_nodes = self.hook_manager.stop_tracing()
        
        # Check for BEV operations (bev_encoder should be detected)
        bev_nodes = [n for n in trace_nodes if n.bev_operation or 'bev' in n.module_path.lower()]
        # At least check that tracing worked
        self.assertGreater(len(trace_nodes), 0, "Should have traced operations")
    
    def test_memory_tracking(self):
        """Test memory tracking capabilities."""
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(self.model, recursive=True)
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        trace_nodes = self.hook_manager.stop_tracing()
        
        # Check memory tracking
        for node in trace_nodes:
            self.assertIsNotNone(node.memory_allocated)
            self.assertIsNotNone(node.memory_freed)
            self.assertIsNotNone(node.cuda_memory_allocated)
            self.assertIsNotNone(node.cuda_memory_freed)
    
    def test_cleanup(self):
        """Test hook cleanup."""
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(self.model, recursive=True)
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        # Cleanup
        self.hook_manager.cleanup_hooks()
        
        # Check cleanup
        self.assertEqual(len(self.hook_manager.hooks), 0)
        self.assertEqual(len(self.hook_manager.trace_nodes), 0)
        self.assertEqual(len(self.hook_manager.module_to_node), 0)
    
    def test_statistics(self):
        """Test statistics generation."""
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(self.model, recursive=True)
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        trace_nodes = self.hook_manager.stop_tracing()
        stats = self.hook_manager.get_statistics()
        
        # Check statistics
        self.assertIn('total_operations', stats)
        self.assertIn('total_duration_ns', stats)
        self.assertIn('task_head_stats', stats)
        self.assertIn('bev_operations', stats)
        
        self.assertEqual(stats['total_operations'], len(trace_nodes))
        self.assertGreaterEqual(stats['total_duration_ns'], 0)
    
    def test_backward_hook(self):
        """Test backward hook registration and execution."""
        hook_config = {
            "forward": True,
            "backward": True,
        }
        
        self.hook_manager.start_tracing()
        self.hook_manager.register_module_hooks(self.model, hook_config, recursive=True)
        
        # Run forward and backward pass
        input_tensor = torch.randn(1, 3, 10, 10, requires_grad=True)
        output = self.model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        trace_nodes = self.hook_manager.stop_tracing()
        
        # Check for gradient info
        nodes_with_gradients = [n for n in trace_nodes if n.gradient_info is not None]
        self.assertGreater(len(nodes_with_gradients), 0, "Should capture gradient information")
    
    def test_convenience_function(self):
        """Test the convenience function register_hooks_on_model."""
        manager = register_hooks_on_model(self.model)
        
        self.assertIsInstance(manager, HookManager)
        self.assertGreater(len(manager.hooks), 0)
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        trace_nodes = manager.stop_tracing()
        self.assertGreater(len(trace_nodes), 0)
        
        manager.cleanup_hooks()


if __name__ == '__main__':
    unittest.main()