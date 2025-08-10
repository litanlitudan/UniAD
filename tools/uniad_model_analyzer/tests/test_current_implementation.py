#!/usr/bin/env python3
"""
Test script for UniAD Model Analyzer current implementation.
Tests data structures, hook manager, and basic functionality.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        # Test core imports
        from core.data_structures import (
            TaskHead,
            AnalysisStage,
            TensorShape,
            TraceNode,
            MemoryProfile,
            AnalysisConfig,
            AnalysisResult,
        )
        print("✓ Data structures imported successfully")
        
        from core.hook_manager import (
            HookManager,
            register_hooks_on_model,
        )
        print("✓ Hook manager imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_structures():
    """Test data structure functionality."""
    print("\nTesting data structures...")
    
    from core.data_structures import (
        TaskHead,
        AnalysisStage,
        TensorShape,
        TraceNode,
        MemoryProfile,
        AnalysisConfig,
    )
    
    # Test TaskHead enum
    assert TaskHead.TRACK.value == "track"
    assert TaskHead.PLANNING.value == "planning"
    print("✓ TaskHead enum works")
    
    # Test TensorShape
    test_tensor = torch.randn(2, 3, 224, 224)
    tensor_shape = TensorShape.from_tensor(test_tensor)
    assert tensor_shape.shape == (2, 3, 224, 224)
    assert tensor_shape.dtype == torch.float32
    assert tensor_shape.memory_bytes == test_tensor.element_size() * test_tensor.numel()
    print(f"✓ TensorShape works: {tensor_shape}")
    
    # Test TraceNode
    node = TraceNode(
        id="test_node",
        name="TestOp",
        module_path="model.test",
        start_time=1000.0,
        duration=500.0,
        memory_allocated=1024,
        memory_freed=512,
        task_head=TaskHead.TRACK,
        bev_operation=True,
    )
    assert node.get_memory_delta() == 512
    assert node.task_head == TaskHead.TRACK
    print("✓ TraceNode works")
    
    # Test MemoryProfile
    mem_profile = MemoryProfile(
        peak_allocated=1024 * 1024 * 1024,  # 1GB
        peak_reserved=2 * 1024 * 1024 * 1024,  # 2GB
    )
    mem_profile.task_head_memory[TaskHead.TRACK] = 500 * 1024 * 1024
    mem_profile.task_head_memory[TaskHead.PLANNING] = 300 * 1024 * 1024
    distribution = mem_profile.get_task_head_distribution()
    assert TaskHead.TRACK in distribution
    print(f"✓ MemoryProfile works: {distribution}")
    
    # Test AnalysisConfig
    config = AnalysisConfig(
        model_config="test_config.py",
        checkpoint="test_checkpoint.pth",
        stage=AnalysisStage.STAGE_1,
        temporal_frames=3,
    )
    config.validate()  # Should not raise
    print("✓ AnalysisConfig validation works")
    
    # Test invalid config
    try:
        bad_config = AnalysisConfig(
            model_config="test.py",
            temporal_frames=4,  # Invalid: must be 3 or 5
        )
        bad_config.validate()
        print("✗ Config validation should have failed")
        return False
    except ValueError as e:
        print(f"✓ Config validation catches errors: {e}")
    
    return True


def test_hook_manager():
    """Test hook manager functionality."""
    print("\nTesting hook manager...")
    
    from core.hook_manager import HookManager
    from core.data_structures import TaskHead
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3)
            # Simulate UniAD task heads
            self.track_head = nn.Linear(128, 10)
            self.seg_head = nn.Linear(128, 20)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            # Flatten for heads
            x_flat = x.mean(dim=[2, 3])
            track_out = self.track_head(x_flat)
            seg_out = self.seg_head(x_flat)
            return track_out, seg_out
    
    model = SimpleModel()
    hook_manager = HookManager()
    
    # Test hook registration
    hook_manager.start_tracing()
    hook_manager.register_module_hooks(model, recursive=True)
    print(f"✓ Registered hooks on model with {len(hook_manager.hooks)} hooks")
    
    # Test forward pass tracing
    input_tensor = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(input_tensor)
    
    # Stop tracing and get nodes
    trace_nodes = hook_manager.stop_tracing()
    print(f"✓ Traced {len(trace_nodes)} operations")
    
    # Verify trace nodes
    if trace_nodes:
        first_node = trace_nodes[0]
        print(f"  First operation: {first_node.name} at {first_node.module_path}")
        
        # Check task head identification
        track_nodes = [n for n in trace_nodes if n.task_head == TaskHead.TRACK]
        seg_nodes = [n for n in trace_nodes if n.task_head == TaskHead.SEGMENTATION]
        print(f"  Found {len(track_nodes)} track head operations")
        print(f"  Found {len(seg_nodes)} segmentation head operations")
    
    # Test statistics
    stats = hook_manager.get_statistics()
    print(f"✓ Statistics: {stats['total_operations']} ops, "
          f"{stats.get('bev_operations', 0)} BEV ops")
    
    # Test cleanup
    hook_manager.cleanup_hooks()
    assert len(hook_manager.hooks) == 0
    assert len(hook_manager.trace_nodes) == 0
    print("✓ Cleanup successful")
    
    return True


def test_hook_manager_with_gradients():
    """Test hook manager with gradient tracking."""
    print("\nTesting hook manager with gradients...")
    
    from core.hook_manager import HookManager
    
    # Simple model for gradient testing
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    hook_manager = HookManager()
    hook_config = {
        "forward": True,
        "backward": True,  # Enable backward hooks
    }
    
    hook_manager.start_tracing()
    hook_manager.register_module_hooks(model, hook_config, recursive=True)
    
    # Forward and backward pass
    input_tensor = torch.randn(5, 10, requires_grad=True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    trace_nodes = hook_manager.stop_tracing()
    
    # Check for gradient info
    nodes_with_gradients = [n for n in trace_nodes if n.gradient_info is not None]
    print(f"✓ Captured gradient info for {len(nodes_with_gradients)} operations")
    
    hook_manager.cleanup_hooks()
    return True


def test_memory_tracking():
    """Test memory tracking capabilities."""
    print("\nTesting memory tracking...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping GPU memory tests")
        return True
    
    from core.hook_manager import HookManager
    
    # Create model and move to GPU
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
    ).cuda()
    
    hook_manager = HookManager()
    hook_manager.start_tracing()
    hook_manager.register_module_hooks(model, recursive=True)
    
    # Run forward pass
    input_tensor = torch.randn(10, 1000).cuda()
    with torch.no_grad():
        output = model(input_tensor)
    
    trace_nodes = hook_manager.stop_tracing()
    
    # Check memory tracking
    total_cuda_memory = sum(node.get_cuda_memory_delta() for node in trace_nodes)
    print(f"✓ Tracked CUDA memory changes: {total_cuda_memory / 1024:.2f} KB")
    
    hook_manager.cleanup_hooks()
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("UniAD Model Analyzer - Testing Current Implementation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("Hook Manager", test_hook_manager),
        ("Gradient Tracking", test_hook_manager_with_gradients),
        ("Memory Tracking", test_memory_tracking),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return all(success for _, success in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)