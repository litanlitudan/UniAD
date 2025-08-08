#!/usr/bin/env python3
"""
Test the integrated tracer with shape recorder and hook manager.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import AnalysisConfig, AnalysisStage, TaskHead
from core.tracer import OperationTracer


class TestModel(nn.Module):
    """Simple test model simulating UniAD structure."""
    
    def __init__(self):
        super().__init__()
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # BEV encoder (simplified)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Task heads
        self.track_head = nn.Conv2d(128, 10, 1)
        self.seg_head = nn.Conv2d(128, 20, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        bev_features = self.bev_encoder(features)
        
        track_out = self.track_head(bev_features)
        seg_out = self.seg_head(bev_features)
        
        return {
            'track': track_out,
            'seg': seg_out,
        }


def test_tracer():
    """Test the integrated operation tracer."""
    print("=" * 70)
    print("Testing Integrated Operation Tracer")
    print("=" * 70)
    
    # Create configuration
    config = AnalysisConfig(
        model_config="test_model.py",
        checkpoint=None,
        stage=AnalysisStage.STAGE_1,
        trace_forward=True,
        trace_backward=False,
        profile_memory=True,
        track_shapes=True,
        track_dtypes=True,
        analyze_bev=True,
        temporal_frames=3,
        num_iterations=1,
        warmup_iterations=0,
    )
    
    print("\n1. Configuration created")
    print(f"   - Stage: {config.stage.value}")
    print(f"   - Memory profiling: {config.profile_memory}")
    print(f"   - Shape tracking: {config.track_shapes}")
    
    # Create model
    model = TestModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n2. Model created with {param_count} parameters")
    
    # Create tracer
    tracer = OperationTracer(config)
    print("\n3. Tracer initialized")
    
    # Create input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 64, 64)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    print(f"\n4. Input shape: {input_tensor.shape}")
    
    # Run tracing
    print("\n5. Starting trace...")
    result = tracer.trace_model(model, input_tensor)
    
    # Analyze results
    print("\n6. Analysis Results:")
    print(f"   - Total operations: {result.total_operations}")
    print(f"   - Total duration: {result.total_duration_ms:.2f} ms")
    print(f"   - Total parameters: {result.total_parameters}")
    
    # Memory profile
    print(f"\n   Memory Profile:")
    print(f"   - Peak allocated: {result.memory_profile.peak_allocated / (1024*1024):.2f} MB")
    print(f"   - Peak reserved: {result.memory_profile.peak_reserved / (1024*1024):.2f} MB")
    
    # Task head stats
    if result.task_head_stats:
        print(f"\n   Task Head Statistics:")
        for task_head, stats in result.task_head_stats.items():
            if isinstance(task_head, TaskHead):
                print(f"   - {task_head.value}:")
            else:
                print(f"   - {task_head}:")
            print(f"     * Operations: {stats.get('operation_count', 0)}")
            print(f"     * Duration: {stats.get('total_duration_ms', 0):.3f} ms")
            print(f"     * Memory: {stats.get('memory_mb', 0):.2f} MB")
    
    # BEV stats
    if result.bev_stats:
        print(f"\n   BEV Statistics:")
        print(f"   - Operations: {result.bev_stats.get('operation_count', 0)}")
        print(f"   - Duration: {result.bev_stats.get('total_duration_ms', 0):.3f} ms")
        print(f"   - Memory: {result.bev_stats.get('memory_mb', 0):.2f} MB")
    
    # Shape analysis
    shape_analysis = tracer.shape_recorder.analyze_shape_patterns()
    print(f"\n   Shape Analysis:")
    print(f"   - Unique shapes: {shape_analysis['total_unique_shapes']}")
    print(f"   - Total tensors tracked: {shape_analysis['total_tensors_tracked']}")
    
    if shape_analysis.get('most_common_shapes'):
        print(f"   - Most common shapes:")
        for shape, count in shape_analysis['most_common_shapes'][:3]:
            print(f"     * {shape}: {count} occurrences")
    
    # Transformations
    transformations = tracer.shape_recorder.get_shape_transformations()
    if transformations:
        print(f"\n   Shape Transformations: {len(transformations)} recorded")
        for i, trans in enumerate(transformations[:3], 1):
            print(f"   {i}. {trans.from_shape} → {trans.to_shape} ({trans.operation})")
            print(f"      Type: {trans.dimension_change.get('type', 'unknown')}")
    
    # Recommendations
    if result.recommendations:
        print(f"\n   Recommendations:")
        for rec in result.recommendations:
            print(f"   - {rec}")
    
    # Warnings
    if result.warnings:
        print(f"\n   Warnings:")
        for warn in result.warnings:
            print(f"   ⚠ {warn}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = test_tracer()
    print(f"\nTracer integration test {'PASSED' if result else 'FAILED'}")