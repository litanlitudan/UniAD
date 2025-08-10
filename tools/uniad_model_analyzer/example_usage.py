#!/usr/bin/env python3
"""
Example usage of UniAD Model Analyzer showing current implementation.
"""

import torch
import torch.nn as nn
from core.data_structures import AnalysisConfig, AnalysisStage, TaskHead
from core.hook_manager import HookManager


class MiniUniAD(nn.Module):
    """Simplified UniAD-like model for demonstration."""

    def __init__(self):
        super().__init__()
        # Shared backbone (like image encoder)
        self.img_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # BEV encoder (simplified)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Task heads (5 heads like UniAD)
        self.track_head = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 10, 1),  # 10 tracking classes
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 20, 1),  # 20 segmentation classes
        )

        self.motion_head = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),  # 2D motion vectors
        )

        self.occ_head = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # Occupancy probability
        )

        self.planning_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # 6D trajectory
        )

    def forward(self, x):
        # Shared feature extraction
        features = self.img_backbone(x)

        # BEV encoding
        bev_features = self.bev_encoder(features)

        # Multi-task heads
        track_out = self.track_head(bev_features)
        seg_out = self.seg_head(bev_features)
        motion_out = self.motion_head(bev_features)
        occ_out = self.occ_head(bev_features)
        planning_out = self.planning_head(bev_features)

        return {
            'track': track_out,
            'seg': seg_out,
            'motion': motion_out,
            'occ': occ_out,
            'planning': planning_out,
        }


def analyze_model():
    """Demonstrate model analysis workflow."""

    print("=" * 70)
    print("UniAD Model Analyzer - Example Usage")
    print("=" * 70)

    # 1. Create configuration
    config = AnalysisConfig(
        model_config="mini_uniad_config.py",
        checkpoint=None,
        stage=AnalysisStage.STAGE_1,
        trace_forward=True,
        trace_backward=False,  # Disable backward tracing for now
        profile_memory=True,
        temporal_frames=3,
        task_heads=[TaskHead.TRACK, TaskHead.PLANNING],  # Analyze specific heads
        export_formats=["json", "markdown"],
    )

    print("\n1. Configuration created:")
    print(f"   - Stage: {config.stage.value}")
    print(f"   - Temporal frames: {config.temporal_frames}")
    print(f"   - Task heads to analyze: {[h.value for h in config.task_heads]}")

    # 2. Create model
    model = MiniUniAD()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"\n2. Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 3. Setup hook manager
    hook_manager = HookManager()
    hook_config = {
        "forward": config.trace_forward,
        "backward": config.trace_backward,
    }

    hook_manager.start_tracing()
    hook_manager.register_module_hooks(model, hook_config, recursive=True)

    print(f"\n3. Registered {len(hook_manager.hooks)} hooks on model")

    # 4. Run model with sample input
    batch_size = 2
    temporal_frames = config.temporal_frames
    input_shape = (batch_size * temporal_frames, 3, 224, 224)

    if torch.cuda.is_available():
        input_tensor = torch.randn(input_shape).cuda()
    else:
        input_tensor = torch.randn(input_shape)

    print(f"\n4. Running model with input shape: {input_shape}")

    # Forward pass
    if config.trace_backward:
        input_tensor.requires_grad = True
        outputs = model(input_tensor)
        # Simulate loss computation - ensure we get a tensor
        loss_terms = [out.mean() for out in outputs.values()]
        if loss_terms:
            loss = torch.stack(loss_terms).sum()
        else:
            # Fallback: create a dummy loss if no outputs
            loss = torch.tensor(0.0, requires_grad=True)
        print(f"   - Forward pass complete")
        try:
            loss.backward()
            print(f"   - Backward pass complete")
        except RuntimeError as e:
            print(f"   - Backward pass skipped (hook compatibility issue)")
    else:
        with torch.no_grad():
            outputs = model(input_tensor)
        print(f"   - Forward pass complete (no gradients)")

    # 5. Stop tracing and analyze
    trace_nodes = hook_manager.stop_tracing()
    stats = hook_manager.get_statistics()

    print(f"\n5. Analysis Results:")
    print(f"   - Total operations traced: {stats['total_operations']}")
    print(f"   - Total duration: {stats['total_duration_ns'] / 1e6:.2f} ms")
    print(f"   - BEV operations: {stats['bev_operations']}")

    # Task head breakdown
    print(f"\n   Task Head Statistics:")
    for task_head, head_stats in stats['task_head_stats'].items():
        if head_stats['count'] > 0:
            print(f"   - {task_head.value if hasattr(task_head, 'value') else task_head}:")
            print(f"     * Operations: {head_stats['count']}")
            print(f"     * Duration: {head_stats['duration'] / 1e6:.2f} ms")
            print(f"     * Memory: {head_stats['memory'] / 1024:.2f} KB")

    # 6. Analyze trace nodes
    print(f"\n6. Detailed Operation Analysis:")

    # Find most time-consuming operations
    sorted_nodes = sorted(trace_nodes, key=lambda n: n.duration, reverse=True)[:5]
    print(f"\n   Top 5 Time-Consuming Operations:")
    for i, node in enumerate(sorted_nodes, 1):
        print(f"   {i}. {node.name} ({node.module_path})")
        print(f"      Duration: {node.duration / 1e6:.3f} ms")
        if node.output_shapes:
            print(f"      Output shape: {node.output_shapes[0].shape}")

    # Find BEV operations
    bev_nodes = [n for n in trace_nodes if n.bev_operation]
    if bev_nodes:
        print(f"\n   BEV Operations ({len(bev_nodes)} total):")
        for node in bev_nodes[:3]:
            print(f"   - {node.name} at {node.module_path}")

    # Find operations by task head
    for task_head in [TaskHead.TRACK, TaskHead.PLANNING]:
        head_nodes = [n for n in trace_nodes if n.task_head == task_head]
        if head_nodes:
            print(f"\n   {task_head.value.capitalize()} Head Operations ({len(head_nodes)} total):")
            for node in head_nodes[:3]:
                print(f"   - {node.name} at {node.module_path}")

    # 7. Memory analysis
    if torch.cuda.is_available():
        total_cuda_memory = sum(node.get_cuda_memory_delta() for node in trace_nodes)
        print(f"\n7. Memory Analysis:")
        print(f"   - Total CUDA memory delta: {total_cuda_memory / (1024*1024):.2f} MB")

        # Find memory-intensive operations
        memory_nodes = sorted(
            [n for n in trace_nodes if n.get_cuda_memory_delta() > 0],
            key=lambda n: n.get_cuda_memory_delta(),
            reverse=True
        )[:3]
        if memory_nodes:
            print(f"\n   Top Memory-Consuming Operations:")
            for node in memory_nodes:
                print(f"   - {node.name}: {node.get_cuda_memory_delta() / 1024:.2f} KB")

    # 8. Cleanup
    hook_manager.cleanup_hooks()
    print(f"\n8. Cleanup complete - hooks removed")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return trace_nodes, stats


if __name__ == "__main__":
    trace_nodes, stats = analyze_model()

    print(f"\nNext steps:")
    print("- Implement remaining analyzers for deeper insights")
    print("- Add visualization components for Mermaid diagrams")
    print("- Create export functionality for results")
    print("- Build CLI interface for easy usage")