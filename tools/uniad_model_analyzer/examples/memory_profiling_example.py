#!/usr/bin/env python3
"""
Example of using the Memory Profiler to analyze memory usage in UniAD.

This script demonstrates how to profile memory usage, identify bottlenecks,
and get optimization suggestions for the 30-50GB GPU memory requirements.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TensorShape, TaskHead
from analyzers.memory_profiler import MemoryProfiler, MemorySnapshot, OptimizationSuggestion


class MemoryIntensiveModel(nn.Module):
    """Model simulating UniAD's memory-intensive operations."""
    
    def __init__(self, num_frames=3, num_cameras=6):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        
        # Large backbone network
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Memory-intensive temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        
        # Large feature maps for BEV
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, 3, padding=1),  # Very large feature maps
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        
        # Multiple task heads (memory distributed across heads)
        self.task_heads = nn.ModuleDict({
            'track': nn.Conv2d(2048, 256, 1),
            'seg': nn.Conv2d(2048, 512, 1),
            'motion': nn.Conv2d(2048, 128, 1),
            'occ': nn.Conv2d(2048, 64, 1),
            'planning': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            ),
        })
    
    def forward(self, x):
        # Process through backbone
        features = self.backbone(x)
        
        # Temporal attention (memory intensive)
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H * W).transpose(1, 2)
        attended, _ = self.temporal_attention(features_flat, features_flat, features_flat)
        features = attended.transpose(1, 2).view(B, C, H, W)
        
        # BEV encoding (very memory intensive)
        bev_features = self.bev_encoder(features)
        
        # Multiple task heads
        outputs = {}
        for task, head in self.task_heads.items():
            outputs[task] = head(bev_features)
        
        return outputs


def create_memory_intensive_trace(num_frames=3, batch_size=2) -> List[TraceNode]:
    """Create trace data simulating memory-intensive UniAD operations."""
    trace_nodes = []
    node_id = 0
    start_time = 0
    
    # Initial memory state
    current_allocated = 0
    current_reserved = 0
    
    # Backbone operations (moderate memory)
    backbone_layers = [
        ('Conv2d', 100 * 1024 * 1024),  # 100MB
        ('BatchNorm2d', 50 * 1024 * 1024),  # 50MB
        ('ReLU', 0),  # In-place
        ('Conv2d', 200 * 1024 * 1024),  # 200MB
        ('BatchNorm2d', 100 * 1024 * 1024),  # 100MB
        ('ReLU', 0),  # In-place
        ('Conv2d', 400 * 1024 * 1024),  # 400MB
        ('BatchNorm2d', 200 * 1024 * 1024),  # 200MB
        ('ReLU', 0),  # In-place
    ]
    
    for layer_name, memory_alloc in backbone_layers:
        node = TraceNode(
            id=f"node_{node_id}",
            name=layer_name,
            module_path=f"backbone.{node_id}",
            start_time=start_time,
            duration=2000000,  # 2ms
            memory_allocated=memory_alloc,
            memory_freed=0,
            cuda_memory_allocated=memory_alloc,
            cuda_memory_freed=0,
            task_head=TaskHead.NONE,
        )
        
        current_allocated += memory_alloc
        current_reserved = max(current_reserved, current_allocated)
        
        # Add shape information for first conv
        if node_id == 0:
            node.input_shapes = [TensorShape(
                shape=(batch_size * num_frames, 3, 640, 480),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,  # Training mode
                memory_bytes=batch_size * num_frames * 3 * 640 * 480 * 4
            )]
            node.output_shapes = [TensorShape(
                shape=(batch_size * num_frames, 128, 320, 240),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=True,
                memory_bytes=batch_size * num_frames * 128 * 320 * 240 * 4
            )]
            node.gradient_info = {'requires_grad': True, 'grad_fn': 'ConvolutionBackward'}
        
        trace_nodes.append(node)
        node_id += 1
        start_time += 2000000
    
    # Temporal attention (very memory intensive)
    attention_memory = 500 * 1024 * 1024  # 500MB for attention matrices
    node = TraceNode(
        id=f"node_{node_id}",
        name="MultiheadAttention",
        module_path="temporal_attention",
        start_time=start_time,
        duration=10000000,  # 10ms - attention is slow
        memory_allocated=attention_memory,
        memory_freed=100 * 1024 * 1024,  # Free 100MB
        cuda_memory_allocated=attention_memory,
        cuda_memory_freed=100 * 1024 * 1024,
        task_head=TaskHead.NONE,
        temporal_frame=None,  # Processes all frames
    )
    node.gradient_info = {'requires_grad': True, 'grad_fn': 'AttentionBackward'}
    
    current_allocated += attention_memory - 100 * 1024 * 1024
    trace_nodes.append(node)
    node_id += 1
    start_time += 10000000
    
    # BEV encoder (extremely memory intensive)
    bev_layers = [
        ('Conv2d', 1024 * 1024 * 1024),  # 1GB!
        ('BatchNorm2d', 512 * 1024 * 1024),  # 512MB
        ('ReLU', 0),  # In-place
        ('Conv2d', 2048 * 1024 * 1024),  # 2GB!! Peak memory
        ('BatchNorm2d', 1024 * 1024 * 1024),  # 1GB
        ('ReLU', 0),  # In-place
    ]
    
    for layer_idx, (layer_name, memory_alloc) in enumerate(bev_layers):
        node = TraceNode(
            id=f"node_{node_id}",
            name=layer_name,
            module_path=f"bev_encoder.{layer_idx}",
            start_time=start_time,
            duration=5000000,  # 5ms
            memory_allocated=memory_alloc,
            memory_freed=0,
            cuda_memory_allocated=memory_alloc,
            cuda_memory_freed=0,
            task_head=TaskHead.NONE,
            bev_operation=True,
        )
        node.gradient_info = {'requires_grad': True, 'grad_fn': f'{layer_name}Backward'}
        
        current_allocated += memory_alloc
        current_reserved = max(current_reserved, current_allocated)
        
        # Mark peak memory operation
        if memory_alloc == 2048 * 1024 * 1024:
            node.module_path += "_PEAK_MEMORY"
        
        trace_nodes.append(node)
        node_id += 1
        start_time += 5000000
    
    # Task heads (distributed memory across heads)
    task_heads_memory = [
        (TaskHead.TRACK, 'track', 256 * 1024 * 1024),  # 256MB
        (TaskHead.SEGMENTATION, 'seg', 512 * 1024 * 1024),  # 512MB
        (TaskHead.MOTION, 'motion', 128 * 1024 * 1024),  # 128MB
        (TaskHead.OCCUPANCY, 'occ', 64 * 1024 * 1024),  # 64MB
        (TaskHead.PLANNING, 'planning', 384 * 1024 * 1024),  # 384MB
    ]
    
    for task_head, task_name, memory_alloc in task_heads_memory:
        # Some memory is freed as we process each head
        memory_freed = memory_alloc // 4  # Free 25% of allocated
        
        node = TraceNode(
            id=f"node_{node_id}",
            name=f"{task_name}_head",
            module_path=f"task_heads.{task_name}",
            start_time=start_time,
            duration=3000000,  # 3ms
            memory_allocated=memory_alloc,
            memory_freed=memory_freed,
            cuda_memory_allocated=memory_alloc,
            cuda_memory_freed=memory_freed,
            task_head=task_head,
        )
        node.gradient_info = {'requires_grad': True, 'grad_fn': f'{task_name}Backward'}
        
        current_allocated += memory_alloc - memory_freed
        
        trace_nodes.append(node)
        node_id += 1
        start_time += 3000000
    
    # Add final cleanup operations
    cleanup_memory = current_allocated // 2  # Free half of remaining memory
    node = TraceNode(
        id=f"node_{node_id}",
        name="cleanup",
        module_path="optimizer.step",
        start_time=start_time,
        duration=1000000,  # 1ms
        memory_allocated=0,
        memory_freed=cleanup_memory,
        cuda_memory_allocated=0,
        cuda_memory_freed=cleanup_memory,
        task_head=TaskHead.NONE,
    )
    
    trace_nodes.append(node)
    
    return trace_nodes


def demonstrate_memory_profiling():
    """Demonstrate memory profiling capabilities."""
    
    print("=" * 80)
    print("UniAD Memory Profiling Example")
    print("=" * 80)
    
    # Create memory profiler with UniAD's target memory
    print("\n1. Initializing Memory Profiler:")
    print("-" * 40)
    
    profiler = MemoryProfiler(
        target_memory_gb=40.0,  # UniAD's typical GPU memory requirement
        optimization_threshold=0.8  # Trigger optimizations at 80% usage
    )
    
    print(f"Target memory: {profiler.target_memory_gb} GB")
    print(f"Warning threshold: {profiler.warning_threshold / (1024**3):.1f} GB")
    print(f"Critical threshold: {profiler.critical_threshold / (1024**3):.1f} GB")
    print(f"Optimization threshold: {profiler.optimization_threshold * 100:.0f}%")
    
    # Create simulated trace data
    print("\n2. Creating Memory-Intensive Trace Data:")
    print("-" * 40)
    
    trace_data = create_memory_intensive_trace(num_frames=3, batch_size=2)
    print(f"Generated {len(trace_data)} trace nodes")
    
    total_allocated = sum(n.cuda_memory_allocated for n in trace_data)
    total_freed = sum(n.cuda_memory_freed for n in trace_data)
    net_memory = total_allocated - total_freed
    
    print(f"Total allocated: {total_allocated / (1024**3):.2f} GB")
    print(f"Total freed: {total_freed / (1024**3):.2f} GB")
    print(f"Net memory usage: {net_memory / (1024**3):.2f} GB")
    
    # Profile memory usage
    print("\n3. Profiling Memory Usage:")
    print("-" * 40)
    
    memory_analysis = profiler.profile_memory(trace_data)
    
    # Summary
    summary = memory_analysis['summary']
    print(f"Peak memory: {summary['peak_memory_gb']:.2f} GB")
    print(f"Memory utilization: {summary['memory_utilization']:.1f}%")
    print(f"Status: {summary['status']}")
    print(f"Optimization suggestions: {summary['optimization_suggestions']}")
    
    # Peak analysis
    print("\n4. Peak Memory Analysis:")
    print("-" * 40)
    
    peak = memory_analysis['peak_analysis']
    if 'no_peak_found' not in peak:
        print(f"Peak operation: {peak['peak_operation']}")
        print(f"Peak module: {peak['peak_module']}")
        print(f"Peak timestamp: {peak['peak_timestamp_ms']:.2f} ms")
        print(f"Peak percentage of target: {peak['peak_percentage']:.1f}%")
    
    # Allocation patterns
    print("\n5. Memory Allocation Patterns:")
    print("-" * 40)
    
    patterns = memory_analysis['allocation_patterns']
    if patterns:
        # Sort by total memory
        sorted_patterns = sorted(patterns.items(), 
                               key=lambda x: x[1]['total_mb'], 
                               reverse=True)
        
        print("Top memory-consuming operations:")
        for op_type, stats in sorted_patterns[:5]:
            print(f"  {op_type}:")
            print(f"    - Count: {stats['count']}")
            print(f"    - Total: {stats['total_mb']:.1f} MB")
            print(f"    - Average: {stats['avg_mb']:.1f} MB")
            print(f"    - Max: {stats['max_mb']:.1f} MB")
    
    # Task head distribution
    print("\n6. Task Head Memory Distribution:")
    print("-" * 40)
    
    task_distribution = memory_analysis['task_head_distribution']
    if task_distribution:
        total_task_memory = sum(info['memory_mb'] for info in task_distribution.values())
        print(f"Total task head memory: {total_task_memory:.1f} MB")
        
        for task_head, info in task_distribution.items():
            print(f"  {task_head}:")
            print(f"    - Memory: {info['memory_mb']:.1f} MB")
            print(f"    - Percentage: {info['percentage']:.1f}%")
    
    # Memory regions
    print("\n7. Memory Regions Analysis:")
    print("-" * 40)
    
    regions = memory_analysis['memory_regions']
    if regions:
        print(f"Identified {len(regions)} memory regions")
        
        for region in regions[:3]:  # Show top 3
            print(f"\n  Region: {region['name']}")
            print(f"    - Duration: {region['duration_ms']:.2f} ms")
            print(f"    - Operations: {region['operation_count']}")
            print(f"    - Peak memory: {region['peak_memory_mb']:.1f} MB")
            print(f"    - Efficiency: {region['efficiency']:.2%}")
    
    # Optimization suggestions
    print("\n8. Memory Optimization Suggestions:")
    print("-" * 40)
    
    suggestions = memory_analysis['optimization_suggestions']
    if suggestions:
        # Group by severity
        by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
        for suggestion in suggestions:
            by_severity[suggestion['severity']].append(suggestion)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if by_severity[severity]:
                print(f"\n  {severity.upper()} Priority:")
                for sugg in by_severity[severity][:2]:  # Show top 2 per severity
                    print(f"    • {sugg['description']}")
                    print(f"      Target: {sugg['target']}")
                    print(f"      Expected savings: {sugg['expected_savings_mb']:.0f} MB")
                    print(f"      Difficulty: {sugg['difficulty']}")
                    if sugg['techniques']:
                        print(f"      Techniques: {', '.join(sugg['techniques'][:2])}")
    
    # Memory lifecycle analysis
    print("\n9. Memory Lifecycle Analysis:")
    print("-" * 40)
    
    lifecycle = profiler.analyze_memory_lifecycle(trace_data)
    
    # Allocation frequency
    if lifecycle['allocation_frequency']:
        print("Allocation patterns:")
        for module, freq_info in list(lifecycle['allocation_frequency'].items())[:3]:
            print(f"  {module}:")
            print(f"    - Count: {freq_info['count']}")
            print(f"    - Avg interval: {freq_info['avg_interval_ns'] / 1e6:.2f} ms")
            print(f"    - Pattern: {freq_info['pattern']}")
    
    # Memory leaks
    if lifecycle['memory_leaks']:
        print("\n⚠️ Potential memory leaks detected:")
        for leak in lifecycle['memory_leaks']:
            print(f"  {leak['module']}:")
            print(f"    - Allocated: {leak['allocated_mb']:.1f} MB")
            print(f"    - Freed: {leak['freed_mb']:.1f} MB")
            print(f"    - Leaked: {leak['leaked_mb']:.1f} MB")
    
    # Gradient checkpointing suggestions
    print("\n10. Gradient Checkpointing Analysis:")
    print("-" * 40)
    
    checkpointing = profiler.suggest_gradient_checkpointing(trace_data)
    if checkpointing:
        total_savings = sum(s['expected_savings_mb'] for s in checkpointing)
        print(f"Potential total savings: {total_savings:.1f} MB")
        
        print("\nTop checkpointing candidates:")
        for i, suggestion in enumerate(checkpointing[:3], 1):
            print(f"  {i}. {suggestion['module']}")
            print(f"     - Operation: {suggestion['operation']}")
            print(f"     - Activation memory: {suggestion['activation_memory_mb']:.1f} MB")
            print(f"     - Expected savings: {suggestion['expected_savings_mb']:.1f} MB")
            print(f"     - Recomputation time: {suggestion['recomputation_time_ms']:.2f} ms")
    
    # Mixed precision opportunities
    print("\n11. Mixed Precision Analysis:")
    print("-" * 40)
    
    mixed_precision = profiler.analyze_mixed_precision_opportunities(trace_data)
    
    print(f"FP32 operations: {mixed_precision['fp32_operations']}")
    print(f"FP16 compatible: {mixed_precision['fp16_compatible']}")
    print(f"Current memory: {mixed_precision['current_memory_gb']:.2f} GB")
    print(f"Potential savings: {mixed_precision['potential_savings_gb']:.2f} GB")
    print(f"Conversion percentage: {mixed_precision['conversion_percentage']:.1f}%")
    
    if mixed_precision['recommendations']:
        print("\nMixed precision recommendations:")
        for rec in mixed_precision['recommendations'][:3]:
            print(f"  • {rec}")
    
    # Memory efficiency score
    print("\n12. Overall Memory Efficiency:")
    print("-" * 40)
    
    efficiency = memory_analysis['memory_efficiency']
    print(f"Memory efficiency score: {efficiency:.2%}")
    
    if efficiency < 0.5:
        print("⚠️ Low memory efficiency detected")
        print("Consider implementing the suggested optimizations")
    elif efficiency < 0.7:
        print("ℹ️ Moderate memory efficiency")
        print("Some optimizations could improve performance")
    else:
        print("✅ Good memory efficiency")
    
    # Export analysis
    print("\n13. Exporting Analysis:")
    print("-" * 40)
    
    export_data = profiler.export_analysis()
    print(f"Export contains {len(export_data)} analysis sections:")
    for key in export_data.keys():
        print(f"  - {key}")
    
    print("\n" + "=" * 80)
    print("Memory Profiling Complete!")
    print("=" * 80)
    
    return memory_analysis


def main():
    """Main entry point."""
    try:
        analysis = demonstrate_memory_profiling()
        
        print("\nKey Insights for UniAD Memory Optimization:")
        print("-" * 40)
        print("1. BEV encoder operations consume the most memory (2GB peak)")
        print("2. Temporal attention is a significant memory bottleneck")
        print("3. Task heads together use ~1.3GB of memory")
        print("4. Gradient checkpointing could save significant memory")
        print("5. Mixed precision training is highly recommended")
        
        print("\nPriority Optimizations:")
        print("-" * 40)
        print("1. Enable automatic mixed precision (AMP)")
        print("2. Implement gradient checkpointing for BEV encoder")
        print("3. Use in-place operations where possible")
        print("4. Consider reducing batch size or temporal frames")
        print("5. Optimize task head memory allocation")
        
        print("\nMemory Budget Recommendations:")
        print("-" * 40)
        print("• Minimum: 32GB GPU (with optimizations)")
        print("• Recommended: 40GB GPU (standard configuration)")
        print("• Optimal: 48GB GPU (for larger batch sizes)")
        
        return 0
        
    except Exception as e:
        print(f"Error during memory profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())