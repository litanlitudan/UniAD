#!/usr/bin/env python3
"""
Example of using the Temporal Analyzer to analyze multi-frame aggregation in UniAD.

This script demonstrates how to analyze temporal patterns, memory usage across frames,
and attention mechanisms in UniAD's temporal processing.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TensorShape
from analyzers.temporal_analyzer import TemporalAnalyzer, TemporalFrameProfile


class TemporalModel(nn.Module):
    """Simplified model with temporal processing similar to UniAD."""
    
    def __init__(self, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        
        # Feature extractor per frame
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Temporal self-attention (like UniAD's temporal_self_attention)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True
        )
        
        # Temporal aggregation
        self.temporal_aggregator = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # BEV features after temporal fusion
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        Forward pass with temporal processing.
        
        Args:
            x: Input tensor of shape (B*T, C, H, W) where T is num_frames
        """
        batch_size = x.shape[0] // self.num_frames
        
        # Process each frame
        frame_features = []
        for t in range(self.num_frames):
            frame_idx = slice(t * batch_size, (t + 1) * batch_size)
            frame_feat = self.frame_encoder(x[frame_idx])
            frame_features.append(frame_feat)
        
        # Stack for temporal processing
        # Shape: (B, T, C, H, W)
        temporal_features = torch.stack(frame_features, dim=1)
        
        # Flatten spatial dimensions for attention
        B, T, C, H, W = temporal_features.shape
        temporal_features_flat = temporal_features.view(B, T, C * H * W)
        
        # Temporal self-attention
        attended_features, attention_weights = self.temporal_attention(
            temporal_features_flat,
            temporal_features_flat,
            temporal_features_flat
        )
        
        # Reshape back
        attended_features = attended_features.view(B, T, C, H, W)
        
        # Temporal aggregation
        # Reshape for Conv1d: (B, C*H*W, T)
        features_for_agg = attended_features.view(B, C * H * W, T)
        aggregated = self.temporal_aggregator(features_for_agg)
        
        # Reshape to spatial: (B, 128, H, W)
        aggregated = aggregated.view(B, 128, H, W)
        
        # BEV encoding
        bev_features = self.bev_encoder(aggregated)
        
        return bev_features, attention_weights


def create_simulated_trace(num_frames=3, batch_size=2):
    """Create simulated trace data for temporal analysis."""
    trace_nodes = []
    node_id = 0
    
    # Simulate frame encoding operations
    for frame in range(num_frames):
        # Frame encoder operations
        for layer_idx, layer_name in enumerate(['Conv2d', 'BatchNorm2d', 'ReLU']):
            node = TraceNode(
                id=f"node_{node_id}",
                name=layer_name,
                module_path=f"frame_encoder.{layer_idx}",
                temporal_frame=frame,
                start_time=node_id * 1000000,  # in nanoseconds
                duration=500000,  # 0.5ms
                memory_allocated=1024 * 1024 * (layer_idx + 1),  # 1-3 MB
                memory_freed=0,
                cuda_memory_allocated=1024 * 1024 * (layer_idx + 1),
                cuda_memory_freed=0,
            )
            
            # Add shape information
            if layer_idx == 0:  # Conv2d
                node.input_shapes = [TensorShape(
                    shape=(batch_size, 3, 224, 224),
                    dtype=torch.float32,
                    device='cuda:0',
                    requires_grad=False,
                    memory_bytes=batch_size * 3 * 224 * 224 * 4
                )]
                node.output_shapes = [TensorShape(
                    shape=(batch_size, 64, 224, 224),
                    dtype=torch.float32,
                    device='cuda:0',
                    requires_grad=False,
                    memory_bytes=batch_size * 64 * 224 * 224 * 4
                )]
            
            trace_nodes.append(node)
            node_id += 1
    
    # Simulate temporal attention operations
    for head_idx in range(8):  # 8 attention heads
        node = TraceNode(
            id=f"node_{node_id}",
            name="MultiheadAttention",
            module_path=f"temporal_attention.head_{head_idx}",
            temporal_frame=None,  # Processes all frames
            start_time=node_id * 1000000,
            duration=2000000,  # 2ms - attention is expensive
            memory_allocated=5 * 1024 * 1024,  # 5MB
            memory_freed=0,
            cuda_memory_allocated=5 * 1024 * 1024,
            cuda_memory_freed=0,
        )
        trace_nodes.append(node)
        node_id += 1
    
    # Simulate temporal aggregation
    node = TraceNode(
        id=f"node_{node_id}",
        name="Conv1d",
        module_path="temporal_aggregator",
        temporal_frame=None,  # Aggregates all frames
        start_time=node_id * 1000000,
        duration=1000000,  # 1ms
        memory_allocated=3 * 1024 * 1024,  # 3MB
        memory_freed=2 * 1024 * 1024,  # Free 2MB
        cuda_memory_allocated=3 * 1024 * 1024,
        cuda_memory_freed=2 * 1024 * 1024,
    )
    trace_nodes.append(node)
    node_id += 1
    
    # Simulate BEV encoding
    for layer_idx, layer_name in enumerate(['Conv2d', 'BatchNorm2d', 'ReLU']):
        node = TraceNode(
            id=f"node_{node_id}",
            name=layer_name,
            module_path=f"bev_encoder.{layer_idx}",
            temporal_frame=None,  # Post-aggregation
            start_time=node_id * 1000000,
            duration=800000,  # 0.8ms
            memory_allocated=2 * 1024 * 1024,  # 2MB
            memory_freed=0,
            cuda_memory_allocated=2 * 1024 * 1024,
            cuda_memory_freed=0,
            bev_operation=True,  # Mark as BEV operation
        )
        trace_nodes.append(node)
        node_id += 1
    
    return trace_nodes


def demonstrate_temporal_analysis():
    """Demonstrate temporal analysis capabilities."""
    
    print("=" * 80)
    print("UniAD Temporal Analysis Example")
    print("=" * 80)
    
    # Test with 3-frame configuration (UniAD default)
    print("\n1. Analyzing 3-frame temporal configuration:")
    print("-" * 40)
    
    analyzer_3 = TemporalAnalyzer(num_frames=3)
    trace_3 = create_simulated_trace(num_frames=3)
    
    analysis_3 = analyzer_3.analyze_temporal_flow(trace_3)
    
    print(f"Number of frames: {analysis_3['num_frames']}")
    print(f"Temporal efficiency score: {analysis_3['temporal_efficiency']:.2%}")
    
    print("\nFrame-wise analysis:")
    for frame_data in analysis_3['frame_profiles']:
        print(f"  Frame {frame_data['frame_index']}:")
        print(f"    - Operations: {frame_data['operation_count']}")
        print(f"    - Memory: {frame_data['memory_mb']:.2f} MB")
        print(f"    - Duration: {frame_data['duration_ms']:.2f} ms")
        print(f"    - Attention ops: {frame_data['attention_ops']}")
    
    print("\nTemporal patterns detected:")
    for pattern_type, pattern_data in analysis_3['temporal_patterns'].items():
        if pattern_data['count'] > 0:
            print(f"  {pattern_type}:")
            print(f"    - Count: {pattern_data['count']}")
            print(f"    - Total memory: {pattern_data['total_memory_mb']:.2f} MB")
            print(f"    - Avg operations: {pattern_data['avg_operations']:.1f}")
    
    # Test with 5-frame configuration
    print("\n2. Analyzing 5-frame temporal configuration:")
    print("-" * 40)
    
    analyzer_5 = TemporalAnalyzer(num_frames=5)
    trace_5 = create_simulated_trace(num_frames=5)
    
    analysis_5 = analyzer_5.analyze_temporal_flow(trace_5)
    
    print(f"Number of frames: {analysis_5['num_frames']}")
    print(f"Temporal efficiency score: {analysis_5['temporal_efficiency']:.2%}")
    
    # Compare memory usage
    print("\n3. Queue Memory Analysis:")
    print("-" * 40)
    
    queue_3 = analyzer_3.analyze_queue_memory(3)
    queue_5 = analyzer_5.analyze_queue_memory(5)
    
    print(f"3-frame queue:")
    print(f"  - Total memory: {queue_3['total_memory'] / (1024**3):.2f} GB")
    print(f"  - Optimal for: {queue_3['optimal_for']}")
    print(f"  - Memory efficiency: {queue_3['memory_efficiency']:.2%}")
    
    print(f"\n5-frame queue:")
    print(f"  - Total memory: {queue_5['total_memory'] / (1024**3):.2f} GB")
    print(f"  - Optimal for: {queue_5['optimal_for']}")
    print(f"  - Memory efficiency: {queue_5['memory_efficiency']:.2%}")
    
    print("\nOptimization suggestions:")
    for suggestion in queue_5['optimization_suggestions']:
        print(f"  - {suggestion}")
    
    # Attention pattern analysis
    print("\n4. Temporal Attention Analysis:")
    print("-" * 40)
    
    attention_data = analyzer_3.visualize_temporal_attention()
    
    if attention_data['attention_matrices']:
        print(f"Found {len(attention_data['attention_matrices'])} attention operations")
        for i, attn in enumerate(attention_data['attention_matrices'][:3]):
            print(f"  Attention {i+1}:")
            print(f"    - Shape: {attn['shape']}")
            print(f"    - Memory: {attn['memory_mb']:.2f} MB")
            print(f"    - Duration: {attn['duration_ms']:.2f} ms")
    
    if attention_data['frame_connections']:
        print(f"\nFrame-to-frame connections: {len(attention_data['frame_connections'])}")
        for conn in attention_data['frame_connections'][:3]:
            print(f"  Frame {conn['from_frame']} → Frame {conn['to_frame']}:")
            print(f"    - Operations: {conn['operation_count']}")
            print(f"    - Memory: {conn['total_memory_mb']:.2f} MB")
    
    # Memory scaling analysis
    print("\n5. Memory Scaling Analysis:")
    print("-" * 40)
    
    memory_scaling = analysis_3['memory_scaling']
    if 'no_frame_data' not in memory_scaling:
        print(f"Total memory: {memory_scaling['total_memory_mb']:.2f} MB")
        print(f"Average per frame: {memory_scaling['avg_memory_per_frame_mb']:.2f} MB")
        print(f"Memory std dev: {memory_scaling['memory_std_mb']:.2f} MB")
        print(f"Scaling factor: {memory_scaling['scaling_factor']:.2f}")
        print(f"Scaling efficiency: {memory_scaling['scaling_efficiency']:.2%}")
    
    # Export analysis
    print("\n6. Exporting Analysis Results:")
    print("-" * 40)
    
    export_data = analyzer_3.export_analysis()
    print(f"Export contains {len(export_data)} analysis sections")
    for key in export_data.keys():
        print(f"  - {key}")
    
    print("\n" + "=" * 80)
    print("Temporal Analysis Complete!")
    print("=" * 80)
    
    return analysis_3, analysis_5


def main():
    """Main entry point."""
    try:
        analysis_3, analysis_5 = demonstrate_temporal_analysis()
        
        print("\nKey Insights:")
        print("-" * 40)
        print("1. 3-frame configuration is more memory-efficient")
        print("2. 5-frame configuration provides better temporal context")
        print("3. Attention operations are the most memory-intensive")
        print("4. Temporal aggregation helps reduce memory after fusion")
        print("5. BEV encoding happens after temporal fusion")
        
        return 0
        
    except Exception as e:
        print(f"Error during temporal analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())