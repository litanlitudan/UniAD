#!/usr/bin/env python3
"""
Example of using the BEV Analyzer to analyze Bird's Eye View operations in UniAD.

This script demonstrates how to analyze BEV encoder operations, spatial transformations,
camera-to-BEV projections, and the Lift-Splat-Shoot pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TensorShape
from analyzers.bev_analyzer import BEVAnalyzer, BEVGridProfile, SpatialTransformation


class SimpleBEVModel(nn.Module):
    """Simplified BEV model similar to UniAD's BEV encoder."""
    
    def __init__(self, num_cameras=6, bev_h=200, bev_w=200):
        super().__init__()
        self.num_cameras = num_cameras
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # Camera feature extractors (simplified)
        self.camera_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ) for _ in range(num_cameras)
        ])
        
        # Lift: Depth prediction for each camera
        self.depth_net = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),  # Depth bins
        )
        
        # Splat: Project to BEV (simplified as conv for example)
        self.lift_splat_shoot = nn.Conv2d(
            128 * num_cameras,  # Concatenated camera features
            256,  # BEV features
            1
        )
        
        # BEV encoder layers
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # BEV pooling for efficiency
        self.bev_pool = nn.MaxPool2d(2, 2)  # Downsample BEV grid
        
        # Final BEV features
        self.bev_output = nn.Conv2d(512, 256, 1)
    
    def forward(self, camera_inputs):
        """
        Forward pass through BEV encoder.
        
        Args:
            camera_inputs: List of camera tensors, each (B, 3, H, W)
        """
        # Process each camera
        camera_features = []
        for i, cam_input in enumerate(camera_inputs):
            feat = self.camera_encoders[i](cam_input)
            camera_features.append(feat)
        
        # Lift stage: Predict depth for each camera
        depth_features = []
        for feat in camera_features:
            depth = self.depth_net(feat)
            depth_features.append(depth)
        
        # Splat stage: Project to BEV space (simplified)
        # In reality, this involves complex geometric transformations
        B = camera_features[0].shape[0]
        
        # Concatenate all camera features
        all_cam_features = torch.cat(camera_features, dim=1)
        
        # Resize to BEV resolution
        bev_features = F.interpolate(
            all_cam_features,
            size=(self.bev_h, self.bev_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Shoot stage: Generate BEV features
        bev_features = self.lift_splat_shoot(bev_features)
        
        # BEV encoding
        bev_encoded = self.bev_encoder(bev_features)
        
        # Optional pooling for efficiency
        bev_pooled = self.bev_pool(bev_encoded)
        
        # Final BEV output
        bev_output = self.bev_output(bev_pooled)
        
        return bev_output


def create_bev_trace_data(num_cameras=6, bev_h=200, bev_w=200) -> List[TraceNode]:
    """Create simulated trace data for BEV analysis."""
    trace_nodes = []
    node_id = 0
    start_time = 0
    
    # Camera feature extraction (6 cameras)
    for cam_idx in range(num_cameras):
        for layer_idx, layer_name in enumerate(['Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d']):
            node = TraceNode(
                id=f"node_{node_id}",
                name=layer_name,
                module_path=f"camera_encoders.{cam_idx}.{layer_idx}",
                start_time=start_time,
                duration=1000000,  # 1ms
                memory_allocated=0,
                memory_freed=0,
                cuda_memory_allocated=5 * 1024 * 1024,  # 5MB
                cuda_memory_freed=0,
            )
            
            # Add shape information for first conv
            if layer_idx == 0:
                node.input_shapes = [TensorShape(
                    shape=(2, 3, 640, 480),  # Camera input resolution
                    dtype=torch.float32,
                    device='cuda:0',
                    requires_grad=False,
                    memory_bytes=2 * 3 * 640 * 480 * 4
                )]
                node.output_shapes = [TensorShape(
                    shape=(2, 64, 640, 480),
                    dtype=torch.float32,
                    device='cuda:0',
                    requires_grad=False,
                    memory_bytes=2 * 64 * 640 * 480 * 4
                )]
            
            trace_nodes.append(node)
            node_id += 1
            start_time += 1000000
    
    # Depth prediction (Lift stage)
    for layer_name in ['Conv2d', 'ReLU', 'Conv2d']:
        node = TraceNode(
            id=f"node_{node_id}",
            name=layer_name,
            module_path=f"depth_net.{layer_name}",
            start_time=start_time,
            duration=2000000,  # 2ms - depth prediction is expensive
            memory_allocated=0,
            memory_freed=0,
            cuda_memory_allocated=8 * 1024 * 1024,  # 8MB
            cuda_memory_freed=0,
        )
        trace_nodes.append(node)
        node_id += 1
        start_time += 2000000
    
    # Lift-Splat-Shoot transformation
    node = TraceNode(
        id=f"node_{node_id}",
        name="LiftSplatShoot",
        module_path="lift_splat_shoot",
        start_time=start_time,
        duration=5000000,  # 5ms - complex transformation
        memory_allocated=0,
        memory_freed=0,
        cuda_memory_allocated=20 * 1024 * 1024,  # 20MB
        cuda_memory_freed=5 * 1024 * 1024,  # Free 5MB
        bev_operation=True,
    )
    
    # Add BEV grid shape
    node.output_shapes = [TensorShape(
        shape=(2, 256, bev_h, bev_w),  # BEV features
        dtype=torch.float32,
        device='cuda:0',
        requires_grad=False,
        memory_bytes=2 * 256 * bev_h * bev_w * 4
    )]
    
    trace_nodes.append(node)
    node_id += 1
    start_time += 5000000
    
    # BEV encoder layers
    for layer_idx, layer_name in enumerate(['Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'ReLU']):
        node = TraceNode(
            id=f"node_{node_id}",
            name=layer_name,
            module_path=f"bev_encoder.{layer_idx}",
            start_time=start_time,
            duration=1500000,  # 1.5ms
            memory_allocated=0,
            memory_freed=0,
            cuda_memory_allocated=10 * 1024 * 1024,  # 10MB
            cuda_memory_freed=0,
            bev_operation=True,
        )
        
        # Add BEV shapes
        if 'Conv2d' in layer_name:
            node.input_shapes = [TensorShape(
                shape=(2, 256 if layer_idx == 0 else 512, bev_h, bev_w),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=False,
                memory_bytes=0  # Calculated automatically
            )]
            node.output_shapes = [TensorShape(
                shape=(2, 256 if layer_idx == 0 else 512, bev_h, bev_w),
                dtype=torch.float32,
                device='cuda:0',
                requires_grad=False,
                memory_bytes=0
            )]
        
        trace_nodes.append(node)
        node_id += 1
        start_time += 1500000
    
    # BEV pooling
    node = TraceNode(
        id=f"node_{node_id}",
        name="MaxPool2d",
        module_path="bev_pool",
        start_time=start_time,
        duration=500000,  # 0.5ms
        memory_allocated=0,
        memory_freed=0,
        cuda_memory_allocated=3 * 1024 * 1024,  # 3MB
        cuda_memory_freed=10 * 1024 * 1024,  # Free 10MB after pooling
        bev_operation=True,
    )
    
    node.input_shapes = [TensorShape(
        shape=(2, 512, bev_h, bev_w),
        dtype=torch.float32,
        device='cuda:0',
        requires_grad=False,
        memory_bytes=0
    )]
    node.output_shapes = [TensorShape(
        shape=(2, 512, bev_h // 2, bev_w // 2),  # Pooled
        dtype=torch.float32,
        device='cuda:0',
        requires_grad=False,
        memory_bytes=0
    )]
    
    trace_nodes.append(node)
    node_id += 1
    start_time += 500000
    
    # Final BEV output
    node = TraceNode(
        id=f"node_{node_id}",
        name="Conv2d",
        module_path="bev_output",
        start_time=start_time,
        duration=1000000,  # 1ms
        memory_allocated=0,
        memory_freed=0,
        cuda_memory_allocated=5 * 1024 * 1024,  # 5MB
        cuda_memory_freed=0,
        bev_operation=True,
    )
    
    trace_nodes.append(node)
    
    return trace_nodes


def demonstrate_bev_analysis():
    """Demonstrate BEV analysis capabilities."""
    
    print("=" * 80)
    print("UniAD BEV Analysis Example")
    print("=" * 80)
    
    # Create BEV analyzer with UniAD's typical configuration
    print("\n1. Initializing BEV Analyzer:")
    print("-" * 40)
    
    analyzer = BEVAnalyzer(
        grid_size=(200, 200),  # UniAD's BEV grid resolution
        spatial_extent=(-51.2, 51.2, -51.2, 51.2)  # 102.4m x 102.4m coverage
    )
    
    grid_config = analyzer._get_grid_configuration()
    print(f"BEV Grid Resolution: {grid_config['grid_resolution']}")
    print(f"Spatial Coverage: {grid_config['spatial_extent_m']['total_area_m2']:.1f} m²")
    print(f"Meters per pixel: X={grid_config['meters_per_pixel']['x']:.2f}, "
          f"Y={grid_config['meters_per_pixel']['y']:.2f}")
    print(f"Total grid points: {grid_config['total_grid_points']:,}")
    
    # Create simulated trace data
    print("\n2. Creating Simulated BEV Trace Data:")
    print("-" * 40)
    
    trace_data = create_bev_trace_data(num_cameras=6)
    print(f"Generated {len(trace_data)} trace nodes")
    print(f"Camera operations: {sum(1 for n in trace_data if 'camera' in n.module_path.lower())}")
    print(f"BEV operations: {sum(1 for n in trace_data if n.bev_operation)}")
    
    # Analyze BEV encoder
    print("\n3. Analyzing BEV Encoder Operations:")
    print("-" * 40)
    
    bev_analysis = analyzer.analyze_bev_encoder(trace_data)
    
    # BEV operations summary
    bev_ops = bev_analysis['bev_operations']
    if 'no_bev_operations_found' not in bev_ops:
        print(f"Total BEV operations: {bev_ops['total_bev_operations']}")
        print(f"Unique operation types: {bev_ops['unique_operation_types']}")
        print(f"Total duration: {bev_ops['total_duration_ms']:.2f} ms")
        print(f"Total memory: {bev_ops['total_memory_mb']:.2f} MB")
        
        print("\nOperation breakdown:")
        for op_type, count in bev_ops['operation_breakdown'].items():
            print(f"  - {op_type}: {count}")
    
    # Spatial transformations
    print("\n4. Spatial Transformations Analysis:")
    print("-" * 40)
    
    transformations = bev_analysis['spatial_transformations']
    if transformations:
        print(f"Found {len(transformations)} spatial transformations")
        for i, trans in enumerate(transformations[:3], 1):
            print(f"\nTransformation {i}:")
            print(f"  Type: {trans['type']}")
            print(f"  Resolution: {trans['from_resolution']} → {trans['to_resolution']}")
            print(f"  Scale factor: {trans['scale_factor']:.2f}")
            print(f"  View change: {trans['view_change']}")
            print(f"  Duration: {trans['duration_ms']:.2f} ms")
    
    # Camera projection analysis
    print("\n5. Camera-to-BEV Projection Analysis:")
    print("-" * 40)
    
    projection = bev_analysis['camera_projection']
    if 'no_camera_projection_found' not in projection:
        print(f"Projection operations: {projection['projection_operations']}")
        print(f"Total time: {projection['total_projection_time_ms']:.2f} ms")
        print(f"Total memory: {projection['total_projection_memory_mb']:.2f} MB")
        print(f"Avg time per camera: {projection['avg_time_per_camera_ms']:.2f} ms")
        print(f"Projection efficiency: {projection['projection_efficiency']:.2%}")
    
    # Lift-Splat-Shoot analysis
    print("\n6. Lift-Splat-Shoot Pipeline Analysis:")
    print("-" * 40)
    
    lss_analysis = analyzer.analyze_lift_splat_shoot(trace_data)
    
    print(f"Total LSS operations: {lss_analysis['total_lss_operations']}")
    print(f"LSS memory usage: {lss_analysis['lss_memory_mb']:.2f} MB")
    
    # Lift stage
    lift_stage = lss_analysis['lift_stage']
    if 'no_lift_operations' not in lift_stage:
        print(f"\nLift Stage:")
        print(f"  Operations: {lift_stage['operation_count']}")
        print(f"  Time: {lift_stage['total_time_ms']:.2f} ms")
        print(f"  Memory: {lift_stage['memory_mb']:.2f} MB")
        print(f"  Depth estimation: {lift_stage['depth_estimation']}")
    
    # Splat stage
    splat_stage = lss_analysis['splat_stage']
    if 'no_splat_operations' not in splat_stage:
        print(f"\nSplat Stage:")
        print(f"  Operations: {splat_stage['operation_count']}")
        print(f"  Time: {splat_stage['total_time_ms']:.2f} ms")
        print(f"  Memory: {splat_stage['memory_mb']:.2f} MB")
    
    # Memory analysis
    print("\n7. BEV Memory Analysis:")
    print("-" * 40)
    
    memory_analysis = bev_analysis['memory_analysis']
    if 'no_bev_profiles' not in memory_analysis:
        if 'total_bev_memory_mb' in memory_analysis:
            print(f"Total BEV memory: {memory_analysis['total_bev_memory_mb']:.2f} MB")
        if 'memory_efficiency' in memory_analysis:
            print(f"Memory efficiency: {memory_analysis['memory_efficiency']:.2%}")
        
        if 'stage_memory' in memory_analysis:
            print("\nMemory by stage:")
            for stage, mem_info in memory_analysis['stage_memory'].items():
                print(f"  {stage}:")
                print(f"    - Memory: {mem_info['memory_mb']:.2f} MB")
                print(f"    - Per channel: {mem_info['memory_per_channel_kb']:.2f} KB")
                print(f"    - Per pixel: {mem_info['memory_per_pixel_bytes']:.1f} bytes")
    
    # Performance metrics
    print("\n8. Performance Metrics:")
    print("-" * 40)
    
    performance = bev_analysis['performance_metrics']
    if 'no_performance_data' not in performance:
        if 'total_bev_time_ms' in performance:
            print(f"Total BEV processing time: {performance['total_bev_time_ms']:.2f} ms")
        if 'avg_operation_time_us' in performance:
            print(f"Average operation time: {performance['avg_operation_time_us']:.2f} μs")
        if 'performance_score' in performance:
            print(f"Performance score: {performance['performance_score']:.2%}")
        
        if 'bottlenecks' in performance and performance['bottlenecks']:
            print("\nTop bottlenecks:")
            for i, bottleneck in enumerate(performance['bottlenecks'][:3], 1):
                print(f"  {i}. {bottleneck['operation']}: {bottleneck['time_ms']:.2f} ms ({bottleneck['percentage']:.1f}%)")
    
    # Optimization opportunities
    print("\n9. Optimization Opportunities:")
    print("-" * 40)
    
    optimizations = bev_analysis['optimization_opportunities']
    if optimizations:
        for i, opt in enumerate(optimizations, 1):
            print(f"{i}. {opt}")
    
    # Export analysis
    print("\n10. Exporting Analysis:")
    print("-" * 40)
    
    export_data = analyzer.export_analysis()
    print(f"Export contains {len(export_data)} analysis sections:")
    for key in export_data.keys():
        print(f"  - {key}")
    
    print("\n" + "=" * 80)
    print("BEV Analysis Complete!")
    print("=" * 80)
    
    return bev_analysis


def main():
    """Main entry point."""
    try:
        analysis = demonstrate_bev_analysis()
        
        print("\nKey Insights for UniAD BEV Processing:")
        print("-" * 40)
        print("1. Multi-camera fusion is the most memory-intensive stage")
        print("2. BEV grid resolution (200x200) balances accuracy and efficiency")
        print("3. Lift-Splat-Shoot provides geometric consistency")
        print("4. Pooling operations help reduce memory after BEV encoding")
        print("5. Spatial transformations are critical for performance")
        
        print("\nRecommendations:")
        print("-" * 40)
        print("1. Consider mixed precision for BEV features")
        print("2. Use cached camera-to-BEV projections when possible")
        print("3. Implement adaptive BEV resolution for distant regions")
        print("4. Optimize depth prediction in the Lift stage")
        print("5. Profile memory usage during multi-camera fusion")
        
        return 0
        
    except Exception as e:
        print(f"Error during BEV analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())