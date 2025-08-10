"""
BEV Analyzer for UniAD Model Analyzer.

This module analyzes Bird's Eye View (BEV) encoder operations in UniAD,
including spatial transformations, pooling operations, and grid generation.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from core.data_structures import TraceNode, TaskHead, TensorShape


@dataclass
class BEVGridProfile:
    """Profile for BEV grid operations."""
    
    grid_resolution: Tuple[int, int]  # (H, W) in BEV space
    feature_channels: int
    spatial_extent: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    
    # Memory metrics
    memory_bytes: int = 0
    memory_per_channel: float = 0.0
    memory_per_pixel: float = 0.0
    
    # Operation counts
    projection_ops: int = 0
    pooling_ops: int = 0
    aggregation_ops: int = 0
    transformation_ops: int = 0
    
    # Performance metrics
    total_duration_ns: float = 0.0
    avg_operation_time_ns: float = 0.0


@dataclass
class SpatialTransformation:
    """Represents a spatial transformation in BEV processing."""
    
    operation_type: str  # 'projection', 'pooling', 'view_transform', 'sampling'
    from_resolution: Tuple[int, int]
    to_resolution: Tuple[int, int]
    from_view: str  # 'perspective', 'bev', 'multi_camera'
    to_view: str
    
    # Transformation details
    scale_factor: float = 1.0
    interpolation_mode: str = 'bilinear'
    preserve_aspect_ratio: bool = True
    
    # Performance
    duration_ns: float = 0.0
    memory_cost: int = 0
    
    # Associated operations
    operations: List[TraceNode] = field(default_factory=list)


class BEVAnalyzer:
    """
    Analyzer for BEV encoder operations in UniAD.
    
    UniAD's BEV encoder transforms multi-camera features into a unified
    bird's eye view representation for downstream tasks.
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (200, 200),
                 spatial_extent: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)):
        """
        Initialize the BEV analyzer.
        
        Args:
            grid_size: BEV grid resolution (H, W)
            spatial_extent: Spatial coverage in meters (x_min, x_max, y_min, y_max)
        """
        self.grid_size = grid_size
        self.spatial_extent = spatial_extent
        
        # BEV profiles at different stages
        self.bev_profiles: Dict[str, BEVGridProfile] = {}
        
        # Spatial transformations
        self.transformations: List[SpatialTransformation] = []
        
        # Camera-to-BEV mappings
        self.camera_to_bev_ops: List[TraceNode] = []
        
        # BEV-specific operations
        self.bev_operations: List[TraceNode] = []
        
        # Keywords for BEV operations
        self.bev_keywords = [
            'bev', 'bird', 'view', 'lift', 'splat', 'shoot',
            'bev_encoder', 'view_transform', 'grid_sample'
        ]
        
        # Projection keywords
        self.projection_keywords = [
            'project', 'lift', 'transform', 'warp',
            'homography', 'perspective', 'extrinsic', 'intrinsic'
        ]
        
        # Pooling keywords
        self.pooling_keywords = [
            'pool', 'voxel', 'pillar', 'aggregate',
            'reduce', 'compress', 'downsample'
        ]
    
    def analyze_bev_encoder(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze BEV encoder operations.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Comprehensive BEV analysis
        """
        # Identify BEV operations
        self._identify_bev_operations(trace_data)
        
        # Analyze spatial transformations
        self._analyze_spatial_transformations(trace_data)
        
        # Profile BEV grids at different stages
        self._profile_bev_grids(trace_data)
        
        # Analyze camera-to-BEV projection
        self._analyze_camera_projection(trace_data)
        
        # Generate analysis report
        analysis = {
            'bev_grid_config': self._get_grid_configuration(),
            'bev_operations': self._summarize_bev_operations(),
            'spatial_transformations': self._summarize_transformations(),
            'camera_projection': self._analyze_projection_efficiency(),
            'memory_analysis': self._analyze_bev_memory(),
            'performance_metrics': self._calculate_performance_metrics(),
            'optimization_opportunities': self._identify_optimizations(),
        }
        
        return analysis
    
    def analyze_lift_splat_shoot(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze Lift-Splat-Shoot operations specifically.
        
        LSS is a common method for BEV generation from multi-camera inputs.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            LSS-specific analysis
        """
        lift_ops = []
        splat_ops = []
        shoot_ops = []
        
        for node in trace_data:
            module_lower = node.module_path.lower()
            
            if 'lift' in module_lower or 'depth' in module_lower:
                lift_ops.append(node)
            elif 'splat' in module_lower or 'scatter' in module_lower:
                splat_ops.append(node)
            elif 'shoot' in module_lower or 'ray' in module_lower:
                shoot_ops.append(node)
        
        return {
            'lift_stage': self._analyze_lift_stage(lift_ops),
            'splat_stage': self._analyze_splat_stage(splat_ops),
            'shoot_stage': self._analyze_shoot_stage(shoot_ops),
            'total_lss_operations': len(lift_ops) + len(splat_ops) + len(shoot_ops),
            'lss_memory_mb': sum(
                op.get_cuda_memory_delta() for op in lift_ops + splat_ops + shoot_ops
            ) / (1024 * 1024),
        }
    
    def profile_bev_grid(self, grid_shape: Tuple[int, ...], 
                        operations: List[TraceNode]) -> BEVGridProfile:
        """
        Profile a specific BEV grid configuration.
        
        Args:
            grid_shape: Shape of the BEV grid tensor
            operations: Operations on this grid
            
        Returns:
            BEV grid profile
        """
        # Extract grid dimensions
        if len(grid_shape) >= 4:
            batch, channels, height, width = grid_shape[-4:]
        else:
            height, width = self.grid_size
            channels = grid_shape[1] if len(grid_shape) > 1 else 256
        
        profile = BEVGridProfile(
            grid_resolution=(height, width),
            feature_channels=channels,
            spatial_extent=self.spatial_extent
        )
        
        # Analyze operations
        for op in operations:
            profile.memory_bytes += op.get_cuda_memory_delta()
            profile.total_duration_ns += op.duration
            
            # Classify operation
            module_lower = op.module_path.lower()
            if any(kw in module_lower for kw in self.projection_keywords):
                profile.projection_ops += 1
            elif any(kw in module_lower for kw in self.pooling_keywords):
                profile.pooling_ops += 1
            elif 'aggregate' in module_lower or 'fusion' in module_lower:
                profile.aggregation_ops += 1
            else:
                profile.transformation_ops += 1
        
        # Calculate metrics
        total_ops = (profile.projection_ops + profile.pooling_ops + 
                    profile.aggregation_ops + profile.transformation_ops)
        
        if total_ops > 0:
            profile.avg_operation_time_ns = profile.total_duration_ns / total_ops
        
        if channels > 0:
            profile.memory_per_channel = profile.memory_bytes / channels
        
        grid_pixels = height * width
        if grid_pixels > 0:
            profile.memory_per_pixel = profile.memory_bytes / grid_pixels
        
        return profile
    
    def _identify_bev_operations(self, trace_data: List[TraceNode]) -> None:
        """Identify BEV-specific operations in the trace."""
        self.bev_operations = []
        
        for node in trace_data:
            module_lower = node.module_path.lower()
            
            # Check for BEV keywords
            is_bev = any(kw in module_lower for kw in self.bev_keywords)
            
            # Also check for BEV-sized tensors (200x200 grid)
            if not is_bev and node.output_shapes:
                for shape in node.output_shapes:
                    if len(shape.shape) >= 4:
                        h, w = shape.shape[-2:]
                        if (h, w) == self.grid_size:
                            is_bev = True
                            break
            
            if is_bev:
                self.bev_operations.append(node)
                node.bev_operation = True
    
    def _analyze_spatial_transformations(self, trace_data: List[TraceNode]) -> None:
        """Analyze spatial transformations in BEV processing."""
        self.transformations = []
        
        for i, node in enumerate(trace_data):
            if not node.input_shapes or not node.output_shapes:
                continue
            
            # Check for spatial dimension changes
            for in_shape, out_shape in zip(node.input_shapes, node.output_shapes):
                if len(in_shape.shape) >= 4 and len(out_shape.shape) >= 4:
                    in_h, in_w = in_shape.shape[-2:]
                    out_h, out_w = out_shape.shape[-2:]
                    
                    # Significant spatial change
                    if (in_h, in_w) != (out_h, out_w):
                        trans = self._create_transformation(
                            node, (in_h, in_w), (out_h, out_w)
                        )
                        self.transformations.append(trans)
    
    def _create_transformation(self, node: TraceNode,
                              from_res: Tuple[int, int],
                              to_res: Tuple[int, int]) -> SpatialTransformation:
        """Create a spatial transformation object."""
        # Determine transformation type
        module_lower = node.module_path.lower()
        
        if 'project' in module_lower or 'lift' in module_lower:
            op_type = 'projection'
            from_view = 'perspective'
            to_view = 'bev'
        elif 'pool' in module_lower:
            op_type = 'pooling'
            from_view = to_view = 'bev'
        elif 'transform' in module_lower:
            op_type = 'view_transform'
            from_view = 'perspective'
            to_view = 'bev'
        elif 'sample' in module_lower:
            op_type = 'sampling'
            from_view = to_view = 'bev'
        else:
            op_type = 'transformation'
            from_view = to_view = 'unknown'
        
        # Calculate scale factor
        scale_h = to_res[0] / from_res[0] if from_res[0] > 0 else 1.0
        scale_w = to_res[1] / from_res[1] if from_res[1] > 0 else 1.0
        scale_factor = (scale_h + scale_w) / 2
        
        trans = SpatialTransformation(
            operation_type=op_type,
            from_resolution=from_res,
            to_resolution=to_res,
            from_view=from_view,
            to_view=to_view,
            scale_factor=scale_factor,
            duration_ns=node.duration,
            memory_cost=node.get_cuda_memory_delta(),
        )
        
        trans.operations.append(node)
        
        return trans
    
    def _profile_bev_grids(self, trace_data: List[TraceNode]) -> None:
        """Profile BEV grids at different processing stages."""
        # Group BEV operations by stage
        stage_ops = defaultdict(list)
        
        for op in self.bev_operations:
            # Determine stage based on module path
            if 'encoder' in op.module_path.lower():
                stage = 'encoder'
            elif 'decoder' in op.module_path.lower():
                stage = 'decoder'
            elif 'fusion' in op.module_path.lower():
                stage = 'fusion'
            elif 'head' in op.module_path.lower():
                stage = 'output'
            else:
                stage = 'intermediate'
            
            stage_ops[stage].append(op)
        
        # Profile each stage
        for stage, ops in stage_ops.items():
            if ops:
                # Find representative shape
                grid_shape = None
                for op in ops:
                    if op.output_shapes:
                        for shape in op.output_shapes:
                            if len(shape.shape) >= 4:
                                h, w = shape.shape[-2:]
                                if abs(h - self.grid_size[0]) < 50 and abs(w - self.grid_size[1]) < 50:
                                    grid_shape = shape.shape
                                    break
                
                if grid_shape:
                    self.bev_profiles[stage] = self.profile_bev_grid(grid_shape, ops)
    
    def _analyze_camera_projection(self, trace_data: List[TraceNode]) -> None:
        """Analyze camera-to-BEV projection operations."""
        self.camera_to_bev_ops = []
        
        for node in trace_data:
            module_lower = node.module_path.lower()
            
            # Look for camera projection operations
            if any(kw in module_lower for kw in ['camera', 'cam', 'img', 'image']):
                # Check if output might be BEV
                if node.output_shapes:
                    for shape in node.output_shapes:
                        if len(shape.shape) >= 4:
                            h, w = shape.shape[-2:]
                            # Check if close to BEV grid size
                            if abs(h - self.grid_size[0]) < 50 and abs(w - self.grid_size[1]) < 50:
                                self.camera_to_bev_ops.append(node)
                                break
    
    def _get_grid_configuration(self) -> Dict[str, Any]:
        """Get BEV grid configuration details."""
        x_min, x_max, y_min, y_max = self.spatial_extent
        
        return {
            'grid_resolution': self.grid_size,
            'spatial_extent_m': {
                'x_range': [x_min, x_max],
                'y_range': [y_min, y_max],
                'total_area_m2': (x_max - x_min) * (y_max - y_min),
            },
            'meters_per_pixel': {
                'x': (x_max - x_min) / self.grid_size[1],
                'y': (y_max - y_min) / self.grid_size[0],
            },
            'total_grid_points': self.grid_size[0] * self.grid_size[1],
        }
    
    def _summarize_bev_operations(self) -> Dict[str, Any]:
        """Summarize BEV operations."""
        if not self.bev_operations:
            return {'no_bev_operations_found': True}
        
        operation_types = defaultdict(int)
        for op in self.bev_operations:
            operation_types[op.name] += 1
        
        return {
            'total_bev_operations': len(self.bev_operations),
            'unique_operation_types': len(operation_types),
            'operation_breakdown': dict(operation_types),
            'total_duration_ms': sum(op.duration for op in self.bev_operations) / 1e6,
            'total_memory_mb': sum(
                op.get_cuda_memory_delta() for op in self.bev_operations
            ) / (1024 * 1024),
        }
    
    def _summarize_transformations(self) -> List[Dict[str, Any]]:
        """Summarize spatial transformations."""
        summaries = []
        
        for trans in self.transformations:
            summary = {
                'type': trans.operation_type,
                'from_resolution': trans.from_resolution,
                'to_resolution': trans.to_resolution,
                'scale_factor': trans.scale_factor,
                'view_change': f"{trans.from_view} -> {trans.to_view}",
                'duration_ms': trans.duration_ns / 1e6,
                'memory_mb': trans.memory_cost / (1024 * 1024),
            }
            summaries.append(summary)
        
        return summaries
    
    def _analyze_projection_efficiency(self) -> Dict[str, Any]:
        """Analyze camera-to-BEV projection efficiency."""
        if not self.camera_to_bev_ops:
            return {'no_camera_projection_found': True}
        
        total_proj_time = sum(op.duration for op in self.camera_to_bev_ops)
        total_proj_memory = sum(op.get_cuda_memory_delta() for op in self.camera_to_bev_ops)
        
        return {
            'projection_operations': len(self.camera_to_bev_ops),
            'total_projection_time_ms': total_proj_time / 1e6,
            'total_projection_memory_mb': total_proj_memory / (1024 * 1024),
            'avg_time_per_camera_ms': total_proj_time / 1e6 / 6,  # Assuming 6 cameras
            'projection_efficiency': self._calculate_projection_efficiency(),
        }
    
    def _calculate_projection_efficiency(self) -> float:
        """Calculate projection efficiency score."""
        if not self.camera_to_bev_ops:
            return 0.0
        
        # Efficiency based on operations per pixel and time
        total_ops = len(self.camera_to_bev_ops)
        total_time = sum(op.duration for op in self.camera_to_bev_ops)
        grid_pixels = self.grid_size[0] * self.grid_size[1]
        
        if grid_pixels == 0 or total_time == 0:
            return 0.0
        
        # Operations per pixel (lower is better)
        ops_per_pixel = total_ops / grid_pixels
        
        # Time per pixel in nanoseconds (lower is better)
        time_per_pixel = total_time / grid_pixels
        
        # Efficiency score (normalized)
        efficiency = (
            (1.0 / (ops_per_pixel * 1000 + 1)) * 0.5 +  # Fewer ops is better
            (1.0 / (time_per_pixel / 1000 + 1)) * 0.5   # Less time is better
        )
        
        return min(1.0, max(0.0, efficiency))
    
    def _analyze_bev_memory(self) -> Dict[str, Any]:
        """Analyze BEV memory usage."""
        if not self.bev_profiles:
            return {'no_bev_profiles': True}
        
        total_memory = sum(p.memory_bytes for p in self.bev_profiles.values())
        
        stage_memory = {}
        for stage, profile in self.bev_profiles.items():
            stage_memory[stage] = {
                'memory_mb': profile.memory_bytes / (1024 * 1024),
                'memory_per_channel_kb': profile.memory_per_channel / 1024,
                'memory_per_pixel_bytes': profile.memory_per_pixel,
            }
        
        return {
            'total_bev_memory_mb': total_memory / (1024 * 1024),
            'stage_memory': stage_memory,
            'memory_efficiency': self._calculate_memory_efficiency(),
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency for BEV operations."""
        if not self.bev_profiles:
            return 0.0
        
        # Calculate theoretical minimum memory
        grid_pixels = self.grid_size[0] * self.grid_size[1]
        typical_channels = 256
        theoretical_min = grid_pixels * typical_channels * 4  # float32
        
        # Actual memory usage
        actual_memory = sum(p.memory_bytes for p in self.bev_profiles.values())
        
        if actual_memory == 0:
            return 0.0
        
        # Efficiency is theoretical / actual (closer to 1 is better)
        efficiency = theoretical_min / actual_memory
        
        return min(1.0, max(0.0, efficiency))
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate BEV performance metrics."""
        if not self.bev_operations:
            return {'no_performance_data': True}
        
        total_time = sum(op.duration for op in self.bev_operations)
        
        # Group by operation type
        time_by_type = defaultdict(float)
        for op in self.bev_operations:
            time_by_type[op.name] += op.duration
        
        # Find bottlenecks
        sorted_types = sorted(time_by_type.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_bev_time_ms': total_time / 1e6,
            'avg_operation_time_us': (total_time / len(self.bev_operations)) / 1e3,
            'bottlenecks': [
                {'operation': op, 'time_ms': t / 1e6, 'percentage': t / total_time * 100}
                for op, t in sorted_types[:5]
            ],
            'performance_score': self._calculate_performance_score(),
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall BEV performance score."""
        if not self.bev_operations:
            return 0.0
        
        total_time = sum(op.duration for op in self.bev_operations)
        grid_pixels = self.grid_size[0] * self.grid_size[1]
        
        # Target: Process BEV in under 10ms
        target_time_ns = 10 * 1e6  # 10ms in nanoseconds
        
        if total_time == 0:
            return 1.0
        
        # Score based on how close to target
        score = target_time_ns / total_time
        
        return min(1.0, max(0.0, score))
    
    def _identify_optimizations(self) -> List[str]:
        """Identify optimization opportunities for BEV processing."""
        optimizations = []
        
        # Check for inefficient transformations
        for trans in self.transformations:
            if trans.scale_factor < 0.5:
                optimizations.append(
                    f"Large downsampling detected ({trans.from_resolution} -> {trans.to_resolution}). "
                    "Consider multi-scale processing."
                )
        
        # Check memory usage
        if self.bev_profiles:
            total_memory = sum(p.memory_bytes for p in self.bev_profiles.values())
            if total_memory > 500 * 1024 * 1024:  # 500MB
                optimizations.append(
                    f"High BEV memory usage ({total_memory / (1024 * 1024):.1f}MB). "
                    "Consider reducing feature channels or using mixed precision."
                )
        
        # Check for redundant operations
        if len(self.bev_operations) > 100:
            optimizations.append(
                f"Large number of BEV operations ({len(self.bev_operations)}). "
                "Consider operation fusion or simplification."
            )
        
        # Check projection efficiency
        proj_efficiency = self._calculate_projection_efficiency()
        if proj_efficiency < 0.5:
            optimizations.append(
                "Low camera-to-BEV projection efficiency. "
                "Consider optimizing view transformation or using cached projections."
            )
        
        # Check for BEV grid resolution
        if self.grid_size[0] * self.grid_size[1] > 50000:
            optimizations.append(
                f"Large BEV grid ({self.grid_size}). "
                "Consider adaptive resolution or region-of-interest processing."
            )
        
        if not optimizations:
            optimizations.append("BEV processing appears well-optimized.")
        
        return optimizations
    
    def _analyze_lift_stage(self, lift_ops: List[TraceNode]) -> Dict[str, Any]:
        """Analyze the lift stage of LSS."""
        if not lift_ops:
            return {'no_lift_operations': True}
        
        return {
            'operation_count': len(lift_ops),
            'total_time_ms': sum(op.duration for op in lift_ops) / 1e6,
            'memory_mb': sum(op.get_cuda_memory_delta() for op in lift_ops) / (1024 * 1024),
            'depth_estimation': any('depth' in op.module_path.lower() for op in lift_ops),
        }
    
    def _analyze_splat_stage(self, splat_ops: List[TraceNode]) -> Dict[str, Any]:
        """Analyze the splat stage of LSS."""
        if not splat_ops:
            return {'no_splat_operations': True}
        
        return {
            'operation_count': len(splat_ops),
            'total_time_ms': sum(op.duration for op in splat_ops) / 1e6,
            'memory_mb': sum(op.get_cuda_memory_delta() for op in splat_ops) / (1024 * 1024),
            'scatter_operations': sum(1 for op in splat_ops if 'scatter' in op.name.lower()),
        }
    
    def _analyze_shoot_stage(self, shoot_ops: List[TraceNode]) -> Dict[str, Any]:
        """Analyze the shoot stage of LSS."""
        if not shoot_ops:
            return {'no_shoot_operations': True}
        
        return {
            'operation_count': len(shoot_ops),
            'total_time_ms': sum(op.duration for op in shoot_ops) / 1e6,
            'memory_mb': sum(op.get_cuda_memory_delta() for op in shoot_ops) / (1024 * 1024),
        }
    
    def export_analysis(self) -> Dict[str, Any]:
        """Export complete BEV analysis."""
        return {
            'grid_configuration': self._get_grid_configuration(),
            'bev_operations': self._summarize_bev_operations(),
            'transformations': self._summarize_transformations(),
            'memory_analysis': self._analyze_bev_memory(),
            'performance': self._calculate_performance_metrics(),
            'optimizations': self._identify_optimizations(),
        }