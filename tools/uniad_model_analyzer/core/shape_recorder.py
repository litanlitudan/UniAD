"""
Shape Recorder Module for UniAD Model Analyzer.

This module provides efficient tensor shape and dtype tracking with memory-aware caching.
It captures shape transformations through the network and analyzes dimension changes.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import numpy as np
import torch

from .data_structures import TensorShape, TaskHead


@dataclass
class ShapeTransformation:
    """Represents a shape transformation between operations."""
    
    from_shape: Tuple[int, ...]
    to_shape: Tuple[int, ...]
    operation: str
    module_path: str
    
    # Transformation metadata
    dimension_change: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate dimension changes after initialization."""
        self.dimension_change = self._analyze_transformation()
    
    def _analyze_transformation(self) -> Dict[str, Any]:
        """Analyze the transformation between shapes."""
        from_dims = len(self.from_shape)
        to_dims = len(self.to_shape)
        
        # Calculate size changes
        from_size = np.prod(self.from_shape) if self.from_shape else 0
        to_size = np.prod(self.to_shape) if self.to_shape else 0
        
        analysis = {
            "dimension_change": to_dims - from_dims,
            "size_ratio": to_size / from_size if from_size > 0 else float('inf'),
            "total_size_change": to_size - from_size,
            "type": self._classify_transformation(),
        }
        
        # Add dimension-specific analysis
        if from_dims == to_dims and from_dims > 0:
            analysis["dim_changes"] = [
                (i, self.to_shape[i] - self.from_shape[i]) 
                for i in range(from_dims)
                if self.to_shape[i] != self.from_shape[i]
            ]
        
        return analysis
    
    def _classify_transformation(self) -> str:
        """Classify the type of transformation."""
        from_dims = len(self.from_shape)
        to_dims = len(self.to_shape)
        
        if from_dims == to_dims:
            if self.from_shape == self.to_shape:
                return "identity"
            elif from_dims >= 2:
                # Check for common patterns
                if self.from_shape[0] == self.to_shape[0]:  # Batch preserved
                    return "spatial_transform"
                else:
                    return "reshape"
        elif to_dims > from_dims:
            return "expansion"
        else:
            return "reduction"


class ShapeRecorder:
    """
    Records and analyzes tensor shapes throughout model execution.
    
    This class efficiently tracks tensor shapes, data types, and transformations
    with memory-aware caching for UniAD's large model analysis.
    """
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize the shape recorder.
        
        Args:
            cache_size: Maximum number of unique shapes to cache
        """
        self.cache_size = cache_size
        
        # Shape storage with LRU-like behavior using OrderedDict
        self.shape_cache: OrderedDict[str, TensorShape] = OrderedDict()
        
        # Transformation tracking
        self.transformations: List[ShapeTransformation] = []
        
        # Statistics
        self.shape_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.dtype_counts: Dict[torch.dtype, int] = defaultdict(int)
        self.device_counts: Dict[str, int] = defaultdict(int)
        
        # UniAD-specific tracking
        self.task_head_shapes: Dict[TaskHead, List[TensorShape]] = defaultdict(list)
        self.bev_shapes: List[TensorShape] = []
        self.temporal_shapes: Dict[int, List[TensorShape]] = defaultdict(list)
        
        # Memory tracking
        self.total_memory_tracked: int = 0
        self.peak_shape_memory: int = 0
        
        # Shape patterns (for detecting common UniAD patterns)
        self.common_patterns = self._init_common_patterns()
    
    def _init_common_patterns(self) -> Dict[str, Tuple[int, ...]]:
        """Initialize common shape patterns in UniAD."""
        return {
            # BEV grid patterns (200x200 at 0.512m resolution)
            "bev_grid_small": (200, 200),
            "bev_grid_medium": (100, 100),
            "bev_grid_large": (400, 400),
            
            # Common image sizes
            "image_full": (900, 1600),  # Full resolution
            "image_crop": (928, 1600),  # After cropping
            "image_resize": (320, 800),  # Common resize
            
            # Feature dimensions
            "bev_features": (256,),  # BEV feature channels
            "img_features": (256,),  # Image feature channels
            
            # Temporal dimensions
            "temporal_3": (3,),  # 3-frame temporal
            "temporal_5": (5,),  # 5-frame temporal
        }
    
    def record_tensor(self, 
                     tensor: torch.Tensor, 
                     context: Dict[str, Any]) -> TensorShape:
        """
        Record a tensor's shape and metadata.
        
        Args:
            tensor: PyTorch tensor to record
            context: Additional context (module_path, task_head, etc.)
            
        Returns:
            TensorShape object with recorded information
        """
        # Create tensor shape
        shape_obj = TensorShape.from_tensor(tensor)
        
        # Generate cache key
        cache_key = self._generate_cache_key(shape_obj, context)
        
        # LRU cache management
        if cache_key in self.shape_cache:
            # Move to end (most recently used)
            self.shape_cache.move_to_end(cache_key)
        else:
            # Add new shape
            if len(self.shape_cache) >= self.cache_size:
                # Remove least recently used
                self.shape_cache.popitem(last=False)
            self.shape_cache[cache_key] = shape_obj
        
        # Update statistics
        self.shape_counts[shape_obj.shape] += 1
        self.dtype_counts[shape_obj.dtype] += 1
        self.device_counts[shape_obj.device] += 1
        
        # Track memory
        self.total_memory_tracked += shape_obj.memory_bytes
        self.peak_shape_memory = max(self.peak_shape_memory, shape_obj.memory_bytes)
        
        # UniAD-specific tracking
        if "task_head" in context and context["task_head"] != TaskHead.NONE:
            self.task_head_shapes[context["task_head"]].append(shape_obj)
        
        if context.get("bev_operation", False):
            self.bev_shapes.append(shape_obj)
        
        if "temporal_frame" in context and context["temporal_frame"] is not None:
            self.temporal_shapes[context["temporal_frame"]].append(shape_obj)
        
        return shape_obj
    
    def _generate_cache_key(self, 
                           shape: TensorShape, 
                           context: Dict[str, Any]) -> str:
        """Generate a unique cache key for a shape."""
        # Include shape, dtype, and key context elements
        key_parts = [
            str(shape.shape),
            str(shape.dtype),
            str(shape.device),
            context.get("module_path", ""),
        ]
        return "|".join(key_parts)
    
    def record_transformation(self,
                            from_tensor: Optional[torch.Tensor],
                            to_tensor: torch.Tensor,
                            operation: str,
                            module_path: str) -> ShapeTransformation:
        """
        Record a shape transformation between operations.
        
        Args:
            from_tensor: Input tensor (can be None for source ops)
            to_tensor: Output tensor
            operation: Operation name
            module_path: Module path in model
            
        Returns:
            ShapeTransformation object
        """
        from_shape = tuple(from_tensor.shape) if from_tensor is not None else ()
        to_shape = tuple(to_tensor.shape)
        
        transformation = ShapeTransformation(
            from_shape=from_shape,
            to_shape=to_shape,
            operation=operation,
            module_path=module_path,
        )
        
        self.transformations.append(transformation)
        return transformation
    
    def get_shape_transformations(self) -> List[ShapeTransformation]:
        """Get all recorded shape transformations."""
        return self.transformations.copy()
    
    def analyze_shape_patterns(self) -> Dict[str, Any]:
        """
        Analyze shape patterns and identify common UniAD patterns.
        
        Returns:
            Dictionary with pattern analysis results
        """
        analysis = {
            "total_unique_shapes": len(set(self.shape_counts.keys())),
            "total_tensors_tracked": sum(self.shape_counts.values()),
            "most_common_shapes": self._get_most_common_shapes(10),
            "identified_patterns": self._identify_patterns(),
            "dimension_distribution": self._analyze_dimensions(),
            "memory_analysis": self._analyze_memory(),
        }
        
        # UniAD-specific analysis
        analysis["task_head_analysis"] = self._analyze_task_heads()
        analysis["bev_analysis"] = self._analyze_bev_shapes()
        analysis["temporal_analysis"] = self._analyze_temporal_shapes()
        
        return analysis
    
    def _get_most_common_shapes(self, top_k: int = 10) -> List[Tuple[Tuple[int, ...], int]]:
        """Get the most common shapes."""
        sorted_shapes = sorted(
            self.shape_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_shapes[:top_k]
    
    def _identify_patterns(self) -> Dict[str, List[Tuple[int, ...]]]:
        """Identify known UniAD patterns in recorded shapes."""
        identified = defaultdict(list)
        
        for shape in self.shape_counts.keys():
            # Check BEV grid patterns
            if len(shape) >= 2:
                spatial_dims = shape[-2:]
                if spatial_dims in [v for k, v in self.common_patterns.items() if "bev_grid" in k]:
                    identified["bev_grids"].append(shape)
            
            # Check temporal patterns
            if len(shape) >= 1:
                if shape[0] in [3, 5]:  # Common temporal dimensions
                    identified["temporal"].append(shape)
            
            # Check image patterns
            if len(shape) == 4:  # NCHW format
                h, w = shape[-2:]
                for pattern_name, pattern_shape in self.common_patterns.items():
                    if "image" in pattern_name and len(pattern_shape) == 2:
                        if (h, w) == pattern_shape:
                            identified["image_sizes"].append(shape)
                            break
        
        return dict(identified)
    
    def _analyze_dimensions(self) -> Dict[str, Any]:
        """Analyze dimension statistics."""
        all_dims = [len(shape) for shape in self.shape_counts.keys()]
        if not all_dims:
            return {}
        
        return {
            "min_dims": min(all_dims),
            "max_dims": max(all_dims),
            "avg_dims": np.mean(all_dims),
            "dim_distribution": dict(zip(*np.unique(all_dims, return_counts=True))),
        }
    
    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        return {
            "total_memory_tracked_mb": self.total_memory_tracked / (1024 * 1024),
            "peak_shape_memory_mb": self.peak_shape_memory / (1024 * 1024),
            "avg_tensor_size_kb": (
                self.total_memory_tracked / sum(self.shape_counts.values()) / 1024
                if sum(self.shape_counts.values()) > 0 else 0
            ),
            "dtype_distribution": {
                str(dtype): count 
                for dtype, count in self.dtype_counts.items()
            },
        }
    
    def _analyze_task_heads(self) -> Dict[str, Any]:
        """Analyze shapes by task head."""
        analysis = {}
        
        for task_head, shapes in self.task_head_shapes.items():
            if shapes:
                unique_shapes = set(s.shape for s in shapes)
                total_memory = sum(s.memory_bytes for s in shapes)
                
                analysis[task_head.value] = {
                    "total_tensors": len(shapes),
                    "unique_shapes": len(unique_shapes),
                    "total_memory_mb": total_memory / (1024 * 1024),
                    "common_shapes": list(unique_shapes)[:5],  # Top 5
                }
        
        return analysis
    
    def _analyze_bev_shapes(self) -> Dict[str, Any]:
        """Analyze BEV-related shapes."""
        if not self.bev_shapes:
            return {"no_bev_operations": True}
        
        unique_bev = set(s.shape for s in self.bev_shapes)
        
        # Check for standard BEV grid sizes
        standard_grids = []
        for shape in unique_bev:
            if len(shape) >= 2:
                h, w = shape[-2:]
                if (h, w) in [(200, 200), (100, 100), (400, 400)]:
                    standard_grids.append((h, w))
        
        return {
            "total_bev_tensors": len(self.bev_shapes),
            "unique_bev_shapes": len(unique_bev),
            "standard_grids_found": standard_grids,
            "total_bev_memory_mb": sum(s.memory_bytes for s in self.bev_shapes) / (1024 * 1024),
        }
    
    def _analyze_temporal_shapes(self) -> Dict[str, Any]:
        """Analyze temporal frame shapes."""
        if not self.temporal_shapes:
            return {"no_temporal_tracking": True}
        
        analysis = {}
        for frame_idx, shapes in self.temporal_shapes.items():
            if shapes:
                analysis[f"frame_{frame_idx}"] = {
                    "tensor_count": len(shapes),
                    "total_memory_mb": sum(s.memory_bytes for s in shapes) / (1024 * 1024),
                    "unique_shapes": len(set(s.shape for s in shapes)),
                }
        
        return analysis
    
    def export_shape_data(self) -> Dict[str, Any]:
        """
        Export all shape analysis data.
        
        Returns:
            Dictionary containing all shape data and analysis
        """
        return {
            "shape_counts": dict(self.shape_counts),
            "dtype_counts": {str(k): v for k, v in self.dtype_counts.items()},
            "device_counts": dict(self.device_counts),
            "transformations": [
                {
                    "from": t.from_shape,
                    "to": t.to_shape,
                    "operation": t.operation,
                    "module_path": t.module_path,
                    "analysis": t.dimension_change,
                }
                for t in self.transformations
            ],
            "analysis": self.analyze_shape_patterns(),
            "cache_stats": {
                "cache_size": len(self.shape_cache),
                "max_cache_size": self.cache_size,
                "cache_hit_rate": "N/A",  # Would need to track hits/misses
            },
        }
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.shape_cache.clear()
        self.transformations.clear()
        self.shape_counts.clear()
        self.dtype_counts.clear()
        self.device_counts.clear()
        self.task_head_shapes.clear()
        self.bev_shapes.clear()
        self.temporal_shapes.clear()
        self.total_memory_tracked = 0
        self.peak_shape_memory = 0