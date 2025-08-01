"""Data structures for PyTorch Operation Tracer"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class TensorInfo:
    """Detailed tensor information including shape and semantics"""
    shape: Tuple[int, ...]
    dtype: str = "float32"
    device: str = "cuda"
    semantic_dims: Optional[Dict[int, str]] = None  # e.g., {0: "batch", 1: "channels"}
    
    def __str__(self):
        return f"({','.join(map(str, self.shape))})"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'semantic_dims': self.semantic_dims
        }
    
    @classmethod
    def from_tuple(cls, shape: Tuple, dtype: str = "float32", device: str = "cuda"):
        """Create TensorInfo from shape tuple"""
        return cls(shape=shape, dtype=dtype, device=device)


@dataclass
class TraceNode:
    """Enhanced data structure for tracing operations in UniAD"""
    
    # Basic information
    operation: str
    module_path: str
    input_shapes: List[TensorInfo]
    output_shapes: List[TensorInfo]
    
    # UniAD-specific fields
    task_head: Optional[str] = None  # track/seg/motion/occ/planning
    temporal_index: Optional[int] = None  # frame index in queue
    is_frozen: bool = False  # for frozen BEV encoder in stage 2
    
    # Performance metrics
    memory_usage: float = 0.0  # MB
    compute_time: float = 0.0  # ms
    flops: Optional[int] = None
    
    # BEV-specific
    is_bev_operation: bool = False
    bev_grid_size: Optional[Tuple[int, int]] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    feeds_into: List[str] = field(default_factory=list)
    
    # Shape transformation tracking
    shape_transform: Optional[str] = None  # e.g., "flatten", "reshape", "permute"
    
    # Unique identifier
    node_id: str = ""
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"{self.module_path}_{id(self)}"
    
    def get_shape_change_summary(self):
        """Summarize how shapes change through this operation"""
        if not self.input_shapes or not self.output_shapes:
            return ""
        
        in_shape = self.input_shapes[0].shape
        out_shape = self.output_shapes[0].shape
        
        # Detect common transformations
        if len(in_shape) != len(out_shape):
            return f"Reshape: {in_shape} → {out_shape}"
        elif in_shape != out_shape:
            return f"Transform: {in_shape} → {out_shape}"
        else:
            return f"Preserve: {in_shape}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'operation': self.operation,
            'module_path': self.module_path,
            'input_shapes': [s.to_dict() for s in self.input_shapes],
            'output_shapes': [s.to_dict() for s in self.output_shapes],
            'task_head': self.task_head,
            'temporal_index': self.temporal_index,
            'is_frozen': self.is_frozen,
            'memory_usage': self.memory_usage,
            'compute_time': self.compute_time,
            'flops': self.flops,
            'is_bev_operation': self.is_bev_operation,
            'bev_grid_size': self.bev_grid_size,
            'depends_on': self.depends_on,
            'feeds_into': self.feeds_into,
            'shape_transform': self.shape_transform,
            'node_id': self.node_id
        }