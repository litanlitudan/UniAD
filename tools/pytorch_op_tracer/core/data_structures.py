"""Data structures for PyTorch Operation Tracer"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# Supported data types in PyTorch
PYTORCH_DTYPES = {
    # Floating point types
    'float32': {'bytes': 4, 'alias': ['fp32', 'float']},
    'float16': {'bytes': 2, 'alias': ['fp16', 'half']},
    'bfloat16': {'bytes': 2, 'alias': ['bf16']},
    'float64': {'bytes': 8, 'alias': ['fp64', 'double']},
    
    # Integer types
    'int64': {'bytes': 8, 'alias': ['long']},
    'int32': {'bytes': 4, 'alias': ['int']},
    'int16': {'bytes': 2, 'alias': ['short']},
    'int8': {'bytes': 1, 'alias': ['byte']},
    'uint8': {'bytes': 1, 'alias': []},
    
    # Other types
    'bool': {'bytes': 1, 'alias': []},
    'complex64': {'bytes': 8, 'alias': []},
    'complex128': {'bytes': 16, 'alias': []},
}

# Mixed precision configurations for UniAD
UNIAD_DTYPE_CONFIGS = {
    'default': {
        'backbone': 'float32',
        'bev_encoder': 'float32',
        'task_heads': 'float32',
    },
    'mixed_precision': {
        'backbone': 'float16',  # FP16 for CNN operations
        'bev_encoder': 'float16',  # FP16 for transformer
        'task_heads': 'float32',  # FP32 for final outputs
    },
    'bfloat16': {
        'backbone': 'bfloat16',  # BF16 maintains range
        'bev_encoder': 'bfloat16',
        'task_heads': 'float32',
    },
    'int8_quantized': {
        'backbone': 'int8',  # Quantized backbone
        'bev_encoder': 'float16',
        'task_heads': 'float32',
    }
}


@dataclass
class VisualizationMetadata:
    """Metadata for interactive dataflow visualization"""
    node_id: str  # Unique identifier for visualization
    display_name: str  # Human-readable name
    position: Tuple[float, float]  # X, Y coordinates
    color: str  # Color based on operation type or memory usage
    size: float  # Node size based on importance metric
    expanded: bool = True  # Whether children are visible
    highlight: bool = False  # Whether node is highlighted
    tooltip_data: Dict[str, Any] = field(default_factory=dict)  # Data for hover tooltip


@dataclass
class TensorInfo:
    """Detailed tensor information including shape, dtype, and semantics"""
    shape: Tuple[int, ...]
    dtype: str = "float32"  # float32, float16, bfloat16, int8, int32, bool, etc.
    device: str = "cuda"
    semantic_dims: Optional[Dict[int, str]] = None  # e.g., {0: "batch", 1: "channels"}
    requires_grad: bool = True
    is_quantized: bool = False
    memory_bytes: Optional[int] = None  # Actual memory usage considering dtype
    
    def __str__(self):
        dtype_str = self.dtype.replace('float', 'fp').replace('bfloat', 'bf')
        return f"[{','.join(map(str, self.shape))}]@{dtype_str}"
    
    def memory_size(self) -> int:
        """Calculate memory size in bytes based on shape and dtype"""
        if self.memory_bytes is not None:
            return self.memory_bytes
        
        # Calculate based on dtype
        dtype_bytes = {
            'float32': 4, 'float': 4, 'fp32': 4,
            'float16': 2, 'half': 2, 'fp16': 2,
            'bfloat16': 2, 'bf16': 2,
            'float64': 8, 'double': 8, 'fp64': 8,
            'int64': 8, 'long': 8,
            'int32': 4, 'int': 4,
            'int16': 2, 'short': 2,
            'int8': 1, 'byte': 1,
            'uint8': 1,
            'bool': 1
        }
        
        bytes_per_element = dtype_bytes.get(self.dtype.lower(), 4)
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        
        return num_elements * bytes_per_element
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'semantic_dims': self.semantic_dims,
            'requires_grad': self.requires_grad,
            'is_quantized': self.is_quantized,
            'memory_bytes': self.memory_bytes,
            'memory_mb': self.memory_size() / (1024 * 1024)
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
    
    # Visualization metadata (optional)
    visualization_metadata: Optional[VisualizationMetadata] = None
    
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
        result = {
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
        
        # Add visualization metadata if present
        if self.visualization_metadata:
            result['visualization_metadata'] = {
                'node_id': self.visualization_metadata.node_id,
                'display_name': self.visualization_metadata.display_name,
                'position': self.visualization_metadata.position,
                'color': self.visualization_metadata.color,
                'size': self.visualization_metadata.size,
                'expanded': self.visualization_metadata.expanded,
                'highlight': self.visualization_metadata.highlight,
                'tooltip_data': self.visualization_metadata.tooltip_data
            }
        
        return result