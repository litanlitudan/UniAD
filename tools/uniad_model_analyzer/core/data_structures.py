"""
Data structures and type definitions for UniAD Model Analyzer.

This module defines the core data structures used throughout the analyzer,
including trace nodes, configuration options, and memory profiles.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import torch


class TaskHead(Enum):
    """UniAD task head types."""
    TRACK = "track"
    SEGMENTATION = "seg"
    MOTION = "motion"
    OCCUPANCY = "occ"
    PLANNING = "planning"
    NONE = "none"  # For operations not associated with a specific head


class AnalysisStage(Enum):
    """UniAD training stages."""
    STAGE_1 = 1  # Perception only
    STAGE_2 = 2  # End-to-end with frozen BEV encoder


@dataclass
class TensorShape:
    """Represents the shape and metadata of a tensor."""
    
    shape: Tuple[int, ...]  # Tensor dimensions
    dtype: torch.dtype  # Data type
    device: str  # Device (cpu, cuda:0, etc.)
    requires_grad: bool  # Whether tensor requires gradients
    memory_bytes: int  # Memory usage in bytes
    stride: Optional[Tuple[int, ...]] = None  # Memory stride
    is_contiguous: bool = True  # Whether tensor is contiguous in memory
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorShape":
        """Create TensorShape from a PyTorch tensor."""
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=str(tensor.device),
            requires_grad=tensor.requires_grad,
            memory_bytes=tensor.element_size() * tensor.numel(),
            stride=tuple(tensor.stride()) if tensor.stride() else None,
            is_contiguous=tensor.is_contiguous()
        )
    
    def __str__(self) -> str:
        """String representation of tensor shape."""
        dtype_str = str(self.dtype).replace("torch.", "")
        return f"{self.shape} {dtype_str} @{self.device}"


@dataclass
class TraceNode:
    """Represents a single operation in the traced computation graph."""
    
    # Core identification
    id: str  # Unique node identifier
    name: str  # Operation/module name
    module_path: str  # Full module path in model (e.g., "model.bev_encoder.layer1")
    
    # Timing and memory
    start_time: float  # Operation start time (nanoseconds)
    duration: float  # Execution duration (nanoseconds)
    memory_allocated: int  # Memory allocated during op (bytes)
    memory_freed: int  # Memory freed during op (bytes)
    cuda_memory_allocated: int = 0  # CUDA memory allocated (bytes)
    cuda_memory_freed: int = 0  # CUDA memory freed (bytes)
    
    # Tensor information
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shapes: List[TensorShape] = field(default_factory=list)
    
    # UniAD-specific metadata
    task_head: TaskHead = TaskHead.NONE  # Associated task head
    temporal_frame: Optional[int] = None  # Temporal frame index (0-4)
    bev_operation: bool = False  # Is BEV-related operation
    stage: Optional[AnalysisStage] = None  # Training stage
    
    # Graph structure
    parent: Optional[str] = None  # Parent node ID
    children: List[str] = field(default_factory=list)  # Child node IDs
    dependencies: List[str] = field(default_factory=list)  # Cross-branch dependencies
    
    # Additional metadata
    operation_type: Optional[str] = None  # Type of operation (conv, linear, attention, etc.)
    flops: Optional[int] = None  # Floating point operations
    parameters: Optional[int] = None  # Number of parameters in this module
    gradient_info: Optional[Dict[str, Any]] = None  # Gradient-related information
    
    def get_memory_delta(self) -> int:
        """Get net memory change for this operation."""
        return self.memory_allocated - self.memory_freed
    
    def get_cuda_memory_delta(self) -> int:
        """Get net CUDA memory change for this operation."""
        return self.cuda_memory_allocated - self.cuda_memory_freed
    
    def get_total_input_memory(self) -> int:
        """Calculate total memory of all input tensors."""
        return sum(shape.memory_bytes for shape in self.input_shapes)
    
    def get_total_output_memory(self) -> int:
        """Calculate total memory of all output tensors."""
        return sum(shape.memory_bytes for shape in self.output_shapes)


@dataclass
class MemoryProfile:
    """Memory profiling information for the model."""
    
    # Overall statistics
    peak_allocated: int  # Peak memory allocation (bytes)
    peak_reserved: int  # Peak reserved memory (bytes)
    current_allocated: int = 0  # Current allocated memory (bytes)
    current_reserved: int = 0  # Current reserved memory (bytes)
    
    # Breakdown by component
    activation_memory: Dict[str, int] = field(default_factory=dict)  # Memory for activations
    parameter_memory: Dict[str, int] = field(default_factory=dict)  # Memory for parameters
    gradient_memory: Dict[str, int] = field(default_factory=dict)  # Memory for gradients
    optimizer_memory: Dict[str, int] = field(default_factory=dict)  # Memory for optimizer states
    
    # Task head breakdown (UniAD-specific)
    task_head_memory: Dict[TaskHead, int] = field(default_factory=dict)  # Memory per task head
    
    # Temporal breakdown (UniAD-specific)
    temporal_memory: List[int] = field(default_factory=list)  # Memory per temporal frame
    
    # BEV-specific memory (UniAD-specific)
    bev_encoder_memory: int = 0  # Memory used by BEV encoder
    bev_features_memory: int = 0  # Memory for BEV feature maps
    
    # Optimization potential
    fp16_savings: int = 0  # Potential FP16 memory savings
    int8_savings: int = 0  # Potential INT8 memory savings
    pruning_savings: int = 0  # Potential pruning savings
    gradient_checkpointing_savings: int = 0  # Potential gradient checkpointing savings
    
    # Memory timeline
    memory_timeline: List[Tuple[float, int]] = field(default_factory=list)  # (timestamp, memory) pairs
    
    def get_total_memory(self) -> int:
        """Get total memory usage."""
        return self.current_allocated
    
    def get_optimization_potential(self) -> int:
        """Get total potential memory savings."""
        return self.fp16_savings + self.int8_savings + self.pruning_savings + self.gradient_checkpointing_savings
    
    def get_task_head_distribution(self) -> Dict[TaskHead, float]:
        """Get memory distribution across task heads as percentages."""
        total = sum(self.task_head_memory.values())
        if total == 0:
            return {}
        return {head: (mem / total) * 100 for head, mem in self.task_head_memory.items()}


@dataclass
class AnalysisConfig:
    """Configuration for model analysis."""
    
    # Model configuration
    model_config: str  # Path to UniAD config file
    checkpoint: Optional[str] = None  # Model checkpoint path
    stage: AnalysisStage = AnalysisStage.STAGE_1  # UniAD training stage
    
    # Tracing options
    trace_forward: bool = True  # Trace forward pass
    trace_backward: bool = False  # Trace backward pass
    trace_optimizer: bool = False  # Trace optimizer steps
    task_heads: List[TaskHead] = field(default_factory=lambda: list(TaskHead))  # Heads to trace
    max_trace_depth: int = -1  # Maximum module nesting depth (-1 for unlimited)
    
    # Analysis options
    profile_memory: bool = True  # Enable memory profiling
    track_shapes: bool = True  # Track tensor shapes
    track_dtypes: bool = True  # Track data types
    analyze_bev: bool = True  # BEV-specific analysis
    analyze_temporal: bool = True  # Temporal aggregation analysis
    temporal_frames: int = 3  # Number of temporal frames (3 or 5)
    
    # Performance options
    use_cuda: bool = True  # Use CUDA if available
    batch_size: int = 1  # Batch size for analysis
    num_iterations: int = 1  # Number of iterations to profile
    warmup_iterations: int = 0  # Warmup iterations before profiling
    
    # Visualization options
    generate_mermaid: bool = True  # Generate Mermaid diagrams
    generate_html: bool = True  # Generate HTML dashboard
    hierarchical_view: bool = True  # Enable expandable views
    max_diagram_nodes: int = 50  # Maximum nodes in diagram
    show_memory_in_diagram: bool = True  # Show memory info in diagrams
    show_shapes_in_diagram: bool = True  # Show tensor shapes in diagrams
    
    # Export options
    export_formats: List[str] = field(default_factory=lambda: ["json", "markdown"])
    output_dir: str = "./analysis_results"
    save_intermediate: bool = False  # Save intermediate analysis results
    compress_output: bool = False  # Compress large output files
    
    # Filtering options
    min_memory_mb: float = 0.1  # Minimum memory (MB) to include in analysis
    min_duration_ms: float = 0.01  # Minimum duration (ms) to include
    include_modules: List[str] = field(default_factory=list)  # Specific modules to include
    exclude_modules: List[str] = field(default_factory=list)  # Modules to exclude
    
    # UniAD-specific options
    analyze_stage_differences: bool = False  # Compare Stage 1 vs Stage 2
    verify_frozen_bev: bool = True  # Verify BEV encoder is frozen in Stage 2
    track_queue_length: bool = True  # Track memory impact of queue length
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.temporal_frames not in [3, 5]:
            raise ValueError(f"temporal_frames must be 3 or 5, got {self.temporal_frames}")
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.max_diagram_nodes < 1:
            raise ValueError(f"max_diagram_nodes must be >= 1, got {self.max_diagram_nodes}")
        
        valid_formats = ["json", "csv", "markdown", "html", "tensorboard", "onnx"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}. Must be one of {valid_formats}")


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    
    config: AnalysisConfig  # Configuration used for analysis
    trace_graph: List[TraceNode]  # Traced operation graph
    memory_profile: MemoryProfile  # Memory profiling results
    
    # Statistics
    total_operations: int = 0  # Total number of operations
    total_duration_ms: float = 0.0  # Total execution time in milliseconds
    total_parameters: int = 0  # Total number of parameters
    total_flops: int = 0  # Total floating point operations
    
    # Task head statistics (UniAD-specific)
    task_head_stats: Dict[TaskHead, Dict[str, Any]] = field(default_factory=dict)
    
    # BEV statistics (UniAD-specific)
    bev_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal statistics (UniAD-specific)
    temporal_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: Optional[str] = None
    analysis_duration_s: float = 0.0
    model_name: str = "UniAD"
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add an optimization recommendation."""
        self.recommendations.append(recommendation)