# PyTorch Operation Tracer and Visualizer for UniAD

This document outlines the architecture design for a script that traces PyTorch operations, records tensor shapes, and visualizes the dataflow using Mermaid diagrams, with specific enhancements for the UniAD multi-task autonomous driving framework.

## 1. Overview

The PyTorch Operation Tracer is designed to help developers understand the flow of tensors through neural network models, with specialized support for UniAD's hierarchical multi-task architecture:

1. Tracing all PyTorch operations during model execution
2. Recording input and output tensor shapes for each operation
3. Visualizing the dataflow as a Mermaid diagram with task-specific views
4. Providing detailed analysis of the model's computational graph
5. **UniAD-specific**: Tracking multi-head interactions, temporal queues, and BEV feature flow
6. **Memory profiling**: Essential for UniAD's 30-50GB GPU memory requirements

## 2. Architecture Components

### 2.1 Core Components

The system consists of the following core components:

1. **OperationTracer**: Hooks into PyTorch modules to trace operations
2. **TensorShapeRecorder**: Records shapes of input/output tensors
3. **DataflowVisualizer**: Generates Mermaid diagrams from trace data
4. **TraceAnalyzer**: Analyzes the trace data for insights
5. **CommandLineInterface**: Provides a user-friendly interface

#### UniAD-Specific Components:

6. **MultiHeadTracer**: Traces UniAD's five task heads (track, seg, motion, occ, planning) and their interactions
7. **TemporalTracer**: Handles temporal queue operations (3-5 frames)
8. **BEVFeatureTracer**: Specialized for BEV encoder/decoder transformations
9. **MemoryProfiler**: Tracks GPU memory usage patterns (critical for 30-50GB requirements)
10. **TaskDependencyAnalyzer**: Analyzes dependencies between task heads

### 2.2 Component Interactions

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ CommandLineUI   │────▶│ OperationTracer   │────▶│ TensorShapeRecorder │
│ --show-shapes   │     │ Track modules     │     │ Shape: [B,C,H,W]    │
│ --expand-modules│     │ Hook operations   │     │ Semantics tracking  │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
                                 │                            │
                                 ├──────────┐                 │
                                 ▼          ▼                 ▼
                         ┌─────────────┐ ┌──────────────┐ ┌───────────────┐
                         │MultiHeadTrace│ │TemporalTrace │ │ Trace Data    │
                         │5 task heads │ │Queue shapes  │ │ w/ shapes     │
                         └─────────────┘ └──────────────┘ └───────────────┘
                                 │          │                 │
                                 ▼          ▼                 ▼
                         ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
                         │ TraceAnalyzer │ │MemoryProfile│ │BEVFeatureTrace│
                         │ Shape changes │ │ MB per tensor│ │[256,200,200]  │
                         └───────────────┘ └──────────────┘ └──────────────┘
                                 │                            ▲
                                 ▼                            │
                        ┌────────────────┐                    │
                        │ DataflowVisual │────────────────────┘
                        │ Mermaid + shapes│
                        └────────────────┘
```

## 3. Detailed Component Design

### 3.1 OperationTracer

The OperationTracer uses PyTorch hooks to intercept forward and backward passes through the model:

- **Module Hooks**: Register hooks on PyTorch modules
- **Function Hooks**: Register hooks on PyTorch autograd functions
- **Tensor Hooks**: Track tensor operations

Key features:
- Track execution order of operations
- Identify connections between operations
- Support for custom filtering of operations

```python
class OperationTracer:
    def __init__(self, model, trace_backward=False, filter_ops=None,
                 stage=2, task_heads=None):
        # Initialize tracer with UniAD-specific options
        self.stage = stage  # Stage 1 or 2
        self.task_heads = task_heads or ['track', 'seg', 'motion', 'occ', 'planning']

    def register_hooks(self):
        # Register hooks on model modules

    def trace(self, inputs):
        # Perform tracing with given inputs

    def get_trace_data(self):
        # Return collected trace data with task head annotations
```

### 3.2 TensorShapeRecorder

Records the shapes of tensors at each operation:

- Input tensor shapes (with dimension semantics)
- Output tensor shapes (with dimension semantics)
- Parameter shapes
- Shape transformations and dimension changes

Handles various tensor containers (lists, tuples, dictionaries).

#### Enhanced Data Structure for UniAD:

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

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

@dataclass
class TraceNode:
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
```

#### Shape Semantics for UniAD

```python
# Common dimension semantics in UniAD
UNIAD_DIM_SEMANTICS = {
    "image": {0: "batch", 1: "num_cams", 2: "channels", 3: "height", 4: "width"},
    "bev": {0: "batch", 1: "channels", 2: "bev_h", 3: "bev_w"},
    "query": {0: "batch", 1: "num_queries", 2: "embed_dims"},
    "temporal": {0: "batch", 1: "num_frames", 2: "channels", 3: "height", 4: "width"},
    "motion": {0: "batch", 1: "num_agents", 2: "num_modes", 3: "coords", 4: "timesteps"},
    "planning": {0: "batch", 1: "timesteps", 2: "coords"},
}
```

#### Data Type (dtype) Support

The tracer supports comprehensive data type tracking for all tensors:

```python
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
```

##### Memory Impact of Data Types

Different data types significantly affect memory usage:

```python
def calculate_tensor_memory(shape: Tuple[int, ...], dtype: str) -> float:
    """Calculate memory usage in MB for a tensor"""
    dtype_info = PYTORCH_DTYPES.get(dtype, PYTORCH_DTYPES['float32'])
    bytes_per_element = dtype_info['bytes']
    
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    memory_bytes = num_elements * bytes_per_element
    memory_mb = memory_bytes / (1024 * 1024)
    
    return memory_mb

# Example: BEV features memory comparison
bev_shape = (1, 256, 200, 200)  # Typical BEV feature shape
print(f"FP32: {calculate_tensor_memory(bev_shape, 'float32'):.1f} MB")
print(f"FP16: {calculate_tensor_memory(bev_shape, 'float16'):.1f} MB")
print(f"BF16: {calculate_tensor_memory(bev_shape, 'bfloat16'):.1f} MB")
print(f"INT8: {calculate_tensor_memory(bev_shape, 'int8'):.1f} MB")
# Output:
# FP32: 39.1 MB
# FP16: 19.5 MB
# BF16: 19.5 MB
# INT8: 9.8 MB
```

### 3.3 DataflowVisualizer

Generates visualizations from the trace data:

- **Mermaid Diagram Generator**: Creates Mermaid syntax for dataflow
- **Graph Simplification**: Simplifies complex graphs for readability
- **Hierarchical Grouping**: Groups operations by module hierarchy
- **Module Expansion**: Default top-level view with expandable modules

Features:
- Customizable node appearance
- Filtering options for large models
- Support for exporting to various formats
- **Hierarchical visualization with module expansion**
- **Interactive drill-down into specific modules**

#### 3.3.1 Hierarchical Visualization

The visualizer supports two modes:

1. **Top-Level View (Default)**: Shows only the high-level modules and their connections
2. **Expanded Module View**: Shows detailed operations within a specific module

```python
class DataflowVisualizer:
    def __init__(self, trace_data, visualization_mode='top-level', 
                 expand_modules=None, show_shapes=True, shape_format='full'):
        """
        Args:
            trace_data: Collected trace data
            visualization_mode: 'top-level' or 'expanded'
            expand_modules: List of module names to expand (e.g., ['BEVFormer', 'TrackHead'])
            show_shapes: Whether to display tensor shapes
            shape_format: 'full' (all dims), 'compact' (abbreviated), 'semantic' (with labels)
        """
        self.trace_data = trace_data
        self.visualization_mode = visualization_mode
        self.expand_modules = expand_modules or []
        self.show_shapes = show_shapes
        self.shape_format = shape_format
        
    def generate_mermaid(self):
        if self.visualization_mode == 'top-level':
            return self._generate_top_level_view()
        else:
            return self._generate_expanded_view()
    
    def _format_tensor_shape(self, tensor_info, context=None):
        """Format tensor shape for display"""
        if not self.show_shapes:
            return ""
        
        shape = tensor_info.shape
        
        if self.shape_format == 'compact':
            # Compact format: [B, C, H, W] → [1, 256, 200, 200]
            return f"[{','.join(map(str, shape))}]"
        
        elif self.shape_format == 'semantic' and tensor_info.semantic_dims:
            # Semantic format: [batch=1, channels=256, bev_h=200, bev_w=200]
            parts = []
            for i, dim in enumerate(shape):
                label = tensor_info.semantic_dims.get(i, f"dim{i}")
                parts.append(f"{label}={dim}")
            return f"[{', '.join(parts)}]"
        
        else:  # 'full' format
            # Full format with context
            shape_str = f"[{', '.join(map(str, shape))}]"
            if context:
                shape_str += f" ({context})"
            return shape_str
    
    def _generate_module_node(self, module_name, module_info):
        """Generate node text for a module"""
        lines = [module_name]
        
        if self.show_shapes and module_info.get('input_shape'):
            lines.append(f"Input: {self._format_tensor_shape(module_info['input_shape'])}")
        
        if self.show_shapes and module_info.get('output_shape'):
            lines.append(f"Output: {self._format_tensor_shape(module_info['output_shape'])}")
        
        if module_info.get('memory'):
            lines.append(f"Memory: {module_info['memory']:.1f}GB")
        
        if module_info.get('extra_info'):
            lines.append(module_info['extra_info'])
        
        return "<br/>".join(lines)
    
    def _generate_top_level_view(self):
        """
        Generate a high-level view showing only major modules:
        - BEVFormer Encoder
        - Task Heads (Track, Seg, Motion, Occ, Planning)
        - Major connections between modules
        - Aggregated tensor shapes
        """
        # Group operations by top-level module
        # Show aggregated information (total memory, FLOPs, etc.)
        # Display input/output shapes for each major module
        pass
    
    def _generate_expanded_view(self):
        """
        Generate detailed view for specified modules:
        - All operations within the module
        - Detailed tensor shapes for each operation
        - Shape transformations between operations
        - Memory usage per operation
        """
        # Expand specified modules while keeping others collapsed
        # Show shape changes through the module
        pass
```

#### 3.3.2 Module Expansion Examples

**Top-Level View (Default)**:
```mermaid
graph TB
    Input["Multi-View Images<br/>Shape: (1, 6, 3, 928, 1600)@fp32"] --> BEVFormer["BEVFormer Encoder<br/>In: (1, 6, 3, 928, 1600)@fp32<br/>Out: (1, 256, 200, 200)@fp32<br/>Memory: 15.2GB<br/>6 layers"]
    BEVFormer --> BEV["BEV Features<br/>Shape: (1, 256, 200, 200)@fp32"]
    
    BEV --> TrackHead["Track Head<br/>In: (1, 256, 200, 200)@fp32<br/>Out: (1, 900, 266)@fp32<br/>Memory: 8GB<br/>900 queries"]
    BEV --> SegHead["Seg Head<br/>In: (1, 256, 200, 200)@fp32<br/>Out: (1, 3, 200, 200)@fp32<br/>Memory: 5GB"]
    
    TrackHead --> MotionHead["Motion Head<br/>In: (1, N, 256)@fp32<br/>Out: (1, N, 6, 2, 6)@fp32<br/>Memory: 4GB"]
    TrackHead --> OccHead["Occ Head<br/>In: (1, N, 256)@fp32<br/>Out: (1, 200, 200, 5)@fp32<br/>Memory: 6GB"]
    
    MotionHead --> PlanHead["Planning Head<br/>In: [(1,N,6,2,6)@fp32, (1,200,200,5)@fp32, (1,N,256)@fp32]<br/>Out: (1, 6, 2)@fp32<br/>Memory: 3GB"]
    OccHead --> PlanHead
    TrackHead -.-> |"Agent Features<br/>(1, N, 256)@fp32"| PlanHead
```

**Expanded BEVFormer View**:
```mermaid
graph TB
    subgraph "BEVFormer Encoder (Expanded)"
        Input["Multi-View Images<br/>(1, 6, 3, 928, 1600)"] --> ResNet["ResNet-101<br/>In: (1, 6, 3, 928, 1600)<br/>Out: (1, 6, 2048, 29, 50)<br/>Frozen"]
        ResNet --> FPN["FPN<br/>In: (1, 6, 2048, 29, 50)<br/>Out: 4 levels<br/>(1, 6, 256, H, W)"]
        
        subgraph "Layer 1"
            FPN --> TSA1["Temporal Self-Attention<br/>In/Out: (1, 40000, 256)<br/>8 heads, 4 points"]
            TSA1 --> SCA1["Spatial Cross-Attention<br/>Query: (1, 40000, 256)<br/>Key/Value: (6, H×W, 256)<br/>Out: (1, 40000, 256)"]
            SCA1 --> FFN1["Feed-Forward<br/>In: (1, 40000, 256)<br/>Hidden: (1, 40000, 2048)<br/>Out: (1, 40000, 256)"]
        end
        
        subgraph "Layer 2-6"
            FFN1 --> Layers["Layers 2-6<br/>In/Out: (1, 40000, 256)<br/>Similar structure"]
        end
        
        Layers --> BEVOut["BEV Output<br/>Reshape: (1, 40000, 256)<br/>→ (1, 256, 200, 200)"]
    end
```

**Mixed Precision View**:
```mermaid
graph TB
    Input["Multi-View Images<br/>Shape: (1, 6, 3, 928, 1600)@fp32"] --> BEVFormer["BEVFormer Encoder<br/>In: (1, 6, 3, 928, 1600)@fp32<br/>Out: (1, 256, 200, 200)@fp16<br/>Memory: 7.6GB (↓50%)<br/>Mixed Precision"]
    BEVFormer --> BEV["BEV Features<br/>Shape: (1, 256, 200, 200)@fp16"]
    
    BEV --> TrackHead["Track Head<br/>In: (1, 256, 200, 200)@fp16<br/>Out: (1, 900, 266)@fp32<br/>Memory: 5GB<br/>FP32 Output"]
    BEV --> SegHead["Seg Head<br/>In: (1, 256, 200, 200)@fp16<br/>Out: (1, 3, 200, 200)@fp32<br/>Memory: 3GB"]
    
    TrackHead --> MotionHead["Motion Head<br/>In: (1, N, 256)@fp32<br/>Out: (1, N, 6, 2, 6)@fp32<br/>Memory: 4GB"]
    TrackHead --> OccHead["Occ Head<br/>In: (1, N, 256)@fp32<br/>Out: (1, 200, 200, 5)@fp16<br/>Memory: 3GB (↓50%)"]
    
    MotionHead --> PlanHead["Planning Head<br/>In: [(1,N,6,2,6)@fp32, (1,200,200,5)@fp16, (1,N,256)@fp32]<br/>Out: (1, 6, 2)@fp32<br/>Memory: 3GB"]
    OccHead --> PlanHead
    TrackHead -.-> |"Agent Features<br/>(1, N, 256)@fp32"| PlanHead
    
    style BEVFormer fill:#ffcccc,stroke:#333,stroke-width:2px
    style BEV fill:#ffcccc,stroke:#333,stroke-width:2px
    style OccHead fill:#ffcccc,stroke:#333,stroke-width:2px
```

### 3.4 TraceAnalyzer

Analyzes the trace data to provide insights:

- Memory usage estimation
- Computational complexity analysis
- Bottleneck identification
- Layer-wise timing analysis

### 3.5 CommandLineInterface

Provides a user-friendly interface:

```
python tools/analysis_tools/trace_pytorch_ops.py \
    --config CONFIG_PATH \
    --checkpoint CHECKPOINT_PATH \
    --input-shape 1,3,224,224 \
    --output mermaid_diagram.md
```

Options:
- Model configuration
- Input specifications
- Visualization options
- Analysis preferences
- **Module expansion controls**

#### 3.5.1 Visualization Options

```bash
# Default top-level view
python tools/analysis_tools/trace_pytorch_ops.py \
    --config CONFIG_PATH \
    --checkpoint CHECKPOINT_PATH \
    --visualization-mode top-level \
    --output top_level_view.md

# Expand specific modules
python tools/analysis_tools/trace_pytorch_ops.py \
    --config CONFIG_PATH \
    --checkpoint CHECKPOINT_PATH \
    --visualization-mode expanded \
    --expand-modules BEVFormer,TrackHead \
    --output expanded_view.md

# Show all details (full expansion)
python tools/analysis_tools/trace_pytorch_ops.py \
    --config CONFIG_PATH \
    --checkpoint CHECKPOINT_PATH \
    --visualization-mode full \
    --output full_details.md
```

#### 3.5.2 Module Selection Patterns

```bash
# Expand by module type
--expand-modules "*Head"  # All task heads
--expand-modules "BEV*"   # All BEV-related modules

# Expand by specific names
--expand-modules "BEVFormer,MotionHead,PlanningHead"

# Expand by depth level
--expand-depth 2  # Show 2 levels deep from top

# Memory-based expansion
--expand-heavy-modules  # Auto-expand modules using >1GB memory
```

#### 3.5.3 Tensor Shape Display Options

```bash
# Shape display control
--show-shapes              # Enable shape display (default: true)
--no-shapes               # Disable shape display
--shape-format full       # Full format: [1, 256, 200, 200]
--shape-format compact    # Compact: [1,256,200,200]
--shape-format semantic   # Semantic: [batch=1, channels=256, bev_h=200, bev_w=200]

# Combined examples
python trace_pytorch_ops.py \
    --config uniad.py \
    --visualization-mode expanded \
    --expand-modules TrackHead \
    --shape-format semantic \
    --output track_head_shapes.md

# Focus on shape transformations
python trace_pytorch_ops.py \
    --config uniad.py \
    --track-shape-changes \
    --highlight-reshapes \
    --output shape_analysis.md

# Memory and shape correlation
python trace_pytorch_ops.py \
    --config uniad.py \
    --show-shapes \
    --annotate-memory-per-element \
    --output memory_shape_analysis.md
```

### 3.6 UniAD-Specific Components

#### 3.6.1 MultiHeadTracer

Traces interactions between UniAD's five task heads:

```python
class MultiHeadTracer:
    def __init__(self, task_heads=['track', 'seg', 'motion', 'occ', 'planning']):
        self.task_heads = task_heads
        self.head_dependencies = {
            'motion': ['track'],
            'occ': ['track'],
            'planning': ['track', 'motion', 'occ']  # Note: track provides agent features directly
        }

    def trace_task_heads(self, model):
        # Track data flow between task heads
        # Identify inter-head dependencies
        pass
```

#### 3.6.2 TemporalTracer

Handles temporal queue operations for multi-frame processing:

```python
class TemporalTracer:
    def __init__(self, queue_length=3):
        self.queue_length = queue_length

    def trace_temporal_flow(self, bev_features):
        # Track how features aggregate over frames
        # Visualize temporal fusion operations
        pass
```

#### 3.6.3 BEVFeatureTracer

Specialized tracer for BEV transformations:

```python
class BEVFeatureTracer:
    def __init__(self):
        self.bev_shape = (200, 200)  # BEV grid size
        self.feature_dim = 256

    def trace_bev_operations(self, encoder, decoder):
        # Track BEV encoder operations
        # Monitor feature propagation through decoder
        # Identify frozen vs trainable components
        pass
```

#### 3.6.4 MemoryProfiler

Profile GPU memory usage:

```python
class MemoryProfiler:
    def __init__(self, stage=2):
        self.stage = stage
        self.expected_usage = {1: 30000, 2: 17000}  # MB

    def profile_gpu_memory(self):
        # Track memory allocation per operation
        # Identify memory bottlenecks
        # Compare against expected usage
        pass
```

## 4. Implementation Plan

### 4.1 Phase 1: Core Tracing with UniAD Awareness

1. Implement basic module hooks with task head detection
2. Record tensor shapes with BEV grid awareness
3. Track execution order and task dependencies
4. Add stage-specific tracing (Stage 1 vs Stage 2)

### 4.2 Phase 2: Temporal and Multi-Head Support

1. Implement temporal queue tracing
2. Add multi-head interaction tracking
3. Visualize task hierarchy (perception → prediction → planning)
4. Track frozen vs trainable components

### 4.3 Phase 3: Memory Profiling and Optimization

1. Implement GPU memory profiling
2. Identify memory bottlenecks (critical for 30-50GB usage)
3. Add memory-efficient tracing modes
4. Compare memory usage between stages

### 4.4 Phase 4: Advanced Visualization

1. Generate task-aware Mermaid diagrams
2. Add temporal dimension visualization
3. Create memory heatmaps
4. Implement hierarchical task views

### 4.5 Phase 5: Analysis and Optimization

1. Task dependency analysis
2. Performance bottleneck identification
3. Stage comparison tools
4. Optimization recommendations

## 5. Usage Examples

### 5.1 Basic Usage

```bash
# Trace UniAD Stage 1 (perception only)
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage1_track_map/base_track_map.py \
    --checkpoint ckpts/uniad_base_track_map.pth \
    --stage 1 \
    --show-shapes

# Trace UniAD Stage 2 (end-to-end)
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --stage 2 \
    --shape-format semantic
```

### 5.2 Advanced Usage

```bash
# Trace specific task heads with memory profiling and shapes
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --task-heads track,motion,planning \
    --temporal-frames 3 \
    --memory-profile \
    --show-shapes \
    --shape-format semantic \
    --output uniad_trace_analysis.md

# Trace with dtype tracking
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --track-dtype \
    --show-dtype \
    --output uniad_dtype_analysis.md

# Trace with mixed precision analysis
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --mixed-precision fp16 \
    --dtype-memory-analysis \
    --output uniad_mixed_precision.md

# Focus on BEV operations with shape tracking
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --bev-focus \
    --filter-ops BEVFormer,BEVEncoder,BEVDecoder \
    --visualize-temporal \
    --track-shape-changes \
    --output bev_flow.md

# Compare Stage 1 and Stage 2 with shape analysis
python tools/analysis_tools/trace_pytorch_ops.py \
    --compare-stages \
    --stage1-config projects/configs/stage1_track_map/base_track_map.py \
    --stage1-ckpt ckpts/uniad_base_track_map.pth \
    --stage2-config projects/configs/stage2_e2e/base_e2e.py \
    --stage2-ckpt ckpts/uniad_base_e2e.pth \
    --show-shapes \
    --output stage_comparison.md
```

## 6. Challenges and Considerations

### 6.1 Performance Impact

- Tracing adds overhead to model execution
- Memory usage can be significant for large models

### 6.2 Complex Models

- Handling recurrent connections
- Dealing with dynamic control flow
- Supporting custom PyTorch extensions

### 6.3 Visualization Complexity

- Large models produce complex diagrams
- Need effective simplification strategies
- Balance between detail and readability

### 6.4 UniAD-Specific Challenges

- **Memory Management**: Handling 30-50GB GPU memory requirements
- **Temporal Complexity**: Visualizing multi-frame aggregation
- **Task Dependencies**: Tracking complex inter-head dependencies
- **Stage Differences**: Different architectures between Stage 1 and 2

## 7. Sample Visualization Output

### 7.1 Top-Level View (Default)

```mermaid
graph TB
    %% Default view showing only major modules with tensor shapes
    Input["Multi-View Images<br/>Input: (1, 6, 3, 928, 1600)<br/>Memory: 0.1GB"] --> Backbone["ResNet-101 + FPN<br/>Output: (1, 6, 256, 116, 200)<br/>Frozen: Stage 2<br/>Memory: 2.5GB"]
    
    Backbone --> BEVEncoder["BEVFormer Encoder<br/>Input: (1, 6, 256, H, W) multi-scale<br/>Output: (1, 256, 200, 200)<br/>6 layers<br/>Memory: 15.2GB"]
    
    BEVEncoder --> BEVFeatures["BEV Features<br/>Shape: (1, 256, 200, 200)<br/>Memory: 0.4GB"]
    
    BEVFeatures --> TrackHead["Track Head<br/>Input: (1, 256, 200, 200)<br/>Output: (1, 900, 10+256)<br/>900 queries<br/>Memory: 8GB"]
    
    BEVFeatures --> SegHead["Seg Head<br/>Input: (1, 256, 200, 200)<br/>Output: (1, 3, 200, 200)<br/>3 classes<br/>Memory: 5GB"]
    
    TrackHead --> MotionHead["Motion Head<br/>Input: (1, N, 256)<br/>Output: (1, N, 6, 2, 6)<br/>6 modes, 6 timesteps<br/>Memory: 4GB"]
    
    TrackHead --> OccHead["Occ Head<br/>Input: (1, N, 256)<br/>Output: (1, 200, 200, 5)<br/>Future: 3s<br/>Memory: 6GB"]
    
    MotionHead --> PlanHead["Planning Head<br/>Input: [(1,N,6,2,6), (1,200,200,5), (1,N,256)]<br/>Output: (1, 6, 2)<br/>Trajectory: 3s<br/>Memory: 3GB"]
    OccHead --> PlanHead
    TrackHead -.-> |"Agent Features<br/>(1, N, 256)"| PlanHead
    
    %% Style for different module types
    style Backbone fill:#e1e1e1,stroke:#333,stroke-width:2px
    style BEVEncoder fill:#ffcc99,stroke:#333,stroke-width:2px
    style TrackHead fill:#ff9999,stroke:#333,stroke-width:2px
    style MotionHead fill:#9999ff,stroke:#333,stroke-width:2px
    style PlanHead fill:#99ff99,stroke:#333,stroke-width:2px
```

### 7.2 Expanded Track Head View

```mermaid
graph TB
    %% Expanded view of Track Head module with tensor shapes
    BEVFeatures["BEV Features<br/>Shape: (1, 256, 200, 200)"] --> TrackTransformer
    
    subgraph "Track Head (Expanded)"
        TrackTransformer["Track Transformer<br/>Input: (1, 256, 200, 200)<br/>6 layers"]
        
        subgraph "Decoder Layer 1"
            TrackTransformer --> SelfAttn1["Self-Attention<br/>Input: (1, 900, 256)<br/>Output: (1, 900, 256)<br/>8 heads"]
            SelfAttn1 --> CrossAttn1["Cross-Attention<br/>Query: (1, 900, 256)<br/>Key/Value: (1, 40000, 256)<br/>Output: (1, 900, 256)"]
            CrossAttn1 --> FFN1["FFN<br/>Input: (1, 900, 256)<br/>Hidden: (1, 900, 2048)<br/>Output: (1, 900, 256)"]
        end
        
        FFN1 --> MoreLayers["Layers 2-6<br/>Input/Output: (1, 900, 256)<br/>Similar structure"]
        
        MoreLayers --> ClassHead["Classification<br/>Input: (1, 900, 256)<br/>Output: (1, 900, 10)<br/>10 classes"]
        MoreLayers --> BoxHead["Box Regression<br/>Input: (1, 900, 256)<br/>Output: (1, 900, 10)<br/>3D boxes"]
        MoreLayers --> TrackingHead["Tracking<br/>Input: (1, 900, 256)<br/>Output: (1, 900, 256)<br/>Instance embeddings"]
        
        ClassHead --> NMS["NMS<br/>Input: (1, 900, 10+10)<br/>Output: (1, N, 10+10)<br/>N ≤ 300"]
        BoxHead --> NMS
        TrackingHead --> TrackOutput["Track Results<br/>Shape: (1, N, 10+10+256)<br/>Memory: 0.5GB"]
        NMS --> TrackOutput
    end
```

### 7.3 Expanded Planning Head View

```mermaid
graph TB
    %% Expanded view of Planning Head with tensor shapes
    subgraph "Inputs to Planning"
        TrackResults["Agent Features (Track Head)<br/>Shape: (1, N, 256)<br/>N detected objects"]
        MotionPred["Motion Predictions<br/>Shape: (1, N, 6, 2, 6)<br/>6 modes × 6 timesteps"]
        OccPred["Occupancy Predictions<br/>Shape: (1, 200, 200, 5)<br/>5 future frames"]
    end
    
    subgraph "Planning Head (Expanded)"
        TrackResults --> FeatureAgg["Feature Aggregation<br/>Input: [(1,N,256), (1,N,6,2,6), (1,200,200,5)]<br/>Output: (1, 256)"]
        MotionPred --> FeatureAgg
        OccPred --> FeatureAgg
        
        FeatureAgg --> PlanningGRU["Planning GRU<br/>Input: (1, 1, 256)<br/>Hidden: (1, 1, 256)<br/>6 unrolls"]
        
        PlanningGRU --> TrajDecoder["Trajectory Decoder<br/>Input: (1, 6, 256)<br/>Output: (1, 6, 2)<br/>6 timesteps × (x,y)"]
        
        TrajDecoder --> CollisionCheck["Collision Checker<br/>Input: Traj(1,6,2) + Occ(1,200,200,5)<br/>Output: (1, 6, 3)<br/>(x,y,collision_prob)"]
        
        CollisionCheck --> FinalTraj["Final Trajectory<br/>Shape: (1, 6, 2)<br/>6 waypoints @ 0.5s"]
    end
    
    style FeatureAgg fill:#ffeecc,stroke:#333,stroke-width:2px
    style PlanningGRU fill:#ccffcc,stroke:#333,stroke-width:2px
    style CollisionCheck fill:#ffcccc,stroke:#333,stroke-width:2px
```

### 7.4 Memory-Based Auto-Expansion View

```mermaid
graph TB
    %% Auto-expanded view showing modules >5GB memory with tensor shapes
    Input["Input<br/>Shape: (1, 6, 3, 928, 1600)"] --> Backbone["Backbone<br/>Output: (1, 6, 256, H, W)<br/>2.5GB"]
    
    Backbone --> BEVEncoder
    
    subgraph "BEVFormer Encoder (15.2GB) - Expanded"
        BEVEncoder["BEVFormer Entry<br/>Input: (1, 6, 256, H, W)"] --> TSA["Temporal Self-Attn<br/>Input: (1, 40000, 256)<br/>Output: (1, 40000, 256)<br/>6 layers<br/>Memory: 7GB"]
        TSA --> SCA["Spatial Cross-Attn<br/>Query: (1, 40000, 256)<br/>Key/Value: (6, H×W, 256)<br/>Output: (1, 40000, 256)<br/>6 layers<br/>Memory: 8.2GB"]
    end
    
    SCA --> BEVFeatures["BEV Features<br/>Shape: (1, 256, 200, 200)"]
    
    subgraph "Track Head (8GB) - Expanded"
        BEVFeatures --> TrackDec["Track Decoder<br/>Input: (1, 256, 200, 200)<br/>Output: (1, 900, 256)<br/>Memory: 5GB"]
        TrackDec --> TrackPost["Post-processing<br/>Input: (1, 900, 266)<br/>Output: (1, N, 266)<br/>Memory: 3GB"]
    end
    
    TrackPost --> SmallModules["Other Modules<br/>(< 5GB each)"]
    
    subgraph "Occ Head (6GB) - Expanded"
        BEVFeatures --> OccConv["Occ Conv Layers<br/>Input: (1, 256, 200, 200)<br/>Output: (1, 128, 200, 200)<br/>Memory: 4GB"]
        OccConv --> OccPredict["Occ Prediction<br/>Input: (1, 128, 200, 200)<br/>Output: (1, 200, 200, 5)<br/>Memory: 2GB"]
    end
    
    %% Collapsed modules (< 5GB) with shapes
    BEVFeatures --> SegHead["Seg Head<br/>In: (1, 256, 200, 200)<br/>Out: (1, 3, 200, 200)<br/>5GB"]
    SmallModules --> MotionHead["Motion Head<br/>In: (1, N, 256)<br/>Out: (1, N, 6, 2, 6)<br/>4GB"]
    SmallModules --> PlanHead["Plan Head<br/>In: [(1,N,6,2,6), (1,200,200,5)]<br/>Out: (1, 6, 2)<br/>3GB"]
```

### 7.5 Shape Transformation View

```mermaid
graph TB
    %% Shape transformations through BEVFormer encoder
    subgraph "BEVFormer Shape Transformations"
        Input["Multi-View Images<br/>(1, 6, 3, 928, 1600)"]
        
        Input --> Conv1["ResNet Conv1<br/>Transform: (1,6,3,928,1600)<br/>→ (1,6,64,464,800)"]
        
        Conv1 --> MaxPool["MaxPool<br/>Transform: (1,6,64,464,800)<br/>→ (1,6,64,232,400)"]
        
        MaxPool --> ResBlocks["ResNet Blocks<br/>Transform: (1,6,64,232,400)<br/>→ (1,6,2048,29,50)"]
        
        ResBlocks --> FPN["FPN<br/>Multi-scale outputs:<br/>(1,6,256,29,50)<br/>(1,6,256,58,100)<br/>(1,6,256,116,200)<br/>(1,6,256,232,400)"]
        
        FPN --> Flatten["Flatten & Concat<br/>Transform: 4 scales<br/>→ (1, 120000, 256)"]
        
        Flatten --> BEVQuery["BEV Queries<br/>Transform: (1, 120000, 256)<br/>→ (1, 40000, 256)"]
        
        BEVQuery --> TSA["Temporal Self-Attn<br/>Preserve: (1, 40000, 256)"]
        
        TSA --> SCA["Spatial Cross-Attn<br/>Preserve: (1, 40000, 256)"]
        
        SCA --> Reshape["Reshape to Grid<br/>Transform: (1, 40000, 256)<br/>→ (1, 256, 200, 200)"]
        
        style Input fill:#e1e1e1
        style Reshape fill:#99ff99
        style FPN fill:#ffcc99
    end
```

### 7.6 Complete UniAD Tensor Flow

```mermaid
graph TB
    %% Complete tensor shape flow through UniAD pipeline
    subgraph "Input Processing"
        Img["6 Camera Images<br/>(1, 6, 3, 928, 1600)@uint8<br/>13.4 MB each"]
        CAN["CAN Bus Data<br/>(1, 18)@fp32<br/>Ego motion"]
    end
    
    subgraph "Feature Extraction"
        Img --> Backbone["ResNet-101<br/>Transform: (1,6,3,928,1600)@uint8<br/>→ (1,6,2048,29,50)@fp32"]
        Backbone --> FPN["FPN<br/>Multi-scale:<br/>(1,6,256,29,50)@fp32<br/>(1,6,256,58,100)@fp32<br/>(1,6,256,116,200)@fp32<br/>(1,6,256,232,400)@fp32"]
    end
    
    subgraph "BEV Generation"
        FPN --> BEVEnc["BEVFormer<br/>6 layers<br/>(1,40000,256)@fp32"]
        CAN --> BEVEnc
        BEVEnc --> BEVGrid["BEV Grid<br/>(1,256,200,200)@fp32<br/>39.1 MB"]
    end
    
    subgraph "Perception Tasks"
        BEVGrid --> Track["Track Head<br/>Output: (1,N,266)@fp32<br/>N objects"]
        BEVGrid --> Seg["Seg Head<br/>Output: (1,3,200,200)@fp32<br/>3 classes"]
    end
    
    subgraph "Prediction Tasks"
        Track --> Motion["Motion Head<br/>Input: (1,N,266)@fp32<br/>Output: (1,N,6,2,6)@fp32<br/>6 modes"]
        Track --> Occ["Occ Head<br/>Input: (1,N,266)@fp32<br/>Output: (1,200,200,5)@fp32<br/>5 frames"]
    end
    
    subgraph "Planning Task"
        Motion --> Plan["Planning Head<br/>Inputs: [(1,N,6,2,6), (1,200,200,5), (1,N,266)]<br/>Output: (1,6,2)<br/>3s trajectory"]
        Occ --> Plan
        Track --> Plan
    end
    
    style Img fill:#e1e1e1
    style BEVGrid fill:#ffcc99
    style Plan fill:#99ff99
```

### 7.7 Temporal Shape Flow

```mermaid
graph LR
    %% Temporal shape flow across frames
    subgraph "Frame t-2"
        Img_t2["Images<br/>(1,6,3,928,1600)"] --> BEV_t2["BEV<br/>(1,256,200,200)"]
    end
    
    subgraph "Frame t-1"
        Img_t1["Images<br/>(1,6,3,928,1600)"] --> BEV_t1["BEV<br/>(1,256,200,200)"]
        BEV_t2 -.->|Ego-motion<br/>compensation| BEV_t1
    end
    
    subgraph "Frame t (current)"
        Img_t["Images<br/>(1,6,3,928,1600)"] --> BEV_t["BEV<br/>(1,256,200,200)"]
        BEV_t1 -.->|Ego-motion<br/>compensation| BEV_t
    end
    
    subgraph "Temporal Aggregation"
        BEV_t2 --> Queue["BEV Queue<br/>(1,3,256,200,200)<br/>Stage1: 5 frames<br/>Stage2: 3 frames"]
        BEV_t1 --> Queue
        BEV_t --> Queue
        
        Queue --> TSA["Temporal Self-Attention<br/>Input: (1,40000,256)×3<br/>Output: (1,40000,256)"]
    end
    
    TSA --> Final["Temporally Enhanced BEV<br/>(1,256,200,200)"]
    
    style BEV_t fill:#99ff99
    style TSA fill:#ffcc99
```

### 7.8 Memory Usage Heatmap

```
Operation               | Memory (GB) | Percentage | Visual
------------------------|-------------|------------|--------
BEV Encoder            | 15.2        | 30.4%      | ████████████████████████████████░░░░░░░░
Track Head             | 8.0         | 16.0%      | ████████████████░░░░░░░░░░░░░░░░░░░░░░░░
Temporal Aggregation   | 7.5         | 15.0%      | ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░
Seg Head               | 5.0         | 10.0%      | ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Occ Head               | 6.0         | 12.0%      | ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Motion Head            | 4.0         | 8.0%       | ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Planning Head          | 3.0         | 6.0%       | ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Others                 | 1.3         | 2.6%       | ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Total                  | 50.0        | 100%       | ████████████████████████████████████████
```

## 8. Implementation Details for Hierarchical Visualization

### 8.1 Module Hierarchy Detection

```python
class ModuleHierarchyAnalyzer:
    def __init__(self, model):
        self.model = model
        self.module_tree = self._build_module_tree()
        
    def _build_module_tree(self):
        """Build hierarchical tree of modules"""
        tree = {}
        for name, module in self.model.named_modules():
            parts = name.split('.')
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {'_module': None, '_children': {}}
                current = current[part]['_children']
        return tree
    
    def get_top_level_modules(self):
        """Return list of top-level module names"""
        return list(self.module_tree.keys())
    
    def get_module_info(self, module_name):
        """Get aggregated info for a module"""
        total_memory = 0
        total_flops = 0
        num_operations = 0
        # Aggregate from all child operations
        return {
            'memory': total_memory,
            'flops': total_flops,
            'operations': num_operations,
            'has_children': bool(children)
        }
```

### 8.2 Visualization State Management

```python
class VisualizationState:
    def __init__(self):
        self.expanded_modules = set()
        self.visualization_mode = 'top-level'
        self.depth_limit = None
        self.memory_threshold = None
        
    def should_expand_module(self, module_name, module_info):
        """Determine if a module should be expanded"""
        if self.visualization_mode == 'full':
            return True
        elif self.visualization_mode == 'top-level':
            return False
        elif module_name in self.expanded_modules:
            return True
        elif self.memory_threshold and module_info['memory'] > self.memory_threshold:
            return True
        return False
```

### 8.3 Interactive Features (Future)

```javascript
// Planned interactive features for web-based visualization
class InteractiveMermaidDiagram {
    constructor(diagramData) {
        this.data = diagramData;
        this.expandedModules = new Set();
    }
    
    onModuleClick(moduleId) {
        if (this.expandedModules.has(moduleId)) {
            this.collapseModule(moduleId);
        } else {
            this.expandModule(moduleId);
        }
        this.rerender();
    }
    
    expandModule(moduleId) {
        // Fetch detailed operations for this module
        // Update diagram with expanded view
    }
    
    collapseModule(moduleId) {
        // Show only aggregated info
        // Update diagram with collapsed view
    }
}
```

## 9. Future Extensions

- Integration with profiling tools (PyTorch Profiler, NVIDIA Nsight)
- Support for distributed training analysis
- **Interactive web-based visualization with clickable module expansion**
- Real-time tracing during training
- Automatic optimization suggestions based on bottlenecks
- Integration with TensorBoard for comprehensive analysis
- Support for custom UniAD variants and configurations
- **Export to interactive HTML with D3.js visualization**
- **Module search and filtering capabilities**
- **Diff visualization between model versions**
