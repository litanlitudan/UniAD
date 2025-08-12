# Architecture Visualization: Text-based Model Loading

## Mock UniADTrack Model Structure

```mermaid
graph TB
    Input["Input Images<br/>[B, 3, H, W]"]
    
    subgraph "Mock Backbone"
        Conv1["Conv2d(3→64, k=7, s=2)"]
        BN1["BatchNorm2d(64)"]
        ReLU1["ReLU"]
        Pool["MaxPool2d(k=3, s=2)"]
        Conv2["Conv2d(64→256, k=3)"]
    end
    
    subgraph "Mock Neck"
        NeckConv["Conv2d(256→256, k=3)"]
        NeckBN["BatchNorm2d(256)"]
        NeckReLU["ReLU"]
    end
    
    subgraph "Mock BEV Encoder"
        BEVConv["Conv2d(256→256, k=3)"]
        BEVBN["BatchNorm2d(256)"]
        BEVReLU["ReLU"]
    end
    
    subgraph "Mock Detection Head"
        DetConv1["Conv2d(256→128, k=3)"]
        DetReLU["ReLU"]
        DetConv2["Conv2d(128→10, k=1)"]
    end
    
    Input --> Conv1 --> BN1 --> ReLU1 --> Pool --> Conv2
    Conv2 --> NeckConv --> NeckBN --> NeckReLU
    NeckReLU --> BEVConv --> BEVBN --> BEVReLU
    BEVReLU --> DetConv1 --> DetReLU --> DetConv2
    
    DetConv2 --> Output["Detection Output<br/>[B, 10, H', W']"]
```

## Mock UniAD Full Model Structure

```mermaid
graph TB
    Input["Multi-Camera Input<br/>[B, 6, 3, H, W]"]
    
    subgraph "Base Model (UniADTrack)"
        BaseModel["Mock UniADTrack<br/>(as above)"]
    end
    
    subgraph "Task Heads"
        Track["Track Head<br/>Detection Output"]
        Seg["Segmentation Head<br/>Conv2d(256→4, k=1)"]
        Motion["Motion Head<br/>Linear(256→144)"]
        Occ["Occupancy Head<br/>Conv2d(256→2, k=1)"]
        Planning["Planning Head<br/>Linear(256→12)"]
    end
    
    Input --> BaseModel
    BaseModel --> |"BEV Features"| Track
    BaseModel --> |"BEV Features"| Seg
    BaseModel --> |"Pooled Features"| Motion
    BaseModel --> |"BEV Features"| Occ
    BaseModel --> |"Pooled Features"| Planning
    
    Track --> TrackOut["Detections"]
    Seg --> SegOut["Segmentation Map"]
    Motion --> MotionOut["Motion Predictions<br/>(12 steps × 6 modes × 2D)"]
    Occ --> OccOut["Occupancy Grid"]
    Planning --> PlanOut["Ego Trajectory<br/>(6 steps × 2D)"]
```

## Comparison: Real vs Mock Models

### Real UniAD Components
- **ResNet101-DCN**: ~45M parameters
- **FPN**: Multi-scale feature pyramid
- **BEVFormer**: 6-layer transformer encoder
- **Perception Transformer**: Complex attention mechanisms
- **Task-specific Decoders**: Specialized architectures

### Mock Model Simplifications
- **Simple CNN**: ~17M parameters
- **Basic convolutions**: No transformers
- **Simplified heads**: Minimal task decoders
- **No temporal modeling**: Single-frame only
- **No attention**: Basic feed-forward

## Memory Footprint Comparison

| Component | Real Model | Mock Model | Reduction |
|-----------|------------|------------|-----------|
| Backbone | ~180 MB | ~65 MB | 64% |
| BEV Encoder | ~150 MB | ~10 MB | 93% |
| Track Head | ~50 MB | ~5 MB | 90% |
| Motion Head | ~30 MB | ~1 MB | 97% |
| **Total** | **~500 MB** | **~85 MB** | **83%** |

## Forward Pass Comparison

### Real Model Forward Pass
1. Multi-view image encoding with ResNet101-DCN
2. FPN for multi-scale features
3. BEVFormer spatial-temporal aggregation
4. Transformer-based object queries
5. Task-specific decoding with attention

### Mock Model Forward Pass
1. Simple CNN feature extraction
2. Basic convolution for "BEV" features
3. Direct convolution for detection
4. Simple linear layers for other tasks
5. No complex interactions

## Benefits of Mock Models for Analysis

1. **Faster Iteration**: 10x faster forward pass
2. **Lower Memory**: 80%+ reduction
3. **Clear Structure**: Easy to understand flow
4. **No Dependencies**: Works without mmdet3d
5. **Customizable**: Easy to modify for testing

## When to Use Each Approach

### Use Real Models For:
- Training on nuScenes dataset
- Evaluation and benchmarking
- Production deployment
- Research comparisons

### Use Mock Models For:
- Architecture visualization
- Operation tracing
- Memory profiling
- Quick prototyping
- Educational purposes