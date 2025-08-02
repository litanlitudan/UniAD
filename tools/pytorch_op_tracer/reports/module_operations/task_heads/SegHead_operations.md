# SegHead - Detailed Operation Analysis

**Category**: task_heads
**Description**: BEV segmentation head for lanes and drivable area
**Generated**: 2025-08-01 22:17:51

## Module Statistics

- **Total Parameters**: 0.8M
- **Memory Footprint**: ~20MB
- **Computational Complexity**: 0.3 GFLOPs

## Architecture Breakdown

### seg_convs

- **conv1**: `Conv2d(256, 128, 3)` - feature extraction
- **conv2**: `Conv2d(128, 64, 3)` - feature extraction
- **conv3**: `Conv2d(64, 32, 3)` - feature extraction

### seg_classifier

### operations

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Conv2d | 5 | convolution | 2 * in_ch * out_ch * k_h * k_w * out_h * out_w |
| BatchNorm2d | 1 | normalization | 2 * num_features * H * W |
| ReLU | 1 | activation | num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| convolution | 5 | Conv2d |
| normalization | 1 | BatchNorm2d |
| activation | 1 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Convolution-heavy**: High compute for spatial feature extraction

