# FPN - Detailed Operation Analysis

**Category**: backbone
**Description**: Feature Pyramid Network for multi-scale features
**Generated**: 2025-08-01 22:17:51

## Module Statistics

- **Total Parameters**: 3.5M
- **Memory Footprint**: ~15MB
- **Computational Complexity**: 0.5 GFLOPs

## Architecture Breakdown

### lateral_convs

- **lateral_conv0**: `Conv2d(512, 256, 1)` - 1x1 conv for C2
- **lateral_conv1**: `Conv2d(1024, 256, 1)` - 1x1 conv for C3
- **lateral_conv2**: `Conv2d(2048, 256, 1)` - 1x1 conv for C4

### fpn_convs

- **fpn_conv0**: `Conv2d(256, 256, 3)` - 3x3 conv for P2
- **fpn_conv1**: `Conv2d(256, 256, 3)` - 3x3 conv for P3
- **fpn_conv2**: `Conv2d(256, 256, 3)` - 3x3 conv for P4

### extra_convs

- **extra_conv**: `Conv2d(256, 256, 3, 2)` - 3x3 stride 2 for P5

### operations

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Conv2d | 8 | convolution | 2 * in_ch * out_ch * k_h * k_w * out_h * out_w |
| ReLU | 1 | activation | num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| convolution | 8 | Conv2d |
| activation | 1 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Convolution-heavy**: High compute for spatial feature extraction

