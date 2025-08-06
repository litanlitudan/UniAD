# OccHead - Detailed Operation Analysis

**Category**: task_heads
**Description**: 3D occupancy and flow prediction
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 2.1M
- **Memory Footprint**: ~40MB
- **Computational Complexity**: 0.8 GFLOPs

## Architecture Breakdown

### occ_convs

- **3d_conv1**: `Conv3d(256, 128, 3)` - 3D feature extraction
- **3d_conv2**: `Conv3d(128, 64, 3)` - 3D feature extraction
- **3d_conv3**: `Conv3d(64, 32, 3)` - 3D feature extraction

### flow_head

### semantic_head

### operations

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Conv3d | 6 | convolution | 2 * in_ch * out_ch * k_h * k_w * k_d * out_h * out_w * out_d |
| ReLU | 1 | activation | num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| convolution | 6 | Conv3d |
| activation | 1 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Convolution-heavy**: High compute for spatial feature extraction

