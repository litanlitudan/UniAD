# ResNet101-DCN - Detailed Operation Analysis

**Category**: backbone
**Description**: Image backbone with deformable convolutions
**Generated**: 2025-08-01 22:17:51

## Module Statistics

- **Total Parameters**: 44.5M
- **Memory Footprint**: ~180MB
- **Computational Complexity**: 8.0 GFLOPs

## Architecture Breakdown

### stem

- **conv1**: `Conv2d(3, 64, 7, 2)` - 7x7 conv, stride 2
- **bn1**: `BatchNorm2d(64,)` - batch norm
- **relu**: `ReLU()` - activation
- **maxpool**: `MaxPool2d(3, 2, 1)` - 3x3 max pool, stride 2

### layer1

- **blocks**: 3
- **channels**: 64 → 64 → 256
- **ops_per_block**: Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → Add → ReLU

### layer2

- **blocks**: 4
- **channels**: 256 → 128 → 512
- **ops_per_block**: Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → Add → ReLU

### layer3

- **blocks**: 23
- **channels**: 512 → 256 → 1024
- **ops_per_block**: Conv2d → BatchNorm2d → ReLU → DCNv2 → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → Add → ReLU
- **dcn_enabled**: True

### layer4

- **blocks**: 3
- **channels**: 1024 → 512 → 2048
- **ops_per_block**: Conv2d → BatchNorm2d → ReLU → DCNv2 → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → Add → ReLU
- **dcn_enabled**: True

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| BatchNorm2d | 13 | normalization | 2 * num_features * H * W |
| ReLU | 13 | activation | num_elements |
| Conv2d | 11 | convolution | 2 * in_ch * out_ch * k_h * k_w * out_h * out_w |
| DCNv2 | 2 | deformable_convolution | 2 * in_ch * out_ch * k_h * k_w * out_h * out_w + offset_conv + mask_conv |
| MaxPool2d | 1 | custom | - |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| normalization | 13 | BatchNorm2d |
| activation | 13 | ReLU |
| convolution | 11 | Conv2d |
| deformable_convolution | 2 | DCNv2 |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Convolution-heavy**: High compute for spatial feature extraction

