# TrackHead - Detailed Operation Analysis

**Category**: task_heads
**Description**: 3D object detection and tracking head
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 2.8M
- **Memory Footprint**: ~50MB
- **Computational Complexity**: 0.5 GFLOPs

## Architecture Breakdown

### cls_branches

- **layers**: 2
- **ops_per_layer**: Linear → LayerNorm → ReLU → Linear
- **output_dim**: 10

### reg_branches

- **layers**: 3
- **ops_per_layer**: Linear → LayerNorm → ReLU
- **output_dim**: 10

### track_embed

- **ops**: Linear → ReLU → Linear
- **output_dim**: 256

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Linear | 5 | linear | 2 * in_features * out_features |
| ReLU | 3 | activation | num_elements |
| LayerNorm | 2 | normalization | 2 * normalized_shape |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| linear | 5 | Linear |
| activation | 3 | ReLU |
| normalization | 2 | LayerNorm |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Dense layers**: Matrix multiplication dominated

