# MSDeformableAttention3D - Detailed Operation Analysis

**Category**: transformer
**Description**: Multi-scale deformable attention module for 3D feature aggregation
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 0.5M
- **Memory Footprint**: ~5MB
- **Computational Complexity**: 0.2 GFLOPs

## Architecture Breakdown

### components

- **sampling_offsets**: `Linear(256, 256)` - 2D offset prediction
- **attention_weights**: `Linear(256, 64)` - attention weight prediction
- **value_proj**: `Linear(256, 256)` - value projection
- **output_proj**: `Linear(256, 256)` - output projection

### operations

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Linear | 5 | linear | 2 * in_features * out_features |
| Softmax | 1 | activation | 2 * num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| linear | 5 | Linear |
| activation | 1 | Softmax |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Dense layers**: Matrix multiplication dominated

