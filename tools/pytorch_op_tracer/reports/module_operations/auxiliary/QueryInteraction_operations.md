# QueryInteraction - Detailed Operation Analysis

**Category**: auxiliary
**Description**: Query refinement module
**Generated**: 2025-08-01 22:17:51

## Module Statistics

- **Total Parameters**: 0.2M
- **Memory Footprint**: ~5MB

## Architecture Breakdown

### ops

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Linear | 2 | linear | 2 * in_features * out_features |
| LayerNorm | 1 | normalization | 2 * normalized_shape |
| ReLU | 1 | activation | num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| linear | 2 | Linear |
| normalization | 1 | LayerNorm |
| activation | 1 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Dense layers**: Matrix multiplication dominated

