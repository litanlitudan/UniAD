# MemoryBank - Detailed Operation Analysis

**Category**: auxiliary
**Description**: Temporal feature storage for tracking
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 0.3M
- **Memory Footprint**: ~10MB

## Architecture Breakdown

### memory_len

### feature_dim

### ops

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Linear | 1 | linear | 2 * in_features * out_features |
| Softmax | 1 | activation | 2 * num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| linear | 1 | Linear |
| activation | 1 | Softmax |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Dense layers**: Matrix multiplication dominated

