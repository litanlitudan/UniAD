# PlanningHead - Detailed Operation Analysis

**Category**: task_heads
**Description**: Ego vehicle trajectory planning
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 1.2M
- **Memory Footprint**: ~25MB
- **Computational Complexity**: 0.3 GFLOPs

## Architecture Breakdown

### planning_decoder

- **gru_layer**: `GRU(512, 256)` - temporal modeling
- **output_mlp**: Linear → ReLU → Linear → Tanh

### cost_volume

- **ops**: Conv1d → ReLU → Conv1d
- **channels**: 256 → 128 → 64

### planning_steps

### output_dim

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Linear | 2 | linear | 2 * in_features * out_features |
| ReLU | 2 | activation | num_elements |
| GRU | 1 | custom | - |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| linear | 2 | Linear |
| activation | 2 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Dense layers**: Matrix multiplication dominated

