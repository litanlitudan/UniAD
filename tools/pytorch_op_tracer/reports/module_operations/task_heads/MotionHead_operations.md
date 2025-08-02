# MotionHead - Detailed Operation Analysis

**Category**: task_heads
**Description**: Multi-modal motion prediction for agents
**Generated**: 2025-08-01 22:17:51

## Module Statistics

- **Total Parameters**: 1.5M
- **Memory Footprint**: ~30MB
- **Computational Complexity**: 0.4 GFLOPs

## Architecture Breakdown

### motion_decoder

- **layers**: LSTM → Linear → ReLU → Linear
- **hidden_size**: 128
- **num_modes**: 6
- **future_steps**: 12

### mode_prob

- **ops**: Linear → Softmax
- **output_dim**: 6

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| Linear | 3 | linear | 2 * in_features * out_features |
| LSTM | 1 | recurrent | 4 * (input_size + hidden_size + 1) * hidden_size * seq_len |
| ReLU | 1 | activation | num_elements |
| Softmax | 1 | activation | 2 * num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| linear | 3 | Linear |
| activation | 2 | ReLU, Softmax |
| recurrent | 1 | LSTM |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Dense layers**: Matrix multiplication dominated

