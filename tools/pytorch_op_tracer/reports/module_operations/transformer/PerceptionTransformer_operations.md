# PerceptionTransformer - Detailed Operation Analysis

**Category**: transformer
**Description**: Main transformer for object queries and detection
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 8.9M
- **Memory Footprint**: ~35MB
- **Computational Complexity**: 1.8 GFLOPs

## Architecture Breakdown

### decoder_layers

- **num_layers**: 6
- **per_layer**: {'self_attention': {'ops': ['LayerNorm', 'MultiheadAttention', 'Dropout', 'Add'], 'embed_dim': 256, 'num_heads': 8}, 'cross_attention': {'ops': ['LayerNorm', 'MultiheadAttention', 'Dropout', 'Add'], 'embed_dim': 256, 'num_heads': 8}, 'ffn': {'ops': ['LayerNorm', 'Linear', 'ReLU', 'Dropout', 'Linear', 'Dropout', 'Add'], 'hidden_dim': 2048}}

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| LayerNorm | 3 | normalization | 2 * normalized_shape |
| MultiheadAttention | 2 | attention | 4 * seq_len^2 * embed_dim + 2 * seq_len * embed_dim^2 |
| Linear | 2 | linear | 2 * in_features * out_features |
| ReLU | 1 | activation | num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| normalization | 3 | LayerNorm |
| attention | 2 | MultiheadAttention |
| linear | 2 | Linear |
| activation | 1 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Attention-based**: Quadratic complexity with sequence length
- **Dense layers**: Matrix multiplication dominated

