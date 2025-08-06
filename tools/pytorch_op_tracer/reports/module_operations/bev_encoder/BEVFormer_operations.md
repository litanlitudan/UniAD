# BEVFormer - Detailed Operation Analysis

**Category**: bev_encoder
**Description**: Bird's Eye View transformer encoder with spatio-temporal attention
**Generated**: 2025-08-05 17:24:05

## Module Statistics

- **Total Parameters**: 12.3M
- **Memory Footprint**: ~150MB
- **Computational Complexity**: 3.2 GFLOPs

## Architecture Breakdown

### bev_queries

- **learnable_queries**: `Parameter(200, 200, 256)` - BEV grid queries
- **positional_encoding**: `LearnedPositionalEncoding(200, 200, 256)` - 2D positional encoding

### encoder_layers

- **num_layers**: 6
- **per_layer**: {'temporal_self_attention': {'ops': ['LayerNorm', 'MultiheadAttention', 'Dropout', 'Add'], 'embed_dim': 256, 'num_heads': 8}, 'spatial_cross_attention': {'ops': ['LayerNorm', 'MSDeformableAttention3D', 'Dropout', 'Add'], 'embed_dim': 256, 'num_levels': 4, 'num_points': 8}, 'ffn': {'ops': ['LayerNorm', 'Linear', 'ReLU', 'Dropout', 'Linear', 'Dropout', 'Add'], 'hidden_dim': 512}}

## Operation Summary

| Operation | Count | Category | Complexity |
|-----------|-------|----------|------------|
| LayerNorm | 3 | normalization | 2 * normalized_shape |
| Linear | 2 | linear | 2 * in_features * out_features |
| Parameter | 1 | custom | - |
| LearnedPositionalEncoding | 1 | custom | - |
| MultiheadAttention | 1 | attention | 4 * seq_len^2 * embed_dim + 2 * seq_len * embed_dim^2 |
| ReLU | 1 | activation | num_elements |

## Operation Categories

| Category | Operation Count | Operations |
|----------|-----------------|------------|
| normalization | 3 | LayerNorm |
| linear | 2 | Linear |
| attention | 1 | MultiheadAttention |
| activation | 1 | ReLU |

## Memory and Compute Analysis

### Memory Breakdown
- **Parameters**: Weights and biases storage
- **Activations**: Intermediate feature maps
- **Gradients**: Backward pass storage (training only)

### Compute Patterns
- **Attention-based**: Quadratic complexity with sequence length
- **Dense layers**: Matrix multiplication dominated

