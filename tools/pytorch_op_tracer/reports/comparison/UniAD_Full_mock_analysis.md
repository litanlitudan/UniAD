# UniAD_Full Mock Model Analysis

Generated at: 2025-08-05 17:26:06

## Model Statistics

- Total Parameters: 4,980,198
- Model Size: 19.0 MB (fp32)

## Module Structure

```
backbone: Sequential
  backbone.0: Conv2d
  backbone.1: BatchNorm2d
  backbone.2: ReLU
  backbone.3: Conv2d
transformer: TransformerEncoder
  transformer.layers: ModuleList
    transformer.layers.0: TransformerEncoderLayer
      transformer.layers.0.self_attn: MultiheadAttention
        transformer.layers.0.self_attn.out_proj: NonDynamicallyQuantizableLinear
      transformer.layers.0.linear1: Linear
      transformer.layers.0.dropout: Dropout
      transformer.layers.0.linear2: Linear
      transformer.layers.0.norm1: LayerNorm
      transformer.layers.0.norm2: LayerNorm
      transformer.layers.0.dropout1: Dropout
      transformer.layers.0.dropout2: Dropout
    transformer.layers.1: TransformerEncoderLayer
      transformer.layers.1.self_attn: MultiheadAttention
        transformer.layers.1.self_attn.out_proj: NonDynamicallyQuantizableLinear
      transformer.layers.1.linear1: Linear
      transformer.layers.1.dropout: Dropout
      transformer.layers.1.linear2: Linear
      transformer.layers.1.norm1: LayerNorm
      transformer.layers.1.norm2: LayerNorm
      transformer.layers.1.dropout1: Dropout
      transformer.layers.1.dropout2: Dropout
    transformer.layers.2: TransformerEncoderLayer
      transformer.layers.2.self_attn: MultiheadAttention
        transformer.layers.2.self_attn.out_proj: NonDynamicallyQuantizableLinear
      transformer.layers.2.linear1: Linear
      transformer.layers.2.dropout: Dropout
      transformer.layers.2.linear2: Linear
      transformer.layers.2.norm1: LayerNorm
      transformer.layers.2.norm2: LayerNorm
      transformer.layers.2.dropout1: Dropout
      transformer.layers.2.dropout2: Dropout
    transformer.layers.3: TransformerEncoderLayer
      transformer.layers.3.self_attn: MultiheadAttention
        transformer.layers.3.self_attn.out_proj: NonDynamicallyQuantizableLinear
      transformer.layers.3.linear1: Linear
      transformer.layers.3.dropout: Dropout
      transformer.layers.3.linear2: Linear
      transformer.layers.3.norm1: LayerNorm
      transformer.layers.3.norm2: LayerNorm
      transformer.layers.3.dropout1: Dropout
      transformer.layers.3.dropout2: Dropout
    transformer.layers.4: TransformerEncoderLayer
      transformer.layers.4.self_attn: MultiheadAttention
        transformer.layers.4.self_attn.out_proj: NonDynamicallyQuantizableLinear
      transformer.layers.4.linear1: Linear
      transformer.layers.4.dropout: Dropout
      transformer.layers.4.linear2: Linear
      transformer.layers.4.norm1: LayerNorm
      transformer.layers.4.norm2: LayerNorm
      transformer.layers.4.dropout1: Dropout
      transformer.layers.4.dropout2: Dropout
    transformer.layers.5: TransformerEncoderLayer
      transformer.layers.5.self_attn: MultiheadAttention
        transformer.layers.5.self_attn.out_proj: NonDynamicallyQuantizableLinear
      transformer.layers.5.linear1: Linear
      transformer.layers.5.dropout: Dropout
      transformer.layers.5.linear2: Linear
      transformer.layers.5.norm1: LayerNorm
      transformer.layers.5.norm2: LayerNorm
      transformer.layers.5.dropout1: Dropout
      transformer.layers.5.dropout2: Dropout
track_head: Linear
seg_head: Conv2d
motion_head: Linear
occ_head: Conv2d
planning_head: Linear
```

## Operation Types

- Dropout: 18
- Linear: 15
- LayerNorm: 12
- TransformerEncoderLayer: 6
- MultiheadAttention: 6
- NonDynamicallyQuantizableLinear: 6
- Conv2d: 4
- MockUniADFull: 1
- Sequential: 1
- BatchNorm2d: 1
- ReLU: 1
- TransformerEncoder: 1
- ModuleList: 1
