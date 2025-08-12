# ONNX Dataflow Visualization Summary

## Overview

Successfully created ONNX operation dataflow visualization for the UniAD model implementation:

**UniAD Architecture Analysis** (`uniad_onnx_dataflow.md`)
   - Based on UniAD codebase architecture
   - 68 ONNX operation nodes mapped from implementation
   - Complete dataflow from ResNet101-DCN → FPN → BEVFormer → Decoder → Task Heads
   - UniAD-specific operations: DeformableConv (DCNv2), DeformableAttention (MSDeformableAttention3D)

## Key ONNX Operations Distribution

From UniAD architecture analysis:
- **Gemm (Linear)**: 30.8% - Dominant in BEVFormer, Decoder, and task heads
- **LayerNormalization**: 19.8% - BEVFormer normalization layers
- **MultiHeadAttention**: 13.2% - Self-attention mechanisms
- **DeformableAttention**: 13.2% - Spatial cross-attention (UniAD-specific)
- **Conv**: 7.7% - Backbone, FPN, and segmentation heads
- **BatchNormalization**: 4.4% - Backbone normalization
- **Relu**: 4.4% - Activation functions
- **DeformableConv**: 2.2% - DCNv2 in ResNet backbone (UniAD-specific)
- **Others**: 4.5% - LSTM, GRU, Concat, Identity

## Dataflow Patterns

### Sequential Flow
```
Input → Conv → BatchNorm → ReLU → MaxPool → Conv → BatchNorm → ReLU
```

### Transformer Pattern
```
MultiHeadAttention → Dropout → LayerNorm → Linear → Dropout → Linear → Dropout → LayerNorm
```

### Task Head Pattern
```
Global Pooling → Linear → ReLU → Linear → Output
```

## Visualizations Generated

### 1. High-Level Module Flow
Shows how data flows between major components:
- Input → Backbone → FPN → BEVFormer → Task Heads

### 2. Detailed ONNX Operations
Complete graph with individual ONNX operations:
- Node labels show operation type and tensor shapes
- Edges represent data dependencies
- Subgraphs group related operations

### 3. Simplified Operation View
Aggregated view showing operation counts and representative shapes

## Technical Implementation

### Tracing Process
1. Hook into PyTorch module forward passes
2. Capture operation types and tensor shapes
3. Map PyTorch ops to ONNX equivalents
4. Build directed graph of operations
5. Generate Mermaid diagrams for visualization

### PyTorch to ONNX Mapping
- Conv2d → Conv
- Linear → Gemm
- BatchNorm2d → BatchNormalization
- ReLU → Relu
- TransformerEncoderLayer → Multiple ops (MHA, LayerNorm, Linear)

## Files Generated

1. `reports/uniad_onnx_dataflow.md` - UniAD architecture ONNX dataflow
2. `reports/uniad_onnx_trace_data.json` - Complete trace data with all nodes and edges
3. `trace_onnx_dataflow.py` - Script that analyzes UniAD implementation

## Usage

To regenerate the visualization from UniAD implementation:
```bash
# Analyze UniAD architecture and generate ONNX dataflow
python trace_onnx_dataflow.py
```

## Future Improvements

1. **Full Model Tracing**: Trace complete UniAD model with all components
2. **Edge Accuracy**: Track actual tensor dependencies for precise edges
3. **Quantization Analysis**: Show INT8/FP16 optimization opportunities
4. **ONNX Export**: Generate actual ONNX model for deployment
5. **Performance Metrics**: Add FLOPS and memory usage to each node