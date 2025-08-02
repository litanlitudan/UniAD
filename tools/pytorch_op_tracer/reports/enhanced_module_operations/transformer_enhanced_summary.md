# Transformer - Enhanced Analysis Summary

**Generated**: 2025-08-02 01:00:49

## Overview

This category contains 2 modules with detailed operation decomposition analysis.

## Modules in Category

| Module | Description | Parameters | Key Operations |
|--------|-------------|------------|----------------|
| PerceptionTransformer | Main transformer for object queries and detection | 8.9M |  |
| MSDeformableAttention3D | Multi-scale deformable attention module for 3D feature aggregation | 0.5M | Linear |

## Common Decomposition Patterns

### Linear
- **Primitives**: gemm, bias_add
- **Hardware**: elementwise_kernel, cublas_gemm/tensor_core
- **Fusion**: gemm_bias_fusion, gemm_relu_fusion

