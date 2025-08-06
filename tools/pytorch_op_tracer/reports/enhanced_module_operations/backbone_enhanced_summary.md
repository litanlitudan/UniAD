# Backbone - Enhanced Analysis Summary

**Generated**: 2025-08-05 17:24:11

## Overview

This category contains 2 modules with detailed operation decomposition analysis.

## Modules in Category

| Module | Description | Parameters | Key Operations |
|--------|-------------|------------|----------------|
| ResNet101-DCN | Image backbone with deformable convolutions | 44.5M | BatchNorm2d, MaxPool2d, ReLU, Conv2d |
| FPN | Feature Pyramid Network for multi-scale features | 3.5M | Conv2d |

## Common Decomposition Patterns

### BatchNorm2d
- **Primitives**: mean_computation, variance_computation, normalize
- **Hardware**: reduction_kernel, elementwise_kernel
- **Fusion**: bn_relu_fusion, conv_bn_fusion

### Conv2d
- **Primitives**: im2col, gemm, bias_add
- **Hardware**: cublas_gemm/tensor_core, custom_kernel, elementwise_kernel
- **Fusion**: conv_bias_fusion, conv_relu_fusion

### MaxPool2d
- **Primitives**: max_reduction_2d
- **Hardware**: pooling_kernel
- **Fusion**: conv_pool_fusion

### ReLU
- **Primitives**: max_with_zero
- **Hardware**: elementwise_kernel
- **Fusion**: conv_relu_fusion, bn_relu_fusion

