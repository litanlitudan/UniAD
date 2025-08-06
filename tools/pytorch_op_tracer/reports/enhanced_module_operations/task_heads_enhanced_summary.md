# Task_Heads - Enhanced Analysis Summary

**Generated**: 2025-08-05 17:24:24

## Overview

This category contains 5 modules with detailed operation decomposition analysis.

## Modules in Category

| Module | Description | Parameters | Key Operations |
|--------|-------------|------------|----------------|
| TrackHead | 3D object detection and tracking head | 2.8M |  |
| SegHead | BEV segmentation head for lanes and drivable area | 0.8M | Conv2d |
| MotionHead | Multi-modal motion prediction for agents | 1.5M |  |
| OccHead | 3D occupancy and flow prediction | 2.1M | Conv3d |
| PlanningHead | Ego vehicle trajectory planning | 1.2M | GRU |

## Common Decomposition Patterns

### Conv2d
- **Primitives**: im2col, gemm, bias_add
- **Hardware**: cublas_gemm/tensor_core, custom_kernel, elementwise_kernel
- **Fusion**: conv_bias_fusion, conv_relu_fusion

### Conv3d
- **Primitives**: im2col_3d, gemm
- **Hardware**: cublas_gemm/tensor_core, custom_kernel
- **Fusion**: conv3d_bias_fusion

