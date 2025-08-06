# OccHead - Enhanced Operation Analysis

**Category**: task_heads
**Description**: 3D occupancy and flow prediction
**Generated**: 2025-08-05 17:24:23

## Module Statistics

- **Total Parameters**: 2.1M
- **Memory Footprint**: ~40MB
- **Computational Complexity**: 0.8 GFLOPs

## Enhanced Profiling Results

- **Total Operations Profiled**: 35
- **Total Execution Time**: 10.18 ms
- **Total Estimated FLOPs**: 0
- **Effective TFLOPS**: 0.000

## Operation Breakdown with Decomposition

### Conv3d Operations (5 instances)

**Decomposition**:
1. **im2col_3d** (memory_transform)
   - Hardware: custom_kernel
   - FLOPs Formula: `0`
   - Notes: 3D unfold operation
2. **gemm** (compute)
   - Hardware: cublas_gemm/tensor_core
   - FLOPs Formula: `2 * (in_c * k_d * k_h * k_w) * out_c * (out_d * out_h * out_w)`

**Memory Access Pattern**: im2col_3d -> gemm
**Fusion Opportunities**: conv3d_bias_fusion
**Hardware Requirements**:
- tensor_cores: fp16/tf32
- memory_bandwidth: very_high

**Performance**:
- Total Time: 0.00 ms
- Average Time: 0.00 ms

---

### BatchNorm3d Operations (2 instances)


**Performance**:
- Total Time: 0.00 ms
- Average Time: 0.00 ms

---

## Hardware Optimization Opportunities

### Kernel Fusion
No obvious fusion patterns detected in profiled operations
