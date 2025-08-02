# ResNet101-DCN - Enhanced Operation Analysis

**Category**: backbone
**Description**: Image backbone with deformable convolutions
**Generated**: 2025-08-02 01:00:47

## Module Statistics

- **Total Parameters**: 44.5M
- **Memory Footprint**: ~180MB
- **Computational Complexity**: 8.0 GFLOPs

## Enhanced Profiling Results

- **Total Operations Profiled**: 30
- **Total Execution Time**: 0.31 ms
- **Total Estimated FLOPs**: 0
- **Effective TFLOPS**: 0.000

## Operation Breakdown with Decomposition

### Conv2d Operations (1 instances)

**Decomposition**:
1. **im2col** (memory_transform)
   - Hardware: custom_kernel
   - FLOPs Formula: `0`
   - Notes: Unfolds input patches into columns
2. **gemm** (compute)
   - Hardware: cublas_gemm/tensor_core
   - FLOPs Formula: `2 * (in_c * k_h * k_w) * out_c * (out_h * out_w)`
   - Notes: Matrix multiplication of unfolded input and weights
3. **bias_add** (elementwise)
   - Hardware: elementwise_kernel
   - FLOPs Formula: `out_c * out_h * out_w`

**Memory Access Pattern**: im2col_transform -> gemm -> bias
**Fusion Opportunities**: conv_bias_fusion, conv_relu_fusion
**Hardware Requirements**:
- tensor_cores: fp16/tf32
- memory_bandwidth: high

**Performance**:
- Total Time: 0.00 ms
- Average Time: 0.00 ms

---

### BatchNorm2d Operations (1 instances)

**Decomposition**:
1. **mean_computation** (reduction)
   - Hardware: reduction_kernel
   - FLOPs Formula: `channels * height * width`
2. **variance_computation** (reduction)
   - Hardware: reduction_kernel
   - FLOPs Formula: `2 * channels * height * width`
3. **normalize** (elementwise)
   - Hardware: elementwise_kernel
   - FLOPs Formula: `5 * channels * height * width`
   - Notes: (x - mean) / sqrt(var + eps) * gamma + beta

**Memory Access Pattern**: compute_stats -> normalize
**Fusion Opportunities**: bn_relu_fusion, conv_bn_fusion
**Hardware Requirements**:
- memory_bandwidth: moderate

**Performance**:
- Total Time: 0.00 ms
- Average Time: 0.00 ms

---

## Hardware Optimization Opportunities

### Mixed Precision (FP16/TF32)
- 1 operations could benefit from mixed precision
- Potential speedup: ~2x for these operations
- GPU supports Tensor Cores: Yes

### Kernel Fusion
No obvious fusion patterns detected in profiled operations
