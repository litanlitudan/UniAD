# PerceptionTransformer - Enhanced Operation Analysis

**Category**: transformer
**Description**: Main transformer for object queries and detection
**Generated**: 2025-08-05 17:24:14

## Module Statistics

- **Total Parameters**: 8.9M
- **Memory Footprint**: ~35MB
- **Computational Complexity**: 1.8 GFLOPs

## Enhanced Profiling Results

- **Total Operations Profiled**: 48
- **Total Execution Time**: 0.58 ms
- **Total Estimated FLOPs**: 0
- **Effective TFLOPS**: 0.000

## Operation Breakdown with Decomposition

### Linear Operations (2 instances)

**Decomposition**:
1. **gemm** (compute)
   - Hardware: cublas_gemm/tensor_core
   - FLOPs Formula: `2 * in_features * out_features * batch_size`
2. **bias_add** (elementwise)
   - Hardware: elementwise_kernel
   - FLOPs Formula: `out_features * batch_size`

**Memory Access Pattern**: gemm -> bias_add
**Fusion Opportunities**: gemm_bias_fusion, gemm_relu_fusion
**Hardware Requirements**:
- tensor_cores: fp16/tf32/int8

**Performance**:
- Total Time: 0.00 ms
- Average Time: 0.00 ms

---

### LayerNorm Operations (3 instances)

**Decomposition**:
1. **mean_computation** (reduction)
   - Hardware: reduction_kernel
   - FLOPs Formula: `normalized_shape_prod`
2. **variance_computation** (reduction)
   - Hardware: reduction_kernel
   - FLOPs Formula: `2 * normalized_shape_prod`
3. **normalize** (elementwise)
   - Hardware: elementwise_kernel
   - FLOPs Formula: `5 * normalized_shape_prod`

**Memory Access Pattern**: compute_stats -> normalize
**Fusion Opportunities**: layernorm_fusion
**Hardware Requirements**:
- memory_bandwidth: low

**Performance**:
- Total Time: 0.00 ms
- Average Time: 0.00 ms

---

## Hardware Optimization Opportunities

### Mixed Precision (FP16/TF32)
- 2 operations could benefit from mixed precision
- Potential speedup: ~4x for these operations
- GPU supports Tensor Cores: Yes

### Kernel Fusion
No obvious fusion patterns detected in profiled operations
