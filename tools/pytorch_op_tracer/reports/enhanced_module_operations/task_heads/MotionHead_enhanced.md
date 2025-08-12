# MotionHead - Enhanced Operation Analysis

**Category**: task_heads
**Description**: Multi-modal motion prediction for agents
**Generated**: 2025-08-05 17:24:21

## Module Statistics

- **Total Parameters**: 1.5M
- **Memory Footprint**: ~30MB
- **Computational Complexity**: 0.4 GFLOPs

## Enhanced Profiling Results

- **Total Operations Profiled**: 42
- **Total Execution Time**: 0.35 ms
- **Total Estimated FLOPs**: 0
- **Effective TFLOPS**: 0.000

## Operation Breakdown with Decomposition

### Linear Operations (3 instances)

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

## Hardware Optimization Opportunities

### Mixed Precision (FP16/TF32)
- 3 operations could benefit from mixed precision
- Potential speedup: ~6x for these operations
- GPU supports Tensor Cores: Yes

### Kernel Fusion
No obvious fusion patterns detected in profiled operations
