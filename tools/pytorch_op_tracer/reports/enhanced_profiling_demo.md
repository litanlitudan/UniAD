# UniAD Enhanced Profiling Report

**Generated**: 2025-08-01 23:55:21

## GPU Properties

- **Device**: NVIDIA GeForce RTX 3090
- **Compute Capability**: (8, 6)
- **Memory Bandwidth**: 936.0 GB/s
- **FP32 Performance**: 35.6 TFLOPS
- **FP16 Performance**: 71.0 TFLOPS
- **Tensor Cores**: Yes

## Summary Statistics

- **Total Operations**: 52
- **Total CUDA Time**: 132.41 ms
- **Total Estimated FLOPs**: 0
- **Effective TFLOPS**: 0.00

## Operation Breakdown

| Operation | Count | Total Time (ms) | Total FLOPs | Avg Time (ms) |
|-----------|-------|-----------------|-------------|---------------|
| aten | 36 | 78.75 | 0 | 2.19 |
| void at | 7 | 35.62 | 0 | 5.09 |
| void cutlass | 4 | 17.62 | 0 | 4.41 |
| void cudnn | 1 | 0.30 | 0 | 0.30 |
| void xmma_new | 1 | 0.11 | 0 | 0.11 |
| Conv2d | 2 | 0.00 | 0 | 0.00 |
| BatchNorm2d | 1 | 0.00 | 0 | 0.00 |

## Detailed Operation Decomposition

### 1. aten - softmax

**Timing**: CPU: 0.46ms, CUDA: 23.01ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [[8, 12544, 12544], [], []], Output: []


**CUDA Kernels**:
- aten::softmax (custom): 23007.0μs

---

### 2. aten - _softmax

**Timing**: CPU: 0.45ms, CUDA: 23.01ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [[8, 12544, 12544], [], []], Output: []


**CUDA Kernels**:
- aten::_softmax (custom): 23007.0μs

---

### 3. void at - native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float*, int)

**Timing**: CPU: 0.00ms, CUDA: 23.01ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [], Output: []


**CUDA Kernels**:
- void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float*, int) (custom): 23007.0μs

---

### 4. aten - bmm

**Timing**: CPU: 0.20ms, CUDA: 17.45ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [[8, 12544, 16], [8, 16, 12544]], Output: []


**CUDA Kernels**:
- aten::bmm (custom): 7522.0μs
- aten::bmm (custom): 9923.0μs

---

### 5. aten - sum

**Timing**: CPU: 0.09ms, CUDA: 10.87ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [[1, 8, 12544, 12544], [], [], []], Output: []


**CUDA Kernels**:
- aten::sum (custom): 10865.0μs

---

### 6. void at - native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)

**Timing**: CPU: 0.00ms, CUDA: 10.87ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [], Output: []


**CUDA Kernels**:
- void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>) (custom): 2219.0μs
- void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>) (custom): 2882.0μs
- void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>) (custom): 2882.0μs
- void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>) (custom): 2882.0μs

---

### 7. void cutlass - Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x4_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x4_nn_align4::Params)

**Timing**: CPU: 0.00ms, CUDA: 9.92ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [], Output: []


**CUDA Kernels**:
- void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x4_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x4_nn_align4::Params) (cublas): 9923.0μs

---

### 8. void cutlass - Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)

**Timing**: CPU: 0.00ms, CUDA: 7.52ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [], Output: []


**CUDA Kernels**:
- void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params) (cublas): 7522.0μs

---

### 9. aten - div

**Timing**: CPU: 0.11ms, CUDA: 1.51ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [[8, 12544, 16], []], Output: []


**CUDA Kernels**:
- aten::div (custom): 17.0μs
- aten::div (custom): 1489.0μs

---

### 10. void at - native::vectorized_elementwise_kernel<4, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2> >(int, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2>)

**Timing**: CPU: 0.00ms, CUDA: 1.51ms
**Memory**: 0.0MB allocated
**Shapes**: Input: [], Output: []


**CUDA Kernels**:
- void at::native::vectorized_elementwise_kernel<4, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2> >(int, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2>) (elementwise): 17.0μs
- void at::native::vectorized_elementwise_kernel<4, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2> >(int, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2>) (elementwise): 1489.0μs

---

## Hardware Bottleneck Analysis

- **Compute-Bound Operations**: 0 (0.0%)
- **Memory-Bound Operations**: 0 (0.0%)

## Optimization Opportunities


### Tensor Core Optimization
2 operations could benefit from FP16/TF32 for tensor cores:
- Conv2d in conv1
- Conv2d in conv2
