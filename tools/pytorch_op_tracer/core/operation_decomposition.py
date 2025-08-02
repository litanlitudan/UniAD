"""Operation Decomposition Module - Maps PyTorch ops to lower-level primitives"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class OperationPrimitive:
    """Represents a low-level operation primitive"""
    name: str
    category: str  # gemm, elementwise, memory, etc.
    flops_formula: str
    memory_ops_formula: str
    hardware_mapping: str  # cuda kernel, tensor core, etc.
    notes: Optional[str] = None


@dataclass
class DecomposedOperation:
    """Represents a PyTorch operation decomposed into primitives"""
    pytorch_op: str
    primitives: List[OperationPrimitive]
    total_flops_formula: str
    memory_access_pattern: str
    fusion_opportunities: List[str]
    hardware_requirements: Dict[str, Any]


class OperationDecomposer:
    """Decomposes PyTorch operations into lower-level primitives"""
    
    def __init__(self):
        self.decomposition_map = self._build_decomposition_map()
        self.primitive_costs = self._build_primitive_costs()
    
    def _build_decomposition_map(self) -> Dict[str, DecomposedOperation]:
        """Build the mapping of PyTorch ops to their decomposition"""
        return {
            # Convolution operations
            "Conv2d": DecomposedOperation(
                pytorch_op="Conv2d",
                primitives=[
                    OperationPrimitive(
                        name="im2col",
                        category="memory_transform",
                        flops_formula="0",  # Pure memory operation
                        memory_ops_formula="in_c * k_h * k_w * out_h * out_w",
                        hardware_mapping="custom_kernel",
                        notes="Unfolds input patches into columns"
                    ),
                    OperationPrimitive(
                        name="gemm",
                        category="compute",
                        flops_formula="2 * (in_c * k_h * k_w) * out_c * (out_h * out_w)",
                        memory_ops_formula="2 * (in_c * k_h * k_w * out_c + out_c * out_h * out_w)",
                        hardware_mapping="cublas_gemm/tensor_core",
                        notes="Matrix multiplication of unfolded input and weights"
                    ),
                    OperationPrimitive(
                        name="bias_add",
                        category="elementwise",
                        flops_formula="out_c * out_h * out_w",
                        memory_ops_formula="2 * out_c * out_h * out_w",
                        hardware_mapping="elementwise_kernel"
                    )
                ],
                total_flops_formula="2 * in_c * out_c * k_h * k_w * out_h * out_w + out_c * out_h * out_w",
                memory_access_pattern="im2col_transform -> gemm -> bias",
                fusion_opportunities=["conv_bias_fusion", "conv_relu_fusion"],
                hardware_requirements={"tensor_cores": "fp16/tf32", "memory_bandwidth": "high"}
            ),
            
            "Conv3d": DecomposedOperation(
                pytorch_op="Conv3d",
                primitives=[
                    OperationPrimitive(
                        name="im2col_3d",
                        category="memory_transform",
                        flops_formula="0",
                        memory_ops_formula="in_c * k_d * k_h * k_w * out_d * out_h * out_w",
                        hardware_mapping="custom_kernel",
                        notes="3D unfold operation"
                    ),
                    OperationPrimitive(
                        name="gemm",
                        category="compute",
                        flops_formula="2 * (in_c * k_d * k_h * k_w) * out_c * (out_d * out_h * out_w)",
                        memory_ops_formula="2 * (in_c * k_d * k_h * k_w * out_c + out_c * out_d * out_h * out_w)",
                        hardware_mapping="cublas_gemm/tensor_core"
                    )
                ],
                total_flops_formula="2 * in_c * out_c * k_d * k_h * k_w * out_d * out_h * out_w",
                memory_access_pattern="im2col_3d -> gemm",
                fusion_opportunities=["conv3d_bias_fusion"],
                hardware_requirements={"tensor_cores": "fp16/tf32", "memory_bandwidth": "very_high"}
            ),
            
            # Linear operations
            "Linear": DecomposedOperation(
                pytorch_op="Linear",
                primitives=[
                    OperationPrimitive(
                        name="gemm",
                        category="compute",
                        flops_formula="2 * in_features * out_features * batch_size",
                        memory_ops_formula="2 * (in_features * out_features + batch_size * in_features + batch_size * out_features)",
                        hardware_mapping="cublas_gemm/tensor_core"
                    ),
                    OperationPrimitive(
                        name="bias_add",
                        category="elementwise",
                        flops_formula="out_features * batch_size",
                        memory_ops_formula="2 * out_features * batch_size",
                        hardware_mapping="elementwise_kernel"
                    )
                ],
                total_flops_formula="2 * in_features * out_features * batch_size + out_features * batch_size",
                memory_access_pattern="gemm -> bias_add",
                fusion_opportunities=["gemm_bias_fusion", "gemm_relu_fusion"],
                hardware_requirements={"tensor_cores": "fp16/tf32/int8"}
            ),
            
            # Attention operations
            "MultiheadAttention": DecomposedOperation(
                pytorch_op="MultiheadAttention",
                primitives=[
                    OperationPrimitive(
                        name="qkv_projection",
                        category="compute",
                        flops_formula="3 * 2 * embed_dim * embed_dim * seq_len * batch_size",
                        memory_ops_formula="3 * 2 * (embed_dim^2 + embed_dim * seq_len * batch_size)",
                        hardware_mapping="cublas_gemm/tensor_core",
                        notes="Q, K, V linear projections"
                    ),
                    OperationPrimitive(
                        name="reshape_transpose",
                        category="memory_transform",
                        flops_formula="0",
                        memory_ops_formula="3 * embed_dim * seq_len * batch_size",
                        hardware_mapping="memory_kernel",
                        notes="Reshape for multi-head"
                    ),
                    OperationPrimitive(
                        name="scaled_dot_product",
                        category="compute",
                        flops_formula="2 * num_heads * seq_len^2 * (embed_dim // num_heads) * batch_size",
                        memory_ops_formula="2 * num_heads * seq_len^2 * batch_size",
                        hardware_mapping="flash_attention/cublas_gemm",
                        notes="Q @ K.T computation"
                    ),
                    OperationPrimitive(
                        name="softmax",
                        category="elementwise",
                        flops_formula="4 * num_heads * seq_len^2 * batch_size",
                        memory_ops_formula="2 * num_heads * seq_len^2 * batch_size",
                        hardware_mapping="softmax_kernel"
                    ),
                    OperationPrimitive(
                        name="attention_weighted_sum",
                        category="compute",
                        flops_formula="2 * num_heads * seq_len^2 * (embed_dim // num_heads) * batch_size",
                        memory_ops_formula="2 * num_heads * seq_len * embed_dim * batch_size",
                        hardware_mapping="cublas_gemm"
                    ),
                    OperationPrimitive(
                        name="output_projection",
                        category="compute",
                        flops_formula="2 * embed_dim * embed_dim * seq_len * batch_size",
                        memory_ops_formula="2 * (embed_dim^2 + embed_dim * seq_len * batch_size)",
                        hardware_mapping="cublas_gemm/tensor_core"
                    )
                ],
                total_flops_formula="4 * embed_dim^2 * seq_len * batch_size + 4 * seq_len^2 * embed_dim * batch_size",
                memory_access_pattern="qkv_proj -> scaled_dot_product -> softmax -> weighted_sum -> output_proj",
                fusion_opportunities=["flash_attention", "fused_qkv_projection"],
                hardware_requirements={"tensor_cores": "fp16/bf16", "memory_bandwidth": "critical"}
            ),
            
            # Normalization operations
            "BatchNorm2d": DecomposedOperation(
                pytorch_op="BatchNorm2d",
                primitives=[
                    OperationPrimitive(
                        name="mean_computation",
                        category="reduction",
                        flops_formula="channels * height * width",
                        memory_ops_formula="channels * height * width",
                        hardware_mapping="reduction_kernel"
                    ),
                    OperationPrimitive(
                        name="variance_computation",
                        category="reduction",
                        flops_formula="2 * channels * height * width",
                        memory_ops_formula="channels * height * width",
                        hardware_mapping="reduction_kernel"
                    ),
                    OperationPrimitive(
                        name="normalize",
                        category="elementwise",
                        flops_formula="5 * channels * height * width",
                        memory_ops_formula="2 * channels * height * width",
                        hardware_mapping="elementwise_kernel",
                        notes="(x - mean) / sqrt(var + eps) * gamma + beta"
                    )
                ],
                total_flops_formula="8 * channels * height * width",
                memory_access_pattern="compute_stats -> normalize",
                fusion_opportunities=["bn_relu_fusion", "conv_bn_fusion"],
                hardware_requirements={"memory_bandwidth": "moderate"}
            ),
            
            "LayerNorm": DecomposedOperation(
                pytorch_op="LayerNorm",
                primitives=[
                    OperationPrimitive(
                        name="mean_computation",
                        category="reduction",
                        flops_formula="normalized_shape_prod",
                        memory_ops_formula="normalized_shape_prod",
                        hardware_mapping="reduction_kernel"
                    ),
                    OperationPrimitive(
                        name="variance_computation",
                        category="reduction",
                        flops_formula="2 * normalized_shape_prod",
                        memory_ops_formula="normalized_shape_prod",
                        hardware_mapping="reduction_kernel"
                    ),
                    OperationPrimitive(
                        name="normalize",
                        category="elementwise",
                        flops_formula="5 * normalized_shape_prod",
                        memory_ops_formula="2 * normalized_shape_prod",
                        hardware_mapping="elementwise_kernel"
                    )
                ],
                total_flops_formula="8 * normalized_shape_prod",
                memory_access_pattern="compute_stats -> normalize",
                fusion_opportunities=["layernorm_fusion"],
                hardware_requirements={"memory_bandwidth": "low"}
            ),
            
            # Activation operations
            "ReLU": DecomposedOperation(
                pytorch_op="ReLU",
                primitives=[
                    OperationPrimitive(
                        name="max_with_zero",
                        category="elementwise",
                        flops_formula="num_elements",
                        memory_ops_formula="2 * num_elements",
                        hardware_mapping="elementwise_kernel"
                    )
                ],
                total_flops_formula="num_elements",
                memory_access_pattern="elementwise",
                fusion_opportunities=["conv_relu_fusion", "bn_relu_fusion"],
                hardware_requirements={"memory_bandwidth": "low"}
            ),
            
            "GELU": DecomposedOperation(
                pytorch_op="GELU",
                primitives=[
                    OperationPrimitive(
                        name="gelu_computation",
                        category="elementwise",
                        flops_formula="20 * num_elements",  # Approximation using tanh
                        memory_ops_formula="2 * num_elements",
                        hardware_mapping="elementwise_kernel",
                        notes="0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"
                    )
                ],
                total_flops_formula="20 * num_elements",
                memory_access_pattern="elementwise",
                fusion_opportunities=["linear_gelu_fusion"],
                hardware_requirements={"memory_bandwidth": "low"}
            ),
            
            "Softmax": DecomposedOperation(
                pytorch_op="Softmax",
                primitives=[
                    OperationPrimitive(
                        name="max_reduction",
                        category="reduction",
                        flops_formula="dim_size",
                        memory_ops_formula="dim_size",
                        hardware_mapping="reduction_kernel"
                    ),
                    OperationPrimitive(
                        name="exp_subtract_max",
                        category="elementwise",
                        flops_formula="2 * dim_size",
                        memory_ops_formula="2 * dim_size",
                        hardware_mapping="elementwise_kernel"
                    ),
                    OperationPrimitive(
                        name="sum_reduction",
                        category="reduction",
                        flops_formula="dim_size",
                        memory_ops_formula="dim_size",
                        hardware_mapping="reduction_kernel"
                    ),
                    OperationPrimitive(
                        name="divide_by_sum",
                        category="elementwise",
                        flops_formula="dim_size",
                        memory_ops_formula="2 * dim_size",
                        hardware_mapping="elementwise_kernel"
                    )
                ],
                total_flops_formula="5 * dim_size",
                memory_access_pattern="max -> exp -> sum -> normalize",
                fusion_opportunities=["fused_softmax"],
                hardware_requirements={"memory_bandwidth": "moderate"}
            ),
            
            # Pooling operations
            "MaxPool2d": DecomposedOperation(
                pytorch_op="MaxPool2d",
                primitives=[
                    OperationPrimitive(
                        name="max_reduction_2d",
                        category="reduction",
                        flops_formula="channels * output_h * output_w * kernel_h * kernel_w",
                        memory_ops_formula="channels * (input_h * input_w + output_h * output_w)",
                        hardware_mapping="pooling_kernel"
                    )
                ],
                total_flops_formula="channels * output_h * output_w * kernel_h * kernel_w",
                memory_access_pattern="sliding_window_max",
                fusion_opportunities=["conv_pool_fusion"],
                hardware_requirements={"memory_bandwidth": "moderate"}
            ),
            
            # Recurrent operations
            "LSTM": DecomposedOperation(
                pytorch_op="LSTM",
                primitives=[
                    OperationPrimitive(
                        name="gate_gemm",
                        category="compute",
                        flops_formula="4 * 2 * (input_size + hidden_size) * hidden_size * seq_len * batch_size",
                        memory_ops_formula="4 * 2 * ((input_size + hidden_size) * hidden_size + hidden_size * batch_size)",
                        hardware_mapping="cublas_gemm",
                        notes="4 gates: input, forget, cell, output"
                    ),
                    OperationPrimitive(
                        name="gate_activation",
                        category="elementwise",
                        flops_formula="3 * 2 * hidden_size * seq_len * batch_size",  # sigmoid for 3 gates
                        memory_ops_formula="8 * hidden_size * batch_size",
                        hardware_mapping="elementwise_kernel"
                    ),
                    OperationPrimitive(
                        name="cell_update",
                        category="elementwise",
                        flops_formula="5 * hidden_size * seq_len * batch_size",  # forget*c + input*tanh
                        memory_ops_formula="4 * hidden_size * batch_size",
                        hardware_mapping="elementwise_kernel"
                    )
                ],
                total_flops_formula="8 * (input_size + hidden_size + 1) * hidden_size * seq_len * batch_size",
                memory_access_pattern="gate_compute -> activation -> cell_update",
                fusion_opportunities=["fused_lstm_cell"],
                hardware_requirements={"memory_bandwidth": "high", "sequential_dependency": True}
            ),
            
            # Special operations
            "DCNv2": DecomposedOperation(
                pytorch_op="DCNv2",
                primitives=[
                    OperationPrimitive(
                        name="offset_computation",
                        category="compute",
                        flops_formula="2 * offset_channels * out_channels * out_h * out_w",
                        memory_ops_formula="2 * (offset_channels * out_h * out_w)",
                        hardware_mapping="custom_cuda_kernel",
                        notes="Compute sampling offsets"
                    ),
                    OperationPrimitive(
                        name="mask_computation",
                        category="compute",
                        flops_formula="mask_channels * out_channels * out_h * out_w",
                        memory_ops_formula="2 * (mask_channels * out_h * out_w)",
                        hardware_mapping="custom_cuda_kernel",
                        notes="Compute modulation masks"
                    ),
                    OperationPrimitive(
                        name="deformable_im2col",
                        category="memory_transform",
                        flops_formula="2 * in_c * k_h * k_w * out_h * out_w",  # Bilinear interpolation
                        memory_ops_formula="in_c * k_h * k_w * out_h * out_w",
                        hardware_mapping="custom_cuda_kernel",
                        notes="Deformable im2col with bilinear sampling"
                    ),
                    OperationPrimitive(
                        name="modulated_gemm",
                        category="compute",
                        flops_formula="2 * (in_c * k_h * k_w) * out_c * (out_h * out_w)",
                        memory_ops_formula="2 * (in_c * k_h * k_w * out_c + out_c * out_h * out_w)",
                        hardware_mapping="cublas_gemm"
                    )
                ],
                total_flops_formula="2 * in_c * out_c * k_h * k_w * out_h * out_w + 3 * offset_overhead",
                memory_access_pattern="offset_mask_compute -> deformable_sample -> gemm",
                fusion_opportunities=["fused_dcn_kernel"],
                hardware_requirements={"custom_kernels": True, "memory_bandwidth": "very_high"}
            )
        }
    
    def _build_primitive_costs(self) -> Dict[str, Dict[str, float]]:
        """Build cost model for primitive operations"""
        return {
            "gemm": {
                "compute_intensity": 100.0,  # High compute intensity
                "memory_bandwidth_gb": 900.0,  # For V100
                "tensor_core_speedup": 8.0
            },
            "elementwise": {
                "compute_intensity": 1.0,  # Low compute intensity
                "memory_bandwidth_gb": 900.0,
                "fusion_benefit": 2.0
            },
            "reduction": {
                "compute_intensity": 5.0,
                "memory_bandwidth_gb": 900.0,
                "warp_efficiency": 0.8
            },
            "memory_transform": {
                "compute_intensity": 0.0,  # Pure memory operation
                "memory_bandwidth_gb": 900.0,
                "cache_efficiency": 0.6
            }
        }
    
    def decompose(self, operation: str) -> Optional[DecomposedOperation]:
        """Get the decomposition for a PyTorch operation"""
        return self.decomposition_map.get(operation)
    
    def get_primitives(self, operation: str) -> List[OperationPrimitive]:
        """Get the list of primitives for an operation"""
        decomp = self.decompose(operation)
        return decomp.primitives if decomp else []
    
    def estimate_flops(self, operation: str, input_shapes: Dict[str, Any]) -> int:
        """Estimate FLOPs for an operation given input shapes"""
        decomp = self.decompose(operation)
        if not decomp:
            return 0
        
        # Parse the formula and substitute values
        # This is a simplified version - in practice would use a proper expression parser
        formula = decomp.total_flops_formula
        for param, value in input_shapes.items():
            formula = formula.replace(param, str(value))
        
        try:
            return eval(formula)
        except:
            return 0
    
    def get_hardware_requirements(self, operation: str) -> Dict[str, Any]:
        """Get hardware requirements for an operation"""
        decomp = self.decompose(operation)
        return decomp.hardware_requirements if decomp else {}
    
    def analyze_fusion_opportunities(self, operation_sequence: List[str]) -> List[str]:
        """Analyze a sequence of operations for fusion opportunities"""
        fusion_ops = []
        for i in range(len(operation_sequence) - 1):
            curr_op = self.decompose(operation_sequence[i])
            next_op = self.decompose(operation_sequence[i + 1])
            
            if curr_op and next_op:
                # Check if any fusion opportunities match
                for fusion in curr_op.fusion_opportunities:
                    if operation_sequence[i+1].lower() in fusion.lower():
                        fusion_ops.append(fusion)
        
        return fusion_ops