"""Enhanced Profiler with PyTorch Profiler Integration"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import json
from collections import defaultdict

from .operation_decomposition import OperationDecomposer, DecomposedOperation


@dataclass
class KernelInfo:
    """Information about a CUDA kernel launch"""
    name: str
    duration_us: float
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    registers_per_thread: int
    shared_memory_bytes: int
    occupancy: float
    kernel_type: str  # cublas, cudnn, custom, etc.


@dataclass
class EnhancedTraceNode:
    """Enhanced trace node with low-level details"""
    # Basic info
    operation: str
    module_name: str
    
    # Shapes and dtypes
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    input_dtypes: List[torch.dtype]
    output_dtypes: List[torch.dtype]
    
    # Timing
    cpu_time_ms: float
    cuda_time_ms: float
    
    # Memory
    memory_allocated_mb: float
    memory_reserved_mb: float
    memory_active_mb: float
    
    # Kernels
    cuda_kernels: List[KernelInfo] = field(default_factory=list)
    
    # Decomposition
    decomposed_ops: Optional[DecomposedOperation] = None
    estimated_flops: int = 0
    
    # Hardware utilization
    tensor_core_eligible: bool = False
    memory_bandwidth_pct: float = 0.0
    compute_utilization_pct: float = 0.0


class EnhancedProfiler:
    """Enhanced profiler that captures low-level implementation details"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.decomposer = OperationDecomposer()
        self.trace_nodes: List[EnhancedTraceNode] = []
        self.current_module_stack: List[str] = []
        self.gpu_properties = self._get_gpu_properties() if device == 'cuda' else None
        
    def _get_gpu_properties(self) -> Dict[str, Any]:
        """Get GPU properties for hardware-aware analysis"""
        if not torch.cuda.is_available():
            return {}
            
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        
        return {
            "name": props.name,
            "compute_capability": (props.major, props.minor),
            "total_memory_gb": props.total_memory / (1024**3),
            "memory_bandwidth_gb_s": self._estimate_memory_bandwidth(props.name),
            "tensor_cores": props.major >= 7,  # Volta and newer
            "fp16_tflops": self._estimate_fp16_tflops(props.name),
            "fp32_tflops": self._estimate_fp32_tflops(props.name),
            "sm_count": props.multi_processor_count,
            "max_threads_per_sm": props.max_threads_per_multiprocessor if hasattr(props, 'max_threads_per_multiprocessor') else 2048
        }
    
    def _estimate_memory_bandwidth(self, gpu_name: str) -> float:
        """Estimate memory bandwidth based on GPU model"""
        bandwidth_map = {
            "V100": 900,
            "A100": 1555,
            "A10": 600,
            "RTX 3090": 936,
            "RTX 4090": 1008,
            "H100": 3350
        }
        
        for key, bandwidth in bandwidth_map.items():
            if key in gpu_name:
                return bandwidth
        return 500  # Default estimate
    
    def _estimate_fp16_tflops(self, gpu_name: str) -> float:
        """Estimate FP16 TFLOPS based on GPU model"""
        tflops_map = {
            "V100": 125,
            "A100": 312,
            "A10": 125,
            "RTX 3090": 71,
            "RTX 4090": 165,
            "H100": 989
        }
        
        for key, tflops in tflops_map.items():
            if key in gpu_name:
                return tflops
        return 50  # Default estimate
    
    def _estimate_fp32_tflops(self, gpu_name: str) -> float:
        """Estimate FP32 TFLOPS based on GPU model"""
        tflops_map = {
            "V100": 15.7,
            "A100": 19.5,
            "A10": 31.2,
            "RTX 3090": 35.6,
            "RTX 4090": 82.6,
            "H100": 67
        }
        
        for key, tflops in tflops_map.items():
            if key in gpu_name:
                return tflops
        return 10  # Default estimate
    
    def profile_model(self, model: nn.Module, inputs: Any, 
                     capture_kernels: bool = True,
                     warmup_runs: int = 3) -> List[EnhancedTraceNode]:
        """Profile a model with enhanced details"""
        model.to(self.device)
        
        # Move inputs to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, (list, tuple)):
            inputs = [inp.to(self.device) if isinstance(inp, torch.Tensor) else inp 
                     for inp in inputs]
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(inputs)
        
        # Clear any existing traces
        self.trace_nodes.clear()
        
        # Profile with PyTorch profiler
        activities = [ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        with profile(activities=activities, 
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True) as prof:
            # Register hooks for module-level tracking
            hooks = []
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    hook = module.register_forward_hook(
                        self._create_forward_hook(name, module)
                    )
                    hooks.append(hook)
            
            # Run the model
            with torch.no_grad():
                outputs = model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Process profiler results
        self._process_profiler_events(prof, capture_kernels)
        
        return self.trace_nodes
    
    def _create_forward_hook(self, module_name: str, module: nn.Module):
        """Create a forward hook for a module"""
        def hook(module, inputs, outputs):
            # Record module entry
            self.current_module_stack.append(module_name)
            
            # Create a custom record for this module
            with record_function(f"{module.__class__.__name__}::{module_name}"):
                pass
            
            # Record module exit
            self.current_module_stack.pop()
        
        return hook
    
    def _process_profiler_events(self, prof, capture_kernels: bool):
        """Process events from PyTorch profiler"""
        # Group events by module
        module_events = defaultdict(list)
        
        for event in prof.events():
            if "::" in event.name:
                op_type, module_name = event.name.split("::", 1)
                module_events[module_name].append(event)
        
        # Process each module's events
        for module_name, events in module_events.items():
            if not events:
                continue
                
            # Aggregate timing and memory
            cpu_time_us = sum(e.cpu_time_total for e in events)
            cuda_time_us = sum(e.cuda_time_total for e in events) if self.device.type == 'cuda' else 0
            
            # Get shapes from the first event
            input_shapes = []
            output_shapes = []
            if events[0].input_shapes:
                input_shapes = [shape for shape in events[0].input_shapes]
                
            # Estimate memory
            memory_mb = 0
            if hasattr(events[0], 'cpu_memory_usage'):
                memory_mb = events[0].cpu_memory_usage / (1024 * 1024)
            
            # Extract operation type
            op_type = events[0].name.split("::")[0] if "::" in events[0].name else events[0].name
            
            # Get decomposition
            decomposed = self.decomposer.decompose(op_type)
            
            # Create enhanced node
            node = EnhancedTraceNode(
                operation=op_type,
                module_name=module_name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=[torch.float32],  # Default, would need to track actual
                output_dtypes=[torch.float32],
                cpu_time_ms=cpu_time_us / 1000,
                cuda_time_ms=cuda_time_us / 1000,
                memory_allocated_mb=memory_mb,
                memory_reserved_mb=memory_mb * 1.2,  # Estimate
                memory_active_mb=memory_mb,
                decomposed_ops=decomposed
            )
            
            # Extract CUDA kernels if requested
            if capture_kernels and self.device.type == 'cuda':
                node.cuda_kernels = self._extract_cuda_kernels(events)
            
            # Estimate hardware utilization
            if decomposed and self.gpu_properties:
                node = self._estimate_hardware_utilization(node)
            
            self.trace_nodes.append(node)
    
    def _extract_cuda_kernels(self, events) -> List[KernelInfo]:
        """Extract CUDA kernel information from events"""
        kernels = []
        
        # Note: PyTorch profiler's kernel information access varies by version
        # This is a simplified version that works with basic profiling
        for event in events:
            if hasattr(event, 'name') and event.cuda_time_total > 0:
                # Create a simplified kernel info based on event name
                kernel_info = KernelInfo(
                    name=event.name,
                    duration_us=event.cuda_time_total,
                    grid_size=(1, 1, 1),  # Would need actual kernel launch params
                    block_size=(1, 1, 1),  # Would need actual kernel launch params
                    registers_per_thread=0,
                    shared_memory_bytes=0,
                    occupancy=0.0,
                    kernel_type=self._classify_kernel(event.name)
                )
                kernels.append(kernel_info)
        
        return kernels
    
    def _classify_kernel(self, kernel_name: str) -> str:
        """Classify a kernel by its type"""
        kernel_name_lower = kernel_name.lower()
        
        if "gemm" in kernel_name_lower or "blas" in kernel_name_lower:
            return "cublas"
        elif "cudnn" in kernel_name_lower:
            return "cudnn"
        elif "elementwise" in kernel_name_lower:
            return "elementwise"
        elif "reduction" in kernel_name_lower:
            return "reduction"
        else:
            return "custom"
    
    def _estimate_hardware_utilization(self, node: EnhancedTraceNode) -> EnhancedTraceNode:
        """Estimate hardware utilization for a node"""
        if not node.decomposed_ops:
            return node
        
        # Check tensor core eligibility
        if node.input_dtypes[0] in [torch.float16, torch.bfloat16]:
            for primitive in node.decomposed_ops.primitives:
                if "gemm" in primitive.name.lower():
                    node.tensor_core_eligible = True
                    break
        
        # Estimate FLOPS
        shape_params = self._extract_shape_params(node)
        node.estimated_flops = self.decomposer.estimate_flops(
            node.operation, shape_params
        )
        
        # Estimate utilization
        if node.cuda_time_ms > 0:
            # Compute utilization
            achieved_tflops = node.estimated_flops / (node.cuda_time_ms * 1e9)
            if node.tensor_core_eligible:
                peak_tflops = self.gpu_properties["fp16_tflops"]
            else:
                peak_tflops = self.gpu_properties["fp32_tflops"]
            
            node.compute_utilization_pct = (achieved_tflops / peak_tflops) * 100
            
            # Memory bandwidth utilization
            # Simplified: assume we need to read inputs and write outputs
            total_memory_bytes = sum(
                torch.tensor(shape).prod().item() * 4  # Assume 4 bytes per element
                for shape in node.input_shapes + node.output_shapes
            )
            achieved_bandwidth_gb_s = (total_memory_bytes / 1e9) / (node.cuda_time_ms / 1000)
            peak_bandwidth = self.gpu_properties["memory_bandwidth_gb_s"]
            
            node.memory_bandwidth_pct = (achieved_bandwidth_gb_s / peak_bandwidth) * 100
        
        return node
    
    def _extract_shape_params(self, node: EnhancedTraceNode) -> Dict[str, int]:
        """Extract shape parameters for FLOP calculation"""
        params = {}
        
        if node.operation == "Conv2d" and node.input_shapes:
            # Assume NCHW format
            if len(node.input_shapes[0]) == 4:
                params["batch_size"] = node.input_shapes[0][0]
                params["in_c"] = node.input_shapes[0][1]
                params["out_c"] = node.output_shapes[0][1] if node.output_shapes else params["in_c"]
                params["in_h"] = node.input_shapes[0][2]
                params["in_w"] = node.input_shapes[0][3]
                params["out_h"] = node.output_shapes[0][2] if node.output_shapes else params["in_h"]
                params["out_w"] = node.output_shapes[0][3] if node.output_shapes else params["in_w"]
                # Kernel size would need to be tracked separately
                params["k_h"] = 3  # Default
                params["k_w"] = 3  # Default
        
        elif node.operation == "Linear" and node.input_shapes:
            if len(node.input_shapes[0]) >= 2:
                params["batch_size"] = node.input_shapes[0][0]
                params["in_features"] = node.input_shapes[0][-1]
                params["out_features"] = node.output_shapes[0][-1] if node.output_shapes else params["in_features"]
        
        # Add more operation-specific parameter extraction as needed
        
        return params
    
    def generate_report(self, output_path: str):
        """Generate a detailed profiling report"""
        report = {
            "gpu_properties": self.gpu_properties,
            "total_operations": len(self.trace_nodes),
            "operations": []
        }
        
        for node in self.trace_nodes:
            op_info = {
                "operation": node.operation,
                "module": node.module_name,
                "timing": {
                    "cpu_ms": node.cpu_time_ms,
                    "cuda_ms": node.cuda_time_ms
                },
                "memory": {
                    "allocated_mb": node.memory_allocated_mb,
                    "reserved_mb": node.memory_reserved_mb
                },
                "shapes": {
                    "inputs": [list(s) for s in node.input_shapes],
                    "outputs": [list(s) for s in node.output_shapes]
                },
                "hardware": {
                    "tensor_core_eligible": node.tensor_core_eligible,
                    "compute_utilization_%": node.compute_utilization_pct,
                    "memory_bandwidth_%": node.memory_bandwidth_pct,
                    "estimated_flops": node.estimated_flops
                }
            }
            
            # Add kernel information
            if node.cuda_kernels:
                op_info["cuda_kernels"] = [
                    {
                        "name": k.name,
                        "duration_us": k.duration_us,
                        "type": k.kernel_type,
                        "occupancy": k.occupancy
                    }
                    for k in node.cuda_kernels
                ]
            
            # Add decomposition information
            if node.decomposed_ops:
                op_info["decomposition"] = {
                    "primitives": [
                        {
                            "name": p.name,
                            "category": p.category,
                            "hardware_mapping": p.hardware_mapping
                        }
                        for p in node.decomposed_ops.primitives
                    ],
                    "fusion_opportunities": node.decomposed_ops.fusion_opportunities
                }
            
            report["operations"].append(op_info)
        
        # Write report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)