"""
Hook Manager for UniAD Model Analyzer.

This module manages PyTorch hooks for tracing model operations with minimal overhead.
Follows Meta's automated trace patterns for efficient profiling.
"""

import time
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .data_structures import TraceNode, TensorShape, TaskHead


class HookManager:
    """
    Manages forward and backward hooks for PyTorch model tracing.
    
    This class provides a centralized way to register, manage, and clean up
    hooks on PyTorch modules for operation tracing.
    """
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks: List[RemovableHandle] = []
        self.trace_nodes: List[TraceNode] = []
        self.node_counter: int = 0
        self.module_to_node: Dict[nn.Module, str] = {}
        self.active_nodes: List[TraceNode] = []
        self.task_head_mapping: Dict[str, TaskHead] = self._init_task_head_mapping()
        
        # Memory tracking
        self.initial_memory: int = 0
        self.initial_cuda_memory: int = 0
        
        # Module path tracking
        self.module_paths: Dict[nn.Module, str] = {}
        
        # Optimizer hook handles
        self.optimizer_hooks: List[Callable] = []
        
        # Timing
        self.start_time: float = 0.0
        
    def _init_task_head_mapping(self) -> Dict[str, TaskHead]:
        """Initialize mapping of module names to task heads."""
        return {
            "track_head": TaskHead.TRACK,
            "seg_head": TaskHead.SEGMENTATION,
            "motion_head": TaskHead.MOTION,
            "occ_head": TaskHead.OCCUPANCY,
            "planning_head": TaskHead.PLANNING,
            "bev_encoder": TaskHead.NONE,  # BEV encoder is shared
            "img_backbone": TaskHead.NONE,  # Image backbone is shared
        }
    
    def _get_module_path(self, module: nn.Module, parent_name: str = "") -> str:
        """Get the full module path in the model hierarchy."""
        if module in self.module_paths:
            return self.module_paths[module]
        return parent_name if parent_name else module.__class__.__name__
    
    def _identify_task_head(self, module_path: str) -> TaskHead:
        """Identify which task head a module belongs to."""
        path_lower = module_path.lower()
        for key, task_head in self.task_head_mapping.items():
            if key in path_lower:
                return task_head
        return TaskHead.NONE
    
    def _is_bev_operation(self, module_path: str) -> bool:
        """Check if the operation is BEV-related."""
        bev_keywords = ["bev", "bird", "spatial", "deform_attn", "spatial_cross_attention"]
        path_lower = module_path.lower()
        return any(keyword in path_lower for keyword in bev_keywords)
    
    def _get_memory_usage(self) -> Tuple[int, int]:
        """Get current memory usage (CPU and CUDA)."""
        cpu_memory = 0  # PyTorch doesn't provide easy CPU memory tracking
        cuda_memory = 0
        
        if torch.cuda.is_available():
            cuda_memory = torch.cuda.memory_allocated()
        
        return cpu_memory, cuda_memory
    
    def _create_trace_node(self, module: nn.Module, module_path: str) -> TraceNode:
        """Create a new trace node for a module."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        # Get current memory
        cpu_mem, cuda_mem = self._get_memory_usage()
        
        node = TraceNode(
            id=node_id,
            name=module.__class__.__name__,
            module_path=module_path,
            start_time=time.perf_counter_ns() - self.start_time,
            duration=0.0,
            memory_allocated=0,
            memory_freed=0,
            cuda_memory_allocated=cuda_mem - self.initial_cuda_memory,
            cuda_memory_freed=0,
            task_head=self._identify_task_head(module_path),
            bev_operation=self._is_bev_operation(module_path),
        )
        
        # Count parameters if module has them
        try:
            node.parameters = sum(p.numel() for p in module.parameters())
        except Exception:
            node.parameters = 0
        
        return node
    
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], 
                     output: torch.Tensor, module_path: str) -> None:
        """Forward hook to trace operation execution."""
        # Create trace node
        node = self._create_trace_node(module, module_path)
        
        # Record input shapes
        for inp in input:
            if isinstance(inp, torch.Tensor):
                node.input_shapes.append(TensorShape.from_tensor(inp))
        
        # Record output shapes
        if isinstance(output, torch.Tensor):
            node.output_shapes.append(TensorShape.from_tensor(output))
        elif isinstance(output, (tuple, list)):
            for out in output:
                if isinstance(out, torch.Tensor):
                    node.output_shapes.append(TensorShape.from_tensor(out))
        
        # Update timing
        node.duration = time.perf_counter_ns() - self.start_time - node.start_time
        
        # Update memory
        cpu_mem, cuda_mem = self._get_memory_usage()
        node.cuda_memory_freed = cuda_mem - node.cuda_memory_allocated - self.initial_cuda_memory
        
        # Store node
        self.trace_nodes.append(node)
        self.module_to_node[module] = node.id
        
        # Link to parent if in active stack
        if self.active_nodes:
            parent = self.active_nodes[-1]
            node.parent = parent.id
            parent.children.append(node.id)
        
        self.active_nodes.append(node)
    
    def _forward_pre_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...]) -> None:
        """Pre-forward hook to track entry into module."""
        pass  # Currently unused but can be extended
    
    def _backward_hook(self, module: nn.Module, grad_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                      grad_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> None:
        """Backward hook to trace gradient computation."""
        if module in self.module_to_node:
            node_id = self.module_to_node[module]
            # Find the corresponding node
            for node in self.trace_nodes:
                if node.id == node_id:
                    # Record gradient information
                    node.gradient_info = {
                        "grad_input_shapes": [],
                        "grad_output_shapes": [],
                    }
                    
                    # Handle both single tensor and tuple of tensors
                    if isinstance(grad_input, torch.Tensor):
                        grad_inputs = (grad_input,)
                    else:
                        grad_inputs = grad_input if grad_input else ()
                    
                    for grad in grad_inputs:
                        if isinstance(grad, torch.Tensor):
                            node.gradient_info["grad_input_shapes"].append(
                                TensorShape.from_tensor(grad)
                            )
                    
                    if isinstance(grad_output, torch.Tensor):
                        grad_outputs = (grad_output,)
                    else:
                        grad_outputs = grad_output if grad_output else ()
                    
                    for grad in grad_outputs:
                        if isinstance(grad, torch.Tensor):
                            node.gradient_info["grad_output_shapes"].append(
                                TensorShape.from_tensor(grad)
                            )
                    break
    
    def register_module_hooks(self, module: nn.Module, 
                            hook_config: Optional[Dict[str, Any]] = None,
                            recursive: bool = True,
                            parent_name: str = "") -> None:
        """
        Register hooks on a module and optionally its submodules.
        
        Args:
            module: PyTorch module to hook
            hook_config: Configuration for hooks (e.g., which hooks to register)
            recursive: Whether to recursively register on submodules
            parent_name: Parent module name for path construction
        """
        if hook_config is None:
            hook_config = {
                "forward": True,
                "backward": False,
                "forward_pre": False,
            }
        
        # Build module path
        module_name = module.__class__.__name__
        if hasattr(module, '__name__'):
            module_name = module.__name__
        
        if parent_name:
            module_path = f"{parent_name}.{module_name}"
        else:
            module_path = module_name
        
        self.module_paths[module] = str(module_path)
        
        # Register forward hook
        if hook_config.get("forward", True):
            handle = module.register_forward_hook(
                lambda m, i, o: self._forward_hook(m, i, o, str(module_path))
            )
            self.hooks.append(handle)
        
        # Register backward hook
        if hook_config.get("backward", False):
            # Use register_backward_hook instead of register_full_backward_hook
            # to avoid issues with gradient flow
            handle = module.register_backward_hook(self._backward_hook)
            self.hooks.append(handle)
        
        # Register forward pre-hook
        if hook_config.get("forward_pre", False):
            handle = module.register_forward_pre_hook(self._forward_pre_hook)
            self.hooks.append(handle)
        
        # Recursively register on submodules
        if recursive:
            for name, child in module.named_children():
                self.register_module_hooks(
                    child, 
                    hook_config, 
                    recursive=True,
                    parent_name=str(module_path)
                )
    
    def register_optimizer_hooks(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Register hooks for optimizer step tracking.
        
        Args:
            optimizer: PyTorch optimizer to hook
        """
        original_step = optimizer.step
        
        def hooked_step(closure=None):
            """Wrapped optimizer step with tracing."""
            # Record pre-step state
            pre_step_time = time.perf_counter_ns()
            
            # Call original step
            loss = original_step(closure)
            
            # Record post-step state
            post_step_time = time.perf_counter_ns()
            
            # Create a trace node for optimizer step
            node = TraceNode(
                id=f"optimizer_{self.node_counter}",
                name="OptimizerStep",
                module_path=f"optimizer.{optimizer.__class__.__name__}",
                start_time=pre_step_time - self.start_time,
                duration=post_step_time - pre_step_time,
                memory_allocated=0,
                memory_freed=0,
                operation_type="optimizer_step",
            )
            self.node_counter += 1
            self.trace_nodes.append(node)
            
            return loss
        
        optimizer.step = hooked_step
        self.optimizer_hooks.append(original_step)
    
    def cleanup_hooks(self) -> None:
        """
        Remove all registered hooks safely.
        
        This method ensures all hooks are properly removed to prevent
        memory leaks and allow normal model execution.
        """
        # Remove module hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Clear trace data
        self.trace_nodes.clear()
        self.module_to_node.clear()
        self.active_nodes.clear()
        self.module_paths.clear()
        
        # Note: Optimizer hooks need to be restored manually
        # This would require keeping references to original methods
        self.optimizer_hooks.clear()
        
        # Reset counters
        self.node_counter = 0
        self.start_time = 0.0
    
    def start_tracing(self) -> None:
        """Start tracing by recording initial state."""
        self.start_time = time.perf_counter_ns()
        self.initial_memory, self.initial_cuda_memory = self._get_memory_usage()
    
    def stop_tracing(self) -> List[TraceNode]:
        """Stop tracing and return collected trace nodes."""
        # Finalize any remaining active nodes
        self.active_nodes.clear()
        return self.trace_nodes.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the traced operations."""
        if not self.trace_nodes:
            return {}
        
        total_duration = sum(node.duration for node in self.trace_nodes)
        total_memory = sum(node.get_memory_delta() for node in self.trace_nodes)
        total_cuda_memory = sum(node.get_cuda_memory_delta() for node in self.trace_nodes)
        
        # Group by task head
        task_head_stats = defaultdict(lambda: {"count": 0, "duration": 0, "memory": 0})
        for node in self.trace_nodes:
            stats = task_head_stats[node.task_head]
            stats["count"] += 1
            stats["duration"] += int(node.duration)
            stats["memory"] += node.get_cuda_memory_delta()
        
        return {
            "total_operations": len(self.trace_nodes),
            "total_duration_ns": total_duration,
            "total_memory_bytes": total_memory,
            "total_cuda_memory_bytes": total_cuda_memory,
            "task_head_stats": dict(task_head_stats),
            "bev_operations": sum(1 for node in self.trace_nodes if node.bev_operation),
        }


def register_hooks_on_model(model: nn.Module, 
                           config: Optional[Dict[str, Any]] = None) -> HookManager:
    """
    Convenience function to register hooks on a model.
    
    Args:
        model: PyTorch model to instrument
        config: Hook configuration
    
    Returns:
        HookManager instance with registered hooks
    """
    manager = HookManager()
    manager.start_tracing()
    manager.register_module_hooks(model, config)
    return manager