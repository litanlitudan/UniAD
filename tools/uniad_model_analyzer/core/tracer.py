"""
Core Operation Tracer for UniAD Model Analyzer.

This module provides the central orchestration for tracing PyTorch model operations,
integrating hook management, shape recording, and memory profiling.
"""

import time
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

from .data_structures import (
    AnalysisConfig,
    TraceNode,
    MemoryProfile,
    AnalysisResult,
    TaskHead,
    AnalysisStage,
)
from .hook_manager import HookManager
from .shape_recorder import ShapeRecorder, ShapeTransformation


class OperationTracer:
    """
    Central orchestration class for tracing model operations.
    
    Integrates hook management, shape recording, and memory profiling
    to provide comprehensive analysis of UniAD model execution.
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize the operation tracer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        
        # Core components
        self.hook_manager = HookManager()
        self.shape_recorder = ShapeRecorder(cache_size=10000)
        
        # Trace data
        self.trace_nodes: List[TraceNode] = []
        self.memory_profile = MemoryProfile(
            peak_allocated=0,
            peak_reserved=0,
        )
        
        # Analysis results
        self.analysis_result: Optional[AnalysisResult] = None
        
        # Profiling state
        self.profiler: Optional[profile] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        
        # UniAD-specific tracking
        self.stage = config.stage
        self.temporal_frames = config.temporal_frames
        
    def trace_model(self,
                   model: nn.Module,
                   inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
                   additional_context: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        Trace model execution and collect comprehensive metrics.
        
        Args:
            model: PyTorch model to trace (UniAD or components)
            inputs: Model inputs (tensor, dict, or list)
            additional_context: Additional context for analysis
            
        Returns:
            AnalysisResult with complete trace and analysis
        """
        print(f"Starting trace of {model.__class__.__name__}...")
        
        # Prepare for tracing
        self._prepare_tracing(model)
        
        # Run warmup iterations if configured
        if self.config.warmup_iterations > 0:
            print(f"Running {self.config.warmup_iterations} warmup iterations...")
            self._run_warmup(model, inputs)
        
        # Main tracing loop
        print(f"Running {self.config.num_iterations} tracing iterations...")
        for iteration in range(self.config.num_iterations):
            print(f"  Iteration {iteration + 1}/{self.config.num_iterations}")
            self._trace_iteration(model, inputs, iteration)
        
        # Finalize tracing
        self._finalize_tracing()
        
        # Generate analysis result
        self.analysis_result = self._generate_analysis_result()
        
        print("Tracing complete!")
        return self.analysis_result
    
    def _prepare_tracing(self, model: nn.Module) -> None:
        """Prepare for tracing by setting up hooks and profilers."""
        # Clear any previous data
        self.trace_nodes.clear()
        self.shape_recorder.clear()
        
        # Configure hook manager
        hook_config = {
            "forward": self.config.trace_forward,
            "backward": self.config.trace_backward,
            "forward_pre": False,
        }
        
        # Register hooks with enhanced callbacks
        self._register_enhanced_hooks(model, hook_config)
        
        # Start timing
        self.start_time = time.perf_counter()
        self.hook_manager.start_tracing()
        
        # Setup PyTorch profiler if requested
        if self.config.profile_memory:
            self._setup_profiler()
    
    def _register_enhanced_hooks(self, model: nn.Module, hook_config: Dict[str, bool]) -> None:
        """Register hooks with enhanced shape and memory tracking."""
        original_forward_hook = self.hook_manager._forward_hook
        
        def enhanced_forward_hook(module: nn.Module, 
                                 input: Tuple[torch.Tensor, ...],
                                 output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                 module_path: str) -> None:
            """Enhanced forward hook with shape recording."""
            # Call original hook
            original_forward_hook(module, input, output, module_path)
            
            # Get the created node
            if self.hook_manager.trace_nodes:
                node = self.hook_manager.trace_nodes[-1]
                
                # Record shapes if configured
                if self.config.track_shapes:
                    context = {
                        "module_path": module_path,
                        "task_head": node.task_head,
                        "bev_operation": node.bev_operation,
                        "temporal_frame": node.temporal_frame,
                    }
                    
                    # Record input shapes
                    for inp in input:
                        if isinstance(inp, torch.Tensor):
                            self.shape_recorder.record_tensor(inp, context)
                    
                    # Record output shapes
                    if isinstance(output, torch.Tensor):
                        self.shape_recorder.record_tensor(output, context)
                        
                        # Record transformation if we have input
                        if input and isinstance(input[0], torch.Tensor):
                            self.shape_recorder.record_transformation(
                                input[0], output,
                                module.__class__.__name__,
                                module_path
                            )
                    elif isinstance(output, (tuple, list)):
                        for out in output:
                            if isinstance(out, torch.Tensor):
                                self.shape_recorder.record_tensor(out, context)
        
        # Replace hook with enhanced version
        self.hook_manager._forward_hook = enhanced_forward_hook
        
        # Register hooks
        self.hook_manager.register_module_hooks(model, hook_config, recursive=True)
    
    def _run_warmup(self, model: nn.Module, inputs: Any) -> None:
        """Run warmup iterations without recording."""
        # Temporarily disable recording
        original_nodes = self.hook_manager.trace_nodes
        self.hook_manager.trace_nodes = []
        
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                if isinstance(inputs, dict):
                    _ = model(**inputs)
                elif isinstance(inputs, (list, tuple)):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)
        
        # Restore recording
        self.hook_manager.trace_nodes = original_nodes
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _trace_iteration(self, 
                        model: nn.Module, 
                        inputs: Any,
                        iteration: int) -> None:
        """Run a single tracing iteration."""
        # Handle gradient tracking
        if self.config.trace_backward:
            # Ensure inputs require gradients
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.requires_grad_(True)
            elif isinstance(inputs, dict):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.requires_grad_(True)
        
        # Memory snapshot before
        if self.config.profile_memory:
            self._record_memory_snapshot("before_forward")
        
        # Forward pass
        if self.config.trace_backward:
            # With gradients
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
            
            # Compute loss for backward
            if isinstance(outputs, dict):
                loss = sum(o.mean() for o in outputs.values() if isinstance(o, torch.Tensor))
            elif isinstance(outputs, (tuple, list)):
                loss = sum(o.mean() for o in outputs if isinstance(o, torch.Tensor))
            else:
                loss = outputs.mean()
            
            # Memory snapshot after forward
            if self.config.profile_memory:
                self._record_memory_snapshot("after_forward")
            
            # Backward pass
            loss.backward()
            
            # Memory snapshot after backward
            if self.config.profile_memory:
                self._record_memory_snapshot("after_backward")
        else:
            # Without gradients
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                elif isinstance(inputs, (list, tuple)):
                    outputs = model(*inputs)
                else:
                    outputs = model(inputs)
            
            # Memory snapshot after forward
            if self.config.profile_memory:
                self._record_memory_snapshot("after_forward")
        
        # Store trace nodes from this iteration
        self.trace_nodes.extend(self.hook_manager.trace_nodes[len(self.trace_nodes):])
    
    def _record_memory_snapshot(self, stage: str) -> None:
        """Record memory snapshot at different stages."""
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated()
            current_reserved = torch.cuda.memory_reserved()
            
            # Update peaks
            self.memory_profile.peak_allocated = max(
                self.memory_profile.peak_allocated,
                current_allocated
            )
            self.memory_profile.peak_reserved = max(
                self.memory_profile.peak_reserved,
                current_reserved
            )
            
            # Update current
            self.memory_profile.current_allocated = current_allocated
            self.memory_profile.current_reserved = current_reserved
            
            # Record timeline
            timestamp = time.perf_counter() - self.start_time
            self.memory_profile.memory_timeline.append((timestamp, current_allocated))
    
    def _setup_profiler(self) -> None:
        """Setup PyTorch profiler for detailed profiling."""
        # Note: PyTorch profiler integration is optional
        # We handle memory profiling manually for better control
        pass
    
    def _finalize_tracing(self) -> None:
        """Finalize tracing and collect final metrics."""
        # Stop timing
        self.end_time = time.perf_counter()
        
        # Get final trace nodes
        final_nodes = self.hook_manager.stop_tracing()
        if final_nodes and len(final_nodes) > len(self.trace_nodes):
            self.trace_nodes = final_nodes
        
        # Cleanup hooks
        self.hook_manager.cleanup_hooks()
        
        # Finalize profiler (if we use it in the future)
        # Currently handled manually
        
        # Analyze memory by component
        self._analyze_memory_breakdown()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _analyze_memory_breakdown(self) -> None:
        """Analyze memory usage by component."""
        # Group nodes by task head
        for node in self.trace_nodes:
            if node.task_head != TaskHead.NONE:
                current = self.memory_profile.task_head_memory.get(node.task_head, 0)
                self.memory_profile.task_head_memory[node.task_head] = (
                    current + node.get_cuda_memory_delta()
                )
        
        # Analyze BEV memory
        bev_nodes = [n for n in self.trace_nodes if n.bev_operation]
        if bev_nodes:
            self.memory_profile.bev_encoder_memory = sum(
                n.get_cuda_memory_delta() for n in bev_nodes
            )
        
        # Analyze temporal memory
        for node in self.trace_nodes:
            if node.temporal_frame is not None:
                if node.temporal_frame < len(self.memory_profile.temporal_memory):
                    self.memory_profile.temporal_memory[node.temporal_frame] += (
                        node.get_cuda_memory_delta()
                    )
                else:
                    self.memory_profile.temporal_memory.append(
                        node.get_cuda_memory_delta()
                    )
    
    def _generate_analysis_result(self) -> AnalysisResult:
        """Generate comprehensive analysis result."""
        # Calculate statistics
        total_duration = (self.end_time - self.start_time) * 1000  # Convert to ms
        total_operations = len(self.trace_nodes)
        
        # Calculate total parameters
        unique_modules = set()
        total_params = 0
        for node in self.trace_nodes:
            if node.module_path not in unique_modules:
                unique_modules.add(node.module_path)
                if node.parameters:
                    total_params += node.parameters
        
        # Task head statistics
        task_head_stats = {}
        for task_head in TaskHead:
            head_nodes = [n for n in self.trace_nodes if n.task_head == task_head]
            if head_nodes:
                task_head_stats[task_head] = {
                    "operation_count": len(head_nodes),
                    "total_duration_ms": sum(n.duration for n in head_nodes) / 1e6,
                    "avg_duration_ms": (sum(n.duration for n in head_nodes) / len(head_nodes)) / 1e6,
                    "memory_mb": sum(n.get_cuda_memory_delta() for n in head_nodes) / (1024 * 1024),
                }
        
        # BEV statistics
        bev_nodes = [n for n in self.trace_nodes if n.bev_operation]
        bev_stats = {
            "operation_count": len(bev_nodes),
            "total_duration_ms": sum(n.duration for n in bev_nodes) / 1e6 if bev_nodes else 0,
            "memory_mb": sum(n.get_cuda_memory_delta() for n in bev_nodes) / (1024 * 1024) if bev_nodes else 0,
        }
        
        # Shape analysis
        shape_analysis = self.shape_recorder.analyze_shape_patterns()
        
        # Create result
        result = AnalysisResult(
            config=self.config,
            trace_graph=self.trace_nodes,
            memory_profile=self.memory_profile,
            total_operations=total_operations,
            total_duration_ms=total_duration,
            total_parameters=total_params,
            task_head_stats=task_head_stats,
            bev_stats=bev_stats,
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            analysis_duration_s=self.end_time - self.start_time,
        )
        
        # Add shape analysis to result
        result.temporal_stats = shape_analysis.get("temporal_analysis", {})
        
        # Generate recommendations
        self._generate_recommendations(result)
        
        return result
    
    def _generate_recommendations(self, result: AnalysisResult) -> None:
        """Generate optimization recommendations based on analysis."""
        # Memory recommendations
        if self.memory_profile.peak_allocated > 30 * 1024 * 1024 * 1024:  # > 30GB
            result.add_recommendation(
                "High memory usage detected (>30GB). Consider mixed precision training or gradient checkpointing."
            )
        
        # BEV recommendations
        if self.stage == AnalysisStage.STAGE_2 and self.memory_profile.bev_encoder_memory > 0:
            if self.config.verify_frozen_bev:
                # Check if BEV encoder has gradients (should not in Stage 2)
                bev_with_grads = [
                    n for n in self.trace_nodes 
                    if n.bev_operation and n.gradient_info is not None
                ]
                if bev_with_grads:
                    result.add_warning(
                        "BEV encoder has gradients in Stage 2 - it should be frozen!"
                    )
        
        # Task head balance
        if result.task_head_stats:
            memory_values = [stats["memory_mb"] for stats in result.task_head_stats.values()]
            if memory_values:
                max_mem = max(memory_values)
                min_mem = min(memory_values)
                if max_mem > 2 * min_mem and min_mem > 0:
                    result.add_recommendation(
                        "Task head memory imbalance detected. Consider adjusting loss weights or model capacity."
                    )
        
        # Temporal recommendations
        if self.memory_profile.temporal_memory:
            temporal_variation = max(self.memory_profile.temporal_memory) - min(self.memory_profile.temporal_memory)
            if temporal_variation > 1024 * 1024 * 1024:  # > 1GB variation
                result.add_recommendation(
                    "High temporal memory variation. Consider temporal attention optimization."
                )