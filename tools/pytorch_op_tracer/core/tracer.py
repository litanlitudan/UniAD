"""Core tracer for PyTorch operations with UniAD awareness"""

import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .data_structures import TraceNode
from .shape_recorder import TensorShapeRecorder


class OperationTracer:
    """Core tracer for PyTorch operations with UniAD awareness"""
    
    def __init__(self, model: nn.Module, trace_backward: bool = False, 
                 filter_ops: Optional[List[str]] = None,
                 stage: int = 2, task_heads: Optional[List[str]] = None):
        self.model = model
        self.trace_backward = trace_backward
        self.filter_ops = filter_ops
        self.stage = stage
        self.task_heads = task_heads or ['track', 'seg', 'motion', 'occ', 'planning']
        
        self.trace_nodes: List[TraceNode] = []
        self.hooks: List[RemovableHandle] = []
        self.module_to_head: Dict[str, str] = {}
        self.execution_order = 0
        
        # Identify task heads in the model
        self._identify_task_heads()
    
    def _identify_task_heads(self):
        """Map modules to their corresponding task heads"""
        for name, module in self.model.named_modules():
            # UniAD-specific head detection
            if 'track_head' in name or 'pts_bbox_head' in name:
                self.module_to_head[name] = 'track'
            elif 'seg_head' in name or 'panseg' in name:
                self.module_to_head[name] = 'seg'
            elif 'motion_head' in name:
                self.module_to_head[name] = 'motion'
            elif 'occ_head' in name:
                self.module_to_head[name] = 'occ'
            elif 'planning_head' in name:
                self.module_to_head[name] = 'planning'
            elif 'bev' in name.lower():
                self.module_to_head[name] = 'bev'
    
    def _get_task_head(self, module_path: str) -> Optional[str]:
        """Determine which task head a module belongs to"""
        for path, head in self.module_to_head.items():
            if path in module_path:
                return head
        return None
    
    def _create_forward_hook(self, name: str):
        """Create a forward hook for a module"""
        def hook(module, input, output):
            if self.filter_ops and module.__class__.__name__ not in self.filter_ops:
                return
            
            start_time = time.time()
            
            # Extract shapes
            input_shapes = TensorShapeRecorder.extract_shapes(input)
            output_shapes = TensorShapeRecorder.extract_shapes(output)
            
            # Detect shape transformation
            shape_transform = TensorShapeRecorder.detect_shape_transformation(
                input_shapes, output_shapes
            )
            
            # Create trace node
            node = TraceNode(
                operation=module.__class__.__name__,
                module_path=name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                task_head=self._get_task_head(name),
                is_bev_operation='bev' in name.lower(),
                memory_usage=TensorShapeRecorder.estimate_memory(output_shapes),
                compute_time=(time.time() - start_time) * 1000,
                is_frozen=not any(p.requires_grad for p in module.parameters()),
                shape_transform=shape_transform
            )
            
            # Check for BEV grid size
            if node.is_bev_operation and output_shapes:
                for tensor_info in output_shapes:
                    if len(tensor_info.shape) >= 4:  # B, C, H, W
                        node.bev_grid_size = (tensor_info.shape[-2], tensor_info.shape[-1])
                        break
            
            self.trace_nodes.append(node)
            self.execution_order += 1
            
        return hook
    
    def register_hooks(self):
        """Register hooks on model modules"""
        # Define which modules are "meaningful" - either leaf modules or important containers
        meaningful_modules = set()
        
        # Patterns for meaningful intermediate modules in UniAD
        important_patterns = [
            'BEVFormer', 'BEVEncoder', 'BEVDecoder',
            'TrackHead', 'MotionHead', 'OccHead', 'PlanningHead', 'SegHead',
            'PansegformerHead', 'BEVFormerTrackHead',
            'Transformer', 'Encoder', 'Decoder',
            'SelfAttention', 'CrossAttention', 'MultiheadAttention',
            'TemporalSelfAttention', 'SpatialCrossAttention',
            'ResNet', 'FPN', 'ConvModule',
            'FFN', 'MLP',
        ]
        
        for name, module in self.model.named_modules():
            module_class = module.__class__.__name__
            
            # Include if it's a leaf module
            is_leaf = len(list(module.children())) == 0
            
            # Include if it matches important patterns
            is_important = any(pattern in module_class for pattern in important_patterns)
            
            # Include if it's a task head module
            is_task_head = any(head in name for head in ['track_head', 'seg_head', 'motion_head', 
                                                         'occ_head', 'planning_head', 'pts_bbox_head'])
            
            # Include if it's a BEV-related module
            is_bev_module = 'bev' in name.lower() or 'bev' in module_class.lower()
            
            if is_leaf or is_important or is_task_head or is_bev_module:
                meaningful_modules.add((name, module))
        
        # Register hooks on meaningful modules
        for name, module in meaningful_modules:
            hook = module.register_forward_hook(self._create_forward_hook(name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def trace(self, inputs: Any):
        """Perform tracing with given inputs"""
        self.trace_nodes.clear()
        self.execution_order = 0
        
        # Set model to eval mode
        self.model.eval()
        
        # Register hooks
        self.register_hooks()
        
        try:
            with torch.no_grad():
                # Forward pass
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
        finally:
            # Clean up hooks
            self.remove_hooks()
        
        # Build dependencies
        self._build_dependencies()
        
        return outputs
    
    def _build_dependencies(self):
        """Build dependency graph between operations"""
        # Simple heuristic: operations are dependent on previous operations in the same head
        head_last_node = {}
        
        for node in self.trace_nodes:
            if node.task_head:
                if node.task_head in head_last_node:
                    prev_node = head_last_node[node.task_head]
                    node.depends_on.append(prev_node.node_id)
                    prev_node.feeds_into.append(node.node_id)
                head_last_node[node.task_head] = node
    
    def get_trace_data(self) -> List[TraceNode]:
        """Return collected trace data"""
        return self.trace_nodes