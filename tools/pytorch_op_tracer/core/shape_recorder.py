"""Tensor shape recording utilities"""

from typing import Any, List, Tuple, Dict, Optional
import torch

from .data_structures import TensorInfo


# Common dimension semantics in UniAD
UNIAD_DIM_SEMANTICS = {
    "image": {0: "batch", 1: "num_cams", 2: "channels", 3: "height", 4: "width"},
    "bev": {0: "batch", 1: "channels", 2: "bev_h", 3: "bev_w"},
    "query": {0: "batch", 1: "num_queries", 2: "embed_dims"},
    "temporal": {0: "batch", 1: "num_frames", 2: "channels", 3: "height", 4: "width"},
    "motion": {0: "batch", 1: "num_agents", 2: "num_modes", 3: "coords", 4: "timesteps"},
    "planning": {0: "batch", 1: "timesteps", 2: "coords"},
    "track": {0: "batch", 1: "num_queries", 2: "features"},
    "occ": {0: "batch", 1: "bev_h", 2: "bev_w", 3: "num_future_frames"},
    "seg": {0: "batch", 1: "num_classes", 2: "bev_h", 3: "bev_w"},
}


class TensorShapeRecorder:
    """Records shapes of tensors at each operation"""
    
    @staticmethod
    def extract_shapes(tensors: Any, semantic_dims: Optional[Dict[str, Dict[int, str]]] = None, 
                      context: Optional[str] = None) -> List[TensorInfo]:
        """Extract shapes from various tensor containers and return TensorInfo objects
        
        Args:
            tensors: Tensor or container of tensors
            semantic_dims: Optional dict mapping tensor keys to dimension semantics
            context: Optional context hint for semantic inference (e.g., 'image', 'bev', 'motion')
        """
        if isinstance(tensors, torch.Tensor):
            shape = tuple(tensors.shape)
            dtype = str(tensors.dtype).replace('torch.', '')
            device = str(tensors.device)
            
            # Convert dtype to more readable format
            dtype = TensorShapeRecorder._format_dtype(dtype)
            
            # Use provided semantic dims or try to infer from context
            sem_dims = None
            if semantic_dims and isinstance(semantic_dims, dict):
                sem_dims = semantic_dims
            elif context and context in UNIAD_DIM_SEMANTICS:
                # Check if shape matches expected dimensions
                expected_dims = UNIAD_DIM_SEMANTICS[context]
                if len(shape) == len(expected_dims):
                    sem_dims = expected_dims
            else:
                # Try to infer semantic dimensions for common patterns
                sem_dims = TensorShapeRecorder._infer_semantic_dims(shape, context)
            
            return [TensorInfo(shape=shape, dtype=dtype, device=device, semantic_dims=sem_dims)]
            
        elif isinstance(tensors, (list, tuple)):
            infos = []
            for t in tensors:
                infos.extend(TensorShapeRecorder.extract_shapes(t, semantic_dims, context))
            return infos
            
        elif isinstance(tensors, dict):
            infos = []
            for k, v in tensors.items():
                # Pass key-specific semantic info if available
                key_semantic = semantic_dims.get(k) if semantic_dims else None
                # Use key as context hint if no context provided
                key_context = context or k
                infos.extend(TensorShapeRecorder.extract_shapes(v, key_semantic, key_context))
            return infos
        else:
            return []
    
    @staticmethod
    def _infer_semantic_dims(shape: Tuple[int, ...], context: Optional[str] = None) -> Optional[Dict[int, str]]:
        """Infer semantic dimensions from shape and context"""
        # BEV grid detection
        if len(shape) == 4 and shape[2] == 200 and shape[3] == 200:
            return UNIAD_DIM_SEMANTICS["bev"]
        
        # Multi-view image detection  
        if len(shape) == 5 and shape[1] == 6:  # 6 cameras
            return UNIAD_DIM_SEMANTICS["image"]
        
        # Track queries
        if len(shape) == 3 and shape[1] == 900:  # 900 queries in UniAD
            return UNIAD_DIM_SEMANTICS["track"]
        
        # Motion prediction
        if len(shape) == 5 and shape[2] == 6:  # 6 modes
            return UNIAD_DIM_SEMANTICS["motion"]
        
        # Occupancy prediction
        if len(shape) == 4 and shape[1] == 200 and shape[2] == 200:
            return UNIAD_DIM_SEMANTICS["occ"]
        
        # Generic patterns
        if len(shape) == 4:  # Common image tensor
            return {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'}
        elif len(shape) == 3:  # Could be sequence or BEV
            if shape[1] > 100 and shape[2] > 100:  # Likely spatial
                return {0: 'batch', 1: 'x', 2: 'y'}
            else:
                return {0: 'batch', 1: 'seq', 2: 'features'}
        elif len(shape) == 2:
            return {0: 'batch', 1: 'features'}
        
        return None
    
    @staticmethod
    def _format_dtype(dtype: str) -> str:
        """Convert PyTorch dtype to readable format"""
        dtype_map = {
            'float32': 'fp32',
            'float16': 'fp16',
            'bfloat16': 'bf16',
            'float64': 'fp64',
            'int64': 'int64',
            'int32': 'int32',
            'int16': 'int16',
            'int8': 'int8',
            'uint8': 'uint8',
            'bool': 'bool'
        }
        return dtype_map.get(dtype, dtype)
    
    @staticmethod
    def estimate_memory(tensor_infos: List[TensorInfo]) -> float:
        """Estimate memory usage in MB from TensorInfo objects"""
        total_bytes = 0
        
        for info in tensor_infos:
            # Get bytes per element based on dtype
            if 'float16' in info.dtype or 'half' in info.dtype or 'fp16' in info.dtype:
                bytes_per_element = 2
            elif 'bfloat16' in info.dtype or 'bf16' in info.dtype:
                bytes_per_element = 2
            elif 'float32' in info.dtype or 'float' in info.dtype or 'fp32' in info.dtype:
                bytes_per_element = 4
            elif 'float64' in info.dtype or 'double' in info.dtype or 'fp64' in info.dtype:
                bytes_per_element = 8
            elif 'int8' in info.dtype or 'byte' in info.dtype:
                bytes_per_element = 1
            elif 'int16' in info.dtype or 'short' in info.dtype:
                bytes_per_element = 2
            elif 'int32' in info.dtype or 'int' in info.dtype:
                bytes_per_element = 4
            elif 'int64' in info.dtype or 'long' in info.dtype:
                bytes_per_element = 8
            else:
                bytes_per_element = 4  # Default to float32
            
            # Calculate total elements
            total_elements = 1
            for dim in info.shape:
                total_elements *= dim
            
            total_bytes += total_elements * bytes_per_element
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    @staticmethod
    def detect_shape_transformation(input_shapes: List[TensorInfo], 
                                   output_shapes: List[TensorInfo]) -> Optional[str]:
        """Detect common shape transformations between input and output"""
        if not input_shapes or not output_shapes:
            return None
            
        in_shape = input_shapes[0].shape
        out_shape = output_shapes[0].shape
        
        # Flatten detection
        if len(in_shape) > len(out_shape) and len(out_shape) == 2:
            in_elements = 1
            for dim in in_shape[1:]:  # Skip batch dim
                in_elements *= dim
            if in_elements == out_shape[1]:
                return "flatten"
        
        # Reshape detection
        if len(in_shape) != len(out_shape):
            in_elements = 1
            out_elements = 1
            for dim in in_shape:
                in_elements *= dim
            for dim in out_shape:
                out_elements *= dim
            if in_elements == out_elements:
                return "reshape"
        
        # Permute detection (simplified)
        if len(in_shape) == len(out_shape) and sorted(in_shape) == sorted(out_shape):
            if in_shape != out_shape:
                return "permute"
        
        # View/squeeze/unsqueeze
        if 1 in in_shape and 1 not in out_shape:
            return "squeeze"
        elif 1 not in in_shape and 1 in out_shape:
            return "unsqueeze"
        
        return None