"""Temporal queue analyzer for multi-frame processing"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

try:
    from ..core.data_structures import TraceNode, TensorInfo
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode, TensorInfo


class TemporalTracer:
    """Handles temporal queue operations for multi-frame processing"""
    
    def __init__(self, queue_length: int = 3, stage: int = 2):
        """Initialize temporal tracer
        
        Args:
            queue_length: Number of frames in temporal queue (3 for stage 2, 5 for stage 1)
            stage: UniAD training stage (1 or 2)
        """
        self.queue_length = queue_length
        self.stage = stage
        self.temporal_operations = []
        self.ego_motion_nodes = []
        self.temporal_fusion_nodes = []
    
    def analyze_temporal_flow(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze temporal aggregation in the model
        
        Args:
            trace_nodes: List of traced nodes from the model
            
        Returns:
            Comprehensive temporal analysis results
        """
        # Separate temporal nodes by type
        temporal_nodes = []
        ego_motion_nodes = []
        fusion_nodes = []
        bev_queue_nodes = []
        
        for node in trace_nodes:
            # Check for temporal operations
            if self._is_temporal_operation(node):
                temporal_nodes.append(node)
                
                # Classify temporal operation type
                if 'ego' in node.operation.lower() or 'motion_comp' in node.module_path.lower():
                    ego_motion_nodes.append(node)
                elif 'temporal_self_attention' in node.module_path.lower():
                    fusion_nodes.append(node)
                elif 'bev_queue' in node.module_path.lower() or self._is_bev_queue_operation(node):
                    bev_queue_nodes.append(node)
        
        # Analyze frame-wise data
        frames_data = self._analyze_frame_data(temporal_nodes)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(temporal_nodes)
        
        # Analyze ego motion compensation
        ego_motion_analysis = self._analyze_ego_motion(ego_motion_nodes)
        
        # Analyze temporal fusion
        fusion_analysis = self._analyze_temporal_fusion(fusion_nodes)
        
        # Analyze BEV queue operations
        bev_queue_analysis = self._analyze_bev_queue(bev_queue_nodes)
        
        # Calculate temporal overhead
        temporal_overhead = self._calculate_temporal_overhead(temporal_nodes, trace_nodes)
        
        return {
            'queue_length': self.queue_length,
            'stage': self.stage,
            'temporal_nodes_count': len(temporal_nodes),
            'frames_analyzed': len(frames_data),
            'temporal_memory_mb': sum(f['memory'] for f in frames_data.values()),
            'temporal_compute_ms': sum(f['compute'] for f in frames_data.values()),
            'temporal_overhead_percent': temporal_overhead,
            'frame_details': dict(frames_data),
            'temporal_patterns': temporal_patterns,
            'ego_motion_analysis': ego_motion_analysis,
            'fusion_analysis': fusion_analysis,
            'bev_queue_analysis': bev_queue_analysis,
            'recommendations': self._generate_recommendations(temporal_patterns, temporal_overhead)
        }
    
    def _is_temporal_operation(self, node: TraceNode) -> bool:
        """Check if a node represents a temporal operation"""
        # Check temporal index
        if node.temporal_index is not None:
            return True
        
        # Check operation names
        temporal_keywords = ['temporal', 'queue', 'history', 'prev', 'ego', 'motion_comp', 'time']
        op_lower = node.operation.lower()
        path_lower = node.module_path.lower()
        
        return any(keyword in op_lower or keyword in path_lower for keyword in temporal_keywords)
    
    def _is_bev_queue_operation(self, node: TraceNode) -> bool:
        """Check if node is related to BEV queue operations"""
        if not node.is_bev_operation:
            return False
        
        # Check for queue-related shapes (batch, queue_length, channels, H, W)
        for shape_info in node.input_shapes + node.output_shapes:
            if len(shape_info.shape) == 5 and shape_info.shape[1] in [3, 5]:  # queue lengths
                return True
        
        return False
    
    def _analyze_frame_data(self, temporal_nodes: List[TraceNode]) -> Dict[int, Dict[str, Any]]:
        """Analyze data for each temporal frame"""
        frames_data = defaultdict(lambda: {
            'nodes': [], 
            'memory': 0, 
            'compute': 0,
            'operations': set(),
            'shapes': []
        })
        
        for node in temporal_nodes:
            frame_idx = node.temporal_index if node.temporal_index is not None else -1
            frames_data[frame_idx]['nodes'].append(node)
            frames_data[frame_idx]['memory'] += node.memory_usage
            frames_data[frame_idx]['compute'] += node.compute_time
            frames_data[frame_idx]['operations'].add(node.operation)
            
            # Track unique shapes
            for shape_info in node.output_shapes:
                shape_str = str(shape_info.shape)
                if shape_str not in frames_data[frame_idx]['shapes']:
                    frames_data[frame_idx]['shapes'].append(shape_str)
        
        # Convert sets to lists for JSON serialization
        for frame_data in frames_data.values():
            frame_data['operations'] = list(frame_data['operations'])
        
        return frames_data
    
    def _analyze_temporal_patterns(self, temporal_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Identify temporal processing patterns"""
        patterns = {
            'sequential_processing': False,
            'parallel_processing': False,
            'sliding_window': False,
            'attention_based': False,
            'feature_propagation': False
        }
        
        # Check for sequential vs parallel processing
        compute_times = defaultdict(list)
        for node in temporal_nodes:
            if node.temporal_index is not None:
                compute_times[node.temporal_index].append(node.compute_time)
        
        if len(compute_times) > 1:
            # Check if frames are processed in sequence or parallel
            avg_times = [np.mean(times) for times in compute_times.values()]
            if len(set(avg_times)) == 1:  # Similar compute times suggest parallel
                patterns['parallel_processing'] = True
            else:
                patterns['sequential_processing'] = True
        
        # Check for sliding window pattern
        if self.queue_length > 1 and len(temporal_nodes) > self.queue_length:
            patterns['sliding_window'] = True
        
        # Check for attention-based fusion
        attention_keywords = ['attention', 'attn', 'transformer']
        if any(any(kw in node.module_path.lower() for kw in attention_keywords) 
               for node in temporal_nodes):
            patterns['attention_based'] = True
        
        # Check for feature propagation
        if any('propagat' in node.operation.lower() or 'flow' in node.operation.lower() 
               for node in temporal_nodes):
            patterns['feature_propagation'] = True
        
        return patterns
    
    def _analyze_ego_motion(self, ego_motion_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze ego motion compensation"""
        if not ego_motion_nodes:
            return {'enabled': False}
        
        analysis = {
            'enabled': True,
            'node_count': len(ego_motion_nodes),
            'total_memory_mb': sum(node.memory_usage for node in ego_motion_nodes),
            'total_compute_ms': sum(node.compute_time for node in ego_motion_nodes),
            'compensation_methods': set(),
            'transform_shapes': []
        }
        
        for node in ego_motion_nodes:
            # Identify compensation method
            if 'warp' in node.operation.lower():
                analysis['compensation_methods'].add('warping')
            elif 'transform' in node.operation.lower():
                analysis['compensation_methods'].add('transformation')
            elif 'align' in node.operation.lower():
                analysis['compensation_methods'].add('alignment')
            
            # Track transformation shapes
            for shape_info in node.output_shapes:
                shape_str = f"{shape_info.shape}"
                if shape_str not in analysis['transform_shapes']:
                    analysis['transform_shapes'].append(shape_str)
        
        analysis['compensation_methods'] = list(analysis['compensation_methods'])
        
        return analysis
    
    def _analyze_temporal_fusion(self, fusion_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze temporal fusion mechanisms"""
        if not fusion_nodes:
            return {'enabled': False}
        
        analysis = {
            'enabled': True,
            'fusion_type': 'unknown',
            'node_count': len(fusion_nodes),
            'total_memory_mb': sum(node.memory_usage for node in fusion_nodes),
            'total_compute_ms': sum(node.compute_time for node in fusion_nodes),
            'attention_heads': None,
            'fusion_stages': []
        }
        
        # Determine fusion type
        fusion_types = set()
        for node in fusion_nodes:
            if 'attention' in node.operation.lower():
                fusion_types.add('attention')
                # Try to extract number of attention heads
                if hasattr(node, 'extra_info') and 'heads' in str(node.extra_info):
                    try:
                        import re
                        heads_match = re.search(r'(\d+)\s*heads', str(node.extra_info))
                        if heads_match:
                            analysis['attention_heads'] = int(heads_match.group(1))
                    except:
                        pass
            elif 'concat' in node.operation.lower():
                fusion_types.add('concatenation')
            elif 'conv' in node.operation.lower():
                fusion_types.add('convolution')
            elif 'gru' in node.operation.lower() or 'lstm' in node.operation.lower():
                fusion_types.add('recurrent')
        
        if fusion_types:
            analysis['fusion_type'] = ', '.join(sorted(fusion_types))
        
        # Track fusion stages
        for node in fusion_nodes:
            stage_info = {
                'operation': node.operation,
                'input_shape': str(node.input_shapes[0].shape) if node.input_shapes else 'N/A',
                'output_shape': str(node.output_shapes[0].shape) if node.output_shapes else 'N/A',
                'memory_mb': node.memory_usage
            }
            analysis['fusion_stages'].append(stage_info)
        
        return analysis
    
    def _analyze_bev_queue(self, bev_queue_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze BEV queue operations"""
        if not bev_queue_nodes:
            return {'enabled': False}
        
        analysis = {
            'enabled': True,
            'queue_operations': len(bev_queue_nodes),
            'queue_memory_mb': sum(node.memory_usage for node in bev_queue_nodes),
            'queue_shapes': [],
            'update_pattern': 'unknown'
        }
        
        # Analyze queue shapes
        for node in bev_queue_nodes:
            for shape_info in node.output_shapes:
                if len(shape_info.shape) == 5:  # (B, T, C, H, W) format
                    queue_info = {
                        'batch': shape_info.shape[0],
                        'queue_length': shape_info.shape[1],
                        'channels': shape_info.shape[2],
                        'height': shape_info.shape[3],
                        'width': shape_info.shape[4],
                        'memory_mb': np.prod(shape_info.shape) * 4 / (1024 * 1024)  # Assume float32
                    }
                    analysis['queue_shapes'].append(queue_info)
        
        # Determine update pattern
        if any('push' in node.operation.lower() or 'pop' in node.operation.lower() 
               for node in bev_queue_nodes):
            analysis['update_pattern'] = 'FIFO'
        elif any('slide' in node.operation.lower() or 'shift' in node.operation.lower() 
                 for node in bev_queue_nodes):
            analysis['update_pattern'] = 'sliding_window'
        
        return analysis
    
    def _calculate_temporal_overhead(self, temporal_nodes: List[TraceNode], 
                                   all_nodes: List[TraceNode]) -> float:
        """Calculate temporal processing overhead as percentage"""
        if not all_nodes:
            return 0.0
        
        temporal_compute = sum(node.compute_time for node in temporal_nodes)
        total_compute = sum(node.compute_time for node in all_nodes)
        
        if total_compute > 0:
            return (temporal_compute / total_compute) * 100
        return 0.0
    
    def _generate_recommendations(self, patterns: Dict[str, bool], 
                                overhead_percent: float) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # High temporal overhead
        if overhead_percent > 30:
            recommendations.append(
                f"High temporal overhead ({overhead_percent:.1f}%). Consider optimizing temporal fusion."
            )
        
        # Sequential processing inefficiency
        if patterns.get('sequential_processing', False):
            recommendations.append(
                "Sequential temporal processing detected. Consider parallelizing frame processing."
            )
        
        # Memory optimization for sliding window
        if patterns.get('sliding_window', False):
            recommendations.append(
                "Sliding window pattern detected. Ensure efficient memory reuse for frame buffers."
            )
        
        # Attention optimization
        if patterns.get('attention_based', False):
            recommendations.append(
                "Attention-based fusion detected. Consider using sparse attention for longer sequences."
            )
        
        # Stage-specific recommendations
        if self.stage == 1 and self.queue_length > 3:
            recommendations.append(
                f"Stage 1 with queue_length={self.queue_length}. Can reduce to 3 for memory savings."
            )
        
        return recommendations
    
    def trace_temporal_flow(self, bev_features: Any) -> Dict[str, Any]:
        """Trace how features aggregate over frames
        
        Args:
            bev_features: BEV feature tensor or queue
            
        Returns:
            Temporal flow analysis
        """
        # This method would be called during actual tracing
        # to track temporal operations in real-time
        flow_info = {
            'timestamp': None,
            'operation': 'temporal_aggregation',
            'queue_state': self.queue_length,
            'features_shape': None
        }
        
        # Extract shape information if available
        if hasattr(bev_features, 'shape'):
            flow_info['features_shape'] = tuple(bev_features.shape)
        
        return flow_info
    
    def visualize_temporal_flow(self, trace_nodes: List[TraceNode]) -> str:
        """Generate Mermaid diagram for temporal flow
        
        Args:
            trace_nodes: List of traced nodes
            
        Returns:
            Mermaid diagram string
        """
        temporal_analysis = self.analyze_temporal_flow(trace_nodes)
        
        mermaid = ["graph LR"]
        mermaid.append("    %% Temporal flow visualization")
        
        # Add frame nodes
        for i in range(self.queue_length):
            frame_label = f"Frame t-{self.queue_length-1-i}" if i < self.queue_length-1 else "Frame t"
            mermaid.append(f'    F{i}["{frame_label}<br/>Input Features"]')
        
        # Add ego motion compensation
        if temporal_analysis['ego_motion_analysis']['enabled']:
            mermaid.append('    EgoComp["Ego Motion<br/>Compensation"]')
            for i in range(self.queue_length):
                mermaid.append(f'    F{i} --> EgoComp')
        
        # Add temporal fusion
        if temporal_analysis['fusion_analysis']['enabled']:
            fusion_type = temporal_analysis['fusion_analysis']['fusion_type']
            mermaid.append(f'    Fusion["Temporal Fusion<br/>{fusion_type}"]')
            
            if temporal_analysis['ego_motion_analysis']['enabled']:
                mermaid.append('    EgoComp --> Fusion')
            else:
                for i in range(self.queue_length):
                    mermaid.append(f'    F{i} --> Fusion')
        
        # Add output
        mermaid.append('    Output["Temporally Enhanced<br/>BEV Features"]')
        if temporal_analysis['fusion_analysis']['enabled']:
            mermaid.append('    Fusion --> Output')
        
        # Add styling
        mermaid.append('    style EgoComp fill:#ffcc99')
        mermaid.append('    style Fusion fill:#99ccff')
        mermaid.append('    style Output fill:#99ff99')
        
        return '\n'.join(mermaid)