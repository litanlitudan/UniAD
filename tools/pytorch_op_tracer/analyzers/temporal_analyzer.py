"""Temporal queue analyzer for multi-frame processing"""

from collections import defaultdict
from typing import Any, Dict, List, Set, TypedDict
import numpy as np

class FrameData(TypedDict):
    """Type definition for frame data structure"""
    nodes: List[Any]  # List[TraceNode]
    memory: float
    compute: float
    operations: Set[str]
    shapes: List[str]

try:
    from ..core.data_structures import TraceNode
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


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
        frames_data: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            'nodes': [], 
            'memory': 0.0, 
            'compute': 0.0,
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
                # Try to extract number of attention heads from module path
                try:
                    import re
                    heads_match = re.search(r'(\d+)\s*heads?', node.module_path.lower())
                    if heads_match:
                        analysis['attention_heads'] = int(heads_match.group(1))
                except Exception:
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
    
    def generate_animation_frames(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Generate frame-by-frame data for temporal flow animation
        
        This method extracts temporal data suitable for JavaScript animation,
        supporting UniAD's multi-frame processing with proper queue handling.
        
        Args:
            trace_nodes: List of traced nodes from the model
            
        Returns:
            Animation data with frame sequences and temporal dependencies
        """
        animation_data = {
            'metadata': {
                'queue_length': self.queue_length,
                'stage': self.stage,
                'total_frames': self.queue_length,
                'animation_type': 'temporal_flow'
            },
            'frames': [],
            'temporal_dependencies': [],
            'aggregation_patterns': [],
            'flow_sequences': []
        }
        
        # Get temporal analysis
        temporal_analysis = self.analyze_temporal_flow(trace_nodes)
        
        # Generate frame data for animation
        frame_data = self._extract_frame_sequences(trace_nodes)
        animation_data['frames'] = frame_data
        
        # Extract temporal dependencies for animation
        dependencies = self._extract_temporal_dependencies(trace_nodes, temporal_analysis)
        animation_data['temporal_dependencies'] = dependencies
        
        # Generate aggregation patterns
        aggregation_patterns = self._extract_aggregation_patterns(trace_nodes, temporal_analysis)
        animation_data['aggregation_patterns'] = aggregation_patterns
        
        # Create flow sequences for step-by-step animation
        flow_sequences = self._create_flow_sequences(trace_nodes, temporal_analysis)
        animation_data['flow_sequences'] = flow_sequences
        
        return animation_data
    
    def _extract_frame_sequences(self, trace_nodes: List[TraceNode]) -> List[Dict[str, Any]]:
        """Extract frame-by-frame processing data for animation"""
        frames = []
        
        # Group nodes by temporal index
        frame_nodes = defaultdict(list)
        for node in trace_nodes:
            if self._is_temporal_operation(node):
                frame_idx = node.temporal_index if node.temporal_index is not None else -1
                frame_nodes[frame_idx].append(node)
        
        # Generate frame data for each temporal frame
        for frame_idx in range(self.queue_length):
            frame_nodes_list = frame_nodes.get(frame_idx, frame_nodes.get(-1, []))
            
            frame_info = {
                'frame_id': frame_idx,
                'timestamp': f"t-{self.queue_length - 1 - frame_idx}" if frame_idx < self.queue_length - 1 else "t",
                'is_current_frame': frame_idx == self.queue_length - 1,
                'operations': [],
                'bev_features': None,
                'memory_usage_mb': 0,
                'compute_time_ms': 0,
                'feature_shapes': [],
                'ego_motion': None
            }
            
            # Extract operations for this frame
            for node in frame_nodes_list:
                operation_info = {
                    'name': node.operation,
                    'module': node.module_path.split('.')[-1] if '.' in node.module_path else node.module_path,
                    'input_shapes': [str(s.shape) for s in node.input_shapes],
                    'output_shapes': [str(s.shape) for s in node.output_shapes],
                    'memory_mb': node.memory_usage,
                    'compute_ms': node.compute_time,
                    'is_bev_operation': getattr(node, 'is_bev_operation', False),
                    'data_flow': self._extract_operation_dataflow(node)
                }
                frame_info['operations'].append(operation_info)
                frame_info['memory_usage_mb'] += node.memory_usage
                frame_info['compute_time_ms'] += node.compute_time
                
                # Extract BEV feature information
                if getattr(node, 'is_bev_operation', False) and node.output_shapes:
                    frame_info['bev_features'] = {
                        'shape': str(node.output_shapes[0].shape),
                        'dtype': node.output_shapes[0].dtype,
                        'memory_mb': node.output_shapes[0].memory_mb
                    }
                
                # Extract ego motion information
                if 'ego' in node.operation.lower() or 'motion_comp' in node.module_path.lower():
                    frame_info['ego_motion'] = {
                        'operation': node.operation,
                        'compensation_type': self._determine_compensation_type(node),
                        'transform_shape': str(node.output_shapes[0].shape) if node.output_shapes else None
                    }
                
                # Collect unique feature shapes
                for shape_info in node.output_shapes:
                    shape_str = str(shape_info.shape)
                    if shape_str not in frame_info['feature_shapes']:
                        frame_info['feature_shapes'].append(shape_str)
            
            frames.append(frame_info)
        
        return frames
    
    def _extract_temporal_dependencies(self, _trace_nodes: List[TraceNode], 
                                     temporal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract temporal dependencies between frames for animation"""
        dependencies = []
        
        # Frame-to-frame dependencies
        for i in range(1, self.queue_length):
            dependency = {
                'from_frame': f"t-{self.queue_length - 1 - (i-1)}",
                'to_frame': f"t-{self.queue_length - 1 - i}",
                'dependency_type': 'temporal_sequence',
                'operations': ['frame_alignment', 'feature_propagation'],
                'strength': 1.0 - (i * 0.2)  # Decrease dependency strength for older frames
            }
            dependencies.append(dependency)
        
        # Ego motion dependencies
        if temporal_analysis['ego_motion_analysis']['enabled']:
            ego_dependency = {
                'from_frame': 'all_frames',
                'to_frame': 'ego_motion_compensation',
                'dependency_type': 'ego_motion',
                'operations': list(temporal_analysis['ego_motion_analysis']['compensation_methods']),
                'strength': 0.8
            }
            dependencies.append(ego_dependency)
        
        # Temporal fusion dependencies
        if temporal_analysis['fusion_analysis']['enabled']:
            fusion_dependency = {
                'from_frame': 'all_frames',
                'to_frame': 'temporal_fusion',
                'dependency_type': 'fusion',
                'operations': [temporal_analysis['fusion_analysis']['fusion_type']],
                'strength': 1.0
            }
            dependencies.append(fusion_dependency)
        
        return dependencies
    
    def _extract_aggregation_patterns(self, _trace_nodes: List[TraceNode], 
                                    temporal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract temporal aggregation patterns for animation visualization"""
        patterns = []
        
        # Sliding window pattern
        if temporal_analysis['temporal_patterns']['sliding_window']:
            pattern = {
                'type': 'sliding_window',
                'description': f'Sliding window over {self.queue_length} frames',
                'frames_involved': list(range(self.queue_length)),
                'aggregation_method': 'temporal_attention' if temporal_analysis['temporal_patterns']['attention_based'] else 'concatenation',
                'animation_steps': [
                    {'step': 1, 'action': 'highlight_frames', 'frames': list(range(self.queue_length))},
                    {'step': 2, 'action': 'ego_motion_compensation', 'frames': list(range(self.queue_length))},
                    {'step': 3, 'action': 'temporal_fusion', 'method': temporal_analysis['fusion_analysis']['fusion_type']},
                    {'step': 4, 'action': 'output_generation', 'result': 'enhanced_bev_features'}
                ]
            }
            patterns.append(pattern)
        
        # Attention-based aggregation
        if temporal_analysis['temporal_patterns']['attention_based']:
            attention_pattern = {
                'type': 'attention_aggregation',
                'description': 'Attention-based temporal feature fusion',
                'attention_heads': temporal_analysis['fusion_analysis'].get('attention_heads', 'unknown'),
                'frames_involved': list(range(self.queue_length)),
                'attention_flow': self._generate_attention_flow_steps()
            }
            patterns.append(attention_pattern)
        
        # Parallel processing pattern
        if temporal_analysis['temporal_patterns']['parallel_processing']:
            parallel_pattern = {
                'type': 'parallel_processing',
                'description': 'Parallel frame processing with synchronization',
                'frames_involved': list(range(self.queue_length)),
                'processing_stages': [
                    {'stage': 'feature_extraction', 'parallel': True},
                    {'stage': 'ego_motion_compensation', 'parallel': True},
                    {'stage': 'temporal_fusion', 'parallel': False}
                ]
            }
            patterns.append(parallel_pattern)
        
        return patterns
    
    def _create_flow_sequences(self, _trace_nodes: List[TraceNode], 
                             temporal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create step-by-step flow sequences for animation"""
        sequences = []
        
        # Main temporal processing sequence
        main_sequence = {
            'sequence_id': 'main_temporal_flow',
            'description': f'UniAD Stage {self.stage} temporal processing with {self.queue_length} frames',
            'total_steps': 0,
            'steps': []
        }
        
        step_count = 0
        
        # Step 1: Input frames
        step_count += 1
        main_sequence['steps'].append({
            'step_id': step_count,
            'title': 'Input Frame Sequence',
            'description': f'Load {self.queue_length} consecutive frames for temporal processing',
            'duration_ms': 100,
            'elements_to_highlight': [f'frame_{i}' for i in range(self.queue_length)],
            'data_flow': {
                'inputs': [f't-{self.queue_length-1-i}' for i in range(self.queue_length)],
                'outputs': ['frame_queue'],
                'operation': 'frame_loading'
            }
        })
        
        # Step 2: Feature extraction
        step_count += 1
        main_sequence['steps'].append({
            'step_id': step_count,
            'title': 'BEV Feature Extraction',
            'description': 'Extract bird\'s-eye-view features from each frame',
            'duration_ms': 200,
            'elements_to_highlight': ['bev_encoder'],
            'data_flow': {
                'inputs': ['frame_queue'],
                'outputs': ['bev_features_queue'],
                'operation': 'feature_extraction'
            }
        })
        
        # Step 3: Ego motion compensation (if enabled)
        if temporal_analysis['ego_motion_analysis']['enabled']:
            step_count += 1
            compensation_methods = ', '.join(temporal_analysis['ego_motion_analysis']['compensation_methods'])
            main_sequence['steps'].append({
                'step_id': step_count,
                'title': 'Ego Motion Compensation',
                'description': f'Align features across frames using {compensation_methods}',
                'duration_ms': 150,
                'elements_to_highlight': ['ego_motion_comp'],
                'data_flow': {
                    'inputs': ['bev_features_queue', 'ego_motion'],
                    'outputs': ['aligned_features'],
                    'operation': 'motion_compensation'
                }
            })
        
        # Step 4: Temporal fusion
        if temporal_analysis['fusion_analysis']['enabled']:
            step_count += 1
            fusion_type = temporal_analysis['fusion_analysis']['fusion_type']
            main_sequence['steps'].append({
                'step_id': step_count,
                'title': 'Temporal Feature Fusion',
                'description': f'Aggregate temporal information using {fusion_type}',
                'duration_ms': 300,
                'elements_to_highlight': ['temporal_fusion'],
                'data_flow': {
                    'inputs': ['aligned_features'] if temporal_analysis['ego_motion_analysis']['enabled'] else ['bev_features_queue'],
                    'outputs': ['fused_temporal_features'],
                    'operation': 'temporal_fusion'
                }
            })
        
        # Step 5: Output generation
        step_count += 1
        main_sequence['steps'].append({
            'step_id': step_count,
            'title': 'Enhanced BEV Output',
            'description': 'Generate temporally enhanced BEV features for downstream tasks',
            'duration_ms': 100,
            'elements_to_highlight': ['output_features'],
            'data_flow': {
                'inputs': ['fused_temporal_features'],
                'outputs': ['enhanced_bev_features'],
                'operation': 'output_generation'
            }
        })
        
        main_sequence['total_steps'] = step_count
        sequences.append(main_sequence)
        
        # Additional sequence for BEV queue operations (if present)
        if temporal_analysis['bev_queue_analysis']['enabled']:
            queue_sequence = {
                'sequence_id': 'bev_queue_management',
                'description': 'BEV queue update and management operations',
                'total_steps': 3,
                'steps': [
                    {
                        'step_id': 1,
                        'title': 'Queue Update',
                        'description': f'Update BEV queue with new frame ({temporal_analysis["bev_queue_analysis"]["update_pattern"]})',
                        'duration_ms': 50,
                        'elements_to_highlight': ['bev_queue'],
                        'data_flow': {'inputs': ['new_bev_features'], 'outputs': ['updated_queue'], 'operation': 'queue_update'}
                    },
                    {
                        'step_id': 2,
                        'title': 'Queue Access',
                        'description': 'Access historical BEV features from queue',
                        'duration_ms': 30,
                        'elements_to_highlight': ['queue_access'],
                        'data_flow': {'inputs': ['updated_queue'], 'outputs': ['historical_features'], 'operation': 'queue_access'}
                    },
                    {
                        'step_id': 3,
                        'title': 'Memory Management',
                        'description': 'Manage queue memory and cleanup old frames',
                        'duration_ms': 20,
                        'elements_to_highlight': ['memory_manager'],
                        'data_flow': {'inputs': ['queue_state'], 'outputs': ['optimized_queue'], 'operation': 'memory_cleanup'}
                    }
                ]
            }
            sequences.append(queue_sequence)
        
        return sequences
    
    def _extract_operation_dataflow(self, node: TraceNode) -> Dict[str, Any]:
        """Extract data flow information for an operation"""
        return {
            'input_count': len(node.input_shapes),
            'output_count': len(node.output_shapes),
            'input_total_elements': sum(np.prod(s.shape) for s in node.input_shapes),
            'output_total_elements': sum(np.prod(s.shape) for s in node.output_shapes),
            'data_reduction_ratio': self._calculate_data_reduction_ratio(node)
        }
    
    def _calculate_data_reduction_ratio(self, node: TraceNode) -> float:
        """Calculate data reduction ratio for an operation"""
        if not node.input_shapes or not node.output_shapes:
            return 1.0
        
        input_elements = sum(np.prod(s.shape) for s in node.input_shapes)
        output_elements = sum(np.prod(s.shape) for s in node.output_shapes)
        
        if input_elements > 0:
            return output_elements / input_elements
        return 1.0
    
    def _determine_compensation_type(self, node: TraceNode) -> str:
        """Determine the type of ego motion compensation"""
        operation_lower = node.operation.lower()
        
        if 'warp' in operation_lower:
            return 'warping'
        elif 'transform' in operation_lower or 'matrix' in operation_lower:
            return 'transformation_matrix'
        elif 'align' in operation_lower:
            return 'feature_alignment'
        elif 'flow' in operation_lower:
            return 'optical_flow'
        else:
            return 'unknown'
    
    def _generate_attention_flow_steps(self) -> List[Dict[str, Any]]:
        """Generate attention flow steps for animation"""
        return [
            {
                'step': 'query_generation',
                'description': 'Generate queries from current frame features',
                'elements': ['current_frame'],
                'operation': 'linear_projection'
            },
            {
                'step': 'key_value_generation', 
                'description': 'Generate keys and values from all temporal frames',
                'elements': [f'frame_{i}' for i in range(self.queue_length)],
                'operation': 'linear_projection'
            },
            {
                'step': 'attention_computation',
                'description': 'Compute attention weights between frames',
                'elements': ['attention_matrix'],
                'operation': 'scaled_dot_product_attention'
            },
            {
                'step': 'weighted_aggregation',
                'description': 'Aggregate temporal features using attention weights',
                'elements': ['weighted_features'],
                'operation': 'attention_weighted_sum'
            }
        ]