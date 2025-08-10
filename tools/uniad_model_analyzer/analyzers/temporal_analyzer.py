"""
Temporal Analyzer for UniAD Model Analyzer.

This module analyzes multi-frame temporal aggregation patterns in UniAD,
which uses 3-5 frames for temporal feature aggregation.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from core.data_structures import TraceNode


@dataclass
class TemporalFrameProfile:
    """Profile for a single temporal frame."""
    
    frame_index: int
    operation_count: int = 0
    memory_bytes: int = 0
    duration_ns: float = 0.0
    
    # Shape information
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    
    # Feature dimensions
    feature_channels: Set[int] = field(default_factory=set)
    spatial_resolutions: Set[Tuple[int, int]] = field(default_factory=set)
    
    # Operations
    attention_ops: int = 0
    aggregation_ops: int = 0
    transformation_ops: int = 0


@dataclass
class TemporalFlowPattern:
    """Represents a temporal flow pattern in the network."""
    
    pattern_type: str  # 'forward', 'backward', 'bidirectional', 'aggregation'
    start_frame: int
    end_frame: int
    operations: List[TraceNode] = field(default_factory=list)
    total_memory: int = 0
    total_duration: float = 0.0
    
    def add_operation(self, node: TraceNode) -> None:
        """Add an operation to this flow pattern."""
        self.operations.append(node)
        self.total_memory += node.get_cuda_memory_delta()
        self.total_duration += node.duration


class TemporalAnalyzer:
    """
    Analyzer for temporal aggregation patterns in UniAD.
    
    UniAD uses temporal self-attention and BEV feature aggregation
    across multiple frames (typically 3 or 5) for better motion understanding.
    """
    
    def __init__(self, num_frames: int = 3):
        """
        Initialize the temporal analyzer.
        
        Args:
            num_frames: Number of temporal frames (3 or 5)
        """
        self.num_frames = num_frames
        self.frame_profiles: Dict[int, TemporalFrameProfile] = {}
        self.temporal_patterns: List[TemporalFlowPattern] = []
        self.queue_memory_usage: Dict[int, int] = {}
        
        # Temporal operation patterns
        self.temporal_keywords = [
            'temporal', 'time', 'sequence', 'recurrent',
            'lstm', 'gru', 'attention', 'self_attention',
            'queue', 'history', 'previous', 'future'
        ]
        
        # Aggregation patterns
        self.aggregation_keywords = [
            'aggregate', 'fusion', 'merge', 'combine',
            'concat', 'stack', 'mean', 'max', 'sum'
        ]
        
        # Frame-to-frame relationships
        self.frame_relationships: Dict[Tuple[int, int], List[TraceNode]] = defaultdict(list)
    
    def analyze_temporal_flow(self, 
                             trace_data: List[TraceNode],
                             num_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze temporal flow patterns across frames.
        
        Args:
            trace_data: List of trace nodes
            num_frames: Override number of frames
            
        Returns:
            Analysis of temporal patterns
        """
        if num_frames:
            self.num_frames = num_frames
        
        # Initialize frame profiles
        for i in range(self.num_frames):
            self.frame_profiles[i] = TemporalFrameProfile(frame_index=i)
        
        # Analyze operations
        for node in trace_data:
            self._analyze_node(node)
        
        # Identify temporal patterns
        self._identify_temporal_patterns(trace_data)
        
        # Analyze frame relationships
        self._analyze_frame_relationships(trace_data)
        
        # Generate analysis
        analysis = {
            'num_frames': self.num_frames,
            'frame_profiles': self._summarize_frame_profiles(),
            'temporal_patterns': self._summarize_patterns(),
            'aggregation_analysis': self._analyze_aggregation(),
            'attention_analysis': self._analyze_attention_patterns(),
            'memory_scaling': self._analyze_memory_scaling(),
            'temporal_efficiency': self._calculate_temporal_efficiency(),
        }
        
        return analysis
    
    def analyze_queue_memory(self, queue_length: int) -> Dict[str, Any]:
        """
        Analyze memory scaling with temporal queue length.
        
        Args:
            queue_length: Number of frames in temporal queue (3 or 5)
            
        Returns:
            Memory analysis for queue configuration
        """
        # Calculate theoretical memory requirements
        base_memory_per_frame = self._estimate_base_memory()
        
        # Different queue configurations
        configs = {
            3: {
                'total_memory': base_memory_per_frame * 3,
                'optimal_for': 'Real-time processing, lower latency',
                'memory_efficiency': 1.0,
            },
            5: {
                'total_memory': base_memory_per_frame * 5,
                'optimal_for': 'Better temporal understanding, higher accuracy',
                'memory_efficiency': 0.6,  # More memory for marginal gains
            }
        }
        
        config = configs.get(queue_length, configs[3])
        
        # Add actual measurements if available
        if self.frame_profiles:
            actual_memory = sum(
                profile.memory_bytes 
                for profile in list(self.frame_profiles.values())[:queue_length]
            )
            config['actual_memory_mb'] = actual_memory / (1024 * 1024)
            config['memory_per_frame_mb'] = actual_memory / queue_length / (1024 * 1024)
        
        # Memory optimization suggestions
        config['optimization_suggestions'] = self._get_memory_optimizations(queue_length)
        
        return config
    
    def visualize_temporal_attention(self) -> Dict[str, Any]:
        """
        Analyze and prepare visualization data for temporal attention patterns.
        
        Returns:
            Visualization data for temporal attention
        """
        attention_data = {
            'attention_matrices': [],
            'frame_connections': [],
            'attention_strengths': {},
        }
        
        # Find attention operations
        for pattern in self.temporal_patterns:
            attention_ops = [
                op for op in pattern.operations
                if 'attention' in op.module_path.lower()
            ]
            
            if attention_ops:
                # Analyze attention patterns
                for op in attention_ops:
                    if op.output_shapes:
                        # Assuming attention outputs have specific shapes
                        shape = op.output_shapes[0].shape
                        if len(shape) >= 3:  # Batch, frames, features
                            attention_data['attention_matrices'].append({
                                'shape': shape,
                                'memory_mb': op.get_cuda_memory_delta() / (1024 * 1024),
                                'duration_ms': op.duration / 1e6,
                            })
        
        # Analyze frame-to-frame connections
        for (frame1, frame2), operations in self.frame_relationships.items():
            if operations:
                attention_data['frame_connections'].append({
                    'from_frame': frame1,
                    'to_frame': frame2,
                    'operation_count': len(operations),
                    'total_memory_mb': sum(op.get_cuda_memory_delta() for op in operations) / (1024 * 1024),
                })
        
        return attention_data
    
    def _analyze_node(self, node: TraceNode) -> None:
        """Analyze a single node for temporal patterns."""
        # Determine frame assignment
        frame_idx = node.temporal_frame if node.temporal_frame is not None else 0
        
        # Update frame profile if within range
        if 0 <= frame_idx < self.num_frames:
            profile = self.frame_profiles[frame_idx]
            profile.operation_count += 1
            profile.memory_bytes += node.get_cuda_memory_delta()
            profile.duration_ns += node.duration
            
            # Track shapes
            for shape in node.input_shapes:
                if shape.shape not in profile.input_shapes:
                    profile.input_shapes.append(shape.shape)
                    
                # Extract feature dimensions
                if len(shape.shape) >= 2:
                    profile.feature_channels.add(shape.shape[1] if len(shape.shape) > 1 else 0)
                if len(shape.shape) >= 4:
                    profile.spatial_resolutions.add((shape.shape[-2], shape.shape[-1]))
            
            # Classify operation type
            module_lower = node.module_path.lower()
            if any(kw in module_lower for kw in ['attention', 'self_attention']):
                profile.attention_ops += 1
            elif any(kw in module_lower for kw in self.aggregation_keywords):
                profile.aggregation_ops += 1
            else:
                profile.transformation_ops += 1
    
    def _identify_temporal_patterns(self, trace_data: List[TraceNode]) -> None:
        """Identify temporal flow patterns in the trace."""
        current_pattern = None
        
        for node in trace_data:
            # Check if this is a temporal operation
            is_temporal = any(
                kw in node.module_path.lower() 
                for kw in self.temporal_keywords
            )
            
            if is_temporal:
                # Start or continue pattern
                if current_pattern is None:
                    pattern_type = self._classify_pattern(node)
                    current_pattern = TemporalFlowPattern(
                        pattern_type=pattern_type,
                        start_frame=node.temporal_frame or 0,
                        end_frame=node.temporal_frame or 0,
                    )
                
                current_pattern.add_operation(node)
                
                # Update frame range
                if node.temporal_frame is not None:
                    current_pattern.end_frame = max(
                        current_pattern.end_frame, 
                        node.temporal_frame
                    )
            else:
                # End current pattern if exists
                if current_pattern and current_pattern.operations:
                    self.temporal_patterns.append(current_pattern)
                    current_pattern = None
        
        # Don't forget the last pattern
        if current_pattern and current_pattern.operations:
            self.temporal_patterns.append(current_pattern)
    
    def _classify_pattern(self, node: TraceNode) -> str:
        """Classify the type of temporal pattern."""
        module_lower = node.module_path.lower()
        
        if 'attention' in module_lower:
            return 'attention'
        elif any(kw in module_lower for kw in self.aggregation_keywords):
            return 'aggregation'
        elif 'recurrent' in module_lower or 'lstm' in module_lower or 'gru' in module_lower:
            return 'recurrent'
        else:
            return 'transformation'
    
    def _analyze_frame_relationships(self, trace_data: List[TraceNode]) -> None:
        """Analyze relationships between temporal frames."""
        for node in trace_data:
            if node.temporal_frame is not None:
                # Check parent relationships
                if node.parent:
                    # Find parent node
                    parent_node = next(
                        (n for n in trace_data if n.id == node.parent), 
                        None
                    )
                    if parent_node and parent_node.temporal_frame is not None:
                        if parent_node.temporal_frame != node.temporal_frame:
                            self.frame_relationships[
                                (parent_node.temporal_frame, node.temporal_frame)
                            ].append(node)
    
    def _summarize_frame_profiles(self) -> List[Dict[str, Any]]:
        """Summarize frame profiles for output."""
        summaries = []
        
        for frame_idx, profile in self.frame_profiles.items():
            summary = {
                'frame_index': frame_idx,
                'operation_count': profile.operation_count,
                'memory_mb': profile.memory_bytes / (1024 * 1024),
                'duration_ms': profile.duration_ns / 1e6,
                'attention_ops': profile.attention_ops,
                'aggregation_ops': profile.aggregation_ops,
                'feature_channels': list(profile.feature_channels),
                'spatial_resolutions': list(profile.spatial_resolutions),
            }
            summaries.append(summary)
        
        return summaries
    
    def _summarize_patterns(self) -> Dict[str, Any]:
        """Summarize temporal patterns."""
        pattern_summary = defaultdict(lambda: {
            'count': 0,
            'total_memory_mb': 0,
            'total_duration_ms': 0,
            'avg_operations': 0,
        })
        
        for pattern in self.temporal_patterns:
            summary = pattern_summary[pattern.pattern_type]
            summary['count'] += 1
            summary['total_memory_mb'] += int(pattern.total_memory / (1024 * 1024))
            summary['total_duration_ms'] += int(pattern.total_duration / 1e6)
            summary['avg_operations'] += len(pattern.operations)
        
        # Calculate averages
        for pattern_type, summary in pattern_summary.items():
            if summary['count'] > 0:
                summary['avg_operations'] = int(summary['avg_operations'] / summary['count'])
        
        return dict(pattern_summary)
    
    def _analyze_aggregation(self) -> Dict[str, Any]:
        """Analyze aggregation operations across frames."""
        aggregation_ops = []
        
        for pattern in self.temporal_patterns:
            if pattern.pattern_type == 'aggregation':
                for op in pattern.operations:
                    aggregation_ops.append(op)
        
        if not aggregation_ops:
            return {'no_aggregation_found': True}
        
        return {
            'total_aggregation_ops': len(aggregation_ops),
            'aggregation_memory_mb': sum(
                op.get_cuda_memory_delta() for op in aggregation_ops
            ) / (1024 * 1024),
            'aggregation_time_ms': sum(
                op.duration for op in aggregation_ops
            ) / 1e6,
            'aggregation_types': list(set(
                op.name for op in aggregation_ops
            )),
        }
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze temporal attention patterns."""
        attention_ops = []
        
        for pattern in self.temporal_patterns:
            if pattern.pattern_type == 'attention':
                attention_ops.extend(pattern.operations)
        
        if not attention_ops:
            return {'no_attention_found': True}
        
        # Analyze attention complexity
        total_attention_memory = sum(op.get_cuda_memory_delta() for op in attention_ops)
        total_attention_time = sum(op.duration for op in attention_ops)
        
        return {
            'total_attention_ops': len(attention_ops),
            'attention_memory_mb': total_attention_memory / (1024 * 1024),
            'attention_time_ms': total_attention_time / 1e6,
            'attention_layers': len(set(op.module_path for op in attention_ops)),
            'avg_attention_memory_per_op_mb': (
                total_attention_memory / len(attention_ops) / (1024 * 1024)
                if attention_ops else 0
            ),
        }
    
    def _analyze_memory_scaling(self) -> Dict[str, Any]:
        """Analyze how memory scales with temporal frames."""
        if not self.frame_profiles:
            return {'no_frame_data': True}
        
        # Calculate memory per frame
        frame_memories = [
            profile.memory_bytes / (1024 * 1024)
            for profile in self.frame_profiles.values()
        ]
        
        if not frame_memories:
            return {'no_memory_data': True}
        
        # Calculate scaling metrics
        total_memory = sum(frame_memories)
        avg_memory = np.mean(frame_memories)
        std_memory = np.std(frame_memories)
        
        # Estimate scaling
        scaling_factor = total_memory / (avg_memory * self.num_frames) if avg_memory > 0 else 1.0
        
        return {
            'total_memory_mb': total_memory,
            'avg_memory_per_frame_mb': avg_memory,
            'memory_std_mb': std_memory,
            'scaling_factor': scaling_factor,
            'scaling_efficiency': 1.0 / scaling_factor if scaling_factor > 0 else 0,
            'memory_distribution': frame_memories,
        }
    
    def _calculate_temporal_efficiency(self) -> float:
        """
        Calculate temporal processing efficiency.
        
        Returns:
            Efficiency score between 0 and 1
        """
        if not self.frame_profiles:
            return 0.0
        
        # Factors for efficiency
        total_ops = sum(p.operation_count for p in self.frame_profiles.values())
        total_memory = sum(p.memory_bytes for p in self.frame_profiles.values())
        total_time = sum(p.duration_ns for p in self.frame_profiles.values())
        
        if total_ops == 0 or self.num_frames == 0:
            return 0.0
        
        # Calculate metrics
        ops_per_frame = total_ops / self.num_frames
        memory_per_op = total_memory / total_ops if total_ops > 0 else float('inf')
        time_per_op = total_time / total_ops if total_ops > 0 else float('inf')
        
        # Efficiency calculation (normalized)
        efficiency = (
            (ops_per_frame / 100) * 0.3 +  # More ops per frame is good
            (1.0 / (memory_per_op / 1e6 + 1)) * 0.4 +  # Less memory per op is good
            (1.0 / (time_per_op / 1e6 + 1)) * 0.3  # Less time per op is good
        )
        
        return min(1.0, max(0.0, efficiency))
    
    def _estimate_base_memory(self) -> int:
        """Estimate base memory per frame."""
        if self.frame_profiles:
            # Use actual data
            memories = [p.memory_bytes for p in self.frame_profiles.values()]
            return int(np.mean(memories)) if memories else 0
        else:
            # Theoretical estimate for UniAD
            # Assuming BEV features of 256 channels at 200x200 resolution
            bev_features = 256 * 200 * 200 * 4  # float32
            # Add overhead for other features
            return bev_features * 2  # Roughly 80MB per frame
    
    def _get_memory_optimizations(self, queue_length: int) -> List[str]:
        """Get memory optimization suggestions based on queue length."""
        suggestions = []
        
        if queue_length >= 5:
            suggestions.append("Consider gradient checkpointing for temporal attention")
            suggestions.append("Use mixed precision (FP16) for temporal features")
            suggestions.append("Implement temporal feature pruning for older frames")
        
        if queue_length >= 3:
            suggestions.append("Share BEV encoder features across frames when possible")
            suggestions.append("Use memory-efficient attention implementations")
        
        suggestions.append(f"Current {queue_length}-frame setup uses ~{queue_length * 80}MB for BEV features")
        
        return suggestions
    
    def export_analysis(self) -> Dict[str, Any]:
        """Export complete temporal analysis."""
        analysis = self.analyze_temporal_flow([], self.num_frames) if not self.frame_profiles else {}
        
        # Add current state
        analysis.update({
            'num_frames': self.num_frames,
            'frame_profiles': self._summarize_frame_profiles() if self.frame_profiles else [],
            'temporal_patterns': self._summarize_patterns() if self.temporal_patterns else {},
            'queue_memory_analysis': self.analyze_queue_memory(self.num_frames),
            'attention_visualization': self.visualize_temporal_attention(),
        })
        
        return analysis