"""Stage-specific queue visualization for UniAD's multi-frame processing

This module provides specialized visualization for UniAD's temporal queue system,
supporting both Stage 1 (5 frames, perception) and Stage 2 (3 frames, end-to-end)
with proper temporal dependency edge generation and queue state visualization.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque
import numpy as np

try:
    from ..core.data_structures import TraceNode, TensorInfo
    from ..core.visualization_config import InteractiveConfig
    from ..analyzers.temporal_analyzer import TemporalTracer
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode, TensorInfo
    from core.visualization_config import InteractiveConfig
    from analyzers.temporal_analyzer import TemporalTracer


@dataclass
class QueueFrame:
    """Represents a single frame in the temporal queue"""
    frame_index: int
    timestamp: str  # e.g., "t-2", "t-1", "t"
    is_current: bool
    bev_features: Optional[Dict[str, Any]] = None
    operations: List[str] = field(default_factory=list)
    memory_mb: float = 0.0
    compute_ms: float = 0.0
    ego_motion: Optional[Dict[str, Any]] = None
    active_nodes: List[str] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
            'is_current': self.is_current,
            'bev_features': self.bev_features,
            'operations': self.operations,
            'memory_mb': self.memory_mb,
            'compute_ms': self.compute_ms,
            'ego_motion': self.ego_motion,
            'active_nodes': self.active_nodes,
            'dependencies': self.dependencies
        }


@dataclass
class TemporalDependency:
    """Represents a temporal dependency between frames or operations"""
    source_frame: int
    target_frame: int
    dependency_type: str  # "temporal_sequence", "ego_motion", "fusion", "attention"
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    operation: Optional[str] = None
    strength: float = 1.0  # 0.0 to 1.0, indicates dependency strength
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for visualization"""
        return {
            'source_frame': self.source_frame,
            'target_frame': self.target_frame,
            'dependency_type': self.dependency_type,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'operation': self.operation,
            'strength': self.strength,
            'metadata': self.metadata
        }


class QueueVisualizer:
    """Handles visualization of UniAD's temporal queue system"""
    
    def __init__(self, stage: int = 2, config: Optional[InteractiveConfig] = None):
        """Initialize queue visualizer
        
        Args:
            stage: UniAD training stage (1 or 2)
            config: Optional interactive configuration
        """
        self.stage = stage
        self.queue_length = 5 if stage == 1 else 3
        self.config = config or InteractiveConfig()
        
        # Queue state
        self.queue_frames: List[QueueFrame] = []
        self.temporal_dependencies: List[TemporalDependency] = []
        
        # Visualization settings
        self.frame_colors = {
            0: '#ff6b35',  # Current frame (t)
            1: '#4ecdc4',  # Previous frame (t-1)
            2: '#45b7d1',  # Older frame (t-2)
            3: '#95e77e',  # Stage 1 only (t-3)
            4: '#ffd93d'   # Stage 1 only (t-4)
        }
        
        self.dependency_colors = {
            'temporal_sequence': '#95a5a6',
            'ego_motion': '#e74c3c',
            'fusion': '#3498db',
            'attention': '#9b59b6'
        }
    
    def visualize_queue(self, trace_nodes: List[TraceNode], 
                       temporal_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Visualize multi-frame queue processing with stage-specific handling
        
        Args:
            trace_nodes: List of traced nodes from the model
            temporal_analysis: Optional pre-computed temporal analysis
            
        Returns:
            Complete queue visualization data for rendering
        """
        # Use existing temporal analysis or compute new one
        if temporal_analysis is None:
            temporal_tracer = TemporalTracer(queue_length=self.queue_length, stage=self.stage)
            temporal_analysis = temporal_tracer.analyze_temporal_flow(trace_nodes)
        
        # Build queue frames
        self.queue_frames = self._build_queue_frames(trace_nodes, temporal_analysis)
        
        # Generate temporal dependencies
        self.temporal_dependencies = self._generate_temporal_dependencies(
            self.queue_frames, temporal_analysis
        )
        
        # Create visualization data
        visualization_data = {
            'stage': self.stage,
            'queue_length': self.queue_length,
            'frames': [frame.to_dict() for frame in self.queue_frames],
            'dependencies': [dep.to_dict() for dep in self.temporal_dependencies],
            'queue_visualization': self._create_queue_visualization(),
            'temporal_graph': self._create_temporal_graph(),
            'memory_timeline': self._create_memory_timeline(),
            'processing_flow': self._create_processing_flow(),
            'statistics': self._compute_queue_statistics(),
            'recommendations': self._generate_visualization_recommendations(temporal_analysis)
        }
        
        return visualization_data
    
    def _build_queue_frames(self, trace_nodes: List[TraceNode], 
                           temporal_analysis: Dict[str, Any]) -> List[QueueFrame]:
        """Build queue frame representations from trace nodes"""
        frames = []
        
        # Extract frame data from temporal analysis
        frame_details = temporal_analysis.get('frame_details', {})
        
        for i in range(self.queue_length):
            # Create timestamp label
            if i == self.queue_length - 1:
                timestamp = "t"
                is_current = True
            else:
                timestamp = f"t-{self.queue_length - 1 - i}"
                is_current = False
            
            # Get frame-specific data
            frame_data = frame_details.get(i, frame_details.get(-1, {}))
            
            # Extract BEV features if available
            bev_features = None
            for node in frame_data.get('nodes', []):
                if hasattr(node, 'is_bev_operation') and node.is_bev_operation:
                    if node.output_shapes:
                        bev_features = {
                            'shape': str(node.output_shapes[0].shape),
                            'dtype': node.output_shapes[0].dtype,
                            'memory_mb': (node.output_shapes[0].memory_bytes / (1024*1024)) if node.output_shapes[0].memory_bytes else 0
                        }
                    break
            
            # Extract ego motion if available
            ego_motion = None
            for node in frame_data.get('nodes', []):
                if 'ego' in node.operation.lower() or 'motion_comp' in node.module_path.lower():
                    ego_motion = {
                        'operation': node.operation,
                        'compensation_type': self._determine_compensation_type(node),
                        'transform_shape': str(node.output_shapes[0].shape) if node.output_shapes else None
                    }
                    break
            
            # Create frame
            frame = QueueFrame(
                frame_index=i,
                timestamp=timestamp,
                is_current=is_current,
                bev_features=bev_features,
                operations=list(frame_data.get('operations', [])),
                memory_mb=frame_data.get('memory', 0),
                compute_ms=frame_data.get('compute', 0),
                ego_motion=ego_motion,
                active_nodes=[node.id for node in frame_data.get('nodes', []) if hasattr(node, 'id')]
            )
            
            frames.append(frame)
        
        return frames
    
    def _generate_temporal_dependencies(self, frames: List[QueueFrame], 
                                       temporal_analysis: Dict[str, Any]) -> List[TemporalDependency]:
        """Generate temporal dependency edges between frames and operations"""
        dependencies = []
        
        # Frame-to-frame sequential dependencies
        for i in range(1, len(frames)):
            dep = TemporalDependency(
                source_frame=i-1,
                target_frame=i,
                dependency_type='temporal_sequence',
                operation='frame_propagation',
                strength=1.0 - (i * 0.15),  # Decrease strength for older frames
                metadata={'description': f'Sequential flow from {frames[i-1].timestamp} to {frames[i].timestamp}'}
            )
            dependencies.append(dep)
        
        # Ego motion compensation dependencies
        ego_motion_analysis = temporal_analysis.get('ego_motion_analysis', {})
        if ego_motion_analysis.get('enabled', False):
            for i, frame in enumerate(frames):
                if frame.ego_motion:
                    dep = TemporalDependency(
                        source_frame=i,
                        target_frame=-1,  # Special index for ego motion compensation
                        dependency_type='ego_motion',
                        operation=frame.ego_motion.get('compensation_type', 'unknown'),
                        strength=0.8,
                        metadata=frame.ego_motion
                    )
                    dependencies.append(dep)
        
        # Temporal fusion dependencies
        fusion_analysis = temporal_analysis.get('fusion_analysis', {})
        if fusion_analysis.get('enabled', False):
            # All frames contribute to fusion
            for i in range(len(frames)):
                dep = TemporalDependency(
                    source_frame=i,
                    target_frame=-2,  # Special index for temporal fusion
                    dependency_type='fusion',
                    operation=fusion_analysis.get('fusion_type', 'unknown'),
                    strength=1.0 if frames[i].is_current else 0.7,
                    metadata={'fusion_type': fusion_analysis.get('fusion_type', 'unknown')}
                )
                dependencies.append(dep)
        
        # Attention-based dependencies if present
        if temporal_analysis.get('temporal_patterns', {}).get('attention_based', False):
            # Create attention connections between current frame and history
            current_frame_idx = len(frames) - 1
            for i in range(current_frame_idx):
                dep = TemporalDependency(
                    source_frame=i,
                    target_frame=current_frame_idx,
                    dependency_type='attention',
                    operation='temporal_attention',
                    strength=0.6 + (i * 0.1),  # Increase strength for more recent frames
                    metadata={'attention_heads': fusion_analysis.get('attention_heads', 'unknown')}
                )
                dependencies.append(dep)
        
        # BEV queue dependencies
        bev_queue_analysis = temporal_analysis.get('bev_queue_analysis', {})
        if bev_queue_analysis.get('enabled', False):
            update_pattern = bev_queue_analysis.get('update_pattern', 'unknown')
            
            if update_pattern == 'FIFO':
                # FIFO queue pattern
                for i in range(1, len(frames)):
                    dep = TemporalDependency(
                        source_frame=i,
                        target_frame=i-1,
                        dependency_type='queue_update',
                        operation='fifo_shift',
                        strength=0.9,
                        metadata={'pattern': 'FIFO', 'direction': 'forward'}
                    )
                    dependencies.append(dep)
            elif update_pattern == 'sliding_window':
                # Sliding window pattern
                window_size = min(3, len(frames))
                for i in range(len(frames) - window_size + 1):
                    for j in range(i + 1, i + window_size):
                        dep = TemporalDependency(
                            source_frame=i,
                            target_frame=j,
                            dependency_type='queue_update',
                            operation='sliding_window',
                            strength=0.7,
                            metadata={'pattern': 'sliding_window', 'window_size': window_size}
                        )
                        dependencies.append(dep)
        
        return dependencies
    
    def _create_queue_visualization(self) -> Dict[str, Any]:
        """Create queue state visualization data"""
        return {
            'type': 'queue_state',
            'queue_length': self.queue_length,
            'stage': self.stage,
            'frame_layout': 'horizontal',  # or 'circular' for alternative visualization
            'frames': [
                {
                    'index': frame.frame_index,
                    'timestamp': frame.timestamp,
                    'is_current': frame.is_current,
                    'color': self.frame_colors.get(frame.frame_index, '#cccccc'),
                    'position': {
                        'x': 100 + frame.frame_index * 150,
                        'y': 100
                    },
                    'size': 40 if frame.is_current else 35,
                    'memory_mb': frame.memory_mb,
                    'compute_ms': frame.compute_ms,
                    'has_ego_motion': frame.ego_motion is not None,
                    'operation_count': len(frame.operations)
                }
                for frame in self.queue_frames
            ],
            'connections': [
                {
                    'from': dep.source_frame,
                    'to': dep.target_frame,
                    'type': dep.dependency_type,
                    'color': self.dependency_colors.get(dep.dependency_type, '#cccccc'),
                    'strength': dep.strength,
                    'animated': dep.dependency_type in ['temporal_sequence', 'attention']
                }
                for dep in self.temporal_dependencies
            ]
        }
    
    def _create_temporal_graph(self) -> Dict[str, Any]:
        """Create temporal dependency graph for D3.js visualization"""
        nodes = []
        edges = []
        
        # Create nodes for each frame
        for frame in self.queue_frames:
            node = {
                'id': f'frame_{frame.frame_index}',
                'label': frame.timestamp,
                'type': 'frame',
                'is_current': frame.is_current,
                'color': self.frame_colors.get(frame.frame_index, '#cccccc'),
                'size': 30 if frame.is_current else 25,
                'memory_mb': frame.memory_mb,
                'compute_ms': frame.compute_ms,
                'metadata': {
                    'operations': frame.operations,
                    'has_ego_motion': frame.ego_motion is not None,
                    'bev_features': frame.bev_features
                }
            }
            nodes.append(node)
        
        # Add special nodes for ego motion and fusion if present
        special_node_map = {
            -1: {'id': 'ego_motion_comp', 'label': 'Ego Motion', 'type': 'compensation', 'color': '#e74c3c'},
            -2: {'id': 'temporal_fusion', 'label': 'Temporal Fusion', 'type': 'fusion', 'color': '#3498db'}
        }
        
        used_special_nodes = set()
        for dep in self.temporal_dependencies:
            if dep.target_frame in special_node_map:
                used_special_nodes.add(dep.target_frame)
        
        for special_idx in used_special_nodes:
            nodes.append(special_node_map[special_idx])
        
        # Create edges from dependencies
        for dep in self.temporal_dependencies:
            source_id = f'frame_{dep.source_frame}' if dep.source_frame >= 0 else special_node_map.get(dep.source_frame, {}).get('id')
            target_id = f'frame_{dep.target_frame}' if dep.target_frame >= 0 else special_node_map.get(dep.target_frame, {}).get('id')
            
            if source_id and target_id:
                edge = {
                    'source': source_id,
                    'target': target_id,
                    'type': dep.dependency_type,
                    'weight': dep.strength,
                    'color': self.dependency_colors.get(dep.dependency_type, '#cccccc'),
                    'animated': dep.dependency_type in ['temporal_sequence', 'attention'],
                    'metadata': dep.metadata
                }
                edges.append(edge)
        
        # Add output node
        output_node = {
            'id': 'output',
            'label': 'Enhanced BEV',
            'type': 'output',
            'color': '#27ae60',
            'size': 35
        }
        nodes.append(output_node)
        
        # Connect fusion or last frame to output
        if 'temporal_fusion' in [n['id'] for n in nodes]:
            edges.append({
                'source': 'temporal_fusion',
                'target': 'output',
                'type': 'output_flow',
                'weight': 1.0,
                'color': '#27ae60'
            })
        else:
            edges.append({
                'source': f'frame_{self.queue_length - 1}',
                'target': 'output',
                'type': 'output_flow',
                'weight': 1.0,
                'color': '#27ae60'
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'layout': 'hierarchical',  # or 'force-directed'
            'direction': 'left-to-right'
        }
    
    def _create_memory_timeline(self) -> Dict[str, Any]:
        """Create memory usage timeline for queue frames"""
        timeline_data = {
            'labels': [frame.timestamp for frame in self.queue_frames],
            'datasets': [
                {
                    'label': 'Memory Usage (MB)',
                    'data': [frame.memory_mb for frame in self.queue_frames],
                    'borderColor': '#e74c3c',
                    'backgroundColor': 'rgba(231, 76, 60, 0.1)',
                    'tension': 0.4
                },
                {
                    'label': 'Compute Time (ms)',
                    'data': [frame.compute_ms for frame in self.queue_frames],
                    'borderColor': '#3498db',
                    'backgroundColor': 'rgba(52, 152, 219, 0.1)',
                    'tension': 0.4
                }
            ],
            'total_memory_mb': sum(frame.memory_mb for frame in self.queue_frames),
            'total_compute_ms': sum(frame.compute_ms for frame in self.queue_frames),
            'average_memory_mb': np.mean([frame.memory_mb for frame in self.queue_frames]),
            'average_compute_ms': np.mean([frame.compute_ms for frame in self.queue_frames])
        }
        
        return timeline_data
    
    def _create_processing_flow(self) -> List[Dict[str, Any]]:
        """Create step-by-step processing flow for animation"""
        flow_steps = []
        
        # Step 1: Load frames into queue
        flow_steps.append({
            'step': 1,
            'title': f'Load {self.queue_length} Frames',
            'description': f'Initialize temporal queue with {self.queue_length} consecutive frames',
            'frames_involved': list(range(self.queue_length)),
            'operation': 'queue_initialization',
            'duration_ms': 100
        })
        
        # Step 2: BEV feature extraction
        flow_steps.append({
            'step': 2,
            'title': 'Extract BEV Features',
            'description': 'Convert multi-view images to bird\'s-eye-view features',
            'frames_involved': list(range(self.queue_length)),
            'operation': 'bev_encoding',
            'duration_ms': 200
        })
        
        # Step 3: Ego motion compensation (if present)
        has_ego_motion = any(frame.ego_motion for frame in self.queue_frames)
        if has_ego_motion:
            flow_steps.append({
                'step': 3,
                'title': 'Ego Motion Compensation',
                'description': 'Align features across frames using ego vehicle motion',
                'frames_involved': list(range(self.queue_length)),
                'operation': 'motion_compensation',
                'duration_ms': 150
            })
        
        # Step 4: Temporal fusion
        has_fusion = any(dep.dependency_type == 'fusion' for dep in self.temporal_dependencies)
        if has_fusion:
            flow_steps.append({
                'step': len(flow_steps) + 1,
                'title': 'Temporal Feature Fusion',
                'description': 'Aggregate temporal information across frames',
                'frames_involved': list(range(self.queue_length)),
                'operation': 'temporal_fusion',
                'duration_ms': 300
            })
        
        # Step 5: Queue update (for next iteration)
        flow_steps.append({
            'step': len(flow_steps) + 1,
            'title': 'Update Queue',
            'description': 'Shift queue for next frame processing',
            'frames_involved': list(range(self.queue_length)),
            'operation': 'queue_update',
            'duration_ms': 50
        })
        
        return flow_steps
    
    def _compute_queue_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the queue processing"""
        return {
            'stage': self.stage,
            'queue_length': self.queue_length,
            'total_frames': len(self.queue_frames),
            'total_dependencies': len(self.temporal_dependencies),
            'dependency_types': list(set(dep.dependency_type for dep in self.temporal_dependencies)),
            'total_memory_mb': sum(frame.memory_mb for frame in self.queue_frames),
            'total_compute_ms': sum(frame.compute_ms for frame in self.queue_frames),
            'average_memory_per_frame': np.mean([frame.memory_mb for frame in self.queue_frames]),
            'average_compute_per_frame': np.mean([frame.compute_ms for frame in self.queue_frames]),
            'memory_variance': np.var([frame.memory_mb for frame in self.queue_frames]),
            'compute_variance': np.var([frame.compute_ms for frame in self.queue_frames]),
            'has_ego_motion': any(frame.ego_motion for frame in self.queue_frames),
            'has_temporal_fusion': any(dep.dependency_type == 'fusion' for dep in self.temporal_dependencies),
            'has_attention': any(dep.dependency_type == 'attention' for dep in self.temporal_dependencies)
        }
    
    def _generate_visualization_recommendations(self, temporal_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for queue visualization and optimization"""
        recommendations = []
        
        # Stage-specific recommendations
        if self.stage == 1 and self.queue_length == 5:
            recommendations.append(
                "Stage 1 uses 5-frame queue for stable BEV learning. Consider reducing to 3 frames if memory-constrained."
            )
        elif self.stage == 2 and self.queue_length == 3:
            recommendations.append(
                "Stage 2 uses 3-frame queue with frozen BEV encoder for efficient end-to-end training."
            )
        
        # Memory-based recommendations
        total_memory = sum(frame.memory_mb for frame in self.queue_frames)
        if total_memory > 1000:  # 1GB threshold
            recommendations.append(
                f"High queue memory usage ({total_memory:.1f}MB). Consider temporal checkpointing or gradient accumulation."
            )
        
        # Dependency pattern recommendations
        dependency_types = set(dep.dependency_type for dep in self.temporal_dependencies)
        if 'attention' in dependency_types:
            recommendations.append(
                "Attention-based temporal fusion detected. Monitor attention weights for temporal importance."
            )
        
        if 'ego_motion' in dependency_types:
            recommendations.append(
                "Ego motion compensation active. Ensure accurate ego pose estimation for better alignment."
            )
        
        # Processing pattern recommendations
        patterns = temporal_analysis.get('temporal_patterns', {})
        if patterns.get('sequential_processing', False):
            recommendations.append(
                "Sequential frame processing detected. Consider parallelization for faster training."
            )
        
        if patterns.get('sliding_window', False):
            recommendations.append(
                "Sliding window pattern active. Optimize frame buffer reuse to reduce memory overhead."
            )
        
        return recommendations
    
    def _determine_compensation_type(self, node: TraceNode) -> str:
        """Determine the type of ego motion compensation from node"""
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
    
    def export_queue_visualization(self, visualization_data: Dict[str, Any], 
                                  format: str = 'json') -> str:
        """Export queue visualization in specified format
        
        Args:
            visualization_data: Complete visualization data
            format: Export format ('json', 'html', 'mermaid')
            
        Returns:
            Exported visualization string
        """
        if format == 'json':
            return json.dumps(visualization_data, indent=2)
        
        elif format == 'html':
            return self._generate_html_visualization(visualization_data)
        
        elif format == 'mermaid':
            return self._generate_mermaid_diagram(visualization_data)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_visualization(self, visualization_data: Dict[str, Any]) -> str:
        """Generate HTML visualization of queue"""
        html = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<title>UniAD Queue Visualization</title>',
            '<script src="https://d3js.org/d3.v7.min.js"></script>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; }',
            '.queue-frame { fill: #f0f0f0; stroke: #333; stroke-width: 2; }',
            '.current-frame { fill: #ff6b35; }',
            '.dependency { stroke: #95a5a6; stroke-width: 2; fill: none; }',
            '.frame-label { font-size: 12px; text-anchor: middle; }',
            '</style>',
            '</head>',
            '<body>',
            f'<h1>UniAD Stage {self.stage} Queue Visualization</h1>',
            f'<p>Queue Length: {self.queue_length} frames</p>',
            '<svg id="queue-viz" width="800" height="400"></svg>',
            '<script>',
            f'const data = {json.dumps(visualization_data)};',
            'initializeQueueVisualization(data);',
            '</script>',
            '</body>',
            '</html>'
        ]
        
        return '\n'.join(html)
    
    def _generate_mermaid_diagram(self, visualization_data: Dict[str, Any]) -> str:
        """Generate Mermaid diagram of queue processing"""
        mermaid = ['graph LR']
        mermaid.append(f'    %% UniAD Stage {self.stage} Queue Visualization')
        
        # Add frame nodes
        for frame in self.queue_frames:
            label = f"{frame.timestamp}<br/>{frame.memory_mb:.1f}MB"
            style = 'fill:#ff6b35' if frame.is_current else 'fill:#f0f0f0'
            mermaid.append(f'    F{frame.frame_index}["{label}"]')
            mermaid.append(f'    style F{frame.frame_index} {style}')
        
        # Add special nodes
        has_ego_motion = any(frame.ego_motion for frame in self.queue_frames)
        has_fusion = any(dep.dependency_type == 'fusion' for dep in self.temporal_dependencies)
        
        if has_ego_motion:
            mermaid.append('    EgoMotion["Ego Motion<br/>Compensation"]')
            mermaid.append('    style EgoMotion fill:#e74c3c')
        
        if has_fusion:
            mermaid.append('    Fusion["Temporal<br/>Fusion"]')
            mermaid.append('    style Fusion fill:#3498db')
        
        # Add output
        mermaid.append('    Output["Enhanced<br/>BEV Features"]')
        mermaid.append('    style Output fill:#27ae60')
        
        # Add edges
        for dep in self.temporal_dependencies:
            if dep.source_frame >= 0 and dep.target_frame >= 0:
                mermaid.append(f'    F{dep.source_frame} --> F{dep.target_frame}')
            elif dep.target_frame == -1 and has_ego_motion:  # Ego motion
                mermaid.append(f'    F{dep.source_frame} --> EgoMotion')
            elif dep.target_frame == -2 and has_fusion:  # Fusion
                mermaid.append(f'    F{dep.source_frame} --> Fusion')
        
        # Connect to output
        if has_fusion:
            mermaid.append('    Fusion --> Output')
        elif has_ego_motion:
            mermaid.append('    EgoMotion --> Output')
        else:
            mermaid.append(f'    F{self.queue_length - 1} --> Output')
        
        return '\n'.join(mermaid)


def visualize_uniad_queue(trace_nodes: List[TraceNode], stage: int = 2, 
                         export_format: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to visualize UniAD queue
    
    Args:
        trace_nodes: List of traced nodes from the model
        stage: UniAD training stage (1 or 2)
        export_format: Optional export format ('json', 'html', 'mermaid')
        
    Returns:
        Queue visualization data
    """
    visualizer = QueueVisualizer(stage=stage)
    visualization_data = visualizer.visualize_queue(trace_nodes)
    
    if export_format:
        exported = visualizer.export_queue_visualization(visualization_data, export_format)
        print(f"Queue visualization exported as {export_format}")
        if export_format in ['html', 'mermaid']:
            print(exported)
    
    return visualization_data