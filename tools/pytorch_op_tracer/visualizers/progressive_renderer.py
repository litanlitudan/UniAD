"""Progressive rendering for large graph visualizations

This module provides progressive rendering capabilities to handle large graphs
with thousands or tens of thousands of nodes efficiently, implementing chunking,
lazy loading, and viewport-based rendering strategies.
"""

from typing import List, Dict, Any, Optional, Set, Generator
from dataclasses import dataclass, field
import math
from collections import defaultdict

try:
    from ..core.data_structures import TraceNode
    from ..core.visualization_config import InteractiveConfig
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode
    from core.visualization_config import InteractiveConfig


@dataclass
class RenderChunk:
    """Represents a chunk of nodes for progressive rendering"""
    chunk_id: int
    nodes: List[TraceNode]
    priority: float  # Higher priority chunks render first
    bounds: Dict[str, float] = field(default_factory=dict)  # Spatial bounds for viewport culling
    is_loaded: bool = False
    is_visible: bool = True
    memory_mb: float = 0.0
    node_count: int = 0
    
    def __post_init__(self):
        """Calculate chunk properties after initialization"""
        self.node_count = len(self.nodes)
        self.memory_mb = sum(node.memory_usage for node in self.nodes)
        self._calculate_bounds()
    
    def _calculate_bounds(self):
        """Calculate spatial bounds for viewport culling"""
        if not self.nodes:
            return
        
        # Extract positions if available (from visualization metadata)
        positions = []
        for node in self.nodes:
            if hasattr(node, 'visualization_metadata') and node.visualization_metadata:
                # Handle both dict and object-like access patterns
                metadata = node.visualization_metadata
                pos = {}
                
                if isinstance(metadata, dict):
                    pos = metadata.get('position', {})
                else:
                    # Try to get position attribute or use empty dict
                    pos = getattr(metadata, 'position', {})
                
                if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    positions.append((pos['x'], pos['y']))
        
        if positions:
            xs, ys = zip(*positions)
            self.bounds = {
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys),
                'center_x': (min(xs) + max(xs)) / 2,
                'center_y': (min(ys) + max(ys)) / 2
            }


@dataclass
class ViewportState:
    """Represents the current viewport state for culling"""
    x: float = 0.0
    y: float = 0.0
    width: float = 1920.0
    height: float = 1080.0
    zoom: float = 1.0
    
    def contains_bounds(self, bounds: Dict[str, float]) -> bool:
        """Check if bounds intersect with viewport"""
        if not bounds:
            return True  # No bounds means always visible
        
        # Apply zoom to viewport
        effective_width = self.width / self.zoom
        effective_height = self.height / self.zoom
        
        # Check intersection
        viewport_min_x = self.x - effective_width / 2
        viewport_max_x = self.x + effective_width / 2
        viewport_min_y = self.y - effective_height / 2
        viewport_max_y = self.y + effective_height / 2
        
        return not (
            bounds.get('max_x', 0) < viewport_min_x or
            bounds.get('min_x', 0) > viewport_max_x or
            bounds.get('max_y', 0) < viewport_min_y or
            bounds.get('min_y', 0) > viewport_max_y
        )


class ProgressiveRenderer:
    """Handles progressive rendering of large graphs"""
    
    # Default configuration
    DEFAULT_CHUNK_SIZE = 100  # Nodes per chunk
    DEFAULT_INITIAL_CHUNKS = 5  # Number of chunks to load initially
    DEFAULT_LAZY_LOAD_THRESHOLD = 500  # Trigger lazy loading above this many nodes
    DEFAULT_MAX_VISIBLE_NODES = 1000  # Maximum nodes to render at once
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        """Initialize progressive renderer
        
        Args:
            config: Optional interactive configuration
        """
        self.config = config or InteractiveConfig()
        self.chunks: List[RenderChunk] = []
        self.viewport = ViewportState()
        self.loaded_chunks: Set[int] = set()
        self.visible_chunks: Set[int] = set()
        
        # Performance settings
        self.chunk_size = self.DEFAULT_CHUNK_SIZE
        self.initial_chunks = self.DEFAULT_INITIAL_CHUNKS
        self.lazy_load_threshold = self.DEFAULT_LAZY_LOAD_THRESHOLD
        self.max_visible_nodes = getattr(config, 'max_nodes_visible', self.DEFAULT_MAX_VISIBLE_NODES)
        
        # Statistics
        self.total_nodes = 0
        self.loaded_nodes = 0
        self.visible_nodes = 0
    
    def chunk_nodes(self, trace_nodes: List[TraceNode], 
                   chunk_size: Optional[int] = None) -> List[RenderChunk]:
        """Divide nodes into chunks for progressive rendering
        
        Args:
            trace_nodes: List of all trace nodes
            chunk_size: Optional custom chunk size
            
        Returns:
            List of RenderChunk objects
        """
        chunk_size = chunk_size or self.chunk_size
        self.total_nodes = len(trace_nodes)
        
        # Sort nodes by priority (memory usage, compute time, centrality)
        prioritized_nodes = self._prioritize_nodes(trace_nodes)
        
        # Create chunks
        chunks = []
        for i in range(0, len(prioritized_nodes), chunk_size):
            chunk_nodes = prioritized_nodes[i:i + chunk_size]
            
            # Calculate chunk priority based on contained nodes
            chunk_priority = self._calculate_chunk_priority(chunk_nodes, i // chunk_size)
            
            chunk = RenderChunk(
                chunk_id=len(chunks),
                nodes=chunk_nodes,
                priority=chunk_priority
            )
            chunks.append(chunk)
        
        # Sort chunks by priority for rendering order
        chunks.sort(key=lambda c: c.priority, reverse=True)
        
        self.chunks = chunks
        return chunks
    
    def _prioritize_nodes(self, trace_nodes: List[TraceNode]) -> List[TraceNode]:
        """Prioritize nodes for rendering order
        
        Priority factors:
        1. Task head nodes (most important)
        2. High memory consumption nodes
        3. High compute time nodes
        4. BEV operations
        5. Nodes with many connections
        """
        def node_priority(node: TraceNode) -> float:
            priority = 0.0
            
            # Task head priority
            if hasattr(node, 'task_head') and node.task_head:
                task_priorities = {
                    'planning': 1.0,  # Highest priority
                    'track': 0.9,
                    'motion': 0.8,
                    'occ': 0.7,
                    'seg': 0.6
                }
                priority += task_priorities.get(node.task_head, 0.5) * 100
            
            # Memory priority (normalized)
            max_memory = max((n.memory_usage for n in trace_nodes), default=1.0)
            if max_memory == 0:
                max_memory = 1.0  # Avoid division by zero
            priority += (node.memory_usage / max_memory) * 50
            
            # Compute priority (normalized)
            max_compute = max((n.compute_time for n in trace_nodes), default=1.0)
            if max_compute == 0:
                max_compute = 1.0  # Avoid division by zero
            priority += (node.compute_time / max_compute) * 30
            
            # BEV operation priority
            if hasattr(node, 'is_bev_operation') and node.is_bev_operation:
                priority += 20
            
            # Connection count priority (if available)
            if hasattr(node, 'connection_count'):
                priority += getattr(node, 'connection_count', 0) * 2
            
            return priority
        
        return sorted(trace_nodes, key=node_priority, reverse=True)
    
    def _calculate_chunk_priority(self, chunk_nodes: List[TraceNode], 
                                 chunk_index: int) -> float:
        """Calculate rendering priority for a chunk
        
        Args:
            chunk_nodes: Nodes in the chunk
            chunk_index: Index of the chunk (earlier = higher priority)
            
        Returns:
            Chunk priority score
        """
        if not chunk_nodes:
            return 0.0
        
        # Average node priority
        avg_priority = sum(self._get_node_priority(node) for node in chunk_nodes) / len(chunk_nodes)
        
        # Bonus for early chunks
        index_bonus = max(0, 100 - chunk_index * 10)
        
        # Bonus for chunks with critical nodes
        has_critical = any(
            hasattr(node, 'task_head') and node.task_head == 'planning'
            for node in chunk_nodes
        )
        critical_bonus = 50 if has_critical else 0
        
        return avg_priority + index_bonus + critical_bonus
    
    def _get_node_priority(self, node: TraceNode) -> float:
        """Get priority score for a single node"""
        priority = 0.0
        
        if hasattr(node, 'task_head') and node.task_head:
            task_priorities = {
                'planning': 100,
                'track': 90,
                'motion': 80,
                'occ': 70,
                'seg': 60
            }
            priority += task_priorities.get(node.task_head, 50)
        
        # Add memory and compute factors
        priority += min(node.memory_usage, 100)  # Cap at 100
        priority += min(node.compute_time / 10, 50)  # Cap at 50
        
        return priority
    
    def load_initial_chunks(self) -> List[TraceNode]:
        """Load initial high-priority chunks
        
        Returns:
            List of nodes from loaded chunks
        """
        loaded_nodes = []
        chunks_to_load = min(self.initial_chunks, len(self.chunks))
        
        for i in range(chunks_to_load):
            chunk = self.chunks[i]
            if not chunk.is_loaded:
                chunk.is_loaded = True
                self.loaded_chunks.add(chunk.chunk_id)
                loaded_nodes.extend(chunk.nodes)
        
        self.loaded_nodes = len(loaded_nodes)
        print(f"Progressive Renderer: Loaded {self.loaded_nodes} nodes in {chunks_to_load} chunks")
        
        return loaded_nodes
    
    def load_chunk(self, chunk_id: int) -> List[TraceNode]:
        """Load a specific chunk
        
        Args:
            chunk_id: ID of chunk to load
            
        Returns:
            List of nodes from the loaded chunk
        """
        if chunk_id >= len(self.chunks):
            return []
        
        chunk = self.chunks[chunk_id]
        if chunk.is_loaded:
            return []  # Already loaded
        
        chunk.is_loaded = True
        self.loaded_chunks.add(chunk_id)
        self.loaded_nodes += chunk.node_count
        
        return chunk.nodes
    
    def unload_chunk(self, chunk_id: int) -> None:
        """Unload a chunk to free memory
        
        Args:
            chunk_id: ID of chunk to unload
        """
        if chunk_id >= len(self.chunks):
            return
        
        chunk = self.chunks[chunk_id]
        if not chunk.is_loaded:
            return
        
        chunk.is_loaded = False
        self.loaded_chunks.discard(chunk_id)
        self.loaded_nodes -= chunk.node_count
    
    def update_viewport(self, viewport: ViewportState) -> List[TraceNode]:
        """Update viewport and determine visible nodes
        
        Args:
            viewport: New viewport state
            
        Returns:
            List of currently visible nodes
        """
        self.viewport = viewport
        visible_nodes = []
        self.visible_chunks.clear()
        
        for chunk in self.chunks:
            if not chunk.is_loaded:
                continue
            
            # Check if chunk is in viewport
            if viewport.contains_bounds(chunk.bounds):
                chunk.is_visible = True
                self.visible_chunks.add(chunk.chunk_id)
                visible_nodes.extend(chunk.nodes)
            else:
                chunk.is_visible = False
        
        # Limit visible nodes if needed
        if len(visible_nodes) > self.max_visible_nodes:
            # Sort by distance from viewport center and take closest
            visible_nodes = self._cull_distant_nodes(visible_nodes, viewport)
        
        self.visible_nodes = len(visible_nodes)
        return visible_nodes
    
    def _cull_distant_nodes(self, nodes: List[TraceNode], 
                           viewport: ViewportState) -> List[TraceNode]:
        """Cull distant nodes to stay within rendering limits
        
        Args:
            nodes: All potentially visible nodes
            viewport: Current viewport state
            
        Returns:
            Culled list of nodes within limits
        """
        # Calculate distance from viewport center for each node
        node_distances = []
        
        for node in nodes:
            distance = float('inf')
            if hasattr(node, 'visualization_metadata') and node.visualization_metadata:
                # Handle both dict and object-like access patterns
                metadata = node.visualization_metadata
                pos = {}
                
                if isinstance(metadata, dict):
                    pos = metadata.get('position', {})
                else:
                    # Try to get position attribute or use empty dict
                    pos = getattr(metadata, 'position', {})
                
                if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    dx = pos['x'] - viewport.x
                    dy = pos['y'] - viewport.y
                    distance = math.sqrt(dx*dx + dy*dy)
            
            node_distances.append((distance, node))
        
        # Sort by distance and take closest
        node_distances.sort(key=lambda x: x[0])
        return [node for _, node in node_distances[:self.max_visible_nodes]]
    
    def get_lazy_load_generator(self, trace_nodes: List[TraceNode], 
                              batch_size: int = 50) -> Generator[List[TraceNode], None, None]:
        """Create a generator for lazy loading nodes in batches
        
        Args:
            trace_nodes: All trace nodes
            batch_size: Number of nodes per batch
            
        Yields:
            Batches of nodes for progressive loading
        """
        # First, chunk the nodes if not already done
        if not self.chunks:
            self.chunk_nodes(trace_nodes)
        
        # Yield initial high-priority chunks
        initial_nodes = self.load_initial_chunks()
        if initial_nodes:
            yield initial_nodes
        
        # Yield remaining chunks on demand
        for chunk in self.chunks[self.initial_chunks:]:
            if not chunk.is_loaded:
                nodes = self.load_chunk(chunk.chunk_id)
                if nodes:
                    # Yield in smaller batches for smoother loading
                    for i in range(0, len(nodes), batch_size):
                        yield nodes[i:i + batch_size]
    
    def optimize_for_memory(self, memory_limit_mb: float = 1000.0) -> None:
        """Optimize chunk loading based on memory constraints
        
        Args:
            memory_limit_mb: Maximum memory to use for loaded chunks
        """
        current_memory = sum(c.memory_mb for c in self.chunks if c.is_loaded)
        
        if current_memory <= memory_limit_mb:
            return  # Within limits
        
        # Sort loaded chunks by priority and visibility
        loaded_chunks = [c for c in self.chunks if c.is_loaded]
        loaded_chunks.sort(key=lambda c: (c.is_visible, c.priority), reverse=True)
        
        # Unload low-priority invisible chunks until within limit
        for chunk in reversed(loaded_chunks):
            if current_memory <= memory_limit_mb:
                break
            
            if not chunk.is_visible:
                self.unload_chunk(chunk.chunk_id)
                current_memory -= chunk.memory_mb
                print(f"Unloaded chunk {chunk.chunk_id} to free {chunk.memory_mb:.1f}MB")
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """Get current rendering statistics
        
        Returns:
            Dictionary of rendering statistics
        """
        return {
            'total_nodes': self.total_nodes,
            'total_chunks': len(self.chunks),
            'loaded_nodes': self.loaded_nodes,
            'loaded_chunks': len(self.loaded_chunks),
            'visible_nodes': self.visible_nodes,
            'visible_chunks': len(self.visible_chunks),
            'memory_usage_mb': sum(c.memory_mb for c in self.chunks if c.is_loaded),
            'chunk_size': self.chunk_size,
            'viewport': {
                'x': self.viewport.x,
                'y': self.viewport.y,
                'width': self.viewport.width,
                'height': self.viewport.height,
                'zoom': self.viewport.zoom
            }
        }
    
    def generate_lod_levels(self, trace_nodes: List[TraceNode]) -> Dict[str, List[TraceNode]]:
        """Generate Level-of-Detail (LOD) representations
        
        Args:
            trace_nodes: All trace nodes
            
        Returns:
            Dictionary of LOD levels with simplified node lists
        """
        lod_levels = {}
        
        # LOD 0: Full detail (all nodes)
        lod_levels['lod0_full'] = trace_nodes
        
        # LOD 1: High importance nodes only (top 50%)
        high_importance = sorted(
            trace_nodes,
            key=lambda n: n.memory_usage + n.compute_time,
            reverse=True
        )[:len(trace_nodes) // 2]
        lod_levels['lod1_high'] = high_importance
        
        # LOD 2: Critical nodes only (top 20%)
        critical = sorted(
            trace_nodes,
            key=lambda n: n.memory_usage + n.compute_time,
            reverse=True
        )[:len(trace_nodes) // 5]
        lod_levels['lod2_critical'] = critical
        
        # LOD 3: Task heads only
        task_heads = [
            n for n in trace_nodes
            if hasattr(n, 'task_head') and n.task_head
        ]
        lod_levels['lod3_tasks'] = task_heads
        
        # LOD 4: Summary (aggregated by module)
        module_summary = self._create_module_summary(trace_nodes)
        lod_levels['lod4_summary'] = module_summary
        
        return lod_levels
    
    def _create_module_summary(self, trace_nodes: List[TraceNode]) -> List[TraceNode]:
        """Create summary nodes aggregated by module
        
        Args:
            trace_nodes: All trace nodes
            
        Returns:
            List of summary nodes
        """
        module_groups = defaultdict(list)
        
        for node in trace_nodes:
            module_key = node.module_path.split('.')[0] if '.' in node.module_path else node.module_path
            module_groups[module_key].append(node)
        
        summary_nodes = []
        for module_name, nodes in module_groups.items():
            # Create a summary node for each module
            summary_node = TraceNode(
                operation=f"{module_name}_summary",
                module_path=module_name,
                input_shapes=[],
                output_shapes=[],
                memory_usage=sum(n.memory_usage for n in nodes),
                compute_time=sum(n.compute_time for n in nodes),
                temporal_index=None
            )
            
            # Add metadata about summarized nodes (if supported)
            # Store metadata separately to avoid attribute errors
            node_metadata = {'node_count': len(nodes), 'is_summary': True}
            
            # Try to set metadata if possible, but don't fail if not
            try:
                if hasattr(summary_node, '__dict__'):
                    setattr(summary_node, 'metadata', node_metadata)
            except (AttributeError, TypeError):
                # If metadata assignment fails, that's okay
                pass
            
            summary_nodes.append(summary_node)
        
        return summary_nodes
    
    def should_use_progressive(self, node_count: int) -> bool:
        """Determine if progressive rendering should be used
        
        Args:
            node_count: Number of nodes in the graph
            
        Returns:
            True if progressive rendering should be used
        """
        return node_count > self.lazy_load_threshold
    
    def export_chunk_map(self) -> Dict[str, Any]:
        """Export chunk mapping for client-side rendering
        
        Returns:
            Dictionary with chunk information for JavaScript
        """
        chunk_map = {
            'total_nodes': self.total_nodes,
            'chunk_size': self.chunk_size,
            'chunks': []
        }
        
        for chunk in self.chunks:
            chunk_info = {
                'id': chunk.chunk_id,
                'priority': chunk.priority,
                'node_count': chunk.node_count,
                'memory_mb': chunk.memory_mb,
                'bounds': chunk.bounds,
                'is_loaded': chunk.is_loaded,
                'is_visible': chunk.is_visible,
                'node_ids': [getattr(node, 'id', f'node_{i}') for i, node in enumerate(chunk.nodes)]
            }
            chunk_map['chunks'].append(chunk_info)
        
        return chunk_map


def create_progressive_renderer(config: Optional[InteractiveConfig] = None) -> ProgressiveRenderer:
    """Factory function to create a progressive renderer
    
    Args:
        config: Optional interactive configuration
        
    Returns:
        ProgressiveRenderer instance
    """
    return ProgressiveRenderer(config)