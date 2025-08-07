"""Filter engine for large dataflow visualizations with UniAD-specific optimizations"""

import re
from typing import List, Dict, Optional, Any, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

try:
    from ..core.data_structures import TraceNode
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


class FilterType(Enum):
    """Types of filters available"""
    MODULE = "module"
    MEMORY = "memory"
    OPERATION = "operation"
    TASK_HEAD = "task_head"
    BEV_OPERATION = "bev_operation"
    TEMPORAL = "temporal"
    SHAPE = "shape"
    DTYPE = "dtype"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class FilterCriteria:
    """Criteria for filtering nodes"""
    filter_type: FilterType
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, contains, regex, in
    negate: bool = False
    
    def __str__(self):
        op_str = "NOT " if self.negate else ""
        return f"{op_str}{self.filter_type.value} {self.operator} {self.value}"


@dataclass
class SearchResult:
    """Result of a search operation"""
    nodes: List[TraceNode]
    total_matches: int
    filter_applied: str
    execution_time_ms: float
    highlighted_paths: Optional[List[List[str]]] = None  # List of node paths for highlighting


@dataclass
class PathInfo:
    """Information about data paths between nodes"""
    path_nodes: List[str]  # Node IDs in the path
    transformations: List[Dict[str, Any]]  # Shape/dtype transformations along path
    total_memory: float  # Total memory consumption along path
    bottleneck_node: Optional[str] = None  # Node with highest memory/compute


class FilterEngine:
    """Advanced filtering and search engine for large UniAD dataflow visualizations"""
    
    # UniAD-specific patterns
    UNIAD_TASK_HEADS = {
        'track': ['track', 'tracking', 'BEVFormerTrack'],
        'seg': ['seg', 'segmentation', 'panseg', 'Pansegformer'],
        'motion': ['motion', 'prediction', 'MotionHead'],
        'occ': ['occ', 'occupancy', 'OccHead'],
        'planning': ['planning', 'PlanningHead']
    }
    
    BEV_OPERATIONS = [
        'BEVFormer', 'BEVEncoder', 'BEVDecoder', 'BEVGrid',
        'view_transform', 'bev_embed', 'spatial_cross_attention'
    ]
    
    PERFORMANCE_THRESHOLDS = {
        'high_memory': 100.0,    # MB
        'medium_memory': 10.0,   # MB
        'high_compute': 50.0,    # ms
        'medium_compute': 5.0    # ms
    }
    
    def __init__(self, enable_caching: bool = True, max_cache_size: int = 1000):
        """Initialize filter engine
        
        Args:
            enable_caching: Whether to cache filter results
            max_cache_size: Maximum number of cached results
        """
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self._filter_cache: Dict[str, SearchResult] = {}
        self._node_index: Dict[str, TraceNode] = {}
        self._path_cache: Dict[Tuple[str, str], PathInfo] = {}
        
        # Build regex patterns for common filters
        self._compiled_patterns = {}
        self._build_common_patterns()
    
    def _build_common_patterns(self):
        """Pre-compile common regex patterns for performance"""
        patterns = {
            'conv': re.compile(r'conv|Conv', re.IGNORECASE),
            'attention': re.compile(r'attention|Attention', re.IGNORECASE),
            'norm': re.compile(r'norm|Norm|bn|BatchNorm', re.IGNORECASE),
            'linear': re.compile(r'linear|Linear|fc|Dense', re.IGNORECASE),
            'activation': re.compile(r'relu|ReLU|gelu|GELU|silu|SiLU', re.IGNORECASE),
            'bev': re.compile(r'bev|BEV', re.IGNORECASE),
            'transformer': re.compile(r'transformer|Transformer', re.IGNORECASE)
        }
        self._compiled_patterns.update(patterns)
    
    def build_index(self, nodes: List[TraceNode]):
        """Build internal indices for fast filtering
        
        Args:
            nodes: List of trace nodes to index
        """
        self._node_index = {node.node_id: node for node in nodes}
        
        # Clear caches when rebuilding index
        self._filter_cache.clear()
        self._path_cache.clear()
    
    def filter_by_module(self, nodes: List[TraceNode], 
                        module_pattern: str, 
                        use_regex: bool = True) -> SearchResult:
        """Filter nodes by module pattern
        
        Args:
            nodes: List of nodes to filter
            module_pattern: Pattern to match against module_path
            use_regex: Whether to use regex matching
            
        Returns:
            SearchResult with matching nodes
        """
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = f"module_{module_pattern}_{use_regex}"
        if self.enable_caching and cache_key in self._filter_cache:
            cached = self._filter_cache[cache_key]
            # Filter cached results to only include nodes in current set
            current_ids = {node.node_id for node in nodes}
            filtered_nodes = [n for n in cached.nodes if n.node_id in current_ids]
            return SearchResult(
                nodes=filtered_nodes,
                total_matches=len(filtered_nodes),
                filter_applied=cache_key,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        if use_regex:
            try:
                pattern = re.compile(module_pattern, re.IGNORECASE)
                matching_nodes = [
                    node for node in nodes 
                    if pattern.search(node.module_path) or pattern.search(node.operation)
                ]
            except re.error:
                # Fallback to string matching if regex is invalid
                matching_nodes = [
                    node for node in nodes
                    if module_pattern.lower() in node.module_path.lower() 
                    or module_pattern.lower() in node.operation.lower()
                ]
        else:
            matching_nodes = [
                node for node in nodes
                if module_pattern.lower() in node.module_path.lower()
                or module_pattern.lower() in node.operation.lower()
            ]
        
        result = SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=cache_key,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
        # Cache result
        if self.enable_caching and len(self._filter_cache) < self.max_cache_size:
            self._filter_cache[cache_key] = result
        
        return result
    
    def filter_by_memory(self, nodes: List[TraceNode], 
                        threshold: Optional[float] = None,
                        operator: str = "gte",
                        min_memory_mb: Optional[float] = None,
                        max_memory_mb: Optional[float] = None) -> Union[SearchResult, List[TraceNode]]:
        """Filter nodes by memory usage
        
        Args:
            nodes: List of nodes to filter
            threshold: Memory threshold in MB (for old API)
            operator: Comparison operator (gte, lte, gt, lt, eq)
            min_memory_mb: Minimum memory in MB (for new API)
            max_memory_mb: Maximum memory in MB (for new API)
            
        Returns:
            SearchResult with matching nodes or List[TraceNode] for compatibility
        """
        import time
        start_time = time.time()
        
        # Handle new API with min/max parameters
        if min_memory_mb is not None or max_memory_mb is not None:
            matching_nodes = []
            for node in nodes:
                if min_memory_mb is not None and max_memory_mb is not None:
                    if min_memory_mb <= node.memory_usage <= max_memory_mb:
                        matching_nodes.append(node)
                elif min_memory_mb is not None:
                    if node.memory_usage >= min_memory_mb:
                        matching_nodes.append(node)
                elif max_memory_mb is not None:
                    if node.memory_usage <= max_memory_mb:
                        matching_nodes.append(node)
            
            # Return plain list for compatibility with tests
            return matching_nodes
        
        # Handle old API with threshold parameter
        if threshold is None:
            return nodes  # No filtering if no criteria provided
            
        operators = {
            "gte": lambda x, t: x >= t,
            "lte": lambda x, t: x <= t,
            "gt": lambda x, t: x > t,
            "lt": lambda x, t: x < t,
            "eq": lambda x, t: abs(x - t) < 0.01
        }
        
        if operator not in operators:
            raise ValueError(f"Invalid operator: {operator}")
        
        op_func = operators[operator]
        matching_nodes = [
            node for node in nodes
            if op_func(node.memory_usage, threshold)
        ]
        
        # Sort by memory usage (descending)
        matching_nodes.sort(key=lambda x: x.memory_usage, reverse=True)
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=f"memory_{operator}_{threshold}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def filter_by_operation(self, nodes: List[TraceNode], 
                           op_types: Union[str, List[str]]) -> SearchResult:
        """Filter nodes by operation type
        
        Args:
            nodes: List of nodes to filter
            op_types: Operation type(s) to match
            
        Returns:
            SearchResult with matching nodes
        """
        import time
        start_time = time.time()
        
        if isinstance(op_types, str):
            op_types = [op_types]
        
        # Normalize operation types for matching
        op_types_lower = [op.lower() for op in op_types]
        
        matching_nodes = []
        for node in nodes:
            node_op_lower = node.operation.lower()
            if any(op_type in node_op_lower for op_type in op_types_lower):
                matching_nodes.append(node)
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=f"operation_{','.join(op_types)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def filter_by_task_head(self, nodes: List[TraceNode], 
                           task_head: str) -> SearchResult:
        """Filter nodes by UniAD task head
        
        Args:
            nodes: List of nodes to filter
            task_head: Task head name (track, seg, motion, occ, planning)
            
        Returns:
            SearchResult with matching nodes
        """
        import time
        start_time = time.time()
        
        if task_head not in self.UNIAD_TASK_HEADS:
            available = ", ".join(self.UNIAD_TASK_HEADS.keys())
            raise ValueError(f"Invalid task head: {task_head}. Available: {available}")
        
        # Get patterns for this task head
        patterns = self.UNIAD_TASK_HEADS[task_head]
        
        matching_nodes = []
        for node in nodes:
            # Direct task head match
            if node.task_head == task_head:
                matching_nodes.append(node)
                continue
            
            # Pattern matching in module path or operation
            text_to_search = f"{node.module_path} {node.operation}".lower()
            if any(pattern.lower() in text_to_search for pattern in patterns):
                matching_nodes.append(node)
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=f"task_head_{task_head}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def filter_by_bev_operations(self, nodes: List[TraceNode]) -> SearchResult:
        """Filter nodes that are BEV-related operations
        
        Args:
            nodes: List of nodes to filter
            
        Returns:
            SearchResult with BEV nodes
        """
        import time
        start_time = time.time()
        
        matching_nodes = []
        for node in nodes:
            # Direct BEV flag
            if node.is_bev_operation:
                matching_nodes.append(node)
                continue
            
            # Pattern matching
            text_to_search = f"{node.module_path} {node.operation}".lower()
            if any(bev_op.lower() in text_to_search for bev_op in self.BEV_OPERATIONS):
                matching_nodes.append(node)
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied="bev_operations",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def filter_by_shape(self, nodes: List[TraceNode], 
                       shape_pattern: str, 
                       match_input: bool = True,
                       match_output: bool = True) -> SearchResult:
        """Filter nodes by tensor shape pattern
        
        Args:
            nodes: List of nodes to filter
            shape_pattern: Shape pattern (e.g., "200,200" for BEV grid)
            match_input: Whether to match input shapes
            match_output: Whether to match output shapes
            
        Returns:
            SearchResult with matching nodes
        """
        import time
        start_time = time.time()
        
        matching_nodes = []
        for node in nodes:
            shapes_to_check = []
            
            if match_input:
                shapes_to_check.extend(node.input_shapes)
            if match_output:
                shapes_to_check.extend(node.output_shapes)
            
            for tensor_info in shapes_to_check:
                shape_str = ",".join(map(str, tensor_info.shape))
                if shape_pattern in shape_str:
                    matching_nodes.append(node)
                    break
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=f"shape_{shape_pattern}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def filter_by_dtype(self, nodes: List[TraceNode], 
                       dtype: str,
                       match_input: bool = True,
                       match_output: bool = True) -> SearchResult:
        """Filter nodes by tensor data type
        
        Args:
            nodes: List of nodes to filter
            dtype: Data type to match (e.g., "float16", "int8")
            match_input: Whether to match input dtypes
            match_output: Whether to match output dtypes
            
        Returns:
            SearchResult with matching nodes
        """
        import time
        start_time = time.time()
        
        matching_nodes = []
        for node in nodes:
            tensors_to_check = []
            
            if match_input:
                tensors_to_check.extend(node.input_shapes)
            if match_output:
                tensors_to_check.extend(node.output_shapes)
            
            for tensor_info in tensors_to_check:
                if tensor_info.dtype.lower() == dtype.lower():
                    matching_nodes.append(node)
                    break
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=f"dtype_{dtype}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def search_nodes(self, nodes: List[TraceNode], query: str) -> SearchResult:
        """General search across all node attributes
        
        Args:
            nodes: List of nodes to search
            query: Search query string
            
        Returns:
            SearchResult with matching nodes
        """
        import time
        start_time = time.time()
        
        query_lower = query.lower()
        matching_nodes = []
        
        for node in nodes:
            searchable_text = " ".join([
                node.operation,
                node.module_path,
                str(node.task_head) if node.task_head else "",
                str(node.shape_transform) if node.shape_transform else "",
                " ".join([str(s.shape) for s in node.input_shapes]),
                " ".join([str(s.shape) for s in node.output_shapes])
            ]).lower()
            
            if query_lower in searchable_text:
                matching_nodes.append(node)
        
        return SearchResult(
            nodes=matching_nodes,
            total_matches=len(matching_nodes),
            filter_applied=f"search_{query}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def apply_multiple_filters(self, nodes: List[TraceNode], 
                             criteria_list: List[FilterCriteria],
                             combine_with_and: bool = True) -> SearchResult:
        """Apply multiple filter criteria
        
        Args:
            nodes: List of nodes to filter
            criteria_list: List of filter criteria to apply
            combine_with_and: True for AND logic, False for OR logic
            
        Returns:
            SearchResult with nodes matching criteria
        """
        import time
        start_time = time.time()
        
        if not criteria_list:
            return SearchResult(
                nodes=nodes,
                total_matches=len(nodes),
                filter_applied="no_filters",
                execution_time_ms=0.0
            )
        
        matching_sets = []
        filter_descriptions = []
        
        for criteria in criteria_list:
            if criteria.filter_type == FilterType.MODULE:
                result = self.filter_by_module(nodes, criteria.value, use_regex=True)
            elif criteria.filter_type == FilterType.MEMORY:
                result = self.filter_by_memory(nodes, criteria.value, criteria.operator)
            elif criteria.filter_type == FilterType.OPERATION:
                result = self.filter_by_operation(nodes, criteria.value)
            elif criteria.filter_type == FilterType.TASK_HEAD:
                result = self.filter_by_task_head(nodes, criteria.value)
            elif criteria.filter_type == FilterType.BEV_OPERATION:
                result = self.filter_by_bev_operations(nodes)
            elif criteria.filter_type == FilterType.SHAPE:
                result = self.filter_by_shape(nodes, criteria.value)
            elif criteria.filter_type == FilterType.DTYPE:
                result = self.filter_by_dtype(nodes, criteria.value)
            else:
                continue  # Skip unknown filter types
            
            # Handle both SearchResult and List[TraceNode] return types
            if isinstance(result, SearchResult):
                result_nodes = result.nodes
            else:
                result_nodes = result
            matching_nodes = set(node.node_id for node in result_nodes)
            if criteria.negate:
                # Invert the set
                all_node_ids = set(node.node_id for node in nodes)
                matching_nodes = all_node_ids - matching_nodes
            
            matching_sets.append(matching_nodes)
            filter_descriptions.append(str(criteria))
        
        # Combine results
        if combine_with_and:
            # Intersection of all sets
            final_ids = matching_sets[0] if matching_sets else set()
            for node_set in matching_sets[1:]:
                final_ids &= node_set
        else:
            # Union of all sets
            final_ids = set()
            for node_set in matching_sets:
                final_ids |= node_set
        
        # Convert back to node objects
        final_nodes = [node for node in nodes if node.node_id in final_ids]
        
        logic_str = " AND " if combine_with_and else " OR "
        filter_applied = logic_str.join(filter_descriptions)
        
        return SearchResult(
            nodes=final_nodes,
            total_matches=len(final_nodes),
            filter_applied=filter_applied,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def find_data_path(self, nodes: List[TraceNode], 
                      source_node_id: str, 
                      target_node_id: str) -> Optional[PathInfo]:
        """Find data path between two nodes
        
        Args:
            nodes: List of all nodes
            source_node_id: Starting node ID
            target_node_id: Target node ID
            
        Returns:
            PathInfo if path exists, None otherwise
        """
        cache_key = (source_node_id, target_node_id)
        if self.enable_caching and cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        # Build adjacency graph
        graph = defaultdict(list)
        node_lookup = {node.node_id: node for node in nodes}
        
        for node in nodes:
            for dependency in node.depends_on:
                if dependency in node_lookup:
                    graph[dependency].append(node.node_id)
        
        # BFS to find path
        from collections import deque
        queue = deque([(source_node_id, [source_node_id])])
        visited = set()
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_node_id:
                # Found path - analyze transformations
                transformations = []
                total_memory = 0.0
                bottleneck_memory = 0.0
                bottleneck_node = None
                
                for i, node_id in enumerate(path):
                    node = node_lookup[node_id]
                    total_memory += node.memory_usage
                    
                    if node.memory_usage > bottleneck_memory:
                        bottleneck_memory = node.memory_usage
                        bottleneck_node = node_id
                    
                    if i < len(path) - 1:  # Not the last node
                        next_node = node_lookup[path[i + 1]]
                        
                        # Analyze shape transformation
                        transform_info = {
                            'from_node': node_id,
                            'to_node': path[i + 1],
                            'shape_change': None,
                            'dtype_change': None,
                            'memory_change': next_node.memory_usage - node.memory_usage
                        }
                        
                        if node.output_shapes and next_node.input_shapes:
                            out_shape = node.output_shapes[0]
                            in_shape = next_node.input_shapes[0]
                            
                            if out_shape.shape != in_shape.shape:
                                transform_info['shape_change'] = {
                                    'from': out_shape.shape,
                                    'to': in_shape.shape
                                }
                            
                            if out_shape.dtype != in_shape.dtype:
                                transform_info['dtype_change'] = {
                                    'from': out_shape.dtype,
                                    'to': in_shape.dtype
                                }
                        
                        transformations.append(transform_info)
                
                path_info = PathInfo(
                    path_nodes=path,
                    transformations=transformations,
                    total_memory=total_memory,
                    bottleneck_node=bottleneck_node
                )
                
                if self.enable_caching:
                    self._path_cache[cache_key] = path_info
                
                return path_info
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for neighbor in graph[current_id]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        # No path found
        return None
    
    def highlight_data_path(self, nodes: List[TraceNode], 
                           path_info: PathInfo) -> SearchResult:
        """Highlight nodes in a data path
        
        Args:
            nodes: List of all nodes
            path_info: Path information from find_data_path
            
        Returns:
            SearchResult with path nodes highlighted
        """
        import time
        start_time = time.time()
        
        path_node_ids = set(path_info.path_nodes)
        highlighted_nodes = [node for node in nodes if node.node_id in path_node_ids]
        
        return SearchResult(
            nodes=highlighted_nodes,
            total_matches=len(highlighted_nodes),
            filter_applied=f"data_path_{len(path_info.path_nodes)}_nodes",
            execution_time_ms=(time.time() - start_time) * 1000,
            highlighted_paths=[path_info.path_nodes]
        )
    
    def get_performance_summary(self, nodes: List[TraceNode]) -> Dict[str, Any]:
        """Get performance summary for a set of nodes
        
        Args:
            nodes: List of nodes to analyze
            
        Returns:
            Dictionary with performance statistics
        """
        if not nodes:
            return {}
        
        memory_values = [node.memory_usage for node in nodes]
        compute_values = [node.compute_time for node in nodes if node.compute_time > 0]
        
        summary = {
            'total_nodes': len(nodes),
            'total_memory_mb': sum(memory_values),
            'average_memory_mb': sum(memory_values) / len(memory_values),
            'max_memory_mb': max(memory_values),
            'min_memory_mb': min(memory_values),
            'high_memory_nodes': len([m for m in memory_values if m > self.PERFORMANCE_THRESHOLDS['high_memory']]),
            'medium_memory_nodes': len([m for m in memory_values if self.PERFORMANCE_THRESHOLDS['medium_memory'] < m <= self.PERFORMANCE_THRESHOLDS['high_memory']]),
        }
        
        if compute_values:
            summary.update({
                'total_compute_ms': sum(compute_values),
                'average_compute_ms': sum(compute_values) / len(compute_values),
                'max_compute_ms': max(compute_values),
                'high_compute_nodes': len([c for c in compute_values if c > self.PERFORMANCE_THRESHOLDS['high_compute']]),
            })
        
        # Task head distribution
        task_heads = defaultdict(int)
        for node in nodes:
            if node.task_head:
                task_heads[node.task_head] += 1
        summary['task_head_distribution'] = dict(task_heads)
        
        # BEV operations count
        bev_count = sum(1 for node in nodes if node.is_bev_operation)
        summary['bev_operations'] = bev_count
        
        return summary
    
    def clear_cache(self):
        """Clear all caches"""
        self._filter_cache.clear()
        self._path_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'filter_cache_size': len(self._filter_cache),
            'path_cache_size': len(self._path_cache),
            'max_cache_size': self.max_cache_size,
            'caching_enabled': self.enable_caching
        }