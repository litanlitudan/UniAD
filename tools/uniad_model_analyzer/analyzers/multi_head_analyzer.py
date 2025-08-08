"""
Multi-Head Analyzer for UniAD Model Analyzer.

This module provides specialized analysis for UniAD's 5 task heads:
- Track head: Object tracking
- Segmentation head: Semantic segmentation
- Motion head: Motion prediction
- Occupancy head: Occupancy prediction
- Planning head: Trajectory planning
"""

from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.data_structures import TraceNode, TaskHead


@dataclass
class TaskHeadProfile:
    """Profile information for a single task head."""
    
    task_head: TaskHead
    operation_count: int = 0
    total_duration_ns: float = 0.0
    total_memory_bytes: int = 0
    total_parameters: int = 0
    
    # Detailed metrics
    layer_count: int = 0
    unique_operations: Set[str] = field(default_factory=set)
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    
    # Memory breakdown
    activation_memory: int = 0
    parameter_memory: int = 0
    gradient_memory: int = 0
    
    # Performance metrics
    avg_operation_time_ns: float = 0.0
    max_operation_time_ns: float = 0.0
    min_operation_time_ns: float = float('inf')
    
    # Dependencies
    depends_on: Set[TaskHead] = field(default_factory=set)
    feeds_into: Set[TaskHead] = field(default_factory=set)
    
    def add_operation(self, node: TraceNode) -> None:
        """Add an operation to this task head's profile."""
        self.operation_count += 1
        self.total_duration_ns += node.duration
        self.total_memory_bytes += node.get_cuda_memory_delta()
        
        # Track unique operations
        self.unique_operations.add(node.name)
        
        # Update timing stats
        self.max_operation_time_ns = max(self.max_operation_time_ns, node.duration)
        self.min_operation_time_ns = min(self.min_operation_time_ns, node.duration)
        
        # Track shapes
        for shape in node.input_shapes:
            if shape.shape not in self.input_shapes:
                self.input_shapes.append(shape.shape)
        for shape in node.output_shapes:
            if shape.shape not in self.output_shapes:
                self.output_shapes.append(shape.shape)
        
        # Update parameters if available
        if node.parameters:
            self.total_parameters += node.parameters
    
    def finalize(self) -> None:
        """Finalize profile calculations."""
        if self.operation_count > 0:
            self.avg_operation_time_ns = self.total_duration_ns / self.operation_count
        else:
            self.avg_operation_time_ns = 0.0
            self.min_operation_time_ns = 0.0


class MultiHeadAnalyzer:
    """
    Analyzer for UniAD's multi-task head architecture.
    
    Provides detailed analysis of each task head's operations, memory usage,
    and inter-head dependencies.
    """
    
    def __init__(self):
        """Initialize the multi-head analyzer."""
        self.task_head_profiles: Dict[TaskHead, TaskHeadProfile] = {}
        self.shared_operations: List[TraceNode] = []
        self.dependency_graph: Dict[TaskHead, Set[TaskHead]] = defaultdict(set)
        
        # UniAD-specific head names
        self.head_module_patterns = {
            TaskHead.TRACK: ['track', 'tracking', 'track_head', 'tracker'],
            TaskHead.SEGMENTATION: ['seg', 'segmentation', 'seg_head', 'panseg'],
            TaskHead.MOTION: ['motion', 'motion_head', 'motion_pred', 'flow'],
            TaskHead.OCCUPANCY: ['occ', 'occupancy', 'occ_head', 'occ_pred'],
            TaskHead.PLANNING: ['plan', 'planning', 'planning_head', 'trajectory'],
        }
        
        # Known shared components
        self.shared_patterns = ['backbone', 'neck', 'bev', 'encoder', 'decoder']
    
    def analyze_task_head(self, 
                         head_name: TaskHead,
                         trace_data: List[TraceNode]) -> TaskHeadProfile:
        """
        Analyze a specific task head.
        
        Args:
            head_name: The task head to analyze
            trace_data: List of trace nodes
            
        Returns:
            TaskHeadProfile with analysis results
        """
        profile = TaskHeadProfile(task_head=head_name)
        
        # Filter nodes for this task head
        head_nodes = self._filter_nodes_for_head(head_name, trace_data)
        
        # Analyze each node
        for node in head_nodes:
            profile.add_operation(node)
        
        # Finalize calculations
        profile.finalize()
        
        # Store profile
        self.task_head_profiles[head_name] = profile
        
        return profile
    
    def analyze_head_interactions(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze interactions and dependencies between task heads.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Dictionary with interaction analysis
        """
        # First, identify all task head operations
        for task_head in TaskHead:
            if task_head != TaskHead.NONE:
                self.analyze_task_head(task_head, trace_data)
        
        # Identify shared operations
        self._identify_shared_operations(trace_data)
        
        # Build dependency graph
        self._build_dependency_graph(trace_data)
        
        # Calculate interaction metrics
        interaction_analysis = {
            'shared_operation_count': len(self.shared_operations),
            'shared_memory_mb': sum(n.get_cuda_memory_delta() for n in self.shared_operations) / (1024 * 1024),
            'dependency_graph': self._format_dependency_graph(),
            'head_coupling_scores': self._calculate_coupling_scores(),
            'resource_distribution': self._calculate_resource_distribution(),
        }
        
        return interaction_analysis
    
    def compute_head_metrics(self, trace_data: List[TraceNode]) -> Dict[TaskHead, Dict[str, Any]]:
        """
        Compute comprehensive metrics for each task head.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Dictionary mapping task heads to their metrics
        """
        metrics = {}
        
        # Ensure all heads are analyzed
        for task_head in TaskHead:
            if task_head != TaskHead.NONE:
                if task_head not in self.task_head_profiles:
                    self.analyze_task_head(task_head, trace_data)
                
                profile = self.task_head_profiles.get(task_head)
                if profile and profile.operation_count > 0:
                    metrics[task_head] = {
                        'operation_count': profile.operation_count,
                        'total_duration_ms': profile.total_duration_ns / 1e6,
                        'avg_duration_ms': profile.avg_operation_time_ns / 1e6,
                        'memory_mb': profile.total_memory_bytes / (1024 * 1024),
                        'parameter_count': profile.total_parameters,
                        'unique_operations': len(profile.unique_operations),
                        'layer_count': profile.layer_count,
                        'efficiency_score': self._calculate_efficiency_score(profile),
                    }
        
        return metrics
    
    def _filter_nodes_for_head(self, 
                              head_name: TaskHead,
                              trace_data: List[TraceNode]) -> List[TraceNode]:
        """Filter trace nodes belonging to a specific task head."""
        head_nodes = []
        
        # First, check explicit task head assignment
        for node in trace_data:
            if node.task_head == head_name:
                head_nodes.append(node)
            # Also check module path patterns
            elif head_name in self.head_module_patterns:
                for pattern in self.head_module_patterns[head_name]:
                    if pattern in node.module_path.lower():
                        head_nodes.append(node)
                        break
        
        return head_nodes
    
    def _identify_shared_operations(self, trace_data: List[TraceNode]) -> None:
        """Identify operations shared across multiple task heads."""
        self.shared_operations = []
        
        for node in trace_data:
            # Check if it's a shared component
            is_shared = False
            for pattern in self.shared_patterns:
                if pattern in node.module_path.lower():
                    is_shared = True
                    break
            
            # Also check if marked as NONE (shared)
            if node.task_head == TaskHead.NONE:
                is_shared = True
            
            if is_shared:
                self.shared_operations.append(node)
    
    def _build_dependency_graph(self, trace_data: List[TraceNode]) -> None:
        """Build dependency graph between task heads."""
        # Track data flow between heads
        node_to_head: Dict[str, TaskHead] = {}
        
        # Map nodes to their task heads
        for node in trace_data:
            if node.task_head != TaskHead.NONE:
                node_to_head[node.id] = node.task_head
        
        # Trace dependencies
        for node in trace_data:
            if node.id in node_to_head:
                current_head = node_to_head[node.id]
                
                # Check parent dependencies
                if node.parent and node.parent in node_to_head:
                    parent_head = node_to_head[node.parent]
                    if parent_head != current_head:
                        self.dependency_graph[current_head].add(parent_head)
                
                # Check cross-branch dependencies
                for dep_id in node.dependencies:
                    if dep_id in node_to_head:
                        dep_head = node_to_head[dep_id]
                        if dep_head != current_head:
                            self.dependency_graph[current_head].add(dep_head)
    
    def _format_dependency_graph(self) -> Dict[str, List[str]]:
        """Format dependency graph for output."""
        formatted = {}
        for head, deps in self.dependency_graph.items():
            formatted[head.value] = [d.value for d in deps]
        return formatted
    
    def _calculate_coupling_scores(self) -> Dict[str, float]:
        """Calculate coupling scores between task heads."""
        scores = {}
        
        # Calculate pairwise coupling
        for head1 in TaskHead:
            if head1 == TaskHead.NONE:
                continue
            for head2 in TaskHead:
                if head2 == TaskHead.NONE or head2 == head1:
                    continue
                
                key = f"{head1.value}-{head2.value}"
                
                # Check if there's a dependency
                if head2 in self.dependency_graph.get(head1, set()):
                    # Higher score for direct dependency
                    scores[key] = 1.0
                elif head1 in self.dependency_graph.get(head2, set()):
                    # Reverse dependency
                    scores[key] = 0.5
                else:
                    # No direct dependency
                    scores[key] = 0.0
        
        return scores
    
    def _calculate_resource_distribution(self) -> Dict[str, float]:
        """Calculate resource distribution across task heads."""
        total_memory = 0
        total_time = 0
        total_params = 0
        
        # Calculate totals
        for profile in self.task_head_profiles.values():
            total_memory += profile.total_memory_bytes
            total_time += profile.total_duration_ns
            total_params += profile.total_parameters
        
        # Calculate percentages
        distribution = {}
        for head, profile in self.task_head_profiles.items():
            if head == TaskHead.NONE:
                continue
            
            distribution[head.value] = {
                'memory_percent': (profile.total_memory_bytes / total_memory * 100) if total_memory > 0 else 0,
                'time_percent': (profile.total_duration_ns / total_time * 100) if total_time > 0 else 0,
                'parameter_percent': (profile.total_parameters / total_params * 100) if total_params > 0 else 0,
            }
        
        return distribution
    
    def _calculate_efficiency_score(self, profile: TaskHeadProfile) -> float:
        """
        Calculate efficiency score for a task head.
        
        Score based on:
        - Operations per parameter (higher is better)
        - Memory per operation (lower is better)
        - Average operation time (lower is better)
        """
        if profile.operation_count == 0 or profile.total_parameters == 0:
            return 0.0
        
        # Operations per million parameters
        ops_per_param = profile.operation_count / (profile.total_parameters / 1e6)
        
        # MB per operation
        mb_per_op = (profile.total_memory_bytes / (1024 * 1024)) / profile.operation_count
        
        # Average time in ms
        avg_time_ms = profile.avg_operation_time_ns / 1e6
        
        # Combine into score (normalize and weight)
        score = (
            (ops_per_param / 100) * 0.3 +  # Normalize and weight
            (1.0 / (mb_per_op + 1)) * 0.4 +  # Inverse for lower is better
            (1.0 / (avg_time_ms + 1)) * 0.3  # Inverse for lower is better
        )
        
        return min(1.0, max(0.0, score))  # Clamp to [0, 1]
    
    def generate_head_comparison(self) -> Dict[str, Any]:
        """Generate a comparison analysis across all task heads."""
        if not self.task_head_profiles:
            return {'error': 'No task heads analyzed yet'}
        
        comparison = {
            'head_count': len(self.task_head_profiles),
            'total_operations': sum(p.operation_count for p in self.task_head_profiles.values()),
            'total_memory_mb': sum(p.total_memory_bytes for p in self.task_head_profiles.values()) / (1024 * 1024),
            'head_rankings': {},
        }
        
        # Rank heads by different metrics
        heads_with_ops = [(h, p) for h, p in self.task_head_profiles.items() if p.operation_count > 0]
        
        if heads_with_ops:
            # By operation count
            by_ops = sorted(heads_with_ops, key=lambda x: x[1].operation_count, reverse=True)
            comparison['head_rankings']['by_operations'] = [h.value for h, _ in by_ops]
            
            # By memory usage
            by_memory = sorted(heads_with_ops, key=lambda x: x[1].total_memory_bytes, reverse=True)
            comparison['head_rankings']['by_memory'] = [h.value for h, _ in by_memory]
            
            # By duration
            by_time = sorted(heads_with_ops, key=lambda x: x[1].total_duration_ns, reverse=True)
            comparison['head_rankings']['by_time'] = [h.value for h, _ in by_time]
            
            # By efficiency
            by_efficiency = sorted(
                heads_with_ops,
                key=lambda x: self._calculate_efficiency_score(x[1]),
                reverse=True
            )
            comparison['head_rankings']['by_efficiency'] = [h.value for h, _ in by_efficiency]
        
        return comparison
    
    def export_analysis(self) -> Dict[str, Any]:
        """Export complete multi-head analysis."""
        return {
            'task_head_profiles': {
                head.value: {
                    'operation_count': profile.operation_count,
                    'total_duration_ms': profile.total_duration_ns / 1e6,
                    'memory_mb': profile.total_memory_bytes / (1024 * 1024),
                    'parameters': profile.total_parameters,
                    'unique_operations': list(profile.unique_operations),
                    'efficiency_score': self._calculate_efficiency_score(profile),
                }
                for head, profile in self.task_head_profiles.items()
                if head != TaskHead.NONE
            },
            'shared_operations': {
                'count': len(self.shared_operations),
                'memory_mb': sum(n.get_cuda_memory_delta() for n in self.shared_operations) / (1024 * 1024),
            },
            'dependencies': self._format_dependency_graph(),
            'resource_distribution': self._calculate_resource_distribution(),
            'comparison': self.generate_head_comparison(),
        }