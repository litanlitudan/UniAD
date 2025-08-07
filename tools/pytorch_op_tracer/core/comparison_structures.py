"""Comparison data structures for UniAD task head analysis"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

# UniAD task heads for comparison
UNIAD_TASK_HEADS = ['track', 'seg', 'motion', 'occ', 'planning']

# Task head dependency relationships
TASK_HEAD_DEPENDENCIES = {
    'motion': ['track'],
    'occ': ['track'],
    'planning': ['track', 'motion', 'occ']
}


@dataclass
class ComparisonData:
    """Data structure for comparing two UniAD task heads"""
    
    # Basic comparison metadata
    base_head: str  # Name of base task head
    compare_head: str  # Name of comparison head
    
    # Operation analysis
    common_operations: List[str] = field(default_factory=list)  # Operations in both
    unique_to_base: List[str] = field(default_factory=list)  # Operations only in base
    unique_to_compare: List[str] = field(default_factory=list)  # Operations only in compare
    
    # Performance metrics
    memory_diff: float = 0.0  # Memory difference in MB (positive = compare uses more)
    compute_diff: float = 0.0  # Compute time difference in ms (positive = compare takes longer)
    
    # Detailed metrics per head
    base_metrics: Dict[str, Any] = field(default_factory=dict)
    compare_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Shape analysis
    shape_changes: Dict[str, List[str]] = field(default_factory=dict)  # operation -> shape change descriptions
    
    # Dependency analysis
    dependency_overlap: List[str] = field(default_factory=list)  # Common dependencies
    dependency_differences: Dict[str, List[str]] = field(default_factory=dict)  # Different dependencies
    
    def __post_init__(self):
        """Validate task head names"""
        if self.base_head not in UNIAD_TASK_HEADS:
            raise ValueError(f"Unknown task head: {self.base_head}. Must be one of {UNIAD_TASK_HEADS}")
        if self.compare_head not in UNIAD_TASK_HEADS:
            raise ValueError(f"Unknown task head: {self.compare_head}. Must be one of {UNIAD_TASK_HEADS}")
    
    def get_memory_percentage_diff(self) -> float:
        """Calculate memory difference as percentage of base head usage"""
        base_memory = self.base_metrics.get('total_memory_mb', 0.0)
        if base_memory == 0:
            return 0.0
        return (self.memory_diff / base_memory) * 100
    
    def get_compute_percentage_diff(self) -> float:
        """Calculate compute time difference as percentage of base head usage"""
        base_compute = self.base_metrics.get('total_compute_ms', 0.0)
        if base_compute == 0:
            return 0.0
        return (self.compute_diff / base_compute) * 100
    
    def get_operation_overlap_ratio(self) -> float:
        """Calculate the ratio of common operations to total unique operations"""
        total_operations = len(self.common_operations) + len(self.unique_to_base) + len(self.unique_to_compare)
        if total_operations == 0:
            return 0.0
        return len(self.common_operations) / total_operations
    
    def get_complexity_comparison(self) -> Dict[str, Any]:
        """Compare complexity metrics between task heads"""
        base_ops = len(self.unique_to_base) + len(self.common_operations)
        compare_ops = len(self.unique_to_compare) + len(self.common_operations)
        
        return {
            'base_operation_count': base_ops,
            'compare_operation_count': compare_ops,
            'operation_ratio': compare_ops / base_ops if base_ops > 0 else 0.0,
            'shared_operations': len(self.common_operations),
            'unique_operations_ratio': (len(self.unique_to_base) + len(self.unique_to_compare)) / (base_ops + compare_ops - len(self.common_operations)) if (base_ops + compare_ops - len(self.common_operations)) > 0 else 0.0
        }
    
    def get_dependency_analysis(self) -> Dict[str, Any]:
        """Analyze dependency relationships between compared heads"""
        base_deps = TASK_HEAD_DEPENDENCIES.get(self.base_head, [])
        compare_deps = TASK_HEAD_DEPENDENCIES.get(self.compare_head, [])
        
        return {
            'base_dependencies': base_deps,
            'compare_dependencies': compare_deps,
            'dependency_overlap': list(set(base_deps) & set(compare_deps)),
            'base_unique_deps': list(set(base_deps) - set(compare_deps)),
            'compare_unique_deps': list(set(compare_deps) - set(base_deps)),
            'is_hierarchical': self.base_head in compare_deps or self.compare_head in base_deps
        }
    
    def to_visualization_data(self) -> Dict[str, Any]:
        """Convert to format suitable for side-by-side visualization"""
        return {
            'comparison_id': f"{self.base_head}_vs_{self.compare_head}",
            'base_head': self.base_head,
            'compare_head': self.compare_head,
            
            # Alignment data for side-by-side visualization
            'alignment_scale': {
                'memory_max': max(
                    self.base_metrics.get('total_memory_mb', 0),
                    self.compare_metrics.get('total_memory_mb', 0)
                ),
                'compute_max': max(
                    self.base_metrics.get('total_compute_ms', 0),
                    self.compare_metrics.get('total_compute_ms', 0)
                ),
                'operation_max': max(
                    len(self.unique_to_base) + len(self.common_operations),
                    len(self.unique_to_compare) + len(self.common_operations)
                )
            },
            
            # Visualization categories
            'operation_categories': {
                'common': self.common_operations,
                'base_unique': self.unique_to_base,
                'compare_unique': self.unique_to_compare
            },
            
            # Performance comparison
            'performance_comparison': {
                'memory_diff_mb': self.memory_diff,
                'memory_diff_percent': self.get_memory_percentage_diff(),
                'compute_diff_ms': self.compute_diff,
                'compute_diff_percent': self.get_compute_percentage_diff()
            },
            
            # Complexity analysis
            'complexity_analysis': self.get_complexity_comparison(),
            
            # Dependency analysis
            'dependency_analysis': self.get_dependency_analysis()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'base_head': self.base_head,
            'compare_head': self.compare_head,
            'common_operations': self.common_operations,
            'unique_to_base': self.unique_to_base,
            'unique_to_compare': self.unique_to_compare,
            'memory_diff': self.memory_diff,
            'compute_diff': self.compute_diff,
            'base_metrics': self.base_metrics,
            'compare_metrics': self.compare_metrics,
            'shape_changes': self.shape_changes,
            'dependency_overlap': self.dependency_overlap,
            'dependency_differences': self.dependency_differences,
            'memory_percentage_diff': self.get_memory_percentage_diff(),
            'compute_percentage_diff': self.get_compute_percentage_diff(),
            'operation_overlap_ratio': self.get_operation_overlap_ratio(),
            'complexity_comparison': self.get_complexity_comparison(),
            'dependency_analysis': self.get_dependency_analysis(),
            'visualization_data': self.to_visualization_data()
        }


@dataclass
class MultiComparisonData:
    """Data structure for comparing multiple task heads simultaneously"""
    
    comparisons: List[ComparisonData] = field(default_factory=list)
    reference_head: Optional[str] = None  # Reference head for normalized comparisons
    
    def add_comparison(self, comparison: ComparisonData):
        """Add a pairwise comparison to the collection"""
        self.comparisons.append(comparison)
    
    def get_comparison_matrix(self) -> Dict[Tuple[str, str], ComparisonData]:
        """Get all comparisons as a matrix indexed by (base, compare) pairs"""
        matrix = {}
        for comp in self.comparisons:
            matrix[(comp.base_head, comp.compare_head)] = comp
        return matrix
    
    def get_head_summary(self, head_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific head across all comparisons"""
        head_comparisons = [c for c in self.comparisons if c.base_head == head_name or c.compare_head == head_name]
        
        if not head_comparisons:
            return {}
        
        # Calculate averages when this head is involved
        memory_diffs = []
        compute_diffs = []
        operation_counts = []
        
        for comp in head_comparisons:
            if comp.base_head == head_name:
                memory_diffs.append(-comp.memory_diff)  # Negative because we want from perspective of head_name
                compute_diffs.append(-comp.compute_diff)
                operation_counts.append(len(comp.unique_to_base) + len(comp.common_operations))
            else:
                memory_diffs.append(comp.memory_diff)
                compute_diffs.append(comp.compute_diff)
                operation_counts.append(len(comp.unique_to_compare) + len(comp.common_operations))
        
        return {
            'head_name': head_name,
            'comparison_count': len(head_comparisons),
            'avg_memory_diff_mb': sum(memory_diffs) / len(memory_diffs),
            'avg_compute_diff_ms': sum(compute_diffs) / len(compute_diffs),
            'avg_operation_count': sum(operation_counts) / len(operation_counts),
            'memory_range': (min(memory_diffs), max(memory_diffs)),
            'compute_range': (min(compute_diffs), max(compute_diffs))
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'comparisons': [comp.to_dict() for comp in self.comparisons],
            'reference_head': self.reference_head,
            'comparison_matrix': {
                f"{k[0]}_vs_{k[1]}": v.to_dict() 
                for k, v in self.get_comparison_matrix().items()
            },
            'head_summaries': {
                head: self.get_head_summary(head) 
                for head in UNIAD_TASK_HEADS
                if any(c.base_head == head or c.compare_head == head for c in self.comparisons)
            }
        }