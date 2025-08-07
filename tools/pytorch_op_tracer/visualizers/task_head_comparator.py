"""Task head comparator for UniAD dataflow visualization"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set
import logging

try:
    from ..core.data_structures import TraceNode, TensorInfo
    from ..core.comparison_structures import ComparisonData, MultiComparisonData, UNIAD_TASK_HEADS, TASK_HEAD_DEPENDENCIES
    from ..analyzers.multi_head_analyzer import MultiHeadTracer
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode, TensorInfo
    from core.comparison_structures import ComparisonData, MultiComparisonData, UNIAD_TASK_HEADS, TASK_HEAD_DEPENDENCIES
    from analyzers.multi_head_analyzer import MultiHeadTracer


class TaskHeadComparator:
    """Compares UniAD task heads with aligned visualization and metrics"""
    
    def __init__(self, task_heads: List[str] = None, alignment_mode: str = 'auto'):
        """
        Initialize task head comparator
        
        Args:
            task_heads: List of task heads to analyze (default: all UniAD heads)
            alignment_mode: How to align scales ('auto', 'global', 'pairwise')
        """
        self.task_heads = task_heads or UNIAD_TASK_HEADS
        self.alignment_mode = alignment_mode
        self.multi_head_tracer = MultiHeadTracer(self.task_heads)
        
        # Validate task heads
        for head in self.task_heads:
            if head not in UNIAD_TASK_HEADS:
                raise ValueError(f"Unknown task head: {head}. Must be one of {UNIAD_TASK_HEADS}")
        
        # Comparison cache
        self._comparison_cache = {}
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
    
    def isolate_task_head_dataflow(self, trace_nodes: List[TraceNode], task_head: str) -> Dict[str, Any]:
        """
        Isolate and analyze dataflow for a specific task head
        
        Args:
            trace_nodes: Complete trace of operations
            task_head: Target task head to isolate
            
        Returns:
            Dictionary containing isolated dataflow analysis
        """
        if task_head not in UNIAD_TASK_HEADS:
            raise ValueError(f"Unknown task head: {task_head}")
        
        # Filter nodes belonging to this task head
        head_nodes = [node for node in trace_nodes if node.task_head == task_head]
        
        # Also include shared BEV operations that feed into this head
        shared_bev_nodes = [
            node for node in trace_nodes
            if node.is_bev_operation and (
                task_head in node.feeds_into or 
                any(head_node.node_id in node.feeds_into for head_node in head_nodes)
            )
        ]
        
        # Combine head-specific and relevant shared nodes
        relevant_nodes = head_nodes + shared_bev_nodes
        
        # Calculate metrics
        total_memory = sum(node.memory_usage for node in head_nodes)
        total_compute = sum(node.compute_time for node in head_nodes)
        operation_count = len(head_nodes)
        
        # Analyze dependencies
        dependencies = TASK_HEAD_DEPENDENCIES.get(task_head, [])
        dependency_nodes = [
            node for node in trace_nodes
            if node.task_head in dependencies
        ]
        
        # Analyze shape transformations
        shape_transforms = []
        for node in head_nodes:
            if node.input_shapes and node.output_shapes:
                transform_desc = node.get_shape_change_summary()
                if transform_desc:
                    shape_transforms.append({
                        'operation': node.operation,
                        'transform': transform_desc,
                        'module': node.module_path
                    })
        
        return {
            'task_head': task_head,
            'head_nodes': head_nodes,
            'shared_nodes': shared_bev_nodes,
            'dependency_nodes': dependency_nodes,
            'metrics': {
                'total_memory_mb': total_memory,
                'total_compute_ms': total_compute,
                'operation_count': operation_count,
                'avg_memory_per_op': total_memory / operation_count if operation_count > 0 else 0.0,
                'avg_compute_per_op': total_compute / operation_count if operation_count > 0 else 0.0
            },
            'dependencies': dependencies,
            'shape_transforms': shape_transforms,
            'operations': [node.operation for node in head_nodes],
            'module_paths': list(set(node.module_path for node in head_nodes))
        }
    
    def compare_heads(self, head_traces: Dict[str, List[TraceNode]]) -> MultiComparisonData:
        """
        Compare multiple task heads and return comprehensive comparison data
        
        Args:
            head_traces: Dictionary mapping task head names to their trace nodes
            
        Returns:
            MultiComparisonData containing all pairwise comparisons
        """
        multi_comparison = MultiComparisonData()
        
        # Validate input
        for head_name in head_traces:
            if head_name not in UNIAD_TASK_HEADS:
                raise ValueError(f"Unknown task head: {head_name}")
        
        # Get all trace nodes for analysis
        all_nodes = []
        for nodes in head_traces.values():
            all_nodes.extend(nodes)
        
        # Isolate each task head's dataflow
        head_analyses = {}
        for head_name in head_traces:
            head_analyses[head_name] = self.isolate_task_head_dataflow(all_nodes, head_name)
        
        # Generate pairwise comparisons
        head_names = list(head_traces.keys())
        for i in range(len(head_names)):
            for j in range(i + 1, len(head_names)):
                base_head = head_names[i]
                compare_head = head_names[j]
                
                comparison = self._generate_pairwise_comparison(
                    head_analyses[base_head], 
                    head_analyses[compare_head]
                )
                multi_comparison.add_comparison(comparison)
        
        return multi_comparison
    
    def _generate_pairwise_comparison(self, base_analysis: Dict[str, Any], 
                                    compare_analysis: Dict[str, Any]) -> ComparisonData:
        """Generate pairwise comparison between two task head analyses"""
        
        base_head = base_analysis['task_head']
        compare_head = compare_analysis['task_head']
        
        # Compare operations
        base_ops = set(base_analysis['operations'])
        compare_ops = set(compare_analysis['operations'])
        
        common_operations = list(base_ops & compare_ops)
        unique_to_base = list(base_ops - compare_ops)
        unique_to_compare = list(compare_ops - base_ops)
        
        # Calculate performance differences
        memory_diff = (compare_analysis['metrics']['total_memory_mb'] - 
                      base_analysis['metrics']['total_memory_mb'])
        compute_diff = (compare_analysis['metrics']['total_compute_ms'] - 
                       base_analysis['metrics']['total_compute_ms'])
        
        # Analyze shape changes
        shape_changes = {}
        for transform in base_analysis['shape_transforms'] + compare_analysis['shape_transforms']:
            op = transform['operation']
            if op not in shape_changes:
                shape_changes[op] = []
            shape_changes[op].append(f"{base_head if transform in base_analysis['shape_transforms'] else compare_head}: {transform['transform']}")
        
        # Analyze dependencies
        base_deps = set(base_analysis['dependencies'])
        compare_deps = set(compare_analysis['dependencies'])
        dependency_overlap = list(base_deps & compare_deps)
        dependency_differences = {
            base_head: list(base_deps - compare_deps),
            compare_head: list(compare_deps - base_deps)
        }
        
        return ComparisonData(
            base_head=base_head,
            compare_head=compare_head,
            common_operations=common_operations,
            unique_to_base=unique_to_base,
            unique_to_compare=unique_to_compare,
            memory_diff=memory_diff,
            compute_diff=compute_diff,
            base_metrics=base_analysis['metrics'],
            compare_metrics=compare_analysis['metrics'],
            shape_changes=shape_changes,
            dependency_overlap=dependency_overlap,
            dependency_differences=dependency_differences
        )
    
    def align_scales(self, comparison_data: MultiComparisonData) -> None:
        """
        Align scales across comparisons for fair visualization
        
        Args:
            comparison_data: Multi-comparison data to align scales for
        """
        if self.alignment_mode == 'auto':
            self._auto_align_scales(comparison_data)
        elif self.alignment_mode == 'global':
            self._global_align_scales(comparison_data)
        elif self.alignment_mode == 'pairwise':
            # Pairwise alignment is already handled in individual comparisons
            pass
        else:
            raise ValueError(f"Unknown alignment mode: {self.alignment_mode}")
    
    def _auto_align_scales(self, comparison_data: MultiComparisonData) -> None:
        """Automatically determine best alignment strategy"""
        
        # Calculate scale variance to determine best alignment
        memory_values = []
        compute_values = []
        
        for comp in comparison_data.comparisons:
            memory_values.extend([
                comp.base_metrics.get('total_memory_mb', 0),
                comp.compare_metrics.get('total_memory_mb', 0)
            ])
            compute_values.extend([
                comp.base_metrics.get('total_compute_ms', 0),
                comp.compare_metrics.get('total_compute_ms', 0)
            ])
        
        # If variance is high, use global alignment; otherwise use pairwise
        import statistics
        if len(memory_values) > 1:
            memory_cv = statistics.stdev(memory_values) / statistics.mean(memory_values) if statistics.mean(memory_values) > 0 else 0
            compute_cv = statistics.stdev(compute_values) / statistics.mean(compute_values) if statistics.mean(compute_values) > 0 else 0
            
            # High coefficient of variation suggests global alignment is needed
            if memory_cv > 0.5 or compute_cv > 0.5:
                self._global_align_scales(comparison_data)
                self.logger.info("Using global scale alignment due to high variance")
            else:
                self.logger.info("Using pairwise scale alignment due to low variance")
    
    def _global_align_scales(self, comparison_data: MultiComparisonData) -> None:
        """Apply global scale alignment across all comparisons"""
        
        # Find global max values
        global_memory_max = 0.0
        global_compute_max = 0.0
        
        for comp in comparison_data.comparisons:
            global_memory_max = max(
                global_memory_max,
                comp.base_metrics.get('total_memory_mb', 0),
                comp.compare_metrics.get('total_memory_mb', 0)
            )
            global_compute_max = max(
                global_compute_max,
                comp.base_metrics.get('total_compute_ms', 0),
                comp.compare_metrics.get('total_compute_ms', 0)
            )
        
        # Apply global scales to visualization data
        for comp in comparison_data.comparisons:
            viz_data = comp.to_visualization_data()
            viz_data['alignment_scale'].update({
                'global_memory_max': global_memory_max,
                'global_compute_max': global_compute_max,
                'alignment_mode': 'global'
            })
    
    def generate_diff_view(self, base_head: str, compare_head: str, 
                          head_traces: Dict[str, List[TraceNode]]) -> Dict[str, Any]:
        """
        Generate detailed diff view between two specific task heads
        
        Args:
            base_head: Base task head for comparison
            compare_head: Task head to compare against base
            head_traces: Dictionary mapping task head names to their trace nodes
            
        Returns:
            Dictionary containing detailed diff analysis
        """
        if base_head not in UNIAD_TASK_HEADS or compare_head not in UNIAD_TASK_HEADS:
            raise ValueError("Invalid task head names")
        
        # Get all nodes for analysis
        all_nodes = []
        for nodes in head_traces.values():
            all_nodes.extend(nodes)
        
        # Isolate dataflows
        base_analysis = self.isolate_task_head_dataflow(all_nodes, base_head)
        compare_analysis = self.isolate_task_head_dataflow(all_nodes, compare_head)
        
        # Generate comparison
        comparison = self._generate_pairwise_comparison(base_analysis, compare_analysis)
        
        # Generate detailed diff view
        diff_view = {
            'comparison': comparison.to_dict(),
            'detailed_analysis': {
                'operation_diff': {
                    'added_operations': comparison.unique_to_compare,
                    'removed_operations': comparison.unique_to_base,
                    'common_operations': comparison.common_operations,
                    'operation_changes': self._analyze_operation_changes(base_analysis, compare_analysis)
                },
                'memory_breakdown': {
                    'base_head_memory': base_analysis['metrics']['total_memory_mb'],
                    'compare_head_memory': compare_analysis['metrics']['total_memory_mb'],
                    'memory_diff_mb': comparison.memory_diff,
                    'memory_diff_percent': comparison.get_memory_percentage_diff(),
                    'memory_per_operation': {
                        base_head: base_analysis['metrics']['avg_memory_per_op'],
                        compare_head: compare_analysis['metrics']['avg_memory_per_op']
                    }
                },
                'compute_breakdown': {
                    'base_head_compute': base_analysis['metrics']['total_compute_ms'],
                    'compare_head_compute': compare_analysis['metrics']['total_compute_ms'],
                    'compute_diff_ms': comparison.compute_diff,
                    'compute_diff_percent': comparison.get_compute_percentage_diff(),
                    'compute_per_operation': {
                        base_head: base_analysis['metrics']['avg_compute_per_op'],
                        compare_head: compare_analysis['metrics']['avg_compute_per_op']
                    }
                },
                'shape_analysis': {
                    'shape_changes': comparison.shape_changes,
                    'transform_patterns': self._analyze_transform_patterns(base_analysis, compare_analysis)
                },
                'dependency_analysis': comparison.get_dependency_analysis()
            },
            'visualization_metadata': {
                'diff_type': 'task_head_comparison',
                'base_head': base_head,
                'compare_head': compare_head,
                'alignment_scale': comparison.to_visualization_data()['alignment_scale'],
                'highlight_differences': True,
                'side_by_side': True
            }
        }
        
        return diff_view
    
    def _analyze_operation_changes(self, base_analysis: Dict[str, Any], 
                                 compare_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how operations change between task heads"""
        
        base_modules = set(base_analysis['module_paths'])
        compare_modules = set(compare_analysis['module_paths'])
        
        return {
            'module_changes': {
                'added_modules': list(compare_modules - base_modules),
                'removed_modules': list(base_modules - compare_modules),
                'common_modules': list(base_modules & compare_modules)
            },
            'complexity_change': {
                'operation_count_change': (
                    compare_analysis['metrics']['operation_count'] - 
                    base_analysis['metrics']['operation_count']
                ),
                'avg_memory_per_op_change': (
                    compare_analysis['metrics']['avg_memory_per_op'] - 
                    base_analysis['metrics']['avg_memory_per_op']
                ),
                'avg_compute_per_op_change': (
                    compare_analysis['metrics']['avg_compute_per_op'] - 
                    base_analysis['metrics']['avg_compute_per_op']
                )
            }
        }
    
    def _analyze_transform_patterns(self, base_analysis: Dict[str, Any], 
                                  compare_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze shape transformation patterns between task heads"""
        
        base_transforms = {t['operation']: t['transform'] for t in base_analysis['shape_transforms']}
        compare_transforms = {t['operation']: t['transform'] for t in compare_analysis['shape_transforms']}
        
        common_ops = set(base_transforms.keys()) & set(compare_transforms.keys())
        transform_diffs = {}
        
        for op in common_ops:
            if base_transforms[op] != compare_transforms[op]:
                transform_diffs[op] = {
                    'base_transform': base_transforms[op],
                    'compare_transform': compare_transforms[op]
                }
        
        return {
            'different_transforms': transform_diffs,
            'base_unique_transforms': {
                op: transform for op, transform in base_transforms.items()
                if op not in compare_transforms
            },
            'compare_unique_transforms': {
                op: transform for op, transform in compare_transforms.items()
                if op not in base_transforms
            }
        }
    
    def generate_side_by_side_visualization(self, comparison_data: MultiComparisonData) -> Dict[str, Any]:
        """
        Generate side-by-side visualization data for multiple task head comparisons
        
        Args:
            comparison_data: Multi-comparison data to visualize
            
        Returns:
            Dictionary containing side-by-side visualization specifications
        """
        # Ensure scales are aligned
        self.align_scales(comparison_data)
        
        visualization_data = {
            'visualization_type': 'side_by_side_comparison',
            'alignment_mode': self.alignment_mode,
            'comparisons': [],
            'global_scales': self._calculate_global_scales(comparison_data),
            'head_summaries': {}
        }
        
        # Process each comparison
        for comp in comparison_data.comparisons:
            viz_comp = comp.to_visualization_data()
            viz_comp['visualization_config'] = {
                'show_memory_bars': True,
                'show_compute_bars': True,
                'show_operation_diff': True,
                'show_dependency_arrows': True,
                'highlight_differences': True,
                'color_scheme': 'task_head_specific'
            }
            visualization_data['comparisons'].append(viz_comp)
        
        # Generate head summaries
        for head in self.task_heads:
            summary = comparison_data.get_head_summary(head)
            if summary:
                visualization_data['head_summaries'][head] = summary
        
        return visualization_data
    
    def _calculate_global_scales(self, comparison_data: MultiComparisonData) -> Dict[str, float]:
        """Calculate global scales for consistent visualization"""
        
        memory_values = []
        compute_values = []
        operation_counts = []
        
        for comp in comparison_data.comparisons:
            memory_values.extend([
                comp.base_metrics.get('total_memory_mb', 0),
                comp.compare_metrics.get('total_memory_mb', 0)
            ])
            compute_values.extend([
                comp.base_metrics.get('total_compute_ms', 0),
                comp.compare_metrics.get('total_compute_ms', 0)
            ])
            operation_counts.extend([
                comp.base_metrics.get('operation_count', 0),
                comp.compare_metrics.get('operation_count', 0)
            ])
        
        return {
            'max_memory_mb': max(memory_values) if memory_values else 0.0,
            'max_compute_ms': max(compute_values) if compute_values else 0.0,
            'max_operation_count': max(operation_counts) if operation_counts else 0
        }
    
    def generate_comparison_report(self, comparison_data: MultiComparisonData) -> str:
        """
        Generate a comprehensive text report of task head comparisons
        
        Args:
            comparison_data: Multi-comparison data to report on
            
        Returns:
            Formatted text report
        """
        report_lines = [
            "# UniAD Task Head Comparison Report",
            "",
            f"**Analysis Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Alignment Mode**: {self.alignment_mode}",
            f"**Total Comparisons**: {len(comparison_data.comparisons)}",
            ""
        ]
        
        # Global scales
        global_scales = self._calculate_global_scales(comparison_data)
        report_lines.extend([
            "## Global Performance Scales",
            f"- **Max Memory Usage**: {global_scales['max_memory_mb']:.2f} MB",
            f"- **Max Compute Time**: {global_scales['max_compute_ms']:.2f} ms",
            f"- **Max Operation Count**: {global_scales['max_operation_count']}",
            ""
        ])
        
        # Individual comparisons
        report_lines.append("## Pairwise Comparisons")
        for comp in comparison_data.comparisons:
            report_lines.extend([
                f"### {comp.base_head.upper()} vs {comp.compare_head.upper()}",
                "",
                "**Performance Metrics:**",
                f"- Memory Difference: {comp.memory_diff:+.2f} MB ({comp.get_memory_percentage_diff():+.1f}%)",
                f"- Compute Time Difference: {comp.compute_diff:+.2f} ms ({comp.get_compute_percentage_diff():+.1f}%)",
                f"- Operation Overlap: {comp.get_operation_overlap_ratio():.1%}",
                "",
                "**Operations:**",
                f"- Common Operations: {len(comp.common_operations)}",
                f"- Unique to {comp.base_head}: {len(comp.unique_to_base)}",
                f"- Unique to {comp.compare_head}: {len(comp.unique_to_compare)}",
                ""
            ])
            
            # Dependency analysis
            dep_analysis = comp.get_dependency_analysis()
            if dep_analysis['is_hierarchical']:
                report_lines.append(f"**Note**: Hierarchical relationship detected between {comp.base_head} and {comp.compare_head}")
                report_lines.append("")
        
        # Head summaries
        report_lines.append("## Task Head Summaries")
        for head in self.task_heads:
            summary = comparison_data.get_head_summary(head)
            if summary:
                report_lines.extend([
                    f"### {head.upper()}",
                    f"- Involved in {summary['comparison_count']} comparisons",
                    f"- Average Memory: {summary['avg_memory_diff_mb']:.2f} MB",
                    f"- Average Compute: {summary['avg_compute_diff_ms']:.2f} ms",
                    f"- Average Operations: {summary['avg_operation_count']:.0f}",
                    ""
                ])
        
        return "\n".join(report_lines)