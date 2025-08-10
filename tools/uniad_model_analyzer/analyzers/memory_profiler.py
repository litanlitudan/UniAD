"""
Memory Profiler for UniAD Model Analyzer.

This module provides detailed memory profiling for UniAD models,
tracking GPU memory usage, identifying bottlenecks, and suggesting optimizations.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import torch

from core.data_structures import TraceNode, TaskHead, MemoryProfile


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a specific point."""
    
    timestamp_ns: float
    operation_id: str
    module_path: str
    
    # Memory values in bytes
    allocated: int
    reserved: int
    active: int
    inactive: int
    
    # CUDA specific
    cuda_allocated: int
    cuda_reserved: int
    cuda_cached: int
    
    # Delta from previous snapshot
    allocated_delta: int = 0
    reserved_delta: int = 0
    
    def get_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.allocated / (1024 * 1024)
    
    def get_reserved_mb(self) -> float:
        """Get reserved memory in MB."""
        return self.reserved / (1024 * 1024)


@dataclass
class MemoryRegion:
    """Represents a memory region with specific characteristics."""
    
    name: str
    start_time_ns: float
    end_time_ns: float
    operations: List[TraceNode] = field(default_factory=list)
    
    # Memory characteristics
    peak_memory: int = 0
    avg_memory: float = 0.0
    memory_variance: float = 0.0
    
    # Allocation patterns
    allocation_count: int = 0
    deallocation_count: int = 0
    reallocation_count: int = 0
    
    # Efficiency metrics
    memory_efficiency: float = 0.0
    fragmentation_score: float = 0.0


@dataclass
class OptimizationSuggestion:
    """Memory optimization suggestion."""
    
    category: str  # 'reduction', 'reordering', 'caching', 'precision', 'checkpointing'
    severity: str  # 'critical', 'high', 'medium', 'low'
    target: str  # Module or operation to optimize
    
    description: str
    expected_savings_mb: float
    implementation_difficulty: str  # 'easy', 'medium', 'hard'
    
    # Specific recommendations
    techniques: List[str] = field(default_factory=list)
    code_changes: List[str] = field(default_factory=list)


class MemoryProfiler:
    """
    Advanced memory profiler for UniAD models.
    
    Tracks memory usage patterns, identifies bottlenecks, and provides
    optimization suggestions for the 30-50GB GPU memory requirements.
    """
    
    def __init__(self, 
                 target_memory_gb: float = 40.0,
                 optimization_threshold: float = 0.8):
        """
        Initialize the memory profiler.
        
        Args:
            target_memory_gb: Target GPU memory budget
            optimization_threshold: Threshold for triggering optimizations (0-1)
        """
        self.target_memory_gb = target_memory_gb
        self.optimization_threshold = optimization_threshold
        
        # Memory tracking
        self.snapshots: List[MemorySnapshot] = []
        self.memory_regions: Dict[str, MemoryRegion] = {}
        
        # Peak tracking
        self.peak_memory: int = 0
        self.peak_operation: Optional[TraceNode] = None
        self.peak_timestamp: float = 0
        
        # Memory patterns
        self.allocation_patterns: Dict[str, List[int]] = defaultdict(list)
        self.task_head_memory: Dict[TaskHead, int] = defaultdict(int)
        
        # Optimization suggestions
        self.suggestions: List[OptimizationSuggestion] = []
        
        # Memory thresholds (in bytes)
        self.critical_threshold = int(target_memory_gb * 0.95 * 1024 * 1024 * 1024)
        self.warning_threshold = int(target_memory_gb * 0.8 * 1024 * 1024 * 1024)
        self.target_threshold = int(target_memory_gb * 1024 * 1024 * 1024)
    
    def profile_memory(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Profile memory usage from trace data.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Comprehensive memory analysis
        """
        # Create memory snapshots
        self._create_snapshots(trace_data)
        
        # Identify memory regions
        self._identify_memory_regions(trace_data)
        
        # Analyze allocation patterns
        self._analyze_allocation_patterns(trace_data)
        
        # Track task head memory
        self._track_task_head_memory(trace_data)
        
        # Generate optimization suggestions
        self._generate_optimization_suggestions()
        
        # Create analysis report
        analysis = {
            'summary': self._generate_summary(),
            'peak_analysis': self._analyze_peak_memory(),
            'memory_timeline': self._generate_timeline(),
            'allocation_patterns': self._summarize_allocation_patterns(),
            'task_head_distribution': self._analyze_task_head_distribution(),
            'memory_regions': self._summarize_memory_regions(),
            'optimization_suggestions': self._format_suggestions(),
            'memory_efficiency': self._calculate_memory_efficiency(),
        }
        
        return analysis
    
    def analyze_memory_lifecycle(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze memory lifecycle patterns.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Memory lifecycle analysis
        """
        lifecycle = {
            'allocation_frequency': {},
            'retention_patterns': {},
            'deallocation_timing': {},
            'memory_leaks': [],
        }
        
        # Track allocations and deallocations
        allocations = defaultdict(list)
        deallocations = defaultdict(list)
        
        for node in trace_data:
            if node.cuda_memory_allocated > 0:
                allocations[node.module_path].append({
                    'time': node.start_time,
                    'size': node.cuda_memory_allocated,
                    'operation': node.name,
                })
            
            if node.cuda_memory_freed > 0:
                deallocations[node.module_path].append({
                    'time': node.start_time + node.duration,  # Calculate end time
                    'size': node.cuda_memory_freed,
                    'operation': node.name,
                })
        
        # Analyze patterns
        for module in allocations:
            alloc_times = [a['time'] for a in allocations[module]]
            dealloc_times = [d['time'] for d in deallocations.get(module, [])]
            
            # Calculate allocation frequency
            if len(alloc_times) > 1:
                intervals = np.array(np.diff(sorted(alloc_times)))
                lifecycle['allocation_frequency'][module] = {
                    'count': len(alloc_times),
                    'avg_interval_ns': float(np.mean(intervals)),
                    'pattern': self._classify_allocation_pattern(intervals),
                }
            
            # Check for potential leaks
            total_allocated = sum(a['size'] for a in allocations[module])
            total_freed = sum(d['size'] for d in deallocations.get(module, []))
            
            if total_allocated > total_freed * 1.1:  # 10% threshold
                lifecycle['memory_leaks'].append({
                    'module': module,
                    'allocated_mb': total_allocated / (1024 * 1024),
                    'freed_mb': total_freed / (1024 * 1024),
                    'leaked_mb': (total_allocated - total_freed) / (1024 * 1024),
                })
        
        return lifecycle
    
    def suggest_gradient_checkpointing(self, trace_data: List[TraceNode]) -> List[Dict[str, Any]]:
        """
        Suggest gradient checkpointing opportunities.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            List of checkpointing suggestions
        """
        suggestions = []
        
        # Find large activation tensors
        for node in trace_data:
            if node.gradient_info and node.cuda_memory_allocated > 100 * 1024 * 1024:  # 100MB
                # Check if this is a good checkpointing candidate
                if self._is_checkpointing_candidate(node):
                    suggestions.append({
                        'module': node.module_path,
                        'operation': node.name,
                        'activation_memory_mb': node.cuda_memory_allocated / (1024 * 1024),
                        'expected_savings_mb': node.cuda_memory_allocated * 0.5 / (1024 * 1024),
                        'recomputation_time_ms': node.duration / 1e6,
                        'recommendation': f"Apply gradient checkpointing to {node.module_path}",
                    })
        
        # Sort by potential savings
        suggestions.sort(key=lambda x: x['expected_savings_mb'], reverse=True)
        
        return suggestions[:10]  # Top 10 suggestions
    
    def analyze_mixed_precision_opportunities(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze opportunities for mixed precision training.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Mixed precision analysis
        """
        fp32_operations = []
        fp16_compatible = []
        
        for node in trace_data:
            # Check tensor dtypes
            for shape in node.output_shapes:
                if shape.dtype == torch.float32:
                    fp32_operations.append(node)
                    
                    # Check if compatible with FP16
                    if self._is_fp16_compatible(node):
                        fp16_compatible.append(node)
                    break
        
        total_fp32_memory = sum(n.cuda_memory_allocated for n in fp32_operations)
        potential_savings = sum(n.cuda_memory_allocated * 0.5 for n in fp16_compatible)
        
        return {
            'fp32_operations': len(fp32_operations),
            'fp16_compatible': len(fp16_compatible),
            'current_memory_gb': total_fp32_memory / (1024**3),
            'potential_savings_gb': potential_savings / (1024**3),
            'conversion_percentage': len(fp16_compatible) / len(fp32_operations) * 100 if fp32_operations else 0,
            'recommendations': self._generate_mixed_precision_recommendations(fp16_compatible),
        }
    
    def _create_snapshots(self, trace_data: List[TraceNode]) -> None:
        """Create memory snapshots from trace data."""
        self.snapshots = []
        
        current_allocated = 0
        current_reserved = 0
        
        for node in trace_data:
            # Update current memory state
            current_allocated += node.get_cuda_memory_delta()
            current_reserved = max(current_reserved, current_allocated)
            
            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp_ns=node.start_time,
                operation_id=node.id,
                module_path=node.module_path,
                allocated=current_allocated,
                reserved=current_reserved,
                active=current_allocated,
                inactive=current_reserved - current_allocated,
                cuda_allocated=node.cuda_memory_allocated,
                cuda_reserved=current_reserved,  # Use current_reserved instead
                cuda_cached=0,  # Would need actual CUDA cache info
            )
            
            # Calculate deltas
            if self.snapshots:
                prev = self.snapshots[-1]
                snapshot.allocated_delta = snapshot.allocated - prev.allocated
                snapshot.reserved_delta = snapshot.reserved - prev.reserved
            
            self.snapshots.append(snapshot)
            
            # Track peak
            if current_allocated > self.peak_memory:
                self.peak_memory = current_allocated
                self.peak_operation = node
                self.peak_timestamp = node.start_time
    
    def _identify_memory_regions(self, trace_data: List[TraceNode]) -> None:
        """Identify distinct memory usage regions."""
        self.memory_regions = {}
        
        # Simple region identification based on memory patterns
        region_threshold = self.target_threshold * 0.1  # 10% of target
        
        current_region = None
        region_start = 0
        region_operations = []
        
        for i, snapshot in enumerate(self.snapshots):
            if current_region is None:
                # Start new region
                current_region = f"region_{len(self.memory_regions)}"
                region_start = snapshot.timestamp_ns
                region_operations = []
            
            # Add to current region
            if i < len(trace_data):
                region_operations.append(trace_data[i])
            
            # Check for region boundary
            if abs(snapshot.allocated_delta) > region_threshold:
                # End current region and start new one
                if region_operations:
                    self._create_memory_region(
                        current_region, region_start, 
                        snapshot.timestamp_ns, region_operations
                    )
                
                current_region = None
        
        # Handle last region
        if current_region and region_operations:
            self._create_memory_region(
                current_region, region_start,
                self.snapshots[-1].timestamp_ns if self.snapshots else 0,
                region_operations
            )
    
    def _create_memory_region(self, name: str, start: float, end: float,
                            operations: List[TraceNode]) -> None:
        """Create a memory region from operations."""
        region = MemoryRegion(
            name=name,
            start_time_ns=start,
            end_time_ns=end,
            operations=operations,
        )
        
        # Calculate region statistics
        if operations:
            memories = [op.cuda_memory_allocated for op in operations]
            region.peak_memory = max(memories) if memories else 0
            region.avg_memory = np.mean(memories) if memories else 0
            region.memory_variance = np.var(memories) if memories else 0
            
            # Count allocations/deallocations
            for op in operations:
                if op.cuda_memory_allocated > 0:
                    region.allocation_count += 1
                if op.cuda_memory_freed > 0:
                    region.deallocation_count += 1
            
            # Calculate efficiency
            if region.peak_memory > 0:
                region.memory_efficiency = region.avg_memory / region.peak_memory
        
        self.memory_regions[name] = region
    
    def _analyze_allocation_patterns(self, trace_data: List[TraceNode]) -> None:
        """Analyze memory allocation patterns."""
        self.allocation_patterns = defaultdict(list)
        
        for node in trace_data:
            if node.cuda_memory_allocated > 0:
                # Group by operation type
                pattern_key = node.name
                self.allocation_patterns[pattern_key].append(node.cuda_memory_allocated)
    
    def _track_task_head_memory(self, trace_data: List[TraceNode]) -> None:
        """Track memory usage by task head."""
        self.task_head_memory = defaultdict(int)
        
        for node in trace_data:
            self.task_head_memory[node.task_head] += node.get_cuda_memory_delta()
    
    def _generate_optimization_suggestions(self) -> None:
        """Generate memory optimization suggestions."""
        self.suggestions = []
        
        # Check for high peak memory
        if self.peak_memory > self.warning_threshold:
            self._suggest_peak_reduction()
        
        # Check for inefficient regions
        for region in self.memory_regions.values():
            if region.memory_efficiency < 0.7:
                self._suggest_region_optimization(region)
        
        # Check allocation patterns
        self._suggest_allocation_optimizations()
        
        # Check task head distribution
        self._suggest_task_head_optimizations()
    
    def _suggest_peak_reduction(self) -> None:
        """Suggest ways to reduce peak memory."""
        if self.peak_operation:
            suggestion = OptimizationSuggestion(
                category='reduction',
                severity='critical' if self.peak_memory > self.critical_threshold else 'high',
                target=self.peak_operation.module_path,
                description=f"Peak memory ({self.peak_memory / (1024**3):.2f}GB) exceeds target. "
                          f"Operation: {self.peak_operation.name}",
                expected_savings_mb=(self.peak_memory - self.target_threshold) / (1024 * 1024),
                implementation_difficulty='medium',
                techniques=[
                    "Use gradient checkpointing",
                    "Reduce batch size",
                    "Enable mixed precision (FP16)",
                    "Implement activation checkpointing",
                ],
                code_changes=[
                    "torch.utils.checkpoint.checkpoint(module, inputs)",
                    "with torch.cuda.amp.autocast():",
                    "model = model.half()",
                ],
            )
            self.suggestions.append(suggestion)
    
    def _suggest_region_optimization(self, region: MemoryRegion) -> None:
        """Suggest optimizations for a memory region."""
        if region.memory_efficiency < 0.5:
            suggestion = OptimizationSuggestion(
                category='reordering',
                severity='medium',
                target=region.name,
                description=f"Memory region '{region.name}' has low efficiency ({region.memory_efficiency:.2%})",
                expected_savings_mb=(region.peak_memory - region.avg_memory) / (1024 * 1024),
                implementation_difficulty='medium',
                techniques=[
                    "Reorder operations to reduce peak memory",
                    "Free intermediate tensors earlier",
                    "Use in-place operations where possible",
                ],
                code_changes=[
                    "del intermediate_tensor",
                    "tensor.add_(other)  # In-place",
                    "with torch.no_grad():",
                ],
            )
            self.suggestions.append(suggestion)
    
    def _suggest_allocation_optimizations(self) -> None:
        """Suggest optimizations for allocation patterns."""
        for op_type, allocations in self.allocation_patterns.items():
            if len(allocations) > 10 and np.std(allocations) > np.mean(allocations) * 0.5:
                suggestion = OptimizationSuggestion(
                    category='caching',
                    severity='low',
                    target=op_type,
                    description=f"High allocation variance for '{op_type}' operations",
                    expected_savings_mb=np.std(allocations) / (1024 * 1024),
                    implementation_difficulty='easy',
                    techniques=[
                        "Pre-allocate tensors",
                        "Use tensor caching",
                        "Implement memory pooling",
                    ],
                    code_changes=[
                        "buffer = torch.empty(size, device='cuda')",
                        "cache = {}  # Reuse tensors",
                    ],
                )
                self.suggestions.append(suggestion)
    
    def _suggest_task_head_optimizations(self) -> None:
        """Suggest optimizations for task head memory distribution."""
        total_memory = sum(self.task_head_memory.values())
        
        for task_head, memory in self.task_head_memory.items():
            if task_head != TaskHead.NONE and memory > total_memory * 0.3:
                suggestion = OptimizationSuggestion(
                    category='precision',
                    severity='medium',
                    target=task_head.value,
                    description=f"Task head '{task_head.value}' uses {memory/(1024**3):.2f}GB "
                              f"({memory/total_memory*100:.1f}% of total)",
                    expected_savings_mb=memory * 0.3 / (1024 * 1024),  # Assume 30% reduction possible
                    implementation_difficulty='medium',
                    techniques=[
                        f"Use mixed precision for {task_head.value} head",
                        "Reduce hidden dimensions",
                        "Share features across heads",
                    ],
                    code_changes=[
                        f"{task_head.value}_head = {task_head.value}_head.half()",
                        "nn.Linear(hidden_dim // 2, output_dim)",
                    ],
                )
                self.suggestions.append(suggestion)
    
    def _classify_allocation_pattern(self, intervals: np.ndarray) -> str:
        """Classify allocation pattern based on intervals."""
        if len(intervals) < 2:
            return "sparse"
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        if cv < 0.1:
            return "regular"
        elif cv < 0.5:
            return "periodic"
        else:
            return "irregular"
    
    def _is_checkpointing_candidate(self, node: TraceNode) -> bool:
        """Check if a node is a good candidate for gradient checkpointing."""
        # Good candidates have large activations but relatively fast computation
        if node.cuda_memory_allocated < 50 * 1024 * 1024:  # Less than 50MB
            return False
        
        # Check computation/memory ratio
        compute_intensity = node.duration / node.cuda_memory_allocated if node.cuda_memory_allocated > 0 else 0
        
        # Low compute intensity means good checkpointing candidate
        return compute_intensity < 1e-6  # Threshold in ns/byte
    
    def _is_fp16_compatible(self, node: TraceNode) -> bool:
        """Check if an operation is compatible with FP16."""
        # Conservative list of FP16-compatible operations
        fp16_compatible_ops = [
            'conv', 'linear', 'matmul', 'bmm', 'relu', 'gelu',
            'layernorm', 'batchnorm', 'dropout', 'attention'
        ]
        
        op_lower = node.name.lower()
        return any(op in op_lower for op in fp16_compatible_ops)
    
    def _generate_mixed_precision_recommendations(self, 
                                                  fp16_compatible: List[TraceNode]) -> List[str]:
        """Generate mixed precision recommendations."""
        recommendations = []
        
        if len(fp16_compatible) > 10:
            recommendations.append("Enable automatic mixed precision (AMP) training")
            recommendations.append("Use torch.cuda.amp.GradScaler() for gradient scaling")
        
        # Group by module
        module_memory = defaultdict(int)
        for node in fp16_compatible:
            module = node.module_path.split('.')[0] if '.' in node.module_path else node.module_path
            module_memory[module] += node.cuda_memory_allocated
        
        # Top modules for FP16 conversion
        top_modules = sorted(module_memory.items(), key=lambda x: x[1], reverse=True)[:5]
        for module, memory in top_modules:
            recommendations.append(
                f"Convert {module} to FP16 (saves ~{memory/(2*1024**2):.1f}MB)"
            )
        
        return recommendations
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate memory profiling summary."""
        return {
            'peak_memory_gb': self.peak_memory / (1024**3),
            'target_memory_gb': self.target_memory_gb,
            'memory_utilization': self.peak_memory / self.target_threshold * 100,
            'total_snapshots': len(self.snapshots),
            'memory_regions': len(self.memory_regions),
            'optimization_suggestions': len(self.suggestions),
            'status': self._get_memory_status(),
        }
    
    def _get_memory_status(self) -> str:
        """Get memory status based on peak usage."""
        if self.peak_memory > self.critical_threshold:
            return "critical"
        elif self.peak_memory > self.warning_threshold:
            return "warning"
        elif self.peak_memory > self.target_threshold * 0.5:
            return "optimal"
        else:
            return "underutilized"
    
    def _analyze_peak_memory(self) -> Dict[str, Any]:
        """Analyze peak memory usage."""
        if not self.peak_operation:
            return {'no_peak_found': True}
        
        return {
            'peak_memory_gb': self.peak_memory / (1024**3),
            'peak_operation': self.peak_operation.name,
            'peak_module': self.peak_operation.module_path,
            'peak_timestamp_ms': self.peak_timestamp / 1e6,
            'peak_percentage': self.peak_memory / self.target_threshold * 100,
        }
    
    def _generate_timeline(self) -> List[Dict[str, Any]]:
        """Generate memory timeline for visualization."""
        timeline = []
        
        for snapshot in self.snapshots[::max(1, len(self.snapshots) // 100)]:  # Sample 100 points
            timeline.append({
                'timestamp_ms': snapshot.timestamp_ns / 1e6,
                'allocated_mb': snapshot.allocated / (1024 * 1024),
                'reserved_mb': snapshot.reserved / (1024 * 1024),
                'delta_mb': snapshot.allocated_delta / (1024 * 1024),
            })
        
        return timeline
    
    def _summarize_allocation_patterns(self) -> Dict[str, Any]:
        """Summarize allocation patterns."""
        pattern_summary = {}
        
        for op_type, allocations in self.allocation_patterns.items():
            if allocations:
                pattern_summary[op_type] = {
                    'count': len(allocations),
                    'total_mb': sum(allocations) / (1024 * 1024),
                    'avg_mb': np.mean(allocations) / (1024 * 1024),
                    'std_mb': np.std(allocations) / (1024 * 1024),
                    'max_mb': max(allocations) / (1024 * 1024),
                }
        
        return pattern_summary
    
    def _analyze_task_head_distribution(self) -> Dict[str, Any]:
        """Analyze memory distribution across task heads."""
        total_memory = sum(self.task_head_memory.values())
        
        distribution = {}
        for task_head, memory in self.task_head_memory.items():
            if task_head != TaskHead.NONE:
                distribution[task_head.value] = {
                    'memory_mb': memory / (1024 * 1024),
                    'percentage': memory / total_memory * 100 if total_memory > 0 else 0,
                }
        
        return distribution
    
    def _summarize_memory_regions(self) -> List[Dict[str, Any]]:
        """Summarize memory regions."""
        summaries = []
        
        for name, region in self.memory_regions.items():
            summary = {
                'name': name,
                'duration_ms': (region.end_time_ns - region.start_time_ns) / 1e6,
                'operation_count': len(region.operations),
                'peak_memory_mb': region.peak_memory / (1024 * 1024),
                'avg_memory_mb': region.avg_memory / (1024 * 1024),
                'efficiency': region.memory_efficiency,
                'allocations': region.allocation_count,
                'deallocations': region.deallocation_count,
            }
            summaries.append(summary)
        
        return summaries
    
    def _format_suggestions(self) -> List[Dict[str, Any]]:
        """Format optimization suggestions for output."""
        formatted = []
        
        # Sort by severity and expected savings
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_suggestions = sorted(
            self.suggestions,
            key=lambda x: (severity_order[x.severity], -x.expected_savings_mb)
        )
        
        for suggestion in sorted_suggestions[:20]:  # Top 20 suggestions
            formatted.append({
                'category': suggestion.category,
                'severity': suggestion.severity,
                'target': suggestion.target,
                'description': suggestion.description,
                'expected_savings_mb': suggestion.expected_savings_mb,
                'difficulty': suggestion.implementation_difficulty,
                'techniques': suggestion.techniques,
                'code_changes': suggestion.code_changes,
            })
        
        return formatted
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate overall memory efficiency score."""
        if not self.snapshots:
            return 0.0
        
        # Factors for efficiency
        peak_efficiency = min(1.0, self.target_threshold / self.peak_memory) if self.peak_memory > 0 else 0
        
        # Average utilization
        avg_memory = np.mean([s.allocated for s in self.snapshots])
        utilization_efficiency = avg_memory / self.peak_memory if self.peak_memory > 0 else 0
        
        # Region efficiency
        region_efficiencies = [r.memory_efficiency for r in self.memory_regions.values()]
        avg_region_efficiency = np.mean(region_efficiencies) if region_efficiencies else 0
        
        # Combined score
        efficiency = (
            peak_efficiency * 0.4 +
            utilization_efficiency * 0.3 +
            avg_region_efficiency * 0.3
        )
        
        return min(1.0, max(0.0, efficiency))
    
    def export_analysis(self) -> Dict[str, Any]:
        """Export complete memory analysis."""
        return {
            'summary': self._generate_summary(),
            'peak_analysis': self._analyze_peak_memory(),
            'timeline': self._generate_timeline(),
            'allocation_patterns': self._summarize_allocation_patterns(),
            'task_head_distribution': self._analyze_task_head_distribution(),
            'memory_regions': self._summarize_memory_regions(),
            'suggestions': self._format_suggestions(),
            'efficiency_score': self._calculate_memory_efficiency(),
        }