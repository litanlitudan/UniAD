"""Memory timeline data structures for PyTorch Operation Tracer

This module provides data structures for tracking memory usage over time,
specifically designed for UniAD's multi-task architecture and memory profiling needs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum


class MemoryEventType(Enum):
    """Types of memory events that can occur during model execution"""
    ALLOCATION = "allocation"
    DEALLOCATION = "deallocation"
    PEAK = "peak"
    THRESHOLD_WARNING = "threshold_warning"
    THRESHOLD_CRITICAL = "threshold_critical"


@dataclass
class MemoryTimelineEvent:
    """
    Represents a single memory event in the timeline during model execution.
    
    This dataclass tracks memory allocation/deallocation patterns to help identify
    problematic operations and memory bottlenecks in UniAD's multi-task architecture.
    """
    
    # Core timing and identification
    timestamp: float  # Time in milliseconds since trace start
    operation: str  # Operation name (e.g. "BEVFormerLayer.forward", "MotionHead.decode")
    module_path: str  # Full module path that triggered the event
    
    # Memory tracking
    memory_delta: float  # Memory change in MB (positive for allocation, negative for deallocation)
    cumulative_memory: float  # Total memory usage at this point in MB
    peak_memory: float = 0.0  # Peak memory usage seen so far in MB
    
    # Event classification
    event_type: MemoryEventType = MemoryEventType.ALLOCATION
    
    # Tensor and operation details
    tensor_info: Optional[Dict[str, Any]] = None  # Shape, dtype, device information
    
    # UniAD-specific context
    task_head: Optional[str] = None  # track/seg/motion/occ/planning
    temporal_index: Optional[int] = None  # Frame index in temporal queue
    is_bev_operation: bool = False  # Whether this is a BEV-related operation
    is_frozen_module: bool = False  # Whether module is frozen (Stage 2 BEV encoder)
    
    # Threshold analysis
    memory_threshold_mb: Optional[float] = None  # Configured memory threshold
    exceeds_threshold: bool = False  # Whether this event exceeds threshold
    
    # Performance context
    gpu_utilization: Optional[float] = None  # GPU utilization percentage
    compute_time: Optional[float] = None  # Compute time for this operation in ms
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields after dataclass creation"""
        # Update peak memory if current cumulative is higher
        if self.cumulative_memory > self.peak_memory:
            self.peak_memory = self.cumulative_memory
        
        # Check threshold exceedance
        if self.memory_threshold_mb is not None:
            self.exceeds_threshold = self.cumulative_memory > self.memory_threshold_mb
            
            # Update event type for threshold violations
            if self.exceeds_threshold:
                if self.cumulative_memory > self.memory_threshold_mb * 1.2:  # 20% over threshold
                    self.event_type = MemoryEventType.THRESHOLD_CRITICAL
                else:
                    self.event_type = MemoryEventType.THRESHOLD_WARNING
    
    def is_problematic(self, warning_threshold_mb: float = 15000, critical_threshold_mb: float = 25000) -> bool:
        """
        Determine if this memory event indicates a problematic operation.
        
        Args:
            warning_threshold_mb: Warning threshold in MB (default: 15GB)
            critical_threshold_mb: Critical threshold in MB (default: 25GB)
            
        Returns:
            True if the event indicates problematic memory usage
        """
        return (
            self.cumulative_memory > critical_threshold_mb or
            abs(self.memory_delta) > 5000 or  # Single operation uses >5GB
            self.event_type in [MemoryEventType.THRESHOLD_WARNING, MemoryEventType.THRESHOLD_CRITICAL]
        )
    
    def get_memory_efficiency_score(self) -> float:
        """
        Calculate a memory efficiency score for this event (0.0 to 1.0).
        
        Higher scores indicate more efficient memory usage.
        
        Returns:
            Memory efficiency score between 0.0 and 1.0
        """
        # UniAD expected memory usage by stage
        stage_expected = {
            1: 30000,  # Stage 1: ~30GB
            2: 17000   # Stage 2: ~17GB  
        }
        
        # Estimate stage based on memory usage patterns
        expected_memory = stage_expected[2]  # Default to Stage 2
        if self.cumulative_memory > 25000:
            expected_memory = stage_expected[1]
        
        # Calculate efficiency (closer to expected = higher efficiency)
        if expected_memory == 0:
            return 1.0
        
        efficiency = 1.0 - abs(self.cumulative_memory - expected_memory) / expected_memory
        return max(0.0, min(1.0, efficiency))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization and visualization"""
        return {
            'timestamp': self.timestamp,
            'operation': self.operation,
            'module_path': self.module_path,
            'memory_delta': self.memory_delta,
            'cumulative_memory': self.cumulative_memory,
            'peak_memory': self.peak_memory,
            'event_type': self.event_type.value,
            'tensor_info': self.tensor_info,
            'task_head': self.task_head,
            'temporal_index': self.temporal_index,
            'is_bev_operation': self.is_bev_operation,
            'is_frozen_module': self.is_frozen_module,
            'memory_threshold_mb': self.memory_threshold_mb,
            'exceeds_threshold': self.exceeds_threshold,
            'gpu_utilization': self.gpu_utilization,
            'compute_time': self.compute_time,
            'is_problematic': self.is_problematic(),
            'efficiency_score': self.get_memory_efficiency_score(),
            'metadata': self.metadata
        }
    
    @classmethod
    def create_allocation_event(
        cls, 
        timestamp: float,
        operation: str,
        module_path: str,
        memory_delta: float,
        cumulative_memory: float,
        **kwargs
    ) -> 'MemoryTimelineEvent':
        """
        Factory method for creating memory allocation events.
        
        Args:
            timestamp: Time in milliseconds
            operation: Operation name
            module_path: Module path
            memory_delta: Memory change in MB (should be positive)
            cumulative_memory: Total memory in MB
            **kwargs: Additional optional parameters
            
        Returns:
            MemoryTimelineEvent configured for allocation
        """
        return cls(
            timestamp=timestamp,
            operation=operation,
            module_path=module_path,
            memory_delta=abs(memory_delta),  # Ensure positive for allocation
            cumulative_memory=cumulative_memory,
            event_type=MemoryEventType.ALLOCATION,
            **kwargs
        )
    
    @classmethod
    def create_deallocation_event(
        cls,
        timestamp: float,
        operation: str,
        module_path: str,
        memory_delta: float,
        cumulative_memory: float,
        **kwargs
    ) -> 'MemoryTimelineEvent':
        """
        Factory method for creating memory deallocation events.
        
        Args:
            timestamp: Time in milliseconds
            operation: Operation name
            module_path: Module path
            memory_delta: Memory change in MB (should be negative)
            cumulative_memory: Total memory in MB
            **kwargs: Additional optional parameters
            
        Returns:
            MemoryTimelineEvent configured for deallocation
        """
        return cls(
            timestamp=timestamp,
            operation=operation,
            module_path=module_path,
            memory_delta=-abs(memory_delta),  # Ensure negative for deallocation
            cumulative_memory=cumulative_memory,
            event_type=MemoryEventType.DEALLOCATION,
            **kwargs
        )


@dataclass
class MemoryTimelineAnalysis:
    """
    Analysis results for a sequence of memory timeline events.
    
    Provides summary statistics and insights for memory usage patterns.
    """
    
    events: List[MemoryTimelineEvent] = field(default_factory=list)
    
    # Summary statistics
    total_duration_ms: float = 0.0
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_growth_rate: float = 0.0  # MB per second
    
    # Problem analysis
    problematic_operations: List[str] = field(default_factory=list)
    threshold_violations: int = 0
    efficiency_score: float = 0.0
    
    # Task head breakdown
    task_head_memory: Dict[str, float] = field(default_factory=dict)
    
    def analyze(self) -> None:
        """Analyze the timeline events and compute summary statistics"""
        if not self.events:
            return
        
        # Time range
        start_time = min(event.timestamp for event in self.events)
        end_time = max(event.timestamp for event in self.events)
        self.total_duration_ms = end_time - start_time
        
        # Memory statistics
        cumulative_memories = [event.cumulative_memory for event in self.events]
        self.peak_memory_mb = max(cumulative_memories)
        self.average_memory_mb = sum(cumulative_memories) / len(cumulative_memories)
        
        # Growth rate calculation
        if self.total_duration_ms > 0:
            memory_change = cumulative_memories[-1] - cumulative_memories[0]
            self.memory_growth_rate = memory_change / (self.total_duration_ms / 1000.0)  # MB/s
        
        # Problem analysis
        self.problematic_operations = [
            event.operation for event in self.events 
            if event.is_problematic()
        ]
        self.threshold_violations = sum(1 for event in self.events if event.exceeds_threshold)
        
        # Efficiency score (average of individual scores)
        efficiency_scores = [event.get_memory_efficiency_score() for event in self.events]
        self.efficiency_score = sum(efficiency_scores) / len(efficiency_scores)
        
        # Task head breakdown
        task_memory = {}
        for event in self.events:
            if event.task_head and event.memory_delta > 0:  # Only count allocations
                if event.task_head not in task_memory:
                    task_memory[event.task_head] = 0
                task_memory[event.task_head] += event.memory_delta
        
        self.task_head_memory = task_memory
    
    def get_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on the analysis"""
        recommendations = []
        
        # High memory usage
        if self.peak_memory_mb > 30000:  # > 30GB
            recommendations.append("Consider reducing queue_length or batch_size to lower memory usage")
        
        # Memory growth rate
        if self.memory_growth_rate > 1000:  # > 1GB/s growth
            recommendations.append("High memory growth rate detected - check for memory leaks")
        
        # Task head imbalance
        if self.task_head_memory:
            max_task_memory = max(self.task_head_memory.values())
            min_task_memory = min(self.task_head_memory.values()) if len(self.task_head_memory) > 1 else max_task_memory
            
            if max_task_memory > min_task_memory * 3:  # 3x imbalance
                max_task = max(self.task_head_memory.keys(), key=lambda k: self.task_head_memory[k])
                recommendations.append(f"Task head '{max_task}' uses significantly more memory - consider optimization")
        
        # Low efficiency
        if self.efficiency_score < 0.7:
            recommendations.append("Memory efficiency is below optimal - consider mixed precision training")
        
        # Threshold violations
        if self.threshold_violations > len(self.events) * 0.1:  # > 10% of operations
            recommendations.append("Frequent memory threshold violations - consider lowering memory limits")
        
        return recommendations