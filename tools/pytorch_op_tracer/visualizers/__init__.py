"""Visualizers for PyTorch Operation Tracer"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "DataflowVisualizer", 
    "InteractiveVisualizer", 
    "FilterEngine", 
    "FilterType", 
    "FilterCriteria", 
    "SearchResult", 
    "PathInfo",
    "MemoryTimelineVisualizer",
    "MemoryPeak",
    "MemoryTimelineData",
    "create_memory_timeline_from_trace_nodes",
    "TaskHeadComparator",
    "QueueVisualizer",
    "ProgressiveRenderer"
]

def __getattr__(name):
    """Lazy import modules to avoid circular dependencies"""
    if name == "DataflowVisualizer":
        from .mermaid_visualizer import DataflowVisualizer
        return DataflowVisualizer
    elif name == "InteractiveVisualizer":
        from .interactive_visualizer import InteractiveVisualizer
        return InteractiveVisualizer
    elif name == "FilterEngine":
        from .filter_engine import FilterEngine
        return FilterEngine
    elif name in ["FilterType", "FilterCriteria", "SearchResult", "PathInfo"]:
        from .filter_engine import FilterType, FilterCriteria, SearchResult, PathInfo
        return locals()[name]
    elif name == "MemoryTimelineVisualizer":
        from .memory_timeline import MemoryTimelineVisualizer
        return MemoryTimelineVisualizer
    elif name in ["MemoryPeak", "MemoryTimelineData", "create_memory_timeline_from_trace_nodes"]:
        from .memory_timeline import MemoryPeak, MemoryTimelineData, create_memory_timeline_from_trace_nodes
        return locals()[name]
    elif name == "TaskHeadComparator":
        from .task_head_comparator import TaskHeadComparator
        return TaskHeadComparator
    elif name == "QueueVisualizer":
        from .queue_visualizer import QueueVisualizer
        return QueueVisualizer
    elif name == "ProgressiveRenderer":
        from .progressive_renderer import ProgressiveRenderer
        return ProgressiveRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")