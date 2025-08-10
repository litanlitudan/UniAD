"""
UniAD Model Analyzer

A comprehensive PyTorch model analysis tool specifically designed for the UniAD 
autonomous driving model. Provides deep insights into the model's multi-task 
architecture through operation tracing, memory profiling, and visualization.

Main Features:
- Operation tracing and profiling for all 5 task heads
- Memory profiling for 30-50GB GPU optimization
- BEV encoder and temporal aggregation analysis
- Mixed precision optimization recommendations
- Interactive visualizations with Mermaid diagrams
- Export to multiple formats (JSON, CSV, TensorBoard, ONNX)

Author: UniAD Model Analyzer Team
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "UniAD Model Analyzer Team"

from .core.data_structures import (
    AnalysisConfig, 
    TraceNode, 
    MemoryProfile,
    TaskHead,
    AnalysisStage,
    TensorShape,
    AnalysisResult,
)
from .core.hook_manager import HookManager

__all__ = [
    "AnalysisConfig", 
    "TraceNode",
    "MemoryProfile",
    "TaskHead",
    "AnalysisStage",
    "TensorShape",
    "AnalysisResult",
    "HookManager",
    "__version__",
]