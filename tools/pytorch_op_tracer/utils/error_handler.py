"""Error handling utilities for visualization components

This module provides comprehensive error handling for the visualization pipeline,
including graceful degradation, incomplete trace handling, and memory overflow
protection for UniAD's complex multi-task architecture.
"""

import functools
import logging
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, TypeVar, Tuple, Type
from dataclasses import dataclass
import psutil
import gc
# json import removed as unused

try:
    from ..core.data_structures import TraceNode
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ErrorContext:
    """Context information for error handling"""
    function_name: str
    error_type: str
    error_message: str
    traceback: str
    args: tuple
    kwargs: dict
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_used: Optional[str] = None
    

class VisualizationError(Exception):
    """Base exception for visualization errors"""
    pass


class IncompleteTraceError(VisualizationError):
    """Exception for incomplete trace data"""
    pass


class MemoryOverflowError(VisualizationError):
    """Exception for memory overflow during visualization"""
    pass


class ErrorHandler:
    """Main error handler for visualization components"""
    
    # Memory thresholds
    MEMORY_WARNING_THRESHOLD = 0.75  # 75% memory usage
    MEMORY_CRITICAL_THRESHOLD = 0.90  # 90% memory usage
    
    # Error recovery strategies
    RECOVERY_STRATEGIES = {
        'memory': ['reduce_nodes', 'simplify_visualization', 'use_static'],
        'incomplete': ['mark_missing', 'partial_visualization', 'interpolate'],
        'rendering': ['fallback_renderer', 'static_output', 'text_only'],
        'data': ['validate_and_clean', 'use_defaults', 'skip_invalid']
    }
    
    def __init__(self, enable_recovery: bool = True, 
                 enable_logging: bool = True,
                 max_retries: int = 3):
        """Initialize error handler
        
        Args:
            enable_recovery: Whether to attempt automatic recovery
            enable_logging: Whether to log errors
            max_retries: Maximum number of retry attempts
        """
        self.enable_recovery = enable_recovery
        self.enable_logging = enable_logging
        self.max_retries = max_retries
        self.error_history: List[ErrorContext] = []
        
    def visualization_error_handler(self, fallback_result: Any = None, 
                                   error_types: Tuple[Type[Exception], ...] = (Exception,)):
        """Decorator for handling visualization errors with graceful degradation
        
        Args:
            fallback_result: Result to return if error occurs
            error_types: Tuple of exception types to catch
            
        Returns:
            Decorated function with error handling
        """
        def decorator(func: F):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                last_error = None
                
                while retries <= self.max_retries:
                    try:
                        # Check memory before execution
                        self._check_memory_availability()
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Validate result if applicable
                        if hasattr(result, '__len__') and len(result) == 0:
                            warnings.warn(f"{func.__name__} returned empty result")
                        
                        return result
                        
                    except error_types as e:
                        last_error = e
                        context = self._create_error_context(func, e, args, kwargs)
                        
                        if self.enable_logging:
                            self._log_error(context)
                        
                        self.error_history.append(context)
                        
                        # Attempt recovery
                        if self.enable_recovery and retries < self.max_retries:
                            recovery_result = self._attempt_recovery(
                                func, e, context, args, kwargs
                            )
                            if recovery_result is not None:
                                context.recovery_successful = True
                                return recovery_result
                        
                        retries += 1
                
                # All retries exhausted
                if self.enable_logging:
                    logger.error(
                        f"All recovery attempts failed for {func.__name__}: {last_error}"
                    )
                
                # Return fallback or raise
                if fallback_result is not None:
                    if callable(fallback_result):
                        return fallback_result(*args, **kwargs)
                    return fallback_result
                else:
                    if last_error is not None:
                        raise last_error
                    else:
                        raise RuntimeError("Function failed without specific error")
                    
            return wrapper
        return decorator
    
    def _create_error_context(self, func: Callable, error: Exception, 
                             args: tuple, kwargs: dict) -> ErrorContext:
        """Create error context for logging and recovery"""
        return ErrorContext(
            function_name=func.__name__,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            args=args,
            kwargs=kwargs
        )
    
    def _log_error(self, context: ErrorContext):
        """Log error with context"""
        logger.error(
            f"Error in {context.function_name}: {context.error_type} - {context.error_message}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Traceback:\n{context.traceback}")
    
    def _check_memory_availability(self):
        """Check system memory availability"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0
            
            if usage_percent > self.MEMORY_CRITICAL_THRESHOLD:
                # Force garbage collection
                gc.collect()
                
                # Re-check after GC
                memory = psutil.virtual_memory()
                usage_percent = memory.percent / 100.0
                
                if usage_percent > self.MEMORY_CRITICAL_THRESHOLD:
                    raise MemoryOverflowError(
                        f"Critical memory usage: {usage_percent:.1%}"
                    )
            
            elif usage_percent > self.MEMORY_WARNING_THRESHOLD:
                warnings.warn(
                    f"High memory usage: {usage_percent:.1%}. "
                    "Consider reducing visualization complexity."
                )
                
        except (ImportError, ValueError):
            # psutil not available, skip memory check
            pass
    
    def _attempt_recovery(self, func: Callable, error: Exception, 
                         context: ErrorContext, args: tuple, 
                         kwargs: dict) -> Optional[Any]:
        """Attempt to recover from error
        
        Args:
            func: Original function that failed
            error: The exception that occurred
            context: Error context
            args: Original function arguments
            kwargs: Original function keyword arguments
            
        Returns:
            Recovery result if successful, None otherwise
        """
        context.recovery_attempted = True
        
        # Memory overflow recovery
        if isinstance(error, MemoryOverflowError):
            return self._recover_from_memory_overflow(func, args, kwargs)
        
        # Incomplete trace recovery
        elif isinstance(error, IncompleteTraceError):
            return self._recover_from_incomplete_trace(func, args, kwargs)
        
        # Generic visualization error recovery
        elif isinstance(error, VisualizationError):
            return self._recover_from_visualization_error(func, args, kwargs)
        
        # Default recovery for other errors
        else:
            return self._default_recovery(func, error, args, kwargs)
    
    def _recover_from_memory_overflow(self, func: Callable, 
                                     args: tuple, kwargs: dict) -> Optional[Any]:
        """Recover from memory overflow by reducing complexity"""
        logger.info("Attempting recovery from memory overflow...")
        
        # Strategy 1: Reduce number of nodes
        if 'trace_nodes' in kwargs or (args and isinstance(args[0], list)):
            nodes = kwargs.get('trace_nodes', args[0] if args else [])
            
            if isinstance(nodes, list) and len(nodes) > 100:
                # Keep only top 100 nodes by importance
                reduced_nodes = self._reduce_nodes(nodes, max_nodes=100)
                
                if 'trace_nodes' in kwargs:
                    kwargs['trace_nodes'] = reduced_nodes
                else:
                    args = (reduced_nodes,) + args[1:]
                
                logger.info(f"Reduced nodes from {len(nodes)} to {len(reduced_nodes)}")
                
                # Retry with reduced nodes
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Recovery with reduced nodes failed: {e}")
        
        # Strategy 2: Use simplified visualization
        if 'config' in kwargs:
            kwargs['config'] = self._simplify_config(kwargs.get('config'))
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Recovery with simplified config failed: {e}")
        
        return None
    
    def _recover_from_incomplete_trace(self, func: Callable, 
                                      args: tuple, kwargs: dict) -> Optional[Any]:
        """Recover from incomplete trace by handling missing data"""
        logger.info("Attempting recovery from incomplete trace...")
        
        # Strategy: Mark missing sections and provide partial visualization
        if 'trace_nodes' in kwargs or (args and isinstance(args[0], list)):
            nodes = kwargs.get('trace_nodes', args[0] if args else [])
            
            # Validate and clean nodes
            cleaned_nodes = self._clean_trace_nodes(nodes)
            
            if 'trace_nodes' in kwargs:
                kwargs['trace_nodes'] = cleaned_nodes
            else:
                args = (cleaned_nodes,) + args[1:]
            
            # Add flag to indicate partial data
            kwargs['partial_data'] = True
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Recovery with cleaned nodes failed: {e}")
        
        return None
    
    def _recover_from_visualization_error(self, func: Callable, 
                                         args: tuple, kwargs: dict) -> Optional[Any]:
        """Recover from generic visualization error"""
        logger.info("Attempting recovery from visualization error...")
        
        # Try with fallback renderer
        if 'renderer' in kwargs:
            kwargs['renderer'] = 'fallback'
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Recovery with fallback renderer failed: {e}")
        
        return None
    
    def _default_recovery(self, func: Callable, error: Exception,
                         args: tuple, kwargs: dict) -> Optional[Any]:
        """Default recovery strategy"""
        logger.info(f"Attempting default recovery for {type(error).__name__}")
        
        # Add error flag and retry
        kwargs['error_recovery'] = True
        kwargs['simplified'] = True
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Default recovery failed: {e}")
            return None
    
    def _reduce_nodes(self, nodes: List[TraceNode], 
                     max_nodes: int = 100) -> List[TraceNode]:
        """Reduce nodes to manageable number based on importance"""
        if len(nodes) <= max_nodes:
            return nodes
        
        # Sort by importance (memory + compute)
        sorted_nodes = sorted(
            nodes,
            key=lambda n: (n.memory_usage + n.compute_time),
            reverse=True
        )
        
        # Keep top nodes and add summary node
        kept_nodes = sorted_nodes[:max_nodes - 1]
        
        # Create summary node for removed nodes
        removed_nodes = sorted_nodes[max_nodes - 1:]
        if removed_nodes:
            summary_node = TraceNode(
                operation="[Summary of {} nodes]".format(len(removed_nodes)),
                module_path="summary",
                input_shapes=[],
                output_shapes=[],
                memory_usage=sum(n.memory_usage for n in removed_nodes),
                compute_time=sum(n.compute_time for n in removed_nodes),
                temporal_index=None
            )
            kept_nodes.append(summary_node)
        
        return kept_nodes
    
    def _simplify_config(self, config: Any) -> Any:
        """Simplify configuration to reduce resource usage"""
        if config is None:
            return config
        
        # Disable resource-intensive features
        if hasattr(config, 'enable_animations'):
            config.enable_animations = False
        if hasattr(config, 'max_nodes_visible'):
            config.max_nodes_visible = min(config.max_nodes_visible, 50)
        if hasattr(config, 'enable_minimap'):
            config.enable_minimap = False
        if hasattr(config, 'animation_duration_ms'):
            config.animation_duration_ms = 0
        
        return config
    
    def _clean_trace_nodes(self, nodes: List[TraceNode]) -> List[TraceNode]:
        """Clean and validate trace nodes"""
        cleaned = []
        
        for node in nodes:
            if node is None:
                continue
            
            # Ensure required attributes exist
            if not hasattr(node, 'operation'):
                node.operation = 'unknown'
            if not hasattr(node, 'module_path'):
                node.module_path = 'unknown'
            if not hasattr(node, 'memory_usage'):
                node.memory_usage = 0.0
            if not hasattr(node, 'compute_time'):
                node.compute_time = 0.0
            
            # Validate shapes
            if hasattr(node, 'input_shapes') and node.input_shapes is None:
                node.input_shapes = []
            if hasattr(node, 'output_shapes') and node.output_shapes is None:
                node.output_shapes = []
            
            cleaned.append(node)
        
        return cleaned


def incomplete_trace_handler(trace_nodes: List[TraceNode], 
                            strict: bool = False) -> Dict[str, Any]:
    """Handle incomplete trace data by marking missing sections
    
    Args:
        trace_nodes: List of trace nodes (may be incomplete)
        strict: If True, raise error for missing critical data
        
    Returns:
        Dictionary with trace analysis and missing sections marked
    """
    result = {
        'complete': True,
        'missing_sections': [],
        'warnings': [],
        'node_count': len(trace_nodes),
        'validated_nodes': []
    }
    
    # Check for required task heads in UniAD
    expected_heads = ['track', 'seg', 'motion', 'occ', 'planning']
    found_heads = set()
    
    for node in trace_nodes:
        # Validate node
        if node is None:
            result['warnings'].append("Found None node in trace")
            result['complete'] = False
            continue
        
        # Check for task head
        if hasattr(node, 'task_head') and node.task_head:
            found_heads.add(node.task_head)
        
        # Check for required attributes
        missing_attrs = []
        for attr in ['operation', 'module_path', 'memory_usage', 'compute_time']:
            if not hasattr(node, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            result['warnings'].append(
                f"Node missing attributes: {', '.join(missing_attrs)}"
            )
            result['complete'] = False
            
            # Add default values
            for attr in missing_attrs:
                if attr == 'operation':
                    setattr(node, attr, 'unknown')
                elif attr == 'module_path':
                    setattr(node, attr, 'unknown')
                else:
                    setattr(node, attr, 0.0)
        
        result['validated_nodes'].append(node)
    
    # Check for missing task heads
    missing_heads = set(expected_heads) - found_heads
    if missing_heads:
        result['missing_sections'].extend([f"{h}_head" for h in missing_heads])
        result['complete'] = False
        
        if strict:
            raise IncompleteTraceError(
                f"Missing critical task heads: {', '.join(missing_heads)}"
            )
    
    # Check for temporal data
    has_temporal = any(
        hasattr(node, 'temporal_index') and node.temporal_index is not None
        for node in trace_nodes
    )
    
    if not has_temporal:
        result['warnings'].append("No temporal information found in trace")
        result['missing_sections'].append('temporal_data')
    
    # Check for BEV operations
    has_bev = any(
        hasattr(node, 'is_bev_operation') and node.is_bev_operation
        for node in trace_nodes
    )
    
    if not has_bev:
        result['warnings'].append("No BEV operations found in trace")
        result['missing_sections'].append('bev_operations')
    
    return result


def memory_overflow_handler(func: Callable, memory_limit_mb: float = 1000.0) -> Callable:
    """Handle memory overflow by implementing chunked processing
    
    Args:
        func: Function to wrap
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Wrapped function with memory overflow protection
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check current memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < memory_limit_mb:
                logger.warning(
                    f"Low memory: {available_mb:.1f}MB available, "
                    f"need {memory_limit_mb:.1f}MB"
                )
                
                # Try to free memory
                gc.collect()
                
                # Re-check
                memory = psutil.virtual_memory()
                available_mb = memory.available / (1024 * 1024)
                
                if available_mb < memory_limit_mb:
                    # Use simplified processing
                    kwargs['simplified'] = True
                    kwargs['chunk_processing'] = True
                    
                    # Reduce batch size if applicable
                    if 'batch_size' in kwargs:
                        kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                    
                    logger.info("Using simplified processing due to memory constraints")
        
        except (ImportError, ValueError):
            # psutil not available
            pass
        
        # Execute function
        try:
            return func(*args, **kwargs)
        except MemoryError as e:
            logger.error(f"Memory error in {func.__name__}: {e}")
            
            # Try with minimal configuration
            if 'config' in kwargs:
                kwargs['config'] = create_minimal_config()
            
            # Retry with reduced data
            if 'trace_nodes' in kwargs:
                nodes = kwargs['trace_nodes']
                if len(nodes) > 10:
                    kwargs['trace_nodes'] = nodes[:10]
                    logger.info("Reduced to 10 nodes due to memory error")
            
            try:
                return func(*args, **kwargs)
            except Exception as e2:
                logger.error(f"Recovery failed: {e2}")
                
                # Return minimal fallback
                return create_fallback_result(func.__name__)
    
    return wrapper


def create_minimal_config() -> Dict[str, Any]:
    """Create minimal configuration for memory-constrained environments"""
    return {
        'max_nodes_visible': 10,
        'enable_animations': False,
        'enable_minimap': False,
        'enable_search': False,
        'enable_filter': False,
        'enable_tooltips': False,
        'enable_highlighting': False,
        'animation_duration_ms': 0,
        'layout_algorithm': 'hierarchical'
    }


def create_fallback_result(function_name: str) -> Dict[str, Any]:
    """Create fallback result when all recovery attempts fail"""
    return {
        'error': True,
        'function': function_name,
        'message': 'Visualization failed due to resource constraints',
        'fallback': True,
        'data': None,
        'recommendations': [
            'Close other applications to free memory',
            'Use a smaller subset of data',
            'Disable animations and extra features',
            'Use static visualization mode'
        ]
    }


# Global error handler instance
error_handler = ErrorHandler()

# Convenience decorators
visualization_error_handler = error_handler.visualization_error_handler


def safe_visualization(fallback=None):
    """Convenience decorator for safe visualization functions"""
    return error_handler.visualization_error_handler(
        fallback_result=fallback,
        error_types=(Exception,)
    )