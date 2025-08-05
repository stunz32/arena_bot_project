"""
Method Tracing Decorator System for Arena Bot Deep Debugging

Provides comprehensive instrumentation for method calls including:
- Entry/exit logging with full parameter capture
- Execution timing with microsecond precision  
- Memory usage tracking and leak detection
- Exception context capture with stack traces
- Performance bottleneck identification
- Correlation ID propagation for distributed tracing

Integrates seamlessly with existing S-tier logging infrastructure.
"""

import time
import traceback
import threading
import psutil
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from ..logging_system.logger import get_logger, LogLevel


@dataclass
class MethodExecutionContext:
    """
    Comprehensive context captured during method execution.
    
    Provides detailed information about method calls for debugging
    and performance analysis.
    """
    
    # Method identification
    method_name: str
    module_name: str
    class_name: Optional[str] = None
    
    # Execution tracking
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Context capture
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    thread_name: str = field(default_factory=lambda: threading.current_thread().name)
    
    # Memory tracking
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    
    # Parameters and results
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    output_result: Any = None
    
    # Exception information
    exception_occurred: bool = False
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    
    # Performance metrics
    cpu_percent_before: Optional[float] = None
    cpu_percent_after: Optional[float] = None
    
    def finish_execution(self, result: Any = None, exception: Exception = None) -> None:
        """Mark execution as finished and capture final metrics."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        # Capture memory after execution
        try:
            process = psutil.Process()
            self.memory_after_mb = process.memory_info().rss / 1024 / 1024
            if self.memory_before_mb:
                self.memory_delta_mb = self.memory_after_mb - self.memory_before_mb
        except:
            pass
        
        # Capture CPU after execution  
        try:
            self.cpu_percent_after = psutil.cpu_percent()
        except:
            pass
        
        # Handle result or exception
        if exception:
            self.exception_occurred = True
            self.exception_type = type(exception).__name__
            self.exception_message = str(exception)
            self.exception_traceback = traceback.format_exc()
        else:
            self.output_result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'method_name': self.method_name,
            'module_name': self.module_name,
            'class_name': self.class_name,
            'correlation_id': self.correlation_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'thread_id': self.thread_id,
            'thread_name': self.thread_name,
            'memory_before_mb': self.memory_before_mb,
            'memory_after_mb': self.memory_after_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'input_parameters': self._safe_serialize(self.input_parameters),
            'output_result': self._safe_serialize(self.output_result),
            'exception_occurred': self.exception_occurred,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'cpu_percent_before': self.cpu_percent_before,
            'cpu_percent_after': self.cpu_percent_after
        }
    
    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize objects for logging."""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [self._safe_serialize(item) for item in obj[:10]]  # Limit to first 10 items
            elif isinstance(obj, dict):
                return {k: self._safe_serialize(v) for k, v in list(obj.items())[:10]}  # Limit to first 10 items
            else:
                return f"<{type(obj).__name__}>"
        except:
            return "<serialization_error>"


class MethodTracer:
    """
    Central method tracer for Arena Bot debugging.
    
    Manages method execution contexts, performance tracking,
    and integration with S-tier logging system.
    """
    
    def __init__(self):
        """Initialize method tracer."""
        self.logger = get_logger("arena_bot.debugging.method_tracer")
        self.active_contexts: Dict[str, MethodExecutionContext] = {}
        self.context_lock = threading.RLock()
        
        # Performance tracking
        self.total_traced_calls = 0
        self.total_execution_time_ms = 0.0
        self.failed_calls = 0
        
        # Configuration
        self.capture_parameters = True
        self.capture_memory = True
        self.capture_cpu = True
        self.max_parameter_size = 1000  # Max chars for parameter logging
        
    def start_trace(self, 
                   method_name: str,
                   module_name: str,
                   class_name: Optional[str] = None,
                   parameters: Optional[Dict[str, Any]] = None) -> MethodExecutionContext:
        """Start tracing a method execution."""
        
        # Create execution context
        context = MethodExecutionContext(
            method_name=method_name,
            module_name=module_name,
            class_name=class_name
        )
        
        # Capture memory before execution
        if self.capture_memory:
            try:
                process = psutil.Process()
                context.memory_before_mb = process.memory_info().rss / 1024 / 1024
            except:
                pass
        
        # Capture CPU before execution
        if self.capture_cpu:
            try:
                context.cpu_percent_before = psutil.cpu_percent()
            except:
                pass
        
        # Capture parameters
        if self.capture_parameters and parameters:
            context.input_parameters = self._filter_parameters(parameters)
        
        # Store active context
        with self.context_lock:
            self.active_contexts[context.correlation_id] = context
        
        # Log method entry
        self.logger.trace(
            f"🔍 METHOD_ENTRY: {self._format_method_name(context)}",
            extra={
                'trace_type': 'method_entry',
                'correlation_id': context.correlation_id,
                'method_name': method_name,
                'module_name': module_name,
                'class_name': class_name,
                'thread_id': context.thread_id,
                'thread_name': context.thread_name,
                'memory_before_mb': context.memory_before_mb,
                'parameters': context.input_parameters
            }
        )
        
        return context
    
    def finish_trace(self, 
                    context: MethodExecutionContext,
                    result: Any = None,
                    exception: Exception = None) -> None:
        """Finish tracing a method execution."""
        
        # Finish execution context
        context.finish_execution(result, exception)
        
        # Update statistics
        self.total_traced_calls += 1
        if context.duration_ms:
            self.total_execution_time_ms += context.duration_ms
        if exception:
            self.failed_calls += 1
        
        # Remove from active contexts
        with self.context_lock:
            self.active_contexts.pop(context.correlation_id, None)
        
        # Log method exit
        log_level = LogLevel.ERROR if exception else LogLevel.TRACE
        
        self.logger.log(
            log_level,
            f"🔍 METHOD_EXIT: {self._format_method_name(context)} "
            f"({'FAILED' if exception else 'SUCCESS'}) - {context.duration_ms:.2f}ms",
            extra={
                'trace_type': 'method_exit',
                'correlation_id': context.correlation_id,
                'method_name': context.method_name,
                'module_name': context.module_name,
                'class_name': context.class_name,
                'duration_ms': context.duration_ms,
                'memory_delta_mb': context.memory_delta_mb,
                'exception_occurred': context.exception_occurred,
                'exception_type': context.exception_type,
                'exception_message': context.exception_message,
                'result_type': type(result).__name__ if result is not None else None
            }
        )
        
        # Log performance warning if method is slow
        if context.duration_ms and context.duration_ms > 1000:  # > 1 second
            self.logger.warning(
                f"⚠️ SLOW_METHOD: {self._format_method_name(context)} took {context.duration_ms:.2f}ms",
                extra={
                    'trace_type': 'performance_warning',
                    'correlation_id': context.correlation_id,
                    'duration_ms': context.duration_ms,
                    'memory_delta_mb': context.memory_delta_mb
                }
            )
        
        # Log memory warning if significant memory increase
        if context.memory_delta_mb and context.memory_delta_mb > 50:  # > 50MB increase
            self.logger.warning(
                f"⚠️ MEMORY_INCREASE: {self._format_method_name(context)} increased memory by {context.memory_delta_mb:.2f}MB",
                extra={
                    'trace_type': 'memory_warning',
                    'correlation_id': context.correlation_id,
                    'memory_delta_mb': context.memory_delta_mb,
                    'duration_ms': context.duration_ms
                }
            )
    
    def _format_method_name(self, context: MethodExecutionContext) -> str:
        """Format method name for logging."""
        if context.class_name:
            return f"{context.class_name}.{context.method_name}"
        else:
            return f"{context.module_name}.{context.method_name}"
    
    def _filter_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters for safe logging."""
        filtered = {}
        
        for key, value in parameters.items():
            # Skip sensitive parameters
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                filtered[key] = '<REDACTED>'
                continue
            
            # Limit parameter size
            try:
                str_value = str(value)
                if len(str_value) > self.max_parameter_size:
                    filtered[key] = f"<truncated:{type(value).__name__}:{len(str_value)}chars>"
                else:
                    filtered[key] = value
            except:
                filtered[key] = f"<error_serializing:{type(value).__name__}>"
        
        return filtered
    
    def get_active_traces(self) -> List[MethodExecutionContext]:
        """Get currently active method traces."""
        with self.context_lock:
            return list(self.active_contexts.values())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get method tracing performance statistics."""
        avg_execution_time = (
            self.total_execution_time_ms / self.total_traced_calls
            if self.total_traced_calls > 0 else 0
        )
        
        failure_rate = (
            self.failed_calls / self.total_traced_calls
            if self.total_traced_calls > 0 else 0
        )
        
        return {
            'total_traced_calls': self.total_traced_calls,
            'total_execution_time_ms': self.total_execution_time_ms,
            'average_execution_time_ms': avg_execution_time,
            'failed_calls': self.failed_calls,
            'failure_rate': failure_rate,
            'active_traces': len(self.active_contexts)
        }


# Global method tracer instance
_global_tracer: Optional[MethodTracer] = None
_tracer_lock = threading.Lock()


def get_method_tracer() -> MethodTracer:
    """Get global method tracer instance."""
    global _global_tracer
    
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = MethodTracer()
    
    return _global_tracer


def trace_method(capture_args: bool = True,
                capture_timing: bool = True,
                capture_memory: bool = True,
                capture_exceptions: bool = True,
                enabled: bool = True) -> Callable:
    """
    Decorator for comprehensive method tracing.
    
    Args:
        capture_args: Whether to capture method arguments
        capture_timing: Whether to capture execution timing
        capture_memory: Whether to capture memory usage
        capture_exceptions: Whether to capture exception details
        enabled: Whether tracing is enabled (allows runtime toggle)
    
    Returns:
        Decorated method with tracing instrumentation
    
    Usage:
        @trace_method(capture_args=True, capture_timing=True)
        def my_method(self, param1, param2):
            # Method implementation
            return result
    """
    
    def decorator(func: Callable) -> Callable:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            
            tracer = get_method_tracer()
            
            # Get method information
            method_name = func.__name__
            module_name = func.__module__
            
            # Determine class name if this is a method
            class_name = None
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
            
            # Capture parameters
            parameters = {}
            if capture_args:
                try:
                    # Get function signature
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Convert to dict, excluding 'self'
                    parameters = dict(bound_args.arguments)
                    if 'self' in parameters:
                        parameters.pop('self')
                        
                except Exception as e:
                    parameters = {'_parameter_capture_error': str(e)}
            
            # Start tracing
            context = tracer.start_trace(
                method_name=method_name,
                module_name=module_name,
                class_name=class_name,
                parameters=parameters
            )
            
            try:
                # Execute method
                result = func(*args, **kwargs)
                
                # Finish tracing successfully
                tracer.finish_trace(context, result=result)
                
                return result
                
            except Exception as e:
                # Finish tracing with exception
                if capture_exceptions:
                    tracer.finish_trace(context, exception=e)
                else:
                    tracer.finish_trace(context)
                
                # Re-raise exception
                raise
        
        # Add tracing metadata to wrapper
        wrapper.__traced__ = True
        wrapper.__trace_config__ = {
            'capture_args': capture_args,
            'capture_timing': capture_timing,
            'capture_memory': capture_memory,
            'capture_exceptions': capture_exceptions,
            'enabled': enabled
        }
        
        return wrapper
    
    return decorator


def disable_tracing(func: Callable) -> Callable:
    """
    Decorator to disable tracing for a specific method.
    
    Useful for utility methods or methods called frequently
    that don't need tracing.
    """
    if hasattr(func, '__trace_config__'):
        func.__trace_config__['enabled'] = False
    return func


def enable_tracing(func: Callable) -> Callable:
    """
    Decorator to explicitly enable tracing for a method.
    
    Useful for re-enabling tracing that was previously disabled.
    """
    if hasattr(func, '__trace_config__'):
        func.__trace_config__['enabled'] = True
    return func