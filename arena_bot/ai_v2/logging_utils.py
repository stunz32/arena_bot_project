"""
AI v2 Standardized Logging Utilities

Provides consistent, structured logging across all AI v2 components
with standardized error formats and debugging capabilities.
"""

import logging
import json
import traceback
import time
import uuid
import threading
import inspect
import functools
from datetime import datetime
from typing import Any, Dict, Optional, Union, Callable
from enum import Enum
from contextlib import contextmanager
from queue import Queue


class LogLevel(Enum):
    """Standardized log levels for AI v2 system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categorize log messages for better filtering and analysis."""
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    API_CALL = "api_call"
    DATA_PROCESSING = "data_processing"
    FALLBACK = "fallback"
    PERFORMANCE = "performance"
    CIRCUIT_BREAKER = "circuit_breaker"
    CACHE = "cache"
    RECOVERY = "recovery"
    USER_INPUT = "user_input"
    SYSTEM_HEALTH = "system_health"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    DEBUG = "debug"


class StructuredLogger:
    """
    Structured logger for AI v2 system with consistent formatting.
    
    Provides standardized logging methods with structured data,
    performance tracking, and error context preservation.
    """
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        """Initialize structured logger for a component."""
        self.component_name = component_name
        self.logger = logger or logging.getLogger(f"ai_v2.{component_name}")
        self.start_time = time.time()
        
        # Performance tracking
        self.operation_timings = {}
        self.error_counts = {}
        self.warning_counts = {}
    
    def _format_structured_message(
        self, 
        message: str, 
        category: LogCategory, 
        level: LogLevel,
        **kwargs
    ) -> str:
        """Format message with structured data."""
        structured_data = {
            'timestamp': datetime.now().isoformat(),
            'component': self.component_name,
            'category': category.value,
            'level': level.value,
            'message': message,
            'session_uptime_seconds': round(time.time() - self.start_time, 2)
        }
        
        # Add request ID if available
        request_id = get_current_request_id()
        if request_id:
            structured_data['request_id'] = request_id
        
        # Add operation name if available
        operation_name = get_current_operation_name()
        if operation_name:
            structured_data['operation_name'] = operation_name
        
        # Add any additional context
        if kwargs:
            structured_data['context'] = kwargs
        
        # Create human-readable prefix
        emoji_map = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🚨"
        }
        
        emoji = emoji_map.get(level, "")
        request_prefix = f"[{request_id[:8]}]" if request_id else ""
        human_readable = f"{emoji} {request_prefix} [{self.component_name}] {message}"
        
        # Add structured data as JSON on separate line for processing
        structured_json = json.dumps(structured_data, separators=(',', ':'))
        
        return f"{human_readable}\n[STRUCTURED_LOG] {structured_json}"
    
    def debug(
        self, 
        message: str, 
        category: LogCategory = LogCategory.DATA_PROCESSING,
        **kwargs
    ) -> None:
        """Log debug message with structured data."""
        formatted_message = self._format_structured_message(message, category, LogLevel.DEBUG, **kwargs)
        self.logger.debug(formatted_message)
    
    def info(
        self, 
        message: str, 
        category: LogCategory = LogCategory.DATA_PROCESSING,
        **kwargs
    ) -> None:
        """Log info message with structured data."""
        formatted_message = self._format_structured_message(message, category, LogLevel.INFO, **kwargs)
        self.logger.info(formatted_message)
    
    def warning(
        self, 
        message: str, 
        category: LogCategory = LogCategory.DATA_PROCESSING,
        **kwargs
    ) -> None:
        """Log warning message with structured data."""
        self.warning_counts[category.value] = self.warning_counts.get(category.value, 0) + 1
        formatted_message = self._format_structured_message(message, category, LogLevel.WARNING, **kwargs)
        self.logger.warning(formatted_message)
    
    def error(
        self, 
        message: str, 
        category: LogCategory = LogCategory.DATA_PROCESSING,
        exception: Optional[Exception] = None,
        capture_locals: bool = False,
        **kwargs
    ) -> None:
        """Log error message with structured data and exception details."""
        self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
        
        # Add exception information if provided
        if exception:
            kwargs.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc() if self.logger.isEnabledFor(logging.DEBUG) else None
            })
            
            # Capture local variables at point of failure if requested
            if capture_locals and self.logger.isEnabledFor(logging.DEBUG):
                try:
                    frame = inspect.currentframe().f_back
                    if frame:
                        local_vars = {k: str(v)[:200] for k, v in frame.f_locals.items() 
                                    if not k.startswith('_') and not callable(v)}
                        kwargs['local_variables'] = local_vars
                except Exception:
                    pass  # Don't let variable capture fail the logging
        
        formatted_message = self._format_structured_message(message, category, LogLevel.ERROR, **kwargs)
        self.logger.error(formatted_message)
    
    def critical(
        self, 
        message: str, 
        category: LogCategory = LogCategory.SYSTEM_HEALTH,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> None:
        """Log critical message with structured data."""
        self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
        
        if exception:
            kwargs.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            })
        
        formatted_message = self._format_structured_message(message, category, LogLevel.CRITICAL, **kwargs)
        self.logger.critical(formatted_message)
    
    def log_operation_start(self, operation_name: str, **kwargs) -> str:
        """Log start of operation and return operation ID for tracking."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.operation_timings[operation_id] = time.time()
        
        self.info(
            f"Starting operation: {operation_name}",
            category=LogCategory.PERFORMANCE,
            operation_id=operation_id,
            **kwargs
        )
        
        return operation_id
    
    def log_operation_end(
        self, 
        operation_id: str, 
        success: bool = True, 
        result_summary: Optional[str] = None,
        **kwargs
    ) -> float:
        """Log end of operation and return duration."""
        if operation_id not in self.operation_timings:
            self.warning(
                f"Operation ID {operation_id} not found in timings",
                category=LogCategory.PERFORMANCE
            )
            return 0.0
        
        duration = time.time() - self.operation_timings[operation_id]
        del self.operation_timings[operation_id]
        
        status = "completed successfully" if success else "failed"
        message = f"Operation {operation_id} {status}"
        if result_summary:
            message += f": {result_summary}"
        
        log_method = self.info if success else self.error
        log_method(
            message,
            category=LogCategory.PERFORMANCE,
            operation_id=operation_id,
            duration_seconds=round(duration, 3),
            success=success,
            **kwargs
        )
        
        return duration
    
    def log_api_call(
        self, 
        api_name: str, 
        method: str = "GET",
        response_time_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log API call with standardized format."""
        message = f"API call: {method} {api_name}"
        
        call_data = {
            'api_name': api_name,
            'method': method,
            'response_time_ms': response_time_ms,
            'status_code': status_code
        }
        
        if error:
            call_data['error'] = error
            self.error(
                f"{message} failed: {error}",
                category=LogCategory.API_CALL,
                **call_data,
                **kwargs
            )
        else:
            status_msg = f" ({status_code})" if status_code else ""
            timing_msg = f" in {response_time_ms}ms" if response_time_ms else ""
            self.info(
                f"{message}{status_msg}{timing_msg}",
                category=LogCategory.API_CALL,
                **call_data,
                **kwargs
            )
    
    def log_validation_result(
        self, 
        field_name: str, 
        success: bool,
        error_message: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ) -> None:
        """Log validation result with standardized format."""
        if success:
            self.debug(
                f"Validation passed: {field_name}",
                category=LogCategory.VALIDATION,
                field_name=field_name,
                **kwargs
            )
        else:
            self.error(
                f"Validation failed: {field_name} - {error_message}",
                category=LogCategory.VALIDATION,
                field_name=field_name,
                error_message=error_message,
                invalid_value=str(value) if value is not None else None,
                **kwargs
            )
    
    def log_fallback_activation(
        self, 
        component: str, 
        reason: str,
        fallback_type: str = "default",
        **kwargs
    ) -> None:
        """Log fallback activation with standardized format."""
        self.warning(
            f"Fallback activated for {component}: {reason}",
            category=LogCategory.FALLBACK,
            component=component,
            reason=reason,
            fallback_type=fallback_type,
            **kwargs
        )
    
    def log_cache_operation(
        self, 
        operation: str, 
        cache_key: str,
        hit: bool = False,
        cache_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log cache operation with standardized format."""
        status = "hit" if hit else "miss"
        message = f"Cache {operation}: {cache_key} ({status})"
        
        self.debug(
            message,
            category=LogCategory.CACHE,
            operation=operation,
            cache_key=cache_key,
            cache_hit=hit,
            cache_size=cache_size,
            **kwargs
        )
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get logging statistics for this component."""
        return {
            'component': self.component_name,
            'uptime_seconds': round(time.time() - self.start_time, 2),
            'error_counts': self.error_counts.copy(),
            'warning_counts': self.warning_counts.copy(),
            'active_operations': len(self.operation_timings)
        }


# Thread-local storage for request context
_request_context = threading.local()

class RequestContext:
    """Context manager for request ID tracking across operations."""
    
    def __init__(self, request_id: Optional[str] = None, operation_name: str = "operation"):
        self.request_id = request_id or str(uuid.uuid4())
        self.operation_name = operation_name
        self.parent_context = None
    
    def __enter__(self):
        # Store parent context if it exists
        self.parent_context = getattr(_request_context, 'current', None)
        _request_context.current = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore parent context
        _request_context.current = self.parent_context

def get_current_request_id() -> Optional[str]:
    """Get the current request ID from thread-local storage."""
    context = getattr(_request_context, 'current', None)
    return context.request_id if context else None

def get_current_operation_name() -> Optional[str]:
    """Get the current operation name from thread-local storage."""
    context = getattr(_request_context, 'current', None)
    return context.operation_name if context else None

class GUILogHandler(logging.Handler):
    """Thread-safe logging handler that sends logs to GUI in real-time."""
    
    def __init__(self, gui_queue: Queue, level=logging.NOTSET):
        super().__init__(level)
        self.gui_queue = gui_queue
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def emit(self, record):
        try:
            # Format the record for GUI display
            formatted_msg = self.format(record)
            
            # Extract structured data if present
            structured_data = None
            if '[STRUCTURED_LOG]' in formatted_msg:
                parts = formatted_msg.split('[STRUCTURED_LOG]')
                if len(parts) == 2:
                    human_readable = parts[0].strip()
                    try:
                        structured_data = json.loads(parts[1].strip())
                    except json.JSONDecodeError:
                        pass
                    formatted_msg = human_readable
            
            # Send to GUI queue (non-blocking)
            log_entry = {
                'timestamp': time.time(),
                'level': record.levelname,
                'message': formatted_msg,
                'structured_data': structured_data
            }
            
            # Use non-blocking put to avoid GUI freezing
            try:
                self.gui_queue.put_nowait(log_entry)
            except:
                pass  # Silently drop if queue is full
                
        except Exception:
            # Never let logging handler exceptions crash the application
            pass

# Factory function for creating component loggers
def get_structured_logger(component_name: str, gui_queue: Optional[Queue] = None) -> StructuredLogger:
    """Create a structured logger for an AI v2 component."""
    logger = StructuredLogger(component_name)
    
    # Add GUI handler if queue is provided
    if gui_queue:
        gui_handler = GUILogHandler(gui_queue)
        logger.logger.addHandler(gui_handler)
    
    return logger


# Common logging patterns as convenience functions
def log_component_initialization(logger: StructuredLogger, component_name: str, success: bool, **kwargs) -> None:
    """Log component initialization with standardized format."""
    if success:
        logger.info(
            f"{component_name} initialized successfully",
            category=LogCategory.INITIALIZATION,
            **kwargs
        )
    else:
        logger.error(
            f"{component_name} initialization failed",
            category=LogCategory.INITIALIZATION,
            **kwargs
        )


def log_performance_warning(logger: StructuredLogger, operation: str, duration_ms: float, threshold_ms: float = 1000, **kwargs) -> None:
    """Log performance warning if operation exceeds threshold."""
    if duration_ms > threshold_ms:
        logger.warning(
            f"Performance warning: {operation} took {duration_ms}ms (threshold: {threshold_ms}ms)",
            category=LogCategory.PERFORMANCE,
            operation=operation,
            duration_ms=duration_ms,
            threshold_ms=threshold_ms,
            **kwargs
        )


def log_circuit_breaker_event(logger: StructuredLogger, component: str, event: str, state: str, **kwargs) -> None:
    """Log circuit breaker events with standardized format."""
    logger.warning(
        f"Circuit breaker {event} for {component}: {state}",
        category=LogCategory.CIRCUIT_BREAKER,
        component=component,
        event=event,
        circuit_state=state,
        **kwargs
    )


# Performance Instrumentation Decorators and Context Managers

def log_performance(
    logger: Optional[StructuredLogger] = None,
    component_name: str = "unknown",
    category: LogCategory = LogCategory.PERFORMANCE,
    threshold_ms: float = 1000.0
):
    """
    Decorator to automatically log function performance with request ID tracking.
    
    Args:
        logger: StructuredLogger instance (will create one if None)
        component_name: Component name for logger if creating new one
        category: Log category for performance messages
        threshold_ms: Log warning if execution exceeds this threshold
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            actual_logger = logger or get_structured_logger(component_name)
            
            # Start timing
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function start
            actual_logger.debug(
                f"Function started: {function_name}",
                category=category,
                function_name=function_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log completion
                if duration_ms > threshold_ms:
                    actual_logger.warning(
                        f"Function exceeded threshold: {function_name} took {duration_ms:.2f}ms",
                        category=category,
                        function_name=function_name,
                        duration_ms=duration_ms,
                        threshold_ms=threshold_ms
                    )
                else:
                    actual_logger.debug(
                        f"Function completed: {function_name} in {duration_ms:.2f}ms",
                        category=category,
                        function_name=function_name,
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                # Calculate duration for failed execution
                duration_ms = (time.time() - start_time) * 1000
                
                # Log error with performance context
                actual_logger.error(
                    f"Function failed: {function_name} after {duration_ms:.2f}ms",
                    category=LogCategory.ERROR,
                    function_name=function_name,
                    duration_ms=duration_ms,
                    exception=e,
                    capture_locals=True
                )
                raise
                
        return wrapper
    return decorator

@contextmanager
def time_operation(
    operation_name: str,
    logger: Optional[StructuredLogger] = None,
    component_name: str = "unknown",
    category: LogCategory = LogCategory.PERFORMANCE,
    threshold_ms: float = 1000.0
):
    """
    Context manager for timing operations with structured logging.
    
    Usage:
        with time_operation("database_query", logger):
            # Your operation here
            result = expensive_operation()
    """
    # Get or create logger
    actual_logger = logger or get_structured_logger(component_name)
    
    # Start timing
    start_time = time.time()
    
    # Log operation start
    actual_logger.info(
        f"Operation started: {operation_name}",
        category=category,
        operation_name=operation_name
    )
    
    try:
        yield actual_logger
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log completion
        if duration_ms > threshold_ms:
            actual_logger.warning(
                f"Operation exceeded threshold: {operation_name} took {duration_ms:.2f}ms",
                category=category,
                operation_name=operation_name,
                duration_ms=duration_ms,
                threshold_ms=threshold_ms
            )
        else:
            actual_logger.info(
                f"Operation completed: {operation_name} in {duration_ms:.2f}ms",
                category=category,
                operation_name=operation_name,
                duration_ms=duration_ms
            )
            
    except Exception as e:
        # Calculate duration for failed operation
        duration_ms = (time.time() - start_time) * 1000
        
        # Log error with performance context
        actual_logger.error(
            f"Operation failed: {operation_name} after {duration_ms:.2f}ms",
            category=LogCategory.ERROR,
            operation_name=operation_name,
            duration_ms=duration_ms,
            exception=e,
            capture_locals=True
        )
        raise


# State Change Logging Utilities

def log_state_change(
    logger: StructuredLogger,
    component: str,
    from_state: str,
    to_state: str,
    trigger: str = "unknown",
    **kwargs
) -> None:
    """Log application state changes with standardized format."""
    logger.info(
        f"State change in {component}: {from_state} → {to_state} (trigger: {trigger})",
        category=LogCategory.STATE_CHANGE,
        component=component,
        from_state=from_state,
        to_state=to_state,
        trigger=trigger,
        **kwargs
    )


# Enhanced Error Logging for Complex Failures

def log_complex_error(
    logger: StructuredLogger,
    error_context: str,
    exception: Exception,
    operation_data: Optional[Dict[str, Any]] = None,
    capture_locals: bool = True
) -> None:
    """
    Log complex errors with full context for debugging multi-threaded failures.
    
    Args:
        logger: StructuredLogger instance
        error_context: Description of what was being attempted
        exception: The exception that occurred
        operation_data: Additional data about the operation
        capture_locals: Whether to capture local variables
    """
    error_data = {
        'error_context': error_context,
        'thread_name': threading.current_thread().name,
        'thread_id': threading.get_ident()
    }
    
    if operation_data:
        error_data['operation_data'] = operation_data
    
    logger.error(
        f"Complex error in {error_context}: {str(exception)}",
        category=LogCategory.ERROR,
        exception=exception,
        capture_locals=capture_locals,
        **error_data
    )