"""
Context Enricher for S-Tier Logging System.

This module provides performance-optimized context injection, automatically
enriching log messages with correlation IDs, performance metrics, thread
information, and system context.

Components:
- ContextEnricher: Main context injection engine
- OperationContext: Context manager for operation tracking
- operation_context: Context manager decorator
- timed_operation: Performance timing decorator

Features:
- Automatic correlation ID generation and propagation
- Performance metrics injection (CPU, memory, timing)
- Thread and process information
- Integration with existing monitoring.py
- Operation tracking with hierarchical context
- Minimal performance overhead (<150μs target)
"""

import os
import sys
import time
import threading
import logging
import uuid
import psutil
import traceback
import functools
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
from threading import local
from weakref import WeakSet

# Try to import existing monitoring system
try:
    from ...ai_v2.monitoring import get_performance_monitor, ResourceTracker, get_resource_manager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    # Create fallback classes
    class ResourceTracker:
        def __init__(self, name): pass
        def track_operation(self, name): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def get_performance_monitor(): return None
    def get_resource_manager(): return None


# Thread-local storage for context
_thread_local = local()


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    cpu_start_percent: Optional[float] = None
    cpu_end_percent: Optional[float] = None
    cpu_delta_percent: Optional[float] = None
    error: Optional[str] = None
    custom_metrics: Dict[str, Union[int, float, str]] = field(default_factory=dict)
    
    def finalize(self, end_time: Optional[float] = None, error: Optional[str] = None) -> None:
        """Finalize operation metrics."""
        self.end_time = end_time or time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'operation_id': self.operation_id,
            'name': self.name,
            'duration_ms': self.duration_ms,
            'memory_delta_mb': self.memory_delta_mb,
            'cpu_delta_percent': self.cpu_delta_percent,
            'error': self.error,
            'custom_metrics': self.custom_metrics
        }


class OperationContext:
    """
    Context manager for operation tracking.
    
    Provides hierarchical operation tracking with automatic metrics collection,
    correlation ID propagation, and integration with the logging system.
    """
    
    def __init__(self, 
                 operation_name: str,
                 correlation_id: Optional[str] = None,
                 parent_context: Optional['OperationContext'] = None,
                 track_performance: bool = True,
                 custom_data: Optional[Dict[str, Any]] = None):
        """
        Initialize operation context.
        
        Args:
            operation_name: Name of the operation
            correlation_id: Correlation ID (auto-generated if None)
            parent_context: Parent operation context
            track_performance: Whether to track performance metrics
            custom_data: Additional custom data
        """
        self.operation_name = operation_name
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.parent_context = parent_context
        self.track_performance = track_performance
        self.custom_data = custom_data or {}
        
        # Operation tracking
        self.operation_id = f"{operation_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.is_active = False
        
        # Child operations
        self.child_operations: List['OperationContext'] = []
        
        # Performance tracking
        self._metrics: Optional[OperationMetrics] = None
        self._resource_tracker: Optional[ResourceTracker] = None
        
        # Progress tracking
        self._progress_updates: List[Dict[str, Any]] = []
        self._logged_metrics: Dict[str, Union[int, float, str]] = {}
        
        # Logger
        self._logger = logging.getLogger(f"{__name__}.OperationContext")
        
    def _generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return f"op_{uuid.uuid4().hex[:16]}"
    
    def __enter__(self) -> 'OperationContext':
        """Enter operation context."""
        self.is_active = True
        
        # Set up thread-local context
        if not hasattr(_thread_local, 'context_stack'):
            _thread_local.context_stack = []
        
        _thread_local.context_stack.append(self)
        
        # Initialize performance tracking
        if self.track_performance:
            self._initialize_performance_tracking()
        
        # Log operation start
        self._logger.info(f"Operation started: {self.operation_name}",
                         extra={
                             'correlation_id': self.correlation_id,
                             'operation_id': self.operation_id,
                             'parent_operation': self.parent_context.operation_id if self.parent_context else None
                         })
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit operation context."""
        self.end_time = time.time()
        self.is_active = False
        error_info = None
        
        # Handle exceptions
        if exc_type is not None:
            error_info = f"{exc_type.__name__}: {exc_val}"
            self._logger.error(f"Operation failed: {self.operation_name}",
                              exc_info=(exc_type, exc_val, exc_tb),
                              extra={
                                  'correlation_id': self.correlation_id,
                                  'operation_id': self.operation_id,
                                  'error_type': exc_type.__name__,
                                  'error_message': str(exc_val)
                              })
        
        # Finalize performance tracking
        if self.track_performance:
            self._finalize_performance_tracking(error_info)
        
        # Remove from thread-local context
        if hasattr(_thread_local, 'context_stack') and _thread_local.context_stack:
            if _thread_local.context_stack[-1] is self:
                _thread_local.context_stack.pop()
        
        # Log operation completion
        duration_ms = (self.end_time - self.start_time) * 1000
        
        if error_info is None:
            self._logger.info(f"Operation completed: {self.operation_name}",
                             extra={
                                 'correlation_id': self.correlation_id,
                                 'operation_id': self.operation_id,
                                 'duration_ms': duration_ms,
                                 'child_operations': len(self.child_operations)
                             })
        
        # Add to parent's child operations
        if self.parent_context:
            self.parent_context.child_operations.append(self)
    
    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking."""
        try:
            # Create operation metrics
            self._metrics = OperationMetrics(
                operation_id=self.operation_id,
                name=self.operation_name,
                start_time=self.start_time
            )
            
            # Get baseline performance metrics
            if psutil:
                process = psutil.Process()
                self._metrics.memory_start_mb = process.memory_info().rss / 1024 / 1024
                self._metrics.cpu_start_percent = process.cpu_percent()
            
            # Initialize resource tracker if available
            if MONITORING_AVAILABLE:
                self._resource_tracker = ResourceTracker(self.operation_name)
                self._resource_tracker.start_operation(self.operation_name)
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize performance tracking: {e}")
    
    def _finalize_performance_tracking(self, error_info: Optional[str] = None) -> None:
        """Finalize performance tracking."""
        try:
            if self._metrics:
                # Get final performance metrics
                if psutil:
                    process = psutil.Process()
                    self._metrics.memory_end_mb = process.memory_info().rss / 1024 / 1024
                    self._metrics.cpu_end_percent = process.cpu_percent()
                    
                    # Calculate deltas
                    if self._metrics.memory_start_mb is not None:
                        self._metrics.memory_delta_mb = (
                            self._metrics.memory_end_mb - self._metrics.memory_start_mb
                        )
                    
                    if self._metrics.cpu_start_percent is not None:
                        self._metrics.cpu_delta_percent = (
                            self._metrics.cpu_end_percent - self._metrics.cpu_start_percent
                        )
                
                # Finalize metrics
                self._metrics.finalize(self.end_time, error_info)
                
                # Add custom metrics
                self._metrics.custom_metrics.update(self._logged_metrics)
            
            # Finalize resource tracker
            if self._resource_tracker and MONITORING_AVAILABLE:
                self._resource_tracker.end_operation(self.operation_name)
            
        except Exception as e:
            self._logger.warning(f"Failed to finalize performance tracking: {e}")
    
    def set_progress(self, stage: str, progress: float = 0.0, details: Optional[str] = None) -> None:
        """
        Update operation progress.
        
        Args:
            stage: Current stage name
            progress: Progress value (0.0 to 1.0)
            details: Optional details about current stage
        """
        progress_update = {
            'timestamp': time.time(),
            'stage': stage,
            'progress': max(0.0, min(1.0, progress)),
            'details': details
        }
        
        self._progress_updates.append(progress_update)
        
        # Log progress update
        self._logger.debug(f"Operation progress: {self.operation_name}",
                          extra={
                              'correlation_id': self.correlation_id,
                              'operation_id': self.operation_id,
                              'stage': stage,
                              'progress': progress,
                              'details': details
                          })
    
    def log_metric(self, name: str, value: Union[int, float, str]) -> None:
        """
        Log a custom metric for this operation.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self._logged_metrics[name] = value
        
        self._logger.debug(f"Operation metric: {name}={value}",
                          extra={
                              'correlation_id': self.correlation_id,
                              'operation_id': self.operation_id,
                              'metric_name': name,
                              'metric_value': value
                          })
    
    def log_success(self, message: str, **kwargs) -> None:
        """
        Log successful completion with additional data.
        
        Args:
            message: Success message
            **kwargs: Additional data to log
        """
        extra_data = {
            'correlation_id': self.correlation_id,
            'operation_id': self.operation_id,
            'operation_status': 'success'
        }
        extra_data.update(kwargs)
        
        self._logger.info(f"Operation success: {message}", extra=extra_data)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """
        Log operation warning with additional data.
        
        Args:
            message: Warning message
            **kwargs: Additional data to log
        """
        extra_data = {
            'correlation_id': self.correlation_id,
            'operation_id': self.operation_id,
            'operation_status': 'warning'
        }
        extra_data.update(kwargs)
        
        self._logger.warning(f"Operation warning: {message}", extra=extra_data)
    
    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """
        Log operation error with additional data.
        
        Args:
            message: Error message
            error: Optional exception object
            **kwargs: Additional data to log
        """
        extra_data = {
            'correlation_id': self.correlation_id,
            'operation_id': self.operation_id,
            'operation_status': 'error'
        }
        extra_data.update(kwargs)
        
        if error:
            extra_data.update({
                'error_type': type(error).__name__,
                'error_message': str(error)
            })
        
        self._logger.error(f"Operation error: {message}", 
                          exc_info=error,  
                          extra=extra_data)
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        end_time = self.end_time or time.time()
        return (end_time - self.start_time) * 1000
    
    def get_context_data(self) -> Dict[str, Any]:
        """Get context data for logging enrichment."""
        context_data = {
            'operation_id': self.operation_id,
            'operation_name': self.operation_name,
            'correlation_id': self.correlation_id,
            'elapsed_ms': self.elapsed_ms,
            'is_active': self.is_active,
            'child_operation_count': len(self.child_operations)
        }
        
        # Add parent operation info
        if self.parent_context:
            context_data['parent_operation_id'] = self.parent_context.operation_id
        
        # Add custom data
        if self.custom_data:
            context_data['custom_data'] = self.custom_data
        
        # Add performance metrics if available
        if self._metrics:
            context_data['performance'] = self._metrics.to_dict()
        
        # Add recent progress
        if self._progress_updates:
            context_data['latest_progress'] = self._progress_updates[-1]
        
        return context_data


class ContextEnricher:
    """
    Performance-optimized context injection engine.
    
    Automatically enriches log messages with correlation IDs, performance
    metrics, thread information, and system context. Integrates with
    existing monitoring systems for comprehensive observability.
    
    Features:
    - <150μs context enrichment target
    - Automatic correlation ID propagation
    - System performance metrics injection
    - Thread and process information
    - Integration with existing monitoring.py
    - Caching for performance optimization
    """
    
    def __init__(self, 
                 enable_performance_metrics: bool = True,
                 enable_system_metrics: bool = True,
                 enable_thread_info: bool = True,
                 cache_system_info: bool = True,
                 cache_ttl_seconds: float = 5.0):
        """
        Initialize context enricher.
        
        Args:
            enable_performance_metrics: Enable performance metric collection
            enable_system_metrics: Enable system resource metrics
            enable_thread_info: Enable thread and process information
            cache_system_info: Cache system information for performance
            cache_ttl_seconds: TTL for cached system information
        """
        self.enable_performance_metrics = enable_performance_metrics
        self.enable_system_metrics = enable_system_metrics
        self.enable_thread_info = enable_thread_info
        self.cache_system_info = cache_system_info
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Caching for performance
        self._system_info_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0.0
        self._cache_lock = threading.RLock()
        
        # Process information (cached)
        self._process_id = os.getpid()
        self._process_start_time = time.time()
        
        # Integration with monitoring system
        self._performance_monitor = None
        self._resource_manager = None
        
        if MONITORING_AVAILABLE:
            try:
                self._performance_monitor = get_performance_monitor()
                self._resource_manager = get_resource_manager()
            except Exception as e:
                pass  # Fallback gracefully
        
        # Performance tracking
        self._enrichment_count = 0
        self._total_enrichment_time = 0.0
        self._enrichment_times = deque(maxlen=1000)
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.ContextEnricher")
        
        self._logger.info("ContextEnricher initialized",
                         extra={
                             'performance_metrics': enable_performance_metrics,
                             'system_metrics': enable_system_metrics,
                             'thread_info': enable_thread_info,
                             'monitoring_integration': MONITORING_AVAILABLE
                         })
    
    def enrich_log_message(self, 
                          logger_name: str,
                          level: int,
                          message: str,
                          extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enrich log message with context information.
        
        Args:
            logger_name: Name of the logger
            level: Log level
            message: Log message
            extra_data: Additional data from logging call
            
        Returns:
            Enriched context dictionary
        """
        start_time = time.perf_counter()
        
        try:
            # Base context
            context = {
                'timestamp': time.time(),
                'level': level,
                'logger': logger_name,
                'message': message
            }
            
            # Add correlation ID and operation context
            context['ids'] = self._get_id_context()
            
            # Add thread information
            if self.enable_thread_info:
                context['thread'] = self._get_thread_context()
            
            # Add system metrics
            if self.enable_system_metrics:
                context['system'] = self._get_system_context()
            
            # Add performance metrics
            if self.enable_performance_metrics:
                context['performance'] = self._get_performance_context()
            
            # Add operation context if available
            operation_context = self._get_current_operation_context()
            if operation_context:
                context['operation'] = operation_context
            
            # Add extra data
            if extra_data:
                # Merge extra data, handling special keys
                for key, value in extra_data.items():
                    if key in ('correlation_id', 'operation_id'):
                        context['ids'][key] = value
                    elif key.startswith('metric_'):
                        if 'metrics' not in context:
                            context['metrics'] = {}
                        context['metrics'][key[7:]] = value  # Remove 'metric_' prefix
                    else:
                        if 'context' not in context:
                            context['context'] = {}
                        context['context'][key] = value
            
            # Track enrichment performance
            elapsed_time = time.perf_counter() - start_time
            self._update_enrichment_stats(elapsed_time)
            
            return context
            
        except Exception as e:
            self._logger.error(f"Context enrichment failed: {e}")
            # Return minimal context to prevent logging failure
            return {
                'timestamp': time.time(),
                'level': level,
                'logger': logger_name,
                'message': message,
                'enrichment_error': str(e)
            }
    
    def _get_id_context(self) -> Dict[str, Any]:
        """Get correlation ID and operation context."""
        ids = {
            'process_id': self._process_id,
            'thread_id': threading.current_thread().name,
            'sequence_number': self._get_sequence_number()
        }
        
        # Get correlation ID from current operation context
        current_context = self._get_current_operation_context()
        if current_context and 'correlation_id' in current_context:
            ids['correlation_id'] = current_context['correlation_id']
            ids['operation_id'] = current_context.get('operation_id')
        else:
            # Generate new correlation ID if none exists
            ids['correlation_id'] = f"log_{uuid.uuid4().hex[:16]}"
        
        return ids
    
    def _get_thread_context(self) -> Dict[str, Any]:
        """Get thread and process information."""
        thread = threading.current_thread()
        
        return {
            'thread_name': thread.name,
            'thread_id': thread.ident,
            'is_daemon': thread.daemon,
            'is_alive': thread.is_alive(),
            'active_thread_count': threading.active_count()
        }
    
    def _get_system_context(self) -> Dict[str, Any]:
        """Get system metrics and information."""
        # Use cached system info if available and fresh
        if self.cache_system_info:
            current_time = time.time()
            
            with self._cache_lock:
                if (self._system_info_cache and 
                    current_time - self._cache_timestamp < self.cache_ttl_seconds):
                    return self._system_info_cache.copy()
        
        # Collect fresh system information
        system_info = {
            'process_id': self._process_id,
            'process_uptime_seconds': time.time() - self._process_start_time
        }
        
        try:
            if psutil:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                system_info.update({
                    'memory_rss_mb': memory_info.rss / 1024 / 1024,
                    'memory_vms_mb': memory_info.vms / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None
                })
                
                # System-wide info (expensive, so cache it)
                if not hasattr(self, '_last_system_check') or time.time() - self._last_system_check > 30:
                    self._last_system_check = time.time()
                    try:
                        system_memory = psutil.virtual_memory()
                        system_info.update({
                            'system_memory_percent': system_memory.percent,
                            'system_cpu_percent': psutil.cpu_percent(interval=0.1),
                            'system_load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
                        })
                    except Exception:
                        pass  # Skip system-wide metrics on error
        
        except Exception as e:
            system_info['metrics_error'] = str(e)
        
        # Cache the system info
        if self.cache_system_info:
            with self._cache_lock:
                self._system_info_cache = system_info.copy()
                self._cache_timestamp = time.time()
        
        return system_info
    
    def _get_performance_context(self) -> Dict[str, Any]:
        """Get performance metrics."""
        performance = {}
        
        try:
            # Integration with existing monitoring system
            if self._performance_monitor and MONITORING_AVAILABLE:
                try:
                    monitor_stats = self._performance_monitor.get_metrics_summary()
                    if isinstance(monitor_stats, dict) and 'error' not in monitor_stats:
                        performance['monitoring_stats'] = {
                            'monitoring_state': monitor_stats.get('monitoring_state'),
                            'system_metrics': monitor_stats.get('system_metrics')
                        }
                except Exception:
                    pass  # Skip on error
            
            # Resource manager integration
            if self._resource_manager and MONITORING_AVAILABLE:
                try:
                    health_status = self._resource_manager.get_health_status()
                    if isinstance(health_status, dict):
                        performance['resource_health'] = {
                            'overall_health': health_status.get('overall_health'),
                            'resource_states': health_status.get('resource_states')
                        }
                except Exception:
                    pass  # Skip on error
            
            # Enrichment performance stats
            if self._enrichment_count > 0:
                avg_enrichment_time = self._total_enrichment_time / self._enrichment_count
                performance['enrichment_stats'] = {
                    'average_enrichment_time_us': avg_enrichment_time * 1_000_000,
                    'enrichment_count': self._enrichment_count
                }
        
        except Exception as e:
            performance['performance_error'] = str(e)
        
        return performance
    
    def _get_current_operation_context(self) -> Optional[Dict[str, Any]]:
        """Get current operation context from thread-local storage."""
        try:
            if hasattr(_thread_local, 'context_stack') and _thread_local.context_stack:
                current_context = _thread_local.context_stack[-1]
                return current_context.get_context_data()
        except Exception:
            pass
        
        return None
    
    def _get_sequence_number(self) -> int:
        """Get sequence number for this log entry."""
        if not hasattr(_thread_local, 'sequence_counter'):
            _thread_local.sequence_counter = 0
        
        _thread_local.sequence_counter += 1
        return _thread_local.sequence_counter
    
    def _update_enrichment_stats(self, elapsed_time: float) -> None:
        """Update enrichment performance statistics."""
        self._enrichment_count += 1
        self._total_enrichment_time += elapsed_time
        self._enrichment_times.append(elapsed_time)
        
        # Warn if enrichment is taking too long
        if elapsed_time > 0.0005:  # 500μs warning threshold
            self._logger.warning(f"Slow context enrichment: {elapsed_time * 1000:.1f}ms")
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get context enrichment performance statistics."""
        if self._enrichment_count == 0:
            return {'status': 'no_data'}
        
        avg_time = self._total_enrichment_time / self._enrichment_count
        recent_times = list(self._enrichment_times)[-100:]  # Last 100
        recent_avg = sum(recent_times) / len(recent_times) if recent_times else 0
        
        return {
            'total_enrichments': self._enrichment_count,
            'average_time_us': avg_time * 1_000_000,
            'recent_average_time_us': recent_avg * 1_000_000,
            'max_time_us': max(self._enrichment_times) * 1_000_000 if self._enrichment_times else 0,
            'min_time_us': min(self._enrichment_times) * 1_000_000 if self._enrichment_times else 0,
            'performance_target_met': avg_time < 0.00015,  # 150μs target
            'cache_enabled': self.cache_system_info,
            'monitoring_integration': MONITORING_AVAILABLE
        }


# Context manager function for operation tracking
@contextmanager
def operation_context(operation_name: str,
                     correlation_id: Optional[str] = None,
                     track_performance: bool = True,
                     custom_data: Optional[Dict[str, Any]] = None):
    """
    Context manager for operation tracking.
    
    Args:
        operation_name: Name of the operation
        correlation_id: Optional correlation ID
        track_performance: Whether to track performance metrics
        custom_data: Additional custom data
        
    Yields:
        OperationContext instance
        
    Example:
        with operation_context("card_recognition") as ctx:
            ctx.set_progress("preprocessing", 0.2)
            # ... do work ...
            ctx.log_success("recognition_complete", card_code="AT_001")
    """
    # Get parent context from thread-local if available
    parent_context = None
    if hasattr(_thread_local, 'context_stack') and _thread_local.context_stack:
        parent_context = _thread_local.context_stack[-1]
    
    # Create operation context
    context = OperationContext(
        operation_name=operation_name,
        correlation_id=correlation_id,
        parent_context=parent_context,
        track_performance=track_performance,
        custom_data=custom_data
    )
    
    with context:
        yield context


def timed_operation(operation_name: Optional[str] = None,
                   track_memory: bool = False,
                   log_success: bool = True,
                   log_errors: bool = True):
    """
    Decorator for automatic operation timing and logging.
    
    Args:
        operation_name: Name of operation (defaults to function name)
        track_memory: Whether to track memory usage
        log_success: Whether to log successful completion
        log_errors: Whether to log errors
        
    Example:
        @timed_operation("card_recognition", track_memory=True)
        def recognize_card(image):
            # Implementation
            return result
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with operation_context(name, track_performance=True) as ctx:
                try:
                    # Add memory tracking if requested
                    if track_memory and psutil:
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024
                        ctx.log_metric("start_memory_mb", start_memory)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Log memory usage
                    if track_memory and psutil:
                        end_memory = process.memory_info().rss / 1024 / 1024
                        ctx.log_metric("end_memory_mb", end_memory)
                        ctx.log_metric("memory_delta_mb", end_memory - start_memory)
                    
                    # Log success
                    if log_success:
                        ctx.log_success(f"Operation completed: {name}",
                                       function_name=func.__name__,
                                       result_type=type(result).__name__)
                    
                    return result
                    
                except Exception as e:
                    if log_errors:
                        ctx.log_error(f"Operation failed: {name}", error=e,
                                     function_name=func.__name__)
                    raise
        
        return wrapper
    return decorator


# Global context enricher instance
_global_enricher: Optional[ContextEnricher] = None


def get_context_enricher() -> ContextEnricher:
    """Get global context enricher instance."""
    global _global_enricher
    if _global_enricher is None:
        _global_enricher = ContextEnricher()
    return _global_enricher


def set_context_enricher(enricher: ContextEnricher) -> None:
    """Set global context enricher instance."""
    global _global_enricher
    _global_enricher = enricher


# Module exports
__all__ = [
    'OperationMetrics',
    'OperationContext',
    'ContextEnricher',
    'operation_context',
    'timed_operation',
    'get_context_enricher',
    'set_context_enricher'
]