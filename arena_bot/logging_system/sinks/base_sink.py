"""
Base Sink for S-Tier Logging System.

This module provides the abstract base class and common functionality
for all logging sinks, including error handling, health monitoring,
and performance tracking.

Features:
- Abstract base class with common sink functionality
- Health monitoring and error recovery
- Performance tracking and statistics
- Thread-safe operations
- Configurable error handling and retries
- Graceful degradation under failures
"""

import os
import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass
from collections import deque
from enum import Enum

# Import from our core components
from ..core.hybrid_async_queue import LogMessage
from ..formatters.structured_formatter import StructuredFormatter


class SinkState(Enum):
    """Sink operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    DISABLED = "disabled"
    SHUTDOWN = "shutdown"


class ErrorHandlingStrategy(Enum):
    """Error handling strategies for sinks."""
    IGNORE = "ignore"          # Ignore errors and continue
    LOG_AND_CONTINUE = "log_and_continue"  # Log error but continue
    RETRY = "retry"            # Retry on error with backoff
    DISABLE_ON_ERROR = "disable_on_error"  # Disable sink on error
    ESCALATE = "escalate"      # Escalate to emergency sink


@dataclass
class SinkStats:
    """Statistics for sink performance monitoring."""
    
    sink_name: str
    state: SinkState
    messages_processed: int
    messages_failed: int
    total_processing_time: float
    average_processing_time_ms: float
    error_rate_percent: float
    last_activity: float
    uptime_seconds: float
    
    # Health metrics
    consecutive_errors: int
    last_error_time: Optional[float]
    last_error_message: Optional[str]
    
    # Performance metrics
    throughput_msgs_per_sec: float
    peak_processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sink_name': self.sink_name,
            'state': self.state.value,
            'messages_processed': self.messages_processed,
            'messages_failed': self.messages_failed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time_ms': self.average_processing_time_ms,
            'error_rate_percent': self.error_rate_percent,
            'last_activity': self.last_activity,
            'uptime_seconds': self.uptime_seconds,
            'consecutive_errors': self.consecutive_errors,
            'last_error_time': self.last_error_time,
            'last_error_message': self.last_error_message,
            'throughput_msgs_per_sec': self.throughput_msgs_per_sec,
            'peak_processing_time_ms': self.peak_processing_time_ms
        }


class BaseSink(ABC):
    """
    Abstract base class for all logging sinks.
    
    Provides common functionality including error handling, health monitoring,
    performance tracking, and thread safety. All concrete sinks should inherit
    from this class and implement the abstract methods.
    
    Features:
    - Thread-safe operations with minimal locking
    - Comprehensive error handling with configurable strategies
    - Health monitoring with automatic recovery
    - Performance tracking and statistics
    - Graceful degradation and recovery
    """
    
    def __init__(self,
                 name: str,
                 formatter: Optional[Any] = None,
                 error_strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.LOG_AND_CONTINUE,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 health_check_interval: float = 60.0,
                 enable_performance_tracking: bool = True):
        """
        Initialize base sink.
        
        Args:
            name: Sink name for identification
            formatter: Message formatter (defaults to StructuredFormatter)
            error_strategy: How to handle errors
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            health_check_interval: Health check interval in seconds
            enable_performance_tracking: Enable performance monitoring
        """
        self.name = name
        self.formatter = formatter or StructuredFormatter()
        self.error_strategy = error_strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        self.enable_performance_tracking = enable_performance_tracking
        
        # State management
        self._state = SinkState.INITIALIZING
        self._state_lock = threading.RLock()
        self._initialization_time = time.time()
        self._shutdown_requested = False
        
        # Performance tracking
        self._messages_processed = 0
        self._messages_failed = 0
        self._total_processing_time = 0.0
        self._processing_times = deque(maxlen=1000)  # Last 1000 processing times
        self._peak_processing_time = 0.0
        self._last_activity = time.time()
        
        # Error handling
        self._consecutive_errors = 0
        self._last_error_time: Optional[float] = None
        self._last_error_message: Optional[str] = None
        self._error_history = deque(maxlen=100)  # Last 100 errors
        
        # Health monitoring
        self._last_health_check = time.time()
        self._health_check_failures = 0
        
        # Thread safety
        self._write_lock = threading.RLock()
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
        
        self._logger.info(f"BaseSink '{name}' initialized",
                         extra={
                             'sink_type': self.__class__.__name__,
                             'error_strategy': error_strategy.value,
                             'max_retries': max_retries
                         })
    
    @abstractmethod
    def _write_message(self, formatted_message: str, message: LogMessage) -> bool:
        """
        Write formatted message to sink destination.
        
        This method must be implemented by concrete sink classes.
        
        Args:
            formatted_message: Formatted message string
            message: Original LogMessage object
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _initialize_sink(self) -> bool:
        """
        Initialize sink-specific resources.
        
        This method must be implemented by concrete sink classes.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _cleanup_sink(self) -> bool:
        """
        Cleanup sink-specific resources.
        
        This method must be implemented by concrete sink classes.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _health_check_sink(self) -> bool:
        """
        Perform sink-specific health check.
        
        This method must be implemented by concrete sink classes.
        
        Returns:
            True if sink is healthy, False otherwise
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the sink.
        
        Returns:
            True if initialization successful, False otherwise
        """
        with self._state_lock:
            if self._state != SinkState.INITIALIZING:
                return self._state == SinkState.HEALTHY
            
            try:
                # Perform sink-specific initialization
                if self._initialize_sink():
                    self._state = SinkState.HEALTHY
                    self._logger.info(f"Sink '{self.name}' initialized successfully")
                    return True
                else:
                    self._state = SinkState.ERROR
                    self._logger.error(f"Sink '{self.name}' initialization failed")
                    return False
                    
            except Exception as e:
                self._state = SinkState.ERROR
                self._logger.error(f"Sink '{self.name}' initialization error: {e}")
                return False
    
    def write(self, message: LogMessage) -> bool:
        """
        Write message to sink with error handling and performance tracking.
        
        Args:
            message: LogMessage to write
            
        Returns:
            True if successful, False otherwise
        """
        if self._shutdown_requested:
            return False
        
        # Check if sink is operational
        if not self._is_operational():
            return False
        
        start_time = time.perf_counter()
        success = False
        
        try:
            with self._write_lock:
                # Format message
                try:
                    formatted_message = self.formatter.format(message)
                except Exception as e:
                    self._handle_error(f"Message formatting failed: {e}", message)
                    return False
                
                # Write with retry logic
                success = self._write_with_retry(formatted_message, message)
                
                # Update performance tracking
                elapsed_time = time.perf_counter() - start_time
                self._update_performance_stats(elapsed_time, success)
                
                # Update health state
                if success:
                    self._handle_success()
                else:
                    self._handle_failure("Write operation failed", message)
                
                return success
                
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            self._update_performance_stats(elapsed_time, False)
            self._handle_error(f"Unexpected error in write: {e}", message)
            return False
    
    def _write_with_retry(self, formatted_message: str, message: LogMessage) -> bool:
        """
        Write message with retry logic.
        
        Args:
            formatted_message: Formatted message to write
            message: Original LogMessage object
            
        Returns:
            True if successful, False otherwise
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self._write_message(formatted_message, message):
                    if attempt > 0:
                        self._logger.info(f"Write succeeded on attempt {attempt + 1}")
                    return True
                else:
                    last_error = "Write method returned False"
                    
            except Exception as e:
                last_error = str(e)
                self._logger.warning(f"Write attempt {attempt + 1} failed: {e}")
            
            # Apply retry delay (except on last attempt)
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(min(delay, 30.0))  # Cap at 30 seconds
        
        # All retries failed
        self._logger.error(f"All {self.max_retries + 1} write attempts failed. Last error: {last_error}")
        return False
    
    def _is_operational(self) -> bool:
        """Check if sink is in an operational state."""
        with self._state_lock:
            return self._state in (SinkState.HEALTHY, SinkState.DEGRADED)
    
    def _handle_success(self) -> None:
        """Handle successful write operation."""
        with self._state_lock:
            self._consecutive_errors = 0
            self._last_activity = time.time()
            
            # Recover from degraded state
            if self._state == SinkState.DEGRADED:
                self._state = SinkState.HEALTHY
                self._logger.info(f"Sink '{self.name}' recovered to healthy state")
    
    def _handle_failure(self, error_message: str, message: Optional[LogMessage] = None) -> None:
        """
        Handle write failure according to error strategy.
        
        Args:
            error_message: Error description
            message: Failed LogMessage (optional)
        """
        self._handle_error(error_message, message)
        
        # Apply error handling strategy
        if self.error_strategy == ErrorHandlingStrategy.DISABLE_ON_ERROR:
            if self._consecutive_errors >= 3:  # Disable after 3 consecutive errors
                with self._state_lock:
                    self._state = SinkState.DISABLED
                    self._logger.error(f"Sink '{self.name}' disabled due to consecutive errors")
    
    def _handle_error(self, error_message: str, message: Optional[LogMessage] = None) -> None:
        """
        Handle error occurrence.
        
        Args:
            error_message: Error description
            message: Failed LogMessage (optional)
        """
        current_time = time.time()
        
        with self._state_lock:
            self._consecutive_errors += 1
            self._last_error_time = current_time
            self._last_error_message = error_message
            
            # Add to error history
            self._error_history.append({
                'timestamp': current_time,
                'message': error_message,
                'log_message_id': message.correlation_id if message else None
            })
            
            # Update state if needed
            if self._state == SinkState.HEALTHY and self._consecutive_errors >= 3:
                self._state = SinkState.DEGRADED
                self._logger.warning(f"Sink '{self.name}' degraded due to errors")
        
        # Log error according to strategy
        if self.error_strategy in (ErrorHandlingStrategy.LOG_AND_CONTINUE, 
                                  ErrorHandlingStrategy.RETRY,
                                  ErrorHandlingStrategy.ESCALATE):
            self._logger.error(f"Sink '{self.name}' error: {error_message}")
    
    def _update_performance_stats(self, elapsed_time: float, success: bool) -> None:
        """
        Update performance statistics.
        
        Args:
            elapsed_time: Time taken for operation
            success: Whether operation succeeded
        """
        if not self.enable_performance_tracking:
            return
        
        self._messages_processed += 1
        if not success:
            self._messages_failed += 1
        
        self._total_processing_time += elapsed_time
        self._processing_times.append(elapsed_time)
        self._peak_processing_time = max(self._peak_processing_time, elapsed_time)
        self._last_activity = time.time()
    
    def perform_health_check(self) -> bool:
        """
        Perform comprehensive health check.
        
        Returns:
            True if sink is healthy, False otherwise
        """
        current_time = time.time()
        
        # Check if health check is due
        if current_time - self._last_health_check < self.health_check_interval:
            return self.is_healthy()
        
        self._last_health_check = current_time
        
        try:
            # Perform sink-specific health check
            sink_healthy = self._health_check_sink()
            
            # Check general health indicators
            general_healthy = (
                self._state in (SinkState.HEALTHY, SinkState.DEGRADED) and
                self._consecutive_errors < 10 and  # Not too many errors
                current_time - self._last_activity < 300  # Active within 5 minutes
            )
            
            overall_healthy = sink_healthy and general_healthy
            
            # Update health check failure count
            if overall_healthy:
                self._health_check_failures = 0
            else:
                self._health_check_failures += 1
                
                # Disable sink if health checks keep failing
                if self._health_check_failures >= 3:
                    with self._state_lock:
                        if self._state != SinkState.DISABLED:
                            self._state = SinkState.ERROR
                            self._logger.error(f"Sink '{self.name}' marked as error due to health check failures")
            
            return overall_healthy
            
        except Exception as e:
            self._logger.error(f"Health check failed for sink '{self.name}': {e}")
            self._health_check_failures += 1
            return False
    
    def is_healthy(self) -> bool:
        """
        Check if sink is currently healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        with self._state_lock:
            return self._state == SinkState.HEALTHY
    
    def get_stats(self) -> SinkStats:
        """
        Get comprehensive sink statistics.
        
        Returns:
            SinkStats object with performance and health metrics
        """
        current_time = time.time()
        uptime = current_time - self._initialization_time
        
        # Calculate averages
        avg_processing_time = (
            self._total_processing_time / self._messages_processed
            if self._messages_processed > 0 else 0.0
        )
        
        error_rate = (
            (self._messages_failed / self._messages_processed) * 100
            if self._messages_processed > 0 else 0.0
        )
        
        throughput = (
            self._messages_processed / uptime
            if uptime > 0 else 0.0
        )
        
        return SinkStats(
            sink_name=self.name,
            state=self._state,
            messages_processed=self._messages_processed,
            messages_failed=self._messages_failed,
            total_processing_time=self._total_processing_time,
            average_processing_time_ms=avg_processing_time * 1000,
            error_rate_percent=error_rate,
            last_activity=self._last_activity,
            uptime_seconds=uptime,
            consecutive_errors=self._consecutive_errors,
            last_error_time=self._last_error_time,
            last_error_message=self._last_error_message,
            throughput_msgs_per_sec=throughput,
            peak_processing_time_ms=self._peak_processing_time * 1000
        )
    
    def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Shutdown sink gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            True if shutdown cleanly, False otherwise
        """
        self._shutdown_requested = True
        
        try:
            with self._state_lock:
                if self._state == SinkState.SHUTDOWN:
                    return True
                
                self._logger.info(f"Shutting down sink '{self.name}'")
                
                # Perform sink-specific cleanup
                cleanup_success = self._cleanup_sink()
                
                self._state = SinkState.SHUTDOWN
                
                if cleanup_success:
                    self._logger.info(f"Sink '{self.name}' shutdown successfully")
                else:
                    self._logger.warning(f"Sink '{self.name}' shutdown with warnings")
                
                return cleanup_success
                
        except Exception as e:
            self._logger.error(f"Sink '{self.name}' shutdown error: {e}")
            return False
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        Get recent error history.
        
        Returns:
            List of error records
        """
        return list(self._error_history)
    
    def reset_error_state(self) -> None:
        """Reset error state and counters."""
        with self._state_lock:
            self._consecutive_errors = 0
            self._last_error_time = None
            self._last_error_message = None
            self._health_check_failures = 0
            
            if self._state in (SinkState.ERROR, SinkState.DEGRADED):
                self._state = SinkState.HEALTHY
                self._logger.info(f"Sink '{self.name}' error state reset")


# Module exports
__all__ = [
    'BaseSink',
    'SinkState',
    'ErrorHandlingStrategy', 
    'SinkStats'
]