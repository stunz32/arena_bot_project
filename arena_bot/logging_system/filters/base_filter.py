"""
Base Filter for S-Tier Logging System.

This module provides the abstract base class for all logging filters,
including common functionality for message processing, performance tracking,
and filter chaining.

Features:
- Abstract base class with standardized interface
- Performance tracking and statistics
- Filter chaining and composition
- Thread-safe operations
- Health monitoring and error handling
- Configurable filter parameters
"""

import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from collections import deque
from enum import Enum

# Import from our core components
from ..core.hybrid_async_queue import LogMessage


class FilterResult(Enum):
    """Filter decision results."""
    ACCEPT = "accept"      # Accept message for processing
    REJECT = "reject"      # Reject message completely  
    MODIFY = "modify"      # Accept message with modifications
    DEFER = "defer"        # Defer decision to next filter


class FilterAction(Enum):
    """Actions that can be performed on messages."""
    ALLOW = "allow"            # Allow message through unchanged
    BLOCK = "block"            # Block message completely
    TRANSFORM = "transform"    # Transform message content
    ROUTE = "route"            # Route to specific destinations
    SAMPLE = "sample"          # Sample message (accept some, reject others)


@dataclass
class FilterStats:
    """Statistics for filter performance monitoring."""
    
    filter_name: str
    messages_processed: int
    messages_accepted: int
    messages_rejected: int
    messages_modified: int
    total_processing_time: float
    average_processing_time_us: float
    error_count: int
    last_error_time: Optional[float]
    last_error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filter_name': self.filter_name,
            'messages_processed': self.messages_processed,
            'messages_accepted': self.messages_accepted,
            'messages_rejected': self.messages_rejected,
            'messages_modified': self.messages_modified,
            'total_processing_time': self.total_processing_time,
            'average_processing_time_us': self.average_processing_time_us,
            'acceptance_rate_percent': (
                (self.messages_accepted / self.messages_processed) * 100
                if self.messages_processed > 0 else 0
            ),
            'rejection_rate_percent': (
                (self.messages_rejected / self.messages_processed) * 100
                if self.messages_processed > 0 else 0
            ),
            'error_count': self.error_count,
            'last_error_time': self.last_error_time,
            'last_error_message': self.last_error_message
        }


class BaseFilter(ABC):
    """
    Abstract base class for all logging filters.
    
    Provides common functionality for message filtering including performance
    tracking, error handling, and standardized interfaces. All concrete
    filters should inherit from this class.
    
    Features:
    - Standardized filter interface
    - Performance tracking and statistics
    - Thread-safe operations
    - Error handling and recovery
    - Filter composition support
    """
    
    def __init__(self,
                 name: str,
                 enabled: bool = True,
                 track_performance: bool = True,
                 max_error_count: int = 100):
        """
        Initialize base filter.
        
        Args:
            name: Filter name for identification
            enabled: Whether filter is initially enabled
            track_performance: Enable performance tracking
            max_error_count: Maximum errors before filter is disabled
        """
        self.name = name
        self.enabled = enabled
        self.track_performance = track_performance
        self.max_error_count = max_error_count
        
        # Performance tracking
        self._messages_processed = 0
        self._messages_accepted = 0
        self._messages_rejected = 0
        self._messages_modified = 0
        self._total_processing_time = 0.0
        self._processing_times = deque(maxlen=1000)  # Last 1000 processing times
        
        # Error handling
        self._error_count = 0
        self._last_error_time: Optional[float] = None
        self._last_error_message: Optional[str] = None
        self._error_history = deque(maxlen=100)  # Last 100 errors
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
        
        self._logger.info(f"Filter '{name}' initialized",
                         extra={
                             'filter_type': self.__class__.__name__,
                             'enabled': enabled,
                             'performance_tracking': track_performance
                         })
    
    @abstractmethod
    def _apply_filter(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """
        Apply filter logic to message.
        
        This method must be implemented by concrete filter classes.
        
        Args:
            message: LogMessage to filter
            
        Returns:
            Tuple of (FilterResult, modified_message_or_None)
        """
        pass
    
    def filter(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """
        Filter a log message with performance tracking and error handling.
        
        Args:
            message: LogMessage to filter
            
        Returns:
            Tuple of (FilterResult, modified_message_or_None)
        """
        if not self.enabled:
            return FilterResult.ACCEPT, message
        
        start_time = time.perf_counter() if self.track_performance else 0
        
        try:
            with self._lock:
                # Apply filter logic
                result, modified_message = self._apply_filter(message)
                
                # Update statistics
                self._update_stats(result, start_time)
                
                return result, modified_message
                
        except Exception as e:
            # Handle filter errors
            elapsed_time = time.perf_counter() - start_time if self.track_performance else 0
            self._handle_error(e, elapsed_time)
            
            # Return accept on error to avoid losing messages
            return FilterResult.ACCEPT, message
    
    def _update_stats(self, result: FilterResult, start_time: float) -> None:
        """Update filter statistics."""
        if not self.track_performance:
            return
        
        with self._lock:
            self._messages_processed += 1
            
            if result == FilterResult.ACCEPT:
                self._messages_accepted += 1
            elif result == FilterResult.REJECT:
                self._messages_rejected += 1
            elif result == FilterResult.MODIFY:
                self._messages_modified += 1
                self._messages_accepted += 1  # Modified messages are also accepted
            
            # Track processing time
            elapsed_time = time.perf_counter() - start_time
            self._total_processing_time += elapsed_time
            self._processing_times.append(elapsed_time)
    
    def _handle_error(self, error: Exception, elapsed_time: float) -> None:
        """Handle filter errors."""
        current_time = time.time()
        
        with self._lock:
            self._error_count += 1
            self._last_error_time = current_time
            self._last_error_message = str(error)
            
            # Add to error history
            self._error_history.append({
                'timestamp': current_time,
                'error': str(error),
                'error_type': type(error).__name__,
                'processing_time': elapsed_time
            })
            
            # Disable filter if too many errors
            if self._error_count >= self.max_error_count:
                self.enabled = False
                self._logger.error(f"Filter '{self.name}' disabled due to excessive errors",
                                 extra={
                                     'error_count': self._error_count,
                                     'max_error_count': self.max_error_count
                                 })
            else:
                self._logger.warning(f"Filter '{self.name}' error: {error}",
                                   extra={
                                       'error_count': self._error_count,
                                       'error_type': type(error).__name__
                                   })
    
    def get_stats(self) -> FilterStats:
        """Get comprehensive filter statistics."""
        with self._lock:
            avg_processing_time = (
                self._total_processing_time / self._messages_processed
                if self._messages_processed > 0 else 0.0
            )
            
            return FilterStats(
                filter_name=self.name,
                messages_processed=self._messages_processed,
                messages_accepted=self._messages_accepted,
                messages_rejected=self._messages_rejected,
                messages_modified=self._messages_modified,
                total_processing_time=self._total_processing_time,
                average_processing_time_us=avg_processing_time * 1_000_000,
                error_count=self._error_count,
                last_error_time=self._last_error_time,
                last_error_message=self._last_error_message
            )
    
    def enable(self) -> None:
        """Enable the filter."""
        with self._lock:
            self.enabled = True
            self._logger.info(f"Filter '{self.name}' enabled")
    
    def disable(self) -> None:
        """Disable the filter."""
        with self._lock:
            self.enabled = False
            self._logger.info(f"Filter '{self.name}' disabled")
    
    def reset_stats(self) -> None:
        """Reset filter statistics."""
        with self._lock:
            self._messages_processed = 0
            self._messages_accepted = 0
            self._messages_rejected = 0
            self._messages_modified = 0
            self._total_processing_time = 0.0
            self._processing_times.clear()
            self._logger.info(f"Filter '{self.name}' statistics reset")
    
    def reset_errors(self) -> None:
        """Reset error counters and re-enable if disabled due to errors."""
        with self._lock:
            self._error_count = 0
            self._last_error_time = None
            self._last_error_message = None
            self._error_history.clear()
            
            if not self.enabled:
                self.enabled = True
                self._logger.info(f"Filter '{self.name}' re-enabled after error reset")
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get recent error history."""
        with self._lock:
            return list(self._error_history)
    
    def is_healthy(self) -> bool:
        """Check if filter is healthy and operational."""
        with self._lock:
            return (
                self.enabled and
                self._error_count < self.max_error_count and
                (self._messages_processed == 0 or 
                 self._error_count / self._messages_processed < 0.1)  # < 10% error rate
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        stats = self.get_stats()
        
        with self._lock:
            recent_times = list(self._processing_times)[-100:]  # Last 100
            
            summary = {
                'filter_name': self.name,
                'enabled': self.enabled,
                'healthy': self.is_healthy(),
                'messages_processed': stats.messages_processed,
                'acceptance_rate_percent': stats.to_dict()['acceptance_rate_percent'],
                'average_processing_time_us': stats.average_processing_time_us,
                'error_rate_percent': (
                    (self._error_count / self._messages_processed) * 100
                    if self._messages_processed > 0 else 0
                )
            }
            
            if recent_times:
                summary.update({
                    'recent_avg_processing_time_us': (sum(recent_times) / len(recent_times)) * 1_000_000,
                    'max_processing_time_us': max(recent_times) * 1_000_000,
                    'min_processing_time_us': min(recent_times) * 1_000_000
                })
            
            return summary
    
    def __str__(self) -> str:
        """String representation of filter."""
        return f"{self.__class__.__name__}({self.name}, enabled={self.enabled})"
    
    def __repr__(self) -> str:
        """Detailed string representation of filter."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
               f"enabled={self.enabled}, "
               f"messages_processed={self._messages_processed}, "
               f"error_count={self._error_count})")


class FilterChain:
    """
    Chain of filters for sequential message processing.
    
    Allows multiple filters to be applied in sequence with early termination
    and result aggregation.
    """
    
    def __init__(self, name: str, filters: Optional[List[BaseFilter]] = None):
        """
        Initialize filter chain.
        
        Args:
            name: Chain name for identification
            filters: List of filters in processing order
        """
        self.name = name
        self.filters = filters or []
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.FilterChain.{name}")
    
    def add_filter(self, filter_instance: BaseFilter) -> None:
        """Add filter to end of chain."""
        with self._lock:
            self.filters.append(filter_instance)
            self._logger.info(f"Added filter '{filter_instance.name}' to chain '{self.name}'")
    
    def remove_filter(self, filter_name: str) -> bool:
        """Remove filter from chain by name."""
        with self._lock:
            for i, filter_instance in enumerate(self.filters):
                if filter_instance.name == filter_name:
                    removed_filter = self.filters.pop(i)
                    self._logger.info(f"Removed filter '{removed_filter.name}' from chain '{self.name}'")
                    return True
            return False
    
    def process(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """
        Process message through filter chain.
        
        Args:
            message: LogMessage to process
            
        Returns:
            Tuple of (final_result, final_message)
        """
        current_message = message
        
        with self._lock:
            for filter_instance in self.filters:
                result, modified_message = filter_instance.filter(current_message)
                
                if result == FilterResult.REJECT:
                    return FilterResult.REJECT, None
                elif result == FilterResult.MODIFY and modified_message:
                    current_message = modified_message
                elif result == FilterResult.DEFER:
                    continue  # Let next filter decide
                # For ACCEPT, continue with current message
            
            # If we get here, all filters accepted or deferred
            return FilterResult.ACCEPT, current_message
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics for entire filter chain."""
        with self._lock:
            filter_stats = []
            for filter_instance in self.filters:
                filter_stats.append(filter_instance.get_performance_summary())
            
            return {
                'chain_name': self.name,
                'filter_count': len(self.filters),
                'filters': filter_stats
            }
    
    def enable_all(self) -> None:
        """Enable all filters in chain."""
        with self._lock:
            for filter_instance in self.filters:
                filter_instance.enable()
    
    def disable_all(self) -> None:
        """Disable all filters in chain."""
        with self._lock:
            for filter_instance in self.filters:
                filter_instance.disable()


# Module exports
__all__ = [
    'BaseFilter',
    'FilterResult',
    'FilterAction',
    'FilterStats',
    'FilterChain'
]