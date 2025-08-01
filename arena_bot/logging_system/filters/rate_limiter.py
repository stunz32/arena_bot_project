"""
Rate Limiter Filter for S-Tier Logging System.

This module provides intelligent rate limiting for log messages using token bucket
algorithms, adaptive rate adjustment, and burst handling to prevent log flooding
while preserving critical messages.

Features:
- Token bucket rate limiting with configurable burst capacity
- Per-logger and per-level rate limiting
- Adaptive rate adjustment based on system load
- Critical message bypass
- Statistical analysis and monitoring
- Thread-safe operations with high performance
"""

import time
import threading
import logging
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# Import from our components
from .base_filter import BaseFilter, FilterResult
from ..core.hybrid_async_queue import LogMessage


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"         # Standard token bucket algorithm
    SLIDING_WINDOW = "sliding_window"     # Sliding window counter
    FIXED_WINDOW = "fixed_window"         # Fixed time window counter
    LEAKY_BUCKET = "leaky_bucket"         # Leaky bucket algorithm
    ADAPTIVE = "adaptive"                 # Adaptive rate based on system load


class RateLimitScope(Enum):
    """Scope for rate limiting."""
    GLOBAL = "global"                     # Global rate limiting
    PER_LOGGER = "per_logger"            # Per logger name
    PER_LEVEL = "per_level"              # Per log level
    PER_MESSAGE = "per_message"          # Per unique message content
    PER_COMBINATION = "per_combination"   # Per logger+level combination


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: float                       # Maximum tokens
    tokens: float                        # Current tokens
    fill_rate: float                     # Tokens per second
    last_update: float                   # Last update timestamp
    
    def __post_init__(self):
        if self.tokens > self.capacity:
            self.tokens = self.capacity
    
    def consume(self, tokens: float = 1.0) -> bool:
        """Consume tokens from bucket."""
        self._update_tokens()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _update_tokens(self) -> None:
        """Update token count based on time elapsed."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Add tokens based on fill rate
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.fill_rate))
        self.last_update = current_time
    
    def get_wait_time(self, tokens: float = 1.0) -> float:
        """Get time to wait for tokens to be available."""
        self._update_tokens()
        
        if self.tokens >= tokens:
            return 0.0
        
        needed_tokens = tokens - self.tokens
        return needed_tokens / self.fill_rate
    
    def reset(self) -> None:
        """Reset bucket to full capacity."""
        self.tokens = self.capacity
        self.last_update = time.time()


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.GLOBAL
    
    # Token bucket parameters
    max_rate: float = 100.0              # Messages per second
    burst_capacity: float = 200.0        # Burst capacity
    
    # Sliding window parameters
    window_size_seconds: float = 60.0    # Window size for sliding window
    max_messages_per_window: int = 1000  # Max messages per window
    
    # Adaptive parameters
    enable_adaptive_rate: bool = True    # Enable adaptive rate adjustment
    min_rate: float = 10.0              # Minimum rate (messages/sec)
    max_adaptive_rate: float = 1000.0   # Maximum adaptive rate
    load_threshold_increase: float = 0.7 # Load threshold to increase rate
    load_threshold_decrease: float = 0.3 # Load threshold to decrease rate
    
    # Critical message bypass
    bypass_critical_messages: bool = True # Bypass rate limiting for critical messages
    bypass_error_messages: bool = True    # Bypass rate limiting for error messages
    bypass_patterns: List[str] = field(default_factory=list) # Message patterns to bypass
    
    # Per-scope overrides
    logger_overrides: Dict[str, float] = field(default_factory=dict) # Per-logger rates
    level_overrides: Dict[int, float] = field(default_factory=dict)  # Per-level rates


@dataclass
class RateLimitStats:
    """Rate limiting statistics."""
    scope_key: str
    messages_processed: int
    messages_allowed: int
    messages_dropped: int
    current_rate: float
    bucket_tokens: float
    bucket_capacity: float
    last_drop_time: Optional[float]
    total_wait_time: float


class RateLimiter(BaseFilter):
    """
    Intelligent rate limiting filter for log messages.
    
    Provides sophisticated rate limiting using token bucket algorithms,
    adaptive rate adjustment, and per-scope rate limiting to prevent
    log flooding while preserving critical messages.
    
    Features:
    - Multiple rate limiting strategies
    - Per-logger, per-level, and global rate limiting
    - Adaptive rate adjustment based on system load
    - Critical message bypass
    - Statistical monitoring and analysis
    """
    
    def __init__(self,
                 name: str = "rate_limiter",
                 config: Optional[RateLimitConfig] = None,
                 enable_monitoring: bool = True,
                 enable_statistics: bool = True,
                 cleanup_interval_seconds: float = 300.0,
                 max_buckets: int = 10000,
                 **base_kwargs):
        """
        Initialize rate limiter.
        
        Args:
            name: Filter name for identification
            config: Rate limiting configuration
            enable_monitoring: Enable performance monitoring
            enable_statistics: Enable detailed statistics
            cleanup_interval_seconds: Interval for cleaning up unused buckets
            max_buckets: Maximum number of token buckets to maintain
            **base_kwargs: Arguments for BaseFilter
        """
        super().__init__(name=name, **base_kwargs)
        
        self.config = config or RateLimitConfig()
        self.enable_monitoring = enable_monitoring
        self.enable_statistics = enable_statistics
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.max_buckets = max_buckets
        
        # Token buckets for different scopes
        self.buckets: Dict[str, TokenBucket] = {}
        self.bucket_lock = threading.RLock()
        
        # Statistics tracking
        self.stats: Dict[str, RateLimitStats] = {}
        self.stats_lock = threading.RLock()
        
        # Adaptive rate adjustment
        self.adaptive_rates: Dict[str, float] = {}
        self.load_history = deque(maxlen=100)  # System load history
        self.last_adaptive_adjustment = time.time()
        self.adaptive_lock = threading.RLock()
        
        # Message pattern matching for bypasses
        self._compile_bypass_patterns()
        
        # Cleanup tracking
        self.last_cleanup = time.time()
        self.bucket_access_times: Dict[str, float] = {}
        
        # Performance optimization
        self._scope_key_cache: Dict[tuple, str] = {}
        self._cache_lock = threading.RLock()
        self._cache_size_limit = 10000
        
        self._logger.info(f"RateLimiter '{name}' initialized",
                         extra={
                             'strategy': self.config.strategy.value,
                             'scope': self.config.scope.value,
                             'max_rate': self.config.max_rate,
                             'burst_capacity': self.config.burst_capacity,
                             'adaptive_enabled': self.config.enable_adaptive_rate
                         })
    
    def _apply_filter(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """Apply rate limiting logic."""
        # Check for bypass conditions first
        if self._should_bypass_rate_limit(message):
            return FilterResult.ACCEPT, message
        
        # Get scope key for this message
        scope_key = self._get_scope_key(message)
        
        # Get or create token bucket for this scope
        bucket = self._get_or_create_bucket(scope_key, message)
        
        # Apply adaptive rate adjustment if enabled
        if self.config.enable_adaptive_rate:
            self._apply_adaptive_adjustment(scope_key)
        
        # Try to consume token
        if bucket.consume(1.0):
            # Token consumed - message allowed
            self._update_stats(scope_key, message, allowed=True)
            return FilterResult.ACCEPT, message
        else:
            # No tokens available - message dropped
            self._update_stats(scope_key, message, allowed=False)
            return FilterResult.REJECT, None
    
    def _should_bypass_rate_limit(self, message: LogMessage) -> bool:
        """Check if message should bypass rate limiting."""
        # Critical messages bypass
        if self.config.bypass_critical_messages and message.level >= logging.CRITICAL:
            return True
        
        # Error messages bypass
        if self.config.bypass_error_messages and message.level >= logging.ERROR:
            return True
        
        # Pattern-based bypass
        if self.config.bypass_patterns:
            message_text = message.message.lower()
            for pattern in self._bypass_patterns_compiled:
                if pattern.search(message_text):
                    return True
        
        return False
    
    def _get_scope_key(self, message: LogMessage) -> str:
        """Get scope key for message based on rate limiting scope."""
        # Check cache first
        cache_key = (
            self.config.scope.value,
            message.logger_name,
            message.level,
            hash(message.message) if self.config.scope == RateLimitScope.PER_MESSAGE else 0
        )
        
        with self._cache_lock:
            if cache_key in self._scope_key_cache:
                return self._scope_key_cache[cache_key]
            
            # Limit cache size
            if len(self._scope_key_cache) >= self._cache_size_limit:
                # Remove oldest entries (simple FIFO)
                for _ in range(self._cache_size_limit // 4):
                    self._scope_key_cache.pop(next(iter(self._scope_key_cache)))
        
        # Generate scope key
        if self.config.scope == RateLimitScope.GLOBAL:
            scope_key = "global"
        elif self.config.scope == RateLimitScope.PER_LOGGER:
            scope_key = f"logger:{message.logger_name}"
        elif self.config.scope == RateLimitScope.PER_LEVEL:
            scope_key = f"level:{message.level}"
        elif self.config.scope == RateLimitScope.PER_MESSAGE:
            # Hash message content for unique scope
            message_hash = hashlib.md5(message.message.encode()).hexdigest()[:8]
            scope_key = f"message:{message_hash}"
        elif self.config.scope == RateLimitScope.PER_COMBINATION:
            scope_key = f"combo:{message.logger_name}:{message.level}"
        else:
            scope_key = "global"  # Fallback
        
        # Cache result
        with self._cache_lock:
            self._scope_key_cache[cache_key] = scope_key
        
        return scope_key
    
    def _get_or_create_bucket(self, scope_key: str, message: LogMessage) -> TokenBucket:
        """Get or create token bucket for scope."""
        with self.bucket_lock:
            if scope_key not in self.buckets:
                # Check bucket limit
                if len(self.buckets) >= self.max_buckets:
                    self._cleanup_old_buckets()
                
                # Get rate for this scope
                rate = self._get_rate_for_scope(scope_key, message)
                burst_capacity = max(rate * 2, self.config.burst_capacity)
                
                # Create new bucket
                self.buckets[scope_key] = TokenBucket(
                    capacity=burst_capacity,
                    tokens=burst_capacity,  # Start with full capacity
                    fill_rate=rate,
                    last_update=time.time()
                )
                
                # Initialize stats
                if self.enable_statistics:
                    with self.stats_lock:
                        self.stats[scope_key] = RateLimitStats(
                            scope_key=scope_key,
                            messages_processed=0,
                            messages_allowed=0,
                            messages_dropped=0,
                            current_rate=rate,
                            bucket_tokens=burst_capacity,
                            bucket_capacity=burst_capacity,
                            last_drop_time=None,
                            total_wait_time=0.0
                        )
            
            # Update access time for cleanup
            self.bucket_access_times[scope_key] = time.time()
            
            return self.buckets[scope_key]
    
    def _get_rate_for_scope(self, scope_key: str, message: LogMessage) -> float:
        """Get rate limit for specific scope."""
        base_rate = self.config.max_rate
        
        # Check for logger-specific override
        if self.config.scope == RateLimitScope.PER_LOGGER:
            logger_name = message.logger_name
            if logger_name in self.config.logger_overrides:
                base_rate = self.config.logger_overrides[logger_name]
        
        # Check for level-specific override
        elif self.config.scope == RateLimitScope.PER_LEVEL:
            if message.level in self.config.level_overrides:
                base_rate = self.config.level_overrides[message.level]
        
        # Apply adaptive adjustment if available
        if self.config.enable_adaptive_rate:
            with self.adaptive_lock:
                if scope_key in self.adaptive_rates:
                    adaptive_rate = self.adaptive_rates[scope_key]
                    base_rate = max(self.config.min_rate, 
                                  min(self.config.max_adaptive_rate, adaptive_rate))
        
        return base_rate
    
    def _apply_adaptive_adjustment(self, scope_key: str) -> None:
        """Apply adaptive rate adjustment based on system load."""
        current_time = time.time()
        
        # Only adjust periodically to avoid overhead
        if current_time - self.last_adaptive_adjustment < 5.0:  # Every 5 seconds
            return
        
        try:
            # Get system load (simplified - could use psutil for more accurate metrics)
            with self.adaptive_lock:
                # Estimate load based on bucket utilization
                bucket = self.buckets.get(scope_key)
                if not bucket:
                    return
                
                utilization = 1.0 - (bucket.tokens / bucket.capacity)
                self.load_history.append(utilization)
                
                # Calculate average load over recent history
                if len(self.load_history) < 10:
                    return
                
                avg_load = sum(self.load_history[-10:]) / 10
                
                # Get current adaptive rate or use base rate
                current_rate = self.adaptive_rates.get(scope_key, self.config.max_rate)
                
                # Adjust rate based on load
                if avg_load > self.config.load_threshold_increase:
                    # High load - decrease rate to reduce pressure
                    new_rate = current_rate * 0.8
                elif avg_load < self.config.load_threshold_decrease:
                    # Low load - increase rate to allow more messages
                    new_rate = current_rate * 1.2
                else:
                    new_rate = current_rate
                
                # Apply bounds
                new_rate = max(self.config.min_rate, 
                             min(self.config.max_adaptive_rate, new_rate))
                
                # Update if changed significantly
                if abs(new_rate - current_rate) / current_rate > 0.1:  # > 10% change
                    self.adaptive_rates[scope_key] = new_rate
                    
                    # Update bucket fill rate
                    bucket.fill_rate = new_rate
                    
                    self._logger.debug(f"Adaptive rate adjustment for {scope_key}",
                                     extra={
                                         'old_rate': current_rate,
                                         'new_rate': new_rate,
                                         'avg_load': avg_load,
                                         'utilization': utilization
                                     })
                
                self.last_adaptive_adjustment = current_time
                
        except Exception as e:
            self._logger.warning(f"Adaptive adjustment failed: {e}")
    
    def _update_stats(self, scope_key: str, message: LogMessage, allowed: bool) -> None:
        """Update statistics for rate limiting."""
        if not self.enable_statistics:
            return
        
        with self.stats_lock:
            if scope_key not in self.stats:
                return
            
            stats = self.stats[scope_key]
            stats.messages_processed += 1
            
            if allowed:
                stats.messages_allowed += 1
            else:
                stats.messages_dropped += 1
                stats.last_drop_time = time.time()
            
            # Update bucket info
            bucket = self.buckets.get(scope_key)
            if bucket:
                stats.bucket_tokens = bucket.tokens
                stats.current_rate = bucket.fill_rate
    
    def _cleanup_old_buckets(self) -> None:
        """Clean up old, unused token buckets."""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self.last_cleanup < self.cleanup_interval_seconds:
            return
        
        with self.bucket_lock:
            # Find buckets to remove (not accessed recently)
            cutoff_time = current_time - self.cleanup_interval_seconds
            keys_to_remove = []
            
            for scope_key in self.buckets:
                last_access = self.bucket_access_times.get(scope_key, 0)
                if last_access < cutoff_time:
                    keys_to_remove.append(scope_key)
            
            # Remove old buckets
            for scope_key in keys_to_remove:
                del self.buckets[scope_key]
                self.bucket_access_times.pop(scope_key, None)
                
                # Remove stats too
                if self.enable_statistics:
                    with self.stats_lock:
                        self.stats.pop(scope_key, None)
            
            if keys_to_remove:
                self._logger.debug(f"Cleaned up {len(keys_to_remove)} unused rate limit buckets")
            
            self.last_cleanup = current_time
    
    def _compile_bypass_patterns(self) -> None:
        """Compile bypass patterns for efficient matching."""
        import re
        
        self._bypass_patterns_compiled = []
        for pattern in self.config.bypass_patterns:
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                self._bypass_patterns_compiled.append(compiled_pattern)
            except re.error as e:
                self._logger.warning(f"Invalid bypass pattern '{pattern}': {e}")
    
    def add_logger_override(self, logger_name: str, rate: float) -> None:
        """Add rate override for specific logger."""
        self.config.logger_overrides[logger_name] = rate
        
        # Clear related buckets to force rate update
        with self.bucket_lock:
            keys_to_clear = [k for k in self.buckets.keys() if logger_name in k]
            for key in keys_to_clear:
                del self.buckets[key]
        
        self._logger.info(f"Added logger rate override",
                         extra={'logger_name': logger_name, 'rate': rate})
    
    def add_level_override(self, level: int, rate: float) -> None:
        """Add rate override for specific log level."""
        self.config.level_overrides[level] = rate
        
        # Clear related buckets to force rate update
        with self.bucket_lock:
            keys_to_clear = [k for k in self.buckets.keys() if f"level:{level}" in k]
            for key in keys_to_clear:
                del self.buckets[key]
        
        level_name = {
            10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"
        }.get(level, f"L{level}")
        
        self._logger.info(f"Added level rate override",
                         extra={'level': level, 'level_name': level_name, 'rate': rate})
    
    def add_bypass_pattern(self, pattern: str) -> None:
        """Add bypass pattern for messages."""
        self.config.bypass_patterns.append(pattern)
        self._compile_bypass_patterns()
        
        self._logger.info(f"Added bypass pattern: {pattern}")
    
    def set_global_rate(self, rate: float, burst_capacity: Optional[float] = None) -> None:
        """Set global rate limit."""
        self.config.max_rate = rate
        if burst_capacity is not None:
            self.config.burst_capacity = burst_capacity
        
        # Clear all buckets to force rate update
        with self.bucket_lock:
            self.buckets.clear()
            self.bucket_access_times.clear()
        
        # Clear stats
        if self.enable_statistics:
            with self.stats_lock:
                self.stats.clear()
        
        self._logger.info(f"Global rate limit updated",
                         extra={'rate': rate, 'burst_capacity': burst_capacity})
    
    def get_rate_limit_stats(self) -> List[RateLimitStats]:
        """Get rate limiting statistics for all scopes."""
        if not self.enable_statistics:
            return []
        
        with self.stats_lock:
            return list(self.stats.values())
    
    def get_bucket_info(self, scope_key: Optional[str] = None) -> Dict[str, Any]:
        """Get token bucket information."""
        with self.bucket_lock:
            if scope_key:
                bucket = self.buckets.get(scope_key)
                if bucket:
                    bucket._update_tokens()  # Update before reporting
                    return {
                        'scope_key': scope_key,
                        'tokens': bucket.tokens,
                        'capacity': bucket.capacity,
                        'fill_rate': bucket.fill_rate,
                        'utilization_percent': ((bucket.capacity - bucket.tokens) / bucket.capacity) * 100
                    }
                return {}
            else:
                # Return all buckets
                bucket_info = {}
                for key, bucket in self.buckets.items():
                    bucket._update_tokens()
                    bucket_info[key] = {
                        'tokens': bucket.tokens,
                        'capacity': bucket.capacity,
                        'fill_rate': bucket.fill_rate,
                        'utilization_percent': ((bucket.capacity - bucket.tokens) / bucket.capacity) * 100
                    }
                return bucket_info
    
    def reset_bucket(self, scope_key: str) -> bool:
        """Reset specific token bucket to full capacity."""
        with self.bucket_lock:
            bucket = self.buckets.get(scope_key)
            if bucket:
                bucket.reset()
                self._logger.info(f"Reset token bucket for scope: {scope_key}")
                return True
            return False
    
    def reset_all_buckets(self) -> int:
        """Reset all token buckets to full capacity."""
        with self.bucket_lock:
            count = 0
            for bucket in self.buckets.values():
                bucket.reset()
                count += 1
            
            if count > 0:
                self._logger.info(f"Reset {count} token buckets")
            
            return count
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get comprehensive filter summary."""
        base_summary = self.get_performance_summary()
        
        # Rate limiter specific information
        rate_summary = {
            **base_summary,
            'strategy': self.config.strategy.value,
            'scope': self.config.scope.value,
            'max_rate': self.config.max_rate,
            'burst_capacity': self.config.burst_capacity,
            'adaptive_enabled': self.config.enable_adaptive_rate,
            'bucket_count': len(self.buckets),
            'bypass_critical': self.config.bypass_critical_messages,
            'bypass_error': self.config.bypass_error_messages,
            'bypass_patterns': len(self.config.bypass_patterns),
            'logger_overrides': len(self.config.logger_overrides),
            'level_overrides': len(self.config.level_overrides)
        }
        
        # Add statistics if enabled
        if self.enable_statistics:
            stats = self.get_rate_limit_stats()
            total_processed = sum(s.messages_processed for s in stats)
            total_dropped = sum(s.messages_dropped for s in stats)
            
            rate_summary.update({
                'total_messages_processed': total_processed,
                'total_messages_dropped': total_dropped,
                'global_drop_rate_percent': (
                    (total_dropped / total_processed) * 100
                    if total_processed > 0 else 0
                ),
                'active_scopes': len(stats)
            })
        
        # Add adaptive rates if available
        if self.config.enable_adaptive_rate:
            with self.adaptive_lock:
                rate_summary['adaptive_rates'] = self.adaptive_rates.copy()
                rate_summary['load_history_size'] = len(self.load_history)
        
        return rate_summary


# Module exports
__all__ = [
    'RateLimiter',
    'RateLimitStrategy',
    'RateLimitScope',
    'RateLimitConfig',
    'RateLimitStats',
    'TokenBucket'
]