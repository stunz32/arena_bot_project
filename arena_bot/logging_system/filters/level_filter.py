"""
Level Filter for S-Tier Logging System.

This module provides intelligent level-based filtering with dynamic thresholds,
contextual adjustments, and advanced filtering strategies for optimal
log message routing and noise reduction.

Features:
- Dynamic level threshold adjustment
- Context-aware level filtering
- Logger-specific level overrides
- Emergency level escalation
- Performance-optimized level checks
- Statistical level analysis
"""

import time
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# Import from our components
from .base_filter import BaseFilter, FilterResult
from ..core.hybrid_async_queue import LogMessage


class LevelFilterMode(Enum):
    """Level filtering modes."""
    FIXED = "fixed"                    # Fixed minimum level
    DYNAMIC = "dynamic"               # Dynamic level adjustment
    CONTEXTUAL = "contextual"         # Context-based level decisions
    ADAPTIVE = "adaptive"             # Machine learning-based adaptation


class LevelAdjustmentStrategy(Enum):
    """Strategies for dynamic level adjustment."""
    VOLUME_BASED = "volume_based"         # Adjust based on message volume
    ERROR_RATE_BASED = "error_rate_based" # Adjust based on error rates
    TIME_BASED = "time_based"            # Adjust based on time patterns
    LOAD_BASED = "load_based"            # Adjust based on system load


@dataclass
class LevelOverride:
    """Level override configuration for specific loggers."""
    logger_pattern: str               # Logger name pattern (supports wildcards)
    level: int                       # Override level
    priority: int = 1                # Override priority (higher wins)
    temporary: bool = False          # Temporary override
    expires_at: Optional[float] = None  # Expiration timestamp
    reason: str = ""                 # Reason for override


@dataclass
class LevelStats:
    """Statistics for level-based filtering."""
    level: int
    level_name: str
    count: int
    percentage: float
    recent_count: int
    trend: str  # "increasing", "decreasing", "stable"


class LevelFilter(BaseFilter):
    """
    Intelligent level-based message filter.
    
    Provides sophisticated level-based filtering with dynamic thresholds,
    contextual adjustments, and logger-specific overrides for optimal
    log message routing and noise reduction.
    
    Features:
    - Dynamic level threshold adjustment
    - Logger-specific level overrides
    - Context-aware filtering decisions
    - Emergency level escalation
    - Statistical analysis and trending
    """
    
    def __init__(self,
                 name: str = "level_filter",
                 min_level: int = logging.INFO,
                 mode: LevelFilterMode = LevelFilterMode.FIXED,
                 adjustment_strategy: LevelAdjustmentStrategy = LevelAdjustmentStrategy.VOLUME_BASED,
                 enable_overrides: bool = True,
                 enable_emergency_escalation: bool = True,
                 emergency_error_threshold: int = 10,
                 emergency_escalation_minutes: float = 5.0,
                 volume_adjustment_threshold: int = 1000,
                 **base_kwargs):
        """
        Initialize level filter.
        
        Args:
            name: Filter name for identification
            min_level: Minimum log level to accept
            mode: Level filtering mode
            adjustment_strategy: Dynamic adjustment strategy
            enable_overrides: Enable logger-specific overrides
            enable_emergency_escalation: Enable emergency escalation
            emergency_error_threshold: Error count threshold for escalation
            emergency_escalation_minutes: Duration for emergency escalation
            volume_adjustment_threshold: Message count for volume adjustment
            **base_kwargs: Arguments for BaseFilter
        """
        super().__init__(name=name, **base_kwargs)
        
        self.min_level = min_level
        self.mode = mode
        self.adjustment_strategy = adjustment_strategy
        self.enable_overrides = enable_overrides
        self.enable_emergency_escalation = enable_emergency_escalation
        self.emergency_error_threshold = emergency_error_threshold
        self.emergency_escalation_minutes = emergency_escalation_minutes
        self.volume_adjustment_threshold = volume_adjustment_threshold
        
        # Dynamic state
        self.current_level = min_level
        self.level_adjustment_factor = 0  # Positive = stricter, negative = looser
        
        # Level overrides
        self.overrides: List[LevelOverride] = []
        self.override_cache: Dict[str, int] = {}  # Logger name -> effective level
        self.override_lock = threading.RLock()
        
        # Emergency escalation
        self.emergency_active = False
        self.emergency_start_time: Optional[float] = None
        self.error_count_window = deque(maxlen=emergency_error_threshold)
        
        # Statistics and analysis
        self.level_counts: Dict[int, int] = defaultdict(int)
        self.logger_level_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.recent_messages = deque(maxlen=1000)  # Recent message levels for analysis
        self.analysis_lock = threading.RLock()
        
        # Level names mapping
        self.level_names = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL"
        }
        
        # Performance optimization
        self._level_check_cache: Dict[tuple, bool] = {}
        self._cache_lock = threading.RLock()
        self._cache_size_limit = 10000
        
        self._logger.info(f"LevelFilter '{name}' initialized",
                         extra={
                             'min_level': min_level,
                             'level_name': self.level_names.get(min_level, f"L{min_level}"),
                             'mode': mode.value,
                             'adjustment_strategy': adjustment_strategy.value,
                             'enable_overrides': enable_overrides
                         })
    
    def _apply_filter(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """Apply level-based filtering logic."""
        # Get effective level for this message
        effective_level = self._get_effective_level(message)
        
        # Update statistics
        self._update_level_stats(message)
        
        # Check emergency escalation
        if self.enable_emergency_escalation:
            self._check_emergency_escalation(message)
        
        # Apply dynamic adjustment if enabled
        if self.mode != LevelFilterMode.FIXED:
            self._apply_dynamic_adjustment()
        
        # Make filtering decision
        if message.level >= effective_level:
            return FilterResult.ACCEPT, message
        else:
            return FilterResult.REJECT, None
    
    def _get_effective_level(self, message: LogMessage) -> int:
        """Get effective level for message considering overrides and adjustments."""
        # Check cache first for performance
        cache_key = (message.logger_name, self.current_level, self.level_adjustment_factor)
        
        with self._cache_lock:
            if cache_key in self._level_check_cache:
                return self._level_check_cache[cache_key]
            
            # Limit cache size
            if len(self._level_check_cache) >= self._cache_size_limit:
                # Remove oldest entries (simple FIFO)
                for _ in range(self._cache_size_limit // 4):
                    self._level_check_cache.pop(next(iter(self._level_check_cache)))
        
        # Calculate effective level
        effective_level = self.current_level
        
        # Apply logger-specific overrides
        if self.enable_overrides:
            override_level = self._get_override_level(message.logger_name)
            if override_level is not None:
                effective_level = override_level
        
        # Apply level adjustment factor
        effective_level += self.level_adjustment_factor * 10  # Each factor = 10 level points
        
        # Emergency escalation overrides everything
        if self.emergency_active:
            effective_level = min(effective_level, logging.WARNING)
        
        # Ensure level is within valid range
        effective_level = max(logging.DEBUG, min(logging.CRITICAL, effective_level))
        
        # Cache result
        with self._cache_lock:
            self._level_check_cache[cache_key] = effective_level
        
        return effective_level
    
    def _get_override_level(self, logger_name: str) -> Optional[int]:
        """Get override level for specific logger."""
        with self.override_lock:
            # Check cache first
            if logger_name in self.override_cache:
                return self.override_cache[logger_name]
            
            # Find matching override with highest priority
            best_override = None
            best_priority = -1
            current_time = time.time()
            
            for override in self.overrides:
                # Check if override has expired
                if override.expires_at and current_time > override.expires_at:
                    continue
                
                # Check if logger matches pattern
                if self._matches_logger_pattern(logger_name, override.logger_pattern):
                    if override.priority > best_priority:
                        best_override = override
                        best_priority = override.priority
            
            # Cache result
            level = best_override.level if best_override else None
            self.override_cache[logger_name] = level
            
            return level
    
    def _matches_logger_pattern(self, logger_name: str, pattern: str) -> bool:
        """Check if logger name matches pattern (supports simple wildcards)."""
        if pattern == "*":
            return True
        
        if "*" not in pattern:
            return logger_name == pattern
        
        # Simple wildcard matching
        parts = pattern.split("*")
        if len(parts) == 2:
            prefix, suffix = parts
            return logger_name.startswith(prefix) and logger_name.endswith(suffix)
        
        # More complex patterns could be implemented here
        return logger_name == pattern
    
    def _update_level_stats(self, message: LogMessage) -> None:
        """Update level statistics."""
        with self.analysis_lock:
            # Update overall level counts
            self.level_counts[message.level] += 1
            
            # Update per-logger level counts
            self.logger_level_counts[message.logger_name][message.level] += 1
            
            # Add to recent messages for trend analysis
            self.recent_messages.append({
                'level': message.level,
                'logger': message.logger_name,
                'timestamp': message.timestamp
            })
    
    def _check_emergency_escalation(self, message: LogMessage) -> None:
        """Check if emergency escalation should be activated."""
        current_time = time.time()
        
        # Track error messages in time window
        if message.level >= logging.ERROR:
            self.error_count_window.append(current_time)
        
        # Remove old entries from window
        window_start = current_time - (self.emergency_escalation_minutes * 60)
        while self.error_count_window and self.error_count_window[0] < window_start:
            self.error_count_window.popleft()
        
        # Check if we should activate emergency mode
        if not self.emergency_active and len(self.error_count_window) >= self.emergency_error_threshold:
            self._activate_emergency_escalation()
        
        # Check if we should deactivate emergency mode
        elif self.emergency_active and len(self.error_count_window) < self.emergency_error_threshold // 2:
            self._deactivate_emergency_escalation()
    
    def _activate_emergency_escalation(self) -> None:
        """Activate emergency level escalation."""
        self.emergency_active = True
        self.emergency_start_time = time.time()
        
        # Clear level cache since effective levels will change
        with self._cache_lock:
            self._level_check_cache.clear()
        
        self._logger.warning(f"Emergency escalation activated for filter '{self.name}'",
                           extra={
                               'error_count': len(self.error_count_window),
                               'threshold': self.emergency_error_threshold,
                               'escalation_level': 'WARNING'
                           })
    
    def _deactivate_emergency_escalation(self) -> None:
        """Deactivate emergency level escalation."""
        duration = time.time() - (self.emergency_start_time or 0)
        
        self.emergency_active = False
        self.emergency_start_time = None
        
        # Clear level cache
        with self._cache_lock:
            self._level_check_cache.clear()
        
        self._logger.info(f"Emergency escalation deactivated for filter '{self.name}'",
                         extra={
                             'duration_minutes': duration / 60,
                             'final_error_count': len(self.error_count_window)
                         })
    
    def _apply_dynamic_adjustment(self) -> None:
        """Apply dynamic level adjustment based on strategy."""
        if self.adjustment_strategy == LevelAdjustmentStrategy.VOLUME_BASED:
            self._apply_volume_based_adjustment()
        elif self.adjustment_strategy == LevelAdjustmentStrategy.ERROR_RATE_BASED:
            self._apply_error_rate_adjustment()
        elif self.adjustment_strategy == LevelAdjustmentStrategy.TIME_BASED:
            self._apply_time_based_adjustment()
        elif self.adjustment_strategy == LevelAdjustmentStrategy.LOAD_BASED:
            self._apply_load_based_adjustment()
    
    def _apply_volume_based_adjustment(self) -> None:
        """Adjust level based on message volume."""
        with self.analysis_lock:
            total_messages = sum(self.level_counts.values())
            
            if total_messages > self.volume_adjustment_threshold:
                # High volume - increase filtering (raise level)
                if self.level_adjustment_factor < 2:
                    self.level_adjustment_factor += 1
                    self._clear_level_cache()
                    self._logger.info(f"Volume-based level adjustment: increased strictness",
                                    extra={'total_messages': total_messages,
                                          'adjustment_factor': self.level_adjustment_factor})
            elif total_messages < self.volume_adjustment_threshold // 4:
                # Low volume - decrease filtering (lower level)
                if self.level_adjustment_factor > -1:
                    self.level_adjustment_factor -= 1
                    self._clear_level_cache()
                    self._logger.info(f"Volume-based level adjustment: decreased strictness",
                                    extra={'total_messages': total_messages,
                                          'adjustment_factor': self.level_adjustment_factor})
    
    def _apply_error_rate_adjustment(self) -> None:
        """Adjust level based on error rate."""
        with self.analysis_lock:
            total_messages = sum(self.level_counts.values())
            error_messages = self.level_counts[logging.ERROR] + self.level_counts[logging.CRITICAL]
            
            if total_messages > 0:
                error_rate = error_messages / total_messages
                
                if error_rate > 0.1:  # > 10% errors
                    # High error rate - decrease filtering to capture more context
                    if self.level_adjustment_factor > -2:
                        self.level_adjustment_factor -= 1
                        self._clear_level_cache()
                        self._logger.info(f"Error rate adjustment: decreased strictness",
                                        extra={'error_rate_percent': error_rate * 100,
                                              'adjustment_factor': self.level_adjustment_factor})
                elif error_rate < 0.01:  # < 1% errors
                    # Low error rate - increase filtering
                    if self.level_adjustment_factor < 1:
                        self.level_adjustment_factor += 1
                        self._clear_level_cache()
                        self._logger.info(f"Error rate adjustment: increased strictness",
                                        extra={'error_rate_percent': error_rate * 100,
                                              'adjustment_factor': self.level_adjustment_factor})
    
    def _apply_time_based_adjustment(self) -> None:
        """Adjust level based on time patterns (e.g., business hours)."""
        # This could implement business hours logic, weekend adjustments, etc.
        # For now, implement a simple day/night cycle
        current_time = time.localtime()
        hour = current_time.tm_hour
        
        # Business hours (9 AM - 5 PM): stricter filtering
        # Off hours: looser filtering to catch issues
        if 9 <= hour <= 17:
            target_adjustment = 1  # Stricter during business hours
        else:
            target_adjustment = -1  # Looser during off hours
        
        if self.level_adjustment_factor != target_adjustment:
            self.level_adjustment_factor = target_adjustment
            self._clear_level_cache()
            self._logger.info(f"Time-based level adjustment",
                            extra={'hour': hour,
                                  'adjustment_factor': self.level_adjustment_factor,
                                  'period': 'business_hours' if 9 <= hour <= 17 else 'off_hours'})
    
    def _apply_load_based_adjustment(self) -> None:
        """Adjust level based on system load."""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # High load - increase filtering to reduce processing
            if cpu_percent > 80 or memory_percent > 85:
                if self.level_adjustment_factor < 2:
                    self.level_adjustment_factor += 1
                    self._clear_level_cache()
                    self._logger.info(f"Load-based level adjustment: increased strictness",
                                    extra={'cpu_percent': cpu_percent,
                                          'memory_percent': memory_percent,
                                          'adjustment_factor': self.level_adjustment_factor})
            
            # Low load - decrease filtering
            elif cpu_percent < 30 and memory_percent < 50:
                if self.level_adjustment_factor > -1:
                    self.level_adjustment_factor -= 1
                    self._clear_level_cache()
                    self._logger.info(f"Load-based level adjustment: decreased strictness",
                                    extra={'cpu_percent': cpu_percent,
                                          'memory_percent': memory_percent,
                                          'adjustment_factor': self.level_adjustment_factor})
                    
        except ImportError:
            # psutil not available, skip load-based adjustment
            pass
        except Exception as e:
            self._logger.warning(f"Load-based adjustment failed: {e}")
    
    def _clear_level_cache(self) -> None:
        """Clear the level check cache."""
        with self._cache_lock:
            self._level_check_cache.clear()
    
    def add_override(self, override: LevelOverride) -> None:
        """Add logger-specific level override."""
        with self.override_lock:
            # Remove any existing override for the same pattern
            self.overrides = [o for o in self.overrides if o.logger_pattern != override.logger_pattern]
            
            # Add new override
            self.overrides.append(override)
            
            # Clear override cache
            self.override_cache.clear()
            
            # Clear level cache
            self._clear_level_cache()
            
            self._logger.info(f"Added level override",
                            extra={
                                'logger_pattern': override.logger_pattern,
                                'level': override.level,
                                'level_name': self.level_names.get(override.level, f"L{override.level}"),
                                'priority': override.priority,
                                'temporary': override.temporary,
                                'reason': override.reason
                            })
    
    def remove_override(self, logger_pattern: str) -> bool:
        """Remove logger-specific level override."""
        with self.override_lock:
            initial_count = len(self.overrides)
            self.overrides = [o for o in self.overrides if o.logger_pattern != logger_pattern]
            removed = len(self.overrides) < initial_count
            
            if removed:
                # Clear caches
                self.override_cache.clear()
                self._clear_level_cache()
                
                self._logger.info(f"Removed level override",
                                extra={'logger_pattern': logger_pattern})
            
            return removed
    
    def set_level(self, level: int) -> None:
        """Set minimum level threshold."""
        old_level = self.current_level
        self.current_level = level
        
        # Clear cache since levels changed
        self._clear_level_cache()
        
        self._logger.info(f"Level threshold changed",
                         extra={
                             'old_level': old_level,
                             'new_level': level,
                             'old_level_name': self.level_names.get(old_level, f"L{old_level}"),
                             'new_level_name': self.level_names.get(level, f"L{level}")
                         })
    
    def get_level_stats(self) -> List[LevelStats]:
        """Get detailed level statistics."""
        with self.analysis_lock:
            total_messages = max(sum(self.level_counts.values()), 1)  # Avoid division by zero
            
            # Get recent messages for trend analysis
            recent_cutoff = time.time() - 300  # Last 5 minutes
            recent_counts = defaultdict(int)
            
            for msg in self.recent_messages:
                if msg['timestamp'] >= recent_cutoff:
                    recent_counts[msg['level']] += 1
            
            # Calculate trends (simplified)
            stats = []
            for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
                count = self.level_counts[level]
                recent_count = recent_counts[level]
                
                # Simple trend calculation
                if recent_count > count * 0.7:  # Recent activity is high
                    trend = "increasing"
                elif recent_count < count * 0.3:  # Recent activity is low
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                stats.append(LevelStats(
                    level=level,
                    level_name=self.level_names.get(level, f"L{level}"),
                    count=count,
                    percentage=(count / total_messages) * 100,
                    recent_count=recent_count,
                    trend=trend
                ))
            
            return stats
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get comprehensive filter summary."""
        base_summary = self.get_performance_summary()
        
        level_stats = self.get_level_stats()
        
        summary = {
            **base_summary,
            'current_level': self.current_level,
            'current_level_name': self.level_names.get(self.current_level, f"L{self.current_level}"),
            'mode': self.mode.value,
            'adjustment_strategy': self.adjustment_strategy.value,
            'level_adjustment_factor': self.level_adjustment_factor,
            'emergency_active': self.emergency_active,
            'emergency_duration_minutes': (
                (time.time() - self.emergency_start_time) / 60
                if self.emergency_start_time else 0
            ),
            'override_count': len(self.overrides),
            'level_statistics': [stat.to_dict() if hasattr(stat, 'to_dict') else {
                'level': stat.level,
                'level_name': stat.level_name,
                'count': stat.count,
                'percentage': stat.percentage,
                'recent_count': stat.recent_count,
                'trend': stat.trend
            } for stat in level_stats],
            'cache_size': len(self._level_check_cache)
        }
        
        return summary


# Module exports
__all__ = [
    'LevelFilter',
    'LevelFilterMode',
    'LevelAdjustmentStrategy',
    'LevelOverride',
    'LevelStats'
]