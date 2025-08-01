"""
Filters for the S-Tier Logging System.

This module contains filters that process and route log messages based on
various criteria including level, rate limiting, correlation tracking,
and security considerations.

Filters:
- BaseFilter: Abstract base class for all filters with performance tracking
- LevelFilter: Smart level-based routing with dynamic thresholds
- RateLimiter: Prevent log flooding with token bucket algorithm
- CorrelationFilter: Correlation ID management and distributed tracing
- SecurityFilter: PII detection and credential scrubbing for security
"""

# Import base filter components
from .base_filter import BaseFilter, FilterResult, FilterAction, FilterStats, FilterChain
from .filter_manager import FilterManager, FilterManagerStats

# Import concrete filter implementations
from .level_filter import (
    LevelFilter, 
    LevelFilterMode, 
    LevelAdjustmentStrategy, 
    LevelOverride, 
    LevelStats
)
from .rate_limiter import (
    RateLimiter,
    RateLimitStrategy,
    RateLimitScope,
    RateLimitConfig,
    RateLimitStats,
    TokenBucket
)
from .correlation_filter import (
    CorrelationFilter,
    CorrelationStrategy,
    CorrelationScope,
    CorrelationRouting,
    CorrelationContext,
    CorrelationRule,
    CorrelationConfig,
    CorrelationStats
)
from .security_filter import (
    SecurityFilter,
    SecurityAction,
    DataClassification,
    PIIType,
    SecurityPattern,
    SecurityConfig,
    SecurityEvent
)

__all__ = [
    # Base filter components
    'BaseFilter',
    'FilterResult',
    'FilterAction', 
    'FilterStats',
    'FilterChain',
    'FilterManager',
    'FilterManagerStats',
    
    # Level filter
    'LevelFilter',
    'LevelFilterMode',
    'LevelAdjustmentStrategy',
    'LevelOverride',
    'LevelStats',
    
    # Rate limiter
    'RateLimiter',
    'RateLimitStrategy',
    'RateLimitScope',
    'RateLimitConfig',
    'RateLimitStats',
    'TokenBucket',
    
    # Correlation filter
    'CorrelationFilter',
    'CorrelationStrategy',
    'CorrelationScope',
    'CorrelationRouting',
    'CorrelationContext',
    'CorrelationRule',
    'CorrelationConfig',
    'CorrelationStats',
    
    # Security filter
    'SecurityFilter',
    'SecurityAction',
    'DataClassification',
    'PIIType',
    'SecurityPattern',
    'SecurityConfig',
    'SecurityEvent'
]