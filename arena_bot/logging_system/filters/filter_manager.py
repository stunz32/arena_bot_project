"""
Filter Manager for S-Tier Logging System.

This module provides centralized management of logging filters with
dynamic configuration, performance monitoring, and filter orchestration.

Features:
- Centralized filter lifecycle management
- Dynamic filter configuration and hot-reload
- Performance monitoring and optimization
- Filter chain composition and execution
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .base_filter import BaseFilter, FilterChain, FilterResult, FilterAction
from .level_filter import LevelFilter
from .rate_limiter import RateLimiter
from .correlation_filter import CorrelationFilter
from .security_filter import SecurityFilter


@dataclass
class FilterManagerStats:
    """Statistics for the filter manager."""
    
    active_filters: int = 0
    total_processed: int = 0
    total_filtered: int = 0
    average_processing_time_ms: float = 0.0
    filters_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.filters_by_type is None:
            self.filters_by_type = {}


class FilterManager:
    """
    Centralized filter management for the S-tier logging system.
    
    Manages filter lifecycle, configuration, and orchestration with
    performance monitoring and dynamic reconfiguration capabilities.
    """
    
    def __init__(self):
        """Initialize filter manager."""
        self.filters: List[BaseFilter] = []
        self.filter_chain: Optional[FilterChain] = None
        self._logger = logging.getLogger(__name__)
        self._initialized = False
        self._stats = FilterManagerStats()
    
    async def initialize(self, config: 'LoggingSystemConfig') -> None:
        """
        Initialize filter manager with configuration.
        
        Args:
            config: Logging system configuration
        """
        if self._initialized:
            return
        
        try:
            # Initialize filters based on configuration
            await self._initialize_filters(config)
            
            # Build filter chain
            self._build_filter_chain()
            
            self._initialized = True
            self._logger.info(f"Filter manager initialized with {len(self.filters)} filters")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize filter manager: {e}")
            raise
    
    async def _initialize_filters(self, config: 'LoggingSystemConfig') -> None:
        """Initialize filters from configuration."""
        # Add default filters
        if hasattr(config, 'filters') and config.filters:
            for filter_name, filter_config in config.filters.items():
                await self._create_filter(filter_name, filter_config)
        else:
            # Add basic default filters
            await self._add_default_filters()
    
    async def _add_default_filters(self) -> None:
        """Add default filters when no configuration is provided."""
        # Add basic level filter
        level_filter = LevelFilter("default_level")
        await level_filter.initialize()
        self.filters.append(level_filter)
        
        # Add basic rate limiter
        rate_limiter = RateLimiter("default_rate_limit")
        await rate_limiter.initialize()
        self.filters.append(rate_limiter)
    
    async def _create_filter(self, name: str, config: Any) -> None:
        """Create filter from configuration."""
        try:
            filter_type = getattr(config, 'type', 'level')
            
            if filter_type == 'level':
                filter_instance = LevelFilter(name)
            elif filter_type == 'rate_limit':
                filter_instance = RateLimiter(name)
            elif filter_type == 'correlation':
                filter_instance = CorrelationFilter(name)
            elif filter_type == 'security':
                filter_instance = SecurityFilter(name)
            else:
                self._logger.warning(f"Unknown filter type: {filter_type}")
                return
            
            await filter_instance.initialize(config)
            self.filters.append(filter_instance)
            
        except Exception as e:
            self._logger.error(f"Failed to create filter {name}: {e}")
    
    def _build_filter_chain(self) -> None:
        """Build filter execution chain."""
        if self.filters:
            self.filter_chain = FilterChain(self.filters)
        else:
            self.filter_chain = None
    
    async def should_process(self, record: 'LogRecord') -> bool:
        """
        Check if log record should be processed.
        
        Args:
            record: Log record to check
            
        Returns:
            True if record should be processed
        """
        if not self._initialized or not self.filter_chain:
            return True
        
        try:
            start_time = time.perf_counter()
            
            # Apply filter chain
            result = await self.filter_chain.apply(record)
            
            # Update statistics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(result, processing_time)
            
            return result.action == FilterAction.PASS
            
        except Exception as e:
            self._logger.error(f"Filter processing error: {e}")
            # Default to allowing the record on error
            return True
    
    def _update_stats(self, result: FilterResult, processing_time: float) -> None:
        """Update processing statistics."""
        self._stats.total_processed += 1
        
        if result.action != FilterAction.PASS:
            self._stats.total_filtered += 1
        
        # Update average processing time
        if self._stats.total_processed == 1:
            self._stats.average_processing_time_ms = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self._stats.average_processing_time_ms = (
                alpha * processing_time + 
                (1 - alpha) * self._stats.average_processing_time_ms
            )
    
    async def reconfigure(self, config: 'LoggingSystemConfig') -> None:
        """
        Reconfigure filters with new configuration.
        
        Args:
            config: New logging system configuration
        """
        try:
            # Clear existing filters
            self.filters.clear()
            self.filter_chain = None
            
            # Reinitialize with new configuration
            await self._initialize_filters(config)
            self._build_filter_chain()
            
            self._logger.info(f"Filter manager reconfigured with {len(self.filters)} filters")
            
        except Exception as e:
            self._logger.error(f"Failed to reconfigure filter manager: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get filter performance statistics."""
        # Update filter counts by type
        self._stats.filters_by_type.clear()
        for filter_instance in self.filters:
            filter_type = filter_instance.__class__.__name__
            self._stats.filters_by_type[filter_type] = (
                self._stats.filters_by_type.get(filter_type, 0) + 1
            )
        
        self._stats.active_filters = len(self.filters)
        
        return {
            'active_filters': self._stats.active_filters,
            'total_processed': self._stats.total_processed,
            'total_filtered': self._stats.total_filtered,
            'filter_rate': (
                self._stats.total_filtered / self._stats.total_processed
                if self._stats.total_processed > 0 else 0
            ),
            'average_processing_time_ms': self._stats.average_processing_time_ms,
            'filters_by_type': self._stats.filters_by_type.copy()
        }
    
    def is_healthy(self) -> bool:
        """Check if filter manager is healthy."""
        return (
            self._initialized and
            self._stats.average_processing_time_ms < 10.0  # Less than 10ms average
        )
    
    async def shutdown(self) -> None:
        """Shutdown filter manager."""
        try:
            # Shutdown all filters
            for filter_instance in self.filters:
                if hasattr(filter_instance, 'shutdown'):
                    await filter_instance.shutdown()
            
            self.filters.clear()
            self.filter_chain = None
            self._initialized = False
            
            self._logger.info("Filter manager shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Filter manager shutdown error: {e}")


# Module exports
__all__ = [
    'FilterManager',
    'FilterManagerStats'
]