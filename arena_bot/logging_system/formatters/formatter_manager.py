"""
Formatter Manager for S-Tier Logging System.

This module provides centralized management of logging formatters with
dynamic configuration, performance monitoring, and formatter orchestration.

Features:
- Centralized formatter lifecycle management
- Dynamic formatter configuration and hot-reload
- Performance monitoring and optimization
- Multi-format output generation
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base_formatter import BaseFormatter
from .structured_formatter import StructuredFormatter
from .console_formatter import ConsoleFormatter
from .compression_formatter import CompressionFormatter


@dataclass
class FormatterManagerStats:
    """Statistics for the formatter manager."""
    
    active_formatters: int = 0
    total_formatted: int = 0
    total_errors: int = 0
    average_processing_time_ms: float = 0.0
    formatters_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.formatters_by_type is None:
            self.formatters_by_type = {}


class FormatterManager:
    """
    Centralized formatter management for the S-tier logging system.
    
    Manages formatter lifecycle, configuration, and orchestration with
    performance monitoring and dynamic reconfiguration capabilities.
    """
    
    def __init__(self):
        """Initialize formatter manager."""
        self.formatters: Dict[str, BaseFormatter] = {}
        self._logger = logging.getLogger(__name__)
        self._initialized = False
        self._stats = FormatterManagerStats()
    
    async def initialize(self, config: 'LoggingSystemConfig') -> None:
        """
        Initialize formatter manager with configuration.
        
        Args:
            config: Logging system configuration
        """
        if self._initialized:
            return
        
        try:
            # Initialize formatters based on configuration
            await self._initialize_formatters(config)
            
            self._initialized = True
            self._logger.info(f"Formatter manager initialized with {len(self.formatters)} formatters")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize formatter manager: {e}")
            raise
    
    async def _initialize_formatters(self, config: 'LoggingSystemConfig') -> None:
        """Initialize formatters from configuration."""
        # Add default formatters
        await self._add_default_formatters()
    
    async def _add_default_formatters(self) -> None:
        """Add default formatters."""
        # Add structured formatter
        structured = StructuredFormatter("structured")
        await structured.initialize()
        self.formatters["structured"] = structured
        
        # Add console formatter
        console = ConsoleFormatter("console") 
        await console.initialize()
        self.formatters["console"] = console
        
        # Add compression formatter
        compression = CompressionFormatter("compression")
        await compression.initialize()
        self.formatters["compression"] = compression
    
    async def format_record(self, record: 'LogRecord') -> List[Tuple['LogRecord', Dict[str, Any]]]:
        """
        Format log record using appropriate formatters.
        
        Args:
            record: Log record to format
            
        Returns:
            List of (record, formatted_data) tuples
        """
        if not self._initialized:
            return [(record, record.to_dict())]
        
        try:
            start_time = time.perf_counter()
            
            # Format with all active formatters
            formatted_results = []
            for formatter_name, formatter in self.formatters.items():
                if formatter.is_enabled():
                    formatted_data = await formatter.format(record)
                    formatted_results.append((record, formatted_data))
            
            # If no formatters produced output, use default
            if not formatted_results:
                formatted_results.append((record, record.to_dict()))
            
            # Update statistics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(processing_time, success=True)
            
            return formatted_results
            
        except Exception as e:
            self._logger.error(f"Formatter processing error: {e}")
            self._update_stats(0, success=False)
            # Return default format on error
            return [(record, record.to_dict())]
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        self._stats.total_formatted += 1
        
        if not success:
            self._stats.total_errors += 1
        
        # Update average processing time
        if self._stats.total_formatted == 1:
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
        Reconfigure formatters with new configuration.
        
        Args:
            config: New logging system configuration
        """
        try:
            # Shutdown existing formatters
            for formatter in self.formatters.values():
                if hasattr(formatter, 'shutdown'):
                    await formatter.shutdown()
            
            self.formatters.clear()
            
            # Reinitialize with new configuration
            await self._initialize_formatters(config)
            
            self._logger.info(f"Formatter manager reconfigured with {len(self.formatters)} formatters")
            
        except Exception as e:
            self._logger.error(f"Failed to reconfigure formatter manager: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get formatter performance statistics."""
        # Update formatter counts by type
        self._stats.formatters_by_type.clear()
        for formatter in self.formatters.values():
            formatter_type = formatter.__class__.__name__
            self._stats.formatters_by_type[formatter_type] = (
                self._stats.formatters_by_type.get(formatter_type, 0) + 1
            )
        
        self._stats.active_formatters = len(self.formatters)
        
        return {
            'active_formatters': self._stats.active_formatters,
            'total_formatted': self._stats.total_formatted,
            'total_errors': self._stats.total_errors,
            'error_rate': (
                self._stats.total_errors / self._stats.total_formatted
                if self._stats.total_formatted > 0 else 0
            ),
            'average_processing_time_ms': self._stats.average_processing_time_ms,
            'formatters_by_type': self._stats.formatters_by_type.copy()
        }
    
    def is_healthy(self) -> bool:
        """Check if formatter manager is healthy."""
        return (
            self._initialized and
            len(self.formatters) > 0 and
            self._stats.average_processing_time_ms < 5.0  # Less than 5ms average
        )
    
    async def shutdown(self) -> None:
        """Shutdown formatter manager."""
        try:
            # Shutdown all formatters
            for formatter in self.formatters.values():
                if hasattr(formatter, 'shutdown'):
                    await formatter.shutdown()
            
            self.formatters.clear()
            self._initialized = False
            
            self._logger.info("Formatter manager shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Formatter manager shutdown error: {e}")


# Module exports
__all__ = [
    'FormatterManager',
    'FormatterManagerStats'
]