"""
Base Formatter for S-Tier Logging System.

This module provides the abstract base class for all formatters with
common functionality, performance tracking, and consistent interface.

Features:
- Abstract base class for all formatters
- Common performance tracking and statistics
- Consistent formatter interface and lifecycle
- Error handling and fallback formatting
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

# Import from our core components
from ..core.hybrid_async_queue import LogMessage


@dataclass
class FormatterStats:
    """Statistics for formatter performance tracking."""
    
    total_formatted: int = 0
    total_errors: int = 0
    average_processing_time_us: float = 0.0
    peak_processing_time_us: float = 0.0
    enabled: bool = True


class BaseFormatter(ABC):
    """
    Abstract base class for all formatters in the S-tier logging system.
    
    Provides common functionality including performance tracking, error handling,
    and consistent lifecycle management for all formatter implementations.
    """
    
    def __init__(self, name: str):
        """
        Initialize base formatter.
        
        Args:
            name: Unique name for this formatter instance
        """
        self.name = name
        self._logger = logging.getLogger(f"{__name__}.{name}")
        self._initialized = False
        self._enabled = True
        self._stats = FormatterStats()
    
    async def initialize(self, config: Optional[Any] = None) -> None:
        """
        Initialize formatter with optional configuration.
        
        Args:
            config: Optional formatter-specific configuration
        """
        if self._initialized:
            return
        
        try:
            await self._initialize_formatter(config)
            self._initialized = True
            self._logger.info(f"Formatter {self.name} initialized")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize formatter {self.name}: {e}")
            raise
    
    async def _initialize_formatter(self, config: Optional[Any] = None) -> None:
        """
        Formatter-specific initialization logic.
        Override in subclasses for custom initialization.
        
        Args:
            config: Optional formatter-specific configuration
        """
        pass
    
    @abstractmethod
    async def format(self, record: Union[LogMessage, 'LogRecord']) -> Dict[str, Any]:
        """
        Format log record into target format.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted data as dictionary
        """
        pass
    
    async def format_with_stats(self, record: Union[LogMessage, 'LogRecord']) -> Dict[str, Any]:
        """
        Format record with performance tracking.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted data as dictionary
        """
        if not self._enabled:
            return self._get_disabled_format(record)
        
        start_time = time.perf_counter()
        
        try:
            result = await self.format(record)
            
            # Update performance statistics
            processing_time_us = (time.perf_counter() - start_time) * 1_000_000
            self._update_stats(processing_time_us, success=True)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Formatter {self.name} error: {e}")
            self._update_stats(0, success=False)
            
            # Return fallback format
            return self._get_fallback_format(record, e)
    
    def _update_stats(self, processing_time_us: float, success: bool) -> None:
        """Update performance statistics."""
        self._stats.total_formatted += 1
        
        if not success:
            self._stats.total_errors += 1
            return
        
        # Update peak processing time
        if processing_time_us > self._stats.peak_processing_time_us:
            self._stats.peak_processing_time_us = processing_time_us
        
        # Update average processing time (exponential moving average)
        if self._stats.total_formatted == 1:
            self._stats.average_processing_time_us = processing_time_us
        else:
            alpha = 0.1
            self._stats.average_processing_time_us = (
                alpha * processing_time_us + 
                (1 - alpha) * self._stats.average_processing_time_us
            )
    
    def _get_disabled_format(self, record: Union[LogMessage, 'LogRecord']) -> Dict[str, Any]:
        """Return format when formatter is disabled."""
        return {
            'message': 'Formatter disabled',
            'formatter': self.name,
            'timestamp': time.time()
        }
    
    def _get_fallback_format(self, record: Union[LogMessage, 'LogRecord'], error: Exception) -> Dict[str, Any]:
        """Return fallback format on error."""
        # Try to extract basic info from record
        try:
            if hasattr(record, 'to_dict'):
                base_data = record.to_dict()
            elif hasattr(record, '__dict__'):
                base_data = record.__dict__.copy()
            else:
                base_data = {'message': str(record)}
            
            base_data.update({
                'formatter_error': str(error),
                'formatter': self.name,
                'fallback_format': True
            })
            
            return base_data
            
        except Exception:
            # Ultimate fallback
            return {
                'message': str(record),
                'formatter_error': str(error),
                'formatter': self.name,
                'fallback_format': True,
                'timestamp': time.time()
            }
    
    def is_enabled(self) -> bool:
        """Check if formatter is enabled."""
        return self._enabled and self._initialized
    
    def enable(self) -> None:
        """Enable formatter."""
        self._enabled = True
        self._stats.enabled = True
    
    def disable(self) -> None:
        """Disable formatter."""
        self._enabled = False
        self._stats.enabled = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get formatter performance statistics."""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'enabled': self._enabled,
            'total_formatted': self._stats.total_formatted,
            'total_errors': self._stats.total_errors,
            'error_rate': (
                self._stats.total_errors / self._stats.total_formatted
                if self._stats.total_formatted > 0 else 0
            ),
            'average_processing_time_us': self._stats.average_processing_time_us,
            'peak_processing_time_us': self._stats.peak_processing_time_us
        }
    
    async def shutdown(self) -> None:
        """Shutdown formatter and cleanup resources."""
        try:
            await self._shutdown_formatter()
            self._initialized = False
            self._enabled = False
            self._logger.info(f"Formatter {self.name} shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Formatter {self.name} shutdown error: {e}")
    
    async def _shutdown_formatter(self) -> None:
        """
        Formatter-specific shutdown logic.
        Override in subclasses for custom cleanup.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of formatter."""
        return f"{self.__class__.__name__}(name={self.name}, enabled={self._enabled})"
    
    def __repr__(self) -> str:
        """Detailed representation of formatter."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"enabled={self._enabled}, "
            f"initialized={self._initialized}, "
            f"formatted={self._stats.total_formatted}, "
            f"errors={self._stats.total_errors})"
        )


# Module exports
__all__ = [
    'BaseFormatter',
    'FormatterStats'
]