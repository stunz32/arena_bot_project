"""
Sink Manager for S-Tier Logging System.

This module provides centralized management of logging sinks with
dynamic configuration, performance monitoring, and sink orchestration.

Features:
- Centralized sink lifecycle management
- Dynamic sink configuration and hot-reload
- Performance monitoring and optimization
- Sink routing and load balancing
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .base_sink import BaseSink
from .console_sink import ConsoleSink
from .tiered_file_sink import TieredFileSink
from .network_sink import NetworkSink
from .metrics_sink import MetricsSink
from .emergency_sink import EmergencySink


@dataclass
class SinkManagerStats:
    """Statistics for the sink manager."""
    
    active_sinks: int = 0
    total_messages: int = 0
    total_errors: int = 0
    average_processing_time_ms: float = 0.0
    sinks_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.sinks_by_type is None:
            self.sinks_by_type = {}


class SinkManager:
    """
    Centralized sink management for the S-tier logging system.
    
    Manages sink lifecycle, configuration, and orchestration with
    performance monitoring and dynamic reconfiguration capabilities.
    """
    
    def __init__(self):
        """Initialize sink manager."""
        self.sinks: Dict[str, BaseSink] = {}
        self._logger = logging.getLogger(__name__)
        self._initialized = False
        self._stats = SinkManagerStats()
    
    async def initialize(self, config: 'LoggingSystemConfig') -> None:
        """
        Initialize sink manager with configuration.
        
        Args:
            config: Logging system configuration
        """
        if self._initialized:
            return
        
        try:
            # Initialize sinks based on configuration
            await self._initialize_sinks(config)
            
            self._initialized = True
            self._logger.info(f"Sink manager initialized with {len(self.sinks)} sinks")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize sink manager: {e}")
            raise
    
    async def _initialize_sinks(self, config: 'LoggingSystemConfig') -> None:
        """Initialize sinks from configuration."""
        # Add default sinks
        if hasattr(config, 'sinks') and config.sinks:
            for sink_name, sink_config in config.sinks.items():
                await self._create_sink(sink_name, sink_config)
        else:
            # Add basic default sink
            await self._add_default_sinks()
    
    async def _add_default_sinks(self) -> None:
        """Add default sinks when no configuration is provided."""
        # Add console sink as default
        console_sink = ConsoleSink("default_console")
        await console_sink.initialize()
        self.sinks["default_console"] = console_sink
    
    async def _create_sink(self, name: str, config: Any) -> None:
        """Create sink from configuration."""
        try:
            sink_type = getattr(config, 'type', 'console')
            
            if sink_type == 'console':
                sink_instance = ConsoleSink(name)
            elif sink_type == 'file':
                sink_instance = TieredFileSink(name)
            elif sink_type == 'network':
                sink_instance = NetworkSink(name)
            elif sink_type == 'metrics':
                sink_instance = MetricsSink(name)
            elif sink_type == 'emergency':
                sink_instance = EmergencySink(name)
            else:
                self._logger.warning(f"Unknown sink type: {sink_type}")
                return
            
            await sink_instance.initialize(config)
            self.sinks[name] = sink_instance
            
        except Exception as e:
            self._logger.error(f"Failed to create sink {name}: {e}")
    
    async def emit_record(self, record: 'LogRecord', formatted_data: Dict[str, Any]) -> None:
        """
        Emit log record to appropriate sinks.
        
        Args:
            record: Log record to emit
            formatted_data: Formatted log data
        """
        if not self._initialized:
            return
        
        try:
            start_time = time.perf_counter()
            
            # Emit to all active sinks
            emit_tasks = []
            for sink_name, sink in self.sinks.items():
                if sink.is_enabled():
                    emit_tasks.append(sink.emit(record, formatted_data))
            
            if emit_tasks:
                await asyncio.gather(*emit_tasks, return_exceptions=True)
            
            # Update statistics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(processing_time, success=True)
            
        except Exception as e:
            self._logger.error(f"Sink emit error: {e}")
            self._update_stats(0, success=False)
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        self._stats.total_messages += 1
        
        if not success:
            self._stats.total_errors += 1
        
        # Update average processing time
        if self._stats.total_messages == 1:
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
        Reconfigure sinks with new configuration.
        
        Args:
            config: New logging system configuration
        """
        try:
            # Shutdown existing sinks
            for sink in self.sinks.values():
                if hasattr(sink, 'shutdown'):
                    await sink.shutdown()
            
            self.sinks.clear()
            
            # Reinitialize with new configuration
            await self._initialize_sinks(config)
            
            self._logger.info(f"Sink manager reconfigured with {len(self.sinks)} sinks")
            
        except Exception as e:
            self._logger.error(f"Failed to reconfigure sink manager: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get sink performance statistics."""
        # Update sink counts by type
        self._stats.sinks_by_type.clear()
        for sink in self.sinks.values():
            sink_type = sink.__class__.__name__
            self._stats.sinks_by_type[sink_type] = (
                self._stats.sinks_by_type.get(sink_type, 0) + 1
            )
        
        self._stats.active_sinks = len(self.sinks)
        
        return {
            'active_sinks': self._stats.active_sinks,
            'total_messages': self._stats.total_messages,
            'total_errors': self._stats.total_errors,
            'error_rate': (
                self._stats.total_errors / self._stats.total_messages
                if self._stats.total_messages > 0 else 0
            ),
            'average_processing_time_ms': self._stats.average_processing_time_ms,
            'sinks_by_type': self._stats.sinks_by_type.copy(),
            'sink_details': {
                name: sink.get_stats() if hasattr(sink, 'get_stats') else {}
                for name, sink in self.sinks.items()
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if sink manager is healthy."""
        return (
            self._initialized and
            len(self.sinks) > 0 and
            self._stats.average_processing_time_ms < 50.0  # Less than 50ms average
        )
    
    async def shutdown(self) -> None:
        """Shutdown sink manager."""
        try:
            # Shutdown all sinks
            for sink in self.sinks.values():
                if hasattr(sink, 'shutdown'):
                    await sink.shutdown()
            
            self.sinks.clear()
            self._initialized = False
            
            self._logger.info("Sink manager shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Sink manager shutdown error: {e}")


# Module exports
__all__ = [
    'SinkManager',
    'SinkManagerStats'
]