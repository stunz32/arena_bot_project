"""
Console Sink for S-Tier Logging System.

This module provides console output functionality with intelligent formatting,
color management, and performance optimization for developer-friendly logging.

Features:
- Colorized console output with terminal detection
- Configurable formatting and layout
- Performance-optimized string operations
- Thread-safe console writing
- Intelligent buffering and flushing
- Error handling and fallback formatting
"""

import os
import sys
import time
import threading
import logging
from typing import Any, Dict, Optional, TextIO, Union
from io import StringIO

# Import from our components
from .base_sink import BaseSink, SinkState, ErrorHandlingStrategy
from ..formatters.console_formatter import ConsoleFormatter, ConsoleFormatConfig, detect_terminal_capabilities
from ..core.hybrid_async_queue import LogMessage


class ConsoleSink(BaseSink):
    """
    Console output sink with intelligent formatting and performance optimization.
    
    Provides colorized, human-readable console output optimized for developer
    experience. Includes terminal capability detection, configurable formatting,
    and efficient output management.
    
    Features:
    - Automatic terminal capability detection
    - Configurable color schemes and layouts
    - Thread-safe console writing
    - Performance-optimized output buffering
    - Graceful fallback on formatting errors
    """
    
    def __init__(self,
                 name: str = "console",
                 stream: Optional[TextIO] = None,
                 formatter: Optional[ConsoleFormatter] = None,
                 config: Optional[ConsoleFormatConfig] = None,
                 buffer_size: int = 8192,
                 flush_interval: float = 0.1,
                 force_colors: Optional[bool] = None,
                 encoding: str = "utf-8",
                 errors: str = "replace",
                 **base_kwargs):
        """
        Initialize console sink.
        
        Args:
            name: Sink name for identification
            stream: Output stream (defaults to sys.stdout)
            formatter: Console formatter (auto-created if None)
            config: Console format configuration
            buffer_size: Output buffer size in bytes
            flush_interval: Auto-flush interval in seconds
            force_colors: Force color output (None = auto-detect)
            encoding: Output encoding
            errors: Encoding error handling
            **base_kwargs: Arguments for BaseSink
        """
        # Set up formatter before calling parent __init__
        self.stream = stream or sys.stdout
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.encoding = encoding
        self.errors = errors
        
        # Terminal capabilities
        self.capabilities = detect_terminal_capabilities()
        
        # Override color detection if forced
        if force_colors is not None:
            self.capabilities['supports_colors'] = force_colors
        
        # Create formatter if not provided
        if formatter is None:
            if config is None:
                config = ConsoleFormatConfig()
                # Adjust config based on capabilities
                config.use_colors = self.capabilities['supports_colors']
            
            formatter = ConsoleFormatter(
                config=config, 
                stream=self.stream,
                auto_detect_capabilities=False  # We already detected them
            )
            formatter.capabilities = self.capabilities
        
        # Initialize parent
        super().__init__(name=name, formatter=formatter, **base_kwargs)
        
        # Output management
        self._buffer = StringIO()
        self._buffer_lock = threading.RLock()
        self._last_flush = time.time()
        self._bytes_written = 0
        self._lines_written = 0
        
        # Performance tracking
        self._write_times = []
        self._flush_times = []
        
        # Flush timer for periodic flushing
        self._flush_timer: Optional[threading.Timer] = None
        self._flush_timer_lock = threading.RLock()
        
        self._logger.info(f"ConsoleSink '{name}' initialized",
                         extra={
                             'supports_colors': self.capabilities['supports_colors'],
                             'terminal_width': self.capabilities['terminal_width'],
                             'buffer_size': buffer_size,
                             'stream_type': type(self.stream).__name__
                         })
    
    def _initialize_sink(self) -> bool:
        """Initialize console sink resources."""
        try:
            # Test stream writability
            if hasattr(self.stream, 'write') and hasattr(self.stream, 'flush'):
                # Test write (should be safe for stdout/stderr)
                if self.stream in (sys.stdout, sys.stderr):
                    return True
                else:
                    # For custom streams, test with empty string
                    self.stream.write("")
                    self.stream.flush()
                    return True
            else:
                self._logger.error(f"Stream {self.stream} is not writable")
                return False
                
        except Exception as e:
            self._logger.error(f"Console sink initialization failed: {e}")
            return False
    
    def _cleanup_sink(self) -> bool:
        """Cleanup console sink resources."""
        try:
            # Cancel flush timer
            with self._flush_timer_lock:
                if self._flush_timer:
                    self._flush_timer.cancel()
                    self._flush_timer = None
            
            # Flush any remaining buffered output
            success = self._flush_buffer()
            
            self._logger.info(f"Console sink '{self.name}' cleanup completed",
                             extra={
                                 'bytes_written': self._bytes_written,
                                 'lines_written': self._lines_written,
                                 'cleanup_success': success
                             })
            
            return success
            
        except Exception as e:
            self._logger.error(f"Console sink cleanup failed: {e}")
            return False
    
    def _health_check_sink(self) -> bool:
        """Perform console sink health check."""
        try:
            # Check if stream is still writable
            if not hasattr(self.stream, 'write'):
                return False
            
            # Check if stream is closed
            if hasattr(self.stream, 'closed') and self.stream.closed:
                return False
            
            # For stdout/stderr, check if they're still valid
            if self.stream is sys.stdout:
                return sys.stdout is not None and not sys.stdout.closed
            elif self.stream is sys.stderr:
                return sys.stderr is not None and not sys.stderr.closed
            
            return True
            
        except Exception as e:
            self._logger.warning(f"Console sink health check failed: {e}")
            return False
    
    def _write_message(self, formatted_message: str, message: LogMessage) -> bool:
        """
        Write formatted message to console.
        
        Args:
            formatted_message: Formatted message string
            message: Original LogMessage object
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.perf_counter()
        
        try:
            # Ensure message ends with newline
            if not formatted_message.endswith('\n'):
                formatted_message += '\n'
            
            # Handle encoding
            try:
                if hasattr(self.stream, 'encoding') and self.stream.encoding:
                    # Stream handles encoding
                    output_text = formatted_message
                else:
                    # Manual encoding
                    output_text = formatted_message.encode(
                        self.encoding, errors=self.errors
                    ).decode(self.encoding, errors=self.errors)
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                self._logger.warning(f"Encoding error, using fallback: {e}")
                # Strip non-ASCII characters as fallback
                output_text = ''.join(char if ord(char) < 128 else '?' for char in formatted_message)
            
            # Write to buffer or directly to stream
            if self.buffer_size > 0:
                success = self._write_to_buffer(output_text)
            else:
                success = self._write_to_stream(output_text)
            
            # Update statistics
            if success:
                self._bytes_written += len(output_text)
                self._lines_written += 1
            
            # Track performance
            elapsed_time = time.perf_counter() - start_time
            self._write_times.append(elapsed_time)
            if len(self._write_times) > 1000:
                self._write_times.pop(0)
            
            return success
            
        except Exception as e:
            self._logger.error(f"Console write failed: {e}")
            return False
    
    def _write_to_buffer(self, text: str) -> bool:
        """Write text to buffer with automatic flushing."""
        try:
            with self._buffer_lock:
                self._buffer.write(text)
                
                # Check if buffer needs flushing
                buffer_size = len(self._buffer.getvalue())
                current_time = time.time()
                
                should_flush = (
                    buffer_size >= self.buffer_size or
                    current_time - self._last_flush >= self.flush_interval or
                    '\n' in text  # Flush on newlines for interactive output
                )
                
                if should_flush:
                    return self._flush_buffer()
                else:
                    # Schedule flush if not already scheduled  
                    self._schedule_flush()
                    return True
                    
        except Exception as e:
            self._logger.error(f"Buffer write failed: {e}")
            return False
    
    def _write_to_stream(self, text: str) -> bool:
        """Write text directly to stream."""
        try:
            self.stream.write(text)
            
            # Flush immediately for unbuffered output
            if hasattr(self.stream, 'flush'):
                self.stream.flush()
            
            return True
            
        except Exception as e:
            self._logger.error(f"Stream write failed: {e}")
            return False
    
    def _flush_buffer(self) -> bool:
        """Flush buffered output to stream."""
        start_time = time.perf_counter()
        
        try:
            with self._buffer_lock:
                buffer_content = self._buffer.getvalue()
                
                if buffer_content:
                    # Write to stream
                    self.stream.write(buffer_content)
                    
                    # Flush stream
                    if hasattr(self.stream, 'flush'):
                        self.stream.flush()
                    
                    # Clear buffer
                    self._buffer.seek(0)
                    self._buffer.truncate(0)
                    
                    # Update flush time
                    self._last_flush = time.time()
                    
                    # Track flush performance
                    elapsed_time = time.perf_counter() - start_time
                    self._flush_times.append(elapsed_time)
                    if len(self._flush_times) > 100:
                        self._flush_times.pop(0)
                
                return True
                
        except Exception as e:
            self._logger.error(f"Buffer flush failed: {e}")
            return False
    
    def _schedule_flush(self) -> None:
        """Schedule automatic flush."""
        try:
            with self._flush_timer_lock:
                # Cancel existing timer
                if self._flush_timer:
                    self._flush_timer.cancel()
                
                # Schedule new flush
                self._flush_timer = threading.Timer(
                    self.flush_interval,
                    self._auto_flush
                )
                self._flush_timer.daemon = True
                self._flush_timer.start()
                
        except Exception as e:
            self._logger.warning(f"Flush scheduling failed: {e}")
    
    def _auto_flush(self) -> None:
        """Automatic flush callback."""
        try:
            with self._flush_timer_lock:
                self._flush_timer = None
            
            self._flush_buffer()
            
        except Exception as e:
            self._logger.warning(f"Auto-flush failed: {e}")
    
    def force_flush(self) -> bool:
        """Force immediate flush of buffered output."""
        return self._flush_buffer()
    
    def get_console_stats(self) -> Dict[str, Any]:
        """Get console sink statistics."""
        base_stats = self.get_stats().to_dict()
        
        # Add console-specific stats
        console_stats = {
            'bytes_written': self._bytes_written,
            'lines_written': self._lines_written,
            'buffer_size': self.buffer_size,
            'current_buffer_size': len(self._buffer.getvalue()) if self._buffer else 0,
            'capabilities': self.capabilities,
            'stream_type': type(self.stream).__name__,
            'encoding': self.encoding
        }
        
        # Performance statistics
        if self._write_times:
            console_stats['write_performance'] = {
                'average_write_time_us': (sum(self._write_times) / len(self._write_times)) * 1_000_000,
                'max_write_time_us': max(self._write_times) * 1_000_000,
                'write_samples': len(self._write_times)
            }
        
        if self._flush_times:
            console_stats['flush_performance'] = {
                'average_flush_time_us': (sum(self._flush_times) / len(self._flush_times)) * 1_000_000,
                'max_flush_time_us': max(self._flush_times) * 1_000_000,
                'flush_samples': len(self._flush_times)
            }
        
        # Merge with base stats
        base_stats.update(console_stats)
        return base_stats
    
    def set_color_scheme(self, use_colors: bool) -> None:
        """
        Enable or disable color output.
        
        Args:
            use_colors: Whether to use colors
        """
        if hasattr(self.formatter, 'config'):
            self.formatter.config.use_colors = use_colors and self.capabilities['supports_colors']
            self._logger.info(f"Console colors {'enabled' if self.formatter.config.use_colors else 'disabled'}")
    
    def set_verbosity(self, 
                     show_timestamp: bool = True,
                     show_level: bool = True,
                     show_logger_name: bool = True,
                     show_correlation_id: bool = True,
                     show_thread_info: bool = False,
                     show_performance_summary: bool = True) -> None:
        """
        Configure console output verbosity.
        
        Args:
            show_timestamp: Show timestamp
            show_level: Show log level
            show_logger_name: Show logger name
            show_correlation_id: Show correlation ID
            show_thread_info: Show thread information
            show_performance_summary: Show performance metrics
        """
        if hasattr(self.formatter, 'config'):
            config = self.formatter.config
            config.show_timestamp = show_timestamp
            config.show_level = show_level
            config.show_logger_name = show_logger_name
            config.show_correlation_id = show_correlation_id
            config.show_thread_info = show_thread_info
            config.show_performance_summary = show_performance_summary
            
            self._logger.info("Console verbosity updated",
                             extra={
                                 'timestamp': show_timestamp,
                                 'level': show_level,
                                 'logger': show_logger_name,
                                 'correlation_id': show_correlation_id,
                                 'thread_info': show_thread_info,
                                 'performance': show_performance_summary
                             })


# Module exports
__all__ = [
    'ConsoleSink'
]