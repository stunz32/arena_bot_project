"""
Console Formatter for S-Tier Logging System.

This module provides human-readable console output formatting with colors,
alignment, and intelligent truncation for optimal developer experience.

Features:
- Colorized output with level-based color schemes
- Intelligent field truncation and alignment
- Performance-optimized string operations
- Configurable output format and style
- Terminal capability detection
- Thread-safe color management
"""

import os
import sys
import time
import logging
import threading
from typing import Any, Dict, Optional, List, Union, TextIO
from datetime import datetime
from dataclasses import dataclass

# Import from our core components
from ..core.hybrid_async_queue import LogMessage


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    
    # Reset
    RESET = '\033[0m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright text colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


# Level-based color scheme
LEVEL_COLORS = {
    logging.DEBUG: Colors.BRIGHT_BLACK,
    logging.INFO: Colors.BLUE,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD
}

# Level names
LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT"
}


@dataclass
class ConsoleFormatConfig:
    """Configuration for console formatting."""
    
    # Colors and styling
    use_colors: bool = True
    color_scheme: str = "default"  # default, dark, light, monochrome
    bold_logger_names: bool = True
    dim_timestamps: bool = True
    
    # Layout and alignment
    timestamp_width: int = 23  # "2025-07-31 10:30:45.123"
    level_width: int = 5       # "ERROR"
    logger_width: int = 30     # "arena_bot.core.card_recognizer"
    correlation_id_width: int = 12  # "abc123ef4567"
    
    # Content control
    show_timestamp: bool = True
    show_level: bool = True
    show_logger_name: bool = True
    show_correlation_id: bool = True
    show_thread_info: bool = False
    show_performance_summary: bool = True
    show_operation_context: bool = True
    
    # Message formatting
    max_message_length: Optional[int] = None
    wrap_long_messages: bool = True
    indent_wrapped_lines: bool = True
    
    # Context display
    show_context_summary: bool = True
    max_context_items: int = 3
    context_item_max_length: int = 50


def detect_terminal_capabilities() -> Dict[str, bool]:
    """
    Detect terminal capabilities for optimal formatting.
    
    Returns:
        Dictionary of terminal capabilities
    """
    capabilities = {
        'supports_colors': False,
        'supports_unicode': False,
        'terminal_width': 80,
        'is_tty': False
    }
    
    try:
        # Check if output is a TTY
        capabilities['is_tty'] = sys.stdout.isatty()
        
        # Check color support
        if capabilities['is_tty']:
            term = os.environ.get('TERM', '').lower()
            colorterm = os.environ.get('COLORTERM', '').lower()
            
            # Common terminals that support colors
            color_terms = ['xterm', 'ansi', 'color', 'linux', 'screen', 'tmux']
            capabilities['supports_colors'] = (
                any(term_type in term for term_type in color_terms) or
                'color' in colorterm or
                'truecolor' in colorterm or
                os.environ.get('FORCE_COLOR') == '1'
            )
        
        # Check Unicode support
        encoding = sys.stdout.encoding or 'ascii'
        capabilities['supports_unicode'] = encoding.lower() in ['utf-8', 'utf8']
        
        # Get terminal width
        try:
            if capabilities['is_tty']:
                import shutil
                capabilities['terminal_width'] = shutil.get_terminal_size().columns
        except Exception:
            pass
            
    except Exception:
        pass  # Use defaults
    
    return capabilities


class ConsoleFormatter:
    """
    Human-readable console formatter for log messages.
    
    Provides colorized, well-formatted console output optimized for
    developer experience and terminal readability.
    
    Features:
    - Intelligent color scheme selection
    - Optimal field width and alignment
    - Context summarization
    - Performance information display
    - Terminal capability detection
    """
    
    def __init__(self,
                 config: Optional[ConsoleFormatConfig] = None,
                 stream: Optional[TextIO] = None,
                 auto_detect_capabilities: bool = True):
        """
        Initialize console formatter.
        
        Args:
            config: Console formatting configuration
            stream: Output stream (defaults to sys.stdout)
            auto_detect_capabilities: Auto-detect terminal capabilities
        """
        self.config = config or ConsoleFormatConfig()
        self.stream = stream or sys.stdout
        
        # Terminal capabilities
        if auto_detect_capabilities:
            self.capabilities = detect_terminal_capabilities()
        else:
            self.capabilities = {
                'supports_colors': True,
                'supports_unicode': True,
                'terminal_width': 120,
                'is_tty': True
            }
        
        # Adjust configuration based on capabilities
        if not self.capabilities['supports_colors']:
            self.config.use_colors = False
        
        # Performance tracking
        self._format_count = 0
        self._total_format_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.ConsoleFormatter")
        
        self._logger.info("ConsoleFormatter initialized",
                         extra={
                             'use_colors': self.config.use_colors,
                             'terminal_width': self.capabilities['terminal_width'],
                             'supports_unicode': self.capabilities['supports_unicode']
                         })
    
    def format(self, message: LogMessage) -> str:
        """
        Format LogMessage for console output.
        
        Args:
            message: LogMessage to format
            
        Returns:
            Formatted console string
        """
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                formatted = self._format_message(message)
                
                # Update performance tracking
                elapsed_time = time.perf_counter() - start_time
                self._update_format_stats(elapsed_time)
                
                return formatted
                
        except Exception as e:
            # Fallback formatting
            elapsed_time = time.perf_counter() - start_time
            self._update_format_stats(elapsed_time)
            
            return self._fallback_format(message, e)
    
    def _format_message(self, message: LogMessage) -> str:
        """
        Internal message formatting.
        
        Args:
            message: LogMessage to format
            
        Returns:
            Formatted string
        """
        parts = []
        
        # Timestamp
        if self.config.show_timestamp:
            timestamp_str = self._format_timestamp(message.timestamp)
            if self.config.use_colors and self.config.dim_timestamps:
                timestamp_str = f"{Colors.DIM}{timestamp_str}{Colors.RESET}"
            parts.append(self._pad_field(timestamp_str, self.config.timestamp_width))
        
        # Level
        if self.config.show_level:
            level_str = LEVEL_NAMES.get(message.level, f"L{message.level}")
            if self.config.use_colors:
                color = LEVEL_COLORS.get(message.level, Colors.WHITE)
                level_str = f"{color}{level_str}{Colors.RESET}"
            parts.append(self._pad_field(level_str, self.config.level_width))
        
        # Logger name
        if self.config.show_logger_name:
            logger_str = self._format_logger_name(message.logger_name)
            if self.config.use_colors and self.config.bold_logger_names:
                logger_str = f"{Colors.BOLD}{logger_str}{Colors.RESET}"
            parts.append(self._pad_field(logger_str, self.config.logger_width))
        
        # Correlation ID
        if self.config.show_correlation_id and message.correlation_id:
            corr_id = message.correlation_id[-self.config.correlation_id_width:]
            if self.config.use_colors:
                corr_id = f"{Colors.CYAN}{corr_id}{Colors.RESET}"
            parts.append(self._pad_field(corr_id, self.config.correlation_id_width))
        
        # Main message
        message_str = self._format_main_message(message.message)
        parts.append(message_str)
        
        # Construct main line
        main_line = " ".join(parts)
        
        # Additional lines for context, performance, etc.
        additional_lines = []
        
        # Thread info
        if self.config.show_thread_info and message.thread_id:
            thread_info = f"Thread: {message.thread_id}"
            if self.config.use_colors:
                thread_info = f"{Colors.BRIGHT_BLACK}{thread_info}{Colors.RESET}"
            additional_lines.append(self._indent_line(thread_info))
        
        # Performance summary
        if self.config.show_performance_summary and message.performance:
            perf_summary = self._format_performance_summary(message.performance)
            if perf_summary:
                additional_lines.append(self._indent_line(perf_summary))
        
        # Operation context
        if self.config.show_operation_context and message.operation:
            op_context = self._format_operation_context(message.operation)
            if op_context:
                additional_lines.append(self._indent_line(op_context))
        
        # Context summary
        if self.config.show_context_summary and message.context:
            context_summary = self._format_context_summary(message.context)
            if context_summary:
                additional_lines.append(self._indent_line(context_summary))
        
        # Error information
        if message.error:
            error_info = self._format_error_info(message.error)
            if error_info:
                additional_lines.append(self._indent_line(error_info))
        
        # Combine all lines
        if additional_lines:
            return main_line + "\n" + "\n".join(additional_lines)
        else:
            return main_line
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for console display."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Millisecond precision
        except Exception:
            return f"{timestamp:.3f}"
    
    def _format_logger_name(self, logger_name: str) -> str:
        """Format logger name with intelligent truncation."""
        if len(logger_name) <= self.config.logger_width:
            return logger_name
        
        # Intelligent truncation - keep important parts
        parts = logger_name.split('.')
        if len(parts) > 2:
            # Keep first and last parts, truncate middle
            first = parts[0]
            last = parts[-1]
            available = self.config.logger_width - len(first) - len(last) - 3  # "...""
            
            if available > 0:
                return f"{first}...{last}"
        
        # Simple truncation
        return logger_name[:self.config.logger_width - 3] + "..."
    
    def _format_main_message(self, message: str) -> str:
        """Format the main log message."""
        # Apply length limit if configured
        if self.config.max_message_length and len(message) > self.config.max_message_length:
            message = message[:self.config.max_message_length] + "..."
        
        # Handle line wrapping
        if self.config.wrap_long_messages:
            terminal_width = self.capabilities['terminal_width']
            used_width = (
                (self.config.timestamp_width if self.config.show_timestamp else 0) +
                (self.config.level_width if self.config.show_level else 0) +
                (self.config.logger_width if self.config.show_logger_name else 0) +
                (self.config.correlation_id_width if self.config.show_correlation_id else 0) +
                4  # Spaces between fields
            )
            
            available_width = max(40, terminal_width - used_width)
            
            if len(message) > available_width:
                lines = []
                words = message.split(' ')
                current_line = ""
                
                for word in words:
                    if len(current_line + word) <= available_width:
                        current_line += word + " "
                    else:
                        if current_line:
                            lines.append(current_line.rstrip())
                        current_line = word + " "
                
                if current_line:
                    lines.append(current_line.rstrip())
                
                if len(lines) > 1 and self.config.indent_wrapped_lines:
                    # Indent continuation lines
                    indent = " " * (used_width + 2)
                    wrapped_lines = [lines[0]] + [indent + line for line in lines[1:]]
                    return "\n".join(wrapped_lines)
                else:
                    return "\n".join(lines)
        
        return message
    
    def _format_performance_summary(self, performance: Dict[str, Any]) -> Optional[str]:
        """Format performance data summary."""
        try:
            summary_parts = []
            
            # Duration
            if 'operation_duration_ms' in performance:
                duration = performance['operation_duration_ms']
                if isinstance(duration, (int, float)):
                    summary_parts.append(f"⏱️ {duration:.1f}ms")
            
            # Memory
            if 'memory_used_mb' in performance:
                memory = performance['memory_used_mb']
                if isinstance(memory, (int, float)):
                    summary_parts.append(f"💾 {memory:.1f}MB")
            
            # CPU
            if 'cpu_percent_delta' in performance:
                cpu = performance['cpu_percent_delta']
                if isinstance(cpu, (int, float)):
                    summary_parts.append(f"🖥️ {cpu:.1f}% CPU")
            
            if summary_parts:
                summary = " ".join(summary_parts)
                if self.config.use_colors:
                    summary = f"{Colors.BRIGHT_BLACK}Performance: {summary}{Colors.RESET}"
                else:
                    summary = f"Performance: {summary}"
                return summary
            
        except Exception as e:
            self._logger.warning(f"Performance summary formatting failed: {e}")
        
        return None
    
    def _format_operation_context(self, operation: Dict[str, Any]) -> Optional[str]:
        """Format operation context information."""
        try:
            context_parts = []
            
            # Operation name
            if 'operation_name' in operation:
                name = operation['operation_name']
                context_parts.append(f"Op: {name}")
            
            # Progress
            if 'progress' in operation:
                progress = operation['progress']
                if isinstance(progress, (int, float)):
                    context_parts.append(f"Progress: {progress:.0%}")
            
            # Stage
            if 'stage' in operation:
                stage = operation['stage']
                context_parts.append(f"Stage: {stage}")
            
            if context_parts:
                context_str = " | ".join(context_parts)
                if self.config.use_colors:
                    context_str = f"{Colors.MAGENTA}Operation: {context_str}{Colors.RESET}"
                else:
                    context_str = f"Operation: {context_str}"
                return context_str
            
        except Exception as e:
            self._logger.warning(f"Operation context formatting failed: {e}")
        
        return None
    
    def _format_context_summary(self, context: Dict[str, Any]) -> Optional[str]:
        """Format context data summary."""
        try:
            if not context:
                return None
            
            # Select most important context items
            important_keys = [
                'card_code', 'card_name', 'confidence', 'position',
                'user_id', 'session_id', 'request_id', 'operation_type',
                'error_code', 'status', 'result', 'count'
            ]
            
            summary_items = []
            
            # Add important keys first
            for key in important_keys:
                if key in context and len(summary_items) < self.config.max_context_items:
                    value = context[key]
                    formatted_value = self._format_context_value(value)
                    summary_items.append(f"{key}={formatted_value}")
            
            # Add other keys if we have space
            for key, value in context.items():
                if (key not in important_keys and 
                    len(summary_items) < self.config.max_context_items):
                    formatted_value = self._format_context_value(value)
                    summary_items.append(f"{key}={formatted_value}")
            
            if summary_items:
                summary = " | ".join(summary_items)
                if self.config.use_colors:
                    summary = f"{Colors.GREEN}Context: {summary}{Colors.RESET}"
                else:
                    summary = f"Context: {summary}"
                return summary
            
        except Exception as e:
            self._logger.warning(f"Context summary formatting failed: {e}")
        
        return None
    
    def _format_context_value(self, value: Any) -> str:
        """Format a context value for display."""
        try:
            if isinstance(value, str):
                if len(value) > self.config.context_item_max_length:
                    return value[:self.config.context_item_max_length - 3] + "..."
                return value
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, bool):
                return "true" if value else "false"
            elif value is None:
                return "null"
            else:
                str_value = str(value)
                if len(str_value) > self.config.context_item_max_length:
                    return str_value[:self.config.context_item_max_length - 3] + "..."
                return str_value
        except Exception:
            return "<error>"
    
    def _format_error_info(self, error: Dict[str, Any]) -> Optional[str]:
        """Format error information."""
        try:
            error_parts = []
            
            # Error type
            if 'exception_type' in error:
                error_type = error['exception_type']
                error_parts.append(f"Type: {error_type}")
            
            # Error message
            if 'message' in error:
                message = error['message']
                if len(message) > 100:
                    message = message[:100] + "..."
                error_parts.append(f"Message: {message}")
            
            if error_parts:
                error_str = " | ".join(error_parts)
                if self.config.use_colors:
                    error_str = f"{Colors.BRIGHT_RED}Error: {error_str}{Colors.RESET}"
                else:
                    error_str = f"Error: {error_str}"
                return error_str
            
        except Exception as e:
            self._logger.warning(f"Error info formatting failed: {e}")
        
        return None
    
    def _pad_field(self, text: str, width: int) -> str:
        """Pad field to specified width, handling ANSI codes."""
        # Calculate actual text length excluding ANSI codes
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        
        if len(clean_text) >= width:
            return text
        
        padding = width - len(clean_text)
        return text + " " * padding
    
    def _indent_line(self, line: str, indent: str = "    ") -> str:
        """Indent a line with specified indentation."""
        return indent + line
    
    def _fallback_format(self, message: LogMessage, error: Exception) -> str:
        """Fallback formatting when main formatting fails."""
        try:
            timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
            level = LEVEL_NAMES.get(message.level, f"L{message.level}")
            
            return (f"{timestamp} {level:>5} {message.logger_name}: {message.message} "
                   f"[FORMATTING_ERROR: {type(error).__name__}]")
        except Exception:
            return f"LOG_FORMAT_ERROR: {message.message}"
    
    def _update_format_stats(self, elapsed_time: float) -> None:
        """Update formatting performance statistics."""
        self._format_count += 1
        self._total_format_time += elapsed_time
    
    def get_format_stats(self) -> Dict[str, Any]:
        """Get formatting performance statistics."""
        if self._format_count == 0:
            return {'status': 'no_data'}
        
        avg_time = self._total_format_time / self._format_count
        
        return {
            'total_formats': self._format_count,
            'average_format_time_us': avg_time * 1_000_000,
            'terminal_capabilities': self.capabilities,
            'configuration': {
                'use_colors': self.config.use_colors,
                'terminal_width': self.capabilities['terminal_width'],
                'show_performance_summary': self.config.show_performance_summary,
                'show_context_summary': self.config.show_context_summary
            }
        }


# Module exports
__all__ = [
    'ConsoleFormatter',
    'ConsoleFormatConfig',
    'Colors',
    'LEVEL_COLORS',
    'LEVEL_NAMES',
    'detect_terminal_capabilities'
]