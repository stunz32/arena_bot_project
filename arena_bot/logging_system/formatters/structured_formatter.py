"""
Structured JSON Formatter for S-Tier Logging System.

This module provides high-performance JSON formatting for log messages,
converting LogMessage objects into structured JSON format with optimized
serialization and consistent schema.

Features:
- High-performance JSON serialization (<200μs target)
- Consistent schema with validation
- Optimized for minimal allocations
- Support for custom field mapping
- Error handling and fallback formatting
- Schema versioning and evolution
"""

import json
import time
import logging
import traceback
from typing import Any, Dict, Optional, List, Union, Callable
from datetime import datetime, timezone
from dataclasses import asdict
from collections import OrderedDict

# Import from our core components
from ..core.hybrid_async_queue import LogMessage
from .base_formatter import BaseFormatter


# JSON Schema version for compatibility
SCHEMA_VERSION = "1.0"

# Standard log levels mapping
LOG_LEVELS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO", 
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL"
}

# Field ordering for consistent JSON output
FIELD_ORDER = [
    'timestamp',
    'level', 
    'logger',
    'message',
    'ids',
    'context',
    'performance',
    'system',
    'operation',
    'error',
    'metadata'
]


class StructuredFormatter(BaseFormatter):
    """
    High-performance JSON formatter for log messages.
    
    Converts LogMessage objects into structured JSON format with consistent
    schema, optimized serialization, and comprehensive error handling.
    
    Features:
    - <200μs JSON serialization target
    - Consistent field ordering and schema
    - Custom field mapping and transformation
    - Error handling with fallback formatting
    - Schema validation and versioning
    """
    
    def __init__(self,
                 name: str = "structured",
                 include_timestamp: bool = True,
                 timestamp_format: str = "iso",
                 include_level_name: bool = True,
                 include_thread_info: bool = True,
                 include_performance_metrics: bool = True,
                 include_system_info: bool = True,
                 include_operation_context: bool = True,
                 max_message_length: Optional[int] = None,
                 max_context_depth: int = 10,
                 custom_field_transformers: Optional[Dict[str, Callable]] = None,
                 ensure_ascii: bool = False,
                 indent: Optional[int] = None):
        """
        Initialize structured formatter.
        
        Args:
            include_timestamp: Include timestamp in output
            timestamp_format: Timestamp format ('iso', 'unix', 'readable')
            include_level_name: Include level name (vs just number)
            include_thread_info: Include thread information
            include_performance_metrics: Include performance data
            include_system_info: Include system metrics
            include_operation_context: Include operation context
            max_message_length: Maximum message length (None = no limit)
            max_context_depth: Maximum nesting depth for context objects
            custom_field_transformers: Custom field transformation functions
            ensure_ascii: Ensure ASCII-only JSON output
            indent: JSON indentation (None = compact)
        """
        # Initialize base formatter
        super().__init__(name)
        
        self.include_timestamp = include_timestamp
        self.timestamp_format = timestamp_format
        self.include_level_name = include_level_name
        self.include_thread_info = include_thread_info
        self.include_performance_metrics = include_performance_metrics
        self.include_system_info = include_system_info
        self.include_operation_context = include_operation_context
        self.max_message_length = max_message_length
        self.max_context_depth = max_context_depth
        self.custom_field_transformers = custom_field_transformers or {}
        self.ensure_ascii = ensure_ascii
        self.indent = indent
        
        # Performance tracking
        self._format_count = 0
        self._total_format_time = 0.0
        self._format_errors = 0
        
        # JSON encoder options for performance
        self._json_separators = (',', ':') if indent is None else None
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.StructuredFormatter")
        
        self._logger.info("StructuredFormatter initialized",
                         extra={
                             'timestamp_format': timestamp_format,
                             'performance_metrics': include_performance_metrics,
                             'system_info': include_system_info,
                             'ensure_ascii': ensure_ascii
                         })
    
    async def format(self, record: Union[LogMessage, 'LogRecord']) -> Dict[str, Any]:
        """
        Format LogMessage/LogRecord as structured dictionary.
        
        Args:
            record: LogMessage or LogRecord to format
            
        Returns:
            Formatted data as dictionary
        """
        start_time = time.perf_counter()
        
        try:
            # Convert to structured dictionary
            structured_data = self._to_structured_dict(record)
            
            # Update performance tracking
            elapsed_time = time.perf_counter() - start_time
            self._update_format_stats(elapsed_time, success=True)
            
            return structured_data
            
        except Exception as e:
            # Fallback formatting on error
            elapsed_time = time.perf_counter() - start_time
            self._update_format_stats(elapsed_time, success=False)
            
            return self._fallback_format(record, e)
    
    def _to_structured_dict(self, message: LogMessage) -> Dict[str, Any]:
        """
        Convert LogMessage to structured dictionary.
        
        Args:
            message: LogMessage to convert
            
        Returns:
            Structured dictionary following schema
        """
        # Start with ordered dictionary for consistent field ordering
        structured = OrderedDict()
        
        # Timestamp
        if self.include_timestamp:
            structured['timestamp'] = self._format_timestamp(message.timestamp)
        
        # Level
        if self.include_level_name:
            structured['level'] = LOG_LEVELS.get(message.level, f"LEVEL_{message.level}")
        else:
            structured['level'] = message.level
        
        # Basic fields
        structured['logger'] = message.logger_name
        structured['message'] = self._format_message(message.message)
        
        # IDs section
        ids = OrderedDict()
        ids['correlation_id'] = message.correlation_id
        if message.sequence_number > 0:
            ids['sequence_number'] = message.sequence_number
        ids['thread_id'] = message.thread_id
        ids['process_id'] = message.process_id
        structured['ids'] = ids
        
        # Context data
        if message.context:
            structured['context'] = self._process_context_data(message.context)
        
        # Performance data
        if self.include_performance_metrics and message.performance:
            structured['performance'] = self._process_performance_data(message.performance)
        
        # System data
        if self.include_system_info and message.system:
            structured['system'] = self._process_system_data(message.system)
        
        # Operation context
        if self.include_operation_context and message.operation:
            structured['operation'] = self._process_operation_data(message.operation)
        
        # Error information
        if message.error:
            structured['error'] = self._process_error_data(message.error)
        
        # Metadata
        metadata = OrderedDict()
        metadata['schema_version'] = SCHEMA_VERSION
        metadata['formatted_at'] = time.time()
        if message.priority > 0:
            metadata['priority'] = message.priority
        if message.retry_count > 0:
            metadata['retry_count'] = message.retry_count
        structured['metadata'] = metadata
        
        # Apply custom field transformers
        if self.custom_field_transformers:
            structured = self._apply_custom_transformers(structured)
        
        return structured
    
    def _format_timestamp(self, timestamp: float) -> Union[str, float, int]:
        """
        Format timestamp according to configured format.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Formatted timestamp
        """
        try:
            if self.timestamp_format == "iso":
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return dt.isoformat()
            elif self.timestamp_format == "unix":
                return timestamp
            elif self.timestamp_format == "readable":
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Millisecond precision
            else:
                return timestamp
                
        except Exception as e:
            self._logger.warning(f"Timestamp formatting failed: {e}")
            return timestamp
    
    def _format_message(self, message: str) -> str:
        """
        Format log message with length limits and sanitization.
        
        Args:
            message: Raw message string
            
        Returns:
            Formatted message string
        """
        try:
            # Apply length limit if configured
            if self.max_message_length and len(message) > self.max_message_length:
                message = message[:self.max_message_length] + "... [truncated]"
            
            # Basic sanitization - remove control characters
            message = ''.join(char for char in message if ord(char) >= 32 or char in '\t\n\r')
            
            return message
            
        except Exception as e:
            self._logger.warning(f"Message formatting failed: {e}")
            return str(message)
    
    def _process_context_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process context data with depth limiting and sanitization.
        
        Args:
            context: Raw context dictionary
            
        Returns:
            Processed context dictionary
        """
        try:
            return self._limit_depth(context, self.max_context_depth)
        except Exception as e:
            self._logger.warning(f"Context processing failed: {e}")
            return {"processing_error": str(e)}
    
    def _process_performance_data(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process performance data with numeric validation.
        
        Args:
            performance: Raw performance dictionary
            
        Returns:
            Processed performance dictionary
        """
        try:
            processed = {}
            
            for key, value in performance.items():
                # Validate numeric values
                if isinstance(value, (int, float)):
                    # Check for NaN or infinity
                    if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                        processed[key] = None
                    else:
                        processed[key] = value
                else:
                    processed[key] = value
            
            return processed
            
        except Exception as e:
            self._logger.warning(f"Performance data processing failed: {e}")
            return {"processing_error": str(e)}
    
    def _process_system_data(self, system: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process system data with sanitization.
        
        Args:
            system: Raw system dictionary
            
        Returns:
            Processed system dictionary
        """
        try:
            # Remove potentially sensitive system information
            sensitive_keys = {'environment_vars', 'command_line', 'working_directory'}
            
            processed = {
                key: value for key, value in system.items()
                if key not in sensitive_keys
            }
            
            return processed
            
        except Exception as e:
            self._logger.warning(f"System data processing failed: {e}")
            return {"processing_error": str(e)}
    
    def _process_operation_data(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process operation context data.
        
        Args:
            operation: Raw operation dictionary
            
        Returns:
            Processed operation dictionary
        """
        try:
            return self._limit_depth(operation, self.max_context_depth)
        except Exception as e:
            self._logger.warning(f"Operation data processing failed: {e}")
            return {"processing_error": str(e)}
    
    def _process_error_data(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process error information.
        
        Args:
            error: Raw error dictionary
            
        Returns:
            Processed error dictionary
        """
        try:
            processed = {}
            
            # Include standard error fields
            for key in ['exception_type', 'message', 'stack_trace', 'recovery_action']:
                if key in error:
                    processed[key] = error[key]
            
            # Limit stack trace length
            if 'stack_trace' in processed and isinstance(processed['stack_trace'], str):
                if len(processed['stack_trace']) > 10000:  # 10KB limit
                    processed['stack_trace'] = processed['stack_trace'][:10000] + "\n... [truncated]"
            
            return processed
            
        except Exception as e:
            self._logger.warning(f"Error data processing failed: {e}")
            return {"processing_error": str(e)}
    
    def _limit_depth(self, obj: Any, max_depth: int, current_depth: int = 0) -> Any:
        """
        Recursively limit object nesting depth.
        
        Args:
            obj: Object to process
            max_depth: Maximum allowed depth
            current_depth: Current nesting depth
            
        Returns:
            Depth-limited object
        """
        if current_depth >= max_depth:
            return f"... [max depth {max_depth} exceeded]"
        
        if isinstance(obj, dict):
            return {
                key: self._limit_depth(value, max_depth, current_depth + 1)
                for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [
                self._limit_depth(item, max_depth, current_depth + 1)
                for item in obj
            ]
        else:
            return obj
    
    def _apply_custom_transformers(self, structured: OrderedDict) -> OrderedDict:
        """
        Apply custom field transformers.
        
        Args:
            structured: Structured data dictionary
            
        Returns:
            Transformed dictionary
        """
        try:
            for field_path, transformer in self.custom_field_transformers.items():
                try:
                    # Simple field path resolution (e.g., "context.user_id")
                    keys = field_path.split('.')
                    current = structured
                    
                    # Navigate to parent of target field
                    for key in keys[:-1]:
                        if key in current and isinstance(current[key], dict):
                            current = current[key]
                        else:
                            break
                    else:
                        # Apply transformer to target field
                        target_key = keys[-1]
                        if target_key in current:
                            current[target_key] = transformer(current[target_key])
                
                except Exception as e:
                    self._logger.warning(f"Custom transformer failed for {field_path}: {e}")
            
            return structured
            
        except Exception as e:
            self._logger.warning(f"Custom transformers application failed: {e}")
            return structured
    
    def _json_default_handler(self, obj: Any) -> Any:
        """
        Handle objects that can't be serialized to JSON.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        try:
            # Handle datetime objects
            if isinstance(obj, datetime):
                return obj.isoformat()
            
            # Handle other objects with string representation
            return str(obj)
            
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"
    
    def _fallback_format(self, message: LogMessage, error: Exception) -> str:
        """
        Fallback formatting when main formatting fails.
        
        Args:
            message: LogMessage that failed to format
            error: Exception that caused the failure
            
        Returns:
            Fallback JSON string
        """
        try:
            self._format_errors += 1
            
            # Create minimal fallback structure
            fallback = {
                "timestamp": time.time(),
                "level": LOG_LEVELS.get(message.level, f"LEVEL_{message.level}"),
                "logger": message.logger_name,
                "message": str(message.message),
                "correlation_id": message.correlation_id,
                "formatting_error": {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "fallback_used": True
                },
                "metadata": {
                    "schema_version": SCHEMA_VERSION,
                    "fallback_format": True
                }
            }
            
            return json.dumps(fallback, separators=(',', ':'), ensure_ascii=True)
            
        except Exception as fallback_error:
            # Ultimate fallback - plain text
            return (f'{{"timestamp": {time.time()}, "level": "ERROR", '
                   f'"message": "JSON formatting failed", '
                   f'"error": "{str(fallback_error)}", "fallback": true}}')
    
    def _update_format_stats(self, elapsed_time: float, success: bool) -> None:
        """
        Update formatting performance statistics.
        
        Args:
            elapsed_time: Time taken for formatting
            success: Whether formatting succeeded
        """
        self._format_count += 1
        self._total_format_time += elapsed_time
        
        if not success:
            self._format_errors += 1
        
        # Warn if formatting is taking too long
        if elapsed_time > 0.0002:  # 200μs warning threshold
            self._logger.warning(f"Slow JSON formatting: {elapsed_time * 1000:.1f}ms")
    
    def get_format_stats(self) -> Dict[str, Any]:
        """
        Get formatting performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if self._format_count == 0:
            return {'status': 'no_data'}
        
        avg_time = self._total_format_time / self._format_count
        error_rate = (self._format_errors / self._format_count) * 100
        
        return {
            'total_formats': self._format_count,
            'total_errors': self._format_errors,
            'error_rate_percent': error_rate,
            'average_format_time_us': avg_time * 1_000_000,
            'performance_target_met': avg_time < 0.0002,  # 200μs target
            'configuration': {
                'timestamp_format': self.timestamp_format,
                'include_performance_metrics': self.include_performance_metrics,
                'include_system_info': self.include_system_info,
                'max_context_depth': self.max_context_depth,
                'ensure_ascii': self.ensure_ascii
            }
        }
    
    def validate_schema(self, json_string: str) -> bool:
        """
        Validate JSON string against expected schema.
        
        Args:
            json_string: JSON string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = json.loads(json_string)
            
            # Check required fields
            required_fields = ['timestamp', 'level', 'logger', 'message', 'ids']
            
            for field in required_fields:
                if field not in data:
                    return False
            
            # Check IDs structure
            if not isinstance(data['ids'], dict):
                return False
            
            if 'correlation_id' not in data['ids']:
                return False
            
            # Check metadata
            if 'metadata' in data:
                metadata = data['metadata']
                if not isinstance(metadata, dict):
                    return False
                
                if 'schema_version' not in metadata:
                    return False
            
            return True
            
        except Exception:
            return False


# Module exports
__all__ = [
    'StructuredFormatter',
    'SCHEMA_VERSION',
    'LOG_LEVELS',
    'FIELD_ORDER'
]