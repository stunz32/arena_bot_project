"""
Enhanced S-Tier Logger with Deep Debugging Integration

Extensions to the existing S-tier logging system to support:
- New debug-specific log levels (TRACE_DEEP, STATE_CHANGE, PERFORMANCE_ALERT)
- Integration with debugging infrastructure components
- Enhanced correlation ID propagation for debugging traces
- Automatic context enrichment with debugging metadata
- Performance-optimized debug logging with conditional capture
- Emergency debug mode for critical debugging scenarios

Maintains full compatibility with existing S-tier logging infrastructure.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

# Import existing S-tier logging
from ..logging_system.logger import (
    LogLevel as BaseLogLevel, 
    LogRecord as BaseLogRecord,
    STierLogger as BaseSTierLogger,
    LoggerManager as BaseLoggerManager,
    get_logger as base_get_logger
)


class DebugLogLevel(str, Enum):
    """Enhanced log levels for deep debugging."""
    # Existing levels (for compatibility)
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    
    # New debug-specific levels
    TRACE_DEEP = "TRACE_DEEP"           # Ultra-detailed tracing for debugging
    STATE_CHANGE = "STATE_CHANGE"       # Component state transitions
    PERFORMANCE_ALERT = "PERFORMANCE_ALERT"  # Performance threshold breaches
    METHOD_TRACE = "METHOD_TRACE"       # Method entry/exit tracing
    PIPELINE_TRACE = "PIPELINE_TRACE"   # Data pipeline flow tracing
    HEALTH_CHECK = "HEALTH_CHECK"       # Health monitoring events
    ERROR_PATTERN = "ERROR_PATTERN"     # Error pattern analysis results
    EXCEPTION_DEEP = "EXCEPTION_DEEP"   # Deep exception context
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER" # Circuit breaker state changes
    RECOVERY = "RECOVERY"               # Automated recovery events


@dataclass
class EnhancedLogRecord(BaseLogRecord):
    """
    Enhanced log record with debugging-specific fields.
    
    Extends the base S-tier log record with additional fields
    specifically for debugging and deep analysis.
    """
    
    # Debugging-specific fields
    debug_level: Optional[DebugLogLevel] = None
    method_execution_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    state_change_id: Optional[str] = None
    exception_id: Optional[str] = None
    performance_baseline: Optional[float] = None
    
    # Component context
    component_type: Optional[str] = None
    component_version: Optional[str] = None
    operation_phase: Optional[str] = None
    
    # Debug metadata
    debug_enabled: bool = False
    ultra_debug_enabled: bool = False
    debug_capture_level: int = 1  # 1=basic, 2=detailed, 3=ultra
    
    # Performance impact tracking
    debug_overhead_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced debugging fields."""
        base_dict = super().to_dict()
        
        # Add debugging-specific fields
        debug_dict = {
            'debug_level': self.debug_level.value if self.debug_level else None,
            'method_execution_id': self.method_execution_id,
            'pipeline_id': self.pipeline_id,
            'state_change_id': self.state_change_id,
            'exception_id': self.exception_id,
            'performance_baseline': self.performance_baseline,
            'component_type': self.component_type,
            'component_version': self.component_version,
            'operation_phase': self.operation_phase,
            'debug_enabled': self.debug_enabled,
            'ultra_debug_enabled': self.ultra_debug_enabled,
            'debug_capture_level': self.debug_capture_level,
            'debug_overhead_ms': self.debug_overhead_ms
        }
        
        # Merge with base dictionary
        base_dict.update(debug_dict)
        return base_dict


class DebugModeManager:
    """
    Manages different debug modes and their configurations.
    
    Provides centralized control over debug levels, capture modes,
    and performance optimization for debugging operations.
    """
    
    def __init__(self):
        """Initialize debug mode manager."""
        self.debug_enabled = False
        self.ultra_debug_enabled = False
        self.emergency_debug_enabled = False
        
        # Performance thresholds
        self.performance_thresholds = {
            'method_execution_ms': 100,
            'pipeline_stage_ms': 500,
            'memory_usage_mb': 100,
            'cpu_usage_percent': 80
        }
        
        # Capture levels
        self.capture_level = 1  # 1=basic, 2=detailed, 3=ultra
        
        # Component-specific debug settings
        self.component_debug_levels = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def enable_debug_mode(self, level: str = "standard") -> None:
        """Enable debug mode with specified level."""
        with self.lock:
            if level == "standard":
                self.debug_enabled = True
                self.capture_level = 1
            elif level == "detailed":
                self.debug_enabled = True
                self.capture_level = 2
            elif level == "ultra":
                self.debug_enabled = True
                self.ultra_debug_enabled = True
                self.capture_level = 3
            elif level == "emergency":
                self.debug_enabled = True
                self.ultra_debug_enabled = True
                self.emergency_debug_enabled = True
                self.capture_level = 3
    
    def disable_debug_mode(self) -> None:
        """Disable all debug modes."""
        with self.lock:
            self.debug_enabled = False
            self.ultra_debug_enabled = False
            self.emergency_debug_enabled = False
            self.capture_level = 0
    
    def set_component_debug_level(self, component: str, level: int) -> None:
        """Set debug level for specific component."""
        with self.lock:
            self.component_debug_levels[component] = level
    
    def should_capture_debug(self, component: str = "", level: int = 1) -> bool:
        """Check if debug capture should be performed."""
        with self.lock:
            if not self.debug_enabled:
                return False
            
            # Check component-specific level
            if component in self.component_debug_levels:
                return level <= self.component_debug_levels[component]
            
            # Check global level
            return level <= self.capture_level
    
    def get_debug_context(self) -> Dict[str, Any]:
        """Get current debug context."""
        with self.lock:
            return {
                'debug_enabled': self.debug_enabled,
                'ultra_debug_enabled': self.ultra_debug_enabled,
                'emergency_debug_enabled': self.emergency_debug_enabled,
                'capture_level': self.capture_level,
                'performance_thresholds': self.performance_thresholds.copy(),
                'component_debug_levels': self.component_debug_levels.copy()
            }


class EnhancedSTierLogger(BaseSTierLogger):
    """
    Enhanced S-tier logger with debugging integration.
    
    Extends the base S-tier logger with debugging-specific features
    while maintaining full compatibility with existing logging infrastructure.
    """
    
    def __init__(self, name: str, manager: 'EnhancedLoggerManager', **kwargs):
        """Initialize enhanced logger."""
        super().__init__(name, manager, **kwargs)
        
        # Debug mode manager
        self.debug_manager = DebugModeManager()
        
        # Performance tracking for debug operations
        self.debug_overhead_total = 0.0
        self.debug_operations_count = 0
        
        # Integration with debugging infrastructure
        self._debugging_components = {}
    
    def _create_enhanced_log_record(self,
                                  level: Union[BaseLogLevel, DebugLogLevel],
                                  message: str,
                                  debug_context: Optional[Dict[str, Any]] = None,
                                  **kwargs: Any) -> EnhancedLogRecord:
        """Create enhanced log record with debugging fields."""
        start_time = time.perf_counter()
        
        # Create base record using parent method
        base_record = self._create_log_record(level, message, **kwargs)
        
        # Convert to enhanced record
        enhanced_record = EnhancedLogRecord(
            message=base_record.message,
            level=base_record.level,
            timestamp=base_record.timestamp,
            logger_name=base_record.logger_name,
            correlation_id=base_record.correlation_id,
            trace_id=base_record.trace_id,
            span_id=base_record.span_id,
            user_id=base_record.user_id,
            session_id=base_record.session_id,
            module=base_record.module,
            function=base_record.function,
            line_number=base_record.line_number,
            file_path=base_record.file_path,
            extra=base_record.extra.copy(),
            tags=base_record.tags.copy(),
            processing_time_ms=base_record.processing_time_ms,
            memory_usage_bytes=base_record.memory_usage_bytes,
            is_sensitive=base_record.is_sensitive,
            compliance_tags=base_record.compliance_tags.copy(),
            redacted_fields=base_record.redacted_fields.copy(),
            exception=base_record.exception,
            exception_traceback=base_record.exception_traceback
        )
        
        # Add debugging-specific fields
        if isinstance(level, DebugLogLevel):
            enhanced_record.debug_level = level
        
        # Add debug context
        if debug_context:
            enhanced_record.method_execution_id = debug_context.get('method_execution_id')
            enhanced_record.pipeline_id = debug_context.get('pipeline_id')
            enhanced_record.state_change_id = debug_context.get('state_change_id')
            enhanced_record.exception_id = debug_context.get('exception_id')
            enhanced_record.performance_baseline = debug_context.get('performance_baseline')
            enhanced_record.component_type = debug_context.get('component_type')
            enhanced_record.operation_phase = debug_context.get('operation_phase')
        
        # Add debug mode information
        debug_mode_context = self.debug_manager.get_debug_context()
        enhanced_record.debug_enabled = debug_mode_context['debug_enabled']
        enhanced_record.ultra_debug_enabled = debug_mode_context['ultra_debug_enabled']
        enhanced_record.debug_capture_level = debug_mode_context['capture_level']
        
        # Calculate debug overhead
        debug_overhead = (time.perf_counter() - start_time) * 1000
        enhanced_record.debug_overhead_ms = debug_overhead
        
        # Update performance tracking
        self.debug_overhead_total += debug_overhead
        self.debug_operations_count += 1
        
        return enhanced_record
    
    def log_debug(self,
                 level: DebugLogLevel,
                 message: str,
                 debug_context: Optional[Dict[str, Any]] = None,
                 **kwargs: Any) -> None:
        """Log a debug message with enhanced context."""
        
        # Check if debug logging should be performed
        component = debug_context.get('component_name', '') if debug_context else ''
        capture_level = debug_context.get('capture_level', 1) if debug_context else 1
        
        if not self.debug_manager.should_capture_debug(component, capture_level):
            return
        
        # Create enhanced record
        record = self._create_enhanced_log_record(level, message, debug_context, **kwargs)
        
        # Log using parent's sync method
        self._log_sync(record)
    
    async def alog_debug(self,
                        level: DebugLogLevel,
                        message: str,
                        debug_context: Optional[Dict[str, Any]] = None,
                        **kwargs: Any) -> None:
        """Asynchronously log a debug message with enhanced context."""
        
        # Check if debug logging should be performed
        component = debug_context.get('component_name', '') if debug_context else ''
        capture_level = debug_context.get('capture_level', 1) if debug_context else 1
        
        if not self.debug_manager.should_capture_debug(component, capture_level):
            return
        
        # Create enhanced record
        record = self._create_enhanced_log_record(level, message, debug_context, **kwargs)
        
        # Log using parent's async method
        await self._log_async(record)
    
    # Convenience methods for debug-specific log levels
    def trace_deep(self, message: str, **kwargs: Any) -> None:
        """Log ultra-detailed trace message."""
        self.log_debug(DebugLogLevel.TRACE_DEEP, message, **kwargs)
    
    def state_change(self, message: str, **kwargs: Any) -> None:
        """Log state change event."""
        self.log_debug(DebugLogLevel.STATE_CHANGE, message, **kwargs)
    
    def performance_alert(self, message: str, **kwargs: Any) -> None:
        """Log performance alert."""
        self.log_debug(DebugLogLevel.PERFORMANCE_ALERT, message, **kwargs)
    
    def method_trace(self, message: str, **kwargs: Any) -> None:
        """Log method trace event."""
        self.log_debug(DebugLogLevel.METHOD_TRACE, message, **kwargs)
    
    def pipeline_trace(self, message: str, **kwargs: Any) -> None:
        """Log pipeline trace event."""
        self.log_debug(DebugLogLevel.PIPELINE_TRACE, message, **kwargs)
    
    def health_check(self, message: str, **kwargs: Any) -> None:
        """Log health check event."""
        self.log_debug(DebugLogLevel.HEALTH_CHECK, message, **kwargs)
    
    def error_pattern(self, message: str, **kwargs: Any) -> None:
        """Log error pattern analysis."""
        self.log_debug(DebugLogLevel.ERROR_PATTERN, message, **kwargs)
    
    def exception_deep(self, message: str, **kwargs: Any) -> None:
        """Log deep exception context."""
        self.log_debug(DebugLogLevel.EXCEPTION_DEEP, message, **kwargs)
    
    def circuit_breaker(self, message: str, **kwargs: Any) -> None:
        """Log circuit breaker event."""
        self.log_debug(DebugLogLevel.CIRCUIT_BREAKER, message, **kwargs)
    
    def recovery(self, message: str, **kwargs: Any) -> None:
        """Log recovery event."""
        self.log_debug(DebugLogLevel.RECOVERY, message, **kwargs)
    
    # Async versions
    async def atrace_deep(self, message: str, **kwargs: Any) -> None:
        """Async log ultra-detailed trace message."""
        await self.alog_debug(DebugLogLevel.TRACE_DEEP, message, **kwargs)
    
    async def astate_change(self, message: str, **kwargs: Any) -> None:
        """Async log state change event."""
        await self.alog_debug(DebugLogLevel.STATE_CHANGE, message, **kwargs)
    
    async def aperformance_alert(self, message: str, **kwargs: Any) -> None:
        """Async log performance alert."""
        await self.alog_debug(DebugLogLevel.PERFORMANCE_ALERT, message, **kwargs)
    
    def enable_debug_mode(self, level: str = "standard") -> None:
        """Enable debug mode for this logger."""
        self.debug_manager.enable_debug_mode(level)
        self.info(f"🐛 Debug mode enabled: {level}")
    
    def disable_debug_mode(self) -> None:
        """Disable debug mode for this logger."""
        self.debug_manager.disable_debug_mode()
        self.info("🐛 Debug mode disabled")
    
    def get_debug_performance_stats(self) -> Dict[str, Any]:
        """Get debug performance statistics."""
        base_stats = self.get_performance_stats()
        
        avg_debug_overhead = (
            self.debug_overhead_total / self.debug_operations_count
            if self.debug_operations_count > 0 else 0
        )
        
        debug_stats = {
            'debug_operations_count': self.debug_operations_count,
            'total_debug_overhead_ms': self.debug_overhead_total,
            'average_debug_overhead_ms': avg_debug_overhead,
            'debug_manager_context': self.debug_manager.get_debug_context()
        }
        
        base_stats.update(debug_stats)
        return base_stats


class EnhancedLoggerManager(BaseLoggerManager):
    """
    Enhanced logger manager with debugging integration.
    
    Extends the base logger manager to support debugging infrastructure
    integration and enhanced logging capabilities.
    """
    
    def __init__(self, **kwargs):
        """Initialize enhanced logger manager."""
        super().__init__(**kwargs)
        
        # Global debug mode manager
        self.global_debug_manager = DebugModeManager()
        
        # Integration with debugging components
        self.debugging_components = {}
        
        # Performance monitoring for debug operations
        self.debug_performance_monitor = {
            'total_debug_logs': 0,
            'debug_overhead_ms': 0.0,
            'emergency_debug_activations': 0
        }
    
    def get_logger(self, name: str) -> EnhancedSTierLogger:
        """Get or create an enhanced logger."""
        with self._logger_lock:
            if name not in self._loggers:
                # Create enhanced logger
                logger_config = self.config.loggers.get(name)
                if logger_config:
                    level = BaseLogLevel(logger_config.level.value)
                    propagate = logger_config.propagate
                else:
                    level = BaseLogLevel.INFO
                    propagate = True
                
                enhanced_logger = EnhancedSTierLogger(
                    name=name,
                    manager=self,
                    level=level,
                    propagate=propagate
                )
                
                # Copy global debug settings
                enhanced_logger.debug_manager = self.global_debug_manager
                
                self._loggers[name] = enhanced_logger
            
            return self._loggers[name]
    
    def enable_global_debug_mode(self, level: str = "standard") -> None:
        """Enable debug mode globally for all loggers."""
        self.global_debug_manager.enable_debug_mode(level)
        
        # Update all existing loggers
        with self._logger_lock:
            for logger in self._loggers.values():
                if isinstance(logger, EnhancedSTierLogger):
                    logger.debug_manager = self.global_debug_manager
        
        # Log the activation
        logger = self.get_logger("arena_bot.debugging.enhanced_logger")
        logger.info(f"🐛 Global debug mode enabled: {level}")
    
    def disable_global_debug_mode(self) -> None:
        """Disable debug mode globally for all loggers."""
        self.global_debug_manager.disable_debug_mode()
        
        # Log the deactivation
        logger = self.get_logger("arena_bot.debugging.enhanced_logger")
        logger.info("🐛 Global debug mode disabled")
    
    def activate_emergency_debug(self) -> None:
        """Activate emergency debug mode for critical debugging."""
        self.enable_global_debug_mode("emergency")
        self.debug_performance_monitor['emergency_debug_activations'] += 1
        
        # Log emergency activation
        logger = self.get_logger("arena_bot.debugging.enhanced_logger")
        logger.critical("🚨 EMERGENCY DEBUG MODE ACTIVATED")
    
    def register_debugging_component(self, name: str, component: Any) -> None:
        """Register a debugging component for integration."""
        self.debugging_components[name] = component
        
        logger = self.get_logger("arena_bot.debugging.enhanced_logger")
        logger.info(f"🔧 Debugging component registered: {name}")
    
    def get_debug_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive debug performance statistics."""
        base_stats = self.get_performance_stats()
        
        # Aggregate logger debug stats
        total_debug_operations = 0
        total_debug_overhead = 0.0
        
        with self._logger_lock:
            for logger in self._loggers.values():
                if isinstance(logger, EnhancedSTierLogger):
                    logger_stats = logger.get_debug_performance_stats()
                    total_debug_operations += logger_stats.get('debug_operations_count', 0)
                    total_debug_overhead += logger_stats.get('total_debug_overhead_ms', 0.0)
        
        debug_stats = {
            'global_debug_manager': self.global_debug_manager.get_debug_context(),
            'total_debug_operations': total_debug_operations,
            'total_debug_overhead_ms': total_debug_overhead,
            'average_debug_overhead_ms': (
                total_debug_overhead / total_debug_operations
                if total_debug_operations > 0 else 0
            ),
            'debugging_components': list(self.debugging_components.keys()),
            'performance_monitor': self.debug_performance_monitor.copy()
        }
        
        base_stats.update(debug_stats)
        return base_stats


# Global enhanced logger manager
_global_enhanced_manager: Optional[EnhancedLoggerManager] = None
_enhanced_manager_lock = threading.Lock()


def get_enhanced_logger(name: str) -> EnhancedSTierLogger:
    """
    Get enhanced S-tier logger instance with debugging capabilities.
    
    Args:
        name: Logger name
        
    Returns:
        EnhancedSTierLogger instance
    """
    global _global_enhanced_manager
    
    if not _global_enhanced_manager:
        with _enhanced_manager_lock:
            if not _global_enhanced_manager:
                _global_enhanced_manager = EnhancedLoggerManager()
    
    return _global_enhanced_manager.get_logger(name)


def enable_debug_logging(level: str = "standard") -> None:
    """Enable debug logging globally."""
    global _global_enhanced_manager
    
    if not _global_enhanced_manager:
        get_enhanced_logger("arena_bot.debugging")  # Initialize manager
    
    _global_enhanced_manager.enable_global_debug_mode(level)


def disable_debug_logging() -> None:
    """Disable debug logging globally."""
    global _global_enhanced_manager
    
    if _global_enhanced_manager:
        _global_enhanced_manager.disable_global_debug_mode()


def activate_emergency_debug() -> None:
    """Activate emergency debug mode."""
    global _global_enhanced_manager
    
    if not _global_enhanced_manager:
        get_enhanced_logger("arena_bot.debugging")  # Initialize manager
    
    _global_enhanced_manager.activate_emergency_debug()


# Backward compatibility with existing S-tier logging
def get_logger(name: str) -> EnhancedSTierLogger:
    """
    Get logger with enhanced debugging capabilities.
    
    Maintains compatibility with existing get_logger calls while
    providing enhanced debugging features.
    """
    return get_enhanced_logger(name)