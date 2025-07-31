"""
S-Tier Logging & Observability System for Arena Bot

A high-performance, enterprise-grade logging system with complete observability,
backward compatibility, and integration with existing monitoring systems.

Key Features:
- Fully asynchronous, non-blocking architecture
- Structured JSON logging with automatic context enrichment
- Performance monitoring with correlation tracking
- Circuit breaker protection and emergency protocols
- Seamless integration with existing monitoring.py
- Backward compatible with standard Python logging

Usage:
    # Backward compatible setup (replaces existing logging)
    from arena_bot.logging_system import setup_s_tier_logging
    setup_s_tier_logging()
    
    # Enhanced API for new development
    from arena_bot.logging_system import STierLogger, operation_context, timed_operation
    
    logger = STierLogger.get_logger(__name__)
    
    @timed_operation("my_operation")
    def my_function():
        with operation_context("processing") as ctx:
            logger.info("Processing started", item_count=100)
            # ... processing ...
            ctx.log_success("processing_complete", items_processed=100)

Author: Arena Bot S-Tier Logging System
Version: 1.0.0
"""

import logging
import sys
from typing import Optional, Dict, Any, Callable

# Import core components
from .core.logger_manager import LoggerManager, STierLogger
from .core.context_enricher import operation_context, timed_operation
from .config.defaults import get_default_config
from .diagnostics.health_checker import health_check

# Version information
__version__ = "1.0.0"
__author__ = "Arena Bot S-Tier Logging System"

# Global instances
_global_logger_manager: Optional[LoggerManager] = None
_is_setup: bool = False


def setup_s_tier_logging(
    config: Optional[Dict[str, Any]] = None,
    enable_performance_monitoring: bool = True,
    enable_structured_output: bool = True,
    enable_metrics_integration: bool = True,
    enable_correlation_tracking: bool = True,
    log_level: int = logging.INFO,
    replace_standard_logging: bool = True
) -> LoggerManager:
    """
    Set up the S-Tier logging system.
    
    This function provides backward compatibility by replacing the standard
    Python logging system with the S-Tier implementation. Existing code
    using logging.getLogger() will automatically use S-Tier features.
    
    Args:
        config: Optional configuration dictionary. Uses defaults if None.
        enable_performance_monitoring: Enable performance tracking and metrics
        enable_structured_output: Enable structured JSON output format  
        enable_metrics_integration: Enable integration with monitoring.py
        enable_correlation_tracking: Enable correlation ID tracking
        log_level: Default logging level
        replace_standard_logging: Replace standard logging with S-Tier
        
    Returns:
        LoggerManager instance for advanced usage
        
    Raises:
        RuntimeError: If setup fails due to system constraints
        
    Example:
        # Basic setup - replaces all existing logging
        setup_s_tier_logging()
        
        # Advanced setup with custom configuration
        setup_s_tier_logging(
            config={'queue_size': 65536, 'worker_threads': 4},
            log_level=logging.DEBUG
        )
    """
    global _global_logger_manager, _is_setup
    
    if _is_setup:
        # Already set up, return existing manager
        return _global_logger_manager
    
    try:
        # Load configuration
        if config is None:
            config = get_default_config()
        
        # Update config with parameters
        config.update({
            'enable_performance_monitoring': enable_performance_monitoring,
            'enable_structured_output': enable_structured_output,
            'enable_metrics_integration': enable_metrics_integration,
            'enable_correlation_tracking': enable_correlation_tracking,
            'log_level': log_level,
            'replace_standard_logging': replace_standard_logging
        })
        
        # Create logger manager
        _global_logger_manager = LoggerManager(config)
        
        # Initialize the system
        _global_logger_manager.initialize()
        
        # Replace standard logging if requested
        if replace_standard_logging:
            _global_logger_manager.replace_standard_logging()
        
        _is_setup = True
        
        # Log successful initialization
        logger = _global_logger_manager.get_logger(__name__)
        logger.info("S-Tier Logging System initialized successfully",
                   version=__version__,
                   config_keys=list(config.keys()),
                   performance_monitoring=enable_performance_monitoring,
                   structured_output=enable_structured_output)
        
        return _global_logger_manager
        
    except Exception as e:
        # Fallback to emergency logging
        print(f"CRITICAL: S-Tier logging setup failed: {e}", file=sys.stderr)
        print("Falling back to standard logging", file=sys.stderr)
        raise RuntimeError(f"S-Tier logging setup failed: {e}") from e


def get_logger_manager() -> Optional[LoggerManager]:
    """
    Get the global logger manager instance.
    
    Returns:
        LoggerManager instance if setup has been called, None otherwise
    """
    return _global_logger_manager


def is_initialized() -> bool:
    """
    Check if the S-Tier logging system is initialized.
    
    Returns:
        True if system is initialized and ready
    """
    return _is_setup and _global_logger_manager is not None


def shutdown() -> None:
    """
    Shutdown the S-Tier logging system gracefully.
    
    This will stop all worker threads, flush all buffers, and clean up
    resources. Should be called during application shutdown.
    """
    global _global_logger_manager, _is_setup
    
    if _global_logger_manager is not None:
        try:
            logger = _global_logger_manager.get_logger(__name__)
            logger.info("S-Tier Logging System shutting down gracefully")
            
            _global_logger_manager.shutdown()
            _global_logger_manager = None
            _is_setup = False
            
        except Exception as e:
            print(f"Error during S-Tier logging shutdown: {e}", file=sys.stderr)


def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health information.
    
    Returns:
        Dictionary containing system health metrics and status
    """
    if not _is_setup or _global_logger_manager is None:
        return {
            'status': 'not_initialized',
            'initialized': False,
            'version': __version__
        }
    
    return health_check(_global_logger_manager)


# Context manager for operation tracking
# (imported from core.context_enricher)
__all__ = [
    # Setup and management
    'setup_s_tier_logging',
    'get_logger_manager', 
    'is_initialized',
    'shutdown',
    'get_system_health',
    
    # Enhanced logging API
    'STierLogger',
    'operation_context',
    'timed_operation',
    
    # Health and diagnostics
    'health_check',
    
    # Version info
    '__version__',
    '__author__'
]


# Automatic cleanup on module unload
import atexit
atexit.register(shutdown)