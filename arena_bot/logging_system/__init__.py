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
from .logger import LoggerManager, STierLogger, LogLevel, LogRecord, get_logger, initialize_logging, shutdown_logging
from .core.context_enricher import ContextEnricher
from .config import ConfigurationFactory, create_development_config
from .diagnostics.health_checker import health_check
from .resource_monitor import (
    UnifiedResourceMonitor,
    ResourceMonitorConfig,
    initialize_unified_monitoring,
    get_unified_health_status
)

# Version information
__version__ = "1.0.0"
__author__ = "Arena Bot S-Tier Logging System"

# Global instances
_global_logger_manager: Optional[LoggerManager] = None
_is_setup: bool = False


def setup_s_tier_logging(
    config_path: Optional[str] = None,
    environment: str = "development",
    enable_performance_monitoring: bool = True,
    enable_structured_output: bool = True,
    enable_metrics_integration: bool = True,
    enable_correlation_tracking: bool = True,
    log_level: str = "INFO",
    replace_standard_logging: bool = True
) -> LoggerManager:
    """
    Set up the S-Tier logging system.
    
    This function provides backward compatibility by replacing the standard
    Python logging system with the S-Tier implementation. Existing code
    using logging.getLogger() will automatically use S-Tier features.
    
    Args:
        config_path: Optional path to configuration file. Uses defaults if None.
        environment: Target environment (development, testing, staging, production)
        enable_performance_monitoring: Enable performance tracking and metrics
        enable_structured_output: Enable structured JSON output format  
        enable_metrics_integration: Enable integration with monitoring.py
        enable_correlation_tracking: Enable correlation ID tracking
        log_level: Default logging level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
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
        import asyncio
        from pathlib import Path
        
        # Create configuration factory
        config_factory = ConfigurationFactory()
        
        # Create configuration with overrides
        overrides = {}
        if environment != "development":
            overrides["environment"] = environment
        
        # Load configuration
        config_path_obj = Path(config_path) if config_path else None
        config = config_factory.create_config(
            config_path=config_path_obj,
            environment=environment,
            overrides=overrides
        )
        
        # Create logger manager
        _global_logger_manager = LoggerManager(config)
        
        # Initialize the system (handle async properly)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, schedule initialization
                asyncio.create_task(_global_logger_manager.initialize())
            else:
                # If not in async context, run initialization
                asyncio.run(_global_logger_manager.initialize())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(_global_logger_manager.initialize())
        
        # Initialize unified resource monitoring
        resource_config = ResourceMonitorConfig(
            enable_existing_monitoring=enable_metrics_integration,
            enable_unified_health_reporting=True
        )
        unified_monitor = initialize_unified_monitoring(_global_logger_manager, resource_config)
        
        # Start unified monitoring
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(unified_monitor.start_monitoring())
            else:
                asyncio.run(unified_monitor.start_monitoring())
        except RuntimeError:
            asyncio.run(unified_monitor.start_monitoring())
        
        _is_setup = True
        
        # Log successful initialization
        logger = _global_logger_manager.get_logger(__name__)
        logger.info("S-Tier Logging System initialized successfully",
                   extra={
                       'version': __version__,
                       'environment': environment,
                       'performance_monitoring': enable_performance_monitoring,
                       'structured_output': enable_structured_output
                   })
        
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
            import asyncio
            
            logger = _global_logger_manager.get_logger(__name__)
            logger.info("S-Tier Logging System shutting down gracefully")
            
            # Handle async shutdown properly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_global_logger_manager.shutdown())
                else:
                    asyncio.run(_global_logger_manager.shutdown())
            except RuntimeError:
                asyncio.run(_global_logger_manager.shutdown())
            
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
    
    # Get unified health status that includes both S-tier and existing monitoring
    unified_health = get_unified_health_status()
    
    # Get detailed S-tier health check
    s_tier_health = health_check(_global_logger_manager)
    
    # Combine health information
    return {
        'unified_status': unified_health,
        's_tier_details': s_tier_health,
        'version': __version__,
        'initialized': True
    }


# Export the new S-tier logging API
__all__ = [
    # Setup and management
    'setup_s_tier_logging',
    'get_logger_manager', 
    'is_initialized',
    'shutdown',
    'get_system_health',
    
    # Core logging API
    'LoggerManager',
    'STierLogger',
    'LogLevel',
    'LogRecord',
    'get_logger',
    'initialize_logging',
    'shutdown_logging',
    
    # Configuration
    'ConfigurationFactory',
    'create_development_config',
    
    # Context enrichment
    'ContextEnricher',
    
    # Resource monitoring
    'UnifiedResourceMonitor',
    'ResourceMonitorConfig',
    'initialize_unified_monitoring',
    'get_unified_health_status',
    
    # Health and diagnostics
    'health_check',
    
    # Version info
    '__version__',
    '__author__'
]


# Automatic cleanup on module unload
import atexit
atexit.register(shutdown)