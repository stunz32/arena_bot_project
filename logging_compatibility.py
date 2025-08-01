#!/usr/bin/env python3
"""
Backwards Compatibility Layer for S-Tier Logging Integration

Provides seamless migration path from standard Python logging to S-Tier logging
with zero breaking changes during the transition period. Implements sync/async
dual compatibility with intelligent fallback mechanisms.

Features:
- Drop-in replacement for logging.getLogger()
- Automatic async/sync context detection
- Thread-safe logger caching
- Performance-optimized wrapper layer
- Graceful degradation when S-Tier unavailable
"""

import asyncio
import logging
import threading
import time
import weakref
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps, lru_cache
from contextlib import contextmanager
import inspect

# Import S-Tier logging with fallback
try:
    from arena_bot.logging_system import (
        get_s_tier_logger, 
        setup_s_tier_logging,
        get_system_health
    )
    STIER_AVAILABLE = True
except ImportError:
    STIER_AVAILABLE = False


class CompatibilityLogger:
    """
    Backwards-compatible logger wrapper that provides both sync and async interfaces.
    
    Automatically detects execution context and routes to appropriate logging method
    while maintaining full compatibility with standard Python logging interface.
    """
    
    def __init__(self, name: str, fallback_logger: Optional[logging.Logger] = None):
        """
        Initialize compatibility logger.
        
        Args:
            name: Logger name
            fallback_logger: Standard Python logger for fallback
        """
        self.name = name
        self.fallback_logger = fallback_logger or logging.getLogger(name)
        self._stier_logger: Optional[Any] = None
        self._stier_logger_task: Optional[asyncio.Task] = None
        self._initialization_lock = threading.Lock()
        self._initialized = False
        
        # Performance tracking
        self._call_count = 0
        self._async_call_count = 0
        self._sync_call_count = 0
        self._error_count = 0
    
    def _is_async_context(self) -> bool:
        """Check if we're running in an async context."""
        try:
            loop = asyncio.get_running_loop()
            return loop is not None
        except RuntimeError:
            return False
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the calling code for context enrichment."""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller (skip wrapper methods)
            for _ in range(4):  # Skip _get_caller_info, log method, wrapper, actual caller
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                return {
                    'filename': frame.f_code.co_filename,
                    'function': frame.f_code.co_name,
                    'line_number': frame.f_lineno,
                    'module': frame.f_globals.get('__name__', 'unknown')
                }
        finally:
            del frame  # Avoid reference cycles
        
        return {}
    
    async def _ensure_stier_logger(self) -> Optional[Any]:
        """Ensure S-Tier logger is initialized."""
        if not STIER_AVAILABLE:
            return None
        
        if self._stier_logger is not None:
            return self._stier_logger
        
        with self._initialization_lock:
            if self._stier_logger is not None:
                return self._stier_logger
            
            try:
                self._stier_logger = await get_s_tier_logger(self.name)
                self._initialized = True
                return self._stier_logger
            except Exception as e:
                # Log initialization error with fallback logger
                self.fallback_logger.warning(f"Failed to initialize S-Tier logger for {self.name}: {e}")
                return None
    
    def _sanitize_message_for_windows(self, message: str) -> str:
        """
        Sanitize log messages for Windows console compatibility.
        
        Replaces Unicode emojis with ASCII equivalents to prevent
        UnicodeEncodeError on Windows systems using CP1252 encoding.
        """
        import sys
        
        # Only sanitize on Windows systems
        if sys.platform != 'win32':
            return message
        
        # Emoji to ASCII mapping for Windows compatibility
        emoji_replacements = {
            '🎯': '[TARGET]',
            '✅': '[OK]', 
            '🚀': '[ROCKET]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🔍': '[SEARCH]',
            '📊': '[CHART]',
            '🧠': '[BRAIN]',
            '⚡': '[LIGHTNING]',
            '🖥️': '[DESKTOP]',
            '📁': '[FOLDER]',
            '🔄': '[REFRESH]',
            '💔': '[BROKEN_HEART]',
            '🎮': '[GAME]',
            '⏳': '[HOURGLASS]',
            '🌉': '[BRIDGE]',
            '🏗️': '[CONSTRUCTION]',
            '💡': '[BULB]',
            '🔧': '[WRENCH]',
            '📝': '[MEMO]',
            '📸': '[CAMERA]',
            '🐛': '[BUG]',
            '🏁': '[CHECKERED_FLAG]',
            '💥': '[BOOM]',
            '🛡️': '[SHIELD]',
            '🔒': '[LOCK]',
            '🎉': '[PARTY]',
            '✨': '[SPARKLES]'
        }
        
        # Replace emojis with ASCII equivalents
        sanitized_message = message
        for emoji, replacement in emoji_replacements.items():
            sanitized_message = sanitized_message.replace(emoji, replacement)
        
        return sanitized_message
    
    def _create_log_method(self, level_name: str, level_value: int):
        """Create a logging method that works in both sync and async contexts."""
        
        async def async_log_method(self, message: str, *args, **kwargs):
            """Async version of log method."""
            self._async_call_count += 1
            self._call_count += 1
            
            try:
                # Sanitize message for Windows compatibility
                sanitized_message = self._sanitize_message_for_windows(message)
                
                # Try S-Tier logging first
                stier_logger = await self._ensure_stier_logger()
                if stier_logger:
                    # Enrich context with caller information
                    extra = kwargs.get('extra', {})
                    if not extra.get('caller_info'):
                        extra['caller_info'] = self._get_caller_info()
                        kwargs['extra'] = extra
                    
                    # Call S-Tier logger
                    method = getattr(stier_logger, level_name.lower())
                    await method(sanitized_message, *args, **kwargs)
                    return
            except Exception as e:
                self._error_count += 1
                # Continue to fallback
            
            # Fallback to standard logging with sanitized message
            try:
                sanitized_message = self._sanitize_message_for_windows(message)
                method = getattr(self.fallback_logger, level_name.lower())
                method(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                # Ultimate fallback: strip all non-ASCII characters
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                method = getattr(self.fallback_logger, level_name.lower())
                method(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
        
        def sync_log_method(self, message: str, *args, **kwargs):
            """Sync version of log method."""
            self._sync_call_count += 1
            self._call_count += 1
            
            # Sanitize message for Windows compatibility
            try:
                sanitized_message = self._sanitize_message_for_windows(message)
                method = getattr(self.fallback_logger, level_name.lower())
                method(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                # Ultimate fallback: strip all non-ASCII characters
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                method = getattr(self.fallback_logger, level_name.lower())
                method(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
        
        def universal_log_method(self, message: str, *args, **kwargs):
            """Universal log method that detects context and routes appropriately."""
            if self._is_async_context():
                # We're in async context, but method was called synchronously
                # Schedule the async version
                try:
                    loop = asyncio.get_running_loop()
                    # Create task for async logging (fire and forget)
                    task = loop.create_task(async_log_method(self, message, *args, **kwargs))
                    # Store reference to prevent garbage collection
                    self._stier_logger_task = task
                except Exception:
                    # Fall back to sync method with Windows compatibility
                    sync_log_method(self, message, *args, **kwargs)
            else:
                # Use sync method with Windows compatibility
                sync_log_method(self, message, *args, **kwargs)
        
        return universal_log_method
    
    def __getattr__(self, name: str) -> Any:
        """Dynamic attribute access for logging methods."""
        # Handle standard logging methods
        if name in ('debug', 'info', 'warning', 'warn', 'error', 'exception', 'critical', 'fatal'):
            level_value = getattr(logging, name.upper() if name != 'warn' else 'WARNING')
            method = self._create_log_method(name, level_value)
            # Cache the method
            setattr(self, name, method.__get__(self, type(self)))
            return getattr(self, name)
        
        # Handle other attributes by delegating to fallback logger
        return getattr(self.fallback_logger, name)
    
    # Async versions of logging methods
    async def adebug(self, message: str, *args, **kwargs):
        """Async debug logging."""
        sanitized_message = self._sanitize_message_for_windows(message)
        stier_logger = await self._ensure_stier_logger()
        if stier_logger:
            await stier_logger.debug(sanitized_message, *args, **kwargs)
        else:
            try:
                self.fallback_logger.debug(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                self.fallback_logger.debug(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
    
    async def ainfo(self, message: str, *args, **kwargs):
        """Async info logging."""
        sanitized_message = self._sanitize_message_for_windows(message)
        stier_logger = await self._ensure_stier_logger()
        if stier_logger:
            await stier_logger.info(sanitized_message, *args, **kwargs)
        else:
            try:
                self.fallback_logger.info(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                self.fallback_logger.info(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
    
    async def awarning(self, message: str, *args, **kwargs):
        """Async warning logging."""
        sanitized_message = self._sanitize_message_for_windows(message)
        stier_logger = await self._ensure_stier_logger()
        if stier_logger:
            await stier_logger.warning(sanitized_message, *args, **kwargs)
        else:
            try:
                self.fallback_logger.warning(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                self.fallback_logger.warning(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
    
    async def aerror(self, message: str, *args, **kwargs):
        """Async error logging."""
        sanitized_message = self._sanitize_message_for_windows(message)
        stier_logger = await self._ensure_stier_logger()
        if stier_logger:
            await stier_logger.error(sanitized_message, *args, **kwargs)
        else:
            try:
                self.fallback_logger.error(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                self.fallback_logger.error(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
    
    async def acritical(self, message: str, *args, **kwargs):
        """Async critical logging."""
        sanitized_message = self._sanitize_message_for_windows(message)
        stier_logger = await self._ensure_stier_logger()
        if stier_logger:
            await stier_logger.critical(sanitized_message, *args, **kwargs)
        else:
            try:
                self.fallback_logger.critical(sanitized_message, *args, **kwargs)
            except UnicodeEncodeError:
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                self.fallback_logger.critical(f"[ENCODING_SANITIZED] {ascii_message}", *args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this logger."""
        return {
            'name': self.name,
            'total_calls': self._call_count,
            'async_calls': self._async_call_count,
            'sync_calls': self._sync_call_count,
            'error_count': self._error_count,
            'stier_available': STIER_AVAILABLE,
            'stier_initialized': self._initialized
        }


class LoggerCache:
    """Thread-safe cache for compatibility loggers."""
    
    def __init__(self):
        self._cache: Dict[str, CompatibilityLogger] = {}
        self._lock = threading.Lock()
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_loggers': 0
        }
    
    def get_logger(self, name: str) -> CompatibilityLogger:
        """Get or create a compatibility logger."""
        with self._lock:
            if name in self._cache:
                self._stats['cache_hits'] += 1
                return self._cache[name]
            
            # Create new logger
            logger = CompatibilityLogger(name)
            self._cache[name] = logger
            self._stats['cache_misses'] += 1
            self._stats['total_loggers'] += 1
            
            return logger
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats.copy(),
                'cache_size': len(self._cache),
                'hit_rate': (
                    self._stats['cache_hits'] / 
                    (self._stats['cache_hits'] + self._stats['cache_misses'])
                    if self._stats['cache_hits'] + self._stats['cache_misses'] > 0 
                    else 0
                )
            }
    
    def clear_stats(self):
        """Clear statistics (keep cache)."""
        with self._lock:
            self._stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_loggers': len(self._cache)
            }


# Global logger cache instance
_logger_cache = LoggerCache()


def get_logger(name: str) -> CompatibilityLogger:
    """
    Get a backwards-compatible logger that works with both sync and async code.
    
    This is a drop-in replacement for logging.getLogger() that automatically
    routes to S-Tier logging when available and falls back to standard logging.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        CompatibilityLogger instance
        
    Example:
        # Sync usage (existing code works unchanged)
        logger = get_logger(__name__)
        logger.info("This works in sync context")
        
        # Async usage (new enhanced capabilities)
        logger = get_logger(__name__)
        await logger.ainfo("This works in async context with S-Tier features")
    """
    return _logger_cache.get_logger(name)


async def get_async_logger(name: str) -> CompatibilityLogger:
    """
    Get a logger optimized for async contexts.
    
    Args:
        name: Logger name
        
    Returns:
        CompatibilityLogger with S-Tier logger pre-initialized
    """
    logger = _logger_cache.get_logger(name)
    # Pre-initialize S-Tier logger for better performance
    await logger._ensure_stier_logger()
    return logger


def setup_compatibility_logging(config_path: Optional[str] = None) -> None:
    """
    Setup compatibility logging system.
    
    Args:
        config_path: Path to S-Tier logging configuration file
    """
    if not STIER_AVAILABLE:
        # Setup standard logging as fallback
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        print("⚠️  S-Tier logging unavailable, using standard logging")
        return
    
    # S-Tier logging will be initialized by the main application
    print("✅ Compatibility logging system ready")


async def setup_async_compatibility_logging(config_path: Optional[str] = None) -> None:
    """
    Setup compatibility logging system with async S-Tier initialization.
    
    Args:
        config_path: Path to S-Tier logging configuration file
    """
    if not STIER_AVAILABLE:
        setup_compatibility_logging(config_path)
        return
    
    try:
        # Initialize S-Tier logging system
        await setup_s_tier_logging(
            config_path=config_path or "arena_bot_logging_config.toml",
            environment="production",
            enable_performance_monitoring=True,
            async_enabled=True
        )
        print("✅ S-Tier logging system initialized with compatibility layer")
    except Exception as e:
        print(f"⚠️  S-Tier logging initialization failed: {e}")
        setup_compatibility_logging(config_path)


def get_compatibility_stats() -> Dict[str, Any]:
    """Get statistics for the compatibility logging system."""
    cache_stats = _logger_cache.get_stats()
    
    # Get individual logger stats
    logger_stats = {}
    for name, logger in _logger_cache._cache.items():
        logger_stats[name] = logger.get_stats()
    
    return {
        'cache_stats': cache_stats,
        'logger_stats': logger_stats,
        'stier_available': STIER_AVAILABLE
    }


# Decorator for automatic async logging context
def with_async_logging(logger_name: Optional[str] = None):
    """
    Decorator to automatically setup async logging context for functions.
    
    Args:
        logger_name: Optional logger name (defaults to function module)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get logger for this function
            name = logger_name or func.__module__
            logger = await get_async_logger(name)
            
            # Add logger to function kwargs if not present
            if 'logger' not in kwargs and 'logger' not in inspect.signature(func).parameters:
                # Store logger in a way the function can access it
                func._async_logger = logger
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await logger.aerror(f"Function {func.__name__} failed", extra={
                    'function_name': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, exc_info=True)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get logger for this function
            name = logger_name or func.__module__
            logger = get_logger(name)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed", extra={
                    'function_name': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, exc_info=True)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Module exports
__all__ = [
    'CompatibilityLogger',
    'get_logger',
    'get_async_logger', 
    'setup_compatibility_logging',
    'setup_async_compatibility_logging',
    'get_compatibility_stats',
    'with_async_logging'
]