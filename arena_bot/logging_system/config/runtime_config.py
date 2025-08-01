"""
Runtime Configuration for S-Tier Logging System.

This module provides runtime configuration capabilities for dynamic
configuration changes without system restart.

Features:
- Runtime configuration updates
- Hot-reload capabilities
- Thread-safe configuration changes
- Configuration validation during updates
"""

import logging
import threading
from typing import Any, Dict, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class RuntimeConfig:
    """
    Runtime configuration manager for dynamic configuration updates.
    
    Provides thread-safe configuration updates with validation and
    change notification capabilities.
    """
    
    def __init__(self):
        """Initialize runtime configuration manager."""
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._callbacks: Dict[str, Callable[[str, Any, Any], None]] = {}
        self._history: list = []
        self._max_history = 100
    
    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """
        Set configuration value at runtime.
        
        Args:
            key: Configuration key (dot notation supported)
            value: New value
            validate: Whether to validate the change
            
        Returns:
            True if successfully set, False otherwise
        """
        with self._lock:
            try:
                old_value = self.get(key)
                
                if validate and not self._validate_change(key, value):
                    logger.warning(f"Runtime config validation failed for {key}")
                    return False
                
                # Set the value
                self._set_nested_value(key, value)
                
                # Record change in history
                self._record_change(key, old_value, value)
                
                # Notify callbacks
                self._notify_callbacks(key, old_value, value)
                
                logger.info(f"Runtime config updated: {key} = {value}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set runtime config {key}: {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            return self._get_nested_value(key, default)
    
    def update(self, config_dict: Dict[str, Any], validate: bool = True) -> bool:
        """
        Update multiple configuration values.
        
        Args:
            config_dict: Dictionary of configuration updates
            validate: Whether to validate changes
            
        Returns:
            True if all updates succeeded, False otherwise
        """
        with self._lock:
            success = True
            changes = []
            
            try:
                # Collect all changes first
                for key, value in config_dict.items():
                    old_value = self.get(key)
                    changes.append((key, old_value, value))
                
                # Validate all changes if requested
                if validate:
                    for key, old_value, new_value in changes:
                        if not self._validate_change(key, new_value):
                            logger.warning(f"Validation failed for {key}, aborting update")
                            return False
                
                # Apply all changes
                for key, old_value, new_value in changes:
                    self._set_nested_value(key, new_value)
                    self._record_change(key, old_value, new_value)
                    self._notify_callbacks(key, old_value, new_value)
                
                logger.info(f"Runtime config batch update completed: {len(changes)} changes")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update runtime config: {e}")
                return False
    
    def register_callback(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Register callback for configuration changes.
        
        Args:
            key: Configuration key to watch
            callback: Callback function (key, old_value, new_value)
        """
        with self._lock:
            self._callbacks[key] = callback
            logger.debug(f"Registered runtime config callback for {key}")
    
    def unregister_callback(self, key: str) -> None:
        """
        Unregister callback for configuration changes.
        
        Args:
            key: Configuration key to stop watching
        """
        with self._lock:
            if key in self._callbacks:
                del self._callbacks[key]
                logger.debug(f"Unregistered runtime config callback for {key}")
    
    def get_history(self, limit: Optional[int] = None) -> list:
        """
        Get configuration change history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of change records
        """
        with self._lock:
            history = self._history.copy()
            if limit:
                history = history[-limit:]
            return history
    
    def clear_history(self) -> None:
        """Clear configuration change history."""
        with self._lock:
            self._history.clear()
            logger.debug("Runtime config history cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary.
        
        Returns:
            Current configuration dictionary
        """
        with self._lock:
            return self._config.copy()
    
    def _set_nested_value(self, key: str, value: Any) -> None:
        """Set value using dot notation."""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, key: str, default: Any = None) -> Any:
        """Get value using dot notation."""
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def _validate_change(self, key: str, value: Any) -> bool:
        """
        Validate configuration change.
        
        Args:
            key: Configuration key
            value: New value
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation rules
        validation_rules = {
            'performance.queue_size': lambda v: isinstance(v, int) and 100 <= v <= 1000000,
            'performance.worker_threads': lambda v: isinstance(v, int) and 1 <= v <= 100,
            'performance.max_latency_ms': lambda v: isinstance(v, (int, float)) and v > 0,
            'security.compliance_mode': lambda v: v in ['none', 'gdpr', 'hipaa', 'pci_dss'],
            'system.enabled': lambda v: isinstance(v, bool),
            'diagnostics.health_check_interval': lambda v: isinstance(v, (int, float)) and v > 0
        }
        
        if key in validation_rules:
            return validation_rules[key](value)
        
        # Default validation - check for basic types
        if isinstance(value, (str, int, float, bool, list, dict)):
            return True
        
        return False
    
    def _record_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Record configuration change in history."""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'old_value': old_value,
            'new_value': new_value
        }
        
        self._history.append(change_record)
        
        # Limit history size
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def _notify_callbacks(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify registered callbacks of configuration changes."""
        for callback_key, callback in self._callbacks.items():
            if key.startswith(callback_key) or callback_key == '*':
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Runtime config callback error for {callback_key}: {e}")


# Global runtime configuration instance
_runtime_config = None
_runtime_config_lock = threading.Lock()


def get_runtime_config() -> RuntimeConfig:
    """
    Get global runtime configuration instance.
    
    Returns:
        Global RuntimeConfig instance
    """
    global _runtime_config
    
    if _runtime_config is None:
        with _runtime_config_lock:
            if _runtime_config is None:
                _runtime_config = RuntimeConfig()
    
    return _runtime_config


def set_runtime_value(key: str, value: Any, validate: bool = True) -> bool:
    """
    Set runtime configuration value.
    
    Args:
        key: Configuration key
        value: New value
        validate: Whether to validate the change
        
    Returns:
        True if successfully set
    """
    return get_runtime_config().set(key, value, validate)


def get_runtime_value(key: str, default: Any = None) -> Any:
    """
    Get runtime configuration value.
    
    Args:
        key: Configuration key
        default: Default value
        
    Returns:
        Configuration value or default
    """
    return get_runtime_config().get(key, default)


# Module exports
__all__ = [
    'RuntimeConfig',
    'get_runtime_config',
    'set_runtime_value',
    'get_runtime_value'
]