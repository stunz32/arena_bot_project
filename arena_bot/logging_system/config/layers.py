"""
Configuration Layers for S-Tier Logging System.

This module implements hierarchical configuration layers that support
inheritance, override patterns, and dynamic configuration sources
for enterprise-grade configuration management.

Features:
- Abstract base layer with priority-based ordering
- File-based configuration with hot-reload detection
- Environment variable integration with nested key support
- Runtime configuration overrides
- Default configuration fallbacks
- Performance-optimized configuration merging
"""

import os
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Python 3.11+ TOML support or fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# YAML support (optional)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# JSON support (always available)
import json


class ConfigurationLayer(ABC):
    """
    Abstract base class for configuration layers.
    
    Provides the interface for hierarchical configuration management
    with priority-based ordering and merge capabilities.
    """
    
    def __init__(self, name: str, priority: int = 0):
        """
        Initialize configuration layer.
        
        Args:
            name: Human-readable layer name for debugging
            priority: Layer priority (higher = more important)
        """
        self.name = name
        self.priority = priority
        self._last_update = time.time()
        self._lock = threading.RLock()
    
    @abstractmethod
    def get_values(self) -> Dict[str, Any]:
        """
        Get configuration values from this layer.
        
        Returns:
            Dictionary of configuration values
        """
        pass
    
    def get_priority(self) -> int:
        """Get layer priority for ordering."""
        return self.priority
    
    def get_name(self) -> str:
        """Get layer name for debugging."""
        return self.name
    
    def get_last_update(self) -> float:
        """Get timestamp of last update."""
        with self._lock:
            return self._last_update
    
    def _mark_updated(self) -> None:
        """Mark layer as updated."""
        with self._lock:
            self._last_update = time.time()
    
    def is_available(self) -> bool:
        """Check if layer is available and can provide values."""
        try:
            self.get_values()
            return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of layer."""
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"priority={self.priority}, available={self.is_available()})")


class DefaultConfigLayer(ConfigurationLayer):
    """
    Default configuration layer providing fallback values.
    
    This layer provides sensible defaults for the S-tier logging system
    and serves as the foundation for all other configuration layers.
    """
    
    def __init__(self, name: str = "defaults", priority: int = 0):
        super().__init__(name, priority)
        self._default_values = self._build_default_values()
    
    def get_values(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return self._default_values.copy()
    
    def _build_default_values(self) -> Dict[str, Any]:
        """Build comprehensive default configuration."""
        return {
            "version": 1,
            "environment": "development",
            "system_name": "s-tier-logging",
            "instance_id": f"instance-{os.getpid()}",
            
            # Default loggers
            "loggers": {
                "root": {
                    "name": "root",
                    "level": "INFO",
                    "propagate": False,
                    "handlers": ["console"],
                    "sinks": ["console"],
                    "filters": []
                },
                "app": {
                    "name": "app",
                    "level": "INFO", 
                    "propagate": True,
                    "handlers": ["console"],
                    "sinks": ["console"],
                    "filters": []
                }
            },
            
            # Default handlers (backward compatibility)
            "handlers": {
                "console": {
                    "name": "console",
                    "type": "console",
                    "level": "INFO",
                    "stream": "stdout"
                }
            },
            
            # Default sinks
            "sinks": {
                "console": {
                    "name": "console",
                    "type": "console",
                    "enabled": True,
                    "config": {
                        "colorize": True,
                        "show_timestamp": True
                    },
                    "filters": [],
                    "formatter": "console"
                }
            },
            
            # Default filters
            "filters": {
                "level": {
                    "name": "level",
                    "type": "level",
                    "enabled": True,
                    "priority": 10,
                    "config": {
                        "min_level": "INFO"
                    }
                }
            },
            
            # Security configuration
            "security": {
                "enable_pii_detection": False,
                "pii_redaction_level": "partial", 
                "redact_credentials": True,
                "redact_ip_addresses": False,
                "require_authentication": False,
                "allowed_users": [],
                "allowed_roles": [],
                "enable_audit_trail": False,
                "audit_log_path": None,
                "compliance_mode": "none",
                "encryption": {
                    "enabled": False,
                    "algorithm": "fernet",
                    "key_file_path": None,
                    "key_rotation_days": 90,
                    "encrypt_in_transit": False,
                    "encrypt_at_rest": False
                },
                "retention": {
                    "max_age_days": 30,
                    "max_size_gb": 5.0,
                    "max_files": 100,
                    "compression_after_days": 7,
                    "archive_after_days": 14,
                    "delete_after_days": 30
                }
            },
            
            # Performance configuration
            "performance": {
                "enable_async_processing": False,
                "async_queue_size": 1000,
                "worker_threads": 2,
                "buffer_size": 4096,
                "batch_size": 10,
                "flush_interval_seconds": 5.0,
                "enable_caching": True,
                "cache_size_limit": 1000,
                "cache_ttl_seconds": 300,
                "max_memory_mb": 512,
                "max_disk_usage_gb": 5.0,
                "enable_performance_monitoring": False,
                "performance_sampling_rate": 0.01
            },
            
            # Diagnostics configuration
            "diagnostics": {
                "enable_health_checks": True,
                "health_check_interval_seconds": 60,
                "health_check_timeout_seconds": 5,
                "enable_metrics": False,
                "metrics_export_interval_seconds": 300,
                "metrics_retention_hours": 24,
                "enable_emergency_protocols": False,
                "emergency_activation_threshold": 0.9,
                "emergency_cooldown_minutes": 10,
                "monitor_memory_usage": True,
                "monitor_disk_usage": True,
                "monitor_network_usage": False
            },
            
            # Configuration management
            "config_file_path": "logging_config.toml",
            "auto_reload": False,
            "reload_check_interval": 30.0
        }


class FileConfigLayer(ConfigurationLayer):
    """
    File-based configuration layer with hot-reload support.
    
    Supports TOML, YAML, and JSON configuration files with automatic
    format detection and change monitoring for hot-reload capabilities.
    """
    
    def __init__(self, 
                 config_path: Path, 
                 name: Optional[str] = None,
                 priority: int = 10,
                 auto_detect_format: bool = True,
                 file_format: Optional[str] = None):
        """
        Initialize file configuration layer.
        
        Args:
            config_path: Path to configuration file
            name: Layer name (defaults to filename)
            priority: Layer priority
            auto_detect_format: Automatically detect file format from extension
            file_format: Force specific format ('toml', 'yaml', 'json')
        """
        if name is None:
            name = f"file:{config_path.name}"
        
        super().__init__(name, priority)
        
        self.config_path = Path(config_path)
        self.auto_detect_format = auto_detect_format
        self.file_format = file_format
        
        # Cached values and file monitoring
        self._cached_values: Optional[Dict[str, Any]] = None
        self._file_mtime: Optional[float] = None
        self._file_size: Optional[int] = None
        
        # Determine file format
        if not self.file_format and self.auto_detect_format:
            self.file_format = self._detect_file_format()
        
        # Validate format support
        self._validate_format_support()
    
    def get_values(self) -> Dict[str, Any]:
        """Get configuration values from file with caching and change detection."""
        if not self.config_path.exists():
            return {}
        
        # Check if file has changed
        try:
            stat_info = self.config_path.stat()
            current_mtime = stat_info.st_mtime
            current_size = stat_info.st_size
            
            if (self._file_mtime != current_mtime or 
                self._file_size != current_size or 
                self._cached_values is None):
                
                # Reload file
                self._cached_values = self._load_file()
                self._file_mtime = current_mtime
                self._file_size = current_size
                self._mark_updated()
                
        except Exception as e:
            # If we can't stat the file but have cached values, use them
            if self._cached_values is not None:
                return self._cached_values.copy()
            else:
                raise FileNotFoundError(f"Cannot read configuration file {self.config_path}: {e}")
        
        return self._cached_values.copy() if self._cached_values else {}
    
    def _detect_file_format(self) -> str:
        """Detect file format from extension."""
        suffix = self.config_path.suffix.lower()
        
        if suffix in ['.toml', '.tml']:
            return 'toml'
        elif suffix in ['.yaml', '.yml']:
            return 'yaml'
        elif suffix in ['.json']:
            return 'json'
        else:
            # Default to TOML for unknown extensions
            return 'toml'
    
    def _validate_format_support(self) -> None:
        """Validate that the required format parser is available."""
        if self.file_format == 'toml' and tomllib is None:
            raise ImportError("TOML support not available. Install tomli for Python < 3.11")
        elif self.file_format == 'yaml' and not YAML_AVAILABLE:
            raise ImportError("YAML support not available. Install PyYAML")
    
    def _load_file(self) -> Dict[str, Any]:
        """Load configuration from file based on format."""
        try:
            if self.file_format == 'toml':
                return self._load_toml()
            elif self.file_format == 'yaml':
                return self._load_yaml()
            elif self.file_format == 'json':
                return self._load_json()
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
                
        except Exception as e:
            raise ValueError(f"Failed to parse {self.file_format.upper()} file {self.config_path}: {e}")
    
    def _load_toml(self) -> Dict[str, Any]:
        """Load TOML configuration file."""
        with open(self.config_path, 'rb') as f:
            return tomllib.load(f)
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _load_json(self) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def is_file_changed(self) -> bool:
        """Check if file has changed since last load."""
        if not self.config_path.exists():
            return False
        
        try:
            stat_info = self.config_path.stat()
            return (stat_info.st_mtime != self._file_mtime or 
                    stat_info.st_size != self._file_size)
        except Exception:
            return False
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the configuration file."""
        if not self.config_path.exists():
            return {"exists": False}
        
        try:
            stat_info = self.config_path.stat()
            return {
                "exists": True,
                "path": str(self.config_path),
                "format": self.file_format,
                "size_bytes": stat_info.st_size,
                "modified_time": stat_info.st_mtime,
                "modified_datetime": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "is_readable": os.access(self.config_path, os.R_OK),
                "is_changed": self.is_file_changed()
            }
        except Exception as e:
            return {
                "exists": True,
                "path": str(self.config_path),
                "error": str(e)
            }


class EnvironmentConfigLayer(ConfigurationLayer):
    """
    Environment variable configuration layer.
    
    Supports nested configuration keys using delimiter patterns and
    automatic type conversion for environment variables.
    """
    
    def __init__(self, 
                 prefix: str = "LOGGING_", 
                 name: Optional[str] = None,
                 priority: int = 20,
                 delimiter: str = "__",
                 case_sensitive: bool = False):
        """
        Initialize environment variable configuration layer.
        
        Args:
            prefix: Environment variable prefix to filter on
            name: Layer name
            priority: Layer priority
            delimiter: Delimiter for nested keys (e.g., LOGGING_SECURITY__ENCRYPT_LOGS)
            case_sensitive: Whether to preserve case in keys
        """
        if name is None:
            name = f"env:{prefix.rstrip('_')}"
        
        super().__init__(name, priority)
        
        self.prefix = prefix
        self.delimiter = delimiter
        self.case_sensitive = case_sensitive
        
        # Cache environment variables
        self._cached_env_vars: Optional[Dict[str, str]] = None
        self._env_snapshot_time: Optional[float] = None
        self._cache_ttl = 60.0  # Cache environment for 60 seconds
    
    def get_values(self) -> Dict[str, Any]:
        """Get configuration values from environment variables."""
        current_time = time.time()
        
        # Refresh cache if needed
        if (self._cached_env_vars is None or 
            self._env_snapshot_time is None or
            current_time - self._env_snapshot_time > self._cache_ttl):
            
            self._cached_env_vars = self._capture_environment()
            self._env_snapshot_time = current_time
            self._mark_updated()
        
        # Convert environment variables to nested configuration
        config = {}
        for key, value in self._cached_env_vars.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):]
                if not self.case_sensitive:
                    config_key = config_key.lower()
                
                self._set_nested_value(config, config_key.split(self.delimiter), value)
        
        return config
    
    def _capture_environment(self) -> Dict[str, str]:
        """Capture current environment variables."""
        return dict(os.environ)
    
    def _set_nested_value(self, config: Dict[str, Any], keys: List[str], value: str) -> None:
        """Set nested dictionary value from key path."""
        current = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Handle conflict - convert to dict
                current[key] = {}
            current = current[key]
        
        # Set final value with type conversion
        final_key = keys[-1]
        current[final_key] = self._convert_env_value(value)
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool, List[str]]:
        """Convert environment variable string to appropriate Python type."""
        # Strip quotes if present
        value = value.strip('\'"')
        
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on', 'enabled'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off', 'disabled'):
            return False
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',') if item.strip()]
        
        # Numeric conversion
        try:
            # Try integer first
            if '.' not in value and 'e' not in value.lower():
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Default to string
        return value
    
    def get_env_vars(self) -> Dict[str, str]:
        """Get all environment variables matching the prefix."""
        if self._cached_env_vars is None:
            self._cached_env_vars = self._capture_environment()
        
        return {k: v for k, v in self._cached_env_vars.items() if k.startswith(self.prefix)}
    
    def refresh_cache(self) -> None:
        """Force refresh of environment variable cache."""
        self._cached_env_vars = None
        self._env_snapshot_time = None


class RuntimeConfigLayer(ConfigurationLayer):
    """
    Runtime configuration override layer.
    
    Allows programmatic configuration changes at runtime with
    thread-safe operations and change tracking.
    """
    
    def __init__(self, name: str = "runtime", priority: int = 30):
        """
        Initialize runtime configuration layer.
        
        Args:
            name: Layer name
            priority: Layer priority (highest by default)
        """
        super().__init__(name, priority)
        self._overrides: Dict[str, Any] = {}
        self._override_history: List[Dict[str, Any]] = []
        self._max_history = 100
    
    def get_values(self) -> Dict[str, Any]:
        """Get runtime configuration overrides."""
        with self._lock:
            return self._overrides.copy()
    
    def set_override(self, key: str, value: Any, reason: Optional[str] = None) -> None:
        """
        Set runtime configuration override.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Configuration value
            reason: Optional reason for the override
        """
        with self._lock:
            # Record change in history
            change_record = {
                "timestamp": time.time(),
                "key": key,
                "old_value": self._get_nested_value(self._overrides, key.split('.')),
                "new_value": value,
                "reason": reason
            }
            
            # Set the override
            self._set_nested_override(key.split('.'), value)
            
            # Update history
            self._override_history.append(change_record)
            if len(self._override_history) > self._max_history:
                self._override_history.pop(0)
            
            self._mark_updated()
    
    def remove_override(self, key: str, reason: Optional[str] = None) -> bool:
        """
        Remove runtime configuration override.
        
        Args:
            key: Configuration key to remove
            reason: Optional reason for removal
            
        Returns:
            True if override was removed, False if it didn't exist
        """
        with self._lock:
            old_value = self._get_nested_value(self._overrides, key.split('.'))
            
            if old_value is None:
                return False
            
            # Record removal in history
            change_record = {
                "timestamp": time.time(),
                "key": key,
                "old_value": old_value,
                "new_value": None,
                "reason": reason,
                "action": "remove"
            }
            
            # Remove the override
            removed = self._remove_nested_override(key.split('.'))
            
            if removed:
                self._override_history.append(change_record)
                if len(self._override_history) > self._max_history:
                    self._override_history.pop(0)
                
                self._mark_updated()
            
            return removed
    
    def clear_overrides(self, reason: Optional[str] = None) -> int:
        """
        Clear all runtime configuration overrides.
        
        Args:
            reason: Optional reason for clearing
            
        Returns:
            Number of overrides that were cleared
        """
        with self._lock:
            count = len(self._get_all_keys(self._overrides))
            
            if count > 0:
                # Record clearing in history
                change_record = {
                    "timestamp": time.time(),
                    "key": "*",
                    "old_value": self._overrides.copy(),
                    "new_value": {},
                    "reason": reason,
                    "action": "clear_all"
                }
                
                self._overrides.clear()
                self._override_history.append(change_record)
                
                if len(self._override_history) > self._max_history:
                    self._override_history.pop(0)
                
                self._mark_updated()
            
            return count
    
    def _set_nested_override(self, keys: List[str], value: Any) -> None:
        """Set nested override value."""
        current = self._overrides
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """Get nested value from dictionary."""
        current = data
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        
        return current
    
    def _remove_nested_override(self, keys: List[str]) -> bool:
        """Remove nested override value."""
        if len(keys) == 1:
            return self._overrides.pop(keys[0], None) is not None
        
        current = self._overrides
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]
            return True
        
        return False
    
    def _get_all_keys(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all keys from nested dictionary."""
        keys = []
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, full_key))
        
        return keys
    
    def get_override_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of configuration overrides.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of change records
        """
        with self._lock:
            history = self._override_history.copy()
            if limit is not None:
                history = history[-limit:]
            return history
    
    def get_override_count(self) -> int:
        """Get number of active overrides."""
        with self._lock:
            return len(self._get_all_keys(self._overrides))
    
    def has_override(self, key: str) -> bool:
        """Check if specific override exists."""
        with self._lock:
            return self._get_nested_value(self._overrides, key.split('.')) is not None


# Module exports
__all__ = [
    'ConfigurationLayer',
    'DefaultConfigLayer',
    'FileConfigLayer',
    'EnvironmentConfigLayer',
    'RuntimeConfigLayer'
]