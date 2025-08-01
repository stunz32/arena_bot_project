"""
Configuration Utilities for S-Tier Logging System.

This module provides utility functions for configuration management including
file loading, merging, and data transformation utilities.

Features:
- TOML/JSON/YAML configuration file loading
- Configuration merging and inheritance
- Data structure transformation utilities
- Performance-optimized configuration operations
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TextIO
from collections import defaultdict

# Try to import TOML support (Python 3.11+ has native support)
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python versions
    except ImportError:
        tomllib = None

# Try to import YAML support
try:
    import yaml
except ImportError:
    yaml = None


logger = logging.getLogger(__name__)


def load_toml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from TOML file.
    
    Args:
        file_path: Path to TOML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If TOML parsing fails
        RuntimeError: If TOML library is not available
    """
    if tomllib is None:
        raise RuntimeError("TOML library not available. Install tomli or use Python 3.11+")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            config = tomllib.load(f)
        
        logger.debug(f"Loaded TOML configuration from {file_path}")
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to parse TOML configuration {file_path}: {e}")


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If JSON parsing fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.debug(f"Loaded JSON configuration from {file_path}")
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to parse JSON configuration {file_path}: {e}")


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        file_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If YAML parsing fails
        RuntimeError: If YAML library is not available
    """
    if yaml is None:
        raise RuntimeError("YAML library not available. Install PyYAML")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"Loaded YAML configuration from {file_path}")
        return config or {}
        
    except Exception as e:
        raise ValueError(f"Failed to parse YAML configuration {file_path}: {e}")


def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration file with automatic format detection.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If file format is unsupported or parsing fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.toml':
        return load_toml_config(file_path)
    elif suffix == '.json':
        return load_json_config(file_path)
    elif suffix in ['.yaml', '.yml']:
        return load_yaml_config(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}")


def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries with deep merging.
    
    Later configurations override earlier ones. Lists are replaced, not merged.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}
    
    result = {}
    
    for config in configs:
        if not isinstance(config, dict):
            continue
        
        result = _deep_merge_dict(result, config)
    
    return result


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_config_dict(config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.
    
    Args:
        config: Configuration dictionary to flatten
        separator: Key separator for nested keys
        
    Returns:
        Flattened configuration dictionary
    """
    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    items.extend(_flatten(value, new_key).items())
                elif isinstance(value, list):
                    # Handle lists specially
                    for i, item in enumerate(value):
                        list_key = f"{new_key}[{i}]"
                        if isinstance(item, dict):
                            items.extend(_flatten(item, list_key).items())
                        else:
                            items.append((list_key, item))
                else:
                    items.append((new_key, value))
        else:
            items.append((parent_key, obj))
        
        return dict(items)
    
    return _flatten(config)


def unflatten_config_dict(flat_config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Unflatten a flattened configuration dictionary.
    
    Args:
        flat_config: Flattened configuration dictionary
        separator: Key separator used in flattening
        
    Returns:
        Unflattened configuration dictionary
    """
    result = {}
    
    for key, value in flat_config.items():
        # Handle array indices
        if '[' in key and ']' in key:
            # This is a more complex case, for now just use the key as-is
            result[key] = value
            continue
        
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None, separator: str = '.') -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        default: Default value if key not found
        separator: Key separator
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split(separator)
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any, separator: str = '.') -> None:
    """
    Set configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated key path
        value: Value to set
        separator: Key separator
    """
    keys = key_path.split(separator)
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def validate_config_structure(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """
    Validate that configuration has required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key paths (dot notation)
        
    Returns:
        List of missing keys
    """
    missing = []
    
    for key_path in required_keys:
        if get_config_value(config, key_path) is None:
            missing.append(key_path)
    
    return missing


def get_environment_config(prefix: str = 'LOGGING_') -> Dict[str, Any]:
    """
    Extract configuration from environment variables.
    
    Args:
        prefix: Environment variable prefix
        
    Returns:
        Configuration dictionary from environment variables
    """
    config = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            
            # Convert underscores to dots for nested keys
            config_key = config_key.replace('_', '.')
            
            # Try to parse value as JSON for complex types
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Keep as string if not valid JSON
                parsed_value = value
            
            set_config_value(config, config_key, parsed_value)
    
    return config


def save_config_file(config: Dict[str, Any], file_path: Union[str, Path], format: str = 'json') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Output file path
        format: Output format ('json', 'yaml', 'toml')
        
    Raises:
        ValueError: If format is unsupported
        RuntimeError: If required library is not available
    """
    file_path = Path(file_path)
    
    if format.lower() == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif format.lower() in ['yaml', 'yml']:
        if yaml is None:
            raise RuntimeError("YAML library not available. Install PyYAML")
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif format.lower() == 'toml':
        # Note: Writing TOML requires additional library (tomli-w or similar)
        raise NotImplementedError("TOML writing not implemented yet")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.debug(f"Saved configuration to {file_path} in {format} format")


# Module exports
__all__ = [
    'load_toml_config',
    'load_json_config', 
    'load_yaml_config',
    'load_config_file',
    'merge_configurations',
    'flatten_config_dict',
    'unflatten_config_dict',
    'get_config_value',
    'set_config_value',
    'validate_config_structure',
    'get_environment_config',
    'save_config_file'
]