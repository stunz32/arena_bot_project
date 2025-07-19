"""
Simple configuration management for Arena Bot.

Loads settings from JSON files and provides default values.
Following CLAUDE.md principles - minimal and focused.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any


class Config:
    """Simple configuration container."""
    
    def __init__(self, config_data: Dict[str, Any]):
        self._data = config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self._data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data


def load_config() -> Config:
    """
    Load configuration from file with sensible defaults.
    
    Returns:
        Config instance with loaded settings
    """
    logger = logging.getLogger(__name__)
    
    # Default configuration
    default_config = {
        "screen": {
            "capture_method": "pyqt6",
            "detection_timeout": 100,  # milliseconds
            "debug_screenshots": False
        },
        "detection": {
            "histogram_bins": [50, 60],  # H, S bins from Arena Tracker
            "confidence_threshold": 0.35,  # Arena Tracker's threshold
            "template_threshold_mana": 4.5,  # Arena Tracker's L2 threshold
            "template_threshold_rarity": 9.0,  # Arena Tracker's L2 threshold
            "max_candidates": 15  # Arena Tracker's limit
        },
        "modes": {
            "support_underground": True,
            "redraft_enabled": True,
            "auto_detect_mode": True
        },
        "ui": {
            "overlay_enabled": True,
            "show_confidence": True,
            "recommendation_count": 3
        },
        "debug": {
            "save_screenshots": False,
            "log_level": "INFO",
            "verbose_detection": False
        }
    }
    
    # Try to load from file
    config_file = Path(__file__).parent.parent.parent / "config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults (file config takes precedence)
            merged_config = default_config.copy()
            _deep_merge(merged_config, file_config)
            
            logger.info(f"Configuration loaded from {config_file}")
            return Config(merged_config)
            
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
            logger.info("Using default configuration")
    
    else:
        logger.info("No config file found, using defaults")
    
    return Config(default_config)


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary to merge into
        update: Dictionary with updates
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_config(config: Config, file_path: Path = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        file_path: Optional custom file path
    """
    if file_path is None:
        file_path = Path(__file__).parent.parent.parent / "config.json"
    
    try:
        with open(file_path, 'w') as f:
            json.dump(config._data, f, indent=2)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration saved to {file_path}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save config: {e}")