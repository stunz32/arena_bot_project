"""
Configuration management module for Arena Bot AI Helper v2.

This module provides comprehensive configuration management with:
- Thread-safe configuration access and updates
- Atomic configuration transitions
- Configuration validation and migration
- Draft-aware configuration locking
- Hot-reload capability for development

Usage:
    from arena_bot.config import get_config, update_config, get_config_manager
    
    # Get configuration values
    ai_enabled = get_config("ai_helper.enabled", True)
    overlay_opacity = get_config("ui.overlay_opacity", 0.8)
    
    # Update configuration
    update_config({"ai_helper.confidence_threshold": 0.8})
    
    # Use configuration manager directly
    config_mgr = get_config_manager()
    with config_mgr.draft_lock():
        # Configuration is locked during draft
        pass

Author: Claude (Anthropic)
Created: 2025-07-28
"""

from .config_manager import (
    ConfigurationManager,
    ConfigSnapshot,
    ConfigState,
    get_config_manager,
    get_config,
    update_config
)

from .config_migration import (
    ConfigurationMigrator,
    MigrationStep,
    migrate_config_file
)

__all__ = [
    # Configuration Manager
    "ConfigurationManager",
    "ConfigSnapshot", 
    "ConfigState",
    "get_config_manager",
    "get_config",
    "update_config",
    
    # Migration System
    "ConfigurationMigrator",
    "MigrationStep",
    "migrate_config_file"
]

# Module version
__version__ = "2.0.0"