"""
Migration Helper for S-Tier Logging System Configuration.

This module provides utilities for migrating between configuration versions
and handling configuration schema evolution.

Features:
- Configuration schema migration
- Backward compatibility handling
- Configuration format conversion
- Migration validation and rollback
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MigrationHelper:
    """
    Helper class for configuration migrations and schema evolution.
    
    Provides utilities for migrating configurations between versions,
    converting formats, and maintaining backward compatibility.
    """
    
    # Migration rules for schema versions
    MIGRATION_RULES = {
        "0.9": {
            "target": "1.0",
            "rules": [
                {"action": "rename", "from": "async_enabled", "to": "performance.enable_async"},
                {"action": "move", "from": "log_level", "to": "loggers.root.level"},
                {"action": "add", "key": "version", "value": "1.0"},
                {"action": "add", "key": "system.name", "value": "s-tier-logging"}
            ]
        }
    }
    
    @classmethod
    def detect_version(cls, config: Dict[str, Any]) -> str:
        """
        Detect configuration version.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Detected version string
        """
        # Check for explicit version field
        if "version" in config:
            return config["version"]
        
        # Try to detect version from structure
        if "async_enabled" in config:
            return "0.9"
        
        if "system" in config and "performance" in config:
            return "1.0"
        
        # Default to oldest version
        return "0.9"
    
    @classmethod
    def needs_migration(cls, config: Dict[str, Any], target_version: str = "1.0") -> bool:
        """
        Check if configuration needs migration.
        
        Args:
            config: Configuration dictionary
            target_version: Target version to check against
            
        Returns:
            True if migration is needed
        """
        current_version = cls.detect_version(config)
        return current_version != target_version
    
    @classmethod
    def migrate_config(
        cls, 
        config: Dict[str, Any], 
        target_version: str = "1.0",
        backup: bool = True
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate configuration to target version.
        
        Args:
            config: Configuration dictionary to migrate
            target_version: Target version
            backup: Whether to create backup
            
        Returns:
            Tuple of (migrated_config, list_of_warnings)
        """
        current_version = cls.detect_version(config)
        warnings = []
        
        if current_version == target_version:
            logger.info(f"Configuration already at version {target_version}")
            return config.copy(), warnings
        
        logger.info(f"Migrating configuration from {current_version} to {target_version}")
        
        # Create backup if requested
        if backup:
            cls._create_backup(config, current_version)
        
        # Apply migration rules
        migrated_config = config.copy()
        
        try:
            migrated_config, migration_warnings = cls._apply_migration_rules(
                migrated_config, current_version, target_version
            )
            warnings.extend(migration_warnings)
            
            # Validate migrated configuration
            validation_warnings = cls._validate_migrated_config(migrated_config)
            warnings.extend(validation_warnings)
            
            logger.info(f"Configuration migration completed: {current_version} -> {target_version}")
            
        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
            raise
        
        return migrated_config, warnings
    
    @classmethod
    def _apply_migration_rules(
        cls,
        config: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Apply migration rules for version change."""
        warnings = []
        
        if from_version not in cls.MIGRATION_RULES:
            warning = f"No migration rules found for version {from_version}"
            warnings.append(warning)
            logger.warning(warning)
            return config, warnings
        
        rules = cls.MIGRATION_RULES[from_version]["rules"]
        
        for rule in rules:
            try:
                action = rule["action"]
                
                if action == "rename":
                    cls._apply_rename_rule(config, rule)
                elif action == "move":
                    cls._apply_move_rule(config, rule)
                elif action == "add":
                    cls._apply_add_rule(config, rule)
                elif action == "remove":
                    cls._apply_remove_rule(config, rule)
                elif action == "transform":
                    cls._apply_transform_rule(config, rule)
                else:
                    warning = f"Unknown migration action: {action}"
                    warnings.append(warning)
                    logger.warning(warning)
                    
            except Exception as e:
                warning = f"Failed to apply migration rule {rule}: {e}"
                warnings.append(warning)
                logger.warning(warning)
        
        return config, warnings
    
    @classmethod
    def _apply_rename_rule(cls, config: Dict[str, Any], rule: Dict[str, str]) -> None:
        """Apply rename migration rule."""
        from_key = rule["from"]
        to_key = rule["to"]
        
        if from_key in config:
            value = config.pop(from_key)
            cls._set_nested_value(config, to_key, value)
            logger.debug(f"Renamed {from_key} to {to_key}")
    
    @classmethod
    def _apply_move_rule(cls, config: Dict[str, Any], rule: Dict[str, str]) -> None:
        """Apply move migration rule."""
        from_key = rule["from"]
        to_key = rule["to"]
        
        value = cls._get_nested_value(config, from_key)
        if value is not None:
            cls._remove_nested_value(config, from_key)
            cls._set_nested_value(config, to_key, value)
            logger.debug(f"Moved {from_key} to {to_key}")
    
    @classmethod
    def _apply_add_rule(cls, config: Dict[str, Any], rule: Dict[str, Any]) -> None:
        """Apply add migration rule."""
        key = rule["key"]
        value = rule["value"]
        
        if cls._get_nested_value(config, key) is None:
            cls._set_nested_value(config, key, value)
            logger.debug(f"Added {key} = {value}")
    
    @classmethod
    def _apply_remove_rule(cls, config: Dict[str, Any], rule: Dict[str, str]) -> None:
        """Apply remove migration rule."""
        key = rule["key"]
        
        if cls._get_nested_value(config, key) is not None:
            cls._remove_nested_value(config, key)
            logger.debug(f"Removed {key}")
    
    @classmethod
    def _apply_transform_rule(cls, config: Dict[str, Any], rule: Dict[str, Any]) -> None:
        """Apply transform migration rule."""
        key = rule["key"]
        transform_func = rule["transform"]
        
        value = cls._get_nested_value(config, key)
        if value is not None:
            transformed_value = transform_func(value)
            cls._set_nested_value(config, key, transformed_value)
            logger.debug(f"Transformed {key}")
    
    @classmethod
    def _validate_migrated_config(cls, config: Dict[str, Any]) -> List[str]:
        """Validate migrated configuration."""
        warnings = []
        
        # Check required fields
        required_fields = ["version", "system.name"]
        
        for field in required_fields:
            if cls._get_nested_value(config, field) is None:
                warning = f"Missing required field after migration: {field}"
                warnings.append(warning)
        
        return warnings
    
    @classmethod
    def _create_backup(cls, config: Dict[str, Any], version: str) -> Path:
        """Create backup of configuration before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"config_backup_{version}_{timestamp}.json"
        backup_path = Path(backup_filename)
        
        try:
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            raise
    
    @classmethod
    def _set_nested_value(cls, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    @classmethod
    def _get_nested_value(cls, config: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation."""
        keys = key.split('.')
        current = config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return None
    
    @classmethod
    def _remove_nested_value(cls, config: Dict[str, Any], key: str) -> bool:
        """Remove nested value using dot notation."""
        keys = key.split('.')
        current = config
        
        try:
            for k in keys[:-1]:
                current = current[k]
            
            if keys[-1] in current:
                del current[keys[-1]]
                return True
            
        except (KeyError, TypeError):
            pass
        
        return False
    
    @classmethod
    def convert_format(
        cls,
        config: Dict[str, Any],
        from_format: str,
        to_format: str
    ) -> str:
        """
        Convert configuration between formats.
        
        Args:
            config: Configuration dictionary
            from_format: Source format
            to_format: Target format
            
        Returns:
            Configuration in target format as string
        """
        if to_format.lower() == "json":
            return json.dumps(config, indent=2)
        elif to_format.lower() in ["yaml", "yml"]:
            try:
                import yaml
                return yaml.dump(config, default_flow_style=False)
            except ImportError:
                raise RuntimeError("PyYAML not available for YAML conversion")
        elif to_format.lower() == "toml":
            # Note: TOML writing would require additional library
            raise NotImplementedError("TOML output conversion not implemented")
        else:
            raise ValueError(f"Unsupported target format: {to_format}")
    
    @classmethod
    def get_migration_summary(cls, from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Get summary of migration changes.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            Migration summary dictionary
        """
        if from_version not in cls.MIGRATION_RULES:
            return {
                "from_version": from_version,
                "to_version": to_version,
                "supported": False,
                "changes": []
            }
        
        rules = cls.MIGRATION_RULES[from_version]["rules"]
        
        return {
            "from_version": from_version,
            "to_version": to_version,
            "supported": True,
            "changes": [
                f"{rule['action']}: {rule.get('from', rule.get('key', 'unknown'))}"
                for rule in rules
            ]
        }


# Module exports
__all__ = [
    'MigrationHelper'
]