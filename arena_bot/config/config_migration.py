"""
Configuration Migration System for Arena Bot AI Helper v2.

This module handles configuration migration between different versions of the
AI Helper system, ensuring backward compatibility and smooth upgrades.

Features:
- Version-aware configuration migration
- Backward compatibility preservation
- Automatic schema validation and updates
- Safe migration with rollback capability
- Migration history tracking

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import semantic_version

from ..ai_v2.exceptions import AIHelperConfigurationError, AIHelperValidationError


@dataclass
class MigrationStep:
    """Represents a single migration step."""
    from_version: str
    to_version: str
    description: str
    migration_func: callable
    rollback_func: Optional[callable] = None


class ConfigurationMigrator:
    """
    Handles configuration migration between versions.
    
    This class provides safe, version-aware migration of configuration
    files between different versions of the AI Helper system.
    """
    
    CURRENT_VERSION = "2.0.0"
    MINIMUM_SUPPORTED_VERSION = "1.0.0"
    
    def __init__(self, config_dir: Path):
        """
        Initialize configuration migrator.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Migration steps registry
        self._migration_steps: List[MigrationStep] = []
        self._register_migration_steps()
        
        # Migration history
        self.migration_history_file = self.config_dir / "migration_history.json"
    
    def _register_migration_steps(self) -> None:
        """Register all available migration steps."""
        
        # Migration from legacy config (1.0.0) to AI Helper v2 (2.0.0)
        self._migration_steps.append(MigrationStep(
            from_version="1.0.0",
            to_version="2.0.0",
            description="Migrate from legacy configuration to AI Helper v2",
            migration_func=self._migrate_1_0_to_2_0,
            rollback_func=self._rollback_2_0_to_1_0
        ))
        
        # Future migrations can be added here
        # Example:
        # self._migration_steps.append(MigrationStep(
        #     from_version="2.0.0",
        #     to_version="2.1.0",
        #     description="Add new feature configurations",
        #     migration_func=self._migrate_2_0_to_2_1
        # ))
    
    def detect_config_version(self, config_data: Dict[str, Any]) -> str:
        """
        Detect the version of a configuration file.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            Version string (e.g., "1.0.0", "2.0.0")
        """
        # Check for version field
        if "version" in config_data:
            return config_data["version"]
        
        # Check for AI Helper v2 structure
        if "ai_helper" in config_data and "performance" in config_data:
            return "2.0.0"
        
        # Check for legacy structure
        if "screen" in config_data and "detection" in config_data:
            return "1.0.0"
        
        # Unknown structure - assume latest
        self.logger.warning("Could not detect configuration version, assuming latest")
        return self.CURRENT_VERSION
    
    def needs_migration(self, config_data: Dict[str, Any]) -> bool:
        """
        Check if configuration needs migration.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            True if migration is needed, False otherwise
        """
        current_version = self.detect_config_version(config_data)
        return semantic_version.Version(current_version) < semantic_version.Version(self.CURRENT_VERSION)
    
    def migrate_config(self, config_data: Dict[str, Any], backup: bool = True) -> Dict[str, Any]:
        """
        Migrate configuration to current version.
        
        Args:
            config_data: Configuration data to migrate
            backup: Whether to create backup before migration
            
        Returns:
            Migrated configuration data
            
        Raises:
            AIHelperConfigurationError: If migration fails
        """
        current_version = self.detect_config_version(config_data)
        
        if not self.needs_migration(config_data):
            self.logger.info(f"Configuration already at version {current_version}")
            return config_data
        
        if backup:
            self._create_backup(config_data, current_version)
        
        try:
            # Find migration path
            migration_path = self._find_migration_path(current_version, self.CURRENT_VERSION)
            
            if not migration_path:
                raise AIHelperConfigurationError(
                    f"No migration path found from {current_version} to {self.CURRENT_VERSION}"
                )
            
            # Apply migrations
            migrated_data = config_data.copy()
            for step in migration_path:
                self.logger.info(f"Applying migration: {step.description}")
                migrated_data = step.migration_func(migrated_data)
                
                # Record migration
                self._record_migration(step.from_version, step.to_version, step.description)
            
            # Add version field
            migrated_data["version"] = self.CURRENT_VERSION
            
            self.logger.info(f"Configuration migrated from {current_version} to {self.CURRENT_VERSION}")
            return migrated_data
            
        except Exception as e:
            self.logger.error(f"Configuration migration failed: {e}")
            raise AIHelperConfigurationError(f"Migration failed: {e}")
    
    def _find_migration_path(self, from_version: str, to_version: str) -> Optional[List[MigrationStep]]:
        """
        Find migration path between versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of migration steps or None if no path found
        """
        # Simple linear search for now - could be enhanced with graph algorithms
        path = []
        current_version = from_version
        
        while current_version != to_version:
            found_step = None
            
            for step in self._migration_steps:
                if step.from_version == current_version:
                    # Check if this step moves us closer to target
                    if semantic_version.Version(step.to_version) <= semantic_version.Version(to_version):
                        found_step = step
                        break
            
            if found_step is None:
                return None  # No path found
            
            path.append(found_step)
            current_version = found_step.to_version
        
        return path
    
    def _create_backup(self, config_data: Dict[str, Any], version: str) -> None:
        """Create backup of configuration before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.config_dir / f"config_backup_{version}_{timestamp}.json"
        
        try:
            with open(backup_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration backup created: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration backup: {e}")
    
    def _record_migration(self, from_version: str, to_version: str, description: str) -> None:
        """Record migration in history file."""
        try:
            # Load existing history
            history = []
            if self.migration_history_file.exists():
                with open(self.migration_history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new migration record
            history.append({
                "timestamp": datetime.now().isoformat(),
                "from_version": from_version,
                "to_version": to_version,
                "description": description
            })
            
            # Save updated history
            with open(self.migration_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to record migration history: {e}")
    
    # Migration Functions
    
    def _migrate_1_0_to_2_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate from legacy configuration (1.0.0) to AI Helper v2 (2.0.0).
        
        Args:
            config_data: Legacy configuration data
            
        Returns:
            Migrated configuration data
        """
        migrated = {
            "version": "2.0.0",
            "ai_helper": {
                "enabled": True,
                "max_analysis_time": 2.0,
                "confidence_threshold": config_data.get("detection", {}).get("confidence_threshold", 0.7),
                "archetype_preference": "balanced",
                "enable_explanations": True,
                "enable_visual_overlay": config_data.get("ui", {}).get("overlay_enabled", True),
                "fallback_to_legacy": True
            },
            "performance": {
                "max_memory_mb": 500,
                "max_cpu_percent": 25,
                "analysis_timeout": 10.0,
                "cache_size_mb": 50,
                "enable_monitoring": True,
                "thread_pool_size": 4
            },
            "security": {
                "enable_data_encryption": True,
                "log_sensitive_data": False,
                "require_secure_connections": True,
                "credential_timeout": 3600,
                "enable_audit_logging": True
            },
            "ui": {
                "overlay_opacity": 0.8,
                "hover_delay_ms": 500,
                "font_size": 12,
                "show_confidence_scores": config_data.get("ui", {}).get("show_confidence", True),
                "animation_duration_ms": 300,
                "theme": "auto"
            },
            "logging": {
                "level": config_data.get("debug", {}).get("log_level", "INFO"),
                "max_file_size_mb": 10,
                "max_files": 5,
                "enable_correlation_ids": True,
                "enable_performance_logging": True,
                "log_format": "json"
            },
            "development": {
                "hot_reload_enabled": False,
                "debug_mode": config_data.get("debug", {}).get("verbose_detection", False),
                "enable_profiling": False,
                "test_mode": False,
                "save_debug_screenshots": config_data.get("debug", {}).get("save_screenshots", False)
            },
            "card_evaluation": {
                "enable_ml_models": True,
                "ml_model_timeout": 30.0,
                "fallback_to_heuristics": True,
                "cache_evaluations": True,
                "evaluation_weights": {
                    "base_value": 0.3,
                    "tempo_score": 0.25,
                    "value_score": 0.2,
                    "synergy_score": 0.15,
                    "curve_score": 0.1
                }
            },
            "overlay": {
                "enable_click_through": True,
                "frame_rate_limit": 30,
                "position_offset_x": 0,
                "position_offset_y": 0,
                "auto_hide_delay": 30
            }
        }
        
        # Preserve any legacy-specific settings that might be useful
        if "screen" in config_data:
            screen_config = config_data["screen"]
            migrated["development"]["save_debug_screenshots"] = screen_config.get("debug_screenshots", False)
        
        if "modes" in config_data:
            modes_config = config_data["modes"]
            # Map legacy mode settings to new structure if applicable
            pass
        
        return migrated
    
    def _rollback_2_0_to_1_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rollback from AI Helper v2 (2.0.0) to legacy configuration (1.0.0).
        
        Args:
            config_data: AI Helper v2 configuration data
            
        Returns:
            Legacy configuration data
        """
        rollback_data = {
            "version": "1.0.0",
            "screen": {
                "capture_method": "pyqt6",
                "detection_timeout": 100,
                "debug_screenshots": config_data.get("development", {}).get("save_debug_screenshots", False)
            },
            "detection": {
                "histogram_bins": [50, 60],
                "confidence_threshold": config_data.get("ai_helper", {}).get("confidence_threshold", 0.35),
                "template_threshold_mana": 4.5,
                "template_threshold_rarity": 9.0,
                "max_candidates": 15
            },
            "modes": {
                "support_underground": True,
                "redraft_enabled": True,
                "auto_detect_mode": True
            },
            "ui": {
                "overlay_enabled": config_data.get("ai_helper", {}).get("enable_visual_overlay", True),
                "show_confidence": config_data.get("ui", {}).get("show_confidence_scores", True),
                "recommendation_count": 3
            },
            "debug": {
                "save_screenshots": config_data.get("development", {}).get("save_debug_screenshots", False),
                "log_level": config_data.get("logging", {}).get("level", "INFO"),
                "verbose_detection": config_data.get("development", {}).get("debug_mode", False)
            }
        }
        
        return rollback_data
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get migration history.
        
        Returns:
            List of migration records
        """
        try:
            if self.migration_history_file.exists():
                with open(self.migration_history_file, 'r') as f:
                    return json.load(f)
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to read migration history: {e}")
            return []
    
    def validate_migrated_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Validate migrated configuration.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ["ai_helper", "performance", "security", "ui", "logging"]
            for section in required_sections:
                if section not in config_data:
                    self.logger.error(f"Missing required section after migration: {section}")
                    return False
            
            # Check version field
            if "version" not in config_data:
                self.logger.error("Missing version field after migration")
                return False
            
            # Validate version format
            try:
                semantic_version.Version(config_data["version"])
            except ValueError:
                self.logger.error(f"Invalid version format: {config_data['version']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


def migrate_config_file(config_file: Path, backup: bool = True) -> bool:
    """
    Convenience function to migrate a configuration file.
    
    Args:
        config_file: Path to configuration file
        backup: Whether to create backup
        
    Returns:
        True if migration successful, False otherwise
    """
    try:
        migrator = ConfigurationMigrator(config_file.parent)
        
        # Load current configuration
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Check if migration is needed
        if not migrator.needs_migration(config_data):
            return True
        
        # Perform migration
        migrated_data = migrator.migrate_config(config_data, backup)
        
        # Validate migrated configuration
        if not migrator.validate_migrated_config(migrated_data):
            raise AIHelperConfigurationError("Migrated configuration failed validation")
        
        # Save migrated configuration
        with open(config_file, 'w') as f:
            json.dump(migrated_data, f, indent=2)
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Configuration file migration failed: {e}")
        return False