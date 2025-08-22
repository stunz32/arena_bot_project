"""
Strict configuration validation and patch pinning system.

Provides JSON schema validation for bot_config.json and ensures
patch version compatibility between configuration and card data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration JSON Schema
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "patch_version": {
            "type": "string",
            "pattern": r"^\d+\.\d+(\.\d+)?$",
            "description": "Hearthstone patch version (e.g., '29.0', '29.0.3')"
        },
        "locale": {
            "type": "string",
            "enum": ["enUS", "enGB", "deDE", "esES", "esMX", "frFR", "itIT", "jaJP", "koKR", "plPL", "ptBR", "ruRU", "thTH", "zhCN", "zhTW"],
            "description": "Game locale for card data"
        },
        "screen_scaling": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 3.0,
            "description": "Display scaling factor (1.0 = 100%, 1.25 = 125%, etc.)"
        },
        "capture_method": {
            "type": "string",
            "enum": ["pyqt6", "win32", "mss", "pil"],
            "description": "Screenshot capture method"
        },
        "expected_fps": {
            "type": "integer",
            "minimum": 10,
            "maximum": 120,
            "description": "Expected detection FPS for performance budgets"
        },
        "performance_budgets": {
            "type": "object",
            "properties": {
                "coordinates": {"type": "number", "minimum": 10, "maximum": 200},
                "eligibility_filter": {"type": "number", "minimum": 5, "maximum": 100},
                "histogram_match": {"type": "number", "minimum": 50, "maximum": 500},
                "template_validation": {"type": "number", "minimum": 20, "maximum": 200},
                "ai_advisor": {"type": "number", "minimum": 50, "maximum": 1000},
                "ui_render": {"type": "number", "minimum": 10, "maximum": 100},
                "total": {"type": "number", "minimum": 200, "maximum": 2000}
            },
            "additionalProperties": False,
            "description": "Per-stage performance budgets in milliseconds"
        },
        "detection": {
            "type": "object",
            "properties": {
                "histogram_bins": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 8, "maximum": 256},
                    "minItems": 2,
                    "maxItems": 3,
                    "description": "Histogram bins for H, S, V channels"
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "description": "Minimum confidence for card matches"
                },
                "template_threshold_mana": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 20.0,
                    "description": "Template matching threshold for mana cost"
                },
                "template_threshold_rarity": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 30.0,
                    "description": "Template matching threshold for rarity"
                }
            },
            "additionalProperties": False,
            "description": "Detection algorithm settings"
        },
        "ui": {
            "type": "object", 
            "properties": {
                "overlay_enabled": {"type": "boolean"},
                "show_confidence": {"type": "boolean"},
                "recommendation_count": {"type": "integer", "minimum": 1, "maximum": 10},
                "always_on_top": {"type": "boolean"},
                "click_through": {"type": "boolean"},
                "transparency": {"type": "number", "minimum": 0.1, "maximum": 1.0}
            },
            "additionalProperties": False,
            "description": "UI overlay settings"
        },
        "debug": {
            "type": "object",
            "properties": {
                "save_screenshots": {"type": "boolean"},
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                },
                "verbose_detection": {"type": "boolean"},
                "enable_debug_dumps": {"type": "boolean"}
            },
            "additionalProperties": False,
            "description": "Debug and logging settings"
        }
    },
    "required": ["patch_version", "locale"],
    "additionalProperties": False,
    "description": "Arena Bot configuration schema"
}

# Default configuration values
DEFAULT_CONFIG = {
    "patch_version": "29.0",
    "locale": "enUS",
    "screen_scaling": 1.0,
    "capture_method": "pyqt6",
    "expected_fps": 30,
    "performance_budgets": {
        "coordinates": 60,
        "eligibility_filter": 20,
        "histogram_match": 150,
        "template_validation": 80,
        "ai_advisor": 150,
        "ui_render": 40,
        "total": 500
    },
    "detection": {
        "histogram_bins": [50, 60],
        "confidence_threshold": 0.35,
        "template_threshold_mana": 4.5,
        "template_threshold_rarity": 9.0
    },
    "ui": {
        "overlay_enabled": True,
        "show_confidence": True,
        "recommendation_count": 3,
        "always_on_top": True,
        "click_through": False,
        "transparency": 0.9
    },
    "debug": {
        "save_screenshots": False,
        "log_level": "INFO",
        "verbose_detection": False,
        "enable_debug_dumps": False
    }
}


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    config_data: Optional[Dict[str, Any]] = None


@dataclass
class PatchInfo:
    """Information about a patch version."""
    version: str
    cards_file: Optional[Path] = None
    card_count: int = 0
    last_updated: Optional[datetime] = None


class ConfigValidator:
    """
    Strict configuration validator with JSON schema and patch pinning.
    
    Validates bot_config.json against schema and ensures compatibility
    with available card data patches.
    """
    
    def __init__(self, config_path: Optional[Path] = None, cards_data_dir: Optional[Path] = None):
        """
        Initialize configuration validator.
        
        Args:
            config_path: Path to bot_config.json (default: project root)
            cards_data_dir: Path to card data directory (default: auto-detect)
        """
        self.config_path = config_path or self._find_config_path()
        self.cards_data_dir = cards_data_dir or self._find_cards_data_dir()
        self.logger = logger
        
    def _find_config_path(self) -> Path:
        """Find bot_config.json in project root."""
        # Look for bot_config.json in project root
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:  # Stop at filesystem root
            config_file = current_dir / "bot_config.json"
            if config_file.exists():
                return config_file
            current_dir = current_dir.parent
        
        # Default to project root
        return Path(__file__).parent.parent.parent / "bot_config.json"
    
    def _find_cards_data_dir(self) -> Path:
        """Find card data directory."""
        # Look for data directory
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            data_dir = current_dir / "data"
            if data_dir.exists():
                return data_dir
            current_dir = current_dir.parent
        
        # Default to project root data
        return Path(__file__).parent.parent.parent / "data"
    
    def validate_schema(self, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration against JSON schema.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            ValidationResult with schema validation details
        """
        try:
            import jsonschema
            schema_available = True
        except ImportError:
            schema_available = False
            
        errors = []
        warnings = []
        
        if not schema_available:
            warnings.append("jsonschema not available - using basic validation")
            return self._basic_validation(config_data)
        
        try:
            # Validate against schema
            jsonschema.validate(config_data, CONFIG_SCHEMA)
            
            # Additional logical validations
            self._validate_performance_budgets(config_data, errors, warnings)
            self._validate_detection_settings(config_data, errors, warnings)
            
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                config_data=config_data
            )
            
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
    
    def _basic_validation(self, config_data: Dict[str, Any]) -> ValidationResult:
        """Basic validation when jsonschema is not available."""
        errors = []
        warnings = []
        
        # Check required fields
        if "patch_version" not in config_data:
            errors.append("Missing required field: patch_version")
        
        if "locale" not in config_data:
            errors.append("Missing required field: locale")
        
        # Check patch version format
        if "patch_version" in config_data:
            patch_version = config_data["patch_version"]
            if not isinstance(patch_version, str) or not patch_version:
                errors.append("patch_version must be a non-empty string")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            config_data=config_data
        )
    
    def _validate_performance_budgets(self, config_data: Dict[str, Any], 
                                    errors: List[str], warnings: List[str]):
        """Validate performance budget consistency."""
        budgets = config_data.get("performance_budgets", {})
        if not budgets:
            return
        
        # Check if total budget is reasonable
        stage_sum = sum(budgets.get(stage, 0) for stage in 
                       ["coordinates", "eligibility_filter", "histogram_match", 
                        "template_validation", "ai_advisor", "ui_render"])
        total_budget = budgets.get("total", 0)
        
        if total_budget < stage_sum:
            warnings.append(f"Total budget ({total_budget}ms) less than sum of stages ({stage_sum}ms)")
    
    def _validate_detection_settings(self, config_data: Dict[str, Any],
                                   errors: List[str], warnings: List[str]):
        """Validate detection settings consistency."""
        detection = config_data.get("detection", {})
        if not detection:
            return
        
        # Check histogram bins
        bins = detection.get("histogram_bins", [])
        if len(bins) < 2:
            errors.append("histogram_bins must have at least 2 values")
        
        # Check confidence threshold
        confidence = detection.get("confidence_threshold", 0.35)
        if confidence < 0.1 or confidence > 0.9:
            warnings.append(f"confidence_threshold {confidence} may be too extreme")
    
    def find_available_patches(self) -> Dict[str, PatchInfo]:
        """
        Find available card data patches.
        
        Returns:
            Dictionary mapping patch version to PatchInfo
        """
        patches = {}
        
        if not self.cards_data_dir.exists():
            self.logger.warning(f"Cards data directory not found: {self.cards_data_dir}")
            return patches
        
        # Look for card data files
        for file_path in self.cards_data_dir.glob("cards_*.json"):
            try:
                # Extract patch version from filename
                filename = file_path.stem
                if "_" in filename:
                    version_part = filename.split("_", 1)[1]
                    # Clean version (remove locale if present)
                    if "_" in version_part:
                        version = version_part.split("_")[0]
                    else:
                        version = version_part
                    
                    # Load and count cards
                    with open(file_path, 'r', encoding='utf-8') as f:
                        card_data = json.load(f)
                    
                    card_count = len(card_data) if isinstance(card_data, list) else len(card_data.get('cards', []))
                    
                    patches[version] = PatchInfo(
                        version=version,
                        cards_file=file_path,
                        card_count=card_count,
                        last_updated=datetime.fromtimestamp(file_path.stat().st_mtime)
                    )
                    
            except Exception as e:
                self.logger.warning(f"Failed to process card data file {file_path}: {e}")
        
        return patches
    
    def validate_patch_compatibility(self, config_patch: str) -> ValidationResult:
        """
        Validate patch version compatibility with available card data.
        
        Args:
            config_patch: Patch version from configuration
            
        Returns:
            ValidationResult with patch compatibility details
        """
        errors = []
        warnings = []
        
        available_patches = self.find_available_patches()
        
        if not available_patches:
            errors.append("No card data patches found")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Check for exact match
        if config_patch in available_patches:
            patch_info = available_patches[config_patch]
            if patch_info.card_count == 0:
                warnings.append(f"Patch {config_patch} card data appears empty")
            else:
                self.logger.info(f"Patch {config_patch} found with {patch_info.card_count} cards")
            
            return ValidationResult(valid=True, errors=errors, warnings=warnings)
        
        # Look for compatible patches (same major version)
        major_version = config_patch.split('.')[0]
        compatible_patches = [v for v in available_patches.keys() 
                            if v.startswith(major_version + '.')]
        
        if compatible_patches:
            latest_compatible = max(compatible_patches, key=lambda x: tuple(map(int, x.split('.'))))
            warnings.append(f"Exact patch {config_patch} not found, "
                          f"but compatible patch {latest_compatible} is available")
            return ValidationResult(valid=True, errors=errors, warnings=warnings)
        
        # No compatible patches
        available_list = ', '.join(sorted(available_patches.keys()))
        errors.append(f"Patch {config_patch} not compatible with available patches: {available_list}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)
    
    def load_and_validate_config(self) -> ValidationResult:
        """
        Load and validate complete configuration.
        
        Returns:
            ValidationResult with full validation details
        """
        errors = []
        warnings = []
        
        # Load configuration file
        if not self.config_path.exists():
            # Create default config file
            self.logger.info(f"Creating default configuration at {self.config_path}")
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                config_data = DEFAULT_CONFIG.copy()
            except Exception as e:
                errors.append(f"Failed to create default config: {e}")
                return ValidationResult(valid=False, errors=errors, warnings=warnings)
        else:
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load config from {self.config_path}: {e}")
                return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validate schema
        schema_result = self.validate_schema(config_data)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)
        
        if not schema_result.valid:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validate patch compatibility
        patch_version = config_data.get("patch_version", "unknown")
        patch_result = self.validate_patch_compatibility(patch_version)
        errors.extend(patch_result.errors)
        warnings.extend(patch_result.warnings)
        
        final_valid = len(errors) == 0
        return ValidationResult(
            valid=final_valid,
            errors=errors,
            warnings=warnings,
            config_data=config_data
        )
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """
        Generate human-readable validation summary.
        
        Args:
            result: Validation result to summarize
            
        Returns:
            Formatted validation summary
        """
        lines = []
        
        if result.valid:
            lines.append("✅ Configuration validation passed")
        else:
            lines.append("🚨 Configuration validation failed")
        
        if result.errors:
            lines.append("\nErrors:")
            for error in result.errors:
                lines.append(f"  ❌ {error}")
        
        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        if result.config_data:
            patch_version = result.config_data.get("patch_version", "unknown")
            locale = result.config_data.get("locale", "unknown")
            lines.append(f"\nConfiguration: patch {patch_version}, locale {locale}")
        
        return '\n'.join(lines)


def validate_config_at_startup(config_path: Optional[Path] = None) -> ValidationResult:
    """
    Validate configuration at application startup.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ValidationResult with validation status
        
    Raises:
        SystemExit: If configuration is invalid and cannot continue
    """
    validator = ConfigValidator(config_path=config_path)
    result = validator.load_and_validate_config()
    
    # Log validation results
    summary = validator.get_validation_summary(result)
    if result.valid:
        logger.info(summary)
    else:
        logger.error(summary)
        
        # Fail fast on invalid configuration
        raise SystemExit(f"Configuration validation failed. Fix configuration and restart.\n{summary}")
    
    if result.warnings:
        logger.warning("Configuration has warnings - consider reviewing settings")
    
    return result