"""
Configuration schema validation and patch pinning tests.

Tests the strict configuration validation system and patch version
compatibility checking for bot_config.json.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.utils.config_validation import (
    ConfigValidator, ValidationResult, PatchInfo, 
    validate_config_at_startup, DEFAULT_CONFIG, CONFIG_SCHEMA
)


class TestConfigSchemaValidation:
    """Test configuration schema validation."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test card data fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "data"
    
    @pytest.fixture
    def validator(self, temp_config_dir, test_data_dir):
        """Create validator with test directories."""
        config_path = temp_config_dir / "bot_config.json"
        return ConfigValidator(config_path=config_path, cards_data_dir=test_data_dir)
    
    def test_default_config_schema_validation(self, validator):
        """Test that default configuration passes schema validation."""
        result = validator.validate_schema(DEFAULT_CONFIG)
        
        assert result.valid, f"Default config should be valid: {result.errors}"
        assert len(result.errors) == 0, f"Default config has errors: {result.errors}"
        
        # May have warnings, but should be valid
        if result.warnings:
            print(f"Default config warnings: {result.warnings}")
        
        print("✅ Default configuration passes schema validation")
    
    def test_valid_custom_config(self, validator):
        """Test validation of a valid custom configuration."""
        valid_config = {
            "patch_version": "29.0.1",
            "locale": "enUS",
            "screen_scaling": 1.25,
            "capture_method": "pyqt6",
            "expected_fps": 60,
            "performance_budgets": {
                "coordinates": 50,
                "eligibility_filter": 15,
                "histogram_match": 120,
                "template_validation": 70,
                "ai_advisor": 140,
                "ui_render": 35,
                "total": 450
            },
            "detection": {
                "histogram_bins": [48, 58],
                "confidence_threshold": 0.4,
                "template_threshold_mana": 5.0,
                "template_threshold_rarity": 8.0
            },
            "ui": {
                "overlay_enabled": True,
                "show_confidence": False,
                "recommendation_count": 5,
                "always_on_top": False,
                "click_through": True,
                "transparency": 0.8
            },
            "debug": {
                "save_screenshots": True,
                "log_level": "DEBUG",
                "verbose_detection": True,
                "enable_debug_dumps": True
            }
        }
        
        result = validator.validate_schema(valid_config)
        
        assert result.valid, f"Valid config should pass: {result.errors}"
        assert len(result.errors) == 0
        
        print("✅ Valid custom configuration passes schema validation")
    
    def test_missing_required_fields(self, validator):
        """Test validation fails for missing required fields."""
        # Missing patch_version
        config_missing_patch = {
            "locale": "enUS",
            "screen_scaling": 1.0
        }
        
        result = validator.validate_schema(config_missing_patch)
        assert not result.valid
        assert any("patch_version" in error for error in result.errors)
        
        # Missing locale
        config_missing_locale = {
            "patch_version": "29.0",
            "screen_scaling": 1.0
        }
        
        result = validator.validate_schema(config_missing_locale)
        assert not result.valid
        assert any("locale" in error for error in result.errors)
        
        print("✅ Missing required fields properly rejected")
    
    def test_invalid_field_types(self, validator):
        """Test validation fails for invalid field types."""
        invalid_configs = [
            {
                "patch_version": 29.0,  # Should be string
                "locale": "enUS"
            },
            {
                "patch_version": "29.0",
                "locale": "invalid_locale",  # Invalid enum value
            },
            {
                "patch_version": "29.0",
                "locale": "enUS",
                "screen_scaling": "1.25"  # Should be number
            },
            {
                "patch_version": "29.0",
                "locale": "enUS",
                "expected_fps": -10  # Should be positive
            }
        ]
        
        for i, config in enumerate(invalid_configs):
            result = validator.validate_schema(config)
            assert not result.valid, f"Invalid config {i} should fail validation"
            assert len(result.errors) > 0, f"Invalid config {i} should have errors"
        
        print("✅ Invalid field types properly rejected")
    
    def test_invalid_patch_version_format(self, validator):
        """Test validation of patch version format."""
        invalid_patch_configs = [
            {"patch_version": "29", "locale": "enUS"},  # Too short
            {"patch_version": "29.0.0.1", "locale": "enUS"},  # Too many parts
            {"patch_version": "v29.0", "locale": "enUS"},  # Invalid prefix
            {"patch_version": "29.a", "locale": "enUS"},  # Non-numeric
        ]
        
        for config in invalid_patch_configs:
            result = validator.validate_schema(config)
            assert not result.valid, f"Invalid patch version {config['patch_version']} should fail"
        
        # Valid patch versions should pass
        valid_patch_configs = [
            {"patch_version": "29.0", "locale": "enUS"},
            {"patch_version": "29.0.1", "locale": "enUS"},
            {"patch_version": "30.2.5", "locale": "enUS"},
        ]
        
        for config in valid_patch_configs:
            result = validator.validate_schema(config)
            assert result.valid, f"Valid patch version {config['patch_version']} should pass"
        
        print("✅ Patch version format validation works correctly")
    
    def test_performance_budget_validation(self, validator):
        """Test performance budget validation logic."""
        # Budget where total is less than sum of stages
        config_with_inconsistent_budgets = {
            "patch_version": "29.0",
            "locale": "enUS", 
            "performance_budgets": {
                "coordinates": 100,
                "eligibility_filter": 50,
                "histogram_match": 200,
                "template_validation": 100,
                "ai_advisor": 200,
                "ui_render": 50,
                "total": 300  # Less than sum (700)
            }
        }
        
        result = validator.validate_schema(config_with_inconsistent_budgets)
        # Should be valid but have warnings
        assert result.valid
        assert any("Total budget" in warning for warning in result.warnings)
        
        print("✅ Performance budget validation detects inconsistencies")
    
    def test_additional_properties_rejection(self, validator):
        """Test that additional properties are rejected."""
        config_with_extra_fields = {
            "patch_version": "29.0",
            "locale": "enUS",
            "unknown_field": "should_be_rejected",
            "debug": {
                "log_level": "INFO",
                "extra_debug_field": "also_rejected"
            }
        }
        
        result = validator.validate_schema(config_with_extra_fields)
        assert not result.valid
        assert any("additional" in error.lower() or "unknown" in error.lower() 
                  for error in result.errors)
        
        print("✅ Additional properties properly rejected")


class TestPatchVersionCompatibility:
    """Test patch version compatibility checking."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test card data fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "data"
    
    @pytest.fixture
    def validator(self, test_data_dir):
        """Create validator with test data directory."""
        return ConfigValidator(cards_data_dir=test_data_dir)
    
    def test_find_available_patches(self, validator):
        """Test detection of available card data patches."""
        patches = validator.find_available_patches()
        
        # Should find our test fixtures
        assert "29.0" in patches, f"Should find patch 29.0, found: {list(patches.keys())}"
        assert "28.6" in patches, f"Should find patch 28.6, found: {list(patches.keys())}"
        
        # Check patch info
        patch_29 = patches["29.0"]
        assert patch_29.version == "29.0"
        assert patch_29.card_count == 5  # Our test fixture has 5 cards
        assert patch_29.cards_file is not None
        
        patch_28 = patches["28.6"]
        assert patch_28.version == "28.6"
        assert patch_28.card_count == 2  # Our test fixture has 2 cards
        
        print(f"✅ Found patches: {list(patches.keys())}")
    
    def test_exact_patch_match(self, validator):
        """Test exact patch version matching."""
        # Test exact match for available patch
        result = validator.validate_patch_compatibility("29.0")
        assert result.valid
        assert len(result.errors) == 0
        
        result = validator.validate_patch_compatibility("28.6")
        assert result.valid
        assert len(result.errors) == 0
        
        print("✅ Exact patch matches work correctly")
    
    def test_compatible_patch_matching(self, validator):
        """Test compatible patch version matching."""
        # Test compatible patch (same major version)
        result = validator.validate_patch_compatibility("29.0.1")
        assert result.valid  # Should be compatible with 29.0
        assert any("compatible" in warning for warning in result.warnings)
        
        print("✅ Compatible patch matching works correctly")
    
    def test_incompatible_patch_rejection(self, validator):
        """Test rejection of incompatible patch versions."""
        # Test completely incompatible patch
        result = validator.validate_patch_compatibility("30.0")
        assert not result.valid
        assert any("not compatible" in error for error in result.errors)
        
        # Test invalid patch format
        result = validator.validate_patch_compatibility("invalid_patch")
        assert not result.valid
        
        print("✅ Incompatible patches properly rejected")
    
    def test_no_card_data_available(self, validator):
        """Test behavior when no card data is available."""
        # Create validator with non-existent data directory
        empty_validator = ConfigValidator(cards_data_dir=Path("/nonexistent"))
        
        result = empty_validator.validate_patch_compatibility("29.0")
        assert not result.valid
        assert any("No card data" in error for error in result.errors)
        
        print("✅ Missing card data properly detected")


class TestFullConfigValidation:
    """Test full configuration loading and validation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test card data fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "data"
    
    def test_create_default_config(self, temp_dir, test_data_dir):
        """Test creation of default configuration file."""
        config_path = temp_dir / "bot_config.json"
        validator = ConfigValidator(config_path=config_path, cards_data_dir=test_data_dir)
        
        # Config file should not exist initially
        assert not config_path.exists()
        
        # Load and validate should create default config
        result = validator.load_and_validate_config()
        
        # Should be valid and config file should be created
        assert result.valid, f"Default config creation failed: {result.errors}"
        assert config_path.exists()
        
        # Verify file contents
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config == DEFAULT_CONFIG
        
        print("✅ Default configuration file creation works correctly")
    
    def test_load_valid_config_file(self, temp_dir, test_data_dir):
        """Test loading and validation of valid config file."""
        config_path = temp_dir / "bot_config.json"
        
        # Create valid config file
        valid_config = {
            "patch_version": "29.0",
            "locale": "enUS",
            "screen_scaling": 1.5,
            "capture_method": "mss",
            "expected_fps": 30
        }
        
        with open(config_path, 'w') as f:
            json.dump(valid_config, f, indent=2)
        
        validator = ConfigValidator(config_path=config_path, cards_data_dir=test_data_dir)
        result = validator.load_and_validate_config()
        
        assert result.valid, f"Valid config should pass: {result.errors}"
        assert result.config_data["patch_version"] == "29.0"
        assert result.config_data["locale"] == "enUS"
        
        print("✅ Valid configuration file loading works correctly")
    
    def test_load_invalid_config_file(self, temp_dir, test_data_dir):
        """Test loading and validation of invalid config file."""
        config_path = temp_dir / "bot_config.json"
        
        # Create invalid config file
        invalid_config = {
            "patch_version": "99.0",  # Incompatible patch
            "locale": "invalid_locale",  # Invalid locale
            "screen_scaling": -1.0  # Invalid scaling
        }
        
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f, indent=2)
        
        validator = ConfigValidator(config_path=config_path, cards_data_dir=test_data_dir)
        result = validator.load_and_validate_config()
        
        assert not result.valid
        assert len(result.errors) > 0
        
        print("✅ Invalid configuration file properly rejected")
    
    def test_load_malformed_json(self, temp_dir, test_data_dir):
        """Test handling of malformed JSON config file."""
        config_path = temp_dir / "bot_config.json"
        
        # Create malformed JSON file
        with open(config_path, 'w') as f:
            f.write('{"patch_version": "29.0", "locale": "enUS",}')  # Trailing comma
        
        validator = ConfigValidator(config_path=config_path, cards_data_dir=test_data_dir)
        result = validator.load_and_validate_config()
        
        assert not result.valid
        assert any("Failed to load config" in error for error in result.errors)
        
        print("✅ Malformed JSON properly handled")
    
    def test_validation_summary_formatting(self, temp_dir, test_data_dir):
        """Test validation summary formatting."""
        config_path = temp_dir / "bot_config.json"
        validator = ConfigValidator(config_path=config_path, cards_data_dir=test_data_dir)
        
        # Test valid result summary
        valid_result = ValidationResult(
            valid=True,
            errors=[],
            warnings=["Test warning"],
            config_data={"patch_version": "29.0", "locale": "enUS"}
        )
        
        summary = validator.get_validation_summary(valid_result)
        assert "✅ Configuration validation passed" in summary
        assert "⚠️  Test warning" in summary
        assert "patch 29.0" in summary
        
        # Test invalid result summary
        invalid_result = ValidationResult(
            valid=False,
            errors=["Test error"],
            warnings=["Test warning"],
            config_data=None
        )
        
        summary = validator.get_validation_summary(invalid_result)
        assert "🚨 Configuration validation failed" in summary
        assert "❌ Test error" in summary
        assert "⚠️  Test warning" in summary
        
        print("✅ Validation summary formatting works correctly")


class TestStartupValidation:
    """Test startup validation behavior."""
    
    @pytest.fixture 
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_startup_validation_success(self, temp_dir):
        """Test successful startup validation."""
        # Create valid config
        config_path = temp_dir / "bot_config.json"
        
        # Use default config which should be valid
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        
        # Create minimal card data
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        with open(data_dir / "cards_29.0_enUS.json", 'w') as f:
            json.dump([{"id": "test", "name": "Test Card"}], f)
        
        # Create custom validator with test data directory
        validator = ConfigValidator(config_path=config_path, cards_data_dir=data_dir)
        result = validator.load_and_validate_config()
        
        # Should be valid
        assert result.valid, f"Config should be valid: {result.errors}"
        print("✅ Startup validation success handled correctly")
    
    def test_startup_validation_failure(self, temp_dir):
        """Test startup validation failure behavior."""
        # Create invalid config
        config_path = temp_dir / "bot_config.json"
        
        invalid_config = {
            "patch_version": "invalid",
            "locale": "invalid"
        }
        
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f, indent=2)
        
        # This should raise SystemExit
        with pytest.raises(SystemExit):
            validate_config_at_startup(config_path)
        
        print("✅ Startup validation failure handled correctly")


# Integration test
def test_config_validation_integration():
    """Integration test for complete configuration validation system."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test environment
        config_path = temp_path / "bot_config.json"
        data_dir = temp_path / "data"
        data_dir.mkdir()
        
        # Create card data files
        with open(data_dir / "cards_29.0_enUS.json", 'w') as f:
            json.dump([
                {"id": "fireball", "name": "Fireball", "cost": 4},
                {"id": "polymorph", "name": "Polymorph", "cost": 4}
            ], f)
        
        # Create configuration
        config = {
            "patch_version": "29.0",
            "locale": "enUS", 
            "screen_scaling": 1.25,
            "performance_budgets": {
                "coordinates": 50,
                "eligibility_filter": 15,
                "histogram_match": 120,
                "template_validation": 70,
                "ai_advisor": 140,
                "ui_render": 35,
                "total": 450
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Test complete validation
        validator = ConfigValidator(config_path=config_path, cards_data_dir=data_dir)
        result = validator.load_and_validate_config()
        
        # Should pass all validation
        assert result.valid, f"Integration test failed: {result.errors}"
        assert result.config_data is not None
        assert result.config_data["patch_version"] == "29.0"
        
        # Test patch compatibility
        patches = validator.find_available_patches()
        assert "29.0" in patches
        assert patches["29.0"].card_count == 2
        
        # Test summary
        summary = validator.get_validation_summary(result)
        assert "✅" in summary
        
        print("✅ Configuration validation integration test passed")
        print(f"   Config: patch {config['patch_version']}, scaling {config['screen_scaling']}")
        print(f"   Found {len(patches)} patch(es): {list(patches.keys())}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])