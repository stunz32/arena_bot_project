"""
Test suite for S-Tier Logging System Configuration Layers.

This test suite validates the hierarchical configuration layer system including
file-based configuration, environment variables, runtime overrides, and the
configuration merging logic for enterprise-grade configuration management.

Test Categories:
- Configuration layer base functionality
- File-based configuration with format detection
- Environment variable parsing and nested keys
- Runtime configuration overrides
- Configuration layer priority and merging
- Hot-reload and change detection
- Error handling and edge cases
"""

import os
import sys
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, mock_open
import pytest

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arena_bot.logging_system.config.layers import (
    ConfigurationLayer,
    DefaultConfigLayer,
    FileConfigLayer,
    EnvironmentConfigLayer,
    RuntimeConfigLayer
)


class TestConfigurationLayer:
    """Test cases for base ConfigurationLayer functionality."""
    
    def test_abstract_layer_interface(self):
        """Test that ConfigurationLayer is abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ConfigurationLayer("test", 1)
    
    def test_layer_properties(self):
        """Test layer property methods."""
        layer = DefaultConfigLayer("test", 5)
        
        assert layer.get_name() == "test"
        assert layer.get_priority() == 5
        assert layer.is_available() is True
        assert isinstance(layer.get_last_update(), float)
    
    def test_layer_string_representation(self):
        """Test layer string representations."""
        layer = DefaultConfigLayer("test", 5)
        
        str_repr = str(layer)
        assert "DefaultConfigLayer" in str_repr
        assert "test" in str_repr
        assert "5" in str_repr
        
        repr_str = repr(layer)
        assert "DefaultConfigLayer" in repr_str
        assert "available=True" in repr_str


class TestDefaultConfigLayer:
    """Test cases for DefaultConfigLayer functionality."""
    
    def test_create_default_layer(self):
        """Test creating default configuration layer."""
        layer = DefaultConfigLayer()
        
        assert layer.get_name() == "defaults"
        assert layer.get_priority() == 0
        assert layer.is_available() is True
    
    def test_default_values_structure(self):
        """Test default configuration values structure."""
        layer = DefaultConfigLayer()
        values = layer.get_values()
        
        # Check required top-level keys
        required_keys = [
            "version", "environment", "system_name", "instance_id",
            "loggers", "handlers", "sinks", "filters",
            "security", "performance", "diagnostics"
        ]
        
        for key in required_keys:
            assert key in values, f"Missing required key: {key}"
    
    def test_default_loggers_configuration(self):
        """Test default loggers configuration."""
        layer = DefaultConfigLayer()
        values = layer.get_values()
        
        loggers = values["loggers"]
        assert "root" in loggers
        assert "app" in loggers
        
        root_logger = loggers["root"]
        assert root_logger["level"] == "INFO"
        assert root_logger["propagate"] is False
        
        app_logger = loggers["app"]
        assert app_logger["level"] == "INFO"
        assert app_logger["propagate"] is True
    
    def test_default_security_configuration(self):
        """Test default security configuration."""
        layer = DefaultConfigLayer()
        values = layer.get_values()
        
        security = values["security"]
        assert security["enable_pii_detection"] is False
        assert security["redact_credentials"] is True
        assert security["enable_audit_trail"] is False
        assert security["compliance_mode"] == "none"
        
        # Check nested encryption config
        encryption = security["encryption"]
        assert encryption["enabled"] is False
        assert encryption["algorithm"] == "fernet"
    
    def test_default_performance_configuration(self):
        """Test default performance configuration."""
        layer = DefaultConfigLayer()
        values = layer.get_values()
        
        performance = values["performance"]
        assert performance["enable_async_processing"] is False
        assert performance["worker_threads"] == 2
        assert performance["buffer_size"] == 4096
        assert performance["enable_caching"] is True
    
    def test_values_immutability(self):
        """Test that get_values returns a copy (immutable)."""
        layer = DefaultConfigLayer()
        values1 = layer.get_values()
        values2 = layer.get_values()
        
        # Modify one copy
        values1["version"] = 999
        
        # Other copy should be unchanged
        assert values2["version"] == 1


class TestFileConfigLayer:
    """Test cases for FileConfigLayer functionality."""
    
    def test_create_file_layer_toml(self):
        """Test creating file layer with TOML configuration."""
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            toml_content = """
version = 2
environment = "testing"

[security]
enable_pii_detection = true

[performance]
worker_threads = 8
"""
            f.write(toml_content.encode())
            f.flush()
            
            try:
                layer = FileConfigLayer(Path(f.name))
                
                assert layer.get_name() == f"file:{Path(f.name).name}"
                assert layer.file_format == "toml"
                assert layer.is_available() is True
                
                values = layer.get_values()
                assert values["version"] == 2
                assert values["environment"] == "testing"
                assert values["security"]["enable_pii_detection"] is True
                assert values["performance"]["worker_threads"] == 8
                
            finally:
                os.unlink(f.name)
    
    def test_create_file_layer_json(self):
        """Test creating file layer with JSON configuration."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_content = {
                "version": 3,
                "environment": "production",
                "security": {
                    "enable_audit_trail": True
                }
            }
            f.write(json.dumps(json_content).encode())
            f.flush()
            
            try:
                layer = FileConfigLayer(Path(f.name))
                
                assert layer.get_name() == f"file:{Path(f.name).name}"
                assert layer.file_format == "json"
                assert layer.is_available() is True
                
                values = layer.get_values()
                assert values["version"] == 3
                assert values["environment"] == "production"
                assert values["security"]["enable_audit_trail"] is True
                
            finally:
                os.unlink(f.name)
    
    def test_file_format_detection(self):
        """Test automatic file format detection."""
        # Test TOML detection
        layer_toml = FileConfigLayer(Path("config.toml"))
        assert layer_toml.file_format == "toml"
        
        # Test JSON detection
        layer_json = FileConfigLayer(Path("config.json"))
        assert layer_json.file_format == "json"
        
        # Test YAML detection (if supported)
        layer_yaml = FileConfigLayer(Path("config.yaml"))
        assert layer_yaml.file_format == "yaml"
        
        # Test default format for unknown extension
        layer_unknown = FileConfigLayer(Path("config.conf"))
        assert layer_unknown.file_format == "toml"  # Default
    
    def test_file_not_exists(self):
        """Test handling of non-existent file."""
        layer = FileConfigLayer(Path("non_existent_file.toml"))
        
        values = layer.get_values()
        assert values == {}  # Should return empty dict
        assert layer.is_available() is True  # Should still be available
    
    def test_file_change_detection(self):
        """Test file change detection for hot-reload."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            # Initial content
            initial_content = {"version": 1, "test": "initial"}
            f.write(json.dumps(initial_content).encode())
            f.flush()
            
            try:
                layer = FileConfigLayer(Path(f.name))
                
                # Read initial values
                values1 = layer.get_values()
                assert values1["version"] == 1
                assert values1["test"] == "initial"
                
                # Wait a bit to ensure different mtime
                time.sleep(0.1)
                
                # Update file content
                updated_content = {"version": 2, "test": "updated"}
                with open(f.name, 'w') as update_f:
                    json.dump(updated_content, update_f)
                
                # Read updated values
                values2 = layer.get_values()
                assert values2["version"] == 2
                assert values2["test"] == "updated"
                
                # Check change detection
                assert layer.is_file_changed() is False  # Should be current now
                
            finally:
                os.unlink(f.name)
    
    def test_file_info(self):
        """Test file information retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            f.write(b"version = 1\n")
            f.flush()
            
            try:
                layer = FileConfigLayer(Path(f.name))
                info = layer.get_file_info()
                
                assert info["exists"] is True
                assert info["format"] == "toml"
                assert info["size_bytes"] > 0
                assert "modified_time" in info
                assert "is_readable" in info
                
            finally:
                os.unlink(f.name)
    
    def test_invalid_file_content(self):
        """Test handling of invalid file content."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            # Write invalid JSON
            f.write(b'{"invalid": json content}')
            f.flush()
            
            try:
                layer = FileConfigLayer(Path(f.name))
                
                # Should raise exception on invalid content
                with pytest.raises(Exception):
                    layer.get_values()
                
            finally:
                os.unlink(f.name)


class TestEnvironmentConfigLayer:
    """Test cases for EnvironmentConfigLayer functionality."""
    
    def test_create_environment_layer(self):
        """Test creating environment configuration layer."""
        layer = EnvironmentConfigLayer()
        
        assert layer.get_name() == "env:LOGGING"
        assert layer.get_priority() == 20
        assert layer.prefix == "LOGGING_"
        assert layer.delimiter == "__"
    
    def test_custom_prefix_and_delimiter(self):
        """Test custom prefix and delimiter configuration."""
        layer = EnvironmentConfigLayer(
            prefix="MYAPP_LOG_",
            delimiter="___",
            name="custom_env"
        )
        
        assert layer.get_name() == "custom_env"
        assert layer.prefix == "MYAPP_LOG_"
        assert layer.delimiter == "___"
    
    @patch.dict(os.environ, {
        'LOGGING_VERSION': '5',
        'LOGGING_ENVIRONMENT': 'staging',
        'LOGGING_SECURITY__ENABLE_PII_DETECTION': 'true',
        'LOGGING_SECURITY__ENCRYPTION__ENABLED': 'false',
        'LOGGING_PERFORMANCE__WORKER_THREADS': '16',
        'LOGGING_PERFORMANCE__BUFFER_SIZE': '16384'
    })
    def test_environment_variable_parsing(self):
        """Test environment variable parsing and nesting."""
        layer = EnvironmentConfigLayer()
        values = layer.get_values()
        
        assert values["version"] == 5  # Converted to int
        assert values["environment"] == "staging"  # String
        
        # Test nested values
        assert values["security"]["enable_pii_detection"] is True  # Boolean conversion
        assert values["security"]["encryption"]["enabled"] is False  # Nested boolean
        
        assert values["performance"]["worker_threads"] == 16  # Int conversion
        assert values["performance"]["buffer_size"] == 16384  # Int conversion
    
    def test_value_type_conversion(self):
        """Test automatic value type conversion."""
        layer = EnvironmentConfigLayer()
        
        # Test different value conversions
        test_cases = [
            ("true", True),
            ("True", True),  
            ("TRUE", True),
            ("yes", True),
            ("1", True),
            ("on", True),
            ("enabled", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("no", False),
            ("0", False),
            ("off", False),
            ("disabled", False),
            ("42", 42),
            ("3.14", 3.14),
            ("hello", "hello"),
            ("item1,item2,item3", ["item1", "item2", "item3"]),
            ("'quoted'", "quoted"),
            ('"double_quoted"', "double_quoted")
        ]
        
        for input_val, expected in test_cases:
            result = layer._convert_env_value(input_val)
            assert result == expected, f"Failed to convert '{input_val}' to {expected}, got {result}"
    
    @patch.dict(os.environ, {
        'LOGGING_TEST_STRING': 'hello world',
        'LOGGING_TEST_INT': '42',
        'LOGGING_TEST_FLOAT': '3.14',
        'LOGGING_TEST_BOOL_TRUE': 'yes',
        'LOGGING_TEST_BOOL_FALSE': 'no',
        'LOGGING_TEST_LIST': 'item1,item2,item3'
    })
    def test_get_env_vars(self):
        """Test getting filtered environment variables."""
        layer = EnvironmentConfigLayer()
        env_vars = layer.get_env_vars()
        
        # Should only include variables with the prefix
        prefixed_vars = [k for k in env_vars.keys() if k.startswith('LOGGING_')]
        assert len(prefixed_vars) == 6
        
        assert 'LOGGING_TEST_STRING' in env_vars
        assert 'LOGGING_TEST_INT' in env_vars
        assert env_vars['LOGGING_TEST_STRING'] == 'hello world'
        assert env_vars['LOGGING_TEST_INT'] == '42'
    
    def test_cache_refresh(self):
        """Test environment variable cache refresh."""
        layer = EnvironmentConfigLayer()
        
        # Initial cache
        with patch.dict(os.environ, {'LOGGING_TEST': 'initial'}):
            values1 = layer.get_values()
            assert values1.get("test") == "initial"
        
        # Update environment and force refresh
        with patch.dict(os.environ, {'LOGGING_TEST': 'updated'}):
            layer.refresh_cache()
            values2 = layer.get_values()
            assert values2.get("test") == "updated"
    
    def test_case_sensitivity(self):
        """Test case sensitivity configuration."""
        # Case insensitive (default)
        layer_insensitive = EnvironmentConfigLayer(case_sensitive=False)
        
        with patch.dict(os.environ, {'LOGGING_TEST_KEY': 'value'}):
            values = layer_insensitive.get_values()
            assert "test_key" in values  # Lowercase
        
        # Case sensitive
        layer_sensitive = EnvironmentConfigLayer(case_sensitive=True)
        
        with patch.dict(os.environ, {'LOGGING_TEST_KEY': 'value'}):
            values = layer_sensitive.get_values()
            assert "TEST_KEY" in values  # Uppercase preserved


class TestRuntimeConfigLayer:
    """Test cases for RuntimeConfigLayer functionality."""
    
    def test_create_runtime_layer(self):
        """Test creating runtime configuration layer."""
        layer = RuntimeConfigLayer()
        
        assert layer.get_name() == "runtime"
        assert layer.get_priority() == 30
        assert layer.is_available() is True
    
    def test_set_and_get_overrides(self):
        """Test setting and getting configuration overrides."""
        layer = RuntimeConfigLayer()
        
        # Initially empty
        values = layer.get_values()
        assert values == {}
        
        # Set simple override
        layer.set_override("version", 10, "test override")
        values = layer.get_values()
        assert values["version"] == 10
        
        # Set nested override
        layer.set_override("security.enable_pii_detection", True, "security update")
        values = layer.get_values()
        assert values["security"]["enable_pii_detection"] is True
    
    def test_nested_override_handling(self):
        """Test nested configuration override handling."""
        layer = RuntimeConfigLayer()
        
        # Set multiple nested values
        layer.set_override("database.host", "localhost")
        layer.set_override("database.port", 5432)
        layer.set_override("database.credentials.username", "admin")
        layer.set_override("database.credentials.password", "secret")
        
        values = layer.get_values()
        
        assert values["database"]["host"] == "localhost"
        assert values["database"]["port"] == 5432
        assert values["database"]["credentials"]["username"] == "admin"
        assert values["database"]["credentials"]["password"] == "secret"
    
    def test_remove_override(self):
        """Test removing configuration overrides."""
        layer = RuntimeConfigLayer()
        
        # Set overrides
        layer.set_override("test.key1", "value1")
        layer.set_override("test.key2", "value2")
        
        # Remove one override
        removed = layer.remove_override("test.key1", "no longer needed")
        assert removed is True
        
        values = layer.get_values()
        assert "key1" not in values.get("test", {})
        assert values["test"]["key2"] == "value2"
        
        # Try to remove non-existent override
        removed = layer.remove_override("nonexistent.key")
        assert removed is False
    
    def test_clear_overrides(self):
        """Test clearing all configuration overrides."""
        layer = RuntimeConfigLayer()
        
        # Set multiple overrides
        layer.set_override("key1", "value1")
        layer.set_override("key2", "value2")
        layer.set_override("nested.key", "nested_value")
        
        # Check they exist
        values = layer.get_values()
        assert len(values) > 0
        
        # Clear all
        count = layer.clear_overrides("cleanup")
        assert count == 3  # Should return number of cleared overrides
        
        # Check they're gone
        values = layer.get_values()
        assert values == {}
    
    def test_override_history(self):
        """Test override change history tracking."""
        layer = RuntimeConfigLayer()
        
        # Set some overrides
        layer.set_override("key1", "value1", "initial set")
        layer.set_override("key1", "value2", "update value")
        layer.remove_override("key1", "remove key")
        
        # Get history
        history = layer.get_override_history()
        assert len(history) == 3
        
        # Check history entries
        assert history[0]["key"] == "key1"
        assert history[0]["new_value"] == "value1"
        assert history[0]["reason"] == "initial set"
        
        assert history[1]["key"] == "key1"
        assert history[1]["old_value"] == "value1"
        assert history[1]["new_value"] == "value2"
        assert history[1]["reason"] == "update value"
        
        assert history[2]["key"] == "key1"
        assert history[2]["old_value"] == "value2"
        assert history[2]["new_value"] is None
        assert history[2]["reason"] == "remove key"
        assert history[2]["action"] == "remove"
    
    def test_override_count_and_check(self):
        """Test override counting and checking."""
        layer = RuntimeConfigLayer()
        
        # Initially no overrides
        assert layer.get_override_count() == 0
        assert layer.has_override("nonexistent") is False
        
        # Add overrides
        layer.set_override("key1", "value1")
        layer.set_override("nested.key2", "value2")
        
        assert layer.get_override_count() == 2
        assert layer.has_override("key1") is True
        assert layer.has_override("nested.key2") is True
        assert layer.has_override("nonexistent") is False
    
    def test_thread_safety(self):
        """Test thread safety of runtime configuration."""
        layer = RuntimeConfigLayer()
        results = []
        
        def worker(thread_id):
            """Worker function for threading test."""
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                layer.set_override(key, value)
                
                # Verify immediately
                if layer.has_override(key):
                    retrieved_values = layer.get_values()
                    if key in str(retrieved_values):
                        results.append(f"success_{thread_id}_{i}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) > 0  # Should have some successful operations
        
        # Check final state
        final_values = layer.get_values()
        assert layer.get_override_count() > 0


class TestConfigurationLayerIntegration:
    """Test cases for configuration layer integration and edge cases."""
    
    def test_layer_priority_ordering(self):
        """Test that layers are correctly ordered by priority."""
        layers = [
            DefaultConfigLayer("defaults", 0),
            FileConfigLayer(Path("config.toml"), "file", 10),
            EnvironmentConfigLayer("LOGGING_", "env", 20),
            RuntimeConfigLayer("runtime", 30)
        ]
        
        # Sort by priority
        sorted_layers = sorted(layers, key=lambda x: x.get_priority())
        
        priorities = [layer.get_priority() for layer in sorted_layers]
        assert priorities == [0, 10, 20, 30]
        
        names = [layer.get_name() for layer in sorted_layers]
        assert names == ["defaults", "file", "env", "runtime"]
    
    def test_layer_availability_check(self):
        """Test layer availability checking."""
        # Available layer
        available_layer = DefaultConfigLayer()
        assert available_layer.is_available() is True
        
        # Unavailable layer (non-existent file)
        unavailable_layer = FileConfigLayer(Path("non_existent_file.toml"))
        assert unavailable_layer.is_available() is True  # Returns empty dict, still available
    
    def test_layer_update_tracking(self):
        """Test layer update timestamp tracking."""
        layer = RuntimeConfigLayer()
        
        initial_time = layer.get_last_update()
        assert isinstance(initial_time, float)
        
        # Wait a bit
        time.sleep(0.01)
        
        # Make a change
        layer.set_override("test", "value")
        
        updated_time = layer.get_last_update()
        assert updated_time > initial_time
    
    def test_empty_configuration_handling(self):
        """Test handling of empty configurations."""
        # Empty file layer
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'{}')  # Empty JSON
            f.flush()
            
            try:
                layer = FileConfigLayer(Path(f.name))
                values = layer.get_values()
                assert values == {}
                
            finally:
                os.unlink(f.name)
        
        # Empty environment layer
        layer = EnvironmentConfigLayer(prefix="NONEXISTENT_PREFIX_")
        values = layer.get_values()
        assert values == {}
        
        # Empty runtime layer
        layer = RuntimeConfigLayer()
        values = layer.get_values()
        assert values == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])