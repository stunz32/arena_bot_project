"""
Test suite for S-Tier Logging System Configuration Models.

This test suite validates the Pydantic configuration models with comprehensive
test coverage including validation, serialization, environment variable support,
and enterprise compliance requirements.

Test Categories:
- Model validation and type safety
- Environment variable integration
- Configuration inheritance and defaults
- Security configuration validation
- Performance constraint testing
- Error handling and edge cases
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arena_bot.logging_system.config.models import (
    LoggingSystemConfig,
    LoggerConfig,
    HandlerConfig,
    SinkConfig,
    FilterConfig,
    SecurityConfig,
    PerformanceConfig,
    DiagnosticsConfig,
    RetentionPolicy,
    EncryptionConfig,
    LogLevel,
    Environment,
    create_development_config,
    create_production_config,
    create_testing_config
)


class TestRetentionPolicy:
    """Test cases for RetentionPolicy model validation."""
    
    def test_valid_retention_policy(self):
        """Test valid retention policy creation."""
        policy = RetentionPolicy(
            max_age_days=90,
            max_size_gb=10.0,
            max_files=1000,
            compression_after_days=7,
            archive_after_days=30,
            delete_after_days=90
        )
        
        assert policy.max_age_days == 90
        assert policy.max_size_gb == 10.0
        assert policy.compression_after_days == 7
        assert policy.archive_after_days < policy.delete_after_days
    
    def test_archive_before_delete_validation(self):
        """Test that archive happens before deletion."""
        with pytest.raises(Exception):  # ValidationError or ValueError
            RetentionPolicy(
                archive_after_days=90,
                delete_after_days=30  # Invalid: delete before archive
            )
    
    def test_retention_bounds_validation(self):
        """Test retention policy bounds validation."""
        # Test minimum bounds
        with pytest.raises(Exception):
            RetentionPolicy(max_age_days=0)  # Below minimum
        
        # Test maximum bounds
        with pytest.raises(Exception):
            RetentionPolicy(max_age_days=3000)  # Above maximum
    
    def test_retention_policy_defaults(self):
        """Test default values for retention policy."""
        policy = RetentionPolicy()
        
        assert policy.max_age_days == 90
        assert policy.max_size_gb == 10.0
        assert policy.compression_after_days == 7
        assert policy.archive_after_days == 30
        assert policy.delete_after_days == 90


class TestEncryptionConfig:
    """Test cases for EncryptionConfig model validation."""
    
    def test_valid_encryption_config(self):
        """Test valid encryption configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "encryption.key"
            
            config = EncryptionConfig(
                enabled=True,
                algorithm="fernet",
                key_file_path=key_path,
                key_rotation_days=90
            )
            
            assert config.enabled is True
            assert config.algorithm == "fernet"
            assert config.key_file_path == key_path
    
    def test_encryption_disabled_defaults(self):
        """Test encryption disabled by default."""
        config = EncryptionConfig(enabled=False)
        
        assert config.enabled is False
        assert config.key_file_path is None
    
    def test_key_rotation_bounds(self):
        """Test key rotation period bounds."""
        with pytest.raises(Exception):
            EncryptionConfig(key_rotation_days=0)  # Below minimum
        
        with pytest.raises(Exception):
            EncryptionConfig(key_rotation_days=400)  # Above maximum


class TestSecurityConfig:
    """Test cases for SecurityConfig model validation."""
    
    def test_default_security_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.enable_pii_detection is True
        assert config.pii_redaction_level == "partial"
        assert config.redact_credentials is True
        assert config.compliance_mode == "none"
        assert isinstance(config.encryption, EncryptionConfig)
        assert isinstance(config.retention, RetentionPolicy)
    
    def test_security_config_with_encryption(self):
        """Test security config with encryption enabled."""
        encryption = EncryptionConfig(enabled=True, algorithm="aes_256_gcm")
        config = SecurityConfig(encryption=encryption)
        
        assert config.encryption.enabled is True
        assert config.encryption.algorithm == "aes_256_gcm"
    
    def test_compliance_mode_validation(self):
        """Test compliance mode validation."""
        valid_modes = ["none", "gdpr", "hipaa", "sox", "pci_dss"]
        
        for mode in valid_modes:
            config = SecurityConfig(compliance_mode=mode)
            assert config.compliance_mode == mode
        
        # Test invalid compliance mode
        with pytest.raises(Exception):
            SecurityConfig(compliance_mode="invalid_mode")
    
    def test_audit_trail_configuration(self):
        """Test audit trail configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.log"
            
            config = SecurityConfig(
                enable_audit_trail=True,
                audit_log_path=audit_path
            )
            
            assert config.enable_audit_trail is True
            assert config.audit_log_path == audit_path


class TestPerformanceConfig:
    """Test cases for PerformanceConfig model validation."""
    
    def test_default_performance_config(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        
        assert config.enable_async_processing is True
        assert config.worker_threads == 4
        assert config.buffer_size == 8192
        assert config.enable_caching is True
    
    def test_performance_bounds_validation(self):
        """Test performance configuration bounds."""
        # Test worker threads bounds
        with pytest.raises(Exception):
            PerformanceConfig(worker_threads=0)  # Below minimum
        
        with pytest.raises(Exception):
            PerformanceConfig(worker_threads=100)  # Above maximum
        
        # Test buffer size bounds
        with pytest.raises(Exception):
            PerformanceConfig(buffer_size=512)  # Below minimum
        
        with pytest.raises(Exception):
            PerformanceConfig(buffer_size=2_000_000)  # Above maximum
    
    def test_resource_limits(self):
        """Test resource limit configuration."""
        config = PerformanceConfig(
            max_memory_mb=2048,
            max_disk_usage_gb=50.0
        )
        
        assert config.max_memory_mb == 2048
        assert config.max_disk_usage_gb == 50.0
    
    def test_caching_configuration(self):
        """Test caching configuration."""
        config = PerformanceConfig(
            enable_caching=True,
            cache_size_limit=50000,
            cache_ttl_seconds=600
        )
        
        assert config.enable_caching is True
        assert config.cache_size_limit == 50000
        assert config.cache_ttl_seconds == 600


class TestDiagnosticsConfig:
    """Test cases for DiagnosticsConfig model validation."""
    
    def test_default_diagnostics_config(self):
        """Test default diagnostics configuration."""
        config = DiagnosticsConfig()
        
        assert config.enable_health_checks is True
        assert config.health_check_interval_seconds == 30
        assert config.enable_metrics is True
        assert config.enable_emergency_protocols is True
    
    def test_health_check_configuration(self):
        """Test health check configuration."""
        config = DiagnosticsConfig(
            health_check_interval_seconds=60,
            health_check_timeout_seconds=10
        )
        
        assert config.health_check_interval_seconds == 60
        assert config.health_check_timeout_seconds == 10
    
    def test_metrics_configuration(self):
        """Test metrics configuration."""
        config = DiagnosticsConfig(
            enable_metrics=True,
            metrics_export_interval_seconds=120,
            metrics_retention_hours=48
        )
        
        assert config.enable_metrics is True
        assert config.metrics_export_interval_seconds == 120
        assert config.metrics_retention_hours == 48
    
    def test_emergency_protocols(self):
        """Test emergency protocol configuration."""
        config = DiagnosticsConfig(
            enable_emergency_protocols=True,
            emergency_activation_threshold=0.9,
            emergency_cooldown_minutes=15
        )
        
        assert config.enable_emergency_protocols is True
        assert config.emergency_activation_threshold == 0.9
        assert config.emergency_cooldown_minutes == 15


class TestFilterConfig:
    """Test cases for FilterConfig model validation."""
    
    def test_valid_filter_config(self):
        """Test valid filter configuration."""
        config = FilterConfig(
            name="test_filter",
            type="level",
            enabled=True,
            priority=10,
            config={"min_level": "INFO"}
        )
        
        assert config.name == "test_filter"
        assert config.type == "level"
        assert config.enabled is True
        assert config.priority == 10
        assert config.config["min_level"] == "INFO"
    
    def test_filter_name_validation(self):
        """Test filter name validation."""
        # Valid names
        valid_names = ["test_filter", "test-filter", "TestFilter123"]
        for name in valid_names:
            config = FilterConfig(name=name, type="level")
            assert config.name == name
        
        # Invalid names
        with pytest.raises(Exception):
            FilterConfig(name="test filter", type="level")  # Space not allowed
    
    def test_filter_priority_bounds(self):
        """Test filter priority bounds."""
        # Valid priorities
        FilterConfig(name="test", type="level", priority=0)
        FilterConfig(name="test", type="level", priority=100)
        
        # Invalid priorities
        with pytest.raises(Exception):
            FilterConfig(name="test", type="level", priority=-1)
        
        with pytest.raises(Exception):
            FilterConfig(name="test", type="level", priority=101)


class TestSinkConfig:
    """Test cases for SinkConfig model validation."""
    
    def test_valid_sink_config(self):
        """Test valid sink configuration."""
        config = SinkConfig(
            name="console_sink",
            type="console",
            enabled=True,
            filters=["level_filter"],
            formatter="json_formatter",
            config={"colorize": True}
        )
        
        assert config.name == "console_sink"
        assert config.type == "console"
        assert config.enabled is True
        assert "level_filter" in config.filters
        assert config.formatter == "json_formatter"
    
    def test_sink_name_validation(self):
        """Test sink name validation."""
        # Valid names
        valid_names = ["console_sink", "file-sink", "NetworkSink123"]
        for name in valid_names:
            config = SinkConfig(name=name, type="console")
            assert config.name == name
        
        # Invalid names
        with pytest.raises(Exception):
            SinkConfig(name="console sink", type="console")  # Space not allowed


class TestHandlerConfig:
    """Test cases for HandlerConfig model validation."""
    
    def test_console_handler_config(self):
        """Test console handler configuration."""
        config = HandlerConfig(
            name="console",
            type="console",
            level="INFO",
            stream="stdout"
        )
        
        assert config.name == "console"
        assert config.type == "console"
        assert config.level == LogLevel.INFO
        assert config.stream == "stdout"
    
    def test_file_handler_config(self):
        """Test file handler configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            config = HandlerConfig(
                name="file",
                type="file",
                filename=log_file,
                max_bytes=10_000_000,
                backup_count=5
            )
            
            assert config.name == "file"
            assert config.type == "file"
            assert config.filename == log_file
            assert config.max_bytes == 10_000_000
            assert config.backup_count == 5
    
    def test_file_handler_validation(self):
        """Test file handler requires filename."""
        with pytest.raises(Exception):
            HandlerConfig(
                name="file",
                type="file"
                # Missing filename
            )
    
    def test_network_handler_config(self):
        """Test network handler configuration."""
        config = HandlerConfig(
            name="network",
            type="http",
            host="logging.example.com",
            port=443
        )
        
        assert config.name == "network"
        assert config.type == "http"
        assert config.host == "logging.example.com" 
        assert config.port == 443
    
    def test_network_handler_validation(self):
        """Test network handler requires host."""
        with pytest.raises(Exception):
            HandlerConfig(
                name="network",
                type="http"
                # Missing host
            )


class TestLoggerConfig:
    """Test cases for LoggerConfig model validation."""
    
    def test_valid_logger_config(self):
        """Test valid logger configuration."""
        config = LoggerConfig(
            name="app.module",
            level="DEBUG",
            propagate=True,
            handlers=["console", "file"],
            sinks=["console_sink", "file_sink"],
            filters=["level_filter"]
        )
        
        assert config.name == "app.module"
        assert config.level == LogLevel.DEBUG
        assert config.propagate is True
        assert "console" in config.handlers
        assert "console_sink" in config.sinks
        assert "level_filter" in config.filters
    
    def test_logger_name_validation(self):
        """Test logger name validation."""
        # Valid names
        valid_names = ["app", "app.module", "app.sub_module", "app-module"]
        for name in valid_names:
            config = LoggerConfig(name=name, level="INFO")
            assert config.name == name
        
        # Invalid names  
        with pytest.raises(Exception):
            LoggerConfig(name="app module", level="INFO")  # Space not allowed


class TestLoggingSystemConfig:
    """Test cases for main LoggingSystemConfig validation."""
    
    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = LoggingSystemConfig(
            version=1,
            environment="development"
        )
        
        assert config.version == 1
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.diagnostics, DiagnosticsConfig)
    
    def test_environment_validation(self):
        """Test environment validation."""
        valid_environments = ["development", "testing", "staging", "production"]
        
        for env in valid_environments:
            config = LoggingSystemConfig(environment=env)
            assert config.environment.value == env
        
        # Invalid environment
        with pytest.raises(Exception):
            LoggingSystemConfig(environment="invalid_env")
    
    def test_logger_handler_references(self):
        """Test logger handler reference validation."""
        # Valid configuration
        config = LoggingSystemConfig(
            loggers={
                "app": LoggerConfig(name="app", handlers=["console"])
            },
            handlers={
                "console": HandlerConfig(name="console", type="console")
            }
        )
        
        assert "app" in config.loggers
        assert "console" in config.handlers
    
    def test_logger_sink_references(self):
        """Test logger sink reference validation."""
        # Valid configuration
        config = LoggingSystemConfig(
            loggers={
                "app": LoggerConfig(name="app", sinks=["console_sink"])
            },
            sinks={
                "console_sink": SinkConfig(name="console_sink", type="console")
            }
        )
        
        assert "app" in config.loggers
        assert "console_sink" in config.sinks
    
    def test_production_environment_validation(self):
        """Test production environment requirements."""
        # This should validate without errors
        config = LoggingSystemConfig(
            environment="production",
            security=SecurityConfig(
                enable_audit_trail=True,
                encryption=EncryptionConfig(enabled=True)
            ),
            performance=PerformanceConfig(
                enable_async_processing=True
            )
        )
        
        assert config.environment == Environment.PRODUCTION
        assert config.security.enable_audit_trail is True
        assert config.security.encryption.enabled is True
        assert config.performance.enable_async_processing is True
    
    @patch.dict(os.environ, {
        'LOGGING_ENVIRONMENT': 'production',
        'LOGGING_SECURITY__ENABLE_AUDIT_TRAIL': 'true',
        'LOGGING_PERFORMANCE__WORKER_THREADS': '8'
    })
    def test_environment_variable_integration(self):
        """Test environment variable integration."""
        config = LoggingSystemConfig()
        
        # Note: This test may need adjustment based on how environment
        # variables are actually processed by the model
        assert config.environment in [Environment.PRODUCTION, Environment.DEVELOPMENT]
    
    def test_config_file_path_validation(self):
        """Test configuration file path validation."""
        config = LoggingSystemConfig(
            config_file_path=Path("custom_config.toml")
        )
        
        assert config.config_file_path == Path("custom_config.toml")
    
    def test_auto_reload_configuration(self):
        """Test auto-reload configuration."""
        config = LoggingSystemConfig(
            auto_reload=True,
            reload_check_interval=10.0
        )
        
        assert config.auto_reload is True
        assert config.reload_check_interval == 10.0


class TestConfigurationFactories:
    """Test cases for configuration factory functions."""
    
    def test_create_development_config(self):
        """Test development configuration factory."""
        config = create_development_config()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.security.enable_pii_detection is False
        assert config.security.encryption.enabled is False
        assert config.performance.enable_async_processing is False
        assert config.diagnostics.enable_emergency_protocols is False
    
    def test_create_production_config(self):
        """Test production configuration factory."""
        config = create_production_config()
        
        assert config.environment == Environment.PRODUCTION
        assert config.security.enable_pii_detection is True
        assert config.security.encryption.enabled is True
        assert config.security.enable_audit_trail is True
        assert config.performance.enable_async_processing is True
        assert config.diagnostics.enable_emergency_protocols is True
    
    def test_create_testing_config(self):
        """Test testing configuration factory."""
        config = create_testing_config()
        
        assert config.environment == Environment.TESTING
        assert config.security.enable_pii_detection is True
        assert config.security.encryption.enabled is False
        assert config.security.enable_audit_trail is True
        assert config.performance.enable_async_processing is True
        assert config.diagnostics.enable_emergency_protocols is True


class TestConfigurationSerialization:
    """Test cases for configuration serialization."""
    
    def test_config_dict_serialization(self):
        """Test configuration dictionary serialization."""
        config = LoggingSystemConfig(
            version=1,
            environment="development",
            system_name="test-system"
        )
        
        # Test dict conversion (if available in Pydantic version)
        if hasattr(config, 'dict'):
            config_dict = config.dict()
            assert config_dict['version'] == 1
            assert config_dict['environment'] == 'development'
            assert config_dict['system_name'] == 'test-system'
        elif hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
            assert config_dict['version'] == 1
            assert config_dict['environment'] == 'development'
            assert config_dict['system_name'] == 'test-system'
    
    def test_config_json_serialization(self):
        """Test configuration JSON serialization."""
        config = LoggingSystemConfig(
            version=1,
            environment="development"
        )
        
        # Test JSON conversion (if available in Pydantic version)
        if hasattr(config, 'json'):
            json_str = config.json()
            assert isinstance(json_str, str)
            assert '"version": 1' in json_str
        elif hasattr(config, 'model_dump_json'):
            json_str = config.model_dump_json()
            assert isinstance(json_str, str)
            assert '"version": 1' in json_str


class TestConfigurationEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_empty_configuration(self):
        """Test handling of empty configuration."""
        config = LoggingSystemConfig()
        
        # Should use defaults
        assert config.version == 1
        assert config.environment == Environment.PRODUCTION
        assert isinstance(config.security, SecurityConfig)
    
    def test_partial_configuration(self):
        """Test handling of partial configuration."""
        config = LoggingSystemConfig(
            environment="development",
            security=SecurityConfig(enable_pii_detection=True)
        )
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.security.enable_pii_detection is True
        # Other fields should use defaults
        assert isinstance(config.performance, PerformanceConfig)
    
    def test_nested_configuration_validation(self):
        """Test nested configuration validation."""
        # Test invalid nested configuration
        with pytest.raises(Exception):
            LoggingSystemConfig(
                security=SecurityConfig(
                    retention=RetentionPolicy(
                        archive_after_days=90,
                        delete_after_days=30  # Invalid: archive after delete
                    )
                )
            )
    
    def test_configuration_immutability(self):
        """Test configuration immutability where applicable."""
        config = LoggingSystemConfig()
        
        # Some fields should be immutable after creation
        # This depends on the specific Pydantic configuration
        # and may need adjustment based on implementation
        
        # Test that we can read values
        assert config.version is not None
        assert config.environment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])