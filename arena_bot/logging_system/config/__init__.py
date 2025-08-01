"""
Configuration System for S-Tier Logging System.

This module provides enterprise-grade configuration management with schema validation,
hot-reload capabilities, hierarchical configuration layers, and performance-optimized
access patterns for the S-tier logging system.

Features:
- Pydantic-based schema validation with <100μs performance
- TOML configuration files with native Python 3.11+ support
- Hot-reload and zero-downtime configuration updates
- Hierarchical configuration with inheritance (defaults → files → env → runtime)
- Security-focused design with PII redaction and encryption
- Performance-optimized caching with <10μs access time
- Enterprise compliance and audit trail support
"""

# Import configuration models and validation
from .models import (
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
    create_development_config,
    create_production_config
)

# Import configuration management components
from .manager import (
    ConfigurationManager,
    HierarchicalConfigurationManager,
    SecurityConfigurationManager,
    HighPerformanceConfigCache,
    ConfigurationChangeNotifier,
    ConfigurationFactory,
    get_config_fast,
    initialize_config_cache
)

# Import configuration layers
from .layers import (
    ConfigurationLayer,
    DefaultConfigLayer,
    FileConfigLayer,
    EnvironmentConfigLayer,
    RuntimeConfigLayer
)

# Import validation and schema utilities
from .validation import (
    ConfigurationValidator,
    ValidationError,
    ValidationRule,
    SecurityValidator
)

# Import utilities and helpers
from .utils import (
    load_toml_config,
    merge_configurations,
    flatten_config_dict,
    unflatten_config_dict
)

# Legacy imports for backward compatibility
from .schema import LOGGING_CONFIG_SCHEMA, validate_config
from .defaults import get_default_config, get_production_config, get_development_config
from .runtime_config import RuntimeConfig
from .migration_helper import MigrationHelper

# Module exports
__all__ = [
    # Core configuration models
    'LoggingSystemConfig',
    'LoggerConfig', 
    'HandlerConfig',
    'SinkConfig',
    'FilterConfig',
    'SecurityConfig',
    'PerformanceConfig',
    'DiagnosticsConfig',
    'RetentionPolicy',
    'EncryptionConfig',
    
    # Configuration management
    'ConfigurationManager',
    'HierarchicalConfigurationManager', 
    'SecurityConfigurationManager',
    'HighPerformanceConfigCache',
    'ConfigurationChangeNotifier',
    'ConfigurationFactory',
    
    # Configuration layers
    'ConfigurationLayer',
    'DefaultConfigLayer',
    'FileConfigLayer',
    'EnvironmentConfigLayer',
    'RuntimeConfigLayer',
    
    # Validation
    'ConfigurationValidator',
    'ValidationError',
    'ValidationRule',
    'SecurityValidator',
    
    # Utilities
    'get_config_fast',
    'initialize_config_cache',
    'load_toml_config',
    'merge_configurations',
    'flatten_config_dict',
    'unflatten_config_dict',
    
    # Legacy compatibility
    'LOGGING_CONFIG_SCHEMA',
    'validate_config',
    'get_default_config',
    'get_production_config', 
    'get_development_config',
    'create_development_config',
    'create_production_config',
    'RuntimeConfig',
    'MigrationHelper'
]