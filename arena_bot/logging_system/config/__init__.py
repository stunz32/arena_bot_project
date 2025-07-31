"""
Configuration system for the S-Tier Logging System.

This module handles configuration management including JSON schema validation,
default configurations, runtime reconfiguration, and migration utilities
for existing codebases.

Components:
- Schema: JSON schema validation for configurations
- Defaults: Production-ready default configurations
- RuntimeConfig: Thread-safe runtime reconfiguration
- MigrationHelper: Utilities for migrating existing logging code
"""

from .schema import LOGGING_CONFIG_SCHEMA, validate_config
from .defaults import get_default_config, get_production_config, get_development_config
from .runtime_config import RuntimeConfig
from .migration_helper import MigrationHelper

__all__ = [
    'LOGGING_CONFIG_SCHEMA',
    'validate_config',
    'get_default_config',
    'get_production_config', 
    'get_development_config',
    'RuntimeConfig',
    'MigrationHelper'
]