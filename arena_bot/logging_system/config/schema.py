"""
Configuration Schema for S-Tier Logging System.

This module provides JSON schema definitions and validation functions for
the logging system configuration. Used for legacy compatibility and
schema validation.

Features:
- JSON Schema definitions for all configuration components
- Legacy validation functions for backward compatibility
- Schema evolution and versioning support
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# JSON Schema for the logging system configuration
LOGGING_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "S-Tier Logging System Configuration",
    "description": "Configuration schema for the S-tier logging system",
    "type": "object",
    "properties": {
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+(\\.\\d+)?$",
            "description": "Configuration schema version"
        },
        "system": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "name": {"type": "string", "minLength": 1},
                "environment": {"type": "string", "enum": ["development", "staging", "production"]},
                "debug_mode": {"type": "boolean", "default": False}
            },
            "required": ["name"],
            "additionalProperties": False
        },
        "performance": {
            "type": "object",
            "properties": {
                "enable_async": {"type": "boolean", "default": True},
                "queue_size": {"type": "integer", "minimum": 100, "maximum": 1000000},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "worker_threads": {"type": "integer", "minimum": 1, "maximum": 100},
                "max_latency_ms": {"type": "number", "minimum": 0.1, "maximum": 10000},
                "cache_size": {"type": "integer", "minimum": 100, "maximum": 100000}
            },
            "additionalProperties": False
        },
        "security": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "pii_detection": {"type": "boolean", "default": True},
                "credential_scrubbing": {"type": "boolean", "default": True},
                "encryption_enabled": {"type": "boolean", "default": False},
                "compliance_mode": {
                    "type": "string",
                    "enum": ["none", "gdpr", "hipaa", "pci_dss", "sox", "iso27001"]
                }
            },
            "additionalProperties": False
        },
        "loggers": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_.-]+$": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "handlers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "propagate": {"type": "boolean", "default": True},
                        "disabled": {"type": "boolean", "default": False}
                    },
                    "required": ["level"],
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "handlers": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_.-]+$": {
                    "type": "object",
                    "properties": {
                        "class": {"type": "string", "minLength": 1},
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "formatter": {"type": "string"},
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["class"],
                    "additionalProperties": True
                }
            },
            "additionalProperties": False
        },
        "formatters": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_.-]+$": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string"},
                        "datefmt": {"type": "string"},
                        "style": {"type": "string", "enum": ["%", "{", "$"]},
                        "validate": {"type": "boolean", "default": True}
                    },
                    "additionalProperties": True
                }
            },
            "additionalProperties": False
        },
        "filters": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_.-]+$": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"}
                    },
                    "additionalProperties": True
                }
            },
            "additionalProperties": False
        },
        "sinks": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_.-]+$": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["console", "file", "network", "metrics", "emergency"]
                        },
                        "enabled": {"type": "boolean", "default": True},
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "formatter": {"type": "string"}
                    },
                    "required": ["type"],
                    "additionalProperties": True
                }
            },
            "additionalProperties": False
        },
        "diagnostics": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "health_check_interval": {"type": "number", "minimum": 1, "maximum": 3600},
                "performance_monitoring": {"type": "boolean", "default": True},
                "metrics_collection": {"type": "boolean", "default": True},
                "emergency_protocols": {"type": "boolean", "default": True}
            },
            "additionalProperties": False
        }
    },
    "required": ["version", "system"],
    "additionalProperties": False
}


def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate configuration against schema (legacy function).
    
    This is a simplified validation function for backward compatibility.
    For full validation, use the Pydantic models.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Basic validation checks
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors
        
        # Check required fields
        if 'version' not in config:
            errors.append("Missing required field: version")
        
        if 'system' not in config:
            errors.append("Missing required field: system")
        elif not isinstance(config['system'], dict):
            errors.append("Field 'system' must be a dictionary")
        elif 'name' not in config['system']:
            errors.append("Missing required field: system.name")
        
        # Validate performance settings if present
        if 'performance' in config:
            perf = config['performance']
            if not isinstance(perf, dict):
                errors.append("Field 'performance' must be a dictionary")
            else:
                if 'queue_size' in perf:
                    queue_size = perf['queue_size']
                    if not isinstance(queue_size, int) or queue_size < 100 or queue_size > 1000000:
                        errors.append("Field 'performance.queue_size' must be between 100 and 1000000")
                
                if 'worker_threads' in perf:
                    threads = perf['worker_threads']
                    if not isinstance(threads, int) or threads < 1 or threads > 100:
                        errors.append("Field 'performance.worker_threads' must be between 1 and 100")
        
        # Validate security settings if present
        if 'security' in config:
            security = config['security']
            if not isinstance(security, dict):
                errors.append("Field 'security' must be a dictionary")
            else:
                if 'compliance_mode' in security:
                    mode = security['compliance_mode']
                    valid_modes = ["none", "gdpr", "hipaa", "pci_dss", "sox", "iso27001"]
                    if mode not in valid_modes:
                        errors.append(f"Field 'security.compliance_mode' must be one of: {valid_modes}")
        
        # Validate loggers if present
        if 'loggers' in config:
            loggers = config['loggers']
            if not isinstance(loggers, dict):
                errors.append("Field 'loggers' must be a dictionary")
            else:
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                for logger_name, logger_config in loggers.items():
                    if not isinstance(logger_config, dict):
                        errors.append(f"Logger '{logger_name}' configuration must be a dictionary")
                        continue
                    
                    if 'level' not in logger_config:
                        errors.append(f"Logger '{logger_name}' missing required field: level")
                    elif logger_config['level'] not in valid_levels:
                        errors.append(f"Logger '{logger_name}' level must be one of: {valid_levels}")
        
        # Validate sinks if present
        if 'sinks' in config:
            sinks = config['sinks']
            if not isinstance(sinks, dict):
                errors.append("Field 'sinks' must be a dictionary")
            else:
                valid_types = ["console", "file", "network", "metrics", "emergency"]
                for sink_name, sink_config in sinks.items():
                    if not isinstance(sink_config, dict):
                        errors.append(f"Sink '{sink_name}' configuration must be a dictionary")
                        continue
                    
                    if 'type' not in sink_config:
                        errors.append(f"Sink '{sink_name}' missing required field: type")
                    elif sink_config['type'] not in valid_types:
                        errors.append(f"Sink '{sink_name}' type must be one of: {valid_types}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
        
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the logging system.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "version": "1.0",
        "system": {
            "enabled": True,
            "name": "s-tier-logging",
            "environment": "development",
            "debug_mode": False
        },
        "performance": {
            "enable_async": True,
            "queue_size": 10000,
            "batch_size": 100,
            "worker_threads": 4,
            "max_latency_ms": 50.0,
            "cache_size": 1000
        },
        "security": {
            "enabled": True,
            "pii_detection": True,
            "credential_scrubbing": True,
            "encryption_enabled": False,
            "compliance_mode": "none"
        },
        "diagnostics": {
            "enabled": True,
            "health_check_interval": 60.0,
            "performance_monitoring": True,
            "metrics_collection": True,
            "emergency_protocols": True
        },
        "loggers": {
            "root": {
                "level": "INFO",
                "handlers": ["default_console"]
            }
        },
        "handlers": {
            "default_console": {
                "class": "console",
                "level": "INFO",
                "formatter": "structured"
            }
        },
        "formatters": {
            "structured": {
                "format": "json"
            },
            "console": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "sinks": {
            "default_console": {
                "type": "console",
                "enabled": True,
                "level": "INFO",
                "formatter": "structured"
            }
        }
    }


def upgrade_config_schema(config: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
    """
    Upgrade configuration schema from one version to another.
    
    Args:
        config: Configuration to upgrade
        from_version: Source schema version
        to_version: Target schema version
        
    Returns:
        Upgraded configuration
        
    Raises:
        ValueError: If upgrade path is not supported
    """
    if from_version == to_version:
        return config
    
    logger.info(f"Upgrading configuration schema from {from_version} to {to_version}")
    
    # For now, we only support one version
    if to_version == "1.0":
        # Ensure all required fields are present
        upgraded = config.copy()
        
        if 'version' not in upgraded:
            upgraded['version'] = "1.0"
        
        if 'system' not in upgraded:
            upgraded['system'] = {"name": "s-tier-logging"}
        
        return upgraded
    
    raise ValueError(f"Unsupported schema upgrade path: {from_version} -> {to_version}")


# Module exports
__all__ = [
    'LOGGING_CONFIG_SCHEMA',
    'validate_config',
    'get_default_config',
    'upgrade_config_schema'
]