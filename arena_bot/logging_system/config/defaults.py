"""
Default Configurations for S-Tier Logging System.

This module provides default configuration presets for different environments
and use cases, including development, production, and high-performance setups.

Features:
- Environment-specific configuration presets
- Performance-optimized defaults
- Security-focused production settings
- Development-friendly debugging configurations
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .models import LoggingSystemConfig


def get_default_config() -> Dict[str, Any]:
    """
    Get basic default configuration for the logging system.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "version": "1.0",
        "system": {
            "name": "s-tier-logging",
            "environment": "development",
            "enabled": True,
            "debug_mode": False
        },
        "performance": {
            "enable_async": True,
            "queue_size": 10000,
            "batch_size": 100,
            "worker_threads": 4,
            "max_latency_ms": 50.0,
            "cache_size": 1000,
            "optimization_level": "balanced"
        },
        "security": {
            "enabled": True,
            "pii_detection": True,
            "credential_scrubbing": True,
            "encryption_enabled": False,
            "compliance_mode": "none",
            "audit_trail": False
        },
        "diagnostics": {
            "enabled": True,
            "health_check_interval": 60.0,
            "performance_monitoring": True,
            "metrics_collection": True,
            "emergency_protocols": True,
            "profiling_enabled": False
        },
        "loggers": {
            "root": {
                "level": "INFO",
                "handlers": ["default_console"],
                "propagate": True,
                "disabled": False
            }
        },
        "handlers": {
            "default_console": {
                "class": "console",
                "level": "INFO", 
                "formatter": "structured",
                "filters": []
            }
        },
        "formatters": {
            "structured": {
                "format": "json",
                "include_context": True,
                "include_performance": False
            },
            "console": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "default_level": {
                "type": "level",
                "level": "INFO"
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


def get_development_config() -> Dict[str, Any]:
    """
    Get development environment configuration.
    
    Optimized for debugging and development with enhanced logging
    and diagnostic features enabled.
    
    Returns:
        Development configuration dictionary
    """
    config = get_default_config()
    
    # Development-specific overrides
    config.update({
        "system": {
            **config["system"],
            "environment": "development",
            "debug_mode": True
        },
        "performance": {
            **config["performance"],
            "optimization_level": "debug",
            "queue_size": 5000,
            "worker_threads": 2
        },
        "diagnostics": {
            **config["diagnostics"],
            "profiling_enabled": True,
            "health_check_interval": 30.0
        },
        "loggers": {
            "root": {
                **config["loggers"]["root"],
                "level": "DEBUG"
            },
            "arena_bot": {
                "level": "DEBUG",
                "handlers": ["debug_console", "debug_file"],
                "propagate": False
            }
        },
        "handlers": {
            **config["handlers"],
            "debug_console": {
                "class": "console",
                "level": "DEBUG",
                "formatter": "debug_console"
            },
            "debug_file": {
                "class": "file",
                "level": "DEBUG",
                "formatter": "structured",
                "filename": "logs/debug.log"
            }
        },
        "formatters": {
            **config["formatters"],
            "debug_console": {
                "format": "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s",
                "datefmt": "%H:%M:%S",
                "style": "%"
            }
        },
        "sinks": {
            **config["sinks"],
            "debug_file": {
                "type": "file",
                "enabled": True,
                "level": "DEBUG",
                "formatter": "structured",
                "filename": "logs/debug.log",
                "rotation": "size:10MB"
            }
        }
    })
    
    return config


def get_production_config() -> Dict[str, Any]:
    """
    Get production environment configuration.
    
    Optimized for performance, security, and reliability with
    minimal overhead and enhanced security features.
    
    Returns:
        Production configuration dictionary
    """
    config = get_default_config()
    
    # Production-specific overrides
    config.update({
        "system": {
            **config["system"],
            "environment": "production",
            "debug_mode": False
        },
        "performance": {
            **config["performance"],
            "optimization_level": "maximum",
            "queue_size": 50000,
            "batch_size": 500,
            "worker_threads": 8,
            "max_latency_ms": 25.0,
            "cache_size": 5000
        },
        "security": {
            **config["security"],
            "encryption_enabled": True,
            "compliance_mode": "gdpr",
            "audit_trail": True,
            "pii_detection": True,
            "credential_scrubbing": True
        },
        "diagnostics": {
            **config["diagnostics"],
            "health_check_interval": 300.0,  # 5 minutes
            "profiling_enabled": False
        },
        "loggers": {
            "root": {
                **config["loggers"]["root"],
                "level": "WARNING",
                "handlers": ["production_file", "error_file"]
            },
            "arena_bot": {
                "level": "INFO",
                "handlers": ["application_file"],
                "propagate": False
            },
            "security": {
                "level": "INFO",
                "handlers": ["security_file"],
                "propagate": False
            }
        },
        "handlers": {
            "production_file": {
                "class": "file",
                "level": "WARNING",
                "formatter": "structured",
                "filename": "logs/production.log"
            },
            "error_file": {
                "class": "file", 
                "level": "ERROR",
                "formatter": "structured",
                "filename": "logs/errors.log"
            },
            "application_file": {
                "class": "file",
                "level": "INFO",
                "formatter": "structured",
                "filename": "logs/application.log"
            },
            "security_file": {
                "class": "file",
                "level": "INFO",
                "formatter": "structured", 
                "filename": "logs/security.log"
            }
        },
        "filters": {
            **config["filters"],
            "rate_limiter": {
                "type": "rate_limiter",
                "rate": 1000,
                "per": "minute"
            },
            "security_filter": {
                "type": "security",
                "pii_detection": True,
                "credential_scrubbing": True
            }
        },
        "sinks": {
            "production_file": {
                "type": "file",
                "enabled": True,
                "level": "WARNING",
                "formatter": "structured",
                "filename": "logs/production.log",
                "rotation": "time:1d",
                "compression": "gzip",
                "retention": "30d"
            },
            "error_file": {
                "type": "file",
                "enabled": True,
                "level": "ERROR",
                "formatter": "structured",
                "filename": "logs/errors.log",
                "rotation": "size:100MB",
                "retention": "90d"
            },
            "metrics_sink": {
                "type": "metrics",
                "enabled": True,
                "level": "INFO",
                "endpoint": "http://metrics-server:8080/metrics"
            },
            "emergency_sink": {
                "type": "emergency",
                "enabled": True,
                "level": "CRITICAL",
                "buffer_size": 1000
            }
        }
    })
    
    return config


def get_high_performance_config() -> Dict[str, Any]:
    """
    Get high-performance configuration.
    
    Optimized for maximum throughput and minimal latency
    with reduced logging overhead.
    
    Returns:
        High-performance configuration dictionary
    """
    config = get_default_config()
    
    # High-performance overrides
    config.update({
        "system": {
            **config["system"],
            "environment": "production"
        },
        "performance": {
            **config["performance"],
            "optimization_level": "maximum",
            "queue_size": 100000,
            "batch_size": 1000,
            "worker_threads": 16,
            "max_latency_ms": 10.0,
            "cache_size": 10000
        },
        "security": {
            **config["security"],
            "pii_detection": False,  # Disabled for performance
            "credential_scrubbing": True,
            "encryption_enabled": False
        },
        "diagnostics": {
            **config["diagnostics"],
            "performance_monitoring": False,  # Reduced overhead
            "health_check_interval": 600.0,  # 10 minutes
            "profiling_enabled": False
        },
        "loggers": {
            "root": {
                **config["loggers"]["root"],
                "level": "ERROR"  # Minimal logging
            }
        },
        "sinks": {
            "high_perf_file": {
                "type": "file",
                "enabled": True,
                "level": "ERROR",
                "formatter": "structured",
                "filename": "logs/high-perf.log",
                "rotation": "size:1GB",
                "buffer_size": 100000
            }
        }
    })
    
    return config


def get_config_for_environment(environment: str) -> Dict[str, Any]:
    """
    Get configuration for specific environment.
    
    Args:
        environment: Environment name ('development', 'production', 'high-performance')
        
    Returns:
        Configuration dictionary for the environment
        
    Raises:
        ValueError: If environment is not supported
    """
    environment = environment.lower()
    
    if environment in ['dev', 'development']:
        return get_development_config()
    elif environment in ['prod', 'production']:
        return get_production_config()
    elif environment in ['perf', 'high-performance', 'high_performance']:
        return get_high_performance_config()
    elif environment in ['default', 'basic']:
        return get_default_config()
    else:
        raise ValueError(f"Unsupported environment: {environment}")


def create_pydantic_config(config_dict: Dict[str, Any]) -> LoggingSystemConfig:
    """
    Create Pydantic configuration model from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated LoggingSystemConfig instance
    """
    try:
        return LoggingSystemConfig(**config_dict)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create Pydantic config: {e}")
        # Return default config on error
        return LoggingSystemConfig(**get_default_config())


def save_default_config_file(
    file_path: Optional[Path] = None,
    environment: str = "development",
    format: str = "toml"
) -> Path:
    """
    Save default configuration to file.
    
    Args:
        file_path: Output file path (auto-generated if None)
        environment: Environment configuration to save
        format: File format ('toml', 'json', 'yaml')
        
    Returns:
        Path to saved configuration file
    """
    from .utils import save_config_file
    
    if file_path is None:
        file_path = Path(f"logging_config_{environment}.{format}")
    
    config = get_config_for_environment(environment)
    save_config_file(config, file_path, format)
    
    return file_path


# Module exports
__all__ = [
    'get_default_config',
    'get_development_config',
    'get_production_config', 
    'get_high_performance_config',
    'get_config_for_environment',
    'create_pydantic_config',
    'save_default_config_file'
]