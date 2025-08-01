"""
Configuration Models for S-Tier Logging System.

This module defines Pydantic models for comprehensive configuration management
with enterprise-grade validation, type safety, and performance optimization.

Features:
- Type-safe configuration models with validation
- Nested configuration structures with inheritance
- Performance-optimized serialization/deserialization
- Enterprise security and compliance settings
- Backward compatibility support
- Hot-reload friendly configuration models
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal, Set
from dataclasses import dataclass, field

# Import Pydantic components - v1 compatible
try:
    from pydantic import BaseModel, BaseSettings, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
    PYDANTIC_V2 = False
except ImportError:
    # Fallback for systems without Pydantic
    BaseModel = object
    BaseSettings = object
    Field = lambda default=None, **kwargs: default
    validator = lambda *args, **kwargs: lambda f: f
    root_validator = lambda *args, **kwargs: lambda f: f
    PYDANTIC_AVAILABLE = False
    PYDANTIC_V2 = False


class LogLevel(str, Enum):
    """Standard logging levels with validation."""
    DEBUG = "DEBUG"
    INFO = "INFO"  
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HandlerType(str, Enum):
    """Supported handler types."""
    CONSOLE = "console"
    FILE = "file"
    ROTATING_FILE = "rotating_file"
    TIMED_ROTATING_FILE = "timed_rotating_file"
    SYSLOG = "syslog"
    HTTP = "http"
    SMTP = "smtp"
    CUSTOM = "custom"


class SinkType(str, Enum):
    """Supported sink types for S-tier logging."""
    CONSOLE = "console"
    TIERED_FILE = "tiered_file"
    METRICS = "metrics"
    NETWORK = "network"
    EMERGENCY = "emergency"
    CUSTOM = "custom"


class FilterType(str, Enum):
    """Supported filter types."""
    LEVEL = "level"
    RATE_LIMITER = "rate_limiter"
    CORRELATION = "correlation"
    SECURITY = "security"
    CUSTOM = "custom"


class FormatterType(str, Enum):
    """Supported formatter types."""
    STRUCTURED = "structured"
    CONSOLE = "console"
    COMPRESSION = "compression"
    CUSTOM = "custom"


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class CompressionType(str, Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms."""
    NONE = "none"
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"


class RetentionPolicy(BaseModel):
    """Log retention policy configuration."""
    
    max_age_days: int = Field(
        default=90,
        ge=1,
        le=2555,  # ~7 years max
        description="Maximum age of logs in days"
    )
    max_size_gb: float = Field(
        default=10.0,
        ge=0.1,
        le=1000.0,
        description="Maximum total size of logs in GB"
    )
    max_files: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum number of log files"
    )
    compression_after_days: int = Field(
        default=7,
        ge=0,
        le=365,
        description="Compress logs after N days (0 = immediate)"
    )
    archive_after_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Archive logs to cold storage after N days"
    )
    delete_after_days: int = Field(
        default=90,
        ge=1,
        le=2555,
        description="Permanently delete logs after N days"
    )
    
    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            frozen=True,
            validate_assignment=True
        )
    else:
        class Config:
            frozen = True
            validate_assignment = True
    
    @validator('archive_after_days')
    def archive_before_delete(cls, v, values):
        """Ensure archival happens before deletion."""
        if 'delete_after_days' in values and v >= values['delete_after_days']:
            raise ValueError("archive_after_days must be less than delete_after_days")
        return v


class EncryptionConfig(BaseModel):
    """Encryption configuration for sensitive data."""
    
    enabled: bool = Field(
        default=True,
        description="Enable encryption for sensitive logs"
    )
    algorithm: EncryptionAlgorithm = Field(
        default=EncryptionAlgorithm.FERNET,
        description="Encryption algorithm to use"
    )
    key_file_path: Optional[Path] = Field(
        default=None,
        description="Path to encryption key file"
    )
    key_rotation_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Rotate encryption keys every N days"
    )
    encrypt_in_transit: bool = Field(
        default=True,
        description="Encrypt logs during network transmission"
    )
    encrypt_at_rest: bool = Field(
        default=True,
        description="Encrypt logs when stored to disk"
    )
    
    @validator('key_file_path')
    def validate_key_file(cls, v):
        """Validate encryption key file path."""
        if v and not v.parent.exists():
            raise ValueError(f"Encryption key directory does not exist: {v.parent}")
        return v


class SecurityConfig(BaseModel):
    """Security configuration for enterprise compliance."""
    
    # PII and sensitive data protection
    enable_pii_detection: bool = Field(
        default=True,
        description="Enable automatic PII detection and redaction"
    )
    pii_redaction_level: Literal["partial", "full", "hash"] = Field(
        default="partial",
        description="Level of PII redaction"
    )
    redact_credentials: bool = Field(
        default=True,
        description="Automatically redact credentials and API keys"
    )
    redact_ip_addresses: bool = Field(
        default=False,
        description="Redact IP addresses from logs"
    )
    
    # Access control
    require_authentication: bool = Field(
        default=True,
        description="Require authentication for log access"
    )
    allowed_users: List[str] = Field(
        default_factory=list,
        description="List of users allowed to access logs"
    )
    allowed_roles: List[str] = Field(
        default_factory=list,
        description="List of roles allowed to access logs"
    )
    
    # Audit and compliance
    enable_audit_trail: bool = Field(
        default=True,
        description="Enable comprehensive audit trail"
    )
    audit_log_path: Optional[Path] = Field(
        default=None,
        description="Path for security audit logs"
    )
    compliance_mode: Literal["none", "gdpr", "hipaa", "sox", "pci_dss"] = Field(
        default="none",
        description="Compliance framework to enforce"
    )
    
    # Encryption settings
    encryption: EncryptionConfig = Field(
        default_factory=EncryptionConfig,
        description="Encryption configuration"
    )
    
    # Data retention
    retention: RetentionPolicy = Field(
        default_factory=RetentionPolicy,
        description="Data retention policy"
    )
    
    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            validate_assignment=True
        )
    else:
        class Config:
            validate_assignment = True


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""
    
    # Async processing
    enable_async_processing: bool = Field(
        default=True,
        description="Enable asynchronous log processing"
    )
    async_queue_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Size of async processing queue"
    )
    worker_threads: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of worker threads"
    )
    
    # Buffering and batching
    buffer_size: int = Field(
        default=8192,
        ge=1024,
        le=1048576,  # 1MB max
        description="Buffer size in bytes"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of messages to batch together"
    )
    flush_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Flush interval in seconds"
    )
    
    # Caching
    enable_caching: bool = Field(
        default=True,
        description="Enable performance caching"
    )
    cache_size_limit: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum cache entries"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=86400,  # 24 hours max
        description="Cache entry TTL in seconds"
    )
    
    # Resource limits
    max_memory_mb: int = Field(
        default=1024,
        ge=64,
        le=16384,  # 16GB max
        description="Maximum memory usage in MB"
    )
    max_disk_usage_gb: float = Field(
        default=10.0,
        ge=0.1,
        le=1000.0,
        description="Maximum disk usage in GB"
    )
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    performance_sampling_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Sampling rate for performance metrics (0.0-1.0)"
    )
    
    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            validate_assignment=True
        )
    else:
        class Config:
            validate_assignment = True


class DiagnosticsConfig(BaseModel):
    """Diagnostics and monitoring configuration."""
    
    # Health checks
    enable_health_checks: bool = Field(
        default=True,
        description="Enable system health monitoring"
    )
    health_check_interval_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Health check interval in seconds"
    )
    health_check_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Health check timeout in seconds"
    )
    
    # Metrics collection
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_export_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Metrics export interval in seconds"
    )
    metrics_retention_hours: int = Field(
        default=24,
        ge=1,
        le=8760,  # 1 year max
        description="Metrics retention period in hours"
    )
    
    # Emergency protocols
    enable_emergency_protocols: bool = Field(
        default=True,
        description="Enable emergency failover protocols"
    )
    emergency_activation_threshold: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="System load threshold for emergency activation"
    )
    emergency_cooldown_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Cooldown period after emergency activation"
    )
    
    # Resource monitoring
    monitor_memory_usage: bool = Field(
        default=True,
        description="Monitor memory usage"
    )
    monitor_disk_usage: bool = Field(
        default=True,
        description="Monitor disk usage"
    )
    monitor_network_usage: bool = Field(
        default=True,
        description="Monitor network usage"
    )
    
    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            validate_assignment=True
        )
    else:
        class Config:
            validate_assignment = True


class FilterConfig(BaseModel):
    """Filter configuration."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Filter name"
    )
    type: FilterType = Field(
        ...,
        description="Filter type"
    )
    enabled: bool = Field(
        default=True,
        description="Whether filter is enabled"
    )
    priority: int = Field(
        default=1,
        ge=0,
        le=100,
        description="Filter priority (higher = more important)"
    )
    
    # Filter-specific configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filter-specific configuration parameters"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate filter name."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Filter name must be alphanumeric (with _ and - allowed)")
        return v


class SinkConfig(BaseModel):
    """Sink configuration."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Sink name"
    )
    type: SinkType = Field(
        ...,
        description="Sink type"
    )
    enabled: bool = Field(
        default=True,
        description="Whether sink is enabled"
    )
    
    # Sink-specific configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sink-specific configuration parameters"
    )
    
    # Filters applied to this sink
    filters: List[str] = Field(
        default_factory=list,
        description="List of filter names to apply"
    )
    
    # Formatter for this sink
    formatter: Optional[str] = Field(
        default=None,
        description="Formatter name to use"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate sink name."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Sink name must be alphanumeric (with _ and - allowed)")
        return v


class HandlerConfig(BaseModel):
    """Handler configuration for backward compatibility."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Handler name"
    )
    type: HandlerType = Field(
        ...,
        description="Handler type"
    )
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level"
    )
    
    # File handler configuration
    filename: Optional[Path] = Field(
        default=None,
        description="Log file path"
    )
    max_bytes: int = Field(
        default=100_000_000,  # 100MB
        ge=1000,
        description="Maximum file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of backup files"
    )
    
    # Console handler configuration
    stream: Optional[str] = Field(
        default=None,
        description="Output stream (stdout/stderr)"
    )
    
    # Network handler configuration
    host: Optional[str] = Field(
        default=None,
        description="Remote host for network handlers"
    )
    port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Remote port for network handlers"
    )
    
    # Additional configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Handler-specific configuration"
    )
    
    @validator('filename')
    def validate_filename(cls, v, values):
        """Validate filename for file handlers."""
        handler_type = values.get('type')
        if handler_type in [HandlerType.FILE, HandlerType.ROTATING_FILE, HandlerType.TIMED_ROTATING_FILE]:
            if not v:
                raise ValueError(f"File handlers require filename")
            if not v.parent.exists():
                v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('host')
    def validate_host(cls, v, values):
        """Validate host for network handlers.""" 
        handler_type = values.get('type')
        if handler_type in [HandlerType.HTTP, HandlerType.SYSLOG] and not v:
            raise ValueError(f"Network handlers require host")
        return v


class LoggerConfig(BaseModel):
    """Logger configuration."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Logger name"
    )
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logger level"
    )
    propagate: bool = Field(
        default=True,
        description="Whether to propagate to parent loggers"
    )
    
    # Handlers (for backward compatibility)
    handlers: List[str] = Field(
        default_factory=list,
        description="List of handler names"
    )
    
    # Sinks (for S-tier logging)
    sinks: List[str] = Field(
        default_factory=list,
        description="List of sink names"
    )
    
    # Filters applied to this logger
    filters: List[str] = Field(
        default_factory=list,
        description="List of filter names"
    )
    
    # Additional configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Logger-specific configuration"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate logger name."""
        if not all(part.replace('_', '').replace('-', '').isalnum() for part in v.split('.')):
            raise ValueError("Logger name parts must be alphanumeric (with _ and - allowed)")
        return v


class LoggingSystemConfig(BaseSettings):
    """Main configuration for the S-tier logging system."""
    
    # Configuration metadata
    version: int = Field(
        default=1,
        ge=1,
        description="Configuration schema version"
    )
    environment: Environment = Field(
        default=Environment.PRODUCTION,
        description="Deployment environment"
    )
    
    # System identification
    system_name: str = Field(
        default="s-tier-logging",
        min_length=1,
        max_length=100,
        description="System identifier"
    )
    instance_id: str = Field(
        default_factory=lambda: f"instance-{os.getpid()}",
        description="Unique instance identifier"
    )
    
    # Core component configurations
    loggers: Dict[str, LoggerConfig] = Field(
        default_factory=dict,
        description="Logger configurations"
    )
    handlers: Dict[str, HandlerConfig] = Field(
        default_factory=dict,
        description="Handler configurations (backward compatibility)"
    )
    sinks: Dict[str, SinkConfig] = Field(
        default_factory=dict,
        description="Sink configurations"
    )
    filters: Dict[str, FilterConfig] = Field(
        default_factory=dict,
        description="Filter configurations"
    )
    
    # System configurations
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )
    diagnostics: DiagnosticsConfig = Field(
        default_factory=DiagnosticsConfig,
        description="Diagnostics configuration"
    )
    
    # Configuration management
    config_file_path: Path = Field(
        default=Path("logging_config.toml"),
        description="Path to configuration file"
    )
    auto_reload: bool = Field(
        default=True,
        description="Enable automatic configuration reloading"
    )
    reload_check_interval: float = Field(
        default=5.0,
        ge=1.0,
        description="Configuration reload check interval in seconds"
    )
    
    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_prefix='LOGGING_',
            env_nested_delimiter='__',
            case_sensitive=False,
            validate_assignment=True,
            extra='forbid'
        )
    else:
        class Config:
            env_prefix = 'LOGGING_'
            env_nested_delimiter = '__'
            case_sensitive = False
            validate_assignment = True
            extra = 'forbid'
    
    @validator('loggers')
    def validate_logger_handlers(cls, v, values):
        """Validate that logger handlers/sinks exist."""
        handlers = values.get('handlers', {})
        sinks = values.get('sinks', {})
        
        for logger_name, logger_config in v.items():
            # Check handlers exist
            for handler_name in logger_config.handlers:
                if handler_name not in handlers:
                    raise ValueError(f"Logger '{logger_name}' references unknown handler '{handler_name}'")
            
            # Check sinks exist
            for sink_name in logger_config.sinks:
                if sink_name not in sinks:
                    raise ValueError(f"Logger '{logger_name}' references unknown sink '{sink_name}'")
                    
        return v
    
    @validator('sinks')
    def validate_sink_filters(cls, v, values):
        """Validate that sink filters exist."""
        filters = values.get('filters', {})
        
        for sink_name, sink_config in v.items():
            for filter_name in sink_config.filters:
                if filter_name not in filters:
                    raise ValueError(f"Sink '{sink_name}' references unknown filter '{filter_name}'")
                    
        return v
    
    @root_validator
    def validate_system_consistency(cls, values):
        """Validate overall system consistency."""
        environment = values.get('environment')
        security = values.get('security')
        performance = values.get('performance')
        
        # Production environment validations
        if environment == Environment.PRODUCTION:
            if security and not security.enable_audit_trail:
                raise ValueError("Production environment requires audit trail")
            if security and not security.encryption.enabled:
                raise ValueError("Production environment requires encryption")
            if performance and not performance.enable_async_processing:
                raise ValueError("Production environment should use async processing")
        
        return values


# Default configuration factory functions
def create_development_config() -> LoggingSystemConfig:
    """Create development environment configuration."""
    return LoggingSystemConfig(
        environment=Environment.DEVELOPMENT,
        security=SecurityConfig(
            enable_pii_detection=False,
            encryption=EncryptionConfig(enabled=False),
            enable_audit_trail=False
        ),
        performance=PerformanceConfig(
            enable_async_processing=False,
            buffer_size=1024,
            worker_threads=2
        ),
        diagnostics=DiagnosticsConfig(
            health_check_interval_seconds=60,
            enable_emergency_protocols=False
        )
    )


def create_production_config() -> LoggingSystemConfig:
    """Create production environment configuration."""
    return LoggingSystemConfig(
        environment=Environment.PRODUCTION,
        security=SecurityConfig(
            enable_pii_detection=True,
            encryption=EncryptionConfig(enabled=True),
            enable_audit_trail=True,
            compliance_mode="gdpr"
        ),
        performance=PerformanceConfig(
            enable_async_processing=True,
            buffer_size=8192,
            worker_threads=8,
            max_memory_mb=2048
        ),
        diagnostics=DiagnosticsConfig(
            health_check_interval_seconds=10,
            enable_emergency_protocols=True
        )
    )


def create_testing_config() -> LoggingSystemConfig:
    """Create testing environment configuration."""
    return LoggingSystemConfig(
        environment=Environment.TESTING,
        security=SecurityConfig(
            enable_pii_detection=True,
            encryption=EncryptionConfig(enabled=False),
            enable_audit_trail=True
        ),
        performance=PerformanceConfig(
            enable_async_processing=True,
            buffer_size=4096,
            worker_threads=4
        ),
        diagnostics=DiagnosticsConfig(
            health_check_interval_seconds=30,
            enable_emergency_protocols=True
        )
    )


# Module exports
__all__ = [
    'LogLevel',
    'HandlerType', 
    'SinkType',
    'FilterType',
    'FormatterType',
    'Environment',
    'CompressionType',
    'EncryptionAlgorithm',
    'RetentionPolicy',
    'EncryptionConfig',
    'SecurityConfig',
    'PerformanceConfig',
    'DiagnosticsConfig',
    'FilterConfig',
    'SinkConfig',
    'HandlerConfig',
    'LoggerConfig',
    'LoggingSystemConfig',
    'create_development_config',
    'create_production_config',
    'create_testing_config'
]