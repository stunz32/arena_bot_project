"""
Sinks for the S-Tier Logging System.

This module contains output destinations (sinks) that handle the actual
writing of log messages to various targets including console, files,
metrics databases, network endpoints, and emergency fallbacks.

Sinks:
- BaseSink: Abstract base with common functionality and error handling
- ConsoleSink: Human-readable console output with colors and formatting
- TieredFileSink: Intelligent file logging with rotation and compression
- MetricsSink: Log-to-metrics conversion for monitoring systems
- NetworkSink: High-performance network transmission with reliability
- EmergencySink: Ultra-reliable emergency logging when everything fails
"""

# Import all sink classes and related types
from .base_sink import BaseSink, SinkState, ErrorHandlingStrategy, SinkStats
from .sink_manager import SinkManager, SinkManagerStats
from .console_sink import ConsoleSink
from .tiered_file_sink import (
    TieredFileSink,
    RotationStrategy,
    CompressionType,
    StorageTier,
    TierConfig,
    RotationConfig
)
from .metrics_sink import (
    MetricsSink,
    MetricRule,
    MetricValue,
    MetricType,
    MetricDestination
)
from .network_sink import (
    NetworkSink,
    NetworkEndpoint,
    NetworkConfig,
    NetworkProtocol,
    CompressionType as NetworkCompressionType,
    AuthenticationType
)
from .emergency_sink import (
    EmergencySink,
    EmergencyConfig,
    EmergencyDestination,
    EmergencyTrigger,
    MemoryBuffer
)

# Module exports
__all__ = [
    # Base sink components
    'BaseSink',
    'SinkState', 
    'ErrorHandlingStrategy',
    'SinkStats',
    'SinkManager',
    'SinkManagerStats',
    
    # Console sink
    'ConsoleSink',
    
    # File sink components
    'TieredFileSink',
    'RotationStrategy',
    'CompressionType',
    'StorageTier', 
    'TierConfig',
    'RotationConfig',
    
    # Metrics sink components
    'MetricsSink',
    'MetricRule',
    'MetricValue',
    'MetricType',
    'MetricDestination',
    
    # Network sink components
    'NetworkSink',
    'NetworkEndpoint',
    'NetworkConfig',
    'NetworkProtocol',
    'NetworkCompressionType',
    'AuthenticationType',
    
    # Emergency sink components
    'EmergencySink',
    'EmergencyConfig',
    'EmergencyDestination',
    'EmergencyTrigger',
    'MemoryBuffer'
]