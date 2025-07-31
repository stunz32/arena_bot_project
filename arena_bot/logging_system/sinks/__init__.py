"""
Sinks for the S-Tier Logging System.

This module contains output destinations (sinks) that handle the actual
writing of log messages to various targets including console, files,
metrics databases, network endpoints, and emergency fallbacks.

Sinks:
- BaseSink: Abstract base with common functionality
- ConsoleSink: Thread-safe console output
- TieredFileSink: Level-based file routing with rotation
- MetricsSink: Integration with performance monitoring
- NetworkSink: Remote logging with retry logic
- EmergencySink: Failure fallback destination
"""

from .base_sink import BaseSink
from .console_sink import ConsoleSink
from .tiered_file_sink import TieredFileSink
from .metrics_sink import MetricsSink
from .network_sink import NetworkSink
from .emergency_sink import EmergencySink

__all__ = [
    'BaseSink',
    'ConsoleSink',
    'TieredFileSink',
    'MetricsSink',
    'NetworkSink',
    'EmergencySink'
]