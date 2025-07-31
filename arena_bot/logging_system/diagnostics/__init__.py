"""
Diagnostics and health monitoring for the S-Tier Logging System.

This module provides comprehensive health monitoring, performance profiling,
and emergency recovery protocols to ensure the logging system remains
reliable and does not become a bottleneck.

Components:
- HealthChecker: System health monitoring and reporting
- PerformanceProfiler: Detailed performance analysis and optimization
- EmergencyProtocols: Failure recovery and degraded mode handling
"""

from .health_checker import HealthChecker, health_check
from .performance_profiler import PerformanceProfiler
from .emergency_protocols import EmergencyProtocols

__all__ = [
    'HealthChecker',
    'health_check',
    'PerformanceProfiler',
    'EmergencyProtocols'
]