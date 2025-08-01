"""
Diagnostics Module for S-Tier Logging System.

This module provides comprehensive diagnostics, monitoring, and emergency
response capabilities for the S-tier logging system including health checks,
performance profiling, and automated emergency protocols.

Components:
- HealthChecker: System health monitoring and validation
- PerformanceProfiler: Real-time performance metrics and analysis
- EmergencyProtocol: Automated emergency detection and response

Features:
- Real-time health monitoring with automated recovery
- Performance profiling with bottleneck detection
- Emergency response with circuit breaker protection
- Comprehensive metrics collection and analysis
- Automated system protection and graceful degradation
"""

# Import health checking components
from .health_checker import (
    HealthStatus,
    HealthCheckType,
    HealthCheckResult,
    HealthCheck,
    ComponentHealthCheck,
    PerformanceHealthCheck,
    ResourceHealthCheck,
    SystemHealthReport,
    HealthChecker,
    health_check
)

# Import performance profiling components
from .performance_profiler import (
    MetricType,
    PerformanceLevel,
    MetricValue,
    PerformanceStats,
    PerformanceMetric,
    PerformanceProfiler,
    TimingContext
)

# Import emergency protocol components
from .emergency_protocols import (
    EmergencyLevel,
    EmergencyType,
    CircuitState,
    EmergencyEvent,
    EmergencyCondition,
    ResourceExhaustionCondition,
    PerformanceDegradationCondition,
    ComponentFailureCondition,
    CircuitBreaker,
    EmergencyLogger,
    EmergencyProtocol
)

# Module exports
__all__ = [
    # Health checking
    'HealthStatus',
    'HealthCheckType',
    'HealthCheckResult',
    'HealthCheck',
    'ComponentHealthCheck',
    'PerformanceHealthCheck',
    'ResourceHealthCheck',
    'SystemHealthReport',
    'HealthChecker',
    'health_check',
    
    # Performance profiling
    'MetricType',
    'PerformanceLevel',
    'MetricValue',
    'PerformanceStats',
    'PerformanceMetric',
    'PerformanceProfiler',
    'TimingContext',
    
    # Emergency protocols
    'EmergencyLevel',
    'EmergencyType',
    'CircuitState',
    'EmergencyEvent',
    'EmergencyCondition',
    'ResourceExhaustionCondition',
    'PerformanceDegradationCondition',
    'ComponentFailureCondition',
    'CircuitBreaker',
    'EmergencyLogger',
    'EmergencyProtocol'
]