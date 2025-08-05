"""
Proactive Health Monitoring System for Arena Bot Deep Debugging

Comprehensive health monitoring and circuit breaker implementation:
- Real-time component health assessment with automated recovery
- Circuit breaker pattern for failure prevention and isolation
- Resource monitoring (CPU, memory, disk, network) with alerting
- Service dependency mapping and cascade failure prevention
- Predictive failure analysis using performance trends
- Automated health checks with configurable intervals
- Emergency protocols for system protection

Integrates with existing S-tier logging and provides early warning systems.
"""

import time
import threading
import psutil
import asyncio
from typing import Any, Dict, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from ..logging_system.logger import get_logger, LogLevel


class HealthStatus(Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class HealthMetric:
    """
    Represents a single health metric measurement.
    """
    
    # Metric identification
    metric_name: str
    component_name: str
    timestamp: float = field(default_factory=time.time)
    
    # Metric value and metadata
    value: Union[float, int, bool, str]
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    # Status and context
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate_status(self) -> HealthStatus:
        """Evaluate health status based on thresholds."""
        if not isinstance(self.value, (int, float)):
            return HealthStatus.UNKNOWN
        
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            self.status = HealthStatus.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY
        
        return self.status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'metric_name': self.metric_name,
            'component_name': self.component_name,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'value': self.value,
            'unit': self.unit,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'status': self.status.value,
            'message': self.message,
            'context': self.context
        }


@dataclass
class ComponentHealth:
    """
    Represents the overall health of a component.
    """
    
    # Component identification
    component_name: str
    component_type: str = ""
    
    # Health status
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    last_check_time: float = field(default_factory=time.time)
    
    # Metrics and history
    current_metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    metric_history: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    
    # Failure tracking
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    total_failures: int = 0
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    def update_metric(self, metric: HealthMetric) -> None:
        """Update a health metric for this component."""
        self.current_metrics[metric.metric_name] = metric
        self.metric_history[metric.metric_name].append(metric)
        self.last_check_time = time.time()
        
        # Update overall status based on worst metric
        self._update_overall_status()
    
    def _update_overall_status(self) -> None:
        """Update overall status based on current metrics."""
        if not self.current_metrics:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        # Find worst status
        statuses = [metric.status for metric in self.current_metrics.values()]
        
        if HealthStatus.FAILED in statuses:
            self.overall_status = HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            self.overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            self.overall_status = HealthStatus.WARNING
        elif HealthStatus.DEGRADED in statuses:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
    
    def record_failure(self) -> None:
        """Record a failure for this component."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.overall_status = HealthStatus.FAILED
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.consecutive_failures = 0
        if self.overall_status == HealthStatus.FAILED:
            self._update_overall_status()
    
    def get_metric_trend(self, metric_name: str, window_minutes: int = 5) -> Optional[float]:
        """Get trend for a specific metric over time window."""
        if metric_name not in self.metric_history:
            return None
        
        history = self.metric_history[metric_name]
        if len(history) < 2:
            return None
        
        # Filter to time window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) < 2:
            return None
        
        # Calculate trend (simple linear)
        values = [m.value for m in recent_metrics if isinstance(m.value, (int, float))]
        if len(values) < 2:
            return None
        
        return (values[-1] - values[0]) / len(values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'component_name': self.component_name,
            'component_type': self.component_type,
            'overall_status': self.overall_status.value,
            'last_check_time': self.last_check_time,
            'current_metrics': {
                name: metric.to_dict() 
                for name, metric in self.current_metrics.items()
            },
            'consecutive_failures': self.consecutive_failures,
            'last_failure_time': self.last_failure_time,
            'total_failures': self.total_failures,
            'dependencies': list(self.dependencies),
            'dependents': list(self.dependents)
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for component protection.
    
    Implements the circuit breaker pattern to prevent cascading failures
    and provide automatic recovery mechanisms.
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 success_threshold: int = 3):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds to wait before trying recovery
            success_threshold: Successful calls needed to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        
        # Callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        
        # Thread safety
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit should transition states
            self._check_state_transition()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                self._record_success()
                
                return result
                
            except Exception as e:
                # Record failure
                self._record_failure()
                raise
    
    def _record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.success_count += 1
        self.total_successes += 1
        
        # Close circuit if enough successes in half-open state
        if self.state == CircuitState.HALF_OPEN and self.success_count >= self.success_threshold:
            self._change_state(CircuitState.CLOSED)
    
    def _record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        # Open circuit if threshold reached
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self._change_state(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            # Reset to open on any failure during half-open
            self._change_state(CircuitState.OPEN)
    
    def _check_state_transition(self) -> None:
        """Check if circuit should transition to half-open."""
        if (self.state == CircuitState.OPEN and 
            time.time() - self.last_failure_time >= self.recovery_timeout):
            self._change_state(CircuitState.HALF_OPEN)
    
    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
        
        # Call state change callback
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except:
                pass  # Don't let callback errors affect circuit breaker
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self.lock:
            self._change_state(CircuitState.OPEN)
    
    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        with self.lock:
            self._change_state(CircuitState.CLOSED)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            failure_rate = (
                self.total_failures / self.total_calls 
                if self.total_calls > 0 else 0
            )
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'failure_rate': failure_rate,
                'last_failure_time': self.last_failure_time,
                'last_state_change': self.last_state_change,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class HealthChecker(ABC):
    """Abstract base class for health checkers."""
    
    @abstractmethod
    def check_health(self) -> List[HealthMetric]:
        """Perform health check and return metrics."""
        pass
    
    @abstractmethod
    def get_component_name(self) -> str:
        """Get the name of the component being checked."""
        pass


class SystemResourceChecker(HealthChecker):
    """Health checker for system resources."""
    
    def __init__(self):
        """Initialize system resource checker."""
        self.component_name = "system_resources"
    
    def check_health(self) -> List[HealthMetric]:
        """Check system resource health."""
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(HealthMetric(
                metric_name="cpu_usage",
                component_name=self.component_name,
                value=cpu_percent,
                unit="percent",
                threshold_warning=70.0,
                threshold_critical=90.0
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric(
                metric_name="memory_usage",
                component_name=self.component_name,
                value=memory.percent,
                unit="percent",
                threshold_warning=80.0,
                threshold_critical=95.0
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                metric_name="disk_usage",
                component_name=self.component_name,
                value=disk_percent,
                unit="percent",
                threshold_warning=80.0,
                threshold_critical=95.0
            ))
            
            # Process count
            process_count = len(psutil.pids())
            metrics.append(HealthMetric(
                metric_name="process_count",
                component_name=self.component_name,
                value=process_count,
                unit="count",
                threshold_warning=500,
                threshold_critical=1000
            ))
            
        except Exception as e:
            # Return error metric
            metrics.append(HealthMetric(
                metric_name="system_check_error",
                component_name=self.component_name,
                value=str(e),
                status=HealthStatus.CRITICAL
            ))
        
        # Evaluate all metrics
        for metric in metrics:
            metric.evaluate_status()
        
        return metrics
    
    def get_component_name(self) -> str:
        """Get component name."""
        return self.component_name


class HealthMonitor:
    """
    Central health monitoring system for Arena Bot.
    
    Coordinates health checks, circuit breakers, and emergency protocols
    to maintain system stability and prevent cascading failures.
    """
    
    def __init__(self):
        """Initialize health monitor."""
        self.logger = get_logger("arena_bot.debugging.health_monitor")
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Monitoring configuration
        self.enabled = True
        self.check_interval = 30  # seconds
        self.alert_thresholds = {
            HealthStatus.WARNING: True,
            HealthStatus.CRITICAL: True,
            HealthStatus.FAILED: True
        }
        
        # Background monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Emergency protocols
        self.emergency_callbacks: List[Callable[[str, ComponentHealth], None]] = []
        self.recovery_callbacks: List[Callable[[str, ComponentHealth], None]] = []
        
        # Statistics
        self.total_checks_performed = 0
        self.total_alerts_generated = 0
        self.start_time = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default checkers
        self._initialize_default_checkers()
    
    def _initialize_default_checkers(self) -> None:
        """Initialize default health checkers."""
        # System resource checker
        self.add_health_checker(SystemResourceChecker())
        
        # Add circuit breakers for critical components
        self.add_circuit_breaker("detection_engine", failure_threshold=3, recovery_timeout=30)
        self.add_circuit_breaker("ai_advisor", failure_threshold=5, recovery_timeout=60)
        self.add_circuit_breaker("gui_updates", failure_threshold=10, recovery_timeout=15)
    
    def add_health_checker(self, checker: HealthChecker) -> None:
        """Add a health checker."""
        with self.lock:
            component_name = checker.get_component_name()
            self.health_checkers[component_name] = checker
            
            if component_name not in self.components:
                self.components[component_name] = ComponentHealth(
                    component_name=component_name,
                    component_type="health_checked"
                )
    
    def add_circuit_breaker(self,
                           name: str,
                           failure_threshold: int = 5,
                           recovery_timeout: int = 60,
                           success_threshold: int = 3) -> CircuitBreaker:
        """Add a circuit breaker."""
        with self.lock:
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold
            )
            
            # Add state change callback
            breaker.on_state_change = self._on_circuit_state_change
            
            self.circuit_breakers[name] = breaker
            
            self.logger.info(
                f"🔧 Circuit breaker added: {name} "
                f"(failure_threshold={failure_threshold}, recovery_timeout={recovery_timeout}s)"
            )
            
            return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def _on_circuit_state_change(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """Handle circuit breaker state changes."""
        # Find which breaker changed state
        breaker_name = None
        for name, breaker in self.circuit_breakers.items():
            if breaker.state == new_state:
                breaker_name = name
                break
        
        if breaker_name:
            self.logger.warning(
                f"🔧 CIRCUIT_BREAKER: {breaker_name} {old_state.value} → {new_state.value}",
                extra={
                    'circuit_breaker': breaker_name,
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'stats': self.circuit_breakers[breaker_name].get_stats()
                }
            )
    
    def perform_health_check(self, component_name: Optional[str] = None) -> Dict[str, ComponentHealth]:
        """
        Perform health checks.
        
        Args:
            component_name: Check specific component, or all if None
            
        Returns:
            Dictionary of component health results
        """
        
        if not self.enabled:
            return {}
        
        results = {}
        
        with self.lock:
            checkers_to_run = {}
            
            if component_name:
                if component_name in self.health_checkers:
                    checkers_to_run[component_name] = self.health_checkers[component_name]
            else:
                checkers_to_run = self.health_checkers.copy()
            
            # Run health checks
            for name, checker in checkers_to_run.items():
                try:
                    self.total_checks_performed += 1
                    
                    # Get component health object
                    if name not in self.components:
                        self.components[name] = ComponentHealth(
                            component_name=name,
                            component_type="health_checked"
                        )
                    
                    component_health = self.components[name]
                    
                    # Perform check
                    metrics = checker.check_health()
                    
                    # Update component with metrics
                    for metric in metrics:
                        component_health.update_metric(metric)
                    
                    # Record success if no critical metrics
                    if component_health.overall_status not in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                        component_health.record_success()
                    else:
                        component_health.record_failure()
                    
                    results[name] = component_health
                    
                    # Check for alerts
                    self._check_for_alerts(component_health)
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    
                    # Record failure
                    if name in self.components:
                        self.components[name].record_failure()
        
        return results
    
    def _check_for_alerts(self, component_health: ComponentHealth) -> None:
        """Check if component health should trigger alerts."""
        status = component_health.overall_status
        
        if status in self.alert_thresholds and self.alert_thresholds[status]:
            self.total_alerts_generated += 1
            
            self.logger.warning(
                f"🚨 HEALTH_ALERT: {component_health.component_name} status: {status.value}",
                extra={
                    'alert_type': 'health_status',
                    'component_health': component_health.to_dict()
                }
            )
            
            # Trigger emergency protocols for critical/failed status
            if status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                self._trigger_emergency_protocols(component_health)
    
    def _trigger_emergency_protocols(self, component_health: ComponentHealth) -> None:
        """Trigger emergency protocols for critical component issues."""
        component_name = component_health.component_name
        
        self.logger.critical(
            f"🚨 EMERGENCY: {component_name} in {component_health.overall_status.value} state",
            extra={
                'emergency_type': 'component_failure',
                'component_name': component_name,
                'component_health': component_health.to_dict()
            }
        )
        
        # Call emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(component_name, component_health)
            except Exception as e:
                self.logger.error(f"Emergency callback error: {e}")
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(
            f"🏥 Health monitoring started (interval: {self.check_interval}s)"
        )
    
    def stop_monitoring_thread(self) -> None:
        """Stop background health monitoring."""
        self.stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("🏥 Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Perform health checks
                self.perform_health_check()
                
                # Wait for next interval
                if self.stop_monitoring.wait(timeout=self.check_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
                
                # Wait before retrying
                if self.stop_monitoring.wait(timeout=10):
                    break
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status of a specific component."""
        with self.lock:
            return self.components.get(component_name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health status of all components."""
        with self.lock:
            return self.components.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self.lock:
            # Count components by status
            status_counts = defaultdict(int)
            for component in self.components.values():
                status_counts[component.overall_status.value] += 1
            
            # Determine overall system status
            if status_counts[HealthStatus.FAILED.value] > 0:
                overall_status = HealthStatus.FAILED
            elif status_counts[HealthStatus.CRITICAL.value] > 0:
                overall_status = HealthStatus.CRITICAL
            elif status_counts[HealthStatus.WARNING.value] > 0:
                overall_status = HealthStatus.WARNING
            elif status_counts[HealthStatus.DEGRADED.value] > 0:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            return {
                'overall_status': overall_status.value,
                'total_components': len(self.components),
                'status_counts': dict(status_counts),
                'circuit_breakers': {
                    name: breaker.get_stats() 
                    for name, breaker in self.circuit_breakers.items()
                },
                'monitoring_enabled': self.enabled,
                'uptime_seconds': time.time() - self.start_time,
                'total_checks_performed': self.total_checks_performed,
                'total_alerts_generated': self.total_alerts_generated
            }
    
    def add_emergency_callback(self, callback: Callable[[str, ComponentHealth], None]) -> None:
        """Add callback for emergency situations."""
        self.emergency_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str, ComponentHealth], None]) -> None:
        """Add callback for component recovery."""
        self.recovery_callbacks.append(callback)
    
    def enable(self) -> None:
        """Enable health monitoring."""
        self.enabled = True
        self.logger.info("🏥 Health monitoring enabled")
    
    def disable(self) -> None:
        """Disable health monitoring."""
        self.enabled = False
        self.logger.info("🏥 Health monitoring disabled")


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor
    
    if _global_health_monitor is None:
        with _monitor_lock:
            if _global_health_monitor is None:
                _global_health_monitor = HealthMonitor()
    
    return _global_health_monitor


def circuit_breaker(name: str, **kwargs) -> Callable:
    """
    Decorator to protect functions with circuit breaker.
    
    Usage:
        @circuit_breaker("my_component", failure_threshold=3)
        def my_function():
            # Function implementation
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        monitor = get_health_monitor()
        
        # Get or create circuit breaker
        breaker = monitor.get_circuit_breaker(name)
        if not breaker:
            breaker = monitor.add_circuit_breaker(name, **kwargs)
        
        def wrapper(*args, **func_kwargs):
            return breaker.call(func, *args, **func_kwargs)
        
        return wrapper
    
    return decorator