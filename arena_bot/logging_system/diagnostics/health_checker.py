"""
Health Checker for S-Tier Logging System.

This module provides comprehensive health monitoring for the logging system
including component status checks, performance metrics, resource utilization,
and automated recovery procedures.

Features:
- Real-time component health monitoring
- Performance threshold validation
- Resource utilization tracking
- Automated health recovery procedures
- Integration with emergency protocols
- Comprehensive health reporting
"""

import asyncio
import time
import psutil
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class HealthStatus(str, Enum):
    """Health status levels for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheckType(str, Enum):
    """Types of health checks."""
    COMPONENT = "component"       # Individual component health
    PERFORMANCE = "performance"   # Performance metrics check
    RESOURCE = "resource"         # Resource utilization check
    CONNECTIVITY = "connectivity" # Network/external connectivity
    INTEGRITY = "integrity"       # Data integrity check


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    
    check_name: str
    check_type: HealthCheckType
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    
    # Detailed information
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Threshold information
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    current_value: Optional[float] = None
    
    # Recovery information
    is_recoverable: bool = True
    recovery_action: Optional[str] = None
    recovery_attempted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "check_name": self.check_name,
            "check_type": self.check_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "message": self.message,
            "details": self.details,
            "metrics": self.metrics,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "current_value": self.current_value,
            "is_recoverable": self.is_recoverable,
            "recovery_action": self.recovery_action,
            "recovery_attempted": self.recovery_attempted
        }
    
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status == HealthStatus.HEALTHY
    
    def needs_attention(self) -> bool:
        """Check if result needs attention."""
        return self.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]


class HealthCheck(ABC):
    """
    Abstract base class for health checks.
    
    Provides interface for implementing specific health check logic
    with standardized result reporting and recovery procedures.
    """
    
    def __init__(self, 
                 name: str,
                 check_type: HealthCheckType,
                 interval_seconds: float = 30.0,
                 timeout_seconds: float = 5.0,
                 enabled: bool = True):
        """
        Initialize health check.
        
        Args:
            name: Unique name for the health check
            check_type: Type of health check
            interval_seconds: How often to run the check
            timeout_seconds: Maximum time allowed for check
            enabled: Whether the check is enabled
        """
        self.name = name
        self.check_type = check_type
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        
        # State tracking
        self.last_result: Optional[HealthCheckResult] = None
        self.last_run_time: Optional[float] = None
        self.consecutive_failures = 0
        self.total_runs = 0
        self.total_failures = 0
    
    @abstractmethod
    async def execute_check(self) -> HealthCheckResult:
        """
        Execute the health check logic.
        
        Returns:
            HealthCheckResult with status and details
        """
        pass
    
    async def run_check(self) -> HealthCheckResult:
        """
        Run the health check with timeout and error handling.
        
        Returns:
            HealthCheckResult with execution details
        """
        if not self.enabled:
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=HealthStatus.UNKNOWN,
                message="Health check is disabled"
            )
        
        start_time = time.time()
        self.total_runs += 1
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self.execute_check(),
                timeout=self.timeout_seconds
            )
            
            # Update execution time
            execution_time = time.time() - start_time
            result.metrics["execution_time_seconds"] = execution_time
            
            # Update state tracking
            if result.is_healthy():
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.total_failures += 1
            
            self.last_result = result
            self.last_run_time = start_time
            
            return result
            
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            self.total_failures += 1
            
            result = HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout_seconds}s"
            )
            
            self.last_result = result
            self.last_run_time = start_time
            
            return result
            
        except Exception as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            
            result = HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed with exception: {str(e)}",
                details={"exception_type": type(e).__name__}
            )
            
            self.last_result = result
            self.last_run_time = start_time
            
            return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get health check statistics."""
        return {
            "name": self.name,
            "type": self.check_type.value,
            "enabled": self.enabled,
            "total_runs": self.total_runs,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": (
                (self.total_runs - self.total_failures) / self.total_runs
                if self.total_runs > 0 else 0
            ),
            "last_run_time": self.last_run_time,
            "last_status": self.last_result.status.value if self.last_result else None
        }


class ComponentHealthCheck(HealthCheck):
    """Health check for logging system components."""
    
    def __init__(self, 
                 component_name: str,
                 component_checker: Callable,
                 **kwargs):
        """
        Initialize component health check.
        
        Args:
            component_name: Name of component to check
            component_checker: Function that returns component status
        """
        super().__init__(
            name=f"component_{component_name}",
            check_type=HealthCheckType.COMPONENT,
            **kwargs
        )
        self.component_name = component_name
        self.component_checker = component_checker
    
    async def execute_check(self) -> HealthCheckResult:
        """Execute component health check."""
        try:
            # Call component checker
            if asyncio.iscoroutinefunction(self.component_checker):
                status_info = await self.component_checker()
            else:
                status_info = self.component_checker()
            
            # Interpret results
            if isinstance(status_info, bool):
                status = HealthStatus.HEALTHY if status_info else HealthStatus.CRITICAL
                message = f"Component {self.component_name} is {'healthy' if status_info else 'failed'}"
                details = {}
            elif isinstance(status_info, dict):
                # Detailed status information
                is_healthy = status_info.get('healthy', True)
                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.CRITICAL
                message = status_info.get('message', f"Component {self.component_name} status")
                details = {k: v for k, v in status_info.items() if k not in ['healthy', 'message']}
            else:
                status = HealthStatus.UNKNOWN
                message = f"Unknown status format from {self.component_name}"
                details = {"raw_status": str(status_info)}
            
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=HealthStatus.CRITICAL,
                message=f"Component check failed: {str(e)}",
                details={"exception": str(e), "component": self.component_name}
            )


class PerformanceHealthCheck(HealthCheck):
    """Health check for performance metrics."""
    
    def __init__(self,
                 metric_name: str,
                 metric_getter: Callable,
                 warning_threshold: float,
                 critical_threshold: float,
                 higher_is_better: bool = False,
                 **kwargs):
        """
        Initialize performance health check.
        
        Args:
            metric_name: Name of the performance metric
            metric_getter: Function that returns current metric value
            warning_threshold: Threshold for warning status
            critical_threshold: Threshold for critical status
            higher_is_better: Whether higher values are better
        """
        super().__init__(
            name=f"performance_{metric_name}",
            check_type=HealthCheckType.PERFORMANCE,
            **kwargs
        )
        self.metric_name = metric_name
        self.metric_getter = metric_getter
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.higher_is_better = higher_is_better
    
    async def execute_check(self) -> HealthCheckResult:
        """Execute performance health check."""
        try:
            # Get current metric value
            if asyncio.iscoroutinefunction(self.metric_getter):
                current_value = await self.metric_getter()
            else:
                current_value = self.metric_getter()
            
            # Convert to float if needed
            if not isinstance(current_value, (int, float)):
                raise ValueError(f"Metric value must be numeric, got {type(current_value)}")
            
            current_value = float(current_value)
            
            # Determine status based on thresholds
            if self.higher_is_better:
                if current_value >= self.warning_threshold:
                    status = HealthStatus.HEALTHY
                    message = f"{self.metric_name} is healthy: {current_value}"
                elif current_value >= self.critical_threshold:
                    status = HealthStatus.DEGRADED
                    message = f"{self.metric_name} is below warning threshold: {current_value}"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"{self.metric_name} is below critical threshold: {current_value}"
            else:
                if current_value <= self.warning_threshold:
                    status = HealthStatus.HEALTHY
                    message = f"{self.metric_name} is healthy: {current_value}"
                elif current_value <= self.critical_threshold:
                    status = HealthStatus.DEGRADED
                    message = f"{self.metric_name} is above warning threshold: {current_value}"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"{self.metric_name} is above critical threshold: {current_value}"
            
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=status,
                message=message,
                current_value=current_value,
                warning_threshold=self.warning_threshold,
                critical_threshold=self.critical_threshold,
                metrics={self.metric_name: current_value}
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=HealthStatus.CRITICAL,
                message=f"Performance check failed: {str(e)}",
                details={"exception": str(e), "metric": self.metric_name}
            )


class ResourceHealthCheck(HealthCheck):
    """Health check for resource utilization."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="resource_utilization",
            check_type=HealthCheckType.RESOURCE,
            **kwargs
        )
    
    async def execute_check(self) -> HealthCheckResult:
        """Execute resource utilization check."""
        try:
            # Get system resource information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Collect metrics
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            # Determine overall status
            issues = []
            status = HealthStatus.HEALTHY
            
            if cpu_percent > 90:
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
                status = HealthStatus.CRITICAL
            elif cpu_percent > 80:
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
            
            if memory.percent > 90:
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
                status = HealthStatus.CRITICAL
            elif memory.percent > 80:
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
            
            if disk.percent > 95:
                issues.append(f"Disk usage critical: {disk.percent:.1f}%")
                status = HealthStatus.CRITICAL
            elif disk.percent > 85:
                issues.append(f"Disk usage high: {disk.percent:.1f}%")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
            
            # Create result
            if issues:
                message = "Resource utilization issues: " + "; ".join(issues)
                recovery_action = "Consider scaling resources or optimizing usage"
            else:
                message = "Resource utilization is healthy"
                recovery_action = None
            
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=status,
                message=message,
                metrics=metrics,
                recovery_action=recovery_action,
                details={
                    "cpu_cores": psutil.cpu_count(),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                status=HealthStatus.CRITICAL,
                message=f"Resource check failed: {str(e)}",
                details={"exception": str(e)}
            )


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    
    timestamp: float = field(default_factory=time.time)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    check_results: List[HealthCheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "overall_status": self.overall_status.value,
            "check_results": [result.to_dict() for result in self.check_results],
            "summary": self.summary,
            "recommendations": self.recommendations
        }
    
    def get_status_counts(self) -> Dict[str, int]:
        """Get counts of each status type."""
        counts = {status.value: 0 for status in HealthStatus}
        for result in self.check_results:
            counts[result.status.value] += 1
        return counts
    
    def get_failed_checks(self) -> List[HealthCheckResult]:
        """Get all failed health checks."""
        return [
            result for result in self.check_results
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        ]
    
    def get_degraded_checks(self) -> List[HealthCheckResult]:
        """Get all degraded health checks."""
        return [
            result for result in self.check_results
            if result.status == HealthStatus.DEGRADED
        ]


class HealthChecker:
    """
    Main health checker for the S-tier logging system.
    
    Manages multiple health checks, generates comprehensive reports,
    and provides automated recovery procedures.
    """
    
    def __init__(self, 
                 logger_manager: 'LoggerManager',
                 enable_auto_recovery: bool = True,
                 check_interval: float = 30.0):
        """
        Initialize health checker.
        
        Args:
            logger_manager: Logger manager instance to monitor
            enable_auto_recovery: Enable automatic recovery procedures
            check_interval: Default interval between health checks
        """
        self.logger_manager = logger_manager
        self.enable_auto_recovery = enable_auto_recovery
        self.check_interval = check_interval
        
        # Health checks registry
        self.health_checks: List[HealthCheck] = []
        self._checks_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.total_checks_run = 0
        self.total_failures = 0
        self.last_report: Optional[SystemHealthReport] = None
        self.start_time = time.time()
        
        # Setup default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self) -> None:
        """Setup default health checks for the logging system."""
        # Component health checks
        if self.logger_manager.queue:
            self.add_health_check(ComponentHealthCheck(
                "async_queue",
                lambda: self.logger_manager.queue.is_healthy() if self.logger_manager.queue else False
            ))
        
        if self.logger_manager.worker_pool:
            self.add_health_check(ComponentHealthCheck(
                "worker_pool",
                lambda: self.logger_manager.worker_pool.is_healthy() if self.logger_manager.worker_pool else False
            ))
        
        if self.logger_manager.sink_manager:
            self.add_health_check(ComponentHealthCheck(
                "sink_manager",
                lambda: self.logger_manager.sink_manager.is_healthy() if self.logger_manager.sink_manager else False
            ))
        
        # Performance health checks
        self.add_health_check(PerformanceHealthCheck(
            "logs_per_second",
            self._get_logs_per_second,
            warning_threshold=100,  # Warning if < 100 logs/sec when processing
            critical_threshold=10,  # Critical if < 10 logs/sec
            higher_is_better=True
        ))
        
        self.add_health_check(PerformanceHealthCheck(
            "error_rate",
            self._get_error_rate,
            warning_threshold=0.05,  # Warning if > 5% error rate
            critical_threshold=0.15,  # Critical if > 15% error rate
            higher_is_better=False
        ))
        
        # Resource health check
        self.add_health_check(ResourceHealthCheck())
    
    def _get_logs_per_second(self) -> float:
        """Get current logs per second rate."""
        stats = self.logger_manager.get_performance_stats()
        return stats.get('logs_per_second', 0.0)
    
    def _get_error_rate(self) -> float:
        """Get current error rate."""
        stats = self.logger_manager.get_performance_stats()
        return stats.get('error_rate', 0.0)
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check to the monitoring system."""
        with self._checks_lock:
            self.health_checks.append(health_check)
    
    def remove_health_check(self, check_name: str) -> bool:
        """Remove a health check by name."""
        with self._checks_lock:
            initial_count = len(self.health_checks)
            self.health_checks = [
                check for check in self.health_checks 
                if check.name != check_name
            ]
            return len(self.health_checks) < initial_count
    
    def enable_health_check(self, check_name: str) -> bool:
        """Enable a health check by name."""
        with self._checks_lock:
            for check in self.health_checks:
                if check.name == check_name:
                    check.enabled = True
                    return True
            return False
    
    def disable_health_check(self, check_name: str) -> bool:
        """Disable a health check by name."""
        with self._checks_lock:
            for check in self.health_checks:
                if check.name == check_name:
                    check.enabled = False
                    return True
            return False
    
    async def run_all_checks(self) -> SystemHealthReport:
        """Run all enabled health checks and generate report."""
        check_results = []
        
        # Run all enabled checks
        with self._checks_lock:
            enabled_checks = [check for check in self.health_checks if check.enabled]
        
        # Execute checks concurrently
        if enabled_checks:
            results = await asyncio.gather(
                *[check.run_check() for check in enabled_checks],
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, HealthCheckResult):
                    check_results.append(result)
                    self.total_checks_run += 1
                    if not result.is_healthy():
                        self.total_failures += 1
                elif isinstance(result, Exception):
                    # Handle exceptions from health checks
                    check_results.append(HealthCheckResult(
                        check_name="unknown_check",
                        check_type=HealthCheckType.COMPONENT,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check exception: {str(result)}"
                    ))
                    self.total_failures += 1
        
        # Determine overall status
        overall_status = self._calculate_overall_status(check_results)
        
        # Generate summary
        summary = self._generate_summary(check_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(check_results)
        
        # Create report
        report = SystemHealthReport(
            overall_status=overall_status,
            check_results=check_results,
            summary=summary,
            recommendations=recommendations
        )
        
        self.last_report = report
        
        # Attempt auto-recovery if enabled
        if self.enable_auto_recovery and overall_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
            await self._attempt_auto_recovery(report)
        
        return report
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system status from individual results."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def _generate_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate summary statistics from health check results."""
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(1 for r in results if r.status == status)
        
        return {
            "total_checks": len(results),
            "status_counts": status_counts,
            "healthy_percentage": (
                status_counts.get("healthy", 0) / len(results) * 100
                if results else 0
            ),
            "uptime_seconds": time.time() - self.start_time,
            "total_checks_run": self.total_checks_run,
            "total_failures": self.total_failures,
            "success_rate": (
                (self.total_checks_run - self.total_failures) / self.total_checks_run * 100
                if self.total_checks_run > 0 else 0
            )
        }
    
    def _generate_recommendations(self, results: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []
        
        failed_checks = [r for r in results if not r.is_healthy()]
        
        for result in failed_checks:
            if result.recovery_action and result.recovery_action not in recommendations:
                recommendations.append(result.recovery_action)
        
        # Add general recommendations based on patterns
        critical_count = sum(1 for r in results if r.status == HealthStatus.CRITICAL)
        if critical_count > len(results) * 0.5:
            recommendations.append("System is experiencing widespread issues - consider restarting")
        
        resource_issues = [r for r in results if r.check_type == HealthCheckType.RESOURCE and not r.is_healthy()]
        if resource_issues:
            recommendations.append("Monitor resource usage and consider scaling")
        
        return recommendations
    
    async def _attempt_auto_recovery(self, report: SystemHealthReport) -> None:
        """Attempt automatic recovery for failed health checks."""
        failed_checks = report.get_failed_checks()
        
        for result in failed_checks:
            if result.is_recoverable and not result.recovery_attempted:
                try:
                    # Mark as attempted
                    result.recovery_attempted = True
                    
                    # Attempt recovery based on check type
                    if result.check_type == HealthCheckType.COMPONENT:
                        await self._recover_component(result)
                    elif result.check_type == HealthCheckType.PERFORMANCE:
                        await self._recover_performance(result)
                    elif result.check_type == HealthCheckType.RESOURCE:
                        await self._recover_resource(result)
                
                except Exception as e:
                    # Log recovery failure but don't fail the health check
                    pass
    
    async def _recover_component(self, result: HealthCheckResult) -> None:
        """Attempt to recover a failed component."""
        # This would implement component-specific recovery logic
        # For now, we just log the attempt
        pass
    
    async def _recover_performance(self, result: HealthCheckResult) -> None:
        """Attempt to recover from performance issues."""
        # This could trigger performance optimization procedures
        pass
    
    async def _recover_resource(self, result: HealthCheckResult) -> None:
        """Attempt to recover from resource issues."""
        # This could trigger resource cleanup or scaling
        pass
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_task is not None:
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if self._monitoring_task is not None:
            self._shutdown_event.set()
            await self._monitoring_task
            self._monitoring_task = None
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Run health checks
                await self.run_all_checks()
                
                # Wait for next check interval
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.check_interval
                )
                
            except asyncio.TimeoutError:
                # Normal timeout, continue monitoring
                continue
            except Exception as e:
                # Log error but continue monitoring
                await asyncio.sleep(5)  # Brief pause before retrying
    
    def get_health_check_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all health checks."""
        with self._checks_lock:
            return [check.get_statistics() for check in self.health_checks]
    
    async def shutdown(self) -> None:
        """Shutdown the health checker."""
        await self.stop_monitoring()


def health_check(logger_manager: 'LoggerManager') -> Dict[str, Any]:
    """
    Quick health check function for the logging system.
    
    Args:
        logger_manager: Logger manager instance to check
        
    Returns:
        Dictionary containing health status information
    """
    try:
        # Create temporary health checker
        checker = HealthChecker(logger_manager, enable_auto_recovery=False)
        
        # Run health checks synchronously
        import asyncio
        report = asyncio.run(checker.run_all_checks())
        
        return report.to_dict()
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": time.time()
        }


# Module exports
__all__ = [
    'HealthStatus',
    'HealthCheckType',
    'HealthCheckResult',
    'HealthCheck',
    'ComponentHealthCheck',
    'PerformanceHealthCheck',
    'ResourceHealthCheck',
    'SystemHealthReport',
    'HealthChecker',
    'health_check'
]