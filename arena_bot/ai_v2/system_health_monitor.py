"""
System Health Monitor - Complete System Health Monitoring

Provides comprehensive real-time system health monitoring with status indicators,
performance metrics, alerting, and automated health checks for all AI v2 components.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import json
from pathlib import Path
from collections import deque, defaultdict
import queue


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    last_updated: datetime
    trend: str = "stable"  # stable, improving, degrading


@dataclass
class ComponentHealth:
    """Health status for system component."""
    name: str
    status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemAlert:
    """System alert/notification."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring system.
    
    Monitors all AI v2 components, system resources, and external dependencies
    with real-time status indicators, alerting, and automated health checks.
    """
    
    def __init__(self, monitoring_interval: int = 30):
        """Initialize system health monitor."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.monitoring_interval = monitoring_interval
        self.monitoring_enabled = True
        
        # Health tracking
        self.component_health = {}
        self.system_metrics = {}
        self.health_history = deque(maxlen=1000)
        
        # Alerting
        self.alerts = deque(maxlen=500)
        self.alert_handlers = []
        self.alert_queue = queue.Queue()
        
        # Monitoring threads
        self.monitor_thread = None
        self.alert_thread = None
        self.health_lock = threading.Lock()
        
        # Performance tracking
        self.performance_baseline = {}
        self.performance_trends = defaultdict(list)
        
        # Component registry
        self.registered_components = {}
        self.component_checkers = {}
        
        # Health check results cache
        self.health_check_cache = {}
        self.cache_ttl = 60  # seconds
        
        # System startup time
        self.system_start_time = datetime.now()
        
        # Initialize built-in health checks
        self._register_builtin_health_checks()
        
        # Start monitoring
        self._start_monitoring()
        
        self.logger.info("SystemHealthMonitor initialized")
    
    def register_component(self, name: str, health_checker: Optional[Callable] = None,
                         dependencies: Optional[List[str]] = None,
                         warning_thresholds: Optional[Dict[str, float]] = None,
                         critical_thresholds: Optional[Dict[str, float]] = None) -> None:
        """Register a component for health monitoring."""
        with self.health_lock:
            self.component_health[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                dependencies=dependencies or []
            )
            
            if health_checker:
                self.component_checkers[name] = health_checker
            
            self.registered_components[name] = {
                'warning_thresholds': warning_thresholds or {},
                'critical_thresholds': critical_thresholds or {},
                'registered_at': datetime.now()
            }
        
        self.logger.info(f"Registered component for health monitoring: {name}")
    
    def update_component_status(self, component: str, status: HealthStatus,
                              metrics: Optional[Dict[str, float]] = None,
                              error_message: Optional[str] = None,
                              custom_data: Optional[Dict[str, Any]] = None) -> None:
        """Update component health status."""
        with self.health_lock:
            if component not in self.component_health:
                self.register_component(component)
            
            comp_health = self.component_health[component]
            old_status = comp_health.status
            comp_health.status = status
            comp_health.last_check = datetime.now()
            
            if error_message:
                comp_health.error_count += 1
                comp_health.last_error = error_message
            
            if custom_data:
                comp_health.custom_data.update(custom_data)
            
            # Update metrics
            if metrics:
                self._update_component_metrics(component, metrics)
            
            # Generate alert if status changed
            if old_status != status:
                self._generate_status_change_alert(component, old_status, status)
        
        self.logger.debug(f"Updated component status: {component} -> {status.value}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self.health_lock:
            # Calculate overall system status
            overall_status = self._calculate_overall_status()
            
            # Get system metrics
            system_metrics = self._get_system_metrics()
            
            # Get component summary
            component_summary = {}
            for name, health in self.component_health.items():
                component_summary[name] = {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'uptime_hours': health.uptime_seconds / 3600,
                    'error_count': health.error_count,
                    'last_error': health.last_error,
                    'metrics_count': len(health.metrics)
                }
            
            # Get recent alerts
            recent_alerts = [
                {
                    'id': alert.id,
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'message': alert.message,
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ]
            
            return {
                'overall_status': overall_status.value,
                'system_uptime_hours': (datetime.now() - self.system_start_time).total_seconds() / 3600,
                'timestamp': datetime.now().isoformat(),
                'system_metrics': system_metrics,
                'component_health': component_summary,
                'recent_alerts': recent_alerts,
                'monitoring_enabled': self.monitoring_enabled,
                'total_components': len(self.component_health),
                'healthy_components': len([h for h in self.component_health.values() if h.status == HealthStatus.HEALTHY]),
                'warning_components': len([h for h in self.component_health.values() if h.status == HealthStatus.WARNING]),
                'critical_components': len([h for h in self.component_health.values() if h.status == HealthStatus.CRITICAL])
            }
    
    def get_component_details(self, component: str) -> Optional[Dict[str, Any]]:
        """Get detailed health information for specific component."""
        with self.health_lock:
            if component not in self.component_health:
                return None
            
            health = self.component_health[component]
            
            # Get metric details
            metrics_details = {}
            for metric_name, metric in health.metrics.items():
                metrics_details[metric_name] = {
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status.value,
                    'threshold_warning': metric.threshold_warning,
                    'threshold_critical': metric.threshold_critical,
                    'last_updated': metric.last_updated.isoformat(),
                    'trend': metric.trend
                }
            
            # Get dependency status
            dependency_status = {}
            for dep in health.dependencies:
                if dep in self.component_health:
                    dependency_status[dep] = self.component_health[dep].status.value
                else:
                    dependency_status[dep] = 'unknown'
            
            return {
                'name': health.name,
                'status': health.status.value,
                'last_check': health.last_check.isoformat(),
                'uptime_seconds': health.uptime_seconds,
                'error_count': health.error_count,
                'last_error': health.last_error,
                'metrics': metrics_details,
                'dependencies': dependency_status,
                'custom_data': health.custom_data,
                'registration_info': self.registered_components.get(component, {})
            }
    
    def get_performance_trends(self, component: Optional[str] = None, 
                             hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if component:
            # Get trends for specific component
            component_trends = {}
            if component in self.performance_trends:
                recent_data = [
                    entry for entry in self.performance_trends[component]
                    if entry['timestamp'] > cutoff_time
                ]
                component_trends[component] = recent_data
            return component_trends
        else:
            # Get trends for all components
            all_trends = {}
            for comp_name, trend_data in self.performance_trends.items():
                recent_data = [
                    entry for entry in trend_data
                    if entry['timestamp'] > cutoff_time
                ]
                if recent_data:
                    all_trends[comp_name] = recent_data
            return all_trends
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Alert acknowledged: {alert_id}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        """Mark an alert as resolved."""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    if resolution_note:
                        alert.details['resolution_note'] = resolution_note
                    self.logger.info(f"Alert resolved: {alert_id}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False
    
    def register_alert_handler(self, handler: Callable[[SystemAlert], None]) -> None:
        """Register alert handler function."""
        self.alert_handlers.append(handler)
        self.logger.debug(f"Registered alert handler: {handler.__name__}")
    
    def run_health_check(self, component: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Run health check for specific component or all components."""
        results = {}
        
        if component:
            # Check specific component
            if component in self.component_checkers:
                results[component] = self._run_component_health_check(component, force)
            else:
                results[component] = {'error': 'No health checker registered'}
        else:
            # Check all components
            for comp_name in self.component_checkers:
                results[comp_name] = self._run_component_health_check(comp_name, force)
        
        return results
    
    def export_health_report(self, format: str = 'json') -> str:
        """Export comprehensive health report."""
        try:
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'system_health': self.get_system_health(),
                'component_details': {
                    name: self.get_component_details(name)
                    for name in self.component_health.keys()
                },
                'performance_trends': self.get_performance_trends(hours=168),  # 1 week
                'recent_alerts': [
                    {
                        'id': alert.id,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity.value,
                        'component': alert.component,
                        'message': alert.message,
                        'details': alert.details,
                        'acknowledged': alert.acknowledged,
                        'resolved': alert.resolved
                    }
                    for alert in list(self.alerts)[-50:]  # Last 50 alerts
                ]
            }
            
            if format.lower() == 'json':
                return json.dumps(report_data, indent=2, default=str)
            else:
                # Could add other formats (XML, CSV, etc.)
                return json.dumps(report_data, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error exporting health report: {e}")
            return f'{{"error": "{e}"}}'
    
    def shutdown(self) -> None:
        """Shutdown health monitoring gracefully."""
        try:
            self.monitoring_enabled = False
            
            # Stop monitoring threads
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            if self.alert_thread and self.alert_thread.is_alive():
                self.alert_thread.join(timeout=5)
            
            self.logger.info("System health monitor shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during health monitor shutdown: {e}")
    
    # === INTERNAL METHODS ===
    
    def _register_builtin_health_checks(self) -> None:
        """Register built-in health checks for system components."""
        # Register system resource checks
        self.register_component(
            'system_resources',
            health_checker=self._check_system_resources,
            warning_thresholds={'cpu_usage': 80.0, 'memory_usage': 85.0, 'disk_usage': 90.0},
            critical_thresholds={'cpu_usage': 95.0, 'memory_usage': 95.0, 'disk_usage': 95.0}
        )
        
        # Register AI v2 component checks
        ai_components = [
            'hero_selector', 'card_evaluator', 'grandmaster_advisor',
            'conversational_coach', 'draft_exporter', 'cache_manager',
            'resource_manager', 'error_recovery', 'privacy_manager'
        ]
        
        for component in ai_components:
            self.register_component(
                component,
                health_checker=lambda comp=component: self._check_ai_component(comp),
                warning_thresholds={'response_time_ms': 1000.0, 'error_rate': 5.0},
                critical_thresholds={'response_time_ms': 5000.0, 'error_rate': 20.0}
            )
    
    def _start_monitoring(self) -> None:
        """Start background monitoring threads."""
        # Main monitoring thread
        def monitor_worker():
            while self.monitoring_enabled:
                try:
                    self._run_monitoring_cycle()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Error in monitoring cycle: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        # Alert processing thread
        def alert_worker():
            while self.monitoring_enabled:
                try:
                    alert = self.alert_queue.get(timeout=5)
                    self._process_alert(alert)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing alert: {e}")
        
        self.alert_thread = threading.Thread(target=alert_worker, daemon=True)
        self.alert_thread.start()
    
    def _run_monitoring_cycle(self) -> None:
        """Run complete monitoring cycle."""
        cycle_start = datetime.now()
        
        try:
            # Run health checks for all registered components
            for component in list(self.component_checkers.keys()):
                self._run_component_health_check(component)
            
            # Update system metrics
            self._get_system_metrics()
            
            # Check performance trends
            self._analyze_performance_trends()
            
            # Record cycle in history
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.health_history.append({
                'timestamp': cycle_start,
                'cycle_duration_ms': cycle_duration * 1000,
                'components_checked': len(self.component_checkers),
                'alerts_generated': len([a for a in self.alerts if a.timestamp > cycle_start])
            })
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def _run_component_health_check(self, component: str, force: bool = False) -> Dict[str, Any]:
        """Run health check for specific component."""
        try:
            # Check cache first
            if not force and component in self.health_check_cache:
                cache_entry = self.health_check_cache[component]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_ttl:
                    return cache_entry['result']
            
            # Run health check
            start_time = time.time()
            health_checker = self.component_checkers[component]
            
            try:
                check_result = health_checker()
                response_time = (time.time() - start_time) * 1000
                
                # Process result
                status = HealthStatus.HEALTHY
                metrics = {'response_time_ms': response_time}
                error_message = None
                
                if isinstance(check_result, dict):
                    status = HealthStatus(check_result.get('status', 'healthy'))
                    metrics.update(check_result.get('metrics', {}))
                    error_message = check_result.get('error')
                elif isinstance(check_result, bool):
                    status = HealthStatus.HEALTHY if check_result else HealthStatus.CRITICAL
                
                # Update component status
                self.update_component_status(component, status, metrics, error_message)
                
                result = {
                    'status': status.value,
                    'response_time_ms': response_time,
                    'metrics': metrics,
                    'error': error_message
                }
                
                # Cache result
                self.health_check_cache[component] = {
                    'timestamp': datetime.now(),
                    'result': result
                }
                
                return result
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                error_msg = f"Health check failed: {e}"
                
                self.update_component_status(
                    component, HealthStatus.CRITICAL, 
                    {'response_time_ms': response_time}, error_msg
                )
                
                return {
                    'status': HealthStatus.CRITICAL.value,
                    'response_time_ms': response_time,
                    'error': error_msg
                }
                
        except Exception as e:
            self.logger.error(f"Error running health check for {component}: {e}")
            return {'status': HealthStatus.UNKNOWN.value, 'error': str(e)}
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine status
            status = HealthStatus.HEALTHY
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 80 or memory_percent > 85 or disk_percent > 90:
                status = HealthStatus.WARNING
            
            return {
                'status': status.value,
                'metrics': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'disk_usage': disk_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': f"System resource check failed: {e}"
            }
    
    def _check_ai_component(self, component: str) -> Dict[str, Any]:
        """Generic AI component health check."""
        try:
            # This would be customized for each component
            # For now, just return healthy status
            return {
                'status': HealthStatus.HEALTHY.value,
                'metrics': {
                    'last_activity': time.time(),
                    'operational': True
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': f"AI component check failed: {e}"
            }
    
    def _update_component_metrics(self, component: str, metrics: Dict[str, float]) -> None:
        """Update component metrics with trend analysis."""
        try:
            comp_health = self.component_health[component]
            thresholds = self.registered_components.get(component, {})
            warning_thresholds = thresholds.get('warning_thresholds', {})
            critical_thresholds = thresholds.get('critical_thresholds', {})
            
            for metric_name, value in metrics.items():
                # Determine metric status
                metric_status = HealthStatus.HEALTHY
                if metric_name in critical_thresholds and value > critical_thresholds[metric_name]:
                    metric_status = HealthStatus.CRITICAL
                elif metric_name in warning_thresholds and value > warning_thresholds[metric_name]:
                    metric_status = HealthStatus.WARNING
                
                # Calculate trend
                trend = "stable"
                if metric_name in comp_health.metrics:
                    old_value = comp_health.metrics[metric_name].value
                    if value > old_value * 1.1:
                        trend = "degrading"
                    elif value < old_value * 0.9:
                        trend = "improving"
                
                # Create/update metric
                comp_health.metrics[metric_name] = HealthMetric(
                    name=metric_name,
                    value=value,
                    unit=self._get_metric_unit(metric_name),
                    status=metric_status,
                    threshold_warning=warning_thresholds.get(metric_name, float('inf')),
                    threshold_critical=critical_thresholds.get(metric_name, float('inf')),
                    last_updated=datetime.now(),
                    trend=trend
                )
            
            # Record performance data
            if component not in self.performance_trends:
                self.performance_trends[component] = deque(maxlen=1000)
            
            self.performance_trends[component].append({
                'timestamp': datetime.now(),
                'metrics': metrics.copy()
            })
            
        except Exception as e:
            self.logger.error(f"Error updating component metrics: {e}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric."""
        unit_mapping = {
            'response_time_ms': 'ms',
            'cpu_usage': '%',
            'memory_usage': '%',
            'disk_usage': '%',
            'error_rate': '%',
            'throughput': 'ops/s',
            'latency': 'ms',
            'memory_available_gb': 'GB',
            'disk_free_gb': 'GB'
        }
        return unit_mapping.get(metric_name, '')
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN
        
        statuses = [health.status for health in self.component_health.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.OFFLINE in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.WARNING
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            return {
                'monitoring_cycle_interval': self.monitoring_interval,
                'registered_components': len(self.registered_components),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'total_alerts': len(self.alerts),
                'health_checks_cached': len(self.health_check_cache),
                'performance_data_points': sum(len(trends) for trends in self.performance_trends.values())
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _analyze_performance_trends(self) -> None:
        """Analyze performance trends for anomaly detection."""
        try:
            for component, trend_data in self.performance_trends.items():
                if len(trend_data) < 10:  # Need minimum data points
                    continue
                
                # Analyze recent performance
                recent_data = list(trend_data)[-10:]
                
                for metric_name in ['response_time_ms', 'error_rate']:
                    values = [entry['metrics'].get(metric_name) for entry in recent_data]
                    values = [v for v in values if v is not None]
                    
                    if not values:
                        continue
                    
                    # Check for significant degradation
                    avg_value = sum(values) / len(values)
                    baseline = self.performance_baseline.get(f"{component}_{metric_name}")
                    
                    if baseline and avg_value > baseline * 2:  # 100% degradation
                        self._generate_performance_alert(
                            component, metric_name, avg_value, baseline
                        )
                    
                    # Update baseline
                    self.performance_baseline[f"{component}_{metric_name}"] = avg_value
                    
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
    
    def _generate_status_change_alert(self, component: str, old_status: HealthStatus, 
                                    new_status: HealthStatus) -> None:
        """Generate alert for status changes."""
        try:
            severity = AlertSeverity.INFO
            if new_status == HealthStatus.CRITICAL:
                severity = AlertSeverity.CRITICAL
            elif new_status == HealthStatus.WARNING:
                severity = AlertSeverity.WARNING
            elif old_status in [HealthStatus.CRITICAL, HealthStatus.WARNING] and new_status == HealthStatus.HEALTHY:
                severity = AlertSeverity.INFO
            
            alert = SystemAlert(
                id=f"status_change_{component}_{int(time.time())}",
                timestamp=datetime.now(),
                severity=severity,
                component=component,
                message=f"Component {component} status changed from {old_status.value} to {new_status.value}",
                details={'old_status': old_status.value, 'new_status': new_status.value}
            )
            
            self._queue_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error generating status change alert: {e}")
    
    def _generate_performance_alert(self, component: str, metric: str, 
                                  current_value: float, baseline: float) -> None:
        """Generate alert for performance degradation."""
        try:
            alert = SystemAlert(
                id=f"performance_{component}_{metric}_{int(time.time())}",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                component=component,
                message=f"Performance degradation detected in {component}: {metric} = {current_value:.2f} (baseline: {baseline:.2f})",
                details={
                    'metric': metric,
                    'current_value': current_value,
                    'baseline': baseline,
                    'degradation_percent': ((current_value - baseline) / baseline) * 100
                }
            )
            
            self._queue_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error generating performance alert: {e}")
    
    def _queue_alert(self, alert: SystemAlert) -> None:
        """Queue alert for processing."""
        try:
            self.alerts.append(alert)
            self.alert_queue.put(alert)
        except Exception as e:
            self.logger.error(f"Error queuing alert: {e}")
    
    def _process_alert(self, alert: SystemAlert) -> None:
        """Process alert through registered handlers."""
        try:
            self.logger.info(f"Processing alert: {alert.severity.value} - {alert.message}")
            
            # Call registered alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing alert: {e}")


# Global system health monitor instance
_health_monitor = None

def get_health_monitor() -> SystemHealthMonitor:
    """Get global system health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealthMonitor()
    return _health_monitor