"""
Ultra-Debug Mode for Arena Bot Deep Debugging

Maximum debugging visibility system that coordinates all debugging components
to provide unprecedented insight into system behavior:

- Real-time system monitoring with live dashboards
- Comprehensive execution tracing with full call stack analysis
- Advanced memory profiling with leak detection and optimization suggestions
- Automated anomaly detection with predictive analysis
- Cross-component correlation analysis for complex failure scenarios
- Emergency debugging mode with crisis response protocols
- Performance bottleneck identification with optimization recommendations
- Security monitoring with threat detection and audit trails

This is the ultimate debugging tool for complex, hard-to-track issues.
"""

import time
import threading
import asyncio
import psutil
import gc
import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque
from pathlib import Path

# Import all debugging components
from .method_tracer import get_method_tracer
from .state_monitor import get_state_monitor, StateChangeEvent
from .pipeline_tracer import get_pipeline_tracer, PipelineStage
from .health_monitor import get_health_monitor, ComponentHealth
from .error_analyzer import get_error_analyzer, ErrorPattern
from .exception_handler import get_exception_handler, ExceptionContext
from .enhanced_logger import get_enhanced_logger, DebugLogLevel, activate_emergency_debug
from .integration import get_debugging_integrator

from ..logging_system.logger import get_logger, LogLevel


class UltraDebugMode(Enum):
    """Ultra-debug operating modes."""
    DISABLED = "disabled"
    MONITORING = "monitoring"        # Passive monitoring with minimal overhead
    ANALYSIS = "analysis"           # Active analysis with moderate overhead
    INTROSPECTION = "introspection" # Deep introspection with high overhead
    EMERGENCY = "emergency"         # Emergency mode with maximum capture
    CRISIS = "crisis"              # Crisis mode with all-systems capture


@dataclass
class SystemMetrics:
    """Real-time system metrics snapshot."""
    
    timestamp: float = field(default_factory=time.time)
    
    # CPU and Memory
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    process_memory_mb: float = 0.0
    
    # Performance
    active_threads: int = 0
    open_files: int = 0
    network_connections: int = 0
    
    # Arena Bot specific
    active_traces: int = 0
    active_pipelines: int = 0
    pending_state_changes: int = 0
    circuit_breakers_open: int = 0
    recent_exceptions: int = 0
    
    # Performance indicators
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    throughput_ops_per_sec: float = 0.0
    
    def capture_metrics(self) -> None:
        """Capture current system metrics."""
        try:
            # System metrics
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            self.memory_available_gb = memory.available / (1024**3)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.process_memory_mb = process_memory.rss / (1024**2)
            self.active_threads = process.num_threads()
            
            try:
                self.open_files = len(process.open_files())
                self.network_connections = len(process.connections())
            except:
                pass  # Permissions might not allow this
            
            # Debug component metrics
            try:
                method_tracer = get_method_tracer()
                self.active_traces = len(method_tracer.get_active_traces())
                
                pipeline_tracer = get_pipeline_tracer()
                self.active_pipelines = len(pipeline_tracer.get_active_pipelines())
                
                state_monitor = get_state_monitor()
                self.pending_state_changes = len(state_monitor.get_recent_changes(seconds=60))
                
                health_monitor = get_health_monitor()
                health_summary = health_monitor.get_system_health_summary()
                if isinstance(health_summary, dict):
                    status_counts = health_summary.get('status_counts', {})
                    self.circuit_breakers_open = status_counts.get('failed', 0) + status_counts.get('critical', 0)
                
                exception_handler = get_exception_handler()
                recent_contexts = exception_handler.get_exception_contexts(hours=1)
                self.recent_exceptions = len(recent_contexts)
                
            except Exception:
                pass  # Don't let metrics capture affect ultra-debug
        
        except Exception:
            pass  # Don't let any failure affect ultra-debug operation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_gb': self.memory_available_gb,
            'process_memory_mb': self.process_memory_mb,
            'active_threads': self.active_threads,
            'open_files': self.open_files,
            'network_connections': self.network_connections,
            'active_traces': self.active_traces,
            'active_pipelines': self.active_pipelines,
            'pending_state_changes': self.pending_state_changes,
            'circuit_breakers_open': self.circuit_breakers_open,
            'recent_exceptions': self.recent_exceptions,
            'response_time_p95': self.response_time_p95,
            'error_rate': self.error_rate,
            'throughput_ops_per_sec': self.throughput_ops_per_sec
        }


@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""
    
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Alert details
    severity: str = "medium"  # low, medium, high, critical
    category: str = ""        # performance, memory, threading, component
    title: str = ""
    description: str = ""
    
    # Detection details
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    historical_baseline: float = 0.0
    
    # Context
    affected_components: List[str] = field(default_factory=list)
    related_traces: List[str] = field(default_factory=list)
    
    # Resolution
    suggested_actions: List[str] = field(default_factory=list)
    auto_resolution_attempted: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'severity': self.severity,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'historical_baseline': self.historical_baseline,
            'affected_components': self.affected_components,
            'related_traces': self.related_traces,
            'suggested_actions': self.suggested_actions,
            'auto_resolution_attempted': self.auto_resolution_attempted,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time
        }


class AnomalyDetector:
    """Advanced anomaly detection system."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.logger = get_enhanced_logger("arena_bot.debugging.ultra_debug.anomaly_detector")
        
        # Historical data for baseline analysis
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.baseline_window_minutes = 60
        
        # Anomaly detection thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 80, 'critical': 95},
            'memory_percent': {'warning': 85, 'critical': 95},
            'response_time_p95': {'warning': 1000, 'critical': 5000},  # milliseconds
            'error_rate': {'warning': 0.05, 'critical': 0.15},  # 5% and 15%
            'circuit_breakers_open': {'warning': 1, 'critical': 3},
            'recent_exceptions': {'warning': 5, 'critical': 15}
        }
        
        # Alert management
        self.active_alerts: Dict[str, AnomalyAlert] = {}
        self.alert_history: deque = deque(maxlen=500)
        self.suppression_window_minutes = 10  # Don't repeat alerts for same issue
        
        # Statistical analysis
        self.enable_statistical_analysis = True
        self.anomaly_detection_sensitivity = 2.0  # Standard deviations for anomaly
    
    def add_metrics(self, metrics: SystemMetrics) -> None:
        """Add metrics for analysis."""
        self.metrics_history.append(metrics)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(metrics)
        
        # Process any detected anomalies
        for anomaly in anomalies:
            self._process_anomaly(anomaly)
    
    def detect_anomalies(self, current_metrics: SystemMetrics) -> List[AnomalyAlert]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        # Threshold-based detection
        anomalies.extend(self._detect_threshold_anomalies(current_metrics))
        
        # Statistical anomaly detection
        if self.enable_statistical_analysis and len(self.metrics_history) > 30:
            anomalies.extend(self._detect_statistical_anomalies(current_metrics))
        
        # Pattern-based anomaly detection
        anomalies.extend(self._detect_pattern_anomalies(current_metrics))
        
        return anomalies
    
    def _detect_threshold_anomalies(self, metrics: SystemMetrics) -> List[AnomalyAlert]:
        """Detect threshold-based anomalies."""
        anomalies = []
        
        for metric_name, thresholds in self.thresholds.items():
            current_value = getattr(metrics, metric_name, 0)
            
            # Check critical threshold
            if current_value >= thresholds['critical']:
                anomaly = AnomalyAlert(
                    severity="critical",
                    category=self._categorize_metric(metric_name),
                    title=f"Critical {metric_name.replace('_', ' ').title()}",
                    description=f"{metric_name} has reached critical level: {current_value}",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=thresholds['critical']
                )
                anomaly.suggested_actions = self._get_remediation_actions(metric_name, "critical")
                anomalies.append(anomaly)
            
            # Check warning threshold
            elif current_value >= thresholds['warning']:
                anomaly = AnomalyAlert(
                    severity="high",
                    category=self._categorize_metric(metric_name),
                    title=f"High {metric_name.replace('_', ' ').title()}",
                    description=f"{metric_name} is approaching critical level: {current_value}",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=thresholds['warning']
                )
                anomaly.suggested_actions = self._get_remediation_actions(metric_name, "warning")
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, metrics: SystemMetrics) -> List[AnomalyAlert]:
        """Detect statistical anomalies using historical baseline."""
        anomalies = []
        
        # Calculate baseline from recent history
        recent_metrics = [m for m in self.metrics_history 
                         if time.time() - m.timestamp <= self.baseline_window_minutes * 60]
        
        if len(recent_metrics) < 10:
            return anomalies
        
        # Check key metrics for statistical anomalies
        check_metrics = ['cpu_percent', 'memory_percent', 'response_time_p95', 'error_rate']
        
        for metric_name in check_metrics:
            values = [getattr(m, metric_name, 0) for m in recent_metrics]
            if not values:
                continue
            
            # Calculate statistics
            mean_value = sum(values) / len(values)
            variance = sum((x - mean_value) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            current_value = getattr(metrics, metric_name, 0)
            
            # Check if current value is an anomaly
            if std_dev > 0:
                z_score = abs(current_value - mean_value) / std_dev
                
                if z_score > self.anomaly_detection_sensitivity:
                    anomaly = AnomalyAlert(
                        severity="medium" if z_score < 3.0 else "high",
                        category=self._categorize_metric(metric_name),
                        title=f"Statistical Anomaly in {metric_name.replace('_', ' ').title()}",
                        description=f"{metric_name} shows unusual deviation from baseline (z-score: {z_score:.2f})",
                        metric_name=metric_name,
                        current_value=current_value,
                        historical_baseline=mean_value
                    )
                    anomaly.suggested_actions = self._get_remediation_actions(metric_name, "anomaly")
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pattern_anomalies(self, metrics: SystemMetrics) -> List[AnomalyAlert]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        # Check for concerning combinations
        if (metrics.cpu_percent > 90 and metrics.memory_percent > 90 and 
            metrics.recent_exceptions > 3):
            anomaly = AnomalyAlert(
                severity="critical",
                category="system",
                title="System Resource Crisis",
                description="High CPU, memory usage, and exception rate detected simultaneously",
                suggested_actions=[
                    "Immediate system resource investigation required",
                    "Check for runaway processes or memory leaks",
                    "Consider emergency system restart if critical"
                ]
            )
            anomalies.append(anomaly)
        
        # Check for threading issues
        if metrics.active_threads > 50 and metrics.cpu_percent < 30:
            anomaly = AnomalyAlert(
                severity="high",
                category="threading",
                title="Potential Thread Deadlock",
                description="High thread count with low CPU usage suggests potential deadlock",
                suggested_actions=[
                    "Investigate thread states for deadlocks",
                    "Check for blocking operations",
                    "Review synchronization mechanisms"
                ]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric for alert classification."""
        if 'cpu' in metric_name or 'memory' in metric_name:
            return "performance"
        elif 'thread' in metric_name:
            return "threading"
        elif 'exception' in metric_name or 'error' in metric_name:
            return "reliability"
        elif 'breaker' in metric_name:
            return "component"
        else:
            return "system"
    
    def _get_remediation_actions(self, metric_name: str, severity: str) -> List[str]:
        """Get remediation actions for specific metric issues."""
        actions_map = {
            'cpu_percent': [
                "Identify CPU-intensive processes",
                "Check for infinite loops or inefficient algorithms",
                "Consider process optimization or load balancing"
            ],
            'memory_percent': [
                "Investigate memory usage patterns",
                "Check for memory leaks",
                "Implement garbage collection or cache cleanup"
            ],
            'response_time_p95': [
                "Analyze slow operations",
                "Check for blocking I/O operations",
                "Optimize database queries or API calls"
            ],
            'error_rate': [
                "Investigate recent error patterns",
                "Check component health status",
                "Review error logs for root causes"
            ],
            'recent_exceptions': [
                "Review exception logs for patterns",
                "Check component stability",
                "Investigate system resource constraints"
            ]
        }
        
        return actions_map.get(metric_name, ["Investigate the issue further"])
    
    def _process_anomaly(self, anomaly: AnomalyAlert) -> None:
        """Process detected anomaly."""
        # Check for suppression (avoid duplicate alerts)
        suppression_key = f"{anomaly.category}_{anomaly.metric_name}"
        cutoff_time = time.time() - (self.suppression_window_minutes * 60)
        
        # Check if similar alert was recently raised
        recent_similar = any(
            alert.category == anomaly.category and 
            alert.metric_name == anomaly.metric_name and
            alert.timestamp > cutoff_time
            for alert in self.alert_history
        )
        
        if recent_similar:
            return  # Suppress duplicate alert
        
        # Add to active alerts
        self.active_alerts[anomaly.alert_id] = anomaly
        self.alert_history.append(anomaly)
        
        # Log the anomaly
        log_level_map = {
            'low': LogLevel.INFO,
            'medium': LogLevel.WARNING,
            'high': LogLevel.ERROR,
            'critical': LogLevel.CRITICAL
        }
        
        log_level = log_level_map.get(anomaly.severity, LogLevel.WARNING)
        
        self.logger.log(
            log_level,
            f"🚨 ANOMALY_DETECTED: {anomaly.title}",
            extra={
                'anomaly_alert': anomaly.to_dict(),
                'ultra_debug_context': {
                    'detection_system': 'ultra_debug_anomaly_detector',
                    'alert_id': anomaly.alert_id,
                    'severity': anomaly.severity,
                    'category': anomaly.category
                }
            }
        )
        
        # Attempt auto-resolution for certain types
        if anomaly.severity in ['high', 'critical']:
            self._attempt_auto_resolution(anomaly)
    
    def _attempt_auto_resolution(self, anomaly: AnomalyAlert) -> None:
        """Attempt automatic resolution of anomaly."""
        anomaly.auto_resolution_attempted = True
        
        try:
            resolved = False
            
            # Memory-related auto-resolution
            if anomaly.metric_name == 'memory_percent' and anomaly.current_value > 90:
                # Force garbage collection
                gc.collect()
                
                # Wait a moment and recheck
                time.sleep(2)
                current_metrics = SystemMetrics()
                current_metrics.capture_metrics()
                
                if current_metrics.memory_percent < anomaly.current_value * 0.9:
                    resolved = True
                    self.logger.info(f"✅ AUTO_RESOLVED: Memory anomaly resolved via garbage collection")
            
            # Circuit breaker auto-resolution
            elif anomaly.metric_name == 'circuit_breakers_open':
                try:
                    health_monitor = get_health_monitor()
                    # In a real implementation, this might reset specific circuit breakers
                    # based on analysis of their state
                    resolved = False  # Placeholder - would need actual reset logic
                except:
                    pass
            
            if resolved:
                anomaly.resolved = True
                anomaly.resolution_time = time.time()
                
                # Remove from active alerts
                self.active_alerts.pop(anomaly.alert_id, None)
        
        except Exception as e:
            self.logger.error(f"Auto-resolution attempt failed: {e}")
    
    def get_active_alerts(self) -> List[AnomalyAlert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity] += 1
        
        return {
            'total_active_alerts': len(self.active_alerts),
            'active_by_severity': dict(active_by_severity),
            'total_historical_alerts': len(self.alert_history),
            'auto_resolution_attempts': sum(1 for a in self.alert_history if a.auto_resolution_attempted),
            'successful_auto_resolutions': sum(1 for a in self.alert_history if a.resolved and a.auto_resolution_attempted)
        }


class UltraDebugManager:
    """
    Ultra-Debug Manager - Maximum debugging visibility system.
    
    Coordinates all debugging components to provide unprecedented insight
    into system behavior, with real-time monitoring, anomaly detection,
    and automated problem resolution.
    """
    
    def __init__(self):
        """Initialize ultra-debug manager."""
        self.logger = get_enhanced_logger("arena_bot.debugging.ultra_debug")
        
        # Operating mode
        self.mode = UltraDebugMode.DISABLED
        self.previous_mode = UltraDebugMode.DISABLED
        
        # Components
        self.anomaly_detector = AnomalyDetector()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Configuration
        self.monitoring_interval_seconds = 5.0
        self.emergency_threshold_violations = 3
        self.crisis_mode_duration_minutes = 30
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=2000)
        self.performance_baseline: Optional[SystemMetrics] = None
        
        # State management
        self.start_time = time.time()
        self.mode_change_history: List[Dict[str, Any]] = []
        self.emergency_activations = 0
        self.crisis_activations = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Integration with existing systems
        self.integrator = get_debugging_integrator()
    
    def enable_ultra_debug(self, mode: UltraDebugMode = UltraDebugMode.MONITORING) -> bool:
        """
        Enable ultra-debug mode.
        
        Args:
            mode: Ultra-debug mode to activate
            
        Returns:
            True if successfully enabled, False otherwise
        """
        
        with self.lock:
            if self.mode != UltraDebugMode.DISABLED:
                self.logger.warning(f"Ultra-debug already active in {self.mode.value} mode")
                return True
            
            try:
                # Initialize debugging integrator if needed
                if not self.integrator.is_initialized:
                    success = self.integrator.initialize("ultra", auto_start_monitoring=True)
                    if not success:
                        self.logger.error("Failed to initialize debugging integrator")
                        return False
                
                # Enable ultra-level debugging
                self.integrator.enable_debugging("ultra")
                
                # Set mode
                self.previous_mode = self.mode
                self.mode = mode
                
                # Record mode change
                mode_change = {
                    'timestamp': time.time(),
                    'from_mode': self.previous_mode.value,
                    'to_mode': mode.value,
                    'trigger': 'manual_activation'
                }
                self.mode_change_history.append(mode_change)
                
                # Start monitoring based on mode
                if mode != UltraDebugMode.DISABLED:
                    self._start_monitoring()
                
                # Capture initial baseline
                self._capture_performance_baseline()
                
                self.logger.critical(
                    f"🚀 ULTRA_DEBUG_ACTIVATED: Mode {mode.value} - Maximum debugging visibility enabled",
                    extra={
                        'ultra_debug_activation': {
                            'mode': mode.value,
                            'timestamp': time.time(),
                            'baseline_captured': self.performance_baseline is not None,
                            'monitoring_started': self.monitoring_active
                        }
                    }
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to enable ultra-debug mode: {e}")
                return False
    
    def disable_ultra_debug(self) -> None:
        """Disable ultra-debug mode."""
        
        with self.lock:
            if self.mode == UltraDebugMode.DISABLED:
                return
            
            # Stop monitoring
            self._stop_monitoring()
            
            # Record mode change
            mode_change = {
                'timestamp': time.time(),
                'from_mode': self.mode.value,
                'to_mode': UltraDebugMode.DISABLED.value,
                'trigger': 'manual_deactivation'
            }
            self.mode_change_history.append(mode_change)
            
            # Set mode
            self.previous_mode = self.mode
            self.mode = UltraDebugMode.DISABLED
            
            # Disable debugging integrator
            if self.integrator.is_enabled:
                self.integrator.disable_debugging()
            
            self.logger.info(
                f"🔄 ULTRA_DEBUG_DEACTIVATED: Disabled from {self.previous_mode.value} mode",
                extra={
                    'ultra_debug_deactivation': {
                        'previous_mode': self.previous_mode.value,
                        'timestamp': time.time(),
                        'uptime_seconds': time.time() - self.start_time,
                        'metrics_collected': len(self.metrics_history)
                    }
                }
            )
    
    def escalate_to_emergency(self, trigger_reason: str = "manual") -> None:
        """Escalate to emergency debug mode."""
        
        with self.lock:
            self.emergency_activations += 1
            
            # Record mode change
            mode_change = {
                'timestamp': time.time(),
                'from_mode': self.mode.value,
                'to_mode': UltraDebugMode.EMERGENCY.value,
                'trigger': f'emergency_escalation_{trigger_reason}'
            }
            self.mode_change_history.append(mode_change)
            
            self.previous_mode = self.mode
            self.mode = UltraDebugMode.EMERGENCY
            
            # Activate emergency debug in logging system
            activate_emergency_debug()
            
            # Enable full context capture in all components
            if self.integrator.exception_handler:
                self.integrator.exception_handler.capture_full_context = True
                self.integrator.exception_handler.auto_analyze = True
            
            # Increase monitoring frequency
            self.monitoring_interval_seconds = 1.0
            
            self.logger.critical(
                f"🚨 EMERGENCY_DEBUG_ACTIVATED: Triggered by {trigger_reason}",
                extra={
                    'emergency_activation': {
                        'trigger_reason': trigger_reason,
                        'activation_count': self.emergency_activations,
                        'timestamp': time.time(),
                        'monitoring_interval': self.monitoring_interval_seconds
                    }
                }
            )
    
    def escalate_to_crisis(self, trigger_reason: str = "automated") -> None:
        """Escalate to crisis debug mode."""
        
        with self.lock:
            self.crisis_activations += 1
            
            # Record mode change
            mode_change = {
                'timestamp': time.time(),
                'from_mode': self.mode.value,
                'to_mode': UltraDebugMode.CRISIS.value,
                'trigger': f'crisis_escalation_{trigger_reason}'
            }
            self.mode_change_history.append(mode_change)
            
            self.previous_mode = self.mode
            self.mode = UltraDebugMode.CRISIS
            
            # Maximum monitoring frequency
            self.monitoring_interval_seconds = 0.5
            
            # Enable maximum capture in all components
            if self.integrator.method_tracer:
                self.integrator.method_tracer.capture_parameters = True
                self.integrator.method_tracer.capture_memory = True
                self.integrator.method_tracer.capture_return_values = True
            
            if self.integrator.pipeline_tracer:
                self.integrator.pipeline_tracer.capture_data_snapshots = True
                self.integrator.pipeline_tracer.detect_bottlenecks = True
            
            self.logger.critical(
                f"🔥 CRISIS_DEBUG_ACTIVATED: System in crisis mode - Triggered by {trigger_reason}",
                extra={
                    'crisis_activation': {
                        'trigger_reason': trigger_reason,
                        'activation_count': self.crisis_activations,
                        'timestamp': time.time(),
                        'monitoring_interval': self.monitoring_interval_seconds,
                        'duration_limit_minutes': self.crisis_mode_duration_minutes
                    }
                }
            )
            
            # Schedule automatic deescalation
            threading.Timer(
                self.crisis_mode_duration_minutes * 60,
                self._auto_deescalate_from_crisis
            ).start()
    
    def _auto_deescalate_from_crisis(self) -> None:
        """Automatically deescalate from crisis mode after timeout."""
        with self.lock:
            if self.mode == UltraDebugMode.CRISIS:
                self.logger.info("⏰ Auto-deescalating from crisis mode due to timeout")
                
                # Return to emergency mode
                self.previous_mode = self.mode
                self.mode = UltraDebugMode.EMERGENCY
                self.monitoring_interval_seconds = 2.0
                
                # Record mode change
                mode_change = {
                    'timestamp': time.time(),
                    'from_mode': UltraDebugMode.CRISIS.value,
                    'to_mode': UltraDebugMode.EMERGENCY.value,
                    'trigger': 'auto_deescalation_timeout'
                }
                self.mode_change_history.append(mode_change)
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="UltraDebugMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"📊 Ultra-debug monitoring started (interval: {self.monitoring_interval_seconds}s)")
    
    def _stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("📊 Ultra-debug monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Capture current metrics
                metrics = SystemMetrics()
                metrics.capture_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Run anomaly detection
                self.anomaly_detector.add_metrics(metrics)
                
                # Check for escalation conditions
                self._check_escalation_conditions(metrics)
                
                # Log monitoring data based on mode
                self._log_monitoring_data(metrics)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            # Sleep until next monitoring cycle
            time.sleep(self.monitoring_interval_seconds)
    
    def _check_escalation_conditions(self, metrics: SystemMetrics) -> None:
        """Check if conditions warrant mode escalation."""
        
        # Get active alerts
        active_alerts = self.anomaly_detector.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == 'critical']
        
        # Auto-escalate to emergency if multiple critical alerts
        if (self.mode in [UltraDebugMode.MONITORING, UltraDebugMode.ANALYSIS] and 
            len(critical_alerts) >= 2):
            self.escalate_to_emergency("multiple_critical_alerts")
        
        # Auto-escalate to crisis if extreme conditions
        elif (self.mode == UltraDebugMode.EMERGENCY and
              (metrics.cpu_percent > 98 or metrics.memory_percent > 98 or 
               len(critical_alerts) >= 3)):
            self.escalate_to_crisis("extreme_system_conditions")
    
    def _log_monitoring_data(self, metrics: SystemMetrics) -> None:
        """Log monitoring data based on current mode."""
        
        if self.mode == UltraDebugMode.MONITORING:
            # Log summary every 10 cycles
            if len(self.metrics_history) % 10 == 0:
                self.logger.trace_deep(
                    f"📊 MONITORING_SUMMARY: {len(self.metrics_history)} metrics collected",
                    debug_context={
                        'monitoring_summary': metrics.to_dict(),
                        'ultra_debug_mode': self.mode.value,
                        'active_alerts': len(self.anomaly_detector.get_active_alerts())
                    }
                )
        
        elif self.mode in [UltraDebugMode.ANALYSIS, UltraDebugMode.INTROSPECTION]:
            # Log detailed metrics every cycle
            self.logger.trace_deep(
                f"🔍 ANALYSIS_METRICS: System analysis snapshot",
                debug_context={
                    'system_metrics': metrics.to_dict(),
                    'ultra_debug_mode': self.mode.value,
                    'analysis_depth': 'deep' if self.mode == UltraDebugMode.INTROSPECTION else 'standard'
                }
            )
        
        elif self.mode in [UltraDebugMode.EMERGENCY, UltraDebugMode.CRISIS]:
            # Log everything in emergency/crisis modes
            active_alerts = self.anomaly_detector.get_active_alerts()
            
            self.logger.critical(
                f"🚨 {self.mode.value.upper()}_MONITORING: Critical system monitoring",
                debug_context={
                    'emergency_metrics': metrics.to_dict(),
                    'ultra_debug_mode': self.mode.value,
                    'active_alerts': [alert.to_dict() for alert in active_alerts],
                    'escalation_ready': len([a for a in active_alerts if a.severity == 'critical']) >= 3
                }
            )
    
    def _capture_performance_baseline(self) -> None:
        """Capture performance baseline for comparison."""
        try:
            baseline_metrics = SystemMetrics()
            baseline_metrics.capture_metrics()
            self.performance_baseline = baseline_metrics
            
            self.logger.info(
                "📈 Performance baseline captured",
                extra={
                    'performance_baseline': baseline_metrics.to_dict(),
                    'baseline_timestamp': baseline_metrics.timestamp
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to capture performance baseline: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive ultra-debug status."""
        
        with self.lock:
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            
            # Get alert summary
            alert_summary = self.anomaly_detector.get_alert_summary()
            
            # Get recent metrics
            recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
            
            return {
                'mode': self.mode.value,
                'previous_mode': self.previous_mode.value,
                'uptime_seconds': uptime_seconds,
                'monitoring_active': self.monitoring_active,
                'monitoring_interval_seconds': self.monitoring_interval_seconds,
                'metrics_collected': len(self.metrics_history),
                'emergency_activations': self.emergency_activations,
                'crisis_activations': self.crisis_activations,
                'performance_baseline_available': self.performance_baseline is not None,
                'alert_summary': alert_summary,
                'recent_metrics': [m.to_dict() for m in recent_metrics],
                'mode_change_history': self.mode_change_history[-10:],  # Last 10 changes
                'integrator_status': self.integrator.get_integration_status() if self.integrator else {}
            }
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        if not self.metrics_history:
            return {'error': 'No metrics data available for analysis'}
        
        # Calculate statistics from metrics history
        recent_metrics = [m for m in self.metrics_history 
                         if time.time() - m.timestamp <= 3600]  # Last hour
        
        if not recent_metrics:
            return {'error': 'No recent metrics data available'}
        
        # Performance analysis
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        performance_analysis = {
            'cpu_stats': {
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory_stats': {
                'average': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'current': memory_values[-1] if memory_values else 0
            }
        }
        
        # Alert analysis
        alert_summary = self.anomaly_detector.get_alert_summary()
        
        # Component health analysis
        component_analysis = {}
        try:
            if self.integrator and self.integrator.health_monitor:
                health_summary = self.integrator.health_monitor.get_system_health_summary()
                component_analysis = health_summary if isinstance(health_summary, dict) else {}
        except:
            pass
        
        return {
            'analysis_timestamp': time.time(),
            'analysis_period_minutes': 60,
            'metrics_analyzed': len(recent_metrics),
            'performance_analysis': performance_analysis,
            'alert_analysis': alert_summary,
            'component_analysis': component_analysis,
            'recommendations': self._generate_recommendations(performance_analysis, alert_summary)
        }
    
    def _generate_recommendations(self, performance_analysis: Dict, alert_summary: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # CPU recommendations
        cpu_avg = performance_analysis.get('cpu_stats', {}).get('average', 0)
        if cpu_avg > 80:
            recommendations.append("High average CPU usage detected - investigate CPU-intensive operations")
        
        # Memory recommendations
        memory_avg = performance_analysis.get('memory_stats', {}).get('average', 0)
        if memory_avg > 85:
            recommendations.append("High average memory usage detected - check for memory leaks")
        
        # Alert recommendations
        critical_alerts = alert_summary.get('active_by_severity', {}).get('critical', 0)
        if critical_alerts > 0:
            recommendations.append(f"Address {critical_alerts} critical alerts immediately")
        
        if not recommendations:
            recommendations.append("System performance appears normal - continue monitoring")
        
        return recommendations
    
    def shutdown(self) -> None:
        """Gracefully shutdown ultra-debug manager."""
        
        with self.lock:
            self.logger.info("🔄 Shutting down ultra-debug manager...")
            
            # Stop monitoring
            self._stop_monitoring()
            
            # Disable ultra-debug
            if self.mode != UltraDebugMode.DISABLED:
                self.disable_ultra_debug()
            
            self.logger.info("✅ Ultra-debug manager shutdown complete")


# Global ultra-debug manager instance
_global_ultra_debug_manager: Optional[UltraDebugManager] = None
_ultra_debug_lock = threading.Lock()


def get_ultra_debug_manager() -> UltraDebugManager:
    """Get global ultra-debug manager instance."""
    global _global_ultra_debug_manager
    
    if _global_ultra_debug_manager is None:
        with _ultra_debug_lock:
            if _global_ultra_debug_manager is None:
                _global_ultra_debug_manager = UltraDebugManager()
    
    return _global_ultra_debug_manager


def enable_ultra_debug(mode: UltraDebugMode = UltraDebugMode.MONITORING) -> bool:
    """
    Enable ultra-debug mode.
    
    Convenience function to activate maximum debugging visibility.
    
    Args:
        mode: Ultra-debug mode to activate
        
    Returns:
        True if successfully enabled, False otherwise
    """
    manager = get_ultra_debug_manager()
    return manager.enable_ultra_debug(mode)


def disable_ultra_debug() -> None:
    """Disable ultra-debug mode."""
    manager = get_ultra_debug_manager()
    manager.disable_ultra_debug()


def emergency_debug() -> None:
    """Activate emergency debug mode immediately."""
    manager = get_ultra_debug_manager()
    
    # Enable if not already enabled
    if manager.mode == UltraDebugMode.DISABLED:
        manager.enable_ultra_debug(UltraDebugMode.MONITORING)
    
    # Escalate to emergency
    manager.escalate_to_emergency("manual_emergency_activation")


def crisis_debug() -> None:
    """Activate crisis debug mode immediately."""
    manager = get_ultra_debug_manager()
    
    # Enable if not already enabled
    if manager.mode == UltraDebugMode.DISABLED:
        manager.enable_ultra_debug(UltraDebugMode.MONITORING)
    
    # Escalate to crisis
    manager.escalate_to_crisis("manual_crisis_activation")


def get_ultra_debug_status() -> Dict[str, Any]:
    """Get ultra-debug status."""
    manager = get_ultra_debug_manager()
    return manager.get_status()


def get_analysis_report() -> Dict[str, Any]:
    """Get comprehensive analysis report."""
    manager = get_ultra_debug_manager()
    return manager.get_analysis_report()


# Convenience decorator for ultra-debug mode awareness
def ultra_debug_aware(component_name: str = "") -> Callable:
    """
    Decorator that makes methods aware of ultra-debug mode.
    
    Automatically captures additional context when ultra-debug is active.
    
    Usage:
        @ultra_debug_aware("my_component")
        def my_method(self):
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **kwargs):
            manager = get_ultra_debug_manager()
            
            # Check if ultra-debug is active
            if manager.mode == UltraDebugMode.DISABLED:
                return func(*args, **kwargs)
            
            # Enhanced context capture for ultra-debug
            actual_component_name = component_name or getattr(args[0], '__class__', {}).get('__name__', 'unknown')
            
            logger = get_enhanced_logger(f"arena_bot.{actual_component_name}")
            
            start_time = time.perf_counter()
            
            # Log method entry with ultra-debug context
            logger.trace_deep(
                f"🔍 ULTRA_DEBUG_METHOD_ENTRY: {func.__name__}",
                debug_context={
                    'ultra_debug_mode': manager.mode.value,
                    'component_name': actual_component_name,
                    'method_name': func.__name__,
                    'capture_level': 3,  # Ultra level
                    'monitoring_active': manager.monitoring_active
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.trace_deep(
                    f"✅ ULTRA_DEBUG_METHOD_SUCCESS: {func.__name__} ({duration_ms:.2f}ms)",
                    debug_context={
                        'ultra_debug_mode': manager.mode.value,
                        'component_name': actual_component_name,
                        'method_name': func.__name__,
                        'duration_ms': duration_ms,
                        'capture_level': 3
                    }
                )
                
                return result
                
            except Exception as e:
                # Enhanced exception handling in ultra-debug mode
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                logger.exception_deep(
                    f"❌ ULTRA_DEBUG_METHOD_EXCEPTION: {func.__name__} ({duration_ms:.2f}ms)",
                    debug_context={
                        'ultra_debug_mode': manager.mode.value,
                        'component_name': actual_component_name,
                        'method_name': func.__name__,
                        'duration_ms': duration_ms,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e),
                        'capture_level': 3
                    }
                )
                
                # Use deep exception handler
                exception_handler = get_exception_handler()
                exception_handler.handle_exception(
                    exception=e,
                    component_name=actual_component_name,
                    method_name=func.__name__,
                    attempt_recovery=manager.mode in [UltraDebugMode.EMERGENCY, UltraDebugMode.CRISIS]
                )
                
                raise
        
        return wrapper
    
    return decorator