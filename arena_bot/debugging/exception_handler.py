"""
Deep Exception Context Capture System for Arena Bot Deep Debugging

Comprehensive exception handling with full system state capture:
- Complete system snapshot on any exception occurrence
- Multi-threaded stack trace analysis with thread state capture
- Memory profiling and resource usage at exception time
- Component state analysis and dependency chain investigation
- Exception correlation across distributed operations
- Automated root cause analysis using captured context
- Recovery suggestion engine based on exception patterns

Integrates with existing debugging infrastructure for complete visibility.
"""

import time
import threading
import traceback
import sys
import gc
import psutil
import linecache
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4
from collections import defaultdict

from ..logging_system.logger import get_logger, LogLevel


class ExceptionSeverity(Enum):
    """Severity levels for exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class ThreadSnapshot:
    """
    Snapshot of a thread's state at exception time.
    """
    
    # Thread identification
    thread_id: int
    thread_name: str
    is_daemon: bool
    is_alive: bool
    
    # Thread state
    stack_trace: str = ""
    local_variables: Dict[str, str] = field(default_factory=dict)
    
    # Performance data
    cpu_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'thread_id': self.thread_id,
            'thread_name': self.thread_name,
            'is_daemon': self.is_daemon,
            'is_alive': self.is_alive,
            'stack_trace': self.stack_trace,
            'local_variables': self.local_variables,
            'cpu_percent': self.cpu_percent
        }


@dataclass
class SystemSnapshot:
    """
    Complete system state snapshot at exception time.
    """
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    # System resources
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Process information
    process_id: int = 0
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    process_threads: int = 0
    open_files: int = 0
    
    # Threading information
    active_threads: List[ThreadSnapshot] = field(default_factory=list)
    deadlocked_threads: List[int] = field(default_factory=list)
    
    # Memory analysis
    object_counts: Dict[str, int] = field(default_factory=dict)
    memory_leaks: List[str] = field(default_factory=list)
    
    # Network and I/O
    network_connections: int = 0
    io_counters: Dict[str, int] = field(default_factory=dict)
    
    def capture_system_state(self) -> None:
        """Capture current system state."""
        try:
            # System resources
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            self.memory_available_mb = memory.available / 1024 / 1024
            
            disk = psutil.disk_usage('/')
            self.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Current process
            process = psutil.Process()
            self.process_id = process.pid
            process_memory = process.memory_info()
            self.process_memory_mb = process_memory.rss / 1024 / 1024
            self.process_cpu_percent = process.cpu_percent()
            self.process_threads = process.num_threads()
            
            try:
                self.open_files = len(process.open_files())
            except:
                self.open_files = 0
            
            # Network connections
            try:
                self.network_connections = len(process.connections())
            except:
                self.network_connections = 0
            
            # I/O counters
            try:
                io = process.io_counters()
                self.io_counters = {
                    'read_count': io.read_count,
                    'write_count': io.write_count,
                    'read_bytes': io.read_bytes,
                    'write_bytes': io.write_bytes
                }
            except:
                pass
            
        except Exception as e:
            # Don't let system capture failure affect exception handling
            pass
    
    def capture_threading_state(self) -> None:
        """Capture threading state and detect deadlocks."""
        try:
            # Get all threads
            for thread in threading.enumerate():
                thread_snapshot = ThreadSnapshot(
                    thread_id=thread.ident if thread.ident else 0,
                    thread_name=thread.name,
                    is_daemon=thread.daemon,
                    is_alive=thread.is_alive()
                )
                
                # Try to get stack trace for thread
                try:
                    frame = sys._current_frames().get(thread.ident)
                    if frame:
                        thread_snapshot.stack_trace = ''.join(
                            traceback.format_stack(frame)
                        )
                        
                        # Capture local variables (safely)
                        try:
                            for var_name, var_value in frame.f_locals.items():
                                if not var_name.startswith('_'):
                                    thread_snapshot.local_variables[var_name] = self._safe_repr(var_value)
                        except:
                            pass
                except:
                    pass
                
                self.active_threads.append(thread_snapshot)
            
            # Simple deadlock detection (basic heuristic)
            # In a real implementation, this would be more sophisticated
            long_running_threads = [
                t for t in self.active_threads 
                if t.is_alive and not t.is_daemon and 'sleep' not in t.stack_trace.lower()
            ]
            
            if len(long_running_threads) > 1:
                # Potential deadlock if multiple threads are blocked
                blocked_threads = [
                    t for t in long_running_threads
                    if any(keyword in t.stack_trace.lower() for keyword in ['lock', 'wait', 'acquire'])
                ]
                
                if len(blocked_threads) > 1:
                    self.deadlocked_threads = [t.thread_id for t in blocked_threads]
        
        except Exception:
            pass
    
    def capture_memory_analysis(self) -> None:
        """Capture memory analysis including potential leaks."""
        try:
            # Object counts by type
            object_counts = defaultdict(int)
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] += 1
            
            # Keep only significant object types
            self.object_counts = {
                k: v for k, v in object_counts.items() 
                if v > 100  # Only types with many instances
            }
            
            # Simple leak detection (heuristic)
            suspicious_counts = [
                f"{obj_type}: {count}" 
                for obj_type, count in self.object_counts.items()
                if count > 1000  # Arbitrary threshold
            ]
            
            if suspicious_counts:
                self.memory_leaks = suspicious_counts
        
        except Exception:
            pass
    
    def _safe_repr(self, obj: Any, max_length: int = 100) -> str:
        """Safely represent an object as string."""
        try:
            repr_str = repr(obj)
            if len(repr_str) > max_length:
                return f"{repr_str[:max_length]}..."
            return repr_str
        except:
            return f"<{type(obj).__name__}:repr_error>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'process_id': self.process_id,
            'process_memory_mb': self.process_memory_mb,
            'process_cpu_percent': self.process_cpu_percent,
            'process_threads': self.process_threads,
            'open_files': self.open_files,
            'network_connections': self.network_connections,
            'io_counters': self.io_counters,
            'active_threads': [t.to_dict() for t in self.active_threads],
            'deadlocked_threads': self.deadlocked_threads,
            'object_counts': self.object_counts,
            'memory_leaks': self.memory_leaks
        }


@dataclass
class ComponentStateSnapshot:
    """
    Snapshot of component states at exception time.
    """
    
    # Component health states
    component_health: Dict[str, Any] = field(default_factory=dict)
    
    # Active operations
    active_traces: List[str] = field(default_factory=list)
    active_pipelines: List[str] = field(default_factory=list)
    
    # Circuit breaker states
    circuit_breaker_states: Dict[str, Any] = field(default_factory=dict)
    
    # Recent state changes
    recent_state_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    def capture_component_states(self) -> None:
        """Capture current component states."""
        try:
            # Import debugging components (avoid circular imports)
            from . import get_health_monitor, get_state_monitor, get_pipeline_tracer, get_method_tracer
            
            # Health monitor state
            health_monitor = get_health_monitor()
            self.component_health = health_monitor.get_system_health_summary()
            
            # Circuit breaker states
            for name, breaker in health_monitor.circuit_breakers.items():
                self.circuit_breaker_states[name] = breaker.get_stats()
            
            # State monitor information
            state_monitor = get_state_monitor()
            self.recent_state_changes = [
                event.to_dict() for event in state_monitor.get_recent_changes(seconds=300)  # Last 5 minutes
            ]
            
            # Pipeline tracer information
            pipeline_tracer = get_pipeline_tracer()
            self.active_pipelines = pipeline_tracer.get_active_pipelines()
            
            # Method tracer information
            method_tracer = get_method_tracer()
            self.active_traces = [
                trace.correlation_id for trace in method_tracer.get_active_traces()
            ]
            
            # Performance statistics
            self.performance_stats = {
                'health_monitor': health_monitor.get_performance_stats() if hasattr(health_monitor, 'get_performance_stats') else {},
                'state_monitor': state_monitor.get_performance_stats(),
                'pipeline_tracer': pipeline_tracer.get_performance_stats(),
                'method_tracer': method_tracer.get_performance_stats()
            }
            
        except Exception as e:
            # Don't let component state capture failure affect exception handling
            self.component_health = {'capture_error': str(e)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'component_health': self.component_health,
            'active_traces': self.active_traces,
            'active_pipelines': self.active_pipelines,
            'circuit_breaker_states': self.circuit_breaker_states,
            'recent_state_changes': self.recent_state_changes,
            'performance_stats': self.performance_stats
        }


@dataclass
class ExceptionContext:
    """
    Complete context captured when an exception occurs.
    
    Provides comprehensive information about the system state,
    component states, and execution context at the time of the exception.
    """
    
    # Exception identification
    exception_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Exception information
    exception_type: str = ""
    exception_message: str = ""
    exception_traceback: str = ""
    
    # Context information
    correlation_id: Optional[str] = None
    component_name: str = ""
    method_name: str = ""
    operation_name: str = ""
    
    # Severity and classification
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM
    category: str = "unknown"
    
    # System snapshots
    system_snapshot: SystemSnapshot = field(default_factory=SystemSnapshot)
    component_snapshot: ComponentStateSnapshot = field(default_factory=ComponentStateSnapshot)
    
    # Code context
    source_file: str = ""
    source_line: int = 0
    source_context: List[str] = field(default_factory=list)
    
    # Variables and state
    local_variables: Dict[str, str] = field(default_factory=dict)
    global_variables: Dict[str, str] = field(default_factory=dict)
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: str = ""
    recovery_time_ms: Optional[float] = None
    
    # Analysis results
    root_cause_analysis: List[str] = field(default_factory=list)
    related_exceptions: List[str] = field(default_factory=list)
    remediation_suggestions: List[str] = field(default_factory=list)
    
    def capture_exception_details(self, exception: Exception, tb: Any = None) -> None:
        """Capture detailed exception information."""
        self.exception_type = type(exception).__name__
        self.exception_message = str(exception)
        
        # Get traceback
        if tb is None:
            tb = exception.__traceback__
        
        if tb:
            self.exception_traceback = ''.join(traceback.format_tb(tb))
            
            # Get source context
            try:
                frame = tb.tb_frame
                self.source_file = frame.f_code.co_filename
                self.source_line = tb.tb_lineno
                
                # Get surrounding source lines
                try:
                    lines = []
                    for line_num in range(max(1, self.source_line - 3), self.source_line + 4):
                        line = linecache.getline(self.source_file, line_num).rstrip()
                        marker = " -> " if line_num == self.source_line else "    "
                        lines.append(f"{line_num:4d}{marker}{line}")
                    self.source_context = lines
                except:
                    pass
                
                # Capture local variables
                try:
                    for var_name, var_value in frame.f_locals.items():
                        if not var_name.startswith('_'):
                            self.local_variables[var_name] = self._safe_repr(var_value)
                except:
                    pass
                
                # Capture relevant global variables
                try:
                    for var_name, var_value in frame.f_globals.items():
                        if (not var_name.startswith('_') and 
                            var_name in ['logger', 'config', 'DEBUG', 'ENABLE_DEBUG']):
                            self.global_variables[var_name] = self._safe_repr(var_value)
                except:
                    pass
                
            except:
                pass
    
    def capture_system_context(self) -> None:
        """Capture complete system context."""
        # Capture system state
        self.system_snapshot.capture_system_state()
        self.system_snapshot.capture_threading_state()
        self.system_snapshot.capture_memory_analysis()
        
        # Capture component states
        self.component_snapshot.capture_component_states()
    
    def classify_severity(self) -> ExceptionSeverity:
        """Classify exception severity based on type and context."""
        exception_type_lower = self.exception_type.lower()
        
        # Critical exceptions
        if any(critical_type in exception_type_lower for critical_type in [
            'systemexit', 'keyboardinterrupt', 'memoryerror', 'stackoverflowerror'
        ]):
            self.severity = ExceptionSeverity.CRITICAL
        
        # High severity exceptions
        elif any(high_type in exception_type_lower for high_type in [
            'runtimeerror', 'connectionerror', 'timeouterror', 'ioerror'
        ]):
            self.severity = ExceptionSeverity.HIGH
        
        # Medium severity exceptions
        elif any(medium_type in exception_type_lower for medium_type in [
            'valueerror', 'typeerror', 'attributeerror', 'indexerror'
        ]):
            self.severity = ExceptionSeverity.MEDIUM
        
        # Low severity exceptions
        else:
            self.severity = ExceptionSeverity.LOW
        
        # Adjust based on component
        if self.component_name in ['detection_engine', 'ai_advisor', 'gui']:
            # Critical components get higher severity
            if self.severity == ExceptionSeverity.LOW:
                self.severity = ExceptionSeverity.MEDIUM
            elif self.severity == ExceptionSeverity.MEDIUM:
                self.severity = ExceptionSeverity.HIGH
        
        return self.severity
    
    def analyze_root_cause(self) -> List[str]:
        """Perform automated root cause analysis."""
        causes = []
        
        # Check for resource issues
        if self.system_snapshot.memory_percent > 90:
            causes.append("High memory usage (>90%) may be causing the exception")
        
        if self.system_snapshot.cpu_percent > 95:
            causes.append("High CPU usage (>95%) may be causing performance issues")
        
        # Check for threading issues
        if self.system_snapshot.deadlocked_threads:
            causes.append(f"Potential deadlock detected in threads: {self.system_snapshot.deadlocked_threads}")
        
        # Check for component health issues
        component_health = self.component_snapshot.component_health
        if isinstance(component_health, dict):
            status_counts = component_health.get('status_counts', {})
            if status_counts.get('failed', 0) > 0:
                causes.append(f"Failed components detected: {status_counts.get('failed')} components")
            if status_counts.get('critical', 0) > 0:
                causes.append(f"Critical component health issues: {status_counts.get('critical')} components")
        
        # Check for memory leaks
        if self.system_snapshot.memory_leaks:
            causes.append(f"Potential memory leaks detected: {', '.join(self.system_snapshot.memory_leaks[:3])}")
        
        # Check circuit breaker states
        open_breakers = [
            name for name, stats in self.component_snapshot.circuit_breaker_states.items()
            if stats.get('state') == 'open'
        ]
        if open_breakers:
            causes.append(f"Open circuit breakers: {', '.join(open_breakers)}")
        
        self.root_cause_analysis = causes
        return causes
    
    def generate_remediation_suggestions(self) -> List[str]:
        """Generate remediation suggestions based on analysis."""
        suggestions = []
        
        # Based on exception type
        if 'memory' in self.exception_type.lower():
            suggestions.extend([
                "Monitor memory usage and implement garbage collection",
                "Check for memory leaks in component initialization",
                "Consider reducing cache sizes or implementing LRU eviction"
            ])
        
        elif 'timeout' in self.exception_type.lower():
            suggestions.extend([
                "Increase timeout values for long-running operations",
                "Implement retry mechanisms with exponential backoff",
                "Check network connectivity and service availability"
            ])
        
        elif 'connection' in self.exception_type.lower():
            suggestions.extend([
                "Verify network connectivity and service endpoints",
                "Implement connection pooling and retry logic",
                "Check firewall and security settings"
            ])
        
        # Based on root cause analysis
        for cause in self.root_cause_analysis:
            if "memory usage" in cause:
                suggestions.append("Implement memory monitoring and cleanup procedures")
            elif "deadlock" in cause:
                suggestions.append("Review threading code and implement deadlock detection")
            elif "circuit breaker" in cause:
                suggestions.append("Reset circuit breakers after fixing underlying issues")
        
        # Component-specific suggestions
        if self.component_name == "detection_engine":
            suggestions.extend([
                "Verify screenshot quality and card detection parameters",
                "Check template and histogram databases for corruption",
                "Review coordinate detection accuracy"
            ])
        elif self.component_name == "ai_advisor":
            suggestions.extend([
                "Check AI model availability and response times",
                "Verify input data format and quality",
                "Review AI processing timeouts and retries"
            ])
        
        self.remediation_suggestions = suggestions
        return suggestions
    
    def _safe_repr(self, obj: Any, max_length: int = 200) -> str:
        """Safely represent an object as string."""
        try:
            repr_str = repr(obj)
            if len(repr_str) > max_length:
                return f"{repr_str[:max_length]}..."
            return repr_str
        except:
            return f"<{type(obj).__name__}:repr_error>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'exception_id': self.exception_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'exception_traceback': self.exception_traceback,
            'correlation_id': self.correlation_id,
            'component_name': self.component_name,
            'method_name': self.method_name,
            'operation_name': self.operation_name,
            'severity': self.severity.value,
            'category': self.category,
            'system_snapshot': self.system_snapshot.to_dict(),
            'component_snapshot': self.component_snapshot.to_dict(),
            'source_file': self.source_file,
            'source_line': self.source_line,
            'source_context': self.source_context,
            'local_variables': self.local_variables,
            'global_variables': self.global_variables,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_method': self.recovery_method,
            'recovery_time_ms': self.recovery_time_ms,
            'root_cause_analysis': self.root_cause_analysis,
            'related_exceptions': self.related_exceptions,
            'remediation_suggestions': self.remediation_suggestions
        }


class DeepExceptionHandler:
    """
    Deep exception handler with comprehensive context capture.
    
    Provides enterprise-grade exception handling with complete system
    state capture, automated analysis, and recovery suggestions.
    """
    
    def __init__(self):
        """Initialize deep exception handler."""
        self.logger = get_logger("arena_bot.debugging.exception_handler")
        
        # Configuration
        self.enabled = True
        self.capture_full_context = True
        self.auto_analyze = True
        self.max_contexts_stored = 1000
        
        # Storage
        self.exception_contexts: List[ExceptionContext] = []
        self.contexts_by_component: Dict[str, List[ExceptionContext]] = defaultdict(list)
        
        # Statistics
        self.total_exceptions_handled = 0
        self.exceptions_by_severity: Dict[ExceptionSeverity, int] = defaultdict(int)
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def handle_exception(self,
                        exception: Exception,
                        component_name: str = "",
                        method_name: str = "",
                        operation_name: str = "",
                        correlation_id: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None,
                        attempt_recovery: bool = True) -> ExceptionContext:
        """
        Handle an exception with deep context capture.
        
        Args:
            exception: The exception that occurred
            component_name: Name of the component where exception occurred
            method_name: Name of the method where exception occurred
            operation_name: Name of the operation being performed
            correlation_id: Correlation ID for tracing
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            ExceptionContext with complete analysis
        """
        
        if not self.enabled:
            return None
        
        start_time = time.perf_counter()
        
        with self.lock:
            # Create exception context
            exception_context = ExceptionContext(
                correlation_id=correlation_id,
                component_name=component_name,
                method_name=method_name,
                operation_name=operation_name
            )
            
            # Capture exception details
            exception_context.capture_exception_details(exception)
            
            # Capture system context if enabled
            if self.capture_full_context:
                exception_context.capture_system_context()
            
            # Classify severity
            exception_context.classify_severity()
            
            # Perform analysis if enabled
            if self.auto_analyze:
                exception_context.analyze_root_cause()
                exception_context.generate_remediation_suggestions()
            
            # Store context
            self._store_exception_context(exception_context)
            
            # Update statistics
            self.total_exceptions_handled += 1
            self.exceptions_by_severity[exception_context.severity] += 1
            
            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Log the exception with full context
            self._log_exception_context(exception_context, processing_time)
            
            # Attempt recovery if requested
            if attempt_recovery and exception_context.severity != ExceptionSeverity.FATAL:
                self._attempt_recovery(exception_context)
            
            return exception_context
    
    def _store_exception_context(self, context: ExceptionContext) -> None:
        """Store exception context with size management."""
        # Add to main storage
        self.exception_contexts.append(context)
        
        # Add to component-specific storage
        self.contexts_by_component[context.component_name].append(context)
        
        # Manage storage size
        if len(self.exception_contexts) > self.max_contexts_stored:
            # Remove oldest contexts
            removed_context = self.exception_contexts.pop(0)
            
            # Also remove from component storage
            component_contexts = self.contexts_by_component[removed_context.component_name]
            if removed_context in component_contexts:
                component_contexts.remove(removed_context)
    
    def _log_exception_context(self, context: ExceptionContext, processing_time: float) -> None:
        """Log exception context with appropriate level."""
        
        # Determine log level based on severity
        log_level_map = {
            ExceptionSeverity.LOW: LogLevel.WARNING,
            ExceptionSeverity.MEDIUM: LogLevel.ERROR,
            ExceptionSeverity.HIGH: LogLevel.ERROR,
            ExceptionSeverity.CRITICAL: LogLevel.CRITICAL,
            ExceptionSeverity.FATAL: LogLevel.CRITICAL
        }
        
        log_level = log_level_map[context.severity]
        
        # Create log message
        message = (
            f"🚨 DEEP_EXCEPTION: {context.exception_type} in {context.component_name} "
            f"(severity: {context.severity.value}, analysis: {processing_time:.2f}ms)"
        )
        
        # Log with full context
        self.logger.log(
            log_level,
            message,
            extra={
                'exception_context': context.to_dict(),
                'deep_analysis': {
                    'processing_time_ms': processing_time,
                    'context_capture_enabled': self.capture_full_context,
                    'auto_analysis_enabled': self.auto_analyze,
                    'root_causes_found': len(context.root_cause_analysis),
                    'suggestions_generated': len(context.remediation_suggestions)
                }
            }
        )
        
        # Log root cause analysis separately for visibility
        if context.root_cause_analysis:
            self.logger.warning(
                f"🔍 ROOT_CAUSE_ANALYSIS: {context.exception_id[:8]}",
                extra={
                    'root_cause_analysis': {
                        'exception_id': context.exception_id,
                        'causes': context.root_cause_analysis,
                        'suggestions': context.remediation_suggestions
                    }
                }
            )
    
    def _attempt_recovery(self, context: ExceptionContext) -> None:
        """Attempt automatic recovery based on exception context."""
        self.recovery_attempts += 1
        context.recovery_attempted = True
        
        recovery_start = time.perf_counter()
        
        try:
            # Try component-specific recovery strategies
            component_strategy = self.recovery_strategies.get(context.component_name)
            if component_strategy:
                recovery_successful = component_strategy(context)
                if recovery_successful:
                    context.recovery_successful = True
                    context.recovery_method = f"component_strategy_{context.component_name}"
                    self.successful_recoveries += 1
            
            # Try exception-type specific recovery
            if not context.recovery_successful:
                exception_strategy = self.recovery_strategies.get(context.exception_type)
                if exception_strategy:
                    recovery_successful = exception_strategy(context)
                    if recovery_successful:
                        context.recovery_successful = True
                        context.recovery_method = f"exception_strategy_{context.exception_type}"
                        self.successful_recoveries += 1
            
            # Try generic recovery strategies
            if not context.recovery_successful:
                recovery_successful = self._try_generic_recovery(context)
                if recovery_successful:
                    context.recovery_successful = True
                    context.recovery_method = "generic_strategy"
                    self.successful_recoveries += 1
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
        
        finally:
            context.recovery_time_ms = (time.perf_counter() - recovery_start) * 1000
            
            if context.recovery_successful:
                self.logger.info(
                    f"✅ RECOVERY_SUCCESS: {context.exception_id[:8]} recovered using {context.recovery_method}",
                    extra={
                        'recovery_success': {
                            'exception_id': context.exception_id,
                            'recovery_method': context.recovery_method,
                            'recovery_time_ms': context.recovery_time_ms
                        }
                    }
                )
    
    def _try_generic_recovery(self, context: ExceptionContext) -> bool:
        """Try generic recovery strategies."""
        
        # Memory-related recovery
        if 'memory' in context.exception_type.lower():
            try:
                import gc
                gc.collect()
                return True
            except:
                pass
        
        # Connection-related recovery
        elif 'connection' in context.exception_type.lower():
            # In a real implementation, this might reset connections
            return False
        
        # Timeout-related recovery
        elif 'timeout' in context.exception_type.lower():
            # In a real implementation, this might adjust timeouts
            return False
        
        return False
    
    def add_recovery_strategy(self, 
                            trigger: str,
                            strategy: Callable[[ExceptionContext], bool]) -> None:
        """
        Add a recovery strategy.
        
        Args:
            trigger: Component name or exception type to trigger on
            strategy: Callable that takes ExceptionContext and returns success boolean
        """
        self.recovery_strategies[trigger] = strategy
        
        self.logger.info(f"🔧 Recovery strategy added for: {trigger}")
    
    def get_exception_contexts(self,
                             component_name: Optional[str] = None,
                             severity: Optional[ExceptionSeverity] = None,
                             hours: int = 24,
                             limit: int = 100) -> List[ExceptionContext]:
        """
        Get exception contexts with filtering.
        
        Args:
            component_name: Filter by component name
            severity: Filter by severity level
            hours: Hours to look back
            limit: Maximum number of contexts to return
            
        Returns:
            List of matching exception contexts
        """
        
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            if component_name:
                contexts = self.contexts_by_component[component_name].copy()
            else:
                contexts = self.exception_contexts.copy()
        
        # Apply filters
        filtered_contexts = []
        for context in contexts:
            if context.timestamp < cutoff_time:
                continue
            
            if severity and context.severity != severity:
                continue
            
            filtered_contexts.append(context)
        
        # Sort by timestamp (most recent first)
        filtered_contexts.sort(key=lambda c: c.timestamp, reverse=True)
        
        return filtered_contexts[:limit] if limit > 0 else filtered_contexts
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get exception handling performance statistics."""
        
        with self.lock:
            recovery_rate = (
                self.successful_recoveries / self.recovery_attempts
                if self.recovery_attempts > 0 else 0
            )
            
            return {
                'total_exceptions_handled': self.total_exceptions_handled,
                'exceptions_by_severity': {
                    sev.value: count for sev, count in self.exceptions_by_severity.items()
                },
                'recovery_attempts': self.recovery_attempts,
                'successful_recoveries': self.successful_recoveries,
                'recovery_rate': recovery_rate,
                'stored_contexts': len(self.exception_contexts),
                'enabled': self.enabled,
                'capture_full_context': self.capture_full_context,
                'auto_analyze': self.auto_analyze,
                'recovery_strategies': len(self.recovery_strategies)
            }
    
    def enable(self) -> None:
        """Enable deep exception handling."""
        self.enabled = True
        self.logger.info("🚨 Deep exception handling enabled")
    
    def disable(self) -> None:
        """Disable deep exception handling."""
        self.enabled = False
        self.logger.info("🚨 Deep exception handling disabled")


# Global deep exception handler instance
_global_exception_handler: Optional[DeepExceptionHandler] = None
_handler_lock = threading.Lock()


def get_exception_handler() -> DeepExceptionHandler:
    """Get global deep exception handler instance."""
    global _global_exception_handler
    
    if _global_exception_handler is None:
        with _handler_lock:
            if _global_exception_handler is None:
                _global_exception_handler = DeepExceptionHandler()
    
    return _global_exception_handler


def handle_exception(exception: Exception,
                    component_name: str = "",
                    method_name: str = "",
                    **kwargs) -> ExceptionContext:
    """
    Convenience function to handle exceptions with deep context capture.
    
    Usage:
        try:
            # Your code here
            pass
        except Exception as e:
            context = handle_exception(e, component_name="my_component", method_name="my_method")
            # Optional: re-raise or handle based on context
            raise
    """
    handler = get_exception_handler()
    return handler.handle_exception(
        exception=exception,
        component_name=component_name,
        method_name=method_name,
        **kwargs
    )


def deep_exception_handler(component_name: str = "", 
                          method_name: str = "",
                          attempt_recovery: bool = True) -> Callable:
    """
    Decorator for automatic deep exception handling.
    
    Usage:
        @deep_exception_handler(component_name="my_component", attempt_recovery=True)
        def my_method(self):
            # Method implementation
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle exception with deep context capture
                actual_component_name = component_name or getattr(args[0], '__class__', {}).get('__name__', 'unknown')
                actual_method_name = method_name or func.__name__
                
                context = handle_exception(
                    exception=e,
                    component_name=actual_component_name,
                    method_name=actual_method_name,
                    attempt_recovery=attempt_recovery
                )
                
                # Re-raise exception (context is logged)
                raise
        
        return wrapper
    
    return decorator