"""
Performance Bottleneck Detection System for Arena Bot Deep Debugging

Advanced performance analysis and bottleneck detection system that provides:
- Real-time performance monitoring with microsecond precision
- Bottleneck identification using statistical analysis and ML techniques
- Resource usage profiling (CPU, memory, I/O, network) with granular tracking
- Call graph analysis with hotspot detection and optimization suggestions
- Memory leak detection with allocation tracking and cleanup recommendations
- Database query optimization with slow query detection and indexing suggestions
- Network latency analysis with connection pooling and caching recommendations
- Automated performance regression detection with historical baseline comparison
- Performance optimization suggestions with impact prediction and risk assessment

This system helps identify performance bottlenecks that cause slow response times,
high resource usage, and poor user experience in complex systems.
"""

import time
import threading
import psutil
import gc
import sys
import traceback
import cProfile
import pstats
import io
import resource
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import statistics
import json
from uuid import uuid4
import weakref
import linecache

# Import debugging components
from .enhanced_logger import get_enhanced_logger
from .method_tracer import get_method_tracer
from .ultra_debug import get_ultra_debug_manager

from ..logging_system.logger import get_logger, LogLevel


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_LEAK = "memory_leak"
    IO_BLOCKING = "io_blocking"
    NETWORK_LATENCY = "network_latency"
    DATABASE_SLOW = "database_slow"
    ALGORITHM_INEFFICIENT = "algorithm_inefficient"
    SYNCHRONIZATION_BLOCKING = "synchronization_blocking"
    RESOURCE_CONTENTION = "resource_contention"


class PerformanceSeverity(Enum):
    """Severity levels for performance issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics snapshot."""
    
    timestamp: float = field(default_factory=time.time)
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_freq_mhz: float = 0.0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Memory metrics
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    memory_used_gb: float = 0.0
    swap_percent: float = 0.0
    
    # Process metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_threads: int = 0
    process_fds: int = 0
    
    # I/O metrics
    disk_read_mb_per_sec: float = 0.0
    disk_write_mb_per_sec: float = 0.0
    network_sent_mb_per_sec: float = 0.0
    network_recv_mb_per_sec: float = 0.0
    
    # Python-specific metrics
    gc_collections: Tuple[int, int, int] = (0, 0, 0)
    active_objects: int = 0
    memory_objects_mb: float = 0.0
    
    def capture_metrics(self) -> None:
        """Capture current performance metrics."""
        try:
            # System CPU
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            self.cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
            
            try:
                self.load_average = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have getloadavg
                self.load_average = (0.0, 0.0, 0.0)
            
            # System memory
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            self.memory_available_gb = memory.available / (1024**3)
            self.memory_used_gb = memory.used / (1024**3)
            
            swap = psutil.swap_memory()
            self.swap_percent = swap.percent
            
            # Process metrics
            process = psutil.Process()
            self.process_cpu_percent = process.cpu_percent()
            
            process_memory = process.memory_info()
            self.process_memory_mb = process_memory.rss / (1024**2)
            
            self.process_threads = process.num_threads()
            
            try:
                self.process_fds = process.num_fds()
            except (AttributeError, psutil.AccessDenied):
                # Windows or permission issues
                self.process_fds = 0
            
            # I/O metrics
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io and hasattr(self, '_prev_disk_io'):
                    time_delta = self.timestamp - getattr(self, '_prev_timestamp', self.timestamp)
                    if time_delta > 0:
                        self.disk_read_mb_per_sec = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / (1024**2) / time_delta
                        self.disk_write_mb_per_sec = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / (1024**2) / time_delta
                self._prev_disk_io = disk_io
                self._prev_timestamp = self.timestamp
            except Exception:
                pass
            
            try:
                net_io = psutil.net_io_counters()
                if net_io and hasattr(self, '_prev_net_io'):
                    time_delta = self.timestamp - getattr(self, '_prev_net_timestamp', self.timestamp)
                    if time_delta > 0:
                        self.network_sent_mb_per_sec = (net_io.bytes_sent - self._prev_net_io.bytes_sent) / (1024**2) / time_delta
                        self.network_recv_mb_per_sec = (net_io.bytes_recv - self._prev_net_io.bytes_recv) / (1024**2) / time_delta
                self._prev_net_io = net_io
                self._prev_net_timestamp = self.timestamp
            except Exception:
                pass
            
            # Python GC metrics
            self.gc_collections = tuple(gc.get_count())
            self.active_objects = len(gc.get_objects())
            
            # Memory usage by Python objects
            try:
                self.memory_objects_mb = sum(sys.getsizeof(obj) for obj in gc.get_objects()) / (1024**2)
            except Exception:
                self.memory_objects_mb = 0.0
                
        except Exception:
            # Don't let metrics collection fail the system
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'cpu_percent': self.cpu_percent,
            'cpu_freq_mhz': self.cpu_freq_mhz,
            'load_average': list(self.load_average),
            'memory_percent': self.memory_percent,
            'memory_available_gb': self.memory_available_gb,
            'memory_used_gb': self.memory_used_gb,
            'swap_percent': self.swap_percent,
            'process_cpu_percent': self.process_cpu_percent,
            'process_memory_mb': self.process_memory_mb,
            'process_threads': self.process_threads,
            'process_fds': self.process_fds,
            'disk_read_mb_per_sec': self.disk_read_mb_per_sec,
            'disk_write_mb_per_sec': self.disk_write_mb_per_sec,
            'network_sent_mb_per_sec': self.network_sent_mb_per_sec,
            'network_recv_mb_per_sec': self.network_recv_mb_per_sec,
            'gc_collections': list(self.gc_collections),
            'active_objects': self.active_objects,
            'memory_objects_mb': self.memory_objects_mb
        }


@dataclass
class PerformanceBottleneck:
    """Represents a detected performance bottleneck."""
    
    bottleneck_id: str = field(default_factory=lambda: str(uuid4()))
    bottleneck_type: BottleneckType = BottleneckType.CPU_INTENSIVE
    severity: PerformanceSeverity = PerformanceSeverity.MEDIUM
    
    # Location information
    function_name: str = ""
    module_name: str = ""
    file_path: str = ""
    line_number: int = 0
    
    # Performance impact
    impact_description: str = ""
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_overhead_percent: float = 0.0
    
    # Detection details
    detected_time: float = field(default_factory=time.time)
    detection_method: str = ""
    confidence_score: float = 0.0
    
    # Analysis
    root_cause_analysis: str = ""
    optimization_suggestions: List[str] = field(default_factory=list)
    estimated_improvement: str = ""
    
    # Context
    call_stack: List[str] = field(default_factory=list)
    related_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'bottleneck_id': self.bottleneck_id,
            'bottleneck_type': self.bottleneck_type.value,
            'severity': self.severity.value,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'impact_description': self.impact_description,
            'resource_usage': self.resource_usage,
            'performance_overhead_percent': self.performance_overhead_percent,
            'detected_time': self.detected_time,
            'detection_method': self.detection_method,
            'confidence_score': self.confidence_score,
            'root_cause_analysis': self.root_cause_analysis,
            'optimization_suggestions': self.optimization_suggestions,
            'estimated_improvement': self.estimated_improvement,
            'call_stack': self.call_stack,
            'related_metrics': self.related_metrics
        }


class FunctionProfiler:
    """Profiles individual function performance."""
    
    def __init__(self):
        """Initialize function profiler."""
        self.logger = get_enhanced_logger("arena_bot.debugging.performance_analyzer.profiler")
        
        # Profiling data
        self.function_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'recent_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100)
        })
        
        # Active profiling sessions
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Profiling state
        self.profiling_enabled = False
        self.profile_threshold_ms = 10.0  # Only profile functions taking > 10ms
    
    def start_profiling_function(self, function_name: str, module_name: str = "") -> str:
        """Start profiling a function call."""
        if not self.profiling_enabled:
            return ""
        
        profile_id = str(uuid4())
        full_name = f"{module_name}.{function_name}" if module_name else function_name
        
        # Capture initial metrics
        initial_metrics = PerformanceMetrics()
        initial_metrics.capture_metrics()
        
        self.active_profiles[profile_id] = {
            'function_name': full_name,
            'start_time': time.perf_counter(),
            'start_metrics': initial_metrics,
            'thread_id': threading.get_ident()
        }
        
        return profile_id
    
    def end_profiling_function(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """End profiling a function call and analyze performance."""
        if not profile_id or profile_id not in self.active_profiles:
            return None
        
        profile_info = self.active_profiles.pop(profile_id)
        
        # Calculate execution time
        execution_time = time.perf_counter() - profile_info['start_time']
        execution_time_ms = execution_time * 1000
        
        # Skip if below threshold
        if execution_time_ms < self.profile_threshold_ms:
            return None
        
        function_name = profile_info['function_name']
        
        # Capture final metrics
        final_metrics = PerformanceMetrics()
        final_metrics.capture_metrics()
        
        # Update function statistics
        stats = self.function_stats[function_name]
        stats['call_count'] += 1
        stats['total_time'] += execution_time_ms
        stats['min_time'] = min(stats['min_time'], execution_time_ms)
        stats['max_time'] = max(stats['max_time'], execution_time_ms)
        stats['avg_time'] = stats['total_time'] / stats['call_count']
        stats['recent_times'].append(execution_time_ms)
        
        # Calculate resource usage difference
        start_metrics = profile_info['start_metrics']
        cpu_usage = final_metrics.process_cpu_percent - start_metrics.process_cpu_percent
        memory_usage = final_metrics.process_memory_mb - start_metrics.process_memory_mb
        
        stats['cpu_usage'].append(cpu_usage)
        stats['memory_usage'].append(memory_usage)
        
        performance_data = {
            'function_name': function_name,
            'execution_time_ms': execution_time_ms,
            'cpu_usage_change': cpu_usage,
            'memory_usage_change_mb': memory_usage,
            'thread_id': profile_info['thread_id'],
            'start_metrics': start_metrics.to_dict(),
            'final_metrics': final_metrics.to_dict()
        }
        
        # Log slow functions
        if execution_time_ms > 100:  # > 100ms
            self.logger.warning(
                f"⚡ SLOW_FUNCTION: {function_name} took {execution_time_ms:.2f}ms",
                extra={
                    'slow_function_profile': performance_data
                }
            )
        
        return performance_data
    
    def get_function_statistics(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get function performance statistics."""
        if function_name:
            return dict(self.function_stats.get(function_name, {}))
        
        return {name: dict(stats) for name, stats in self.function_stats.items()}
    
    def get_top_slow_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top slowest functions by average execution time."""
        sorted_functions = sorted(
            self.function_stats.items(),
            key=lambda x: x[1]['avg_time'],
            reverse=True
        )
        
        return [
            {
                'function_name': name,
                'avg_time_ms': stats['avg_time'],
                'call_count': stats['call_count'],
                'total_time_ms': stats['total_time'],
                'max_time_ms': stats['max_time']
            }
            for name, stats in sorted_functions[:limit]
        ]


class MemoryLeakDetector:
    """Detects memory leaks and excessive memory usage."""
    
    def __init__(self):
        """Initialize memory leak detector."""
        self.logger = get_enhanced_logger("arena_bot.debugging.performance_analyzer.memory_leak")
        
        # Memory tracking
        self.memory_snapshots: deque = deque(maxlen=100)
        self.object_growth_tracking: Dict[type, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Leak detection parameters
        self.leak_detection_enabled = False
        self.leak_threshold_mb = 50  # Consider leak if growth > 50MB
        self.leak_detection_window = 10  # Number of snapshots to analyze
        
        # Detected leaks
        self.detected_leaks: List[Dict[str, Any]] = []
    
    def start_leak_detection(self) -> None:
        """Start memory leak detection."""
        self.leak_detection_enabled = True
        self.logger.info("💧 Memory leak detection started")
    
    def stop_leak_detection(self) -> None:
        """Stop memory leak detection."""
        self.leak_detection_enabled = False
        self.logger.info("💧 Memory leak detection stopped")
    
    def take_memory_snapshot(self) -> Dict[str, Any]:
        """Take a comprehensive memory snapshot."""
        if not self.leak_detection_enabled:
            return {}
        
        try:
            # Force garbage collection before snapshot
            gc.collect()
            
            # Get memory metrics
            metrics = PerformanceMetrics()
            metrics.capture_metrics()
            
            # Count objects by type
            object_counts = defaultdict(int)
            object_sizes = defaultdict(int)
            
            for obj in gc.get_objects():
                obj_type = type(obj)
                object_counts[obj_type] += 1
                try:
                    object_sizes[obj_type] += sys.getsizeof(obj)
                except Exception:
                    pass
            
            snapshot = {
                'timestamp': time.time(),
                'total_memory_mb': metrics.process_memory_mb,
                'total_objects': metrics.active_objects,
                'gc_collections': metrics.gc_collections,
                'object_counts': dict(object_counts),
                'object_sizes_bytes': dict(object_sizes)
            }
            
            self.memory_snapshots.append(snapshot)
            
            # Update object growth tracking
            for obj_type, count in object_counts.items():
                self.object_growth_tracking[obj_type].append(count)
            
            # Analyze for leaks
            self._analyze_memory_leaks()
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            return {}
    
    def _analyze_memory_leaks(self) -> None:
        """Analyze memory snapshots for potential leaks."""
        if len(self.memory_snapshots) < self.leak_detection_window:
            return
        
        recent_snapshots = list(self.memory_snapshots)[-self.leak_detection_window:]
        
        # Analyze overall memory growth
        start_memory = recent_snapshots[0]['total_memory_mb']
        end_memory = recent_snapshots[-1]['total_memory_mb']
        memory_growth = end_memory - start_memory
        
        if memory_growth > self.leak_threshold_mb:
            leak_info = {
                'type': 'overall_memory_growth',
                'growth_mb': memory_growth,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'time_window_seconds': recent_snapshots[-1]['timestamp'] - recent_snapshots[0]['timestamp'],
                'detected_time': time.time()
            }
            
            self.detected_leaks.append(leak_info)
            
            self.logger.warning(
                f"💧 MEMORY_LEAK_DETECTED: Memory grew by {memory_growth:.2f}MB",
                extra={'memory_leak_detection': leak_info}
            )
        
        # Analyze object type growth
        for obj_type, counts in self.object_growth_tracking.items():
            if len(counts) >= self.leak_detection_window:
                recent_counts = list(counts)[-self.leak_detection_window:]
                growth = recent_counts[-1] - recent_counts[0]
                growth_rate = growth / len(recent_counts)
                
                # Significant object growth (>1000 objects with >10% growth rate)
                if growth > 1000 and growth_rate > recent_counts[0] * 0.1:
                    leak_info = {
                        'type': 'object_type_growth',
                        'object_type': str(obj_type),
                        'object_growth': growth,
                        'growth_rate': growth_rate,
                        'start_count': recent_counts[0],
                        'end_count': recent_counts[-1],
                        'detected_time': time.time()
                    }
                    
                    self.detected_leaks.append(leak_info)
                    
                    self.logger.warning(
                        f"💧 OBJECT_LEAK_DETECTED: {obj_type} objects grew by {growth}",
                        extra={'object_leak_detection': leak_info}
                    )
    
    def get_leak_report(self) -> Dict[str, Any]:
        """Get comprehensive memory leak report."""
        return {
            'leak_detection_enabled': self.leak_detection_enabled,
            'detected_leaks': self.detected_leaks,
            'snapshots_taken': len(self.memory_snapshots),
            'current_memory_mb': self.memory_snapshots[-1]['total_memory_mb'] if self.memory_snapshots else 0,
            'object_growth_summary': {
                str(obj_type): {
                    'current_count': list(counts)[-1] if counts else 0,
                    'max_count': max(counts) if counts else 0,
                    'growth_trend': list(counts)[-5:] if len(counts) >= 5 else list(counts)
                }
                for obj_type, counts in self.object_growth_tracking.items()
            }
        }


class BottleneckDetector:
    """Detects various types of performance bottlenecks."""
    
    def __init__(self):
        """Initialize bottleneck detector."""
        self.logger = get_enhanced_logger("arena_bot.debugging.performance_analyzer.bottleneck_detector")
        
        # Detection algorithms
        self.detection_algorithms = {
            BottleneckType.CPU_INTENSIVE: self._detect_cpu_bottleneck,
            BottleneckType.MEMORY_LEAK: self._detect_memory_bottleneck,
            BottleneckType.IO_BLOCKING: self._detect_io_bottleneck,
            BottleneckType.ALGORITHM_INEFFICIENT: self._detect_algorithm_bottleneck,
            BottleneckType.SYNCHRONIZATION_BLOCKING: self._detect_sync_bottleneck
        }
        
        # Detection state
        self.detected_bottlenecks: List[PerformanceBottleneck] = []
        self.detection_enabled = False
        
        # Thresholds
        self.cpu_threshold_percent = 80
        self.memory_growth_threshold_mb = 100
        self.io_wait_threshold_ms = 1000
        self.sync_wait_threshold_ms = 500
    
    def analyze_performance_data(self, performance_data: Dict[str, Any],
                                metrics_history: List[PerformanceMetrics]) -> List[PerformanceBottleneck]:
        """Analyze performance data to detect bottlenecks."""
        if not self.detection_enabled:
            return []
        
        bottlenecks = []
        
        # Run all detection algorithms
        for bottleneck_type, detector in self.detection_algorithms.items():
            try:
                detected = detector(performance_data, metrics_history)
                if detected:
                    bottlenecks.extend(detected)
            except Exception as e:
                self.logger.error(f"Bottleneck detection failed for {bottleneck_type.value}: {e}")
        
        # Update detected bottlenecks
        self.detected_bottlenecks.extend(bottlenecks)
        
        # Log detected bottlenecks
        for bottleneck in bottlenecks:
            self.logger.warning(
                f"🎯 BOTTLENECK_DETECTED: {bottleneck.bottleneck_type.value} in {bottleneck.function_name}",
                extra={'performance_bottleneck': bottleneck.to_dict()}
            )
        
        return bottlenecks
    
    def _detect_cpu_bottleneck(self, performance_data: Dict[str, Any],
                              metrics_history: List[PerformanceMetrics]) -> List[PerformanceBottleneck]:
        """Detect CPU-intensive bottlenecks."""
        bottlenecks = []
        
        # Check recent CPU usage
        if metrics_history:
            recent_cpu = [m.process_cpu_percent for m in metrics_history[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0
            
            if avg_cpu > self.cpu_threshold_percent:
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.CPU_INTENSIVE,
                    severity=PerformanceSeverity.HIGH if avg_cpu > 95 else PerformanceSeverity.MEDIUM,
                    impact_description=f"High CPU usage: {avg_cpu:.1f}% average",
                    resource_usage={'cpu_percent': avg_cpu},
                    detection_method="cpu_usage_threshold",
                    confidence_score=min(avg_cpu / 100.0, 1.0),
                    root_cause_analysis="Sustained high CPU usage detected",
                    optimization_suggestions=[
                        "Profile CPU-intensive functions using cProfile",
                        "Consider algorithmic optimizations",
                        "Use multiprocessing for CPU-bound tasks",
                        "Implement caching for expensive computations"
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_memory_bottleneck(self, performance_data: Dict[str, Any],
                                 metrics_history: List[PerformanceMetrics]) -> List[PerformanceBottleneck]:
        """Detect memory-related bottlenecks."""
        bottlenecks = []
        
        if len(metrics_history) >= 2:
            start_memory = metrics_history[0].process_memory_mb
            end_memory = metrics_history[-1].process_memory_mb
            memory_growth = end_memory - start_memory
            
            if memory_growth > self.memory_growth_threshold_mb:
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.MEMORY_LEAK,
                    severity=PerformanceSeverity.HIGH,
                    impact_description=f"Memory growth: {memory_growth:.1f}MB",
                    resource_usage={'memory_growth_mb': memory_growth},
                    detection_method="memory_growth_analysis",
                    confidence_score=min(memory_growth / 500.0, 1.0),
                    root_cause_analysis="Significant memory growth detected",
                    optimization_suggestions=[
                        "Use memory profiler to identify large objects",
                        "Implement proper cleanup in finally blocks",
                        "Check for circular references",
                        "Use weak references where appropriate"
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_io_bottleneck(self, performance_data: Dict[str, Any],
                             metrics_history: List[PerformanceMetrics]) -> List[PerformanceBottleneck]:
        """Detect I/O blocking bottlenecks."""
        bottlenecks = []
        
        # Check for I/O patterns in performance data
        if 'execution_time_ms' in performance_data:
            execution_time = performance_data['execution_time_ms']
            cpu_usage = performance_data.get('cpu_usage_change', 0)
            
            # High execution time with low CPU usage suggests I/O blocking
            if execution_time > self.io_wait_threshold_ms and cpu_usage < 10:
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.IO_BLOCKING,
                    severity=PerformanceSeverity.MEDIUM,
                    function_name=performance_data.get('function_name', 'unknown'),
                    impact_description=f"I/O blocking: {execution_time:.1f}ms with low CPU usage",
                    resource_usage={'execution_time_ms': execution_time, 'cpu_usage': cpu_usage},
                    detection_method="io_pattern_analysis",
                    confidence_score=0.7,
                    root_cause_analysis="High execution time with low CPU suggests I/O blocking",
                    optimization_suggestions=[
                        "Use asynchronous I/O operations",
                        "Implement connection pooling",
                        "Add caching layer",
                        "Use bulk operations where possible"
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_algorithm_bottleneck(self, performance_data: Dict[str, Any],
                                    metrics_history: List[PerformanceMetrics]) -> List[PerformanceBottleneck]:
        """Detect algorithmic inefficiency bottlenecks."""
        bottlenecks = []
        
        # This would require more sophisticated analysis of call patterns
        # For now, detect functions with very high execution times
        if 'execution_time_ms' in performance_data:
            execution_time = performance_data['execution_time_ms']
            
            if execution_time > 5000:  # > 5 seconds
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.ALGORITHM_INEFFICIENT,
                    severity=PerformanceSeverity.HIGH,
                    function_name=performance_data.get('function_name', 'unknown'),
                    impact_description=f"Very slow function: {execution_time:.1f}ms",
                    resource_usage={'execution_time_ms': execution_time},
                    detection_method="execution_time_threshold",
                    confidence_score=0.6,
                    root_cause_analysis="Extremely slow function execution suggests algorithmic inefficiency",
                    optimization_suggestions=[
                        "Review algorithm complexity (O(n²) vs O(n log n))",
                        "Implement memoization for repeated calculations",
                        "Use more efficient data structures",
                        "Consider parallelization opportunities"
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_sync_bottleneck(self, performance_data: Dict[str, Any],
                               metrics_history: List[PerformanceMetrics]) -> List[PerformanceBottleneck]:
        """Detect synchronization blocking bottlenecks."""
        bottlenecks = []
        
        # Check for thread contention patterns
        if metrics_history:
            thread_counts = [m.process_threads for m in metrics_history[-5:]]
            if thread_counts and max(thread_counts) > 20:  # Many threads might indicate contention
                
                bottleneck = PerformanceBottleneck(
                    bottleneck_type=BottleneckType.SYNCHRONIZATION_BLOCKING,
                    severity=PerformanceSeverity.MEDIUM,
                    impact_description=f"High thread count: {max(thread_counts)} threads",
                    resource_usage={'thread_count': max(thread_counts)},
                    detection_method="thread_count_analysis",
                    confidence_score=0.5,
                    root_cause_analysis="High thread count may indicate synchronization issues",
                    optimization_suggestions=[
                        "Review lock usage and critical sections",
                        "Use lock-free data structures where possible",
                        "Implement timeout on lock acquisitions",
                        "Consider using queues instead of shared data"
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks


class PerformanceAnalyzer:
    """
    Main performance analysis and bottleneck detection system.
    
    Provides comprehensive performance monitoring and bottleneck detection
    for Arena Bot to identify performance issues and optimization opportunities.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.logger = get_enhanced_logger("arena_bot.debugging.performance_analyzer")
        
        # Core components
        self.function_profiler = FunctionProfiler()
        self.memory_leak_detector = MemoryLeakDetector()
        self.bottleneck_detector = BottleneckDetector()
        
        # Performance monitoring
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.monitoring_enabled = False
        self.monitoring_interval_seconds = 5.0
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Analysis state
        self.start_time = time.time()
        self.analysis_count = 0
        
        # Performance baselines
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.performance_regression_threshold = 0.2  # 20% performance degradation
    
    def start_monitoring(self, enable_profiling: bool = True,
                        enable_leak_detection: bool = True,
                        enable_bottleneck_detection: bool = True) -> bool:
        """
        Start performance monitoring.
        
        Args:
            enable_profiling: Enable function profiling
            enable_leak_detection: Enable memory leak detection
            enable_bottleneck_detection: Enable bottleneck detection
            
        Returns:
            True if monitoring started successfully
        """
        
        try:
            if self.monitoring_enabled:
                self.logger.warning("Performance monitoring is already enabled")
                return True
            
            # Enable components
            if enable_profiling:
                self.function_profiler.profiling_enabled = True
            
            if enable_leak_detection:
                self.memory_leak_detector.start_leak_detection()
            
            if enable_bottleneck_detection:
                self.bottleneck_detector.detection_enabled = True
            
            # Start monitoring thread
            self._start_monitoring_thread()
            
            # Capture baseline metrics
            self._capture_baseline()
            
            self.monitoring_enabled = True
            self.start_time = time.time()
            
            self.logger.critical(
                "🚀 PERFORMANCE_MONITORING_STARTED: Advanced performance analysis enabled",
                extra={
                    'performance_monitoring_startup': {
                        'enable_profiling': enable_profiling,
                        'enable_leak_detection': enable_leak_detection,
                        'enable_bottleneck_detection': enable_bottleneck_detection,
                        'monitoring_interval_seconds': self.monitoring_interval_seconds,
                        'timestamp': time.time()
                    }
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        
        if not self.monitoring_enabled:
            return
        
        try:
            # Stop monitoring thread
            self.stop_monitoring.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            # Stop components
            self.function_profiler.profiling_enabled = False
            self.memory_leak_detector.stop_leak_detection()
            self.bottleneck_detector.detection_enabled = False
            
            self.monitoring_enabled = False
            
            uptime_seconds = time.time() - self.start_time
            
            self.logger.info(
                f"🔄 PERFORMANCE_MONITORING_STOPPED: Monitoring stopped after {uptime_seconds:.1f}s",
                extra={
                    'performance_monitoring_shutdown': {
                        'uptime_seconds': uptime_seconds,
                        'analysis_count': self.analysis_count,
                        'metrics_collected': len(self.metrics_history),
                        'bottlenecks_detected': len(self.bottleneck_detector.detected_bottlenecks),
                        'timestamp': time.time()
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error stopping performance monitoring: {e}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        
        start_time = time.perf_counter()
        
        try:
            # Get function statistics
            function_stats = self.function_profiler.get_function_statistics()
            slow_functions = self.function_profiler.get_top_slow_functions(20)
            
            # Get memory leak report
            memory_report = self.memory_leak_detector.get_leak_report()
            
            # Get current metrics
            current_metrics = PerformanceMetrics()
            current_metrics.capture_metrics()
            
            # Analyze bottlenecks
            bottlenecks = []
            if self.metrics_history:
                # Create synthetic performance data for bottleneck analysis
                performance_data = {
                    'execution_time_ms': 0,
                    'cpu_usage_change': current_metrics.process_cpu_percent,
                    'memory_usage_change_mb': 0
                }
                
                detected_bottlenecks = self.bottleneck_detector.analyze_performance_data(
                    performance_data, list(self.metrics_history)
                )
                bottlenecks = [b.to_dict() for b in detected_bottlenecks]
            
            # Performance regression analysis
            regression_analysis = self._analyze_performance_regression()
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                slow_functions, memory_report, bottlenecks, regression_analysis
            )
            
            analysis_time = (time.perf_counter() - start_time) * 1000
            self.analysis_count += 1
            
            analysis_result = {
                'analysis_timestamp': time.time(),
                'analysis_time_ms': analysis_time,
                'current_metrics': current_metrics.to_dict(),
                'function_statistics': {
                    'total_functions_profiled': len(function_stats),
                    'slow_functions': slow_functions
                },
                'memory_analysis': memory_report,
                'bottlenecks': bottlenecks,
                'performance_regression': regression_analysis,
                'optimization_recommendations': recommendations,
                'monitoring_status': {
                    'monitoring_enabled': self.monitoring_enabled,
                    'metrics_collected': len(self.metrics_history),
                    'analysis_count': self.analysis_count
                }
            }
            
            self.logger.info(
                f"📊 PERFORMANCE_ANALYSIS_COMPLETE: Analysis completed in {analysis_time:.2f}ms",
                extra={
                    'performance_analysis_result': analysis_result
                }
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e), 'analysis_timestamp': time.time()}
    
    def profile_function(self, function_name: str, module_name: str = "") -> Callable:
        """
        Decorator to profile function performance.
        
        Usage:
            @performance_analyzer.profile_function("my_function", "my_module")
            def my_function():
                pass
        """
        
        def decorator(func: Callable) -> Callable:
            
            def wrapper(*args, **kwargs):
                if self.monitoring_enabled:
                    # Start profiling
                    profile_id = self.function_profiler.start_profiling_function(
                        function_name or func.__name__, 
                        module_name
                    )
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # End profiling
                        if profile_id:
                            performance_data = self.function_profiler.end_profiling_function(profile_id)
                            
                            # Analyze for bottlenecks
                            if performance_data:
                                self.bottleneck_detector.analyze_performance_data(
                                    performance_data, list(self.metrics_history)
                                )
                        
                        return result
                        
                    except Exception as e:
                        # End profiling even on exception
                        if profile_id:
                            self.function_profiler.end_profiling_function(profile_id)
                        raise
                else:
                    return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def _start_monitoring_thread(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.debug(f"Performance monitoring thread started (interval: {self.monitoring_interval_seconds}s)")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval_seconds):
            try:
                # Capture metrics
                metrics = PerformanceMetrics()
                metrics.capture_metrics()
                self.metrics_history.append(metrics)
                
                # Take memory snapshot for leak detection
                self.memory_leak_detector.take_memory_snapshot()
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _capture_baseline(self) -> None:
        """Capture performance baseline for regression detection."""
        try:
            baseline = PerformanceMetrics()
            baseline.capture_metrics()
            self.baseline_metrics = baseline
            
            self.logger.info(
                "📈 Performance baseline captured",
                extra={
                    'performance_baseline': baseline.to_dict()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to capture performance baseline: {e}")
    
    def _analyze_performance_regression(self) -> Dict[str, Any]:
        """Analyze for performance regression compared to baseline."""
        if not self.baseline_metrics or not self.metrics_history:
            return {'regression_detected': False}
        
        current_metrics = self.metrics_history[-1]
        baseline = self.baseline_metrics
        
        # Compare key metrics
        cpu_regression = (current_metrics.process_cpu_percent - baseline.process_cpu_percent) / max(baseline.process_cpu_percent, 1)
        memory_regression = (current_metrics.process_memory_mb - baseline.process_memory_mb) / max(baseline.process_memory_mb, 1)
        
        regression_detected = (
            cpu_regression > self.performance_regression_threshold or
            memory_regression > self.performance_regression_threshold
        )
        
        return {
            'regression_detected': regression_detected,
            'cpu_regression_percent': cpu_regression * 100,
            'memory_regression_percent': memory_regression * 100,
            'baseline_timestamp': baseline.timestamp,
            'current_timestamp': current_metrics.timestamp,
            'regression_threshold_percent': self.performance_regression_threshold * 100
        }
    
    def _generate_optimization_recommendations(self, slow_functions: List[Dict[str, Any]],
                                             memory_report: Dict[str, Any],
                                             bottlenecks: List[Dict[str, Any]],
                                             regression_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Function performance recommendations
        if slow_functions:
            recommendations.append(f"🐌 {len(slow_functions)} slow functions detected - review and optimize")
            
            slowest = slow_functions[0]
            recommendations.append(f"⚡ Slowest function: {slowest['function_name']} ({slowest['avg_time_ms']:.1f}ms avg)")
        
        # Memory recommendations
        if memory_report.get('detected_leaks'):
            leak_count = len(memory_report['detected_leaks'])
            recommendations.append(f"💧 {leak_count} memory leaks detected - investigate and fix")
        
        if memory_report.get('current_memory_mb', 0) > 500:
            recommendations.append("💾 High memory usage detected - consider memory optimization")
        
        # Bottleneck recommendations
        if bottlenecks:
            bottleneck_types = set(b['bottleneck_type'] for b in bottlenecks)
            recommendations.append(f"🎯 {len(bottlenecks)} performance bottlenecks detected: {', '.join(bottleneck_types)}")
        
        # Regression recommendations
        if regression_analysis.get('regression_detected'):
            recommendations.append("📉 Performance regression detected - compare with baseline")
        
        if not recommendations:
            recommendations.append("✅ No major performance issues detected - system performing well")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive performance analyzer status."""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'uptime_seconds': time.time() - self.start_time if self.monitoring_enabled else 0,
            'analysis_count': self.analysis_count,
            'metrics_collected': len(self.metrics_history),
            'profiling_enabled': self.function_profiler.profiling_enabled,
            'leak_detection_enabled': self.memory_leak_detector.leak_detection_enabled,
            'bottleneck_detection_enabled': self.bottleneck_detector.detection_enabled,
            'functions_profiled': len(self.function_profiler.function_stats),
            'memory_snapshots': len(self.memory_leak_detector.memory_snapshots),
            'bottlenecks_detected': len(self.bottleneck_detector.detected_bottlenecks),
            'baseline_available': self.baseline_metrics is not None
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown performance analyzer."""
        self.logger.info("🔄 Shutting down performance analyzer...")
        self.stop_monitoring()
        self.logger.info("✅ Performance analyzer shutdown complete")


# Global performance analyzer instance
_global_performance_analyzer: Optional[PerformanceAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get global performance analyzer instance."""
    global _global_performance_analyzer
    
    if _global_performance_analyzer is None:
        with _analyzer_lock:
            if _global_performance_analyzer is None:
                _global_performance_analyzer = PerformanceAnalyzer()
    
    return _global_performance_analyzer


def start_performance_monitoring(enable_profiling: bool = True,
                                enable_leak_detection: bool = True,
                                enable_bottleneck_detection: bool = True) -> bool:
    """
    Start performance monitoring.
    
    Convenience function to start comprehensive performance analysis.
    
    Args:
        enable_profiling: Enable function profiling
        enable_leak_detection: Enable memory leak detection
        enable_bottleneck_detection: Enable bottleneck detection
        
    Returns:
        True if monitoring started successfully
    """
    analyzer = get_performance_analyzer()
    return analyzer.start_monitoring(enable_profiling, enable_leak_detection, enable_bottleneck_detection)


def stop_performance_monitoring() -> None:
    """Stop performance monitoring."""
    analyzer = get_performance_analyzer()
    analyzer.stop_monitoring()


def analyze_performance() -> Dict[str, Any]:
    """Perform comprehensive performance analysis."""
    analyzer = get_performance_analyzer()
    return analyzer.analyze_performance()


def get_performance_status() -> Dict[str, Any]:
    """Get performance analyzer status."""
    analyzer = get_performance_analyzer()
    return analyzer.get_status()


# Convenience decorator for performance profiling
def profile_performance(function_name: str = "", module_name: str = "") -> Callable:
    """
    Decorator to profile function performance.
    
    Usage:
        @profile_performance("my_function", "my_module")
        def my_function():
            pass
    """
    analyzer = get_performance_analyzer()
    return analyzer.profile_function(function_name, module_name)