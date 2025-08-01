"""
Performance Profiler for S-Tier Logging System.

This module provides comprehensive performance monitoring and profiling
for the logging system with real-time metrics collection, bottleneck
detection, and performance optimization recommendations.

Features:
- Real-time performance metrics collection
- Latency tracking with percentile analysis
- Throughput monitoring and trend analysis
- Memory usage and garbage collection profiling
- Bottleneck detection and analysis
- Performance optimization recommendations
"""

import asyncio
import time
import gc
import threading
import psutil
import statistics
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class MetricType(str, Enum):
    """Types of performance metrics."""
    COUNTER = "counter"         # Incrementing count
    GAUGE = "gauge"            # Current value
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"            # Duration measurements


class PerformanceLevel(str, Enum):
    """Performance level categories."""
    EXCELLENT = "excellent"     # >95th percentile
    GOOD = "good"              # 75-95th percentile
    ACCEPTABLE = "acceptable"   # 50-75th percentile
    POOR = "poor"              # 25-50th percentile
    CRITICAL = "critical"       # <25th percentile


@dataclass
class MetricValue:
    """Single metric measurement."""
    
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """Get age of metric in seconds."""
        return time.time() - self.timestamp


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    stddev: float = 0.0
    
    def update(self, values: List[float]) -> None:
        """Update statistics from list of values."""
        if not values:
            return
        
        self.count = len(values)
        self.sum = sum(values)
        self.min = min(values)
        self.max = max(values)
        self.mean = statistics.mean(values)
        self.median = statistics.median(values)
        
        if self.count >= 20:  # Need sufficient data for percentiles
            sorted_values = sorted(values)
            self.p95 = sorted_values[int(0.95 * len(sorted_values))]
            self.p99 = sorted_values[int(0.99 * len(sorted_values))]
        
        if self.count >= 2:
            self.stddev = statistics.stdev(values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float('inf') else None,
            "max": self.max if self.max != float('-inf') else None,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "stddev": self.stddev
        }


class PerformanceMetric:
    """Container for performance metric data."""
    
    def __init__(self, 
                 name: str,
                 metric_type: MetricType,
                 max_age_seconds: float = 300.0,
                 max_samples: int = 10000):
        """
        Initialize performance metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            max_age_seconds: Maximum age of samples to keep
            max_samples: Maximum number of samples to keep
        """
        self.name = name
        self.metric_type = metric_type
        self.max_age_seconds = max_age_seconds
        self.max_samples = max_samples
        
        self.samples: deque[MetricValue] = deque(maxlen=max_samples)
        self._lock = threading.RLock()
        
        # Cached statistics
        self._cached_stats: Optional[PerformanceStats] = None
        self._cache_timestamp = 0.0
        self._cache_ttl = 1.0  # Cache stats for 1 second
    
    def add_sample(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a sample to the metric."""
        with self._lock:
            sample = MetricValue(value=value, labels=labels or {})
            self.samples.append(sample)
            
            # Invalidate cache
            self._cached_stats = None
            
            # Clean old samples
            self._clean_old_samples()
    
    def _clean_old_samples(self) -> None:
        """Remove samples older than max_age_seconds."""
        if not self.samples:
            return
        
        cutoff_time = time.time() - self.max_age_seconds
        
        # Remove old samples from the left
        while self.samples and self.samples[0].timestamp < cutoff_time:
            self.samples.popleft()
    
    def get_stats(self) -> PerformanceStats:
        """Get aggregated statistics for the metric."""
        with self._lock:
            current_time = time.time()
            
            # Check cache validity
            if (self._cached_stats is not None and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._cached_stats
            
            # Clean old samples
            self._clean_old_samples()
            
            # Calculate statistics
            values = [sample.value for sample in self.samples]
            stats = PerformanceStats()
            stats.update(values)
            
            # Cache results
            self._cached_stats = stats
            self._cache_timestamp = current_time
            
            return stats
    
    def get_recent_values(self, seconds: float = 60.0) -> List[float]:
        """Get values from the last N seconds."""
        with self._lock:
            cutoff_time = time.time() - seconds
            return [
                sample.value for sample in self.samples
                if sample.timestamp >= cutoff_time
            ]
    
    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self.samples.clear()
            self._cached_stats = None


class PerformanceProfiler:
    """
    Main performance profiler for the S-tier logging system.
    
    Collects and analyzes performance metrics with real-time
    monitoring and bottleneck detection capabilities.
    """
    
    def __init__(self, 
                 logger_manager: 'LoggerManager',
                 collection_interval: float = 1.0,
                 retention_seconds: float = 3600.0):
        """
        Initialize performance profiler.
        
        Args:
            logger_manager: Logger manager instance to profile
            collection_interval: How often to collect metrics (seconds)
            retention_seconds: How long to retain metrics
        """
        self.logger_manager = logger_manager
        self.collection_interval = collection_interval
        self.retention_seconds = retention_seconds
        
        # Metrics registry
        self.metrics: Dict[str, PerformanceMetric] = {}
        self._metrics_lock = threading.RLock()
        
        # Background collection
        self._collection_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Baseline measurements for comparison
        self._baseline_metrics: Dict[str, PerformanceStats] = {}
        self._baseline_timestamp: Optional[float] = None
        
        # Performance thresholds
        self.thresholds = {
            'log_latency_ms': {'excellent': 1.0, 'good': 5.0, 'acceptable': 20.0, 'poor': 100.0},
            'queue_depth': {'excellent': 100, 'good': 500, 'acceptable': 1000, 'poor': 5000},
            'cpu_percent': {'excellent': 10, 'good': 30, 'acceptable': 60, 'poor': 85},
            'memory_percent': {'excellent': 20, 'good': 50, 'acceptable': 70, 'poor': 85},
            'error_rate': {'excellent': 0.001, 'good': 0.01, 'acceptable': 0.05, 'poor': 0.15}
        }
        
        # Initialize standard metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize standard performance metrics."""
        # Latency metrics
        self.register_metric("log_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("queue_put_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("queue_get_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("sink_write_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("filter_process_latency_ms", MetricType.HISTOGRAM)
        
        # Throughput metrics
        self.register_metric("logs_per_second", MetricType.GAUGE)
        self.register_metric("bytes_per_second", MetricType.GAUGE)
        self.register_metric("records_processed", MetricType.COUNTER)
        self.register_metric("records_dropped", MetricType.COUNTER)
        self.register_metric("errors_total", MetricType.COUNTER)
        
        # Queue metrics
        self.register_metric("queue_depth", MetricType.GAUGE)
        self.register_metric("queue_utilization_percent", MetricType.GAUGE)
        self.register_metric("queue_overflow_count", MetricType.COUNTER)
        
        # Resource metrics
        self.register_metric("cpu_percent", MetricType.GAUGE)
        self.register_metric("memory_percent", MetricType.GAUGE)
        self.register_metric("memory_usage_mb", MetricType.GAUGE)
        self.register_metric("gc_collections", MetricType.COUNTER)
        self.register_metric("gc_time_seconds", MetricType.HISTOGRAM)
        
        # Worker metrics
        self.register_metric("active_workers", MetricType.GAUGE)
        self.register_metric("worker_utilization_percent", MetricType.GAUGE)
        self.register_metric("worker_idle_time_seconds", MetricType.HISTOGRAM)
    
    def register_metric(self, 
                       name: str, 
                       metric_type: MetricType,
                       max_age_seconds: Optional[float] = None,
                       max_samples: Optional[int] = None) -> None:
        """Register a new performance metric."""
        with self._metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetric(
                    name=name,
                    metric_type=metric_type,
                    max_age_seconds=max_age_seconds or self.retention_seconds,
                    max_samples=max_samples or 10000
                )
    
    def record_metric(self, 
                     name: str, 
                     value: float,
                     labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._metrics_lock:
            if name in self.metrics:
                self.metrics[name].add_sample(value, labels)
    
    def increment_counter(self, name: str, amount: float = 1.0) -> None:
        """Increment a counter metric."""
        self.record_metric(name, amount)
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value)
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram measurement."""
        self.record_metric(name, value)
    
    def time_operation(self, metric_name: str):
        """Context manager for timing operations."""
        return TimingContext(self, metric_name)
    
    async def collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("cpu_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("memory_percent", memory.percent)
            self.set_gauge("memory_usage_mb", memory.used / (1024 * 1024))
            
            # Garbage collection stats
            gc_stats = gc.get_stats()
            if gc_stats:
                total_collections = sum(stat['collections'] for stat in gc_stats)
                self.set_gauge("gc_collections", total_collections)
            
        except Exception:
            # Ignore errors in metric collection
            pass
    
    async def collect_logger_metrics(self) -> None:
        """Collect logging system specific metrics."""
        try:
            # Get logger manager stats
            stats = self.logger_manager.get_performance_stats()
            
            # Throughput metrics
            self.set_gauge("logs_per_second", stats.get('logs_per_second', 0))
            self.set_gauge("error_rate", stats.get('error_rate', 0))
            
            # Queue metrics
            if 'queue_stats' in stats:
                queue_stats = stats['queue_stats']
                self.set_gauge("queue_depth", queue_stats.get('current_size', 0))
                self.set_gauge("queue_utilization_percent", 
                              queue_stats.get('utilization_percent', 0))
            
            # Worker metrics
            if 'worker_pool_stats' in stats:
                worker_stats = stats['worker_pool_stats']
                self.set_gauge("active_workers", worker_stats.get('active_workers', 0))
                self.set_gauge("worker_utilization_percent",
                              worker_stats.get('utilization_percent', 0))
            
        except Exception:
            # Ignore errors in metric collection
            pass
    
    def get_metric_stats(self, name: str) -> Optional[PerformanceStats]:
        """Get statistics for a specific metric."""
        with self._metrics_lock:
            if name in self.metrics:
                return self.metrics[name].get_stats()
            return None
    
    def get_all_metrics(self) -> Dict[str, PerformanceStats]:
        """Get statistics for all metrics."""
        with self._metrics_lock:
            return {
                name: metric.get_stats()
                for name, metric in self.metrics.items()
            }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        analysis = {
            'timestamp': time.time(),
            'overall_performance': self._calculate_overall_performance(),
            'bottlenecks': self._identify_bottlenecks(),
            'trends': self._analyze_trends(),
            'recommendations': self._generate_recommendations(),
            'metrics_summary': self._summarize_metrics()
        }
        
        return analysis
    
    def _calculate_overall_performance(self) -> Dict[str, Any]:
        """Calculate overall system performance level."""
        performance_scores = {}
        
        # Evaluate key metrics against thresholds
        for metric_name, thresholds in self.thresholds.items():
            stats = self.get_metric_stats(metric_name)
            if stats and stats.count > 0:
                # Use appropriate statistic (mean for gauges, p95 for histograms)
                if metric_name.endswith('_ms') or metric_name == 'error_rate':
                    value = stats.p95 if stats.count >= 20 else stats.mean
                else:
                    value = stats.mean
                
                # Determine performance level
                if value <= thresholds['excellent']:
                    level = PerformanceLevel.EXCELLENT
                elif value <= thresholds['good']:
                    level = PerformanceLevel.GOOD
                elif value <= thresholds['acceptable']:
                    level = PerformanceLevel.ACCEPTABLE
                elif value <= thresholds['poor']:
                    level = PerformanceLevel.POOR
                else:
                    level = PerformanceLevel.CRITICAL
                
                performance_scores[metric_name] = {
                    'level': level.value,
                    'value': value,
                    'thresholds': thresholds
                }
        
        # Calculate overall score
        level_weights = {
            PerformanceLevel.EXCELLENT: 5,
            PerformanceLevel.GOOD: 4,
            PerformanceLevel.ACCEPTABLE: 3,
            PerformanceLevel.POOR: 2,
            PerformanceLevel.CRITICAL: 1
        }
        
        if performance_scores:
            weighted_sum = sum(
                level_weights[PerformanceLevel(score['level'])]
                for score in performance_scores.values()
            )
            avg_score = weighted_sum / len(performance_scores)
            
            if avg_score >= 4.5:
                overall_level = PerformanceLevel.EXCELLENT
            elif avg_score >= 3.5:
                overall_level = PerformanceLevel.GOOD
            elif avg_score >= 2.5:
                overall_level = PerformanceLevel.ACCEPTABLE
            elif avg_score >= 1.5:
                overall_level = PerformanceLevel.POOR
            else:
                overall_level = PerformanceLevel.CRITICAL
        else:
            overall_level = PerformanceLevel.ACCEPTABLE
        
        return {
            'overall_level': overall_level.value,
            'metric_scores': performance_scores,
            'score': avg_score if performance_scores else 3.0
        }
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for high latency
        latency_stats = self.get_metric_stats("log_latency_ms")
        if latency_stats and latency_stats.p95 > 50:
            bottlenecks.append({
                'type': 'latency',
                'metric': 'log_latency_ms',
                'severity': 'high' if latency_stats.p95 > 100 else 'medium',
                'description': f"High logging latency: {latency_stats.p95:.1f}ms (p95)",
                'recommendation': "Consider increasing worker threads or optimizing sink performance"
            })
        
        # Check for queue congestion
        queue_stats = self.get_metric_stats("queue_depth")
        if queue_stats and queue_stats.mean > 1000:
            bottlenecks.append({
                'type': 'queue',
                'metric': 'queue_depth',
                'severity': 'high' if queue_stats.mean > 5000 else 'medium',
                'description': f"High queue depth: {queue_stats.mean:.0f} items",
                'recommendation': "Increase worker threads or optimize log processing pipeline"
            })
        
        # Check for high error rate
        error_stats = self.get_metric_stats("error_rate")
        if error_stats and error_stats.mean > 0.05:
            bottlenecks.append({
                'type': 'errors',
                'metric': 'error_rate',
                'severity': 'critical' if error_stats.mean > 0.15 else 'high',
                'description': f"High error rate: {error_stats.mean:.3f}",
                'recommendation': "Investigate log processing errors and fix underlying issues"
            })
        
        # Check for resource constraints
        cpu_stats = self.get_metric_stats("cpu_percent")
        if cpu_stats and cpu_stats.mean > 80:
            bottlenecks.append({
                'type': 'cpu',
                'metric': 'cpu_percent',
                'severity': 'high' if cpu_stats.mean > 90 else 'medium',
                'description': f"High CPU usage: {cpu_stats.mean:.1f}%",
                'recommendation': "Consider scaling hardware or optimizing log processing"
            })
        
        memory_stats = self.get_metric_stats("memory_percent")
        if memory_stats and memory_stats.mean > 80:
            bottlenecks.append({
                'type': 'memory',
                'metric': 'memory_percent',
                'severity': 'high' if memory_stats.mean > 90 else 'medium',
                'description': f"High memory usage: {memory_stats.mean:.1f}%",
                'recommendation': "Reduce retention time or increase available memory"
            })
        
        return bottlenecks
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {}
        
        # Analyze key metrics for trends
        key_metrics = ['logs_per_second', 'log_latency_ms', 'error_rate', 'queue_depth']
        
        for metric_name in key_metrics:
            metric = self.metrics.get(metric_name)
            if metric and len(metric.samples) >= 10:
                # Get recent values for trend analysis
                recent_values = metric.get_recent_values(300)  # Last 5 minutes
                older_values = [
                    sample.value for sample in metric.samples
                    if time.time() - sample.timestamp > 300
                ]
                
                if recent_values and older_values:
                    recent_avg = statistics.mean(recent_values)
                    older_avg = statistics.mean(older_values)
                    
                    # Calculate trend
                    if older_avg > 0:
                        trend_percent = ((recent_avg - older_avg) / older_avg) * 100
                    else:
                        trend_percent = 0
                    
                    if abs(trend_percent) > 5:  # Only report significant trends
                        trends[metric_name] = {
                            'trend_percent': trend_percent,
                            'direction': 'increasing' if trend_percent > 0 else 'decreasing',
                            'recent_avg': recent_avg,
                            'older_avg': older_avg
                        }
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze metrics and generate targeted recommendations
        bottlenecks = self._identify_bottlenecks()
        
        if any(b['type'] == 'latency' for b in bottlenecks):
            recommendations.append("Consider enabling async processing and increasing worker threads")
        
        if any(b['type'] == 'queue' for b in bottlenecks):
            recommendations.append("Optimize sink performance or increase processing capacity")
        
        if any(b['type'] == 'errors' for b in bottlenecks):
            recommendations.append("Review error logs and fix underlying issues")
        
        if any(b['type'] in ['cpu', 'memory'] for b in bottlenecks):
            recommendations.append("Consider scaling hardware resources")
        
        # General recommendations based on performance level
        overall_perf = self._calculate_overall_performance()
        if overall_perf['overall_level'] in ['poor', 'critical']:
            recommendations.append("Consider comprehensive performance tuning")
            recommendations.append("Review configuration settings for optimization opportunities")
        
        return recommendations
    
    def _summarize_metrics(self) -> Dict[str, Any]:
        """Create summary of key metrics."""
        summary = {}
        
        key_metrics = [
            'logs_per_second', 'log_latency_ms', 'error_rate',
            'queue_depth', 'cpu_percent', 'memory_percent'
        ]
        
        for metric_name in key_metrics:
            stats = self.get_metric_stats(metric_name)
            if stats and stats.count > 0:
                summary[metric_name] = {
                    'current': stats.mean,
                    'p95': stats.p95 if stats.count >= 20 else stats.max,
                    'min': stats.min,
                    'max': stats.max,
                    'samples': stats.count
                }
        
        return summary
    
    def set_baseline(self) -> None:
        """Set current metrics as baseline for comparison."""
        self._baseline_metrics = self.get_all_metrics()
        self._baseline_timestamp = time.time()
    
    def compare_to_baseline(self) -> Optional[Dict[str, Any]]:
        """Compare current metrics to baseline."""
        if not self._baseline_metrics:
            return None
        
        comparison = {
            'baseline_timestamp': self._baseline_timestamp,
            'comparison_timestamp': time.time(),
            'metrics': {}
        }
        
        current_metrics = self.get_all_metrics()
        
        for metric_name in current_metrics:
            if metric_name in self._baseline_metrics:
                baseline = self._baseline_metrics[metric_name]
                current = current_metrics[metric_name]
                
                if baseline.count > 0 and current.count > 0:
                    change_percent = ((current.mean - baseline.mean) / baseline.mean) * 100
                    comparison['metrics'][metric_name] = {
                        'baseline_mean': baseline.mean,
                        'current_mean': current.mean,
                        'change_percent': change_percent,
                        'improved': self._is_improvement(metric_name, change_percent)
                    }
        
        return comparison
    
    def _is_improvement(self, metric_name: str, change_percent: float) -> bool:
        """Determine if a change represents an improvement."""
        # For latency and error metrics, lower is better
        if 'latency' in metric_name or 'error' in metric_name:
            return change_percent < 0
        # For throughput metrics, higher is better
        elif 'per_second' in metric_name:
            return change_percent > 0
        # For resource metrics, lower is usually better
        elif 'percent' in metric_name or 'usage' in metric_name:
            return change_percent < 0
        else:
            # Default: lower is better
            return change_percent < 0
    
    async def start_collection(self) -> None:
        """Start continuous metric collection."""
        if self._collection_task is not None:
            return
        
        self._collection_task = asyncio.create_task(self._collection_loop())
    
    async def stop_collection(self) -> None:
        """Stop continuous metric collection."""
        if self._collection_task is not None:
            self._shutdown_event.set()
            await self._collection_task
            self._collection_task = None
    
    async def _collection_loop(self) -> None:
        """Main metric collection loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                await self.collect_system_metrics()
                await self.collect_logger_metrics()
                
                # Wait for next collection interval
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.collection_interval
                )
                
            except asyncio.TimeoutError:
                # Normal timeout, continue collection
                continue
            except Exception:
                # Log error but continue collection
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def shutdown(self) -> None:
        """Shutdown the performance profiler."""
        await self.stop_collection()


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, metric_name: str):
        self.profiler = profiler
        self.metric_name = metric_name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.profiler.record_histogram(self.metric_name, duration_ms)


# Module exports
__all__ = [
    'MetricType',
    'PerformanceLevel',
    'MetricValue',
    'PerformanceStats',
    'PerformanceMetric',
    'PerformanceProfiler',
    'TimingContext'
]