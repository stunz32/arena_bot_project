"""
Metrics Sink for S-Tier Logging System.

This module provides integration with metrics systems, converting log messages
into structured metrics for monitoring, alerting, and observability platforms.

Features:
- Log-to-metrics conversion with configurable rules
- Integration with existing monitoring.py system
- Multi-destination metrics publishing (Prometheus, StatsD, custom)
- Metric aggregation and buffering
- Performance counters and histograms
- Automatic metric labeling and tagging
- Error rate and latency tracking
- Custom metric extraction from log context
"""

import time
import threading
import logging
import json
import socket
import struct
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from statistics import mean, median

# Try to import existing monitoring system
try:
    from ...ai_v2.monitoring import get_performance_monitor, ResourceTracker, get_resource_manager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    # Create fallback classes
    class ResourceTracker:
        def __init__(self, name): pass
        def track_operation(self, name): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def get_performance_monitor(): return None
    def get_resource_manager(): return None

# Import from our components
from .base_sink import BaseSink, SinkState, ErrorHandlingStrategy
from ..formatters.structured_formatter import StructuredFormatter
from ..core.hybrid_async_queue import LogMessage


class MetricType(Enum):
    """Types of metrics that can be extracted."""
    COUNTER = "counter"        # Monotonic counter
    GAUGE = "gauge"           # Point-in-time value
    HISTOGRAM = "histogram"   # Distribution of values
    TIMER = "timer"          # Duration measurements
    SET = "set"              # Unique value tracking


class MetricDestination(Enum):
    """Metric destination types."""
    INTERNAL = "internal"     # Internal metrics collection
    PROMETHEUS = "prometheus" # Prometheus format
    STATSD = "statsd"        # StatsD protocol
    CUSTOM = "custom"        # Custom handler


@dataclass
class MetricRule:
    """Rule for extracting metrics from log messages."""
    name: str                                    # Metric name
    metric_type: MetricType                     # Type of metric
    source_field: str                           # Field to extract value from
    label_fields: List[str] = field(default_factory=list)  # Fields to use as labels
    filter_conditions: Dict[str, Any] = field(default_factory=dict)  # Conditions for metric extraction
    value_transformer: Optional[Callable] = None  # Function to transform extracted value
    aggregation_window_seconds: float = 60.0    # Aggregation window for histograms/timers
    description: str = ""                       # Metric description


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    description: str = ""


class MetricsBuffer:
    """
    Thread-safe buffer for metric aggregation and batching.
    
    Handles different metric types with appropriate aggregation strategies
    and provides efficient batch processing for high-throughput scenarios.
    """
    
    def __init__(self, max_size: int = 10000, flush_interval: float = 10.0):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._lock = threading.RLock()
        
        # Storage for different metric types
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._sets: Dict[str, Set[str]] = defaultdict(set)
        
        # Metadata storage
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Buffer management
        self._last_flush = time.time()
        self._total_metrics = 0
        
        # Logger
        self._logger = logging.getLogger(f"{__name__}.MetricsBuffer")
    
    def add_metric(self, metric: MetricValue) -> None:
        """Add metric to buffer with appropriate aggregation."""
        try:
            with self._lock:
                metric_key = self._get_metric_key(metric)
                
                # Store metadata
                self._metric_metadata[metric_key] = {
                    'labels': metric.labels,
                    'description': metric.description,
                    'type': metric.metric_type.value
                }
                
                # Add to appropriate storage based on type
                if metric.metric_type == MetricType.COUNTER:
                    self._counters[metric_key] += metric.value
                
                elif metric.metric_type == MetricType.GAUGE:
                    self._gauges[metric_key] = metric.value
                
                elif metric.metric_type == MetricType.HISTOGRAM:
                    self._histograms[metric_key].append(metric.value)
                    # Limit histogram samples
                    if len(self._histograms[metric_key]) > 1000:
                        self._histograms[metric_key] = self._histograms[metric_key][-1000:]
                
                elif metric.metric_type == MetricType.TIMER:
                    self._timers[metric_key].append(metric.value)
                    # Limit timer samples
                    if len(self._timers[metric_key]) > 1000:
                        self._timers[metric_key] = self._timers[metric_key][-1000:]
                
                elif metric.metric_type == MetricType.SET:
                    self._sets[metric_key].add(str(metric.value))
                    # Limit set size
                    if len(self._sets[metric_key]) > 10000:
                        oldest_items = list(self._sets[metric_key])[:1000]
                        for item in oldest_items:
                            self._sets[metric_key].discard(item)
                
                self._total_metrics += 1
                
        except Exception as e:
            self._logger.error(f"Failed to add metric {metric.name}: {e}")
    
    def _get_metric_key(self, metric: MetricValue) -> str:
        """Generate unique key for metric including labels."""
        if metric.labels:
            labels_str = ",".join(f"{k}={v}" for k, v in sorted(metric.labels.items()))
            return f"{metric.name}|{labels_str}"
        return metric.name
    
    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        with self._lock:
            return (
                self._total_metrics >= self.max_size or
                time.time() - self._last_flush >= self.flush_interval
            )
    
    def flush(self) -> Dict[str, List[MetricValue]]:
        """Flush buffer and return aggregated metrics."""
        try:
            with self._lock:
                current_time = time.time()
                flushed_metrics = {
                    'counters': [],
                    'gauges': [],
                    'histograms': [],
                    'timers': [],
                    'sets': []
                }
                
                # Process counters
                for key, value in self._counters.items():
                    metadata = self._metric_metadata.get(key, {})
                    metric = MetricValue(
                        name=key.split('|')[0],
                        value=value,
                        metric_type=MetricType.COUNTER,
                        labels=metadata.get('labels', {}),
                        timestamp=current_time,
                        description=metadata.get('description', '')
                    )
                    flushed_metrics['counters'].append(metric)
                
                # Process gauges
                for key, value in self._gauges.items():
                    metadata = self._metric_metadata.get(key, {})
                    metric = MetricValue(
                        name=key.split('|')[0],
                        value=value,
                        metric_type=MetricType.GAUGE,
                        labels=metadata.get('labels', {}),
                        timestamp=current_time,
                        description=metadata.get('description', '')
                    )
                    flushed_metrics['gauges'].append(metric)
                
                # Process histograms
                for key, values in self._histograms.items():
                    if values:
                        metadata = self._metric_metadata.get(key, {})
                        # Create histogram summary metrics
                        base_name = key.split('|')[0]
                        labels = metadata.get('labels', {})
                        
                        # Count
                        count_metric = MetricValue(
                            name=f"{base_name}_count",
                            value=len(values),
                            metric_type=MetricType.COUNTER,
                            labels=labels,
                            timestamp=current_time
                        )
                        flushed_metrics['histograms'].append(count_metric)
                        
                        # Sum
                        sum_metric = MetricValue(
                            name=f"{base_name}_sum",
                            value=sum(values),
                            metric_type=MetricType.COUNTER,
                            labels=labels,
                            timestamp=current_time
                        )
                        flushed_metrics['histograms'].append(sum_metric)
                        
                        # Percentiles
                        sorted_values = sorted(values)
                        percentiles = [50, 95, 99]
                        for p in percentiles:
                            idx = int(len(sorted_values) * p / 100)
                            percentile_labels = labels.copy()
                            percentile_labels['quantile'] = f"0.{p:02d}"
                            
                            percentile_metric = MetricValue(
                                name=f"{base_name}",
                                value=sorted_values[min(idx, len(sorted_values) - 1)],
                                metric_type=MetricType.GAUGE,
                                labels=percentile_labels,
                                timestamp=current_time
                            )
                            flushed_metrics['histograms'].append(percentile_metric)
                
                # Process timers (similar to histograms)
                for key, values in self._timers.items():
                    if values:
                        metadata = self._metric_metadata.get(key, {})
                        base_name = key.split('|')[0]
                        labels = metadata.get('labels', {})
                        
                        # Duration statistics
                        duration_metrics = [
                            MetricValue(f"{base_name}_duration_count", len(values), MetricType.COUNTER, labels, current_time),
                            MetricValue(f"{base_name}_duration_sum", sum(values), MetricType.COUNTER, labels, current_time),
                            MetricValue(f"{base_name}_duration_avg", mean(values), MetricType.GAUGE, labels, current_time),
                            MetricValue(f"{base_name}_duration_median", median(values), MetricType.GAUGE, labels, current_time),
                            MetricValue(f"{base_name}_duration_max", max(values), MetricType.GAUGE, labels, current_time),
                            MetricValue(f"{base_name}_duration_min", min(values), MetricType.GAUGE, labels, current_time)
                        ]
                        flushed_metrics['timers'].extend(duration_metrics)
                
                # Process sets
                for key, value_set in self._sets.items():
                    metadata = self._metric_metadata.get(key, {})
                    metric = MetricValue(
                        name=key.split('|')[0],
                        value=len(value_set),
                        metric_type=MetricType.GAUGE,
                        labels=metadata.get('labels', {}),
                        timestamp=current_time,
                        description=metadata.get('description', '')
                    )
                    flushed_metrics['sets'].append(metric)
                
                # Clear buffers
                self._counters.clear()
                self._gauges.clear()
                self._histograms.clear()
                self._timers.clear()
                self._sets.clear()
                self._metric_metadata.clear()
                
                # Reset state
                self._last_flush = current_time
                self._total_metrics = 0
                
                return flushed_metrics
                
        except Exception as e:
            self._logger.error(f"Buffer flush failed: {e}")
            return {}
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                'total_metrics': self._total_metrics,
                'counters': len(self._counters),
                'gauges': len(self._gauges),
                'histograms': len(self._histograms),
                'timers': len(self._timers),
                'sets': len(self._sets),
                'last_flush': self._last_flush,
                'time_since_flush': time.time() - self._last_flush
            }


class MetricsSink(BaseSink):
    """
    Metrics collection and publishing sink.
    
    Converts log messages into structured metrics using configurable rules
    and publishes them to various destinations for monitoring and alerting.
    
    Features:
    - Flexible metric extraction rules
    - Multiple destination support
    - Metric aggregation and buffering
    - Integration with existing monitoring
    - Performance tracking
    - Error rate monitoring
    """
    
    def __init__(self,
                 name: str = "metrics",
                 formatter: Optional[Any] = None,
                 metric_rules: Optional[List[MetricRule]] = None,
                 destinations: Optional[List[MetricDestination]] = None,
                 buffer_size: int = 10000,
                 flush_interval: float = 10.0,
                 enable_default_metrics: bool = True,
                 enable_monitoring_integration: bool = True,
                 statsd_host: str = "localhost",
                 statsd_port: int = 8125,
                 **base_kwargs):
        """
        Initialize metrics sink.
        
        Args:
            name: Sink name for identification
            formatter: Message formatter (defaults to StructuredFormatter)
            metric_rules: Custom metric extraction rules
            destinations: List of metric destinations
            buffer_size: Maximum metrics in buffer before flush
            flush_interval: Buffer flush interval in seconds
            enable_default_metrics: Enable built-in metric extraction
            enable_monitoring_integration: Integrate with existing monitoring
            statsd_host: StatsD server hostname
            statsd_port: StatsD server port
            **base_kwargs: Arguments for BaseSink
        """
        # Set up formatter before calling parent __init__
        if formatter is None:
            formatter = StructuredFormatter(
                timestamp_format="unix",
                include_performance_metrics=True,
                include_system_info=False  # Reduce overhead for metrics
            )
        
        # Initialize parent
        super().__init__(name=name, formatter=formatter, **base_kwargs)
        
        # Configuration
        self.metric_rules = metric_rules or []
        self.destinations = destinations or [MetricDestination.INTERNAL]
        self.enable_default_metrics = enable_default_metrics
        self.enable_monitoring_integration = enable_monitoring_integration
        self.statsd_host = statsd_host
        self.statsd_port = statsd_port
        
        # Metrics buffer
        self.buffer = MetricsBuffer(max_size=buffer_size, flush_interval=flush_interval)
        
        # Default metric rules
        if enable_default_metrics:
            self._add_default_metric_rules()
        
        # Destination handlers
        self._destination_handlers = {
            MetricDestination.INTERNAL: self._handle_internal_metrics,
            MetricDestination.PROMETHEUS: self._handle_prometheus_metrics,
            MetricDestination.STATSD: self._handle_statsd_metrics,
            MetricDestination.CUSTOM: self._handle_custom_metrics
        }
        
        # State tracking
        self._metrics_extracted = 0
        self._metrics_published = 0
        self._extraction_errors = 0
        self._last_flush_time = time.time()
        
        # Monitoring integration
        self._performance_monitor = None
        self._resource_manager = None
        
        if enable_monitoring_integration and MONITORING_AVAILABLE:
            try:
                self._performance_monitor = get_performance_monitor()
                self._resource_manager = get_resource_manager()
            except Exception as e:
                self._logger.warning(f"Monitoring integration failed: {e}")
        
        # Internal metrics storage
        self._internal_metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._internal_metrics_lock = threading.RLock()
        
        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._flush_thread_stop = threading.Event()
        
        self._logger.info(f"MetricsSink '{name}' initialized",
                         extra={
                             'metric_rules': len(self.metric_rules),
                             'destinations': [d.value for d in self.destinations],
                             'buffer_size': buffer_size,
                             'flush_interval': flush_interval,
                             'monitoring_integration': enable_monitoring_integration and MONITORING_AVAILABLE
                         })
    
    def _add_default_metric_rules(self) -> None:
        """Add default metric extraction rules."""
        default_rules = [
            # Error rate tracking
            MetricRule(
                name="log_messages_total",
                metric_type=MetricType.COUNTER,
                source_field="level",
                label_fields=["level", "logger"],
                description="Total log messages by level and logger"
            ),
            
            # Performance metrics
            MetricRule(
                name="operation_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                source_field="performance.operation_duration_ms",
                value_transformer=lambda x: x / 1000.0 if isinstance(x, (int, float)) else 0,
                label_fields=["operation.operation_name", "logger"],
                filter_conditions={"performance.operation_duration_ms": lambda x: x is not None},
                description="Operation duration in seconds"
            ),
            
            # Memory usage
            MetricRule(
                name="memory_usage_mb",
                metric_type=MetricType.GAUGE,
                source_field="performance.memory_used_mb",
                label_fields=["logger"],
                filter_conditions={"performance.memory_used_mb": lambda x: x is not None},
                description="Memory usage in MB"
            ),
            
            # Error tracking
            MetricRule(
                name="errors_total",
                metric_type=MetricType.COUNTER,
                source_field="level",
                label_fields=["error.exception_type", "logger"],
                filter_conditions={"level": lambda x: x >= logging.ERROR},
                description="Total errors by type and logger"
            ),
            
            # Response time percentiles
            MetricRule(
                name="response_time_seconds",
                metric_type=MetricType.TIMER,
                source_field="context.response_time_ms",
                value_transformer=lambda x: x / 1000.0 if isinstance(x, (int, float)) else 0,
                label_fields=["context.endpoint", "context.method"],
                filter_conditions={"context.response_time_ms": lambda x: x is not None},
                description="HTTP response time in seconds"
            )
        ]
        
        self.metric_rules.extend(default_rules)
    
    def _initialize_sink(self) -> bool:
        """Initialize metrics sink."""
        try:
            # Start background flush thread
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name=f"MetricsSink-{self.name}-Flush",
                daemon=True
            )
            self._flush_thread.start()
            
            return True
            
        except Exception as e:
            self._logger.error(f"Metrics sink initialization failed: {e}")
            return False
    
    def _cleanup_sink(self) -> bool:
        """Cleanup metrics sink."""
        try:
            # Stop flush thread
            if self._flush_thread and self._flush_thread.is_alive():
                self._flush_thread_stop.set()
                self._flush_thread.join(timeout=5.0)
            
            # Final flush
            self._flush_metrics()
            
            self._logger.info(f"MetricsSink '{self.name}' cleanup completed",
                             extra={
                                 'metrics_extracted': self._metrics_extracted,
                                 'metrics_published': self._metrics_published,
                                 'extraction_errors': self._extraction_errors
                             })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Metrics sink cleanup failed: {e}")
            return False
    
    def _health_check_sink(self) -> bool:
        """Perform health check on metrics sink."""
        try:
            # Check if flush thread is running
            if self._flush_thread and not self._flush_thread.is_alive():
                return False
            
            # Check buffer health
            buffer_stats = self.buffer.get_buffer_stats()
            if buffer_stats['time_since_flush'] > self.buffer.flush_interval * 3:
                return False
            
            # Check StatsD connectivity if using StatsD
            if MetricDestination.STATSD in self.destinations:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.settimeout(1.0)
                    sock.connect((self.statsd_host, self.statsd_port))
                    sock.close()
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            self._logger.warning(f"Metrics sink health check failed: {e}")
            return False
    
    def _write_message(self, formatted_message: str, message: LogMessage) -> bool:
        """Extract metrics from log message."""
        try:
            # Parse formatted message back to structured data
            try:
                log_data = json.loads(formatted_message)
            except json.JSONDecodeError:
                # If JSON parsing fails, create minimal structure
                log_data = {
                    'level': message.level,
                    'logger': message.logger_name,
                    'message': message.message,
                    'timestamp': message.timestamp
                }
            
            # Extract metrics using rules
            metrics_extracted = 0
            for rule in self.metric_rules:
                try:
                    metric = self._extract_metric(log_data, rule)
                    if metric:
                        self.buffer.add_metric(metric)
                        metrics_extracted += 1
                except Exception as e:
                    self._extraction_errors += 1
                    self._logger.debug(f"Metric extraction failed for rule {rule.name}: {e}")
            
            self._metrics_extracted += metrics_extracted
            
            # Check if buffer should be flushed
            if self.buffer.should_flush():
                self._flush_metrics()
            
            return True
            
        except Exception as e:
            self._logger.error(f"Metrics extraction failed: {e}")
            return False
    
    def _extract_metric(self, log_data: Dict[str, Any], rule: MetricRule) -> Optional[MetricValue]:
        """Extract metric from log data using rule."""
        try:
            # Check filter conditions
            for condition_field, condition_value in rule.filter_conditions.items():
                field_value = self._get_nested_field(log_data, condition_field)
                
                if callable(condition_value):
                    if not condition_value(field_value):
                        return None
                else:
                    if field_value != condition_value:
                        return None
            
            # Extract source value
            source_value = self._get_nested_field(log_data, rule.source_field)
            if source_value is None:
                return None
            
            # Transform value if transformer provided
            if rule.value_transformer:
                source_value = rule.value_transformer(source_value)
            
            # Ensure value is numeric for most metric types
            if rule.metric_type != MetricType.SET:
                if not isinstance(source_value, (int, float)):
                    try:
                        source_value = float(source_value)
                    except (ValueError, TypeError):
                        return None
            
            # Extract labels
            labels = {}
            for label_field in rule.label_fields:
                label_value = self._get_nested_field(log_data, label_field)
                if label_value is not None:
                    labels[label_field.replace('.', '_')] = str(label_value)
            
            # Create metric
            return MetricValue(
                name=rule.name,
                value=source_value,
                metric_type=rule.metric_type,
                labels=labels,
                timestamp=log_data.get('timestamp'),
                description=rule.description
            )
            
        except Exception as e:
            self._logger.debug(f"Metric extraction failed for {rule.name}: {e}")
            return None
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation."""
        try:
            keys = field_path.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._flush_thread_stop.is_set():
            try:
                if self.buffer.should_flush():
                    self._flush_metrics()
                
                # Sleep for a short interval
                self._flush_thread_stop.wait(1.0)
                
            except Exception as e:
                self._logger.error(f"Flush loop error: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _flush_metrics(self) -> None:
        """Flush buffered metrics to destinations."""
        try:
            # Get metrics from buffer
            metrics_by_type = self.buffer.flush()
            
            if not any(metrics_by_type.values()):
                return  # No metrics to flush
            
            # Flatten metrics for processing
            all_metrics = []
            for metric_list in metrics_by_type.values():
                all_metrics.extend(metric_list)
            
            if not all_metrics:
                return
            
            # Send to each destination
            for destination in self.destinations:
                try:
                    handler = self._destination_handlers.get(destination)
                    if handler:
                        handler(all_metrics)
                        self._metrics_published += len(all_metrics)
                except Exception as e:
                    self._logger.error(f"Failed to publish to {destination.value}: {e}")
            
            self._last_flush_time = time.time()
            
            self._logger.debug(f"Flushed {len(all_metrics)} metrics to {len(self.destinations)} destinations")
            
        except Exception as e:
            self._logger.error(f"Metrics flush failed: {e}")
    
    def _handle_internal_metrics(self, metrics: List[MetricValue]) -> None:
        """Handle internal metrics storage."""
        try:
            with self._internal_metrics_lock:
                for metric in metrics:
                    self._internal_metrics[metric.name].append(metric)
                    
                    # Limit stored metrics per name
                    if len(self._internal_metrics[metric.name]) > 1000:
                        self._internal_metrics[metric.name] = self._internal_metrics[metric.name][-500:]
                        
        except Exception as e:
            self._logger.error(f"Internal metrics handling failed: {e}")
    
    def _handle_prometheus_metrics(self, metrics: List[MetricValue]) -> None:
        """Handle Prometheus format metrics."""
        try:
            # Convert to Prometheus format
            prometheus_lines = []
            
            for metric in metrics:
                # Help line
                if metric.description:
                    prometheus_lines.append(f"# HELP {metric.name} {metric.description}")
                
                # Type line
                metric_type_map = {
                    MetricType.COUNTER: "counter",
                    MetricType.GAUGE: "gauge",
                    MetricType.HISTOGRAM: "histogram",
                    MetricType.TIMER: "histogram",
                    MetricType.SET: "gauge"
                }
                prom_type = metric_type_map.get(metric.metric_type, "gauge")
                prometheus_lines.append(f"# TYPE {metric.name} {prom_type}")
                
                # Metric line
                if metric.labels:
                    labels_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
                    metric_line = f"{metric.name}{{{labels_str}}} {metric.value}"
                else:
                    metric_line = f"{metric.name} {metric.value}"
                
                if metric.timestamp:
                    metric_line += f" {int(metric.timestamp * 1000)}"
                
                prometheus_lines.append(metric_line)
            
            # In a real implementation, this would be sent to Prometheus
            # For now, we'll log it at debug level
            prometheus_output = "\n".join(prometheus_lines)
            self._logger.debug(f"Prometheus metrics:\n{prometheus_output}")
            
        except Exception as e:
            self._logger.error(f"Prometheus metrics handling failed: {e}")
    
    def _handle_statsd_metrics(self, metrics: List[MetricValue]) -> None:
        """Handle StatsD protocol metrics."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            for metric in metrics:
                # Convert to StatsD format
                metric_name = metric.name
                
                # Add labels as tags (if supported)
                if metric.labels:
                    tags = ",".join(f"{k}:{v}" for k, v in metric.labels.items())
                    metric_name = f"{metric_name}|#{tags}"
                
                # Format based on metric type
                if metric.metric_type == MetricType.COUNTER:
                    statsd_line = f"{metric_name}:{metric.value}|c"
                elif metric.metric_type == MetricType.GAUGE:
                    statsd_line = f"{metric_name}:{metric.value}|g"
                elif metric.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                    statsd_line = f"{metric_name}:{metric.value}|h"
                elif metric.metric_type == MetricType.SET:
                    statsd_line = f"{metric_name}:{metric.value}|s"
                else:
                    continue
                
                # Send to StatsD server
                sock.sendto(statsd_line.encode('utf-8'), (self.statsd_host, self.statsd_port))
            
            sock.close()
            
        except Exception as e:
            self._logger.error(f"StatsD metrics handling failed: {e}")
    
    def _handle_custom_metrics(self, metrics: List[MetricValue]) -> None:
        """Handle custom metrics (placeholder for user implementation)."""
        # This is a placeholder for custom metric handling
        # Users can override this method or provide custom handlers
        pass
    
    def get_metrics_stats(self) -> Dict[str, Any]:
        """Get comprehensive metrics sink statistics."""
        base_stats = self.get_stats().to_dict()
        
        # Metrics-specific statistics
        metrics_stats = {
            'metrics_extracted': self._metrics_extracted,
            'metrics_published': self._metrics_published,
            'extraction_errors': self._extraction_errors,
            'last_flush_time': self._last_flush_time,
            'metric_rules_count': len(self.metric_rules),
            'destinations': [d.value for d in self.destinations],
            'buffer_stats': self.buffer.get_buffer_stats()
        }
        
        # Internal metrics summary
        with self._internal_metrics_lock:
            metrics_stats['internal_metrics_summary'] = {
                name: len(values) for name, values in self._internal_metrics.items()
            }
        
        # Merge with base stats
        base_stats.update(metrics_stats)
        return base_stats
    
    def get_internal_metrics(self, metric_name: Optional[str] = None, 
                           limit: int = 100) -> Dict[str, List[MetricValue]]:
        """Get internal metrics data."""
        with self._internal_metrics_lock:
            if metric_name:
                return {metric_name: self._internal_metrics.get(metric_name, [])[-limit:]}
            else:
                return {
                    name: values[-limit:] for name, values in self._internal_metrics.items()
                }
    
    def add_metric_rule(self, rule: MetricRule) -> None:
        """Add custom metric extraction rule."""
        self.metric_rules.append(rule)
        self._logger.info(f"Added metric rule: {rule.name}")
    
    def force_flush(self) -> None:
        """Force immediate flush of buffered metrics."""
        self._flush_metrics()


# Module exports
__all__ = [
    'MetricsSink',
    'MetricRule',
    'MetricValue',
    'MetricType',
    'MetricDestination'
]