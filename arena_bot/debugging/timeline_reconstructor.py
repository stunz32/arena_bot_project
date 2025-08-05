"""
Timeline Reconstruction System for Arena Bot Deep Debugging

Advanced timeline reconstruction and event correlation system that provides:
- Comprehensive event timeline reconstruction with microsecond precision
- Multi-source event correlation (logs, traces, state changes, performance metrics)
- Causality analysis with event dependency mapping and impact tracking
- Interactive timeline visualization with filtering and search capabilities
- Event pattern detection with anomaly identification and trend analysis
- Root cause timeline analysis with backward and forward event tracing
- Timeline-based debugging with precise event sequence reproduction
- Historical timeline comparison with regression detection and baseline analysis
- Automated timeline summarization with key event extraction and impact assessment

This system helps reconstruct the exact sequence of events leading to complex
issues, making it possible to understand causality and debug time-sensitive
problems that span multiple components and systems.
"""

import time
import threading
import bisect
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import json
import statistics
from uuid import uuid4
import heapq

# Import debugging components
from .enhanced_logger import get_enhanced_logger
from .method_tracer import get_method_tracer
from .state_monitor import get_state_monitor, StateChangeEvent
from .pipeline_tracer import get_pipeline_tracer
from .health_monitor import get_health_monitor
from .error_analyzer import get_error_analyzer
from .exception_handler import get_exception_handler
from .performance_analyzer import get_performance_analyzer

from ..logging_system.logger import get_logger, LogLevel


class EventType(Enum):
    """Types of events that can be tracked in the timeline."""
    METHOD_CALL = "method_call"
    STATE_CHANGE = "state_change"
    PIPELINE_STAGE = "pipeline_stage"
    HEALTH_CHECK = "health_check"
    ERROR_OCCURRENCE = "error_occurrence"
    EXCEPTION = "exception"
    PERFORMANCE_METRIC = "performance_metric"
    LOG_ENTRY = "log_entry"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    NETWORK_ACTIVITY = "network_activity"
    DATABASE_OPERATION = "database_operation"


class EventSeverity(Enum):
    """Event severity levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TimelineEvent:
    """Represents a single event in the timeline."""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Event identification
    event_type: EventType = EventType.SYSTEM_EVENT
    severity: EventSeverity = EventSeverity.INFO
    source: str = ""  # Component or system that generated the event
    
    # Event details
    title: str = ""
    description: str = ""
    
    # Context data
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    parent_event_id: Optional[str] = None
    child_event_ids: Set[str] = field(default_factory=set)
    related_event_ids: Set[str] = field(default_factory=set)
    
    # Performance data
    duration_ms: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Impact assessment
    impact_score: float = 0.0
    affected_components: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'source': self.source,
            'title': self.title,
            'description': self.description,
            'context': self.context,
            'metadata': self.metadata,
            'parent_event_id': self.parent_event_id,
            'child_event_ids': list(self.child_event_ids),
            'related_event_ids': list(self.related_event_ids),
            'duration_ms': self.duration_ms,
            'resource_usage': self.resource_usage,
            'impact_score': self.impact_score,
            'affected_components': self.affected_components
        }


@dataclass
class TimelineSegment:
    """Represents a segment of the timeline for analysis."""
    
    segment_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    
    # Segment metadata
    title: str = ""
    description: str = ""
    
    # Events in this segment
    events: List[TimelineEvent] = field(default_factory=list)
    
    # Analysis results
    key_events: List[str] = field(default_factory=list)  # Event IDs
    patterns_detected: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance summary
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration_seconds(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time
    
    def get_event_count(self) -> int:
        """Get number of events in segment."""
        return len(self.events)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        return {
            'segment_id': self.segment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_datetime': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_datetime': datetime.fromtimestamp(self.end_time).isoformat(),
            'duration_seconds': self.get_duration_seconds(),
            'title': self.title,
            'description': self.description,
            'event_count': self.get_event_count(),
            'events': [event.to_dict() for event in self.events],
            'key_events': self.key_events,
            'patterns_detected': self.patterns_detected,
            'anomalies': self.anomalies,
            'performance_summary': self.performance_summary
        }


class EventCollector:
    """Collects events from various debugging components."""
    
    def __init__(self, timeline_reconstructor):
        """Initialize event collector."""
        self.logger = get_enhanced_logger("arena_bot.debugging.timeline_reconstructor.collector")
        self.timeline_reconstructor = timeline_reconstructor
        
        # Collection state
        self.collection_enabled = False
        self.collection_stats = {
            'events_collected': 0,
            'collection_errors': 0,
            'last_collection_time': time.time()
        }
        
        # Integration with debugging components
        self.integrated_components = {}
        
    def start_collection(self) -> bool:
        """Start collecting events from debugging components."""
        try:
            self.collection_enabled = True
            
            # Initialize integrations
            self._initialize_integrations()
            
            self.logger.info("📊 Event collection started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start event collection: {e}")
            return False
    
    def stop_collection(self) -> None:
        """Stop collecting events."""
        self.collection_enabled = False
        self.logger.info("📊 Event collection stopped")
    
    def _initialize_integrations(self) -> None:
        """Initialize integrations with debugging components."""
        try:
            # Method tracer integration
            try:
                method_tracer = get_method_tracer()
                self.integrated_components['method_tracer'] = method_tracer
                self.logger.debug("Integrated with method tracer")
            except Exception:
                pass
            
            # State monitor integration
            try:
                state_monitor = get_state_monitor()
                self.integrated_components['state_monitor'] = state_monitor
                self.logger.debug("Integrated with state monitor")
            except Exception:
                pass
            
            # Pipeline tracer integration
            try:
                pipeline_tracer = get_pipeline_tracer()
                self.integrated_components['pipeline_tracer'] = pipeline_tracer
                self.logger.debug("Integrated with pipeline tracer")
            except Exception:
                pass
            
            # Health monitor integration
            try:
                health_monitor = get_health_monitor()
                self.integrated_components['health_monitor'] = health_monitor
                self.logger.debug("Integrated with health monitor")
            except Exception:
                pass
            
            # Exception handler integration
            try:
                exception_handler = get_exception_handler()
                self.integrated_components['exception_handler'] = exception_handler
                self.logger.debug("Integrated with exception handler")
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integrations: {e}")
    
    def collect_method_trace_event(self, trace_data: Dict[str, Any]) -> None:
        """Collect method trace event."""
        if not self.collection_enabled:
            return
        
        try:
            event = TimelineEvent(
                event_type=EventType.METHOD_CALL,
                severity=EventSeverity.TRACE,
                source=f"method_tracer.{trace_data.get('component_name', 'unknown')}",
                title=f"Method Call: {trace_data.get('method_name', 'unknown')}",
                description=f"Called {trace_data.get('method_name')} with {len(trace_data.get('parameters', {}))} parameters",
                context={
                    'method_name': trace_data.get('method_name'),
                    'component_name': trace_data.get('component_name'),
                    'parameters': trace_data.get('parameters', {}),
                    'return_value': trace_data.get('return_value'),
                    'trace_id': trace_data.get('trace_id')
                },
                duration_ms=trace_data.get('execution_time_ms'),
                resource_usage={
                    'memory_mb': trace_data.get('memory_usage_mb', 0),
                    'cpu_percent': trace_data.get('cpu_usage', 0)
                }
            )
            
            self.timeline_reconstructor.add_event(event)
            self.collection_stats['events_collected'] += 1
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Failed to collect method trace event: {e}")
    
    def collect_state_change_event(self, state_change: StateChangeEvent) -> None:
        """Collect state change event."""
        if not self.collection_enabled:
            return
        
        try:
            severity = EventSeverity.INFO
            if hasattr(state_change, 'is_critical') and state_change.is_critical:
                severity = EventSeverity.WARNING
            
            event = TimelineEvent(
                event_type=EventType.STATE_CHANGE,
                severity=severity,
                source=f"state_monitor.{state_change.component_name}",
                title=f"State Change: {state_change.property_name}",
                description=f"Changed from {state_change.old_value} to {state_change.new_value}",
                context={
                    'component_name': state_change.component_name,
                    'property_name': state_change.property_name,
                    'old_value': state_change.old_value,
                    'new_value': state_change.new_value,
                    'change_type': getattr(state_change, 'change_type', 'unknown')
                },
                impact_score=getattr(state_change, 'impact_score', 0.5),
                affected_components=[state_change.component_name]
            )
            
            self.timeline_reconstructor.add_event(event)
            self.collection_stats['events_collected'] += 1
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Failed to collect state change event: {e}")
    
    def collect_exception_event(self, exception_context: Dict[str, Any]) -> None:
        """Collect exception event."""
        if not self.collection_enabled:
            return
        
        try:
            event = TimelineEvent(
                event_type=EventType.EXCEPTION,
                severity=EventSeverity.ERROR,
                source=f"exception_handler.{exception_context.get('component_name', 'unknown')}",
                title=f"Exception: {exception_context.get('exception_type', 'Unknown')}",
                description=exception_context.get('exception_message', 'No message'),
                context={
                    'exception_type': exception_context.get('exception_type'),
                    'exception_message': exception_context.get('exception_message'),
                    'component_name': exception_context.get('component_name'),
                    'method_name': exception_context.get('method_name'),
                    'stack_trace': exception_context.get('stack_trace'),
                    'exception_id': exception_context.get('exception_id')
                },
                impact_score=0.8,  # Exceptions have high impact
                affected_components=[exception_context.get('component_name', 'unknown')]
            )
            
            self.timeline_reconstructor.add_event(event)
            self.collection_stats['events_collected'] += 1
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Failed to collect exception event: {e}")
    
    def collect_performance_event(self, performance_data: Dict[str, Any]) -> None:
        """Collect performance metric event."""
        if not self.collection_enabled:
            return
        
        try:
            # Determine severity based on performance data
            severity = EventSeverity.DEBUG
            cpu_percent = performance_data.get('cpu_percent', 0)
            memory_percent = performance_data.get('memory_percent', 0)
            
            if cpu_percent > 90 or memory_percent > 90:
                severity = EventSeverity.CRITICAL
            elif cpu_percent > 70 or memory_percent > 80:
                severity = EventSeverity.WARNING
            elif cpu_percent > 50 or memory_percent > 70:
                severity = EventSeverity.INFO
            
            event = TimelineEvent(
                event_type=EventType.PERFORMANCE_METRIC,
                severity=severity,
                source="performance_analyzer",
                title="Performance Metrics Update",
                description=f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%",
                context=performance_data,
                resource_usage={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_mb': performance_data.get('process_memory_mb', 0),
                    'threads': performance_data.get('process_threads', 0)
                },
                impact_score=min((cpu_percent + memory_percent) / 200, 1.0)
            )
            
            self.timeline_reconstructor.add_event(event)
            self.collection_stats['events_collected'] += 1
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Failed to collect performance event: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get event collection statistics."""
        return {
            'collection_enabled': self.collection_enabled,
            'events_collected': self.collection_stats['events_collected'],
            'collection_errors': self.collection_stats['collection_errors'],
            'integrated_components': list(self.integrated_components.keys()),
            'last_collection_time': self.collection_stats['last_collection_time']
        }


class TimelineAnalyzer:
    """Analyzes timeline events for patterns and anomalies."""
    
    def __init__(self):
        """Initialize timeline analyzer."""
        self.logger = get_enhanced_logger("arena_bot.debugging.timeline_reconstructor.analyzer")
        
        # Analysis configuration
        self.pattern_detection_enabled = True
        self.anomaly_detection_enabled = True
        self.causality_analysis_enabled = True
        
        # Pattern detection parameters
        self.min_pattern_frequency = 3
        self.pattern_time_window_seconds = 300  # 5 minutes
        
        # Anomaly detection parameters
        self.anomaly_threshold_std_devs = 2.0
        self.min_events_for_baseline = 50
    
    def analyze_timeline_segment(self, segment: TimelineSegment) -> TimelineSegment:
        """Analyze a timeline segment for patterns and anomalies."""
        
        start_time = time.perf_counter()
        
        try:
            # Detect patterns
            if self.pattern_detection_enabled:
                segment.patterns_detected = self._detect_patterns(segment.events)
            
            # Detect anomalies
            if self.anomaly_detection_enabled:
                segment.anomalies = self._detect_anomalies(segment.events)
            
            # Identify key events
            segment.key_events = self._identify_key_events(segment.events)
            
            # Generate performance summary
            segment.performance_summary = self._generate_performance_summary(segment.events)
            
            # Causality analysis
            if self.causality_analysis_enabled:
                self._analyze_causality(segment.events)
            
            analysis_time = (time.perf_counter() - start_time) * 1000
            
            self.logger.debug(
                f"Timeline segment analyzed in {analysis_time:.2f}ms: "
                f"{len(segment.events)} events, {len(segment.patterns_detected)} patterns, "
                f"{len(segment.anomalies)} anomalies"
            )
            
            return segment
            
        except Exception as e:
            self.logger.error(f"Timeline analysis failed: {e}")
            return segment
    
    def _detect_patterns(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect recurring patterns in events."""
        patterns = []
        
        # Group events by type and source
        event_groups = defaultdict(list)
        for event in events:
            group_key = f"{event.event_type.value}:{event.source}"
            event_groups[group_key].append(event)
        
        # Detect frequency patterns
        for group_key, group_events in event_groups.items():
            if len(group_events) >= self.min_pattern_frequency:
                # Calculate timing patterns
                timestamps = [event.timestamp for event in group_events]
                intervals = [timestamps[i+1] - timestamps[i] 
                           for i in range(len(timestamps) - 1)]
                
                if intervals:
                    avg_interval = statistics.mean(intervals)
                    std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
                    
                    pattern = {
                        'pattern_type': 'frequency',
                        'event_type': group_key.split(':')[0],
                        'source': group_key.split(':')[1],
                        'frequency': len(group_events),
                        'average_interval_seconds': avg_interval,
                        'interval_std_dev': std_interval,
                        'regularity_score': 1.0 / (1.0 + std_interval / max(avg_interval, 0.1)),
                        'confidence': min(len(group_events) / 10.0, 1.0)
                    }
                    patterns.append(pattern)
        
        # Detect sequence patterns
        sequence_patterns = self._detect_sequence_patterns(events)
        patterns.extend(sequence_patterns)
        
        return patterns
    
    def _detect_sequence_patterns(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect event sequence patterns."""
        patterns = []
        
        # Look for common sequences (A -> B -> C)
        sequence_map = defaultdict(int)
        
        for i in range(len(events) - 2):
            sequence = (
                events[i].event_type.value,
                events[i+1].event_type.value,
                events[i+2].event_type.value
            )
            sequence_map[sequence] += 1
        
        # Identify frequent sequences
        for sequence, count in sequence_map.items():
            if count >= self.min_pattern_frequency:
                pattern = {
                    'pattern_type': 'sequence',
                    'sequence': list(sequence),
                    'frequency': count,
                    'confidence': min(count / 5.0, 1.0)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_anomalies(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect anomalies in event timeline."""
        anomalies = []
        
        if len(events) < self.min_events_for_baseline:
            return anomalies
        
        # Time-based anomaly detection
        timestamps = [event.timestamp for event in events]
        time_intervals = [timestamps[i+1] - timestamps[i] 
                         for i in range(len(timestamps) - 1)]
        
        if len(time_intervals) > 10:
            mean_interval = statistics.mean(time_intervals)
            std_interval = statistics.stdev(time_intervals)
            
            # Find unusually long gaps
            for i, interval in enumerate(time_intervals):
                if abs(interval - mean_interval) > self.anomaly_threshold_std_devs * std_interval:
                    anomaly = {
                        'anomaly_type': 'time_gap',
                        'description': f'Unusual time gap: {interval:.2f}s (normal: {mean_interval:.2f}s)',
                        'event_index': i,
                        'severity': 'high' if interval > mean_interval + 3 * std_interval else 'medium',
                        'confidence': min(abs(interval - mean_interval) / (3 * std_interval), 1.0)
                    }
                    anomalies.append(anomaly)
        
        # Severity-based anomaly detection
        severity_counts = defaultdict(int)
        for event in events:
            severity_counts[event.severity.value] += 1
        
        total_events = len(events)
        error_rate = (severity_counts['error'] + severity_counts['critical']) / total_events
        
        if error_rate > 0.1:  # More than 10% errors
            anomaly = {
                'anomaly_type': 'high_error_rate',
                'description': f'High error rate: {error_rate:.1%}',
                'error_rate': error_rate,
                'severity': 'critical' if error_rate > 0.3 else 'high',
                'confidence': min(error_rate * 2, 1.0)
            }
            anomalies.append(anomaly)
        
        # Resource usage anomalies
        resource_anomalies = self._detect_resource_anomalies(events)
        anomalies.extend(resource_anomalies)
        
        return anomalies
    
    def _detect_resource_anomalies(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect resource usage anomalies."""
        anomalies = []
        
        # Collect resource usage data
        cpu_values = []
        memory_values = []
        
        for event in events:
            if event.resource_usage:
                if 'cpu_percent' in event.resource_usage:
                    cpu_values.append(event.resource_usage['cpu_percent'])
                if 'memory_percent' in event.resource_usage:
                    memory_values.append(event.resource_usage['memory_percent'])
                elif 'memory_mb' in event.resource_usage:
                    memory_values.append(event.resource_usage['memory_mb'])
        
        # Analyze CPU usage
        if len(cpu_values) > 10:
            mean_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            
            if max_cpu > 95:
                anomaly = {
                    'anomaly_type': 'cpu_spike',
                    'description': f'CPU spike detected: {max_cpu:.1f}% (avg: {mean_cpu:.1f}%)',
                    'max_value': max_cpu,
                    'average_value': mean_cpu,
                    'severity': 'critical',
                    'confidence': 0.9
                }
                anomalies.append(anomaly)
        
        # Analyze memory usage
        if len(memory_values) > 10:
            # Check for memory leaks (steadily increasing memory)
            if len(memory_values) > 20:
                # Calculate trend
                x_values = list(range(len(memory_values)))
                # Simple linear regression slope
                n = len(memory_values)
                sum_x = sum(x_values)
                sum_y = sum(memory_values)
                sum_xy = sum(x * y for x, y in zip(x_values, memory_values))
                sum_x2 = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                if slope > 0.1:  # Memory increasing significantly
                    anomaly = {
                        'anomaly_type': 'memory_leak',
                        'description': f'Potential memory leak detected (slope: {slope:.3f})',
                        'growth_rate': slope,
                        'severity': 'high',
                        'confidence': min(slope * 5, 1.0)
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _identify_key_events(self, events: List[TimelineEvent]) -> List[str]:
        """Identify key events in the timeline."""
        key_events = []
        
        # Sort events by impact score
        sorted_events = sorted(events, key=lambda e: e.impact_score, reverse=True)
        
        # Take top 20% or at least 5 events
        key_count = max(5, len(events) // 5)
        key_events = [event.event_id for event in sorted_events[:key_count]]
        
        # Always include critical and error events
        for event in events:
            if (event.severity in [EventSeverity.CRITICAL, EventSeverity.ERROR] and 
                event.event_id not in key_events):
                key_events.append(event.event_id)
        
        return key_events
    
    def _generate_performance_summary(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Generate performance summary for events."""
        summary = {
            'total_events': len(events),
            'event_types': {},
            'severity_distribution': {},
            'resource_usage': {},
            'timeline_health_score': 0.0
        }
        
        # Count event types
        for event in events:
            event_type = event.event_type.value
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
            
            severity = event.severity.value
            summary['severity_distribution'][severity] = summary['severity_distribution'].get(severity, 0) + 1
        
        # Aggregate resource usage
        cpu_values = []
        memory_values = []
        
        for event in events:
            if event.resource_usage:
                if 'cpu_percent' in event.resource_usage:
                    cpu_values.append(event.resource_usage['cpu_percent'])
                if 'memory_percent' in event.resource_usage:
                    memory_values.append(event.resource_usage['memory_percent'])
                elif 'memory_mb' in event.resource_usage:
                    memory_values.append(event.resource_usage['memory_mb'])
        
        if cpu_values:
            summary['resource_usage']['cpu'] = {
                'average': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            }
        
        if memory_values:
            summary['resource_usage']['memory'] = {
                'average': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            }
        
        # Calculate health score
        total_events = len(events)
        if total_events > 0:
            critical_count = summary['severity_distribution'].get('critical', 0)
            error_count = summary['severity_distribution'].get('error', 0)
            warning_count = summary['severity_distribution'].get('warning', 0)
            
            health_score = 1.0 - (
                (critical_count * 0.8 + error_count * 0.5 + warning_count * 0.2) / total_events
            )
            summary['timeline_health_score'] = max(0.0, health_score)
        
        return summary
    
    def _analyze_causality(self, events: List[TimelineEvent]) -> None:
        """Analyze causality relationships between events."""
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Look for potential causality relationships
        for i in range(len(sorted_events)):
            current_event = sorted_events[i]
            
            # Look at events within a reasonable time window (e.g., 30 seconds)
            causality_window = 30.0
            
            for j in range(i + 1, len(sorted_events)):
                next_event = sorted_events[j]
                
                # Stop if beyond causality window
                if next_event.timestamp - current_event.timestamp > causality_window:
                    break
                
                # Check for potential causality
                if self._could_be_causal_relationship(current_event, next_event):
                    # Add relationship
                    current_event.child_event_ids.add(next_event.event_id)
                    next_event.parent_event_id = current_event.event_id
                    
                    # Bidirectional relationship
                    current_event.related_event_ids.add(next_event.event_id)
                    next_event.related_event_ids.add(current_event.event_id)
    
    def _could_be_causal_relationship(self, event1: TimelineEvent, event2: TimelineEvent) -> bool:
        """Check if two events could have a causal relationship."""
        
        # Same component events are more likely to be related
        if event1.source == event2.source:
            return True
        
        # Error events following method calls might be related
        if (event1.event_type == EventType.METHOD_CALL and 
            event2.event_type == EventType.EXCEPTION):
            return True
        
        # State changes following method calls
        if (event1.event_type == EventType.METHOD_CALL and 
            event2.event_type == EventType.STATE_CHANGE):
            return True
        
        # Performance events following resource-intensive operations
        if (event1.resource_usage.get('cpu_percent', 0) > 50 and
            event2.event_type == EventType.PERFORMANCE_METRIC):
            return True
        
        return False


class TimelineReconstructor:
    """
    Main timeline reconstruction system.
    
    Reconstructs comprehensive event timelines from multiple debugging
    components and provides advanced analysis capabilities.
    """
    
    def __init__(self):
        """Initialize timeline reconstructor."""
        self.logger = get_enhanced_logger("arena_bot.debugging.timeline_reconstructor")
        
        # Core components
        self.event_collector = EventCollector(self)
        self.timeline_analyzer = TimelineAnalyzer()
        
        # Timeline storage
        self.events: List[TimelineEvent] = []
        self.segments: Dict[str, TimelineSegment] = {}
        
        # Configuration
        self.max_events_in_memory = 10000
        self.auto_segmentation_enabled = True
        self.segment_duration_seconds = 3600  # 1 hour segments
        
        # Reconstruction state
        self.reconstruction_enabled = False
        self.start_time = time.time()
        
        # Performance tracking
        self.reconstruction_stats = {
            'events_processed': 0,
            'segments_created': 0,
            'analysis_count': 0,
            'reconstruction_errors': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def start_reconstruction(self) -> bool:
        """Start timeline reconstruction."""
        
        with self.lock:
            try:
                if self.reconstruction_enabled:
                    self.logger.warning("Timeline reconstruction is already enabled")
                    return True
                
                # Start event collection
                success = self.event_collector.start_collection()
                if not success:
                    return False
                
                self.reconstruction_enabled = True
                self.start_time = time.time()
                
                self.logger.critical(
                    "🕒 TIMELINE_RECONSTRUCTION_STARTED: Advanced event timeline analysis enabled",
                    extra={
                        'timeline_reconstruction_startup': {
                            'max_events_in_memory': self.max_events_in_memory,
                            'auto_segmentation_enabled': self.auto_segmentation_enabled,
                            'segment_duration_seconds': self.segment_duration_seconds,
                            'timestamp': time.time()
                        }
                    }
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start timeline reconstruction: {e}")
                return False
    
    def stop_reconstruction(self) -> None:
        """Stop timeline reconstruction."""
        
        with self.lock:
            if not self.reconstruction_enabled:
                return
            
            # Stop event collection
            self.event_collector.stop_collection()
            
            self.reconstruction_enabled = False
            
            uptime_seconds = time.time() - self.start_time
            
            self.logger.info(
                f"🔄 TIMELINE_RECONSTRUCTION_STOPPED: Reconstruction stopped after {uptime_seconds:.1f}s",
                extra={
                    'timeline_reconstruction_shutdown': {
                        'uptime_seconds': uptime_seconds,
                        'events_processed': self.reconstruction_stats['events_processed'],
                        'segments_created': self.reconstruction_stats['segments_created'],
                        'analysis_count': self.reconstruction_stats['analysis_count'],
                        'timestamp': time.time()
                    }
                }
            )
    
    def add_event(self, event: TimelineEvent) -> None:
        """Add an event to the timeline."""
        
        with self.lock:
            if not self.reconstruction_enabled:
                return
            
            try:
                # Insert event in chronological order
                bisect.insort(self.events, event, key=lambda e: e.timestamp)
                
                # Maintain memory limits
                if len(self.events) > self.max_events_in_memory:
                    # Remove oldest events, but preserve them in segments
                    oldest_events = self.events[:len(self.events) - self.max_events_in_memory]
                    self._archive_events_to_segments(oldest_events)
                    self.events = self.events[len(self.events) - self.max_events_in_memory:]
                
                # Auto-segmentation
                if self.auto_segmentation_enabled:
                    self._check_auto_segmentation()
                
                self.reconstruction_stats['events_processed'] += 1
                
            except Exception as e:
                self.reconstruction_stats['reconstruction_errors'] += 1
                self.logger.error(f"Failed to add event to timeline: {e}")
    
    def _archive_events_to_segments(self, events: List[TimelineEvent]) -> None:
        """Archive events to timeline segments."""
        
        if not events:
            return
        
        # Group events by time periods
        start_time = events[0].timestamp
        current_segment_start = start_time
        current_segment_events = []
        
        for event in events:
            # Check if event should be in a new segment
            if (event.timestamp - current_segment_start > self.segment_duration_seconds and
                current_segment_events):
                
                # Create segment for current events
                self._create_segment(current_segment_events, current_segment_start)
                
                # Start new segment
                current_segment_start = event.timestamp
                current_segment_events = []
            
            current_segment_events.append(event)
        
        # Create final segment
        if current_segment_events:
            self._create_segment(current_segment_events, current_segment_start)
    
    def _create_segment(self, events: List[TimelineEvent], start_time: float) -> TimelineSegment:
        """Create a timeline segment from events."""
        
        if not events:
            return None
        
        end_time = max(event.timestamp for event in events)
        
        segment = TimelineSegment(
            start_time=start_time,
            end_time=end_time,
            title=f"Timeline Segment {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            description=f"Events from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}",
            events=events.copy()
        )
        
        # Analyze the segment
        analyzed_segment = self.timeline_analyzer.analyze_timeline_segment(segment)
        
        # Store segment
        self.segments[segment.segment_id] = analyzed_segment
        self.reconstruction_stats['segments_created'] += 1
        self.reconstruction_stats['analysis_count'] += 1
        
        self.logger.debug(
            f"Created timeline segment: {segment.segment_id} with {len(events)} events"
        )
        
        return analyzed_segment
    
    def _check_auto_segmentation(self) -> None:
        """Check if auto-segmentation should create a new segment."""
        
        if not self.events:
            return
        
        # Check if we should create a segment based on time
        oldest_event_time = self.events[0].timestamp
        current_time = time.time()
        
        if current_time - oldest_event_time > self.segment_duration_seconds:
            # Find cutoff point
            cutoff_time = current_time - self.segment_duration_seconds
            cutoff_index = 0
            
            for i, event in enumerate(self.events):
                if event.timestamp > cutoff_time:
                    cutoff_index = i
                    break
            
            if cutoff_index > 0:
                # Create segment from old events
                old_events = self.events[:cutoff_index]
                self._archive_events_to_segments(old_events)
                
                # Keep recent events
                self.events = self.events[cutoff_index:]
    
    def reconstruct_timeline(self, start_time: float, end_time: float) -> TimelineSegment:
        """Reconstruct timeline for a specific time period."""
        
        with self.lock:
            try:
                # Collect events from the time period
                timeline_events = []
                
                # Get events from memory
                for event in self.events:
                    if start_time <= event.timestamp <= end_time:
                        timeline_events.append(event)
                
                # Get events from segments
                for segment in self.segments.values():
                    if (segment.start_time <= end_time and segment.end_time >= start_time):
                        for event in segment.events:
                            if start_time <= event.timestamp <= end_time:
                                timeline_events.append(event)
                
                # Remove duplicates and sort
                seen_ids = set()
                unique_events = []
                for event in timeline_events:
                    if event.event_id not in seen_ids:
                        unique_events.append(event)
                        seen_ids.add(event.event_id)
                
                unique_events.sort(key=lambda e: e.timestamp)
                
                # Create timeline segment
                segment = TimelineSegment(
                    start_time=start_time,
                    end_time=end_time,
                    title=f"Reconstructed Timeline {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}",
                    description=f"Reconstructed timeline from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}",
                    events=unique_events
                )
                
                # Analyze the reconstructed timeline
                analyzed_segment = self.timeline_analyzer.analyze_timeline_segment(segment)
                
                self.reconstruction_stats['analysis_count'] += 1
                
                self.logger.info(
                    f"🕒 TIMELINE_RECONSTRUCTED: {len(unique_events)} events from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}"
                )
                
                return analyzed_segment
                
            except Exception as e:
                self.reconstruction_stats['reconstruction_errors'] += 1
                self.logger.error(f"Timeline reconstruction failed: {e}")
                
                # Return empty segment on failure
                return TimelineSegment(
                    start_time=start_time,
                    end_time=end_time,
                    title="Reconstruction Failed",
                    description=f"Timeline reconstruction failed: {e}"
                )
    
    def find_root_cause_timeline(self, target_event_id: str, 
                                lookback_seconds: float = 300) -> Optional[TimelineSegment]:
        """Find root cause timeline leading to a specific event."""
        
        # Find the target event
        target_event = None
        for event in self.events:
            if event.event_id == target_event_id:
                target_event = event
                break
        
        if not target_event:
            # Check segments
            for segment in self.segments.values():
                for event in segment.events:
                    if event.event_id == target_event_id:
                        target_event = event
                        break
                if target_event:
                    break
        
        if not target_event:
            self.logger.warning(f"Target event not found: {target_event_id}")
            return None
        
        # Reconstruct timeline leading to the event
        start_time = target_event.timestamp - lookback_seconds
        end_time = target_event.timestamp
        
        root_cause_timeline = self.reconstruct_timeline(start_time, end_time)
        root_cause_timeline.title = f"Root Cause Analysis for {target_event.title}"
        root_cause_timeline.description = f"Events leading to {target_event.title} over {lookback_seconds}s period"
        
        return root_cause_timeline
    
    def get_timeline_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get timeline summary for the last N hours."""
        
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Get timeline for the period
        timeline = self.reconstruct_timeline(start_time, end_time)
        
        return {
            'time_period': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': hours
            },
            'event_summary': {
                'total_events': len(timeline.events),
                'event_types': timeline.performance_summary.get('event_types', {}),
                'severity_distribution': timeline.performance_summary.get('severity_distribution', {})
            },
            'patterns_detected': len(timeline.patterns_detected),
            'anomalies_detected': len(timeline.anomalies),
            'key_events_count': len(timeline.key_events),
            'timeline_health_score': timeline.performance_summary.get('timeline_health_score', 0.0),
            'resource_usage': timeline.performance_summary.get('resource_usage', {})
        }
    
    def export_timeline(self, start_time: float, end_time: float, 
                       export_format: str = 'json') -> str:
        """Export timeline data."""
        
        timeline = self.reconstruct_timeline(start_time, end_time)
        
        if export_format.lower() == 'json':
            return json.dumps(timeline.to_dict(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive timeline reconstructor status."""
        
        with self.lock:
            return {
                'reconstruction_enabled': self.reconstruction_enabled,
                'uptime_seconds': time.time() - self.start_time if self.reconstruction_enabled else 0,
                'events_in_memory': len(self.events),
                'segments_stored': len(self.segments),
                'reconstruction_stats': self.reconstruction_stats.copy(),
                'collection_stats': self.event_collector.get_collection_stats(),
                'memory_limit': self.max_events_in_memory,
                'auto_segmentation_enabled': self.auto_segmentation_enabled,
                'segment_duration_seconds': self.segment_duration_seconds
            }
    
    def shutdown(self) -> None:
        """Gracefully shutdown timeline reconstructor."""
        self.logger.info("🔄 Shutting down timeline reconstructor...")
        self.stop_reconstruction()
        self.logger.info("✅ Timeline reconstructor shutdown complete")


# Global timeline reconstructor instance
_global_timeline_reconstructor: Optional[TimelineReconstructor] = None
_reconstructor_lock = threading.Lock()


def get_timeline_reconstructor() -> TimelineReconstructor:
    """Get global timeline reconstructor instance."""
    global _global_timeline_reconstructor
    
    if _global_timeline_reconstructor is None:
        with _reconstructor_lock:
            if _global_timeline_reconstructor is None:
                _global_timeline_reconstructor = TimelineReconstructor()
    
    return _global_timeline_reconstructor


def start_timeline_reconstruction() -> bool:
    """
    Start timeline reconstruction.
    
    Returns:
        True if reconstruction started successfully
    """
    reconstructor = get_timeline_reconstructor()
    return reconstructor.start_reconstruction()


def stop_timeline_reconstruction() -> None:
    """Stop timeline reconstruction."""
    reconstructor = get_timeline_reconstructor()
    reconstructor.stop_reconstruction()


def reconstruct_timeline(start_time: float, end_time: float) -> TimelineSegment:
    """Reconstruct timeline for a specific time period."""
    reconstructor = get_timeline_reconstructor()
    return reconstructor.reconstruct_timeline(start_time, end_time)


def find_root_cause_timeline(event_id: str, lookback_seconds: float = 300) -> Optional[TimelineSegment]:
    """Find root cause timeline for a specific event."""
    reconstructor = get_timeline_reconstructor()
    return reconstructor.find_root_cause_timeline(event_id, lookback_seconds)


def get_timeline_status() -> Dict[str, Any]:
    """Get timeline reconstructor status."""
    reconstructor = get_timeline_reconstructor()
    return reconstructor.get_status()


# Convenience function for adding custom events
def add_timeline_event(event_type: EventType, title: str, description: str = "",
                      source: str = "custom", severity: EventSeverity = EventSeverity.INFO,
                      context: Dict[str, Any] = None) -> str:
    """
    Add a custom event to the timeline.
    
    Args:
        event_type: Type of the event
        title: Event title
        description: Event description
        source: Event source
        severity: Event severity
        context: Additional context data
        
    Returns:
        Event ID
    """
    reconstructor = get_timeline_reconstructor()
    
    event = TimelineEvent(
        event_type=event_type,
        severity=severity,
        source=source,
        title=title,
        description=description,
        context=context or {}
    )
    
    reconstructor.add_event(event)
    return event.event_id


# Convenience decorator for timeline event tracking
def track_timeline_events(event_type: EventType = EventType.METHOD_CALL,
                         source: str = "", severity: EventSeverity = EventSeverity.DEBUG) -> Callable:
    """
    Decorator to automatically track method calls as timeline events.
    
    Usage:
        @track_timeline_events(EventType.METHOD_CALL, "my_component")
        def my_method():
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **kwargs):
            reconstructor = get_timeline_reconstructor()
            
            if reconstructor.reconstruction_enabled:
                start_time = time.perf_counter()
                
                # Create start event
                event = TimelineEvent(
                    event_type=event_type,
                    severity=severity,
                    source=source or func.__module__,
                    title=f"Method Call: {func.__name__}",
                    description=f"Started {func.__name__} with {len(args)} args, {len(kwargs)} kwargs",
                    context={
                        'function_name': func.__name__,
                        'module_name': func.__module__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Update event with completion info
                    execution_time = (time.perf_counter() - start_time) * 1000
                    event.duration_ms = execution_time
                    event.context['execution_result'] = 'success'
                    
                    if execution_time > 1000:  # >1 second
                        event.severity = EventSeverity.WARNING
                        event.description += f" (SLOW: {execution_time:.1f}ms)"
                    
                    reconstructor.add_event(event)
                    return result
                    
                except Exception as e:
                    # Update event with error info
                    execution_time = (time.perf_counter() - start_time) * 1000
                    event.duration_ms = execution_time
                    event.severity = EventSeverity.ERROR
                    event.context.update({
                        'execution_result': 'error',
                        'exception_type': type(e).__name__,
                        'exception_message': str(e)
                    })
                    event.description += f" (ERROR: {str(e)})"
                    
                    reconstructor.add_event(event)
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator