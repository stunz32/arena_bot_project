"""
Data Flow Tracing System for Arena Bot Deep Debugging

Comprehensive tracking of data flow through the entire processing pipeline:
- Screenshot capture → Preprocessing → Detection → AI Analysis → GUI Display
- Data transformation monitoring with before/after snapshots
- Performance bottleneck identification at each pipeline stage
- Error propagation tracking across component boundaries
- Data quality validation and anomaly detection
- Cross-component correlation with timing analysis

Integrates with existing correlation ID system from S-tier logging.
"""

import time
import threading
import hashlib
import json
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque

from ..logging_system.logger import get_logger, LogLevel


class PipelineStage(Enum):
    """Stages in the Arena Bot processing pipeline."""
    SCREENSHOT_CAPTURE = "screenshot_capture"
    PREPROCESSING = "preprocessing"
    COORDINATE_DETECTION = "coordinate_detection"
    CARD_EXTRACTION = "card_extraction"
    HISTOGRAM_MATCHING = "histogram_matching"
    TEMPLATE_MATCHING = "template_matching"
    VALIDATION = "validation"
    AI_ANALYSIS = "ai_analysis"
    TIER_LOOKUP = "tier_lookup"
    RECOMMENDATION = "recommendation"
    GUI_UPDATE = "gui_update"
    USER_DISPLAY = "user_display"


class DataFlowType(Enum):
    """Types of data flow events."""
    INPUT = "input"           # Data entering a stage
    OUTPUT = "output"         # Data leaving a stage
    TRANSFORMATION = "transformation"  # Data being transformed
    ERROR = "error"          # Error in data processing
    VALIDATION = "validation"  # Data validation event


@dataclass
class DataSnapshot:
    """
    Snapshot of data at a specific point in the pipeline.
    
    Captures essential characteristics without storing full data
    to minimize memory usage while preserving debugging information.
    """
    
    # Data identification
    data_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Data characteristics
    data_type: str = ""
    data_size_bytes: int = 0
    data_hash: Optional[str] = None
    data_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    is_valid: bool = True
    quality_score: float = 1.0
    validation_errors: List[str] = field(default_factory=list)
    
    def capture_data(self, data: Any, max_summary_size: int = 1000) -> None:
        """Capture data characteristics for debugging."""
        try:
            # Basic type information
            self.data_type = type(data).__name__
            
            # Size estimation
            if hasattr(data, '__len__'):
                try:
                    self.data_size_bytes = len(data)
                except:
                    pass
            
            # Hash for change detection
            try:
                if isinstance(data, (str, bytes)):
                    self.data_hash = hashlib.md5(
                        data.encode() if isinstance(data, str) else data
                    ).hexdigest()[:16]
                elif hasattr(data, 'tobytes'):  # numpy arrays, images
                    self.data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
            except:
                pass
            
            # Summary information
            self.data_summary = self._create_summary(data, max_summary_size)
            
        except Exception as e:
            self.data_summary = {'capture_error': str(e)}
    
    def _create_summary(self, data: Any, max_size: int) -> Dict[str, Any]:
        """Create a summary of the data for debugging."""
        try:
            summary = {}
            
            # Handle different data types
            if data is None:
                summary['value'] = None
            elif isinstance(data, (str, int, float, bool)):
                summary['value'] = data if len(str(data)) <= max_size else f"{str(data)[:max_size]}..."
            elif isinstance(data, (list, tuple)):
                summary['length'] = len(data)
                summary['first_few'] = [
                    self._safe_repr(item) for item in data[:5]
                ]
            elif isinstance(data, dict):
                summary['keys'] = list(data.keys())[:10]
                summary['size'] = len(data)
            elif hasattr(data, 'shape'):  # numpy arrays, images
                summary['shape'] = data.shape
                summary['dtype'] = str(data.dtype) if hasattr(data, 'dtype') else None
                if hasattr(data, 'min') and hasattr(data, 'max'):
                    try:
                        summary['min_value'] = float(data.min())
                        summary['max_value'] = float(data.max())
                    except:
                        pass
            else:
                summary['type'] = self.data_type
                if hasattr(data, '__dict__'):
                    summary['attributes'] = list(vars(data).keys())[:10]
            
            return summary
            
        except Exception as e:
            return {'summary_error': str(e)}
    
    def _safe_repr(self, obj: Any) -> str:
        """Safely represent an object as a string."""
        try:
            repr_str = repr(obj)
            return repr_str if len(repr_str) <= 100 else f"{repr_str[:100]}..."
        except:
            return f"<{type(obj).__name__}>"


@dataclass
class DataFlowEvent:
    """
    Represents a single data flow event in the processing pipeline.
    
    Captures comprehensive information about data movement and transformation
    for debugging and performance analysis.
    """
    
    # Event identification
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Pipeline context
    stage: PipelineStage
    flow_type: DataFlowType
    operation_name: str = ""
    
    # Correlation and tracing
    correlation_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Data information
    input_data: Optional[DataSnapshot] = None
    output_data: Optional[DataSnapshot] = None
    
    # Performance metrics
    processing_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_occurred: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Thread information
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    thread_name: str = field(default_factory=lambda: threading.current_thread().name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'stage': self.stage.value,
            'flow_type': self.flow_type.value,
            'operation_name': self.operation_name,
            'correlation_id': self.correlation_id,
            'pipeline_id': self.pipeline_id,
            'parent_event_id': self.parent_event_id,
            'input_data': self.input_data.__dict__ if self.input_data else None,
            'output_data': self.output_data.__dict__ if self.output_data else None,
            'processing_time_ms': self.processing_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'context': self.context,
            'error_occurred': self.error_occurred,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'thread_id': self.thread_id,
            'thread_name': self.thread_name
        }


class PipelineTrace:
    """
    Represents a complete trace through the processing pipeline.
    
    Links all events in a single processing flow from start to finish.
    """
    
    def __init__(self, pipeline_id: str, correlation_id: Optional[str] = None):
        """Initialize pipeline trace."""
        self.pipeline_id = pipeline_id
        self.correlation_id = correlation_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.events: List[DataFlowEvent] = []
        self.stages_completed: set = set()
        self.errors: List[DataFlowEvent] = []
        self.performance_metrics: Dict[str, float] = {}
        
    def add_event(self, event: DataFlowEvent) -> None:
        """Add an event to this pipeline trace."""
        event.pipeline_id = self.pipeline_id
        event.correlation_id = self.correlation_id
        
        self.events.append(event)
        self.stages_completed.add(event.stage)
        
        if event.error_occurred:
            self.errors.append(event)
    
    def finish(self) -> None:
        """Mark pipeline trace as finished."""
        self.end_time = time.time()
        
        # Calculate performance metrics
        total_time = (self.end_time - self.start_time) * 1000  # ms
        self.performance_metrics['total_time_ms'] = total_time
        
        # Calculate stage times
        stage_times = {}
        for event in self.events:
            if event.processing_time_ms:
                stage = event.stage.value
                if stage not in stage_times:
                    stage_times[stage] = 0
                stage_times[stage] += event.processing_time_ms
        
        self.performance_metrics['stage_times_ms'] = stage_times
        
        # Calculate success rate
        total_operations = len(self.events)
        failed_operations = len(self.errors)
        self.performance_metrics['success_rate'] = (
            (total_operations - failed_operations) / total_operations
            if total_operations > 0 else 1.0
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline trace."""
        return {
            'pipeline_id': self.pipeline_id,
            'correlation_id': self.correlation_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_events': len(self.events),
            'stages_completed': len(self.stages_completed),
            'errors': len(self.errors),
            'performance_metrics': self.performance_metrics,
            'is_complete': self.end_time is not None
        }


class PipelineTracer:
    """
    Central data flow tracer for Arena Bot processing pipeline.
    
    Tracks data flow through all stages of processing, monitors
    performance, and provides analysis capabilities.
    """
    
    def __init__(self):
        """Initialize pipeline tracer."""
        self.logger = get_logger("arena_bot.debugging.pipeline_tracer")
        
        # Active traces
        self.active_traces: Dict[str, PipelineTrace] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self.traces_lock = threading.RLock()
        
        # Event history
        self.event_history: deque = deque(maxlen=10000)
        self.events_by_stage: Dict[PipelineStage, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.enabled = True
        self.capture_data_snapshots = True
        self.log_all_events = True
        self.detect_bottlenecks = True
        
        # Performance tracking
        self.total_events_traced = 0
        self.total_pipelines_traced = 0
        self.errors_detected = 0
        self.bottlenecks_detected = 0
        
        # Performance thresholds
        self.stage_time_thresholds = {
            PipelineStage.SCREENSHOT_CAPTURE: 100,     # 100ms
            PipelineStage.PREPROCESSING: 50,           # 50ms
            PipelineStage.CARD_EXTRACTION: 200,       # 200ms
            PipelineStage.HISTOGRAM_MATCHING: 500,    # 500ms
            PipelineStage.AI_ANALYSIS: 1000,          # 1 second
            PipelineStage.GUI_UPDATE: 50              # 50ms
        }
    
    def start_pipeline(self, 
                      operation_name: str = "",
                      correlation_id: Optional[str] = None) -> str:
        """
        Start tracing a new pipeline execution.
        
        Returns:
            Pipeline ID for tracking this execution
        """
        
        if not self.enabled:
            return ""
        
        pipeline_id = str(uuid4())
        
        with self.traces_lock:
            trace = PipelineTrace(pipeline_id, correlation_id)
            self.active_traces[pipeline_id] = trace
        
        self.total_pipelines_traced += 1
        
        self.logger.info(
            f"🚀 PIPELINE_START: {operation_name} (ID: {pipeline_id[:8]})",
            extra={
                'trace_type': 'pipeline_start',
                'pipeline_id': pipeline_id,
                'operation_name': operation_name,
                'correlation_id': correlation_id
            }
        )
        
        return pipeline_id
    
    def finish_pipeline(self, pipeline_id: str) -> Optional[PipelineTrace]:
        """
        Finish tracing a pipeline execution.
        
        Returns:
            Completed PipelineTrace or None if not found
        """
        
        if not self.enabled or not pipeline_id:
            return None
        
        with self.traces_lock:
            trace = self.active_traces.pop(pipeline_id, None)
            
            if trace:
                trace.finish()
                self.completed_traces.append(trace)
                
                # Log completion
                summary = trace.get_summary()
                self.logger.info(
                    f"🏁 PIPELINE_COMPLETE: {pipeline_id[:8]} "
                    f"({summary['total_events']} events, "
                    f"{summary['performance_metrics'].get('total_time_ms', 0):.1f}ms)",
                    extra={
                        'trace_type': 'pipeline_complete',
                        'pipeline_summary': summary
                    }
                )
                
                # Check for performance issues
                self._analyze_pipeline_performance(trace)
        
        return trace
    
    def trace_data_flow(self,
                       stage: PipelineStage,
                       flow_type: DataFlowType,
                       operation_name: str = "",
                       pipeline_id: Optional[str] = None,
                       correlation_id: Optional[str] = None,
                       input_data: Any = None,
                       output_data: Any = None,
                       context: Optional[Dict[str, Any]] = None,
                       processing_time_ms: Optional[float] = None,
                       error: Optional[Exception] = None) -> DataFlowEvent:
        """
        Trace a data flow event in the pipeline.
        
        Args:
            stage: Pipeline stage where event occurred
            flow_type: Type of data flow event
            operation_name: Name of the operation
            pipeline_id: ID of the pipeline this event belongs to
            correlation_id: Correlation ID for tracing
            input_data: Input data (snapshot will be captured)
            output_data: Output data (snapshot will be captured)
            context: Additional context information
            processing_time_ms: Time taken for processing
            error: Exception if an error occurred
            
        Returns:
            DataFlowEvent created for this trace
        """
        
        if not self.enabled:
            return None
        
        # Create event
        event = DataFlowEvent(
            stage=stage,
            flow_type=flow_type,
            operation_name=operation_name,
            correlation_id=correlation_id,
            context=context or {},
            processing_time_ms=processing_time_ms
        )
        
        # Handle error information
        if error:
            event.error_occurred = True
            event.error_type = type(error).__name__
            event.error_message = str(error)
            self.errors_detected += 1
        
        # Capture data snapshots
        if self.capture_data_snapshots:
            if input_data is not None:
                event.input_data = DataSnapshot()
                event.input_data.capture_data(input_data)
            
            if output_data is not None:
                event.output_data = DataSnapshot()
                event.output_data.capture_data(output_data)
        
        # Add to pipeline trace
        if pipeline_id:
            with self.traces_lock:
                trace = self.active_traces.get(pipeline_id)
                if trace:
                    trace.add_event(event)
        
        # Add to event history
        self.event_history.append(event)
        self.events_by_stage[stage].append(event)
        self.total_events_traced += 1
        
        # Log event
        if self.log_all_events:
            log_level = LogLevel.ERROR if error else LogLevel.DEBUG
            
            self.logger.log(
                log_level,
                f"📊 DATA_FLOW: {stage.value} {flow_type.value} "
                f"{operation_name} {'FAILED' if error else 'SUCCESS'} "
                f"({processing_time_ms:.1f}ms)" if processing_time_ms else "",
                extra={
                    'trace_type': 'data_flow',
                    'data_flow_event': event.to_dict()
                }
            )
        
        # Check for bottlenecks
        if self.detect_bottlenecks and processing_time_ms:
            threshold = self.stage_time_thresholds.get(stage, 1000)
            if processing_time_ms > threshold:
                self.bottlenecks_detected += 1
                self.logger.warning(
                    f"⚠️ BOTTLENECK: {stage.value} {operation_name} took {processing_time_ms:.1f}ms "
                    f"(threshold: {threshold}ms)",
                    extra={
                        'trace_type': 'bottleneck',
                        'stage': stage.value,
                        'operation_name': operation_name,
                        'processing_time_ms': processing_time_ms,
                        'threshold_ms': threshold,
                        'pipeline_id': pipeline_id
                    }
                )
        
        return event
    
    def _analyze_pipeline_performance(self, trace: PipelineTrace) -> None:
        """Analyze pipeline performance and log issues."""
        metrics = trace.performance_metrics
        
        # Check total time
        total_time = metrics.get('total_time_ms', 0)
        if total_time > 5000:  # > 5 seconds
            self.logger.warning(
                f"⚠️ SLOW_PIPELINE: {trace.pipeline_id[:8]} took {total_time:.1f}ms",
                extra={
                    'trace_type': 'slow_pipeline',
                    'pipeline_id': trace.pipeline_id,
                    'total_time_ms': total_time,
                    'stage_times': metrics.get('stage_times_ms', {})
                }
            )
        
        # Check success rate
        success_rate = metrics.get('success_rate', 1.0)
        if success_rate < 0.8:  # < 80% success
            self.logger.warning(
                f"⚠️ LOW_SUCCESS_RATE: {trace.pipeline_id[:8]} success rate {success_rate:.1%}",
                extra={
                    'trace_type': 'low_success_rate',
                    'pipeline_id': trace.pipeline_id,
                    'success_rate': success_rate,
                    'total_errors': len(trace.errors)
                }
            )
    
    def get_active_pipelines(self) -> List[str]:
        """Get list of currently active pipeline IDs."""
        with self.traces_lock:
            return list(self.active_traces.keys())
    
    def get_pipeline_trace(self, pipeline_id: str) -> Optional[PipelineTrace]:
        """Get a specific pipeline trace."""
        with self.traces_lock:
            # Check active traces first
            if pipeline_id in self.active_traces:
                return self.active_traces[pipeline_id]
            
            # Check completed traces
            for trace in self.completed_traces:
                if trace.pipeline_id == pipeline_id:
                    return trace
        
        return None
    
    def get_recent_events(self, 
                         stage: Optional[PipelineStage] = None,
                         seconds: int = 60,
                         limit: int = 100) -> List[DataFlowEvent]:
        """Get recent data flow events."""
        cutoff_time = time.time() - seconds
        
        if stage:
            events = list(self.events_by_stage[stage])
        else:
            events = list(self.event_history)
        
        # Filter by time
        recent_events = [
            event for event in events
            if event.timestamp >= cutoff_time
        ]
        
        return recent_events[-limit:] if limit > 0 else recent_events
    
    def get_stage_performance(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get performance statistics for a specific stage."""
        events = list(self.events_by_stage[stage])
        
        if not events:
            return {'stage': stage.value, 'no_data': True}
        
        # Calculate statistics
        processing_times = [
            e.processing_time_ms for e in events
            if e.processing_time_ms is not None
        ]
        
        errors = [e for e in events if e.error_occurred]
        
        stats = {
            'stage': stage.value,
            'total_events': len(events),
            'total_errors': len(errors),
            'error_rate': len(errors) / len(events) if events else 0
        }
        
        if processing_times:
            stats.update({
                'avg_processing_time_ms': sum(processing_times) / len(processing_times),
                'min_processing_time_ms': min(processing_times),
                'max_processing_time_ms': max(processing_times),
                'total_processing_time_ms': sum(processing_times)
            })
        
        return stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.traces_lock:
            active_pipelines = len(self.active_traces)
            completed_pipelines = len(self.completed_traces)
        
        return {
            'total_events_traced': self.total_events_traced,
            'total_pipelines_traced': self.total_pipelines_traced,
            'active_pipelines': active_pipelines,
            'completed_pipelines': completed_pipelines,
            'errors_detected': self.errors_detected,
            'bottlenecks_detected': self.bottlenecks_detected,
            'enabled': self.enabled,
            'capture_data_snapshots': self.capture_data_snapshots
        }
    
    def enable(self) -> None:
        """Enable pipeline tracing."""
        self.enabled = True
        self.logger.info("📊 Pipeline tracing enabled")
    
    def disable(self) -> None:
        """Disable pipeline tracing."""
        self.enabled = False
        self.logger.info("📊 Pipeline tracing disabled")


# Global pipeline tracer instance
_global_pipeline_tracer: Optional[PipelineTracer] = None
_tracer_lock = threading.Lock()


def get_pipeline_tracer() -> PipelineTracer:
    """Get global pipeline tracer instance."""
    global _global_pipeline_tracer
    
    if _global_pipeline_tracer is None:
        with _tracer_lock:
            if _global_pipeline_tracer is None:
                _global_pipeline_tracer = PipelineTracer()
    
    return _global_pipeline_tracer


def trace_pipeline_stage(stage: PipelineStage,
                        operation_name: str = "",
                        pipeline_id: Optional[str] = None,
                        **kwargs) -> Callable:
    """
    Decorator for tracing pipeline stage operations.
    
    Usage:
        @trace_pipeline_stage(PipelineStage.CARD_EXTRACTION, "extract_cards")
        def extract_cards(self, screenshot):
            # Method implementation
            return cards
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **func_kwargs):
            tracer = get_pipeline_tracer()
            
            if not tracer.enabled:
                return func(*args, **func_kwargs)
            
            start_time = time.perf_counter()
            error = None
            result = None
            
            try:
                # Trace input
                tracer.trace_data_flow(
                    stage=stage,
                    flow_type=DataFlowType.INPUT,
                    operation_name=operation_name,
                    pipeline_id=pipeline_id,
                    input_data=args[1] if len(args) > 1 else None,
                    **kwargs
                )
                
                # Execute function
                result = func(*args, **func_kwargs)
                
                # Calculate processing time
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                
                # Trace output
                tracer.trace_data_flow(
                    stage=stage,
                    flow_type=DataFlowType.OUTPUT,
                    operation_name=operation_name,
                    pipeline_id=pipeline_id,
                    output_data=result,
                    processing_time_ms=processing_time_ms,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                error = e
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                
                # Trace error
                tracer.trace_data_flow(
                    stage=stage,
                    flow_type=DataFlowType.ERROR,
                    operation_name=operation_name,
                    pipeline_id=pipeline_id,
                    processing_time_ms=processing_time_ms,
                    error=error,
                    **kwargs
                )
                
                raise
        
        return wrapper
    
    return decorator