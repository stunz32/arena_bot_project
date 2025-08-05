"""
State Change Monitoring System for Arena Bot Deep Debugging

Comprehensive monitoring of component state transitions with:
- Real-time state change detection and logging
- State transition validation and anomaly detection
- Cross-component state correlation and dependency tracking
- State history with rollback capability
- Performance impact analysis of state changes
- Automated state consistency validation

Integrates with existing state management in IntegratedArenaBotGUI and other components.
"""

import time
import threading
import json
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque

from ..logging_system.logger import get_logger, LogLevel


class StateChangeType(Enum):
    """Types of state changes that can occur."""
    TRANSITION = "transition"      # Normal state transition
    INITIALIZATION = "initialization"  # Component initialization
    SHUTDOWN = "shutdown"         # Component shutdown
    ERROR = "error"              # Error state change
    RECOVERY = "recovery"        # Recovery from error
    CONFIGURATION = "configuration"  # Configuration change


@dataclass
class StateChangeEvent:
    """
    Represents a single state change event with comprehensive context.
    
    Captures all relevant information about a state transition for
    debugging and analysis purposes.
    """
    
    # Event identification
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Component information
    component_name: str = ""
    component_type: str = ""
    
    # State information
    from_state: Any = None
    to_state: Any = None
    change_type: StateChangeType = StateChangeType.TRANSITION
    
    # Context and metadata
    trigger: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    # Performance metrics
    change_duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Validation
    is_valid_transition: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Thread information
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    thread_name: str = field(default_factory=lambda: threading.current_thread().name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'component_name': self.component_name,
            'component_type': self.component_type,
            'from_state': self._serialize_state(self.from_state),
            'to_state': self._serialize_state(self.to_state),
            'change_type': self.change_type.value,
            'trigger': self.trigger,
            'context': self.context,
            'correlation_id': self.correlation_id,
            'change_duration_ms': self.change_duration_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'is_valid_transition': self.is_valid_transition,
            'validation_errors': self.validation_errors,
            'thread_id': self.thread_id,
            'thread_name': self.thread_name
        }
    
    def _serialize_state(self, state: Any) -> Any:
        """Safely serialize state for logging."""
        try:
            if state is None:
                return None
            elif isinstance(state, (str, int, float, bool)):
                return state
            elif isinstance(state, Enum):
                return state.value
            elif hasattr(state, '__dict__'):
                return f"<{type(state).__name__}>"
            else:
                return str(state)
        except:
            return f"<{type(state).__name__}:serialization_error>"


class StateValidator:
    """
    Validates state transitions and detects anomalies.
    
    Provides rules-based validation for component state changes
    and identifies potentially problematic transitions.
    """
    
    def __init__(self):
        """Initialize state validator."""
        self.transition_rules: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.forbidden_transitions: Dict[str, Set[tuple]] = defaultdict(set)
        self.required_context: Dict[str, Set[str]] = defaultdict(set)
        
    def add_valid_transition(self, component: str, from_state: str, to_state: str) -> None:
        """Add a valid state transition rule."""
        self.transition_rules[component][from_state].add(to_state)
    
    def add_forbidden_transition(self, component: str, from_state: str, to_state: str) -> None:
        """Add a forbidden state transition."""
        self.forbidden_transitions[component].add((from_state, to_state))
    
    def add_required_context(self, component: str, context_key: str) -> None:
        """Add required context for state changes."""
        self.required_context[component].add(context_key)
    
    def validate_transition(self, event: StateChangeEvent) -> bool:
        """
        Validate a state transition event.
        
        Returns True if valid, False otherwise.
        Updates event.validation_errors with any issues found.
        """
        errors = []
        
        # Check forbidden transitions
        transition = (str(event.from_state), str(event.to_state))
        if transition in self.forbidden_transitions[event.component_name]:
            errors.append(f"Forbidden transition: {transition[0]} -> {transition[1]}")
        
        # Check valid transitions (if rules exist)
        if event.component_name in self.transition_rules:
            from_state_str = str(event.from_state)
            to_state_str = str(event.to_state)
            
            if from_state_str in self.transition_rules[event.component_name]:
                valid_targets = self.transition_rules[event.component_name][from_state_str]
                if valid_targets and to_state_str not in valid_targets:
                    errors.append(f"Invalid transition: {from_state_str} -> {to_state_str}")
        
        # Check required context
        required_keys = self.required_context[event.component_name]
        for key in required_keys:
            if key not in event.context:
                errors.append(f"Missing required context: {key}")
        
        # Update event
        event.validation_errors = errors
        event.is_valid_transition = len(errors) == 0
        
        return event.is_valid_transition


class StateHistory:
    """
    Maintains historical record of state changes.
    
    Provides efficient storage and querying of state change history
    with automatic cleanup and memory management.
    """
    
    def __init__(self, max_events: int = 10000):
        """Initialize state history."""
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.events_by_component: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_states: Dict[str, Any] = {}
        self.lock = threading.RLock()
    
    def add_event(self, event: StateChangeEvent) -> None:
        """Add a state change event to history."""
        with self.lock:
            self.events.append(event)
            self.events_by_component[event.component_name].append(event)
            
            # Update current state
            if event.to_state is not None:
                self.current_states[event.component_name] = event.to_state
    
    def get_current_state(self, component: str) -> Any:
        """Get current state of a component."""
        with self.lock:
            return self.current_states.get(component)
    
    def get_component_history(self, component: str, limit: int = 100) -> List[StateChangeEvent]:
        """Get state change history for a specific component."""
        with self.lock:
            events = list(self.events_by_component[component])
            return events[-limit:] if limit > 0 else events
    
    def get_recent_events(self, seconds: int = 60, limit: int = 1000) -> List[StateChangeEvent]:
        """Get recent state change events."""
        cutoff_time = time.time() - seconds
        
        with self.lock:
            recent_events = [
                event for event in self.events
                if event.timestamp >= cutoff_time
            ]
            return recent_events[-limit:] if limit > 0 else recent_events
    
    def find_events(self, 
                   component: Optional[str] = None,
                   change_type: Optional[StateChangeType] = None,
                   from_state: Optional[str] = None,
                   to_state: Optional[str] = None,
                   limit: int = 100) -> List[StateChangeEvent]:
        """Find events matching criteria."""
        with self.lock:
            events = self.events_by_component[component] if component else self.events
            
            matches = []
            for event in events:
                if change_type and event.change_type != change_type:
                    continue
                if from_state and str(event.from_state) != from_state:
                    continue
                if to_state and str(event.to_state) != to_state:
                    continue
                
                matches.append(event)
                
                if len(matches) >= limit:
                    break
            
            return matches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about state change history."""
        with self.lock:
            component_counts = {
                component: len(events) 
                for component, events in self.events_by_component.items()
            }
            
            change_type_counts = defaultdict(int)
            for event in self.events:
                change_type_counts[event.change_type.value] += 1
            
            return {
                'total_events': len(self.events),
                'active_components': len(self.current_states),
                'component_event_counts': dict(component_counts),
                'change_type_counts': dict(change_type_counts),
                'max_events': self.max_events
            }


class StateMonitor:
    """
    Central state monitoring system for Arena Bot.
    
    Monitors all component state changes, validates transitions,
    maintains history, and provides analysis capabilities.
    """
    
    def __init__(self):
        """Initialize state monitor."""
        self.logger = get_logger("arena_bot.debugging.state_monitor")
        self.validator = StateValidator()
        self.history = StateHistory()
        
        # Configuration
        self.enabled = True
        self.log_all_changes = True
        self.log_invalid_transitions = True
        self.detect_anomalies = True
        
        # Performance tracking
        self.total_changes_monitored = 0
        self.invalid_transitions_detected = 0
        self.anomalies_detected = 0
        
        # Callbacks
        self.change_callbacks: List[Callable[[StateChangeEvent], None]] = []
        self.anomaly_callbacks: List[Callable[[StateChangeEvent], None]] = []
        
        # Setup default validation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules for Arena Bot components."""
        
        # GUI state transitions
        gui_states = ["INIT", "IDLE", "ANALYZING", "WAITING", "ERROR", "SHUTDOWN"]
        for from_state in gui_states:
            for to_state in gui_states:
                if from_state != to_state:
                    self.validator.add_valid_transition("IntegratedArenaBotGUI", from_state, to_state)
        
        # Detection system states
        detection_states = ["INITIALIZING", "READY", "DETECTING", "PROCESSING", "ERROR", "DISABLED"]
        for from_state in detection_states:
            for to_state in detection_states:
                if from_state != to_state:
                    self.validator.add_valid_transition("UltimateDetectionEngine", from_state, to_state)
        
        # AI advisor states  
        ai_states = ["UNINITIALIZED", "INITIALIZING", "READY", "ANALYZING", "ERROR"]
        for from_state in ai_states:
            for to_state in ai_states:
                if from_state != to_state:
                    self.validator.add_valid_transition("GrandmasterAdvisor", from_state, to_state)
        
        # Add forbidden transitions
        self.validator.add_forbidden_transition("IntegratedArenaBotGUI", "SHUTDOWN", "ANALYZING")
        self.validator.add_forbidden_transition("UltimateDetectionEngine", "ERROR", "DETECTING")
    
    def log_state_change(self,
                        component_name: str,
                        from_state: Any,
                        to_state: Any,
                        trigger: str = "",
                        context: Optional[Dict[str, Any]] = None,
                        component_type: str = "",
                        change_type: StateChangeType = StateChangeType.TRANSITION,
                        correlation_id: Optional[str] = None) -> StateChangeEvent:
        """
        Log a state change event.
        
        Args:
            component_name: Name of the component changing state
            from_state: Previous state
            to_state: New state
            trigger: What triggered the state change
            context: Additional context information
            component_type: Type of component
            change_type: Type of state change
            correlation_id: Correlation ID for tracing
            
        Returns:
            StateChangeEvent created for this change
        """
        
        if not self.enabled:
            return None
        
        # Create state change event
        event = StateChangeEvent(
            component_name=component_name,
            component_type=component_type,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            context=context or {},
            change_type=change_type,
            correlation_id=correlation_id
        )
        
        # Validate transition
        is_valid = self.validator.validate_transition(event)
        
        # Add to history
        self.history.add_event(event)
        
        # Update statistics
        self.total_changes_monitored += 1
        if not is_valid:
            self.invalid_transitions_detected += 1
        
        # Log the state change
        if self.log_all_changes:
            log_level = LogLevel.WARNING if not is_valid else LogLevel.INFO
            
            self.logger.log(
                log_level,
                f"🔄 STATE_CHANGE: {component_name} {from_state} → {to_state} "
                f"({'INVALID' if not is_valid else 'VALID'}) trigger: {trigger}",
                extra={
                    'state_change_event': event.to_dict(),
                    'component_name': component_name,
                    'from_state': str(from_state),
                    'to_state': str(to_state),
                    'trigger': trigger,
                    'is_valid_transition': is_valid,
                    'validation_errors': event.validation_errors,
                    'correlation_id': correlation_id
                }
            )
        
        # Log validation errors
        if not is_valid and self.log_invalid_transitions:
            self.logger.error(
                f"❌ INVALID_TRANSITION: {component_name} {from_state} → {to_state}",
                extra={
                    'validation_errors': event.validation_errors,
                    'component_name': component_name,
                    'from_state': str(from_state),
                    'to_state': str(to_state),
                    'trigger': trigger
                }
            )
        
        # Call callbacks
        for callback in self.change_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")
        
        # Check for anomalies
        if self.detect_anomalies:
            self._check_for_anomalies(event)
        
        return event
    
    def _check_for_anomalies(self, event: StateChangeEvent) -> None:
        """Check for state change anomalies."""
        
        # Check for rapid state changes (possible state thrashing)
        recent_events = self.history.get_component_history(event.component_name, 10)
        if len(recent_events) >= 5:
            recent_time = time.time() - 5  # Last 5 seconds
            rapid_changes = [e for e in recent_events if e.timestamp > recent_time]
            
            if len(rapid_changes) >= 5:
                self.anomalies_detected += 1
                self.logger.warning(
                    f"🚨 STATE_ANOMALY: Rapid state changes detected in {event.component_name}",
                    extra={
                        'anomaly_type': 'rapid_state_changes',
                        'component_name': event.component_name,
                        'changes_in_5_seconds': len(rapid_changes),
                        'recent_events': [e.to_dict() for e in rapid_changes]
                    }
                )
                
                # Call anomaly callbacks
                for callback in self.anomaly_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        self.logger.error(f"Anomaly callback error: {e}")
        
        # Check for circular state changes
        if len(recent_events) >= 3:
            last_three = recent_events[-3:]
            states = [str(e.to_state) for e in last_three]
            
            if len(set(states)) == 2 and states[0] == states[2]:
                self.anomalies_detected += 1
                self.logger.warning(
                    f"🚨 STATE_ANOMALY: Circular state changes detected in {event.component_name}",
                    extra={
                        'anomaly_type': 'circular_state_changes',
                        'component_name': event.component_name,
                        'state_sequence': states
                    }
                )
    
    def add_change_callback(self, callback: Callable[[StateChangeEvent], None]) -> None:
        """Add callback to be called on every state change."""
        self.change_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable[[StateChangeEvent], None]) -> None:
        """Add callback to be called when anomalies are detected."""
        self.anomaly_callbacks.append(callback)
    
    def get_current_states(self) -> Dict[str, Any]:
        """Get current states of all monitored components."""
        return self.history.current_states.copy()
    
    def get_component_state(self, component_name: str) -> Any:
        """Get current state of a specific component."""
        return self.history.get_current_state(component_name)
    
    def get_state_history(self, component_name: str, limit: int = 100) -> List[StateChangeEvent]:
        """Get state change history for a component."""
        return self.history.get_component_history(component_name, limit)
    
    def get_recent_changes(self, seconds: int = 60) -> List[StateChangeEvent]:
        """Get recent state changes across all components."""
        return self.history.get_recent_events(seconds)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get state monitoring performance statistics."""
        history_stats = self.history.get_stats()
        
        return {
            'total_changes_monitored': self.total_changes_monitored,
            'invalid_transitions_detected': self.invalid_transitions_detected,
            'anomalies_detected': self.anomalies_detected,
            'enabled': self.enabled,
            'history_stats': history_stats
        }
    
    def enable(self) -> None:
        """Enable state monitoring."""
        self.enabled = True
        self.logger.info("🔄 State monitoring enabled")
    
    def disable(self) -> None:
        """Disable state monitoring."""
        self.enabled = False
        self.logger.info("🔄 State monitoring disabled")


# Global state monitor instance
_global_state_monitor: Optional[StateMonitor] = None
_monitor_lock = threading.Lock()


def get_state_monitor() -> StateMonitor:
    """Get global state monitor instance."""
    global _global_state_monitor
    
    if _global_state_monitor is None:
        with _monitor_lock:
            if _global_state_monitor is None:
                _global_state_monitor = StateMonitor()
    
    return _global_state_monitor


def log_state_change(component_name: str,
                    from_state: Any,
                    to_state: Any,
                    trigger: str = "",
                    context: Optional[Dict[str, Any]] = None,
                    **kwargs) -> StateChangeEvent:
    """
    Convenience function to log state changes.
    
    Uses the global state monitor to log state changes.
    """
    monitor = get_state_monitor()
    return monitor.log_state_change(
        component_name=component_name,
        from_state=from_state,
        to_state=to_state,
        trigger=trigger,
        context=context,
        **kwargs
    )