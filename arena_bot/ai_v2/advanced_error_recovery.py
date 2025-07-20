"""
Advanced Error Recovery - Graceful Degradation System

Provides sophisticated error recovery with graceful degradation for partial data 
availability, circuit breakers, fallback strategies, and intelligent retry logic.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import functools
import inspect
from collections import defaultdict, deque
import random


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CACHE = "cache"
    DEGRADE = "degrade"
    ABORT = "abort"


class ComponentStatus(Enum):
    """Component availability status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    duration_ms: float = 0.0


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    name: str
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3


class AdvancedErrorRecovery:
    """
    Sophisticated error recovery and graceful degradation system.
    
    Provides circuit breakers, intelligent retry logic, fallback strategies,
    and automatic degradation for partial data availability scenarios.
    """
    
    def __init__(self):
        """Initialize advanced error recovery system."""
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(list)
        self.component_status = {}
        
        # Circuit breakers
        self.circuit_breakers = {}
        self.circuit_breaker_lock = threading.Lock()
        
        # Recovery strategies
        self.recovery_strategies = {}
        self.fallback_handlers = {}
        self.degradation_handlers = {}
        
        # Retry configuration
        self.retry_configs = {
            'default': {
                'max_attempts': 3,
                'base_delay': 1.0,
                'max_delay': 30.0,
                'exponential_base': 2.0,
                'jitter': True
            },
            'api_calls': {
                'max_attempts': 5,
                'base_delay': 2.0,
                'max_delay': 60.0,
                'exponential_base': 2.0,
                'jitter': True
            },
            'cache_operations': {
                'max_attempts': 2,
                'base_delay': 0.5,
                'max_delay': 5.0,
                'exponential_base': 1.5,
                'jitter': False
            }
        }
        
        # Performance tracking
        self.recovery_metrics = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'circuit_breaker_trips': 0,
            'fallback_activations': 0,
            'degradation_activations': 0
        }
        
        # Component dependencies
        self.component_dependencies = {
            'hero_selector': ['hsreplay_api', 'cache_manager'],
            'card_evaluator': ['hsreplay_api', 'cache_manager', 'cards_database'],
            'grandmaster_advisor': ['hero_selector', 'card_evaluator'],
            'conversational_coach': ['grandmaster_advisor'],
            'draft_exporter': ['cache_manager']
        }
        
        # Degradation levels for different scenarios
        self.degradation_levels = {
            'hero_data_unavailable': {
                'level': 1,
                'description': 'Use cached hero data and qualitative analysis',
                'affected_components': ['hero_selector'],
                'fallback_data': 'cached_hero_preferences'
            },
            'card_data_unavailable': {
                'level': 2,
                'description': 'Use heuristic-based card evaluation',
                'affected_components': ['card_evaluator'],
                'fallback_data': 'card_database_only'
            },
            'api_completely_down': {
                'level': 3,
                'description': 'Full offline mode with cached data only',
                'affected_components': ['hero_selector', 'card_evaluator'],
                'fallback_data': 'all_cached_data'
            },
            'memory_pressure': {
                'level': 1,
                'description': 'Reduce cache size and disable non-essential features',
                'affected_components': ['cache_manager', 'draft_exporter'],
                'fallback_data': 'minimal_cache'
            }
        }
        
        self.logger.info("AdvancedErrorRecovery initialized")
    
    def register_component(self, component_name: str, health_check: Optional[Callable] = None,
                         fallback_handler: Optional[Callable] = None,
                         degradation_handler: Optional[Callable] = None) -> None:
        """Register a component for error recovery management."""
        self.component_status[component_name] = ComponentStatus.HEALTHY
        
        if fallback_handler:
            self.fallback_handlers[component_name] = fallback_handler
        
        if degradation_handler:
            self.degradation_handlers[component_name] = degradation_handler
        
        # Create circuit breaker for component
        self._create_circuit_breaker(component_name)
        
        self.logger.debug(f"Registered component for error recovery: {component_name}")
    
    def with_error_recovery(self, component: str, operation: str = "default",
                          retry_config: Optional[str] = None, fallback: Optional[Callable] = None):
        """Decorator for automatic error recovery."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_recovery(
                    func, component, operation, retry_config, fallback, *args, **kwargs
                )
            return wrapper
        return decorator
    
    def execute_with_recovery(self, func: Callable, component: str, operation: str = "default",
                            retry_config: Optional[str] = None, fallback: Optional[Callable] = None,
                            *args, **kwargs) -> Any:
        """Execute function with comprehensive error recovery."""
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if not self._check_circuit_breaker(component):
                return self._handle_circuit_breaker_open(component, fallback, *args, **kwargs)
            
            # Get retry configuration
            config = self.retry_configs.get(retry_config or 'default', self.retry_configs['default'])
            
            # Attempt execution with retries
            last_exception = None
            for attempt in range(config['max_attempts']):
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record success
                    self._record_success(component, operation, start_time)
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Record error
                    error_record = self._record_error(component, operation, e, start_time)
                    
                    # Check if we should retry
                    if attempt < config['max_attempts'] - 1 and self._should_retry(e, error_record):
                        delay = self._calculate_retry_delay(attempt, config)
                        self.logger.debug(f"Retrying {component}.{operation} in {delay:.2f}s (attempt {attempt + 1})")
                        time.sleep(delay)
                        continue
                    else:
                        break
            
            # All retries failed, try recovery strategies
            return self._attempt_recovery(component, operation, last_exception, fallback, *args, **kwargs)
            
        except Exception as e:
            # Ultimate fallback
            self.logger.error(f"Complete failure in {component}.{operation}: {e}")
            self.recovery_metrics['failed_recoveries'] += 1
            
            if fallback:
                try:
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
            
            raise e
    
    def trigger_graceful_degradation(self, scenario: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger graceful degradation for specific scenario."""
        try:
            if scenario not in self.degradation_levels:
                self.logger.error(f"Unknown degradation scenario: {scenario}")
                return False
            
            degradation_config = self.degradation_levels[scenario]
            
            self.logger.warning(f"Triggering graceful degradation: {degradation_config['description']}")
            
            # Update component statuses
            for component in degradation_config['affected_components']:
                self.component_status[component] = ComponentStatus.DEGRADED
                
                # Call component-specific degradation handler
                if component in self.degradation_handlers:
                    try:
                        self.degradation_handlers[component](scenario, context or {})
                    except Exception as e:
                        self.logger.error(f"Error in degradation handler for {component}: {e}")
            
            # Apply system-wide degradation measures
            self._apply_system_degradation(degradation_config, context or {})
            
            self.recovery_metrics['degradation_activations'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error triggering graceful degradation: {e}")
            return False
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health and recovery status."""
        try:
            health_report = {
                'overall_status': 'healthy',
                'component_status': {},
                'circuit_breakers': {},
                'error_summary': {},
                'recovery_metrics': self.recovery_metrics.copy(),
                'degraded_components': []
            }
            
            # Check component status
            for component, status in self.component_status.items():
                health_report['component_status'][component] = status.value
                if status in [ComponentStatus.DEGRADED, ComponentStatus.FAILING, ComponentStatus.OFFLINE]:
                    health_report['degraded_components'].append(component)
            
            # Check circuit breakers
            with self.circuit_breaker_lock:
                for name, breaker in self.circuit_breakers.items():
                    health_report['circuit_breakers'][name] = {
                        'state': breaker.state,
                        'failure_count': breaker.failure_count,
                        'last_failure': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
                    }
            
            # Analyze recent errors
            recent_errors = [e for e in self.error_history if 
                           (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
            
            health_report['error_summary'] = {
                'total_recent_errors': len(recent_errors),
                'critical_errors': len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]),
                'high_errors': len([e for e in recent_errors if e.severity == ErrorSeverity.HIGH]),
                'error_rate_per_hour': len(recent_errors)
            }
            
            # Determine overall status
            if health_report['degraded_components']:
                if any(self.component_status[comp] == ComponentStatus.OFFLINE for comp in health_report['degraded_components']):
                    health_report['overall_status'] = 'critical'
                else:
                    health_report['overall_status'] = 'degraded'
            elif health_report['error_summary']['error_rate_per_hour'] > 50:
                health_report['overall_status'] = 'warning'
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {'error': str(e), 'overall_status': 'unknown'}
    
    def recover_component(self, component: str) -> bool:
        """Attempt to recover a specific component."""
        try:
            self.logger.info(f"Attempting to recover component: {component}")
            
            # Reset circuit breaker
            self._reset_circuit_breaker(component)
            
            # Update component status
            self.component_status[component] = ComponentStatus.HEALTHY
            
            # Call component-specific recovery if available
            if component in self.fallback_handlers:
                try:
                    recovery_result = self.fallback_handlers[component]()
                    if recovery_result:
                        self.logger.info(f"Component {component} recovered successfully")
                        return True
                except Exception as e:
                    self.logger.error(f"Component recovery handler failed for {component}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recovering component {component}: {e}")
            return False
    
    # === INTERNAL METHODS ===
    
    def _create_circuit_breaker(self, component: str, failure_threshold: int = 5) -> None:
        """Create circuit breaker for component."""
        with self.circuit_breaker_lock:
            self.circuit_breakers[component] = CircuitBreakerState(
                name=component,
                failure_threshold=failure_threshold
            )
    
    def _check_circuit_breaker(self, component: str) -> bool:
        """Check if circuit breaker allows operation."""
        with self.circuit_breaker_lock:
            if component not in self.circuit_breakers:
                return True
            
            breaker = self.circuit_breakers[component]
            current_time = datetime.now()
            
            if breaker.state == "CLOSED":
                return True
            elif breaker.state == "OPEN":
                # Check if recovery timeout has passed
                if (breaker.last_failure_time and 
                    (current_time - breaker.last_failure_time).total_seconds() > breaker.recovery_timeout):
                    breaker.state = "HALF_OPEN"
                    breaker.success_count = 0
                    return True
                return False
            elif breaker.state == "HALF_OPEN":
                return breaker.success_count < breaker.half_open_max_calls
            
            return False
    
    def _record_success(self, component: str, operation: str, start_time: float) -> None:
        """Record successful operation."""
        duration_ms = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        with self.circuit_breaker_lock:
            if component in self.circuit_breakers:
                breaker = self.circuit_breakers[component]
                breaker.success_count += 1
                breaker.last_success_time = datetime.now()
                
                if breaker.state == "HALF_OPEN" and breaker.success_count >= breaker.half_open_max_calls:
                    breaker.state = "CLOSED"
                    breaker.failure_count = 0
        
        # Update component status
        if component in self.component_status:
            if self.component_status[component] != ComponentStatus.HEALTHY:
                self.component_status[component] = ComponentStatus.HEALTHY
                self.logger.info(f"Component {component} restored to healthy status")
    
    def _record_error(self, component: str, operation: str, error: Exception, start_time: float) -> ErrorRecord:
        """Record error occurrence."""
        duration_ms = (time.time() - start_time) * 1000
        
        # Determine error severity
        severity = self._determine_error_severity(error)
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            duration_ms=duration_ms
        )
        
        # Store error
        self.error_history.append(error_record)
        self.error_patterns[component].append(error_record)
        self.recovery_metrics['total_errors'] += 1
        
        # Update circuit breaker
        with self.circuit_breaker_lock:
            if component in self.circuit_breakers:
                breaker = self.circuit_breakers[component]
                breaker.failure_count += 1
                breaker.last_failure_time = datetime.now()
                
                if breaker.failure_count >= breaker.failure_threshold and breaker.state == "CLOSED":
                    breaker.state = "OPEN"
                    self.recovery_metrics['circuit_breaker_trips'] += 1
                    self.logger.warning(f"Circuit breaker opened for {component}")
        
        # Update component status
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.component_status[component] = ComponentStatus.FAILING
        elif severity == ErrorSeverity.MEDIUM:
            self.component_status[component] = ComponentStatus.DEGRADED
        
        return error_record
    
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if error_type in ['MemoryError', 'SystemExit', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['ConnectionError', 'Timeout', 'HTTPError']:
            return ErrorSeverity.HIGH
        
        if any(keyword in error_message for keyword in ['database', 'connection', 'network', 'timeout']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['ValueError', 'KeyError', 'IndexError']:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _should_retry(self, error: Exception, error_record: ErrorRecord) -> bool:
        """Determine if operation should be retried."""
        # Don't retry critical errors
        if error_record.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry certain error types
        non_retryable_errors = ['ValueError', 'TypeError', 'AttributeError']
        if type(error).__name__ in non_retryable_errors:
            return False
        
        # Check error message for non-retryable patterns
        non_retryable_patterns = ['permission denied', 'unauthorized', 'forbidden', 'not found']
        error_message = str(error).lower()
        if any(pattern in error_message for pattern in non_retryable_patterns):
            return False
        
        return True
    
    def _calculate_retry_delay(self, attempt: int, config: Dict[str, Any]) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base_delay = config['base_delay']
        exponential_base = config['exponential_base']
        max_delay = config['max_delay']
        jitter = config['jitter']
        
        # Exponential backoff
        delay = base_delay * (exponential_base ** attempt)
        delay = min(delay, max_delay)
        
        # Add jitter to avoid thundering herd
        if jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _attempt_recovery(self, component: str, operation: str, error: Exception,
                         fallback: Optional[Callable], *args, **kwargs) -> Any:
        """Attempt various recovery strategies."""
        self.logger.info(f"Attempting recovery for {component}.{operation}")
        
        # Strategy 1: Try component-specific fallback
        if component in self.fallback_handlers:
            try:
                result = self.fallback_handlers[component](*args, **kwargs)
                self.recovery_metrics['successful_recoveries'] += 1
                self.recovery_metrics['fallback_activations'] += 1
                return result
            except Exception as e:
                self.logger.debug(f"Component fallback failed: {e}")
        
        # Strategy 2: Try provided fallback
        if fallback:
            try:
                result = fallback(*args, **kwargs)
                self.recovery_metrics['successful_recoveries'] += 1
                return result
            except Exception as e:
                self.logger.debug(f"Provided fallback failed: {e}")
        
        # Strategy 3: Try cached data
        try:
            cached_result = self._try_cached_fallback(component, operation, *args, **kwargs)
            if cached_result is not None:
                self.recovery_metrics['successful_recoveries'] += 1
                return cached_result
        except Exception as e:
            self.logger.debug(f"Cache fallback failed: {e}")
        
        # Strategy 4: Trigger graceful degradation
        degradation_scenario = self._determine_degradation_scenario(component, error)
        if degradation_scenario:
            self.trigger_graceful_degradation(degradation_scenario)
            # Return minimal result
            return self._get_minimal_result(component, operation)
        
        # All recovery strategies failed
        self.recovery_metrics['failed_recoveries'] += 1
        raise error
    
    def _handle_circuit_breaker_open(self, component: str, fallback: Optional[Callable], 
                                   *args, **kwargs) -> Any:
        """Handle circuit breaker open state."""
        self.logger.warning(f"Circuit breaker open for {component}, using fallback")
        
        if fallback:
            return fallback(*args, **kwargs)
        
        # Try cached data
        cached_result = self._try_cached_fallback(component, "circuit_breaker", *args, **kwargs)
        if cached_result is not None:
            return cached_result
        
        # Return minimal result
        return self._get_minimal_result(component, "circuit_breaker")
    
    def _try_cached_fallback(self, component: str, operation: str, *args, **kwargs) -> Any:
        """Try to get result from cache."""
        try:
            # This would integrate with the cache manager
            # For now, return None to indicate cache miss
            return None
        except Exception:
            return None
    
    def _get_minimal_result(self, component: str, operation: str) -> Any:
        """Get minimal result for graceful degradation."""
        # Return component-appropriate minimal results
        minimal_results = {
            'hero_selector': {'recommended_hero_index': 0, 'confidence_level': 0.3},
            'card_evaluator': {'score': 0.5, 'confidence': 0.2},
            'grandmaster_advisor': {'recommended_pick_index': 0, 'confidence_level': 0.2},
            'conversational_coach': 'I\'m having trouble right now, but let\'s continue.',
            'draft_exporter': {'success': False, 'message': 'Export temporarily unavailable'}
        }
        
        return minimal_results.get(component, {'status': 'degraded'})
    
    def _determine_degradation_scenario(self, component: str, error: Exception) -> Optional[str]:
        """Determine appropriate degradation scenario."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if 'memory' in error_message or error_type == 'MemoryError':
            return 'memory_pressure'
        elif 'api' in error_message or 'connection' in error_message:
            if component == 'hero_selector':
                return 'hero_data_unavailable'
            elif component == 'card_evaluator':
                return 'card_data_unavailable'
            else:
                return 'api_completely_down'
        
        return None
    
    def _apply_system_degradation(self, degradation_config: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Apply system-wide degradation measures."""
        try:
            fallback_data = degradation_config['fallback_data']
            
            if fallback_data == 'minimal_cache':
                # Reduce cache size
                try:
                    from .resource_manager import get_resource_manager
                    resource_manager = get_resource_manager()
                    resource_manager.optimize_memory(force=True)
                except Exception as e:
                    self.logger.debug(f"Could not optimize memory: {e}")
            
            elif fallback_data == 'all_cached_data':
                # Switch to full offline mode
                self.logger.warning("Switching to full offline mode")
                # Implementation would disable API calls and use only cached data
            
        except Exception as e:
            self.logger.error(f"Error applying system degradation: {e}")
    
    def _reset_circuit_breaker(self, component: str) -> None:
        """Reset circuit breaker to closed state."""
        with self.circuit_breaker_lock:
            if component in self.circuit_breakers:
                breaker = self.circuit_breakers[component]
                breaker.state = "CLOSED"
                breaker.failure_count = 0
                breaker.success_count = 0
                self.logger.info(f"Reset circuit breaker for {component}")


# Global error recovery instance
_error_recovery = None

def get_error_recovery() -> AdvancedErrorRecovery:
    """Get global error recovery instance."""
    global _error_recovery
    if _error_recovery is None:
        _error_recovery = AdvancedErrorRecovery()
    return _error_recovery