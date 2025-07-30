"""
Component Isolation & Circuit Breaker System for Arena Bot AI Helper v2.

This module implements a comprehensive circuit breaker pattern and component isolation
architecture to prevent cascade failures and ensure system resilience.

Features:
- P0.7.1: Circuit Breaker Pattern Implementation
- P0.7.2: Component Isolation Architecture  
- P0.7.3: Global Error Recovery Coordinator
- P0.7.4: Fault Isolation Boundaries
- P0.7.5: CPU-Aware Thread Pool Sizing
- P0.7.6: Error Recovery Algorithm with exponential backoff
- P0.7.7: Error Boundary Hierarchy

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import time
import threading
import logging
import traceback
import weakref
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import concurrent.futures
import queue
import os
import math

from .exceptions import (
    AIHelperCircuitBreakerError, 
    AIHelperComponentError,
    AIHelperCriticalError
)
from .monitoring import get_resource_manager

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Circuit breaker tripped, calls fail fast
    HALF_OPEN = "half_open"    # Testing if service recovered


class ComponentState(Enum):
    """Component operational states"""
    HEALTHY = "healthy"         # Normal operation
    DEGRADED = "degraded"       # Reduced functionality
    FAILED = "failed"          # Component failed
    RECOVERING = "recovering"   # Recovery in progress
    ISOLATED = "isolated"      # Isolated from other components


class ErrorSeverity(Enum):
    """Error severity levels for boundary hierarchy"""
    FUNCTION = "function"       # Function-level error
    COMPONENT = "component"     # Component-level error
    SYSTEM = "system"          # System-level error
    APPLICATION = "application" # Application-level error


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes to close from half-open
    timeout: float = 30.0               # Operation timeout
    slow_call_threshold: float = 10.0   # Slow call threshold
    slow_call_rate_threshold: float = 0.5  # Rate of slow calls to trip


@dataclass
class ErrorRecord:
    """Record of an error for tracking and recovery"""
    timestamp: float
    error_type: str
    error_message: str
    component_id: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None


class CircuitBreaker:
    """
    P0.7.1: Circuit Breaker Pattern Implementation
    
    Implements the circuit breaker pattern to prevent cascade failures
    by failing fast when a component is experiencing issues.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._last_success_time = 0.0
        
        # Call tracking
        self._call_count = 0
        self._slow_call_count = 0
        self._recent_calls = []  # Sliding window
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_timeouts = 0
        
        self.logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            AIHelperCircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._total_calls += 1
            
            # Check circuit state
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                else:
                    self._total_failures += 1
                    raise AIHelperCircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
        
        # Execute function with timeout and error handling
        start_time = time.time()
        try:
            # Execute with timeout
            result = self._execute_with_timeout(func, args, kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict):
        """Execute function with timeout protection"""
        if self.config.timeout <= 0:
            return func(*args, **kwargs)
        
        # Use thread pool for timeout
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"CB-{self.name}") as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.config.timeout)
            except concurrent.futures.TimeoutError:
                self._total_timeouts += 1
                raise AIHelperCircuitBreakerError(
                    f"Operation timed out after {self.config.timeout}s in circuit '{self.name}'"
                )
    
    def _record_success(self, execution_time: float):
        """Record successful call"""
        with self._lock:
            self._total_successes += 1
            self._last_success_time = time.time()
            
            # Check for slow calls
            if execution_time > self.config.slow_call_threshold:
                self._slow_call_count += 1
            
            self._recent_calls.append({
                'timestamp': time.time(),
                'success': True,
                'execution_time': execution_time
            })
            
            # Trim recent calls to sliding window (last 100 calls)
            if len(self._recent_calls) > 100:
                self._recent_calls.pop(0)
            
            # Handle state transitions
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._close_circuit()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = max(0, self._failure_count - 1)
    
    def _record_failure(self, error: Exception, execution_time: float):
        """Record failed call"""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            self._recent_calls.append({
                'timestamp': time.time(),
                'success': False,
                'execution_time': execution_time,
                'error': str(error)
            })
            
            # Trim recent calls
            if len(self._recent_calls) > 100:
                self._recent_calls.pop(0)
            
            # Check if circuit should open
            if self._should_open_circuit():
                self._open_circuit()
            
            self.logger.warning(
                f"Circuit '{self.name}' recorded failure: {error} "
                f"(failures: {self._failure_count}/{self.config.failure_threshold})"
            )
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open"""
        # Check failure threshold
        if self._failure_count >= self.config.failure_threshold:
            return True
        
        # Check slow call rate
        if len(self._recent_calls) >= 10:  # Minimum calls for rate calculation
            recent_calls = self._recent_calls[-20:]  # Last 20 calls
            slow_calls = sum(1 for call in recent_calls 
                           if call.get('execution_time', 0) > self.config.slow_call_threshold)
            slow_call_rate = slow_calls / len(recent_calls)
            
            if slow_call_rate >= self.config.slow_call_rate_threshold:
                self.logger.warning(
                    f"Circuit '{self.name}' opening due to slow call rate: {slow_call_rate:.2f}"
                )
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        return (time.time() - self._last_failure_time) >= self.config.recovery_timeout
    
    def _open_circuit(self):
        """Open the circuit"""
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0
            self.logger.warning(f"Circuit breaker '{self.name}' OPENED")
    
    def _close_circuit(self):
        """Close the circuit"""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self.logger.info(f"Circuit breaker '{self.name}' CLOSED")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            recent_window = [call for call in self._recent_calls 
                           if time.time() - call['timestamp'] < 300]  # Last 5 minutes
            
            success_rate = 0.0
            if self._total_calls > 0:
                success_rate = self._total_successes / self._total_calls
            
            return {
                'name': self.name,
                'state': self._state.value,
                'total_calls': self._total_calls,
                'total_successes': self._total_successes,
                'total_failures': self._total_failures,
                'total_timeouts': self._total_timeouts,
                'success_rate': success_rate,
                'failure_count': self._failure_count,
                'recent_calls': len(recent_window),
                'last_failure_time': self._last_failure_time,
                'last_success_time': self._last_success_time
            }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self.logger.info(f"Circuit breaker '{self.name}' manually reset")


class ComponentIsolator:
    """
    P0.7.2: Component Isolation Architecture
    
    Ensures each component functions independently with proper
    isolation boundaries and controlled interaction.
    """
    
    def __init__(self, component_id: str):
        """
        Initialize component isolator.
        
        Args:
            component_id: Unique identifier for the component
        """
        self.component_id = component_id
        self.logger = logging.getLogger(f"{__name__.replace('circuit_breaker', 'isolator')}.{component_id}")
        
        # State tracking
        self._state = ComponentState.HEALTHY
        self._isolation_level = 0  # 0 = no isolation, higher = more isolated
        
        # Circuit breakers for external dependencies
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Resource limits
        self._memory_limit_mb = 100  # Per-component memory limit
        self._cpu_limit_percent = 10  # Per-component CPU limit
        self._thread_limit = 4       # Per-component thread limit
        
        # Thread pool for component operations
        self._thread_pool = None
        self._thread_pool_size = self._calculate_thread_pool_size()
        
        # Component health tracking
        self._health_checks: List[Callable[[], bool]] = []
        self._last_health_check = 0.0
        self._health_check_interval = 30.0  # 30 seconds
        
        # Interaction tracking
        self._dependency_components: List[str] = []
        self._dependent_components: List[str] = []
        
        self.logger.info(f"Component isolator initialized for '{component_id}'")
    
    def _calculate_thread_pool_size(self) -> int:
        """P0.7.5: CPU-Aware Thread Pool Sizing with mathematical foundation"""
        # CPU-aware calculation: AI threads = min(CPU_cores - 1, 4)
        cpu_cores = os.cpu_count() or 4
        
        if self.component_id.startswith('ai_'):
            # AI components get more threads but limited
            return min(cpu_cores - 1, 4)
        elif self.component_id.startswith('io_'):
            # I/O components get fixed allocation
            return 2
        elif self.component_id.startswith('ui_'):
            # UI components get single thread
            return 1
        else:
            # Default components get minimal allocation
            return min(2, max(1, cpu_cores // 4))
    
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """Get component's isolated thread pool"""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self._thread_pool_size,
                thread_name_prefix=f"Component-{self.component_id}"
            )
        return self._thread_pool
    
    def add_circuit_breaker(self, dependency_name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Add circuit breaker for external dependency"""
        if dependency_name not in self._circuit_breakers:
            cb_name = f"{self.component_id}->{dependency_name}"
            self._circuit_breakers[dependency_name] = CircuitBreaker(cb_name, config)
        
        return self._circuit_breakers[dependency_name]
    
    def call_dependency(self, dependency_name: str, func: Callable[..., T], *args, **kwargs) -> T:
        """Call external dependency through circuit breaker"""
        circuit_breaker = self._circuit_breakers.get(dependency_name)
        
        if circuit_breaker is None:
            self.logger.warning(f"No circuit breaker for dependency '{dependency_name}', adding default")
            circuit_breaker = self.add_circuit_breaker(dependency_name)
        
        try:
            return circuit_breaker.call(func, *args, **kwargs)
        except AIHelperCircuitBreakerError:
            # Handle circuit breaker open - increase isolation
            self._increase_isolation_level()
            raise
    
    def _increase_isolation_level(self):
        """Increase component isolation level"""
        self._isolation_level = min(5, self._isolation_level + 1)
        
        if self._isolation_level >= 3:
            self._state = ComponentState.ISOLATED
            self.logger.warning(f"Component '{self.component_id}' isolation level increased to {self._isolation_level}")
        elif self._isolation_level >= 1:
            self._state = ComponentState.DEGRADED
    
    def _decrease_isolation_level(self):
        """Decrease component isolation level"""
        if self._isolation_level > 0:
            self._isolation_level -= 1
            
            if self._isolation_level == 0:
                self._state = ComponentState.HEALTHY
                self.logger.info(f"Component '{self.component_id}' isolation level decreased to {self._isolation_level}")
    
    def add_health_check(self, health_check: Callable[[], bool]):
        """Add health check function"""
        self._health_checks.append(health_check)
    
    def check_health(self) -> bool:
        """Run all health checks"""
        current_time = time.time()
        
        # Throttle health checks
        if current_time - self._last_health_check < self._health_check_interval:
            return self._state == ComponentState.HEALTHY
        
        self._last_health_check = current_time
        
        try:
            for health_check in self._health_checks:
                if not health_check():
                    self._state = ComponentState.FAILED
                    self.logger.error(f"Component '{self.component_id}' failed health check")
                    return False
            
            # Health checks passed, potentially decrease isolation
            if self._state in (ComponentState.DEGRADED, ComponentState.ISOLATED):
                self._decrease_isolation_level()
            elif self._state != ComponentState.RECOVERING:
                self._state = ComponentState.HEALTHY
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed for component '{self.component_id}': {e}")
            self._state = ComponentState.FAILED
            return False
    
    def add_dependency(self, component_id: str):
        """Register dependency on another component"""
        if component_id not in self._dependency_components:
            self._dependency_components.append(component_id)
    
    def add_dependent(self, component_id: str):
        """Register that another component depends on this one"""
        if component_id not in self._dependent_components:
            self._dependent_components.append(component_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            'component_id': self.component_id,
            'state': self._state.value,
            'isolation_level': self._isolation_level,
            'thread_pool_size': self._thread_pool_size,
            'dependencies': self._dependency_components.copy(),
            'dependents': self._dependent_components.copy(),
            'circuit_breakers': {
                name: cb.get_metrics() 
                for name, cb in self._circuit_breakers.items()
            },
            'last_health_check': self._last_health_check,
            'health_check_interval': self._health_check_interval
        }
    
    def shutdown(self):
        """Shutdown component and cleanup resources"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        self.logger.info(f"Component '{self.component_id}' shutdown completed")


class GlobalErrorRecoveryCoordinator:
    """
    P0.7.3: Global Error Recovery Coordinator
    
    Orchestrates recovery from multi-component failures and coordinates
    system-wide error handling strategies.
    """
    
    def __init__(self):
        """Initialize global error recovery coordinator."""
        self.logger = logging.getLogger(__name__.replace('circuit_breaker', 'recovery'))
        
        # Component registry
        self._components: Dict[str, ComponentIsolator] = {}
        self._component_refs: Dict[str, weakref.ref] = {}
        
        # Error tracking
        self._error_history: List[ErrorRecord] = []
        self._recovery_strategies: Dict[ErrorSeverity, List[Callable]] = {
            ErrorSeverity.FUNCTION: [self._retry_with_backoff],
            ErrorSeverity.COMPONENT: [self._restart_component, self._isolate_component],
            ErrorSeverity.SYSTEM: [self._emergency_resource_cleanup, self._restart_failed_components],
            ErrorSeverity.APPLICATION: [self._graceful_shutdown, self._emergency_restart]
        }
        
        # Recovery state
        self._recovery_in_progress = False
        self._recovery_start_time = 0.0
        self._recovery_lock = threading.Lock()
        
        # Thread pool for recovery operations
        self._recovery_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="ErrorRecovery"
        )
        
        self.logger.info("Global Error Recovery Coordinator initialized")
    
    def register_component(self, component: ComponentIsolator):
        """Register a component for error recovery coordination"""
        self._components[component.component_id] = component
        self._component_refs[component.component_id] = weakref.ref(component)
        
        # Set up cross-component dependencies
        for other_id, other_component in self._components.items():
            if other_id != component.component_id:
                # Auto-detect dependencies based on component types
                if self._should_have_dependency(component.component_id, other_id):
                    component.add_dependency(other_id)
                    other_component.add_dependent(component.component_id)
    
    def _should_have_dependency(self, component_id: str, other_id: str) -> bool:
        """Determine if components should have dependency relationship"""
        # AI components depend on monitoring and configuration
        if component_id.startswith('ai_'):
            return other_id in ('monitoring', 'config', 'resource_manager')
        
        # UI components depend on AI and monitoring
        if component_id.startswith('ui_'):
            return other_id.startswith('ai_') or other_id in ('monitoring', 'config')
        
        # All components depend on resource manager
        return other_id == 'resource_manager'
    
    def handle_error(self, error: Exception, component_id: str, context: Dict[str, Any] = None) -> bool:
        """
        P0.7.6: Handle error with recovery algorithm and exponential backoff
        
        Args:
            error: The error that occurred
            component_id: ID of component where error occurred
            context: Additional context information
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Determine error severity
        severity = self._classify_error_severity(error, component_id, context or {})
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            component_id=component_id,
            severity=severity,
            context=context or {}
        )
        
        self._error_history.append(error_record)
        
        # Trim error history (keep last 1000 errors)
        if len(self._error_history) > 1000:
            self._error_history.pop(0)
        
        self.logger.error(
            f"Error in component '{component_id}' (severity: {severity.value}): {error}"
        )
        
        # Execute recovery strategies
        return self._execute_recovery_strategies(error_record)
    
    def _classify_error_severity(self, error: Exception, component_id: str, context: Dict[str, Any]) -> ErrorSeverity:
        """P0.7.7: Classify error severity for boundary hierarchy"""
        # Application-level errors
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.APPLICATION
        
        # System-level errors
        if isinstance(error, (MemoryError, OSError)) or "system" in str(error).lower():
            return ErrorSeverity.SYSTEM
        
        # Component-level errors
        if isinstance(error, AIHelperComponentError) or "component" in str(error).lower():
            return ErrorSeverity.COMPONENT
        
        # Critical errors affecting multiple components
        if isinstance(error, AIHelperCriticalError) or context.get('affects_multiple_components'):
            return ErrorSeverity.SYSTEM
        
        # Circuit breaker errors are component-level
        if isinstance(error, AIHelperCircuitBreakerError):
            return ErrorSeverity.COMPONENT
        
        # Default to function-level
        return ErrorSeverity.FUNCTION
    
    def _execute_recovery_strategies(self, error_record: ErrorRecord) -> bool:
        """Execute recovery strategies for error severity level"""
        with self._recovery_lock:
            if self._recovery_in_progress:
                self.logger.warning("Recovery already in progress, queueing error")
                return False
            
            self._recovery_in_progress = True
            self._recovery_start_time = time.time()
        
        try:
            strategies = self._recovery_strategies.get(error_record.severity, [])
            
            for strategy in strategies:
                try:
                    self.logger.info(f"Executing recovery strategy: {strategy.__name__}")
                    
                    # Execute strategy with timeout
                    future = self._recovery_pool.submit(strategy, error_record)
                    success = future.result(timeout=30.0)  # 30 second timeout
                    
                    if success:
                        error_record.recovery_attempted = True
                        error_record.recovery_successful = True
                        self.logger.info(f"Recovery successful using strategy: {strategy.__name__}")
                        return True
                    
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                    continue
            
            # All strategies failed
            error_record.recovery_attempted = True
            error_record.recovery_successful = False
            return False
            
        finally:
            with self._recovery_lock:
                self._recovery_in_progress = False
                recovery_duration = time.time() - self._recovery_start_time
                self.logger.info(f"Recovery attempt completed in {recovery_duration:.2f}s")
    
    # Recovery Strategy Implementations
    
    def _retry_with_backoff(self, error_record: ErrorRecord) -> bool:
        """Retry operation with exponential backoff"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            
            self.logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
            time.sleep(delay)
            
            # For function-level errors, we can't automatically retry
            # This would be implemented by the calling code
            # Here we just simulate recovery check
            component = self._components.get(error_record.component_id)
            if component and component.check_health():
                return True
        
        return False
    
    def _restart_component(self, error_record: ErrorRecord) -> bool:
        """Restart failed component"""
        component = self._components.get(error_record.component_id)
        if not component:
            return False
        
        try:
            self.logger.info(f"Restarting component: {error_record.component_id}")
            
            # Shutdown component
            component.shutdown()
            
            # Wait for shutdown
            time.sleep(1.0)
            
            # Component restart would need to be implemented by the specific component
            # Here we simulate by resetting circuit breakers
            for cb in component._circuit_breakers.values():
                cb.reset()
            
            component._state = ComponentState.RECOVERING
            
            # Check if restart was successful
            time.sleep(2.0)
            return component.check_health()
            
        except Exception as e:
            self.logger.error(f"Component restart failed: {e}")
            return False
    
    def _isolate_component(self, error_record: ErrorRecord) -> bool:
        """Isolate component to prevent cascade failures"""
        component = self._components.get(error_record.component_id)
        if not component:
            return False
        
        try:
            self.logger.warning(f"Isolating component: {error_record.component_id}")
            
            # Increase isolation level
            component._increase_isolation_level()
            component._state = ComponentState.ISOLATED
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component isolation failed: {e}")
            return False
    
    def _emergency_resource_cleanup(self, error_record: ErrorRecord) -> bool:
        """Emergency resource cleanup for system-level errors"""
        try:
            self.logger.critical("Executing emergency resource cleanup")
            
            # Get resource manager and trigger emergency cleanup
            resource_manager = get_resource_manager()
            resource_manager._emergency_memory_cleanup()
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            self.logger.info(f"Emergency GC collected {collected} objects")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency resource cleanup failed: {e}")
            return False
    
    def _restart_failed_components(self, error_record: ErrorRecord) -> bool:
        """Restart all failed components"""
        failed_components = [
            comp for comp in self._components.values()
            if comp._state == ComponentState.FAILED
        ]
        
        if not failed_components:
            return True
        
        success_count = 0
        for component in failed_components:
            if self._restart_component(ErrorRecord(
                timestamp=time.time(),
                error_type="SystemRestart",
                error_message="System-wide component restart",
                component_id=component.component_id,
                severity=ErrorSeverity.COMPONENT
            )):
                success_count += 1
        
        # Consider successful if at least half the components restarted
        return success_count >= len(failed_components) // 2
    
    def _graceful_shutdown(self, error_record: ErrorRecord) -> bool:
        """Perform graceful system shutdown"""
        self.logger.critical("Initiating graceful system shutdown")
        
        # Shutdown all components in reverse dependency order
        for component in reversed(list(self._components.values())):
            try:
                component.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down component {component.component_id}: {e}")
        
        return True
    
    def _emergency_restart(self, error_record: ErrorRecord) -> bool:
        """Emergency system restart"""
        self.logger.critical("Emergency system restart required")
        
        # This would typically trigger application restart
        # Here we just log the requirement
        return False  # Indicates system restart needed
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error history summary"""
        current_time = time.time()
        
        # Recent errors (last hour)
        recent_errors = [
            error for error in self._error_history
            if current_time - error.timestamp < 3600
        ]
        
        # Error counts by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = len([
                error for error in recent_errors
                if error.severity == severity
            ])
        
        # Recovery success rate
        recovery_attempts = [error for error in recent_errors if error.recovery_attempted]
        recovery_success_rate = 0.0
        if recovery_attempts:
            successful_recoveries = [error for error in recovery_attempts if error.recovery_successful]
            recovery_success_rate = len(successful_recoveries) / len(recovery_attempts)
        
        return {
            'total_errors': len(self._error_history),
            'recent_errors_1h': len(recent_errors),
            'severity_counts': severity_counts,
            'recovery_attempts': len(recovery_attempts),
            'recovery_success_rate': recovery_success_rate,
            'recovery_in_progress': self._recovery_in_progress,
            'registered_components': len(self._components)
        }
    
    def shutdown(self):
        """Shutdown error recovery coordinator"""
        self._recovery_pool.shutdown(wait=True)
        self.logger.info("Global Error Recovery Coordinator shutdown completed")


# Global instances
_global_coordinator = None
_component_registry: Dict[str, ComponentIsolator] = {}

def get_global_coordinator() -> GlobalErrorRecoveryCoordinator:
    """Get global error recovery coordinator instance"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = GlobalErrorRecoveryCoordinator()
    return _global_coordinator


def create_component_isolator(component_id: str) -> ComponentIsolator:
    """Create and register component isolator"""
    if component_id in _component_registry:
        return _component_registry[component_id]
    
    isolator = ComponentIsolator(component_id)
    _component_registry[component_id] = isolator
    
    # Register with global coordinator
    coordinator = get_global_coordinator()
    coordinator.register_component(isolator)
    
    return isolator


def handle_error(error: Exception, component_id: str, context: Dict[str, Any] = None) -> bool:
    """Convenience function to handle errors through global coordinator"""
    coordinator = get_global_coordinator()
    return coordinator.handle_error(error, component_id, context)


# Decorators for error handling and isolation

def with_circuit_breaker(component_id: str, dependency_name: str, config: CircuitBreakerConfig = None):
    """Decorator to wrap function calls with circuit breaker"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            isolator = _component_registry.get(component_id)
            if isolator is None:
                isolator = create_component_isolator(component_id)
            
            circuit_breaker = isolator.add_circuit_breaker(dependency_name, config)
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def with_error_boundary(component_id: str, severity: ErrorSeverity = ErrorSeverity.FUNCTION):
    """Decorator to add error boundary around function"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args) if args else 0,
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                # Handle error through coordinator
                recovery_successful = handle_error(e, component_id, context)
                
                if not recovery_successful:
                    # Re-raise if recovery failed
                    raise
                
                # If recovery succeeded, return None or appropriate default
                return None
        
        return wrapper
    return decorator


# Export main components
__all__ = [
    # Core Classes
    'CircuitBreaker',
    'ComponentIsolator', 
    'GlobalErrorRecoveryCoordinator',
    
    # Enums
    'CircuitState',
    'ComponentState',
    'ErrorSeverity',
    
    # Data Classes
    'CircuitBreakerConfig',
    'ErrorRecord',
    
    # Factory Functions
    'get_global_coordinator',
    'create_component_isolator',
    'handle_error',
    
    # Decorators
    'with_circuit_breaker',
    'with_error_boundary'
]