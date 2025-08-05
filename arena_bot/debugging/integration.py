"""
Deep Debugging System Integration for Arena Bot

Comprehensive integration module that connects all debugging components
with the existing Arena Bot infrastructure:

- Seamless integration with existing S-tier logging system
- Automatic instrumentation of core Arena Bot components
- Integration with IntegratedArenaBotGUI debug modes
- Enhanced error handling throughout the application
- Performance monitoring integration
- Configuration system updates for debug profiles
- Backward compatibility with existing functionality

This module serves as the main entry point for enabling deep debugging
across the entire Arena Bot application.
"""

import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Set
from pathlib import Path

# Import debugging components
from .method_tracer import get_method_tracer, trace_method
from .state_monitor import get_state_monitor, log_state_change
from .pipeline_tracer import get_pipeline_tracer, trace_pipeline_stage, PipelineStage
from .health_monitor import get_health_monitor, circuit_breaker
from .error_analyzer import get_error_analyzer, analyze_error
from .exception_handler import get_exception_handler, handle_exception, deep_exception_handler
from .enhanced_logger import get_enhanced_logger, enable_debug_logging, activate_emergency_debug

# Import existing Arena Bot components for integration
try:
    from ..logging_system.logger import get_logger as get_base_logger
except ImportError:
    # Fallback if import fails
    def get_base_logger(name):
        import logging
        return logging.getLogger(name)


class DeepDebuggingIntegrator:
    """
    Central integration manager for deep debugging system.
    
    Coordinates all debugging components and provides unified
    interface for enabling/disabling debugging features.
    """
    
    def __init__(self):
        """Initialize debugging integrator."""
        self.logger = get_enhanced_logger("arena_bot.debugging.integration")
        
        # Component references
        self.method_tracer = None
        self.state_monitor = None
        self.pipeline_tracer = None
        self.health_monitor = None
        self.error_analyzer = None
        self.exception_handler = None
        
        # Integration state
        self.is_initialized = False
        self.is_enabled = False
        self.integration_level = "standard"  # standard, detailed, ultra
        
        # Performance tracking
        self.integration_start_time = 0
        self.components_integrated = 0
        self.integration_errors = 0
        
        # Configuration
        self.auto_instrument_components = True
        self.enable_background_monitoring = True
        self.emergency_debug_threshold = 5  # errors before emergency mode
        
        # Thread safety
        self.lock = threading.RLock()
    
    def initialize(self, 
                  integration_level: str = "standard",
                  auto_start_monitoring: bool = True) -> bool:
        """
        Initialize the deep debugging system.
        
        Args:
            integration_level: Level of integration (standard, detailed, ultra)
            auto_start_monitoring: Whether to start background monitoring
            
        Returns:
            True if initialization successful, False otherwise
        """
        
        if self.is_initialized:
            self.logger.warning("Deep debugging system already initialized")
            return True
        
        self.integration_start_time = time.time()
        
        try:
            with self.lock:
                self.logger.info("🚀 Initializing deep debugging system...")
                
                # Set integration level
                self.integration_level = integration_level
                
                # Initialize core components
                success = self._initialize_core_components()
                if not success:
                    self.logger.error("Failed to initialize core components")
                    return False
                
                # Setup integration with existing infrastructure
                success = self._setup_infrastructure_integration()
                if not success:
                    self.logger.error("Failed to setup infrastructure integration")
                    return False
                
                # Start background monitoring if requested
                if auto_start_monitoring:
                    self._start_background_monitoring()
                
                # Mark as initialized
                self.is_initialized = True
                
                initialization_time = time.time() - self.integration_start_time
                self.logger.info(
                    f"✅ Deep debugging system initialized successfully "
                    f"(level: {integration_level}, time: {initialization_time:.2f}s)"
                )
                
                return True
                
        except Exception as e:
            self.integration_errors += 1
            self.logger.error(f"Deep debugging initialization failed: {e}")
            return False
    
    def _initialize_core_components(self) -> bool:
        """Initialize all core debugging components."""
        try:
            # Initialize method tracer
            self.method_tracer = get_method_tracer()
            self.components_integrated += 1
            self.logger.debug("✅ Method tracer initialized")
            
            # Initialize state monitor
            self.state_monitor = get_state_monitor()
            self.components_integrated += 1
            self.logger.debug("✅ State monitor initialized")
            
            # Initialize pipeline tracer
            self.pipeline_tracer = get_pipeline_tracer()
            self.components_integrated += 1
            self.logger.debug("✅ Pipeline tracer initialized")
            
            # Initialize health monitor
            self.health_monitor = get_health_monitor()
            
            # Start health monitoring if in detailed/ultra mode
            if self.integration_level in ["detailed", "ultra"]:
                self.health_monitor.start_monitoring()
            
            self.components_integrated += 1
            self.logger.debug("✅ Health monitor initialized")
            
            # Initialize error analyzer
            self.error_analyzer = get_error_analyzer()
            self.components_integrated += 1
            self.logger.debug("✅ Error analyzer initialized")
            
            # Initialize exception handler
            self.exception_handler = get_exception_handler()
            self.components_integrated += 1
            self.logger.debug("✅ Exception handler initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Core component initialization failed: {e}")
            return False
    
    def _setup_infrastructure_integration(self) -> bool:
        """Setup integration with existing Arena Bot infrastructure."""
        try:
            # Enable debug logging based on integration level
            if self.integration_level == "standard":
                enable_debug_logging("standard")
            elif self.integration_level == "detailed":
                enable_debug_logging("detailed")
            elif self.integration_level == "ultra":
                enable_debug_logging("ultra")
            
            # Setup error analyzer integration with logging
            if self.error_analyzer:
                self._integrate_error_analyzer()
            
            # Setup health monitoring integration
            if self.health_monitor:
                self._integrate_health_monitoring()
            
            # Setup state monitoring integration
            if self.state_monitor:
                self._integrate_state_monitoring()
            
            self.logger.debug("✅ Infrastructure integration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Infrastructure integration failed: {e}")
            return False
    
    def _integrate_error_analyzer(self) -> None:
        """Integrate error analyzer with logging system."""
        # This would be enhanced to hook into the logging system
        # to automatically analyze logged errors
        pass
    
    def _integrate_health_monitoring(self) -> None:
        """Integrate health monitoring with system components."""
        # Add emergency callback for critical health issues
        def emergency_callback(component_name: str, component_health):
            if component_health.overall_status.value in ['critical', 'failed']:
                self.logger.critical(
                    f"🚨 EMERGENCY: Component {component_name} in critical state"
                )
                
                # Activate emergency debug if multiple failures
                if self.health_monitor.total_alerts_generated >= self.emergency_debug_threshold:
                    activate_emergency_debug()
        
        self.health_monitor.add_emergency_callback(emergency_callback)
    
    def _integrate_state_monitoring(self) -> None:
        """Integrate state monitoring with component state changes."""
        # Add callback for state change anomalies
        def anomaly_callback(state_event):
            self.logger.warning(
                f"🚨 STATE_ANOMALY: {state_event.component_name} - "
                f"anomaly detected in state transitions"
            )
        
        self.state_monitor.add_anomaly_callback(anomaly_callback)
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if not self.enable_background_monitoring:
            return
        
        try:
            # Start health monitoring background task
            if self.health_monitor and not self.health_monitor.monitoring_thread:
                self.health_monitor.start_monitoring()
            
            self.logger.debug("✅ Background monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background monitoring: {e}")
    
    def enable_debugging(self, level: str = None) -> None:
        """
        Enable debugging with specified level.
        
        Args:
            level: Debug level (standard, detailed, ultra, emergency)
                  If None, uses current integration level
        """
        
        if not self.is_initialized:
            self.logger.error("Deep debugging system not initialized")
            return
        
        with self.lock:
            debug_level = level or self.integration_level
            
            # Enable debug logging
            if debug_level == "emergency":
                activate_emergency_debug()
            else:
                enable_debug_logging(debug_level)
            
            # Enable component tracing based on level
            if debug_level in ["detailed", "ultra", "emergency"]:
                # Enable more comprehensive tracing
                if self.method_tracer:
                    self.method_tracer.capture_parameters = True
                    self.method_tracer.capture_memory = True
                
                if self.pipeline_tracer:
                    self.pipeline_tracer.capture_data_snapshots = True
                    self.pipeline_tracer.detect_bottlenecks = True
            
            if debug_level in ["ultra", "emergency"]:
                # Enable ultra-detailed debugging
                if self.exception_handler:
                    self.exception_handler.capture_full_context = True
                    self.exception_handler.auto_analyze = True
                
                if self.error_analyzer:
                    self.error_analyzer.auto_analyze = True
            
            self.is_enabled = True
            
            self.logger.info(f"🐛 Deep debugging enabled (level: {debug_level})")
    
    def disable_debugging(self) -> None:
        """Disable all debugging features."""
        
        with self.lock:
            # Disable debug logging
            from .enhanced_logger import disable_debug_logging
            disable_debug_logging()
            
            # Disable component features
            if self.method_tracer:
                self.method_tracer.capture_parameters = False
                self.method_tracer.capture_memory = False
            
            if self.pipeline_tracer:
                self.pipeline_tracer.capture_data_snapshots = False
                self.pipeline_tracer.detect_bottlenecks = False
            
            if self.exception_handler:
                self.exception_handler.capture_full_context = False
                self.exception_handler.auto_analyze = False
            
            if self.error_analyzer:
                self.error_analyzer.auto_analyze = False
            
            self.is_enabled = False
            
            self.logger.info("🐛 Deep debugging disabled")
    
    def instrument_component(self, 
                           component_instance: Any,
                           component_name: str,
                           methods_to_trace: Optional[List[str]] = None) -> bool:
        """
        Automatically instrument a component with debugging.
        
        Args:
            component_instance: Instance of component to instrument
            component_name: Name of the component
            methods_to_trace: Specific methods to trace, or None for auto-detection
            
        Returns:
            True if instrumentation successful, False otherwise
        """
        
        if not self.is_initialized:
            return False
        
        try:
            # Auto-detect methods if not specified
            if methods_to_trace is None:
                methods_to_trace = [
                    method for method in dir(component_instance)
                    if (callable(getattr(component_instance, method)) and 
                        not method.startswith('_') and
                        method not in ['info', 'debug', 'warning', 'error'])
                ]
            
            # Apply method tracing
            instrumented_methods = 0
            for method_name in methods_to_trace:
                try:
                    if hasattr(component_instance, method_name):
                        original_method = getattr(component_instance, method_name)
                        
                        # Create traced version
                        traced_method = trace_method(
                            capture_args=True,
                            capture_timing=True,
                            capture_memory=self.integration_level in ["detailed", "ultra"]
                        )(original_method)
                        
                        # Replace method
                        setattr(component_instance, method_name, traced_method)
                        instrumented_methods += 1
                        
                except Exception as e:
                    self.logger.debug(f"Failed to instrument {method_name}: {e}")
            
            self.logger.info(
                f"🔧 Instrumented component {component_name} "
                f"({instrumented_methods} methods traced)"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component instrumentation failed: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        
        uptime = time.time() - self.integration_start_time if self.integration_start_time else 0
        
        status = {
            'is_initialized': self.is_initialized,
            'is_enabled': self.is_enabled,
            'integration_level': self.integration_level,
            'uptime_seconds': uptime,
            'components_integrated': self.components_integrated,
            'integration_errors': self.integration_errors,
            'component_status': {}
        }
        
        # Get component status
        if self.method_tracer:
            status['component_status']['method_tracer'] = self.method_tracer.get_performance_stats()
        
        if self.state_monitor:
            status['component_status']['state_monitor'] = self.state_monitor.get_performance_stats()
        
        if self.pipeline_tracer:
            status['component_status']['pipeline_tracer'] = self.pipeline_tracer.get_performance_stats()
        
        if self.health_monitor:
            status['component_status']['health_monitor'] = self.health_monitor.get_performance_stats() if hasattr(self.health_monitor, 'get_performance_stats') else {}
        
        if self.error_analyzer:
            status['component_status']['error_analyzer'] = self.error_analyzer.get_performance_stats()
        
        if self.exception_handler:
            status['component_status']['exception_handler'] = self.exception_handler.get_performance_stats()
        
        return status
    
    def shutdown(self) -> None:
        """Gracefully shutdown the debugging system."""
        
        with self.lock:
            self.logger.info("🔄 Shutting down deep debugging system...")
            
            # Disable debugging
            if self.is_enabled:
                self.disable_debugging()
            
            # Stop health monitoring
            if self.health_monitor:
                self.health_monitor.stop_monitoring_thread()
            
            # Mark as not initialized
            self.is_initialized = False
            
            self.logger.info("✅ Deep debugging system shutdown complete")


# Global integrator instance
_global_integrator: Optional[DeepDebuggingIntegrator] = None
_integrator_lock = threading.Lock()


def get_debugging_integrator() -> DeepDebuggingIntegrator:
    """Get global debugging integrator instance."""
    global _global_integrator
    
    if _global_integrator is None:
        with _integrator_lock:
            if _global_integrator is None:
                _global_integrator = DeepDebuggingIntegrator()
    
    return _global_integrator


def initialize_deep_debugging(integration_level: str = "standard",
                            auto_start_monitoring: bool = True) -> bool:
    """
    Initialize the deep debugging system.
    
    Args:
        integration_level: Level of debugging integration
        auto_start_monitoring: Whether to start background monitoring
        
    Returns:
        True if successful, False otherwise
    """
    integrator = get_debugging_integrator()
    return integrator.initialize(integration_level, auto_start_monitoring)


def enable_deep_debugging(level: str = "standard") -> None:
    """Enable deep debugging with specified level."""
    integrator = get_debugging_integrator()
    
    if not integrator.is_initialized:
        initialize_deep_debugging(level)
    
    integrator.enable_debugging(level)
    
    # If ultra level is requested, also enable ultra-debug mode
    if level == "ultra":
        try:
            from .ultra_debug import enable_ultra_debug, UltraDebugMode
            enable_ultra_debug(UltraDebugMode.INTROSPECTION)
        except ImportError:
            pass  # Ultra-debug not available


def disable_deep_debugging() -> None:
    """Disable deep debugging."""
    integrator = get_debugging_integrator()
    integrator.disable_debugging()
    
    # Also disable ultra-debug mode if active
    try:
        from .ultra_debug import disable_ultra_debug
        disable_ultra_debug()
    except ImportError:
        pass  # Ultra-debug not available


def instrument_arena_bot_component(component_instance: Any, 
                                 component_name: str,
                                 methods_to_trace: Optional[List[str]] = None) -> bool:
    """
    Instrument an Arena Bot component with debugging.
    
    Convenience function for instrumenting Arena Bot components.
    """
    integrator = get_debugging_integrator()
    
    if not integrator.is_initialized:
        initialize_deep_debugging()
    
    return integrator.instrument_component(component_instance, component_name, methods_to_trace)


def get_debugging_status() -> Dict[str, Any]:
    """Get comprehensive debugging system status."""
    integrator = get_debugging_integrator()
    return integrator.get_integration_status()


# Convenience decorators for easy integration
def auto_instrument(component_name: str, methods: Optional[List[str]] = None) -> Callable:
    """
    Class decorator to automatically instrument a component.
    
    Usage:
        @auto_instrument("my_component", ["method1", "method2"])
        class MyComponent:
            def method1(self):
                pass
    """
    
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Instrument after initialization
            instrument_arena_bot_component(self, component_name, methods)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def debug_mode_aware(component_name: str = "") -> Callable:
    """
    Method decorator that logs state changes and performance.
    
    Usage:
        @debug_mode_aware("my_component")
        def my_method(self):
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **kwargs):
            integrator = get_debugging_integrator()
            
            if not integrator.is_enabled:
                return func(*args, **kwargs)
            
            # Log method entry
            actual_component_name = component_name or getattr(args[0], '__class__', {}).get('__name__', 'unknown')
            
            logger = get_enhanced_logger(f"arena_bot.{actual_component_name}")
            logger.method_trace(
                f"🔍 METHOD_ENTRY: {func.__name__}",
                debug_context={
                    'component_name': actual_component_name,
                    'method_name': func.__name__,
                    'capture_level': 1
                }
            )
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.method_trace(
                    f"✅ METHOD_SUCCESS: {func.__name__} ({duration_ms:.2f}ms)",
                    debug_context={
                        'component_name': actual_component_name,
                        'method_name': func.__name__,
                        'duration_ms': duration_ms,
                        'capture_level': 1
                    }
                )
                
                return result
                
            except Exception as e:
                # Log exception
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.exception_deep(
                    f"❌ METHOD_EXCEPTION: {func.__name__} ({duration_ms:.2f}ms)",
                    debug_context={
                        'component_name': actual_component_name,
                        'method_name': func.__name__,
                        'duration_ms': duration_ms,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e),
                        'capture_level': 2
                    }
                )
                
                # Analyze exception if debugging is enabled
                if integrator.error_analyzer:
                    integrator.error_analyzer.analyze_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        component_name=actual_component_name,
                        method_name=func.__name__
                    )
                
                raise
        
        return wrapper
    
    return decorator