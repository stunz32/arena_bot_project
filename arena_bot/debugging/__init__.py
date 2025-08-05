"""
Deep Debugging Infrastructure for Arena Bot

Advanced instrumentation, tracing, and analysis systems for complete visibility
into complex failure scenarios and hard-to-track errors.

## Quick Start

```python
from arena_bot.debugging import (
    initialize_deep_debugging, 
    enable_deep_debugging,
    instrument_arena_bot_component
)

# Initialize the debugging system
initialize_deep_debugging(integration_level="detailed")

# Enable debugging
enable_deep_debugging("standard")

# Instrument a component
instrument_arena_bot_component(my_component, "my_component")
```

## Components:
- Method tracing with comprehensive context capture
- State change monitoring and correlation  
- Data flow tracking through the entire pipeline
- Proactive health monitoring and circuit breakers
- Error pattern analysis and prediction
- Deep exception context capture
- Enhanced logging with debug-specific levels
- Seamless integration with existing infrastructure
- Emergency debug mode for critical scenarios
- Ultra-debug mode with maximum visibility and anomaly detection
- Advanced dependency tracking and analysis
- Performance bottleneck detection and optimization
- Automated issue classification and root cause analysis
- Timeline reconstruction and event correlation
- Interactive debugging dashboard with real-time monitoring
- Configurable debug profiles for different environments
"""

# Core debugging components
from .method_tracer import trace_method, MethodTracer, get_method_tracer
from .state_monitor import StateMonitor, StateChangeEvent, get_state_monitor, log_state_change
from .pipeline_tracer import PipelineTracer, DataFlowEvent, get_pipeline_tracer, trace_pipeline_stage, PipelineStage
from .health_monitor import HealthMonitor, ComponentHealth, get_health_monitor, circuit_breaker
from .error_analyzer import ErrorPatternAnalyzer, ErrorPattern, get_error_analyzer, analyze_error
from .exception_handler import DeepExceptionHandler, ExceptionContext, get_exception_handler, handle_exception, deep_exception_handler

# Enhanced logging
from .enhanced_logger import (
    get_enhanced_logger, 
    enable_debug_logging, 
    disable_debug_logging, 
    activate_emergency_debug,
    DebugLogLevel
)

# Integration and convenience functions
from .integration import (
    initialize_deep_debugging,
    enable_deep_debugging, 
    disable_deep_debugging,
    instrument_arena_bot_component,
    get_debugging_status,
    auto_instrument,
    debug_mode_aware,
    get_debugging_integrator
)

# Ultra-debug capabilities
from .ultra_debug import (
    UltraDebugMode,
    UltraDebugManager,
    get_ultra_debug_manager,
    enable_ultra_debug,
    disable_ultra_debug,
    emergency_debug,
    crisis_debug,
    get_ultra_debug_status,
    get_analysis_report,
    ultra_debug_aware
)

# Advanced debugging components
from .dependency_tracker import (
    DependencyTracker,
    get_dependency_tracker,
    start_dependency_tracking,
    stop_dependency_tracking,
    analyze_dependencies,
    get_dependency_status,
    track_dependency,
    DependencyType
)

from .performance_analyzer import (
    PerformanceAnalyzer,
    get_performance_analyzer,
    start_performance_monitoring,
    stop_performance_monitoring,
    analyze_performance,
    get_performance_status,
    profile_performance,
    BottleneckType,
    PerformanceSeverity
)

from .debug_config import (
    DebugConfigManager,
    DebugProfile,
    DebugEnvironment,
    get_debug_config_manager,
    activate_debug_profile,
    get_active_profile,
    create_custom_profile,
    get_profile_recommendations,
    optimize_current_profile,
    get_debug_config_status
)

from .issue_classifier import (
    IssueClassifier,
    ClassifiedIssue,
    IssueCategory,
    IssueSeverity,
    get_issue_classifier,
    start_issue_classification,
    stop_issue_classification,
    classify_issue,
    provide_classification_feedback,
    get_classification_status,
    auto_classify_issues
)

from .timeline_reconstructor import (
    TimelineReconstructor,
    TimelineEvent,
    TimelineSegment,
    EventType,
    EventSeverity,
    get_timeline_reconstructor,
    start_timeline_reconstruction,
    stop_timeline_reconstruction,
    reconstruct_timeline,
    find_root_cause_timeline,
    get_timeline_status,
    add_timeline_event,
    track_timeline_events
)

# Debug dashboard
try:
    from .debug_dashboard import (
        get_debug_dashboard,
        start_debug_dashboard,
        stop_debug_dashboard,
        get_dashboard_url,
        get_dashboard_status,
        launch_dashboard,
        DashboardConfig
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Main exports for easy use
__all__ = [
    # Quick start functions
    'initialize_deep_debugging',
    'enable_deep_debugging',
    'disable_deep_debugging',
    'instrument_arena_bot_component',
    'get_debugging_status',
    
    # Enhanced logging
    'get_enhanced_logger',
    'enable_debug_logging',
    'disable_debug_logging', 
    'activate_emergency_debug',
    'DebugLogLevel',
    
    # Decorators and convenience
    'trace_method',
    'circuit_breaker',
    'deep_exception_handler',
    'auto_instrument',
    'debug_mode_aware',
    'ultra_debug_aware',
    
    # Core component access
    'get_method_tracer',
    'get_state_monitor', 
    'get_pipeline_tracer',
    'get_health_monitor',
    'get_error_analyzer',
    'get_exception_handler',
    'get_debugging_integrator',
    
    # Utility functions
    'log_state_change',
    'analyze_error',
    'handle_exception',
    
    # Data classes and enums
    'StateChangeEvent',
    'DataFlowEvent', 
    'ComponentHealth',
    'ErrorPattern',
    'ExceptionContext',
    'PipelineStage',
    
    # Ultra-debug functions
    'enable_ultra_debug',
    'disable_ultra_debug',
    'emergency_debug',
    'crisis_debug',
    'get_ultra_debug_status',
    'get_analysis_report',
    'get_ultra_debug_manager',
    
    # Advanced debugging functions
    'start_dependency_tracking',
    'stop_dependency_tracking',
    'analyze_dependencies',
    'get_dependency_status',
    'track_dependency',
    'start_performance_monitoring',
    'stop_performance_monitoring',
    'analyze_performance',
    'get_performance_status',
    'profile_performance',
    'start_issue_classification',
    'stop_issue_classification',
    'classify_issue',
    'provide_classification_feedback',
    'get_classification_status',
    'auto_classify_issues',
    'start_timeline_reconstruction',
    'stop_timeline_reconstruction',
    'reconstruct_timeline',
    'find_root_cause_timeline',
    'get_timeline_status',
    'add_timeline_event',
    'track_timeline_events',
    
    # Configuration functions
    'activate_debug_profile',
    'get_active_profile',
    'create_custom_profile',
    'get_profile_recommendations',
    'optimize_current_profile',
    'get_debug_config_status',
    
    # Dashboard functions (if available)
    'launch_dashboard',
    
    # Core classes (for advanced usage)
    'MethodTracer',
    'StateMonitor',
    'PipelineTracer', 
    'HealthMonitor',
    'ErrorPatternAnalyzer',
    'DeepExceptionHandler',
    'UltraDebugManager',
    'UltraDebugMode',
    'DependencyTracker',
    'PerformanceAnalyzer',
    'IssueClassifier',
    'TimelineReconstructor',
    'DebugConfigManager',
    'DebugProfile',
    
    # Data classes and enums
    'DependencyType',
    'BottleneckType',
    'PerformanceSeverity',
    'IssueCategory',
    'IssueSeverity',
    'ClassifiedIssue',
    'TimelineEvent',
    'TimelineSegment',
    'EventType',
    'EventSeverity',
    'DebugEnvironment'
]

__version__ = "3.0.0"