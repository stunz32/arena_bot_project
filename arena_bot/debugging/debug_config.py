"""
Debug Configuration System for Arena Bot Deep Debugging

Advanced configuration management system for debugging profiles and settings:
- Pre-configured debug profiles for different scenarios (development, testing, production)
- Dynamic configuration hot-reloading with validation and rollback capabilities
- Environment-specific settings with inheritance and override mechanisms
- Performance impact estimation and optimization recommendations
- Configuration templates for common debugging scenarios
- Integration with all debugging components with centralized control
- Configuration history and rollback capabilities with change tracking
- Automated optimization based on system performance and resource constraints

This system provides a centralized way to manage all debugging configurations
and profiles, making it easy to switch between different debugging modes
and optimize performance based on current needs.
"""

import os
import json
import yaml
import time
import threading
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from copy import deepcopy
import logging

# Import debugging components
from .enhanced_logger import get_enhanced_logger
from .ultra_debug import UltraDebugMode

from ..logging_system.logger import get_logger, LogLevel


class DebugEnvironment(Enum):
    """Debug environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EMERGENCY = "emergency"


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class DebugProfile:
    """Complete debugging profile configuration."""
    
    # Profile metadata
    profile_name: str = "default"
    description: str = ""
    environment: DebugEnvironment = DebugEnvironment.DEVELOPMENT
    created_time: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    
    # Core debugging settings
    debug_enabled: bool = True
    debug_level: str = "standard"  # minimal, standard, detailed, ultra
    
    # Component-specific settings
    method_tracing: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'capture_parameters': False,
        'capture_return_values': False,
        'capture_memory': False,
        'max_trace_depth': 10,
        'trace_threshold_ms': 1.0
    })
    
    state_monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'track_changes': True,
        'correlation_enabled': True,
        'max_history_size': 1000
    })
    
    pipeline_tracing: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'capture_data_snapshots': False,
        'detect_bottlenecks': True,
        'max_pipeline_history': 500
    })
    
    health_monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'check_interval_seconds': 30,
        'circuit_breaker_enabled': True,
        'auto_recovery': True,
        'health_threshold': 0.8
    })
    
    error_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'pattern_detection': True,
        'ml_analysis': False,
        'correlation_window_minutes': 60
    })
    
    exception_handling: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'capture_full_context': False,
        'auto_analyze': True,
        'attempt_recovery': False
    })
    
    ultra_debug: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'mode': UltraDebugMode.DISABLED.value,
        'monitoring_interval_seconds': 5.0,
        'anomaly_detection': True,
        'auto_escalation': True
    })
    
    dependency_tracking: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'track_imports': True,
        'track_calls': True,
        'auto_analyze': True,
        'analysis_interval_minutes': 5
    })
    
    performance_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'function_profiling': True,
        'memory_leak_detection': True,
        'bottleneck_detection': True,
        'monitoring_interval_seconds': 5.0
    })
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'enhanced_logging': True,
        'debug_log_level': 'DEBUG',
        'emergency_debug': False,
        'log_retention_hours': 24,
        'structured_logging': True
    })
    
    # Dashboard and UI settings
    dashboard: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'auto_open_browser': True,
        'port': 8888,
        'update_interval_seconds': 2.0
    })
    
    # Performance and resource limits
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        'max_memory_usage_mb': 1000,
        'max_cpu_usage_percent': 50,
        'max_debug_overhead_percent': 10,
        'auto_disable_on_limit': True
    })
    
    # Integration settings
    integration: Dict[str, Any] = field(default_factory=lambda: {
        'auto_instrument': False,
        'instrument_external_libs': False,
        'integration_level': 'standard'
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DebugProfile':
        """Create profile from dictionary."""
        # Handle enum conversion
        if 'environment' in data and isinstance(data['environment'], str):
            data['environment'] = DebugEnvironment(data['environment'])
        
        return cls(**data)
    
    def estimate_performance_impact(self) -> Dict[str, Any]:
        """Estimate the performance impact of this profile."""
        impact_score = 0.0
        impact_details = {}
        
        # Method tracing impact
        if self.method_tracing['enabled']:
            tracing_impact = 0.1
            if self.method_tracing['capture_parameters']:
                tracing_impact += 0.05
            if self.method_tracing['capture_return_values']:
                tracing_impact += 0.05
            if self.method_tracing['capture_memory']:
                tracing_impact += 0.1
            
            impact_score += tracing_impact
            impact_details['method_tracing'] = tracing_impact
        
        # Ultra-debug impact
        if self.ultra_debug['enabled']:
            ultra_impact = 0.2
            mode = self.ultra_debug.get('mode', UltraDebugMode.DISABLED.value)
            if mode == UltraDebugMode.EMERGENCY.value:
                ultra_impact = 0.4
            elif mode == UltraDebugMode.CRISIS.value:
                ultra_impact = 0.6
            
            impact_score += ultra_impact
            impact_details['ultra_debug'] = ultra_impact
        
        # Performance analysis impact
        if self.performance_analysis['enabled']:
            perf_impact = 0.15
            if self.performance_analysis['function_profiling']:
                perf_impact += 0.05
            if self.performance_analysis['memory_leak_detection']:
                perf_impact += 0.05
            
            impact_score += perf_impact
            impact_details['performance_analysis'] = perf_impact
        
        # Dependency tracking impact
        if self.dependency_tracking['enabled']:
            dep_impact = 0.1
            if self.dependency_tracking['track_calls']:
                dep_impact += 0.05
            
            impact_score += dep_impact
            impact_details['dependency_tracking'] = dep_impact
        
        # Other components (lighter impact)
        component_impacts = {
            'state_monitoring': 0.02 if self.state_monitoring['enabled'] else 0,
            'pipeline_tracing': 0.03 if self.pipeline_tracing['enabled'] else 0,
            'health_monitoring': 0.01 if self.health_monitoring['enabled'] else 0,
            'error_analysis': 0.02 if self.error_analysis['enabled'] else 0,
            'exception_handling': 0.01 if self.exception_handling['enabled'] else 0
        }
        
        for component, impact in component_impacts.items():
            impact_score += impact
            if impact > 0:
                impact_details[component] = impact
        
        # Normalize to percentage
        impact_score = min(impact_score * 100, 100)
        
        return {
            'overall_impact_percent': impact_score,
            'component_impacts': impact_details,
            'severity': self._get_impact_severity(impact_score),
            'recommendations': self._get_impact_recommendations(impact_score, impact_details)
        }
    
    def _get_impact_severity(self, impact_percent: float) -> str:
        """Get impact severity level."""
        if impact_percent < 5:
            return "minimal"
        elif impact_percent < 15:
            return "low"
        elif impact_percent < 30:
            return "medium"
        elif impact_percent < 50:
            return "high"
        else:
            return "critical"
    
    def _get_impact_recommendations(self, impact_percent: float, 
                                   impact_details: Dict[str, float]) -> List[str]:
        """Get recommendations for reducing performance impact."""
        recommendations = []
        
        if impact_percent > 30:
            recommendations.append("Consider using a lighter debug profile for production")
        
        # Component-specific recommendations
        for component, impact in impact_details.items():
            if impact > 0.1:  # 10% impact
                if component == 'method_tracing':
                    recommendations.append("Reduce method tracing scope or disable parameter capture")
                elif component == 'ultra_debug':
                    recommendations.append("Use lower ultra-debug mode or disable for non-critical scenarios")
                elif component == 'performance_analysis':
                    recommendations.append("Reduce performance monitoring frequency")
        
        if not recommendations:
            recommendations.append("Performance impact is acceptable for this profile")
        
        return recommendations


class DebugConfigManager:
    """
    Manages debug configuration profiles and settings.
    
    Provides centralized configuration management for all debugging components
    with support for profiles, hot-reloading, and performance optimization.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize debug config manager."""
        self.logger = get_enhanced_logger("arena_bot.debugging.config_manager")
        
        # Configuration directory
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".arena_bot" / "debug_config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Profile management
        self.profiles: Dict[str, DebugProfile] = {}
        self.active_profile: Optional[DebugProfile] = None
        self.default_profile_name = "development"
        
        # Configuration state
        self.config_history: List[Dict[str, Any]] = []
        self.hot_reload_enabled = False
        self.file_watchers: Dict[str, float] = {}  # filename -> last_modified
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Built-in profiles
        self._create_builtin_profiles()
        
        # Load existing profiles
        self._load_profiles()
    
    def _create_builtin_profiles(self) -> None:
        """Create built-in debug profiles."""
        
        # Development profile - Full debugging enabled
        dev_profile = DebugProfile(
            profile_name="development",
            description="Full debugging for development environment",
            environment=DebugEnvironment.DEVELOPMENT,
            debug_level="detailed",
            method_tracing={
                'enabled': True,
                'capture_parameters': True,
                'capture_return_values': True,
                'capture_memory': True,
                'max_trace_depth': 15,
                'trace_threshold_ms': 0.1
            },
            ultra_debug={
                'enabled': True,
                'mode': UltraDebugMode.MONITORING.value,
                'monitoring_interval_seconds': 2.0,
                'anomaly_detection': True,
                'auto_escalation': True
            },
            dependency_tracking={
                'enabled': True,
                'track_imports': True,
                'track_calls': True,
                'auto_analyze': True,
                'analysis_interval_minutes': 5
            },
            performance_analysis={
                'enabled': True,
                'function_profiling': True,
                'memory_leak_detection': True,
                'bottleneck_detection': True,
                'monitoring_interval_seconds': 3.0
            },
            dashboard={
                'enabled': True,
                'auto_open_browser': True,
                'port': 8888,
                'update_interval_seconds': 1.0
            }
        )
        
        # Testing profile - Moderate debugging for testing
        test_profile = DebugProfile(
            profile_name="testing",
            description="Moderate debugging for testing environment",
            environment=DebugEnvironment.TESTING,
            debug_level="standard",
            method_tracing={
                'enabled': True,
                'capture_parameters': False,
                'capture_return_values': False,
                'capture_memory': False,
                'max_trace_depth': 10,
                'trace_threshold_ms': 1.0
            },
            ultra_debug={
                'enabled': True,
                'mode': UltraDebugMode.ANALYSIS.value,
                'monitoring_interval_seconds': 5.0,
                'anomaly_detection': True,
                'auto_escalation': False
            },
            performance_analysis={
                'enabled': True,
                'function_profiling': True,
                'memory_leak_detection': True,
                'bottleneck_detection': False,
                'monitoring_interval_seconds': 10.0
            },
            dashboard={
                'enabled': False,
                'auto_open_browser': False,
                'port': 8889
            }
        )
        
        # Production profile - Minimal debugging for production
        prod_profile = DebugProfile(
            profile_name="production",
            description="Minimal debugging for production environment",
            environment=DebugEnvironment.PRODUCTION,
            debug_level="minimal",
            method_tracing={
                'enabled': False,
                'capture_parameters': False,
                'capture_return_values': False,
                'capture_memory': False,
                'max_trace_depth': 5,
                'trace_threshold_ms': 10.0
            },
            state_monitoring={
                'enabled': True,
                'track_changes': False,
                'correlation_enabled': False,
                'max_history_size': 100
            },
            health_monitoring={
                'enabled': True,
                'check_interval_seconds': 60,
                'circuit_breaker_enabled': True,
                'auto_recovery': True,
                'health_threshold': 0.9
            },
            error_analysis={
                'enabled': True,
                'pattern_detection': True,
                'ml_analysis': False,
                'correlation_window_minutes': 30
            },
            ultra_debug={
                'enabled': False,
                'mode': UltraDebugMode.DISABLED.value,
                'anomaly_detection': False,
                'auto_escalation': False
            },
            dependency_tracking={'enabled': False},
            performance_analysis={'enabled': False},
            dashboard={'enabled': False},
            resource_limits={
                'max_memory_usage_mb': 500,
                'max_cpu_usage_percent': 20,
                'max_debug_overhead_percent': 5,
                'auto_disable_on_limit': True
            }
        )
        
        # Emergency profile - Crisis debugging
        emergency_profile = DebugProfile(
            profile_name="emergency",
            description="Maximum debugging for emergency situations",
            environment=DebugEnvironment.EMERGENCY,
            debug_level="ultra",
            method_tracing={
                'enabled': True,
                'capture_parameters': True,
                'capture_return_values': True,
                'capture_memory': True,
                'max_trace_depth': 20,
                'trace_threshold_ms': 0.01
            },
            ultra_debug={
                'enabled': True,
                'mode': UltraDebugMode.CRISIS.value,
                'monitoring_interval_seconds': 0.5,
                'anomaly_detection': True,
                'auto_escalation': True
            },
            dependency_tracking={
                'enabled': True,
                'track_imports': True,
                'track_calls': True,
                'auto_analyze': True,
                'analysis_interval_minutes': 1
            },
            performance_analysis={
                'enabled': True,
                'function_profiling': True,
                'memory_leak_detection': True,
                'bottleneck_detection': True,
                'monitoring_interval_seconds': 1.0
            },
            logging={
                'enhanced_logging': True,
                'debug_log_level': 'TRACE',
                'emergency_debug': True,
                'log_retention_hours': 72,
                'structured_logging': True
            },
            dashboard={
                'enabled': True,
                'auto_open_browser': True,
                'port': 8887,
                'update_interval_seconds': 0.5
            },
            resource_limits={
                'max_memory_usage_mb': 2000,
                'max_cpu_usage_percent': 80,
                'max_debug_overhead_percent': 50,
                'auto_disable_on_limit': False
            }
        )
        
        # Store built-in profiles
        self.profiles = {
            "development": dev_profile,
            "testing": test_profile,
            "production": prod_profile,
            "emergency": emergency_profile
        }
    
    def _load_profiles(self) -> None:
        """Load profiles from configuration files."""
        try:
            profiles_dir = self.config_dir / "profiles"
            if not profiles_dir.exists():
                return
            
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        profile_data = json.load(f)
                    
                    profile = DebugProfile.from_dict(profile_data)
                    self.profiles[profile.profile_name] = profile
                    
                    self.logger.debug(f"Loaded debug profile: {profile.profile_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load profile {profile_file}: {e}")
            
            # Load YAML profiles
            for profile_file in profiles_dir.glob("*.yaml"):
                try:
                    with open(profile_file, 'r') as f:
                        profile_data = yaml.safe_load(f)
                    
                    profile = DebugProfile.from_dict(profile_data)
                    self.profiles[profile.profile_name] = profile
                    
                    self.logger.debug(f"Loaded debug profile: {profile.profile_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load profile {profile_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}")
    
    def save_profile(self, profile: DebugProfile, format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Save a debug profile to file."""
        try:
            profiles_dir = self.config_dir / "profiles"
            profiles_dir.mkdir(exist_ok=True)
            
            # Update modification time
            profile.last_modified = time.time()
            
            if format == ConfigFormat.JSON:
                profile_file = profiles_dir / f"{profile.profile_name}.json"
                with open(profile_file, 'w') as f:
                    json.dump(profile.to_dict(), f, indent=2, default=str)
            
            elif format == ConfigFormat.YAML:
                profile_file = profiles_dir / f"{profile.profile_name}.yaml"
                with open(profile_file, 'w') as f:
                    yaml.dump(profile.to_dict(), f, default_flow_style=False)
            
            # Update in memory
            self.profiles[profile.profile_name] = profile
            
            self.logger.info(f"Saved debug profile: {profile.profile_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile.profile_name}: {e}")
            return False
    
    def load_profile(self, profile_name: str) -> Optional[DebugProfile]:
        """Load a specific debug profile."""
        return self.profiles.get(profile_name)
    
    def activate_profile(self, profile_name: str) -> bool:
        """Activate a debug profile."""
        with self.lock:
            profile = self.profiles.get(profile_name)
            if not profile:
                self.logger.error(f"Debug profile not found: {profile_name}")
                return False
            
            # Store previous profile in history
            if self.active_profile:
                self.config_history.append({
                    'timestamp': time.time(),
                    'action': 'profile_change',
                    'previous_profile': self.active_profile.profile_name,
                    'new_profile': profile_name
                })
            
            self.active_profile = profile
            
            # Apply profile settings to debugging components
            self._apply_profile_settings(profile)
            
            self.logger.critical(
                f"🔄 DEBUG_PROFILE_ACTIVATED: {profile_name} profile activated",
                extra={
                    'debug_profile_activation': {
                        'profile_name': profile_name,
                        'environment': profile.environment.value,
                        'debug_level': profile.debug_level,
                        'performance_impact': profile.estimate_performance_impact(),
                        'timestamp': time.time()
                    }
                }
            )
            
            return True
    
    def _apply_profile_settings(self, profile: DebugProfile) -> None:
        """Apply profile settings to debugging components."""
        try:
            # Import and configure debugging components
            from .integration import get_debugging_integrator
            
            integrator = get_debugging_integrator()
            
            # Initialize if needed
            if not integrator.is_initialized:
                success = integrator.initialize(profile.debug_level)
                if not success:
                    self.logger.error("Failed to initialize debugging integrator")
                    return
            
            # Enable/disable debugging based on profile
            if profile.debug_enabled:
                integrator.enable_debugging(profile.debug_level)
            else:
                integrator.disable_debugging()
            
            # Configure method tracing
            if integrator.method_tracer and profile.method_tracing['enabled']:
                tracer = integrator.method_tracer
                # Configure tracer settings based on profile
                if hasattr(tracer, 'capture_parameters'):
                    tracer.capture_parameters = profile.method_tracing.get('capture_parameters', False)
                if hasattr(tracer, 'capture_return_values'):
                    tracer.capture_return_values = profile.method_tracing.get('capture_return_values', False)
                if hasattr(tracer, 'capture_memory'):
                    tracer.capture_memory = profile.method_tracing.get('capture_memory', False)
            
            # Configure ultra-debug
            if profile.ultra_debug['enabled']:
                try:
                    from .ultra_debug import get_ultra_debug_manager, UltraDebugMode
                    
                    ultra_manager = get_ultra_debug_manager()
                    mode_str = profile.ultra_debug.get('mode', UltraDebugMode.DISABLED.value)
                    mode = UltraDebugMode(mode_str)
                    
                    ultra_manager.enable_ultra_debug(mode)
                    
                    # Configure monitoring interval
                    interval = profile.ultra_debug.get('monitoring_interval_seconds', 5.0)
                    ultra_manager.monitoring_interval_seconds = interval
                    
                except ImportError:
                    self.logger.warning("Ultra-debug not available")
            
            # Configure dependency tracking
            if profile.dependency_tracking['enabled']:
                try:
                    from .dependency_tracker import start_dependency_tracking
                    
                    start_dependency_tracking(
                        track_imports=profile.dependency_tracking.get('track_imports', True),
                        track_calls=profile.dependency_tracking.get('track_calls', True),
                        auto_analyze=profile.dependency_tracking.get('auto_analyze', True)
                    )
                    
                except ImportError:
                    self.logger.warning("Dependency tracking not available")
            
            # Configure performance analysis
            if profile.performance_analysis['enabled']:
                try:
                    from .performance_analyzer import start_performance_monitoring
                    
                    start_performance_monitoring(
                        enable_profiling=profile.performance_analysis.get('function_profiling', True),
                        enable_leak_detection=profile.performance_analysis.get('memory_leak_detection', True),
                        enable_bottleneck_detection=profile.performance_analysis.get('bottleneck_detection', True)
                    )
                    
                except ImportError:
                    self.logger.warning("Performance analysis not available")
            
            # Configure dashboard
            if profile.dashboard['enabled']:
                try:
                    from .debug_dashboard import start_debug_dashboard, DashboardConfig
                    
                    dashboard_config = DashboardConfig(
                        port=profile.dashboard.get('port', 8888),
                        auto_open_browser=profile.dashboard.get('auto_open_browser', True)
                    )
                    
                    start_debug_dashboard(dashboard_config)
                    
                except ImportError:
                    self.logger.warning("Debug dashboard not available")
            
        except Exception as e:
            self.logger.error(f"Failed to apply profile settings: {e}")
    
    def create_custom_profile(self, base_profile_name: str, custom_name: str,
                             overrides: Dict[str, Any]) -> Optional[DebugProfile]:
        """Create a custom profile based on an existing profile."""
        base_profile = self.profiles.get(base_profile_name)
        if not base_profile:
            self.logger.error(f"Base profile not found: {base_profile_name}")
            return None
        
        # Create a copy of the base profile
        custom_profile_data = base_profile.to_dict()
        custom_profile_data['profile_name'] = custom_name
        custom_profile_data['description'] = f"Custom profile based on {base_profile_name}"
        custom_profile_data['created_time'] = time.time()
        custom_profile_data['last_modified'] = time.time()
        
        # Apply overrides
        def deep_update(base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
            """Deep update dictionary."""
            for key, value in updates.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(custom_profile_data, overrides)
        
        # Create custom profile
        custom_profile = DebugProfile.from_dict(custom_profile_data)
        
        # Save and store
        self.profiles[custom_name] = custom_profile
        self.save_profile(custom_profile)
        
        self.logger.info(f"Created custom debug profile: {custom_name}")
        return custom_profile
    
    def get_profile_recommendations(self, environment: DebugEnvironment,
                                   performance_budget: float = 10.0) -> List[Dict[str, Any]]:
        """Get profile recommendations based on environment and performance budget."""
        recommendations = []
        
        for profile_name, profile in self.profiles.items():
            if profile.environment != environment:
                continue
            
            impact = profile.estimate_performance_impact()
            impact_percent = impact['overall_impact_percent']
            
            if impact_percent <= performance_budget:
                recommendations.append({
                    'profile_name': profile_name,
                    'description': profile.description,
                    'performance_impact': impact,
                    'suitability_score': min(100 - impact_percent + 50, 100),
                    'recommendation_reason': self._get_recommendation_reason(profile, impact_percent, performance_budget)
                })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return recommendations
    
    def _get_recommendation_reason(self, profile: DebugProfile, 
                                  impact_percent: float, budget: float) -> str:
        """Get recommendation reason for a profile."""
        if impact_percent <= budget * 0.5:
            return f"Low performance impact ({impact_percent:.1f}%) - safe for {profile.environment.value}"
        elif impact_percent <= budget:
            return f"Moderate performance impact ({impact_percent:.1f}%) - suitable for {profile.environment.value}"
        else:
            return f"High performance impact ({impact_percent:.1f}%) - use with caution"
    
    def optimize_active_profile(self) -> Optional[DebugProfile]:
        """Optimize the active profile based on current system performance."""
        if not self.active_profile:
            return None
        
        try:
            # Get current system metrics
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_percent = psutil.virtual_memory().percent
            
            # Create optimized profile
            optimized_profile = deepcopy(self.active_profile)
            optimized_profile.profile_name = f"{self.active_profile.profile_name}_optimized"
            optimized_profile.description = f"Auto-optimized version of {self.active_profile.profile_name}"
            optimized_profile.last_modified = time.time()
            
            # Apply optimizations based on system load
            if cpu_percent > 70:
                # High CPU - reduce CPU-intensive debugging
                optimized_profile.method_tracing['capture_parameters'] = False
                optimized_profile.method_tracing['capture_return_values'] = False
                optimized_profile.method_tracing['trace_threshold_ms'] *= 2
                
                if optimized_profile.ultra_debug['enabled']:
                    interval = optimized_profile.ultra_debug['monitoring_interval_seconds']
                    optimized_profile.ultra_debug['monitoring_interval_seconds'] = max(interval * 2, 10.0)
                
                optimized_profile.performance_analysis['monitoring_interval_seconds'] *= 2
            
            if memory_percent > 80:
                # High memory - reduce memory-intensive debugging
                optimized_profile.method_tracing['capture_memory'] = False
                optimized_profile.state_monitoring['max_history_size'] = min(
                    optimized_profile.state_monitoring['max_history_size'], 500
                )
                optimized_profile.pipeline_tracing['max_pipeline_history'] = min(
                    optimized_profile.pipeline_tracing['max_pipeline_history'], 250
                )
            
            # Verify optimization improved performance impact
            original_impact = self.active_profile.estimate_performance_impact()
            optimized_impact = optimized_profile.estimate_performance_impact()
            
            if optimized_impact['overall_impact_percent'] < original_impact['overall_impact_percent']:
                self.profiles[optimized_profile.profile_name] = optimized_profile
                self.save_profile(optimized_profile)
                
                self.logger.info(
                    f"🎯 PROFILE_OPTIMIZED: Created optimized profile with {original_impact['overall_impact_percent'] - optimized_impact['overall_impact_percent']:.1f}% less impact"
                )
                
                return optimized_profile
            
        except Exception as e:
            self.logger.error(f"Failed to optimize profile: {e}")
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive configuration manager status."""
        active_profile_info = None
        if self.active_profile:
            active_profile_info = {
                'name': self.active_profile.profile_name,
                'environment': self.active_profile.environment.value,
                'debug_level': self.active_profile.debug_level,
                'performance_impact': self.active_profile.estimate_performance_impact()
            }
        
        return {
            'active_profile': active_profile_info,
            'available_profiles': list(self.profiles.keys()),
            'config_directory': str(self.config_dir),
            'hot_reload_enabled': self.hot_reload_enabled,
            'config_history_size': len(self.config_history),
            'profiles_loaded': len(self.profiles)
        }
    
    def export_profile(self, profile_name: str, export_path: str,
                      format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Export a profile to a specific location."""
        profile = self.profiles.get(profile_name)
        if not profile:
            return False
        
        try:
            export_file = Path(export_path)
            
            if format == ConfigFormat.JSON:
                with open(export_file, 'w') as f:
                    json.dump(profile.to_dict(), f, indent=2, default=str)
            
            elif format == ConfigFormat.YAML:
                with open(export_file, 'w') as f:
                    yaml.dump(profile.to_dict(), f, default_flow_style=False)
            
            self.logger.info(f"Exported profile {profile_name} to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export profile: {e}")
            return False
    
    def import_profile(self, import_path: str) -> Optional[str]:
        """Import a profile from a file."""
        try:
            import_file = Path(import_path)
            
            if import_file.suffix == '.json':
                with open(import_file, 'r') as f:
                    profile_data = json.load(f)
            elif import_file.suffix in ['.yaml', '.yml']:
                with open(import_file, 'r') as f:
                    profile_data = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported file format: {import_file.suffix}")
                return None
            
            profile = DebugProfile.from_dict(profile_data)
            
            # Ensure unique name
            original_name = profile.profile_name
            counter = 1
            while profile.profile_name in self.profiles:
                profile.profile_name = f"{original_name}_{counter}"
                counter += 1
            
            self.profiles[profile.profile_name] = profile
            self.save_profile(profile)
            
            self.logger.info(f"Imported profile: {profile.profile_name}")
            return profile.profile_name
            
        except Exception as e:
            self.logger.error(f"Failed to import profile: {e}")
            return None


# Global config manager instance
_global_config_manager: Optional[DebugConfigManager] = None
_config_lock = threading.Lock()


def get_debug_config_manager() -> DebugConfigManager:
    """Get global debug config manager instance."""
    global _global_config_manager
    
    if _global_config_manager is None:
        with _config_lock:
            if _global_config_manager is None:
                _global_config_manager = DebugConfigManager()
    
    return _global_config_manager


def activate_debug_profile(profile_name: str) -> bool:
    """
    Activate a debug profile.
    
    Convenience function to activate a debug profile and apply all settings.
    
    Args:
        profile_name: Name of the profile to activate
        
    Returns:
        True if profile was activated successfully
    """
    config_manager = get_debug_config_manager()
    return config_manager.activate_profile(profile_name)


def get_active_profile() -> Optional[DebugProfile]:
    """Get the currently active debug profile."""
    config_manager = get_debug_config_manager()
    return config_manager.active_profile


def create_custom_profile(base_profile: str, custom_name: str,
                         overrides: Dict[str, Any]) -> Optional[DebugProfile]:
    """Create a custom debug profile."""
    config_manager = get_debug_config_manager()
    return config_manager.create_custom_profile(base_profile, custom_name, overrides)


def get_profile_recommendations(environment: DebugEnvironment,
                               performance_budget: float = 10.0) -> List[Dict[str, Any]]:
    """Get debug profile recommendations."""
    config_manager = get_debug_config_manager()
    return config_manager.get_profile_recommendations(environment, performance_budget)


def optimize_current_profile() -> Optional[DebugProfile]:
    """Optimize the currently active profile."""
    config_manager = get_debug_config_manager()
    return config_manager.optimize_active_profile()


def get_debug_config_status() -> Dict[str, Any]:
    """Get debug configuration status."""
    config_manager = get_debug_config_manager()
    return config_manager.get_status()