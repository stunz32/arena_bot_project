"""
Resource Monitor Integration for S-Tier Logging System.

This module provides integration between the S-tier logging system and the
existing AI v2 monitoring infrastructure, enabling unified resource monitoring,
health checks, and emergency protocols across both systems.

Features:
- Seamless integration with existing GlobalResourceManager
- Shared resource thresholds and emergency protocols
- Unified health reporting and metrics collection
- Coordinated resource recovery procedures
- Performance profiler integration
- Emergency protocol coordination
"""

import asyncio
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import existing monitoring infrastructure
try:
    from ..ai_v2.monitoring import (
        GlobalResourceManager,
        PerformanceMonitor,
        ResourceType,
        ResourceState,
        ResourceThresholds,
        get_resource_manager,
        get_performance_monitor,
        LockFreeRingBuffer
    )
    EXISTING_MONITORING_AVAILABLE = True
except ImportError:
    # Fallback if existing monitoring is not available
    EXISTING_MONITORING_AVAILABLE = False
    GlobalResourceManager = None
    PerformanceMonitor = None

# Import S-tier logging diagnostics
from .diagnostics import (
    HealthChecker,
    PerformanceProfiler,
    EmergencyProtocol,
    MetricType as STierMetricType,
    PerformanceLevel
)


@dataclass
class ResourceMonitorConfig:
    """Configuration for resource monitor integration."""
    
    # Integration settings
    enable_existing_monitoring: bool = True
    enable_shared_emergency_protocols: bool = True
    enable_unified_health_reporting: bool = True
    
    # Resource thresholds (inherits from existing if available)
    memory_warning_mb: float = 400
    memory_critical_mb: float = 450
    memory_emergency_mb: float = 480
    
    cpu_warning_percent: float = 15
    cpu_critical_percent: float = 20
    cpu_emergency_percent: float = 23
    
    # Integration intervals
    sync_interval_seconds: float = 5.0
    health_check_interval_seconds: float = 30.0
    
    # Emergency coordination
    coordinate_emergency_responses: bool = True
    shared_recovery_protocols: bool = True


class LoggingResourceAdapter:
    """
    Adapter to bridge S-tier logging metrics with existing monitoring.
    
    Provides a translation layer between the S-tier logging system's
    performance profiler and the existing AI v2 monitoring infrastructure.
    """
    
    def __init__(self, 
                 logger_manager: 'LoggerManager',
                 existing_monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize logging resource adapter.
        
        Args:
            logger_manager: S-tier logger manager instance
            existing_monitor: Existing performance monitor (optional)
        """
        self.logger_manager = logger_manager
        self.existing_monitor = existing_monitor
        self.logger = logging.getLogger(__name__)
        
        # Metric translation mappings
        self.metric_mappings = {
            'logs_per_second': 'logging_throughput_lps',
            'log_latency_ms': 'logging_latency_ms',
            'error_rate': 'logging_error_rate',
            'queue_depth': 'logging_queue_depth',
            'worker_utilization_percent': 'logging_worker_utilization'
        }
        
        # Last sync timestamp
        self._last_sync = 0.0
        self._sync_lock = threading.RLock()
    
    def sync_metrics(self) -> None:
        """Synchronize S-tier logging metrics with existing monitoring."""
        if not self.existing_monitor or not EXISTING_MONITORING_AVAILABLE:
            return
        
        with self._sync_lock:
            current_time = time.time()
            
            try:
                # Get performance stats from logger manager
                stats = self.logger_manager.get_performance_stats()
                
                # Map and record metrics in existing monitoring system
                for s_tier_metric, existing_metric in self.metric_mappings.items():
                    if s_tier_metric in stats:
                        value = stats[s_tier_metric]
                        
                        # Determine metric type based on name
                        if 'rate' in s_tier_metric or 'percent' in s_tier_metric:
                            from ..ai_v2.monitoring import MetricType
                            metric_type = MetricType.GAUGE
                        elif 'count' in s_tier_metric or 'total' in s_tier_metric:
                            metric_type = MetricType.COUNTER
                        else:
                            metric_type = MetricType.GAUGE
                        
                        self.existing_monitor.record_metric(
                            existing_metric,
                            value,
                            metric_type,
                            tags={'source': 's_tier_logging'}
                        )
                
                # Record queue-specific metrics
                if 'queue_stats' in stats:
                    queue_stats = stats['queue_stats']
                    for metric, value in queue_stats.items():
                        self.existing_monitor.record_metric(
                            f'logging_queue_{metric}',
                            value,
                            MetricType.GAUGE,
                            tags={'source': 's_tier_logging'}
                        )
                
                # Record worker pool metrics
                if 'worker_pool_stats' in stats:
                    worker_stats = stats['worker_pool_stats']
                    for metric, value in worker_stats.items():
                        self.existing_monitor.record_metric(
                            f'logging_workers_{metric}',
                            value,
                            MetricType.GAUGE,
                            tags={'source': 's_tier_logging'}
                        )
                
                self._last_sync = current_time
                
            except Exception as e:
                self.logger.error(f"Failed to sync metrics with existing monitoring: {e}")
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            'last_sync_timestamp': self._last_sync,
            'last_sync_age_seconds': time.time() - self._last_sync,
            'existing_monitor_available': self.existing_monitor is not None,
            'metrics_mapped': len(self.metric_mappings)
        }


class UnifiedResourceMonitor:
    """
    Unified resource monitor that coordinates between S-tier logging
    and existing AI v2 monitoring systems.
    
    Provides centralized resource monitoring, health checks, and
    emergency response coordination across both systems.
    """
    
    def __init__(self, 
                 logger_manager: 'LoggerManager',
                 config: Optional[ResourceMonitorConfig] = None):
        """
        Initialize unified resource monitor.
        
        Args:
            logger_manager: S-tier logger manager instance
            config: Resource monitor configuration
        """
        self.logger_manager = logger_manager
        self.config = config or ResourceMonitorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize existing monitoring integration
        self.existing_resource_manager: Optional[GlobalResourceManager] = None
        self.existing_performance_monitor: Optional[PerformanceMonitor] = None
        self.resource_adapter: Optional[LoggingResourceAdapter] = None
        
        if EXISTING_MONITORING_AVAILABLE and self.config.enable_existing_monitoring:
            self._initialize_existing_monitoring()
        
        # Initialize S-tier diagnostics
        self.health_checker = HealthChecker(logger_manager)
        self.performance_profiler = PerformanceProfiler(logger_manager)
        self.emergency_protocol = EmergencyProtocol(logger_manager)
        
        # Unified state tracking
        self._unified_resource_state: Dict[str, Any] = {}
        self._emergency_coordination_active = False
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.start_time = time.time()
        self.total_resource_alerts = 0
        self.total_emergency_responses = 0
        self.total_recoveries = 0
        
        self.logger.info("Unified resource monitor initialized")
    
    def _initialize_existing_monitoring(self) -> None:
        """Initialize integration with existing monitoring systems."""
        try:
            # Get existing monitoring instances
            self.existing_resource_manager = get_resource_manager()
            self.existing_performance_monitor = get_performance_monitor()
            
            # Initialize resource adapter
            self.resource_adapter = LoggingResourceAdapter(
                self.logger_manager,
                self.existing_performance_monitor
            )
            
            # Register S-tier logging with existing resource manager
            if self.existing_resource_manager:
                self.existing_resource_manager.register_component(
                    self.logger_manager,
                    cleanup_handler=self._emergency_cleanup_handler
                )
                
                # Start existing monitoring if not already running
                self.existing_resource_manager.start_monitoring()
            
            self.logger.info("Successfully integrated with existing monitoring systems")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize existing monitoring integration: {e}")
            self.existing_resource_manager = None
            self.existing_performance_monitor = None
            self.resource_adapter = None
    
    def _emergency_cleanup_handler(self) -> None:
        """Emergency cleanup handler for existing resource manager."""
        try:
            self.logger.warning("Emergency cleanup requested by existing resource manager")
            
            # Trigger S-tier logging emergency protocols
            if self.emergency_protocol:
                asyncio.create_task(self._coordinate_emergency_response("resource_exhaustion"))
            
            # Clear S-tier logging caches and buffers
            if hasattr(self.logger_manager, 'queue') and self.logger_manager.queue:
                # Emergency queue drain
                pass  # Queue emergency drain would be implemented here
            
            # Reduce S-tier logging resource usage
            self._reduce_logging_resource_usage()
            
        except Exception as e:
            self.logger.error(f"Emergency cleanup handler failed: {e}")
    
    def _reduce_logging_resource_usage(self) -> None:
        """Reduce S-tier logging resource usage during emergencies."""
        try:
            # Reduce log levels temporarily
            for logger_name in self.logger_manager.list_loggers():
                logger = self.logger_manager.get_logger(logger_name)
                if hasattr(logger, 'set_level'):
                    # Temporarily elevate to WARNING level
                    logger.set_level('WARNING')
            
            # Clear performance profiler caches
            if self.performance_profiler:
                # Clear metric caches
                pass  # Profiler cache clearing would be implemented here
            
            self.logger.info("Reduced S-tier logging resource usage for emergency")
            
        except Exception as e:
            self.logger.error(f"Failed to reduce logging resource usage: {e}")
    
    async def start_monitoring(self) -> None:
        """Start unified resource monitoring."""
        if self._monitoring_task is not None:
            return
        
        # Start S-tier diagnostics
        await self.health_checker.start_monitoring()
        await self.performance_profiler.start_collection()
        await self.emergency_protocol.start_monitoring()
        
        # Start unified monitoring tasks
        self._monitoring_task = asyncio.create_task(self._unified_monitoring_loop())
        
        if self.resource_adapter:
            self._sync_task = asyncio.create_task(self._metric_sync_loop())
        
        self.logger.info("Unified resource monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop unified resource monitoring."""
        if self._monitoring_task is not None:
            self._shutdown_event.set()
            
            # Wait for tasks to complete
            await asyncio.gather(
                self._monitoring_task,
                self._sync_task or asyncio.sleep(0),
                return_exceptions=True
            )
            
            self._monitoring_task = None
            self._sync_task = None
        
        # Stop S-tier diagnostics
        await self.health_checker.stop_monitoring()
        await self.performance_profiler.stop_collection()
        await self.emergency_protocol.stop_monitoring()
        
        # Stop existing monitoring if we started it
        if self.existing_resource_manager:
            self.existing_resource_manager.stop_monitoring()
        
        self.logger.info("Unified resource monitoring stopped")
    
    async def _unified_monitoring_loop(self) -> None:
        """Main unified monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect unified health status
                await self._collect_unified_health()
                
                # Check for emergency conditions
                await self._check_unified_emergency_conditions()
                
                # Update unified resource state
                self._update_unified_resource_state()
                
                # Wait for next monitoring cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.health_check_interval_seconds
                )
                
            except asyncio.TimeoutError:
                # Normal timeout, continue monitoring
                continue
            except Exception as e:
                self.logger.error(f"Unified monitoring loop error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _metric_sync_loop(self) -> None:
        """Metric synchronization loop."""
        while not self._shutdown_event.is_set():
            try:
                # Sync metrics with existing monitoring
                if self.resource_adapter:
                    self.resource_adapter.sync_metrics()
                
                # Wait for next sync cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.sync_interval_seconds
                )
                
            except asyncio.TimeoutError:
                # Normal timeout, continue syncing
                continue
            except Exception as e:
                self.logger.error(f"Metric sync loop error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _collect_unified_health(self) -> None:
        """Collect unified health status from all monitoring systems."""
        try:
            # Get S-tier health status
            s_tier_health = await self.health_checker.run_all_checks()
            
            # Get existing monitoring health status
            existing_health = None
            if self.existing_resource_manager:
                existing_health = self.existing_resource_manager.get_health_status()
            
            # Combine health information
            self._unified_resource_state = {
                'timestamp': time.time(),
                's_tier_health': s_tier_health.to_dict(),
                'existing_health': existing_health,
                'overall_status': self._calculate_overall_health_status(s_tier_health, existing_health)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect unified health: {e}")
    
    def _calculate_overall_health_status(self, 
                                       s_tier_health: 'SystemHealthReport',
                                       existing_health: Optional[Dict[str, Any]]) -> str:
        """Calculate overall health status from all monitoring sources."""
        # S-tier health status
        s_tier_status = s_tier_health.overall_status.value
        
        # Existing monitoring health status
        existing_status = "unknown"
        if existing_health:
            existing_status = existing_health.get('overall_health', 'unknown')
        
        # Determine combined status (worst case wins)
        status_priority = {
            'critical': 4,
            'emergency': 4,
            'unhealthy': 3,
            'degraded': 2,
            'warning': 2,
            'healthy': 1,
            'unknown': 0
        }
        
        s_tier_priority = status_priority.get(s_tier_status, 0)
        existing_priority = status_priority.get(existing_status, 0)
        
        max_priority = max(s_tier_priority, existing_priority)
        
        # Map back to status
        for status, priority in status_priority.items():
            if priority == max_priority and max_priority > 0:
                return status
        
        return 'unknown'
    
    async def _check_unified_emergency_conditions(self) -> None:
        """Check for emergency conditions across all monitoring systems."""
        try:
            # Check if existing monitoring has emergency conditions
            existing_emergency = False
            if self.existing_resource_manager:
                existing_health = self.existing_resource_manager.get_health_status()
                existing_emergency = existing_health.get('overall_health') in ['emergency', 'critical']
            
            # Check S-tier emergency conditions
            s_tier_emergency = len(self.emergency_protocol.active_emergencies) > 0
            
            # Coordinate emergency response if needed
            if (existing_emergency or s_tier_emergency) and not self._emergency_coordination_active:
                await self._coordinate_emergency_response("unified_emergency")
            elif not existing_emergency and not s_tier_emergency and self._emergency_coordination_active:
                await self._coordinate_emergency_recovery()
            
        except Exception as e:
            self.logger.error(f"Failed to check unified emergency conditions: {e}")
    
    async def _coordinate_emergency_response(self, emergency_type: str) -> None:
        """Coordinate emergency response across all monitoring systems."""
        if not self.config.coordinate_emergency_responses:
            return
        
        try:
            self._emergency_coordination_active = True
            self.total_emergency_responses += 1
            
            self.logger.critical(f"Coordinating unified emergency response: {emergency_type}")
            
            # Notify existing monitoring of S-tier emergency
            if self.existing_resource_manager and emergency_type.startswith("s_tier"):
                # This would trigger existing monitoring emergency protocols
                pass
            
            # Apply coordinated resource reduction
            await self._apply_coordinated_resource_reduction()
            
            # Log emergency coordination
            self.logger.warning("Unified emergency response coordination active")
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate emergency response: {e}")
    
    async def _coordinate_emergency_recovery(self) -> None:
        """Coordinate emergency recovery across all monitoring systems."""
        try:
            self._emergency_coordination_active = False
            self.total_recoveries += 1
            
            self.logger.info("Coordinating unified emergency recovery")
            
            # Restore normal resource usage patterns
            await self._restore_normal_resource_usage()
            
            self.logger.info("Unified emergency recovery coordination complete")
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate emergency recovery: {e}")
    
    async def _apply_coordinated_resource_reduction(self) -> None:
        """Apply coordinated resource reduction across all systems."""
        try:
            # Reduce S-tier logging resource usage
            self._reduce_logging_resource_usage()
            
            # Reduce diagnostic collection rates
            if self.performance_profiler:
                # Reduce profiler collection frequency
                pass
            
            # Pause non-critical health checks
            if self.health_checker:
                # Disable non-critical health checks
                pass
            
            self.logger.info("Applied coordinated resource reduction")
            
        except Exception as e:
            self.logger.error(f"Failed to apply coordinated resource reduction: {e}")
    
    async def _restore_normal_resource_usage(self) -> None:
        """Restore normal resource usage patterns."""
        try:
            # Restore normal log levels
            for logger_name in self.logger_manager.list_loggers():
                logger = self.logger_manager.get_logger(logger_name)
                if hasattr(logger, 'set_level'):
                    # Restore to INFO level (or configured level)
                    logger.set_level('INFO')
            
            # Resume normal diagnostic collection
            if self.performance_profiler:
                # Resume normal profiler collection
                pass
            
            # Resume all health checks
            if self.health_checker:
                # Re-enable all health checks
                pass
            
            self.logger.info("Restored normal resource usage patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to restore normal resource usage: {e}")
    
    def _update_unified_resource_state(self) -> None:
        """Update unified resource state tracking."""
        try:
            current_time = time.time()
            
            # Update resource state with current information
            self._unified_resource_state.update({
                'last_update': current_time,
                'uptime_seconds': current_time - self.start_time,
                'emergency_coordination_active': self._emergency_coordination_active,
                'total_alerts': self.total_resource_alerts,
                'total_emergency_responses': self.total_emergency_responses,
                'total_recoveries': self.total_recoveries
            })
            
            # Add performance profiler stats
            if self.performance_profiler:
                profiler_analysis = self.performance_profiler.analyze_performance()
                self._unified_resource_state['performance_analysis'] = profiler_analysis
            
            # Add sync statistics
            if self.resource_adapter:
                sync_stats = self.resource_adapter.get_sync_stats()
                self._unified_resource_state['sync_stats'] = sync_stats
            
        except Exception as e:
            self.logger.error(f"Failed to update unified resource state: {e}")
    
    def get_unified_status(self) -> Dict[str, Any]:
        """Get comprehensive unified monitoring status."""
        return {
            'timestamp': time.time(),
            'monitoring_active': self._monitoring_task is not None,
            'existing_monitoring_integrated': EXISTING_MONITORING_AVAILABLE and self.config.enable_existing_monitoring,
            'emergency_coordination_active': self._emergency_coordination_active,
            'resource_state': self._unified_resource_state,
            'diagnostics_status': {
                'health_checker_active': self.health_checker is not None,
                'performance_profiler_active': self.performance_profiler is not None,
                'emergency_protocol_active': self.emergency_protocol is not None
            },
            'statistics': {
                'uptime_seconds': time.time() - self.start_time,
                'total_resource_alerts': self.total_resource_alerts,
                'total_emergency_responses': self.total_emergency_responses,
                'total_recoveries': self.total_recoveries
            }
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for external health checks."""
        unified_status = self.get_unified_status()
        
        # Determine overall health
        resource_state = unified_status.get('resource_state', {})
        overall_status = resource_state.get('overall_status', 'unknown')
        
        # Map to simple health status
        if overall_status in ['critical', 'emergency']:
            health = 'unhealthy'
        elif overall_status in ['degraded', 'warning']:
            health = 'degraded'
        elif overall_status == 'healthy':
            health = 'healthy'
        else:
            health = 'unknown'
        
        return {
            'status': health,
            'monitoring_active': unified_status['monitoring_active'],
            'emergency_active': unified_status['emergency_coordination_active'],
            'uptime_seconds': unified_status['statistics']['uptime_seconds'],
            'overall_status': overall_status
        }
    
    async def shutdown(self) -> None:
        """Shutdown unified resource monitor."""
        await self.stop_monitoring()
        
        # Shutdown S-tier diagnostics
        if self.health_checker:
            await self.health_checker.shutdown()
        
        if self.performance_profiler:
            await self.performance_profiler.shutdown()
        
        if self.emergency_protocol:
            await self.emergency_protocol.shutdown()
        
        self.logger.info("Unified resource monitor shutdown complete")


# Global instance
_global_unified_monitor: Optional[UnifiedResourceMonitor] = None


def initialize_unified_monitoring(logger_manager: 'LoggerManager',
                                config: Optional[ResourceMonitorConfig] = None) -> UnifiedResourceMonitor:
    """
    Initialize unified resource monitoring.
    
    Args:
        logger_manager: S-tier logger manager instance
        config: Resource monitor configuration
        
    Returns:
        Initialized UnifiedResourceMonitor instance
    """
    global _global_unified_monitor
    
    _global_unified_monitor = UnifiedResourceMonitor(logger_manager, config)
    return _global_unified_monitor


async def start_unified_monitoring() -> None:
    """Start unified resource monitoring."""
    if _global_unified_monitor:
        await _global_unified_monitor.start_monitoring()


async def stop_unified_monitoring() -> None:
    """Stop unified resource monitoring."""
    if _global_unified_monitor:
        await _global_unified_monitor.stop_monitoring()


def get_unified_monitor() -> Optional[UnifiedResourceMonitor]:
    """Get global unified resource monitor instance."""
    return _global_unified_monitor


def get_unified_health_status() -> Dict[str, Any]:
    """Get unified health status."""
    if _global_unified_monitor:
        return _global_unified_monitor.get_health_summary()
    else:
        return {
            'status': 'not_initialized',
            'monitoring_active': False,
            'emergency_active': False,
            'uptime_seconds': 0,
            'overall_status': 'unknown'
        }


# Module exports
__all__ = [
    'ResourceMonitorConfig',
    'LoggingResourceAdapter',
    'UnifiedResourceMonitor',
    'initialize_unified_monitoring',
    'start_unified_monitoring',
    'stop_unified_monitoring',
    'get_unified_monitor',
    'get_unified_health_status'
]