"""
Resource Manager - Complete Resource Management with Memory Optimization

Provides comprehensive resource management for dual data streams (hero + card data)
with memory optimization, connection pooling, and performance monitoring.
"""

import logging
import psutil
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import gc
import weakref
from collections import deque
import queue
import concurrent.futures


class ResourceType(Enum):
    """Types of resources being managed."""
    MEMORY = "memory"
    NETWORK = "network"
    CPU = "cpu"
    DISK = "disk"
    CACHE = "cache"
    API_CALLS = "api_calls"


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    resource_type: ResourceType
    current_usage: float
    max_usage: float
    percentage_used: float
    last_updated: datetime
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0


@dataclass
class MemoryProfile:
    """Memory usage profile for optimization."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    process_memory_mb: float
    cache_memory_mb: float
    hero_data_memory_mb: float
    card_data_memory_mb: float
    optimization_potential_mb: float


class ResourceManager:
    """
    Comprehensive resource management system.
    
    Manages memory, network connections, API rate limits, and performance
    optimization for dual hero/card data streams.
    """
    
    def __init__(self, max_memory_mb: int = 512, max_connections: int = 10):
        """Initialize resource manager."""
        self.logger = logging.getLogger(__name__)
        
        # Resource limits
        self.max_memory_mb = max_memory_mb
        self.max_connections = max_connections
        self.max_api_calls_per_minute = 60
        
        # Resource tracking
        self.resource_usage = {}
        self.memory_profile = None
        
        # Connection management
        self.connection_pool = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.connection_lock = threading.Lock()
        
        # API rate limiting
        self.api_call_history = deque()
        self.api_call_lock = threading.Lock()
        
        # Memory optimization
        self.memory_pressure_threshold = 0.85
        self.gc_threshold = 0.90
        self.cleanup_handlers = []
        self.weak_references = []
        
        # Background monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
        
        # Thread pool for async operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.performance_metrics = {
            'memory_optimizations': 0,
            'gc_collections': 0,
            'connection_reuses': 0,
            'cache_evictions': 0,
            'api_rate_limits': 0
        }
        
        # Initialize monitoring
        self._start_monitoring()
        
        self.logger.info(f"ResourceManager initialized with {max_memory_mb}MB memory limit")
    
    def acquire_connection(self, timeout: float = 30.0) -> Optional[Any]:
        """Acquire a connection from the pool."""
        try:
            with self.connection_lock:
                if self.active_connections < self.max_connections:
                    # Create new connection
                    connection = self._create_connection()
                    if connection:
                        self.active_connections += 1
                        return connection
                
                # Try to get from pool
                try:
                    connection = self.connection_pool.get(timeout=timeout)
                    self.performance_metrics['connection_reuses'] += 1
                    return connection
                except queue.Empty:
                    self.logger.warning("Connection pool timeout")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error acquiring connection: {e}")
            return None
    
    def release_connection(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        try:
            if connection and self._validate_connection(connection):
                # Return to pool if not full
                try:
                    self.connection_pool.put_nowait(connection)
                except queue.Full:
                    # Pool is full, close connection
                    self._close_connection(connection)
                    with self.connection_lock:
                        self.active_connections -= 1
            else:
                # Invalid connection, create new one
                with self.connection_lock:
                    self.active_connections -= 1
                    
        except Exception as e:
            self.logger.error(f"Error releasing connection: {e}")
    
    def check_api_rate_limit(self) -> bool:
        """Check if API call is within rate limits."""
        with self.api_call_lock:
            current_time = datetime.now()
            
            # Remove calls older than 1 minute
            while self.api_call_history and (current_time - self.api_call_history[0]).seconds >= 60:
                self.api_call_history.popleft()
            
            # Check if we can make another call
            if len(self.api_call_history) >= self.max_api_calls_per_minute:
                self.performance_metrics['api_rate_limits'] += 1
                return False
            
            # Record this call
            self.api_call_history.append(current_time)
            return True
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """Optimize memory usage across all components."""
        try:
            optimization_results = {
                'memory_before_mb': self.get_memory_usage(),
                'actions_taken': [],
                'memory_freed_mb': 0
            }
            
            # Check if optimization is needed
            current_usage = self.get_resource_usage(ResourceType.MEMORY)
            if not force and current_usage.percentage_used < self.memory_pressure_threshold * 100:
                optimization_results['actions_taken'].append("No optimization needed")
                return optimization_results
            
            # 1. Garbage collection
            if current_usage.percentage_used > self.gc_threshold * 100:
                gc.collect()
                self.performance_metrics['gc_collections'] += 1
                optimization_results['actions_taken'].append("Forced garbage collection")
            
            # 2. Clean up weak references
            cleaned_refs = self._cleanup_weak_references()
            if cleaned_refs > 0:
                optimization_results['actions_taken'].append(f"Cleaned {cleaned_refs} weak references")
            
            # 3. Call registered cleanup handlers
            for handler in self.cleanup_handlers:
                try:
                    handler()
                    optimization_results['actions_taken'].append(f"Called cleanup handler: {handler.__name__}")
                except Exception as e:
                    self.logger.warning(f"Cleanup handler failed: {e}")
            
            # 4. Optimize cache if available
            try:
                from .intelligent_cache_manager import get_cache_manager
                cache_manager = get_cache_manager()
                cache_results = cache_manager.optimize_cache()
                if 'memory_freed_mb' in cache_results:
                    optimization_results['actions_taken'].append(f"Cache optimization freed {cache_results['memory_freed_mb']:.1f}MB")
            except Exception as e:
                self.logger.debug(f"Cache optimization not available: {e}")
            
            # 5. Force Python garbage collection again
            collected = gc.collect()
            if collected > 0:
                optimization_results['actions_taken'].append(f"Collected {collected} objects")
            
            # Calculate memory freed
            memory_after = self.get_memory_usage()
            optimization_results['memory_after_mb'] = memory_after
            optimization_results['memory_freed_mb'] = optimization_results['memory_before_mb'] - memory_after
            
            self.performance_metrics['memory_optimizations'] += 1
            self.logger.info(f"Memory optimization completed: {optimization_results}")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
            return {'error': str(e)}
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get detailed memory usage profile."""
        try:
            # System memory info
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Calculate component memory usage
            cache_memory = 0
            hero_data_memory = 0
            card_data_memory = 0
            
            try:
                from .intelligent_cache_manager import get_cache_manager
                cache_manager = get_cache_manager()
                cache_stats = cache_manager.get_statistics()
                cache_memory = cache_stats.memory_usage_mb
            except Exception:
                pass
            
            # Estimate hero vs card data memory (placeholder logic)
            estimated_data_memory = process_memory.rss / (1024 * 1024) - cache_memory
            hero_data_memory = estimated_data_memory * 0.3  # Estimate 30% for hero data
            card_data_memory = estimated_data_memory * 0.7  # Estimate 70% for card data
            
            # Calculate optimization potential
            optimization_potential = max(0, process_memory.rss / (1024 * 1024) - self.max_memory_mb * 0.8)
            
            self.memory_profile = MemoryProfile(
                total_memory_mb=memory_info.total / (1024 * 1024),
                available_memory_mb=memory_info.available / (1024 * 1024),
                used_memory_mb=memory_info.used / (1024 * 1024),
                process_memory_mb=process_memory.rss / (1024 * 1024),
                cache_memory_mb=cache_memory,
                hero_data_memory_mb=hero_data_memory,
                card_data_memory_mb=card_data_memory,
                optimization_potential_mb=optimization_potential
            )
            
            return self.memory_profile
            
        except Exception as e:
            self.logger.error(f"Error getting memory profile: {e}")
            return MemoryProfile(0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_resource_usage(self, resource_type: ResourceType) -> ResourceUsage:
        """Get current usage for specified resource type."""
        try:
            current_time = datetime.now()
            
            if resource_type == ResourceType.MEMORY:
                process = psutil.Process()
                memory_info = process.memory_info()
                current_mb = memory_info.rss / (1024 * 1024)
                percentage = (current_mb / self.max_memory_mb) * 100
                
                return ResourceUsage(
                    resource_type=resource_type,
                    current_usage=current_mb,
                    max_usage=self.max_memory_mb,
                    percentage_used=percentage,
                    last_updated=current_time
                )
                
            elif resource_type == ResourceType.NETWORK:
                percentage = (self.active_connections / self.max_connections) * 100
                
                return ResourceUsage(
                    resource_type=resource_type,
                    current_usage=self.active_connections,
                    max_usage=self.max_connections,
                    percentage_used=percentage,
                    last_updated=current_time
                )
                
            elif resource_type == ResourceType.API_CALLS:
                with self.api_call_lock:
                    current_calls = len(self.api_call_history)
                    percentage = (current_calls / self.max_api_calls_per_minute) * 100
                
                return ResourceUsage(
                    resource_type=resource_type,
                    current_usage=current_calls,
                    max_usage=self.max_api_calls_per_minute,
                    percentage_used=percentage,
                    last_updated=current_time
                )
                
            elif resource_type == ResourceType.CPU:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                return ResourceUsage(
                    resource_type=resource_type,
                    current_usage=cpu_percent,
                    max_usage=100.0,
                    percentage_used=cpu_percent,
                    last_updated=current_time
                )
                
            else:
                # Default case
                return ResourceUsage(
                    resource_type=resource_type,
                    current_usage=0,
                    max_usage=100,
                    percentage_used=0,
                    last_updated=current_time
                )
                
        except Exception as e:
            self.logger.error(f"Error getting resource usage for {resource_type}: {e}")
            return ResourceUsage(resource_type, 0, 100, 0, current_time)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def register_cleanup_handler(self, handler: callable) -> None:
        """Register a cleanup handler for memory optimization."""
        self.cleanup_handlers.append(handler)
        self.logger.debug(f"Registered cleanup handler: {handler.__name__}")
    
    def add_weak_reference(self, obj: Any) -> None:
        """Add weak reference for automatic cleanup."""
        try:
            weak_ref = weakref.ref(obj)
            self.weak_references.append(weak_ref)
        except TypeError:
            # Object doesn't support weak references
            pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            # Update resource usage
            memory_usage = self.get_resource_usage(ResourceType.MEMORY)
            network_usage = self.get_resource_usage(ResourceType.NETWORK)
            api_usage = self.get_resource_usage(ResourceType.API_CALLS)
            cpu_usage = self.get_resource_usage(ResourceType.CPU)
            
            return {
                'resource_usage': {
                    'memory': {
                        'current_mb': memory_usage.current_usage,
                        'max_mb': memory_usage.max_usage,
                        'percentage': memory_usage.percentage_used
                    },
                    'network': {
                        'active_connections': network_usage.current_usage,
                        'max_connections': network_usage.max_usage,
                        'percentage': network_usage.percentage_used
                    },
                    'api_calls': {
                        'current_calls': api_usage.current_usage,
                        'max_calls': api_usage.max_usage,
                        'percentage': api_usage.percentage_used
                    },
                    'cpu': {
                        'percentage': cpu_usage.percentage_used
                    }
                },
                'performance_counters': self.performance_metrics,
                'memory_profile': self.get_memory_profile(),
                'health_status': self._get_health_status()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Shutdown resource manager gracefully."""
        try:
            # Stop monitoring
            self.monitoring_enabled = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Close all connections
            while not self.connection_pool.empty():
                try:
                    connection = self.connection_pool.get_nowait()
                    self._close_connection(connection)
                except queue.Empty:
                    break
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Final memory optimization
            self.optimize_memory(force=True)
            
            self.logger.info("Resource manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource manager shutdown: {e}")
    
    # === INTERNAL METHODS ===
    
    def _start_monitoring(self) -> None:
        """Start background resource monitoring."""
        def monitor_worker():
            while self.monitoring_enabled:
                try:
                    self._monitor_resources()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Error in resource monitoring: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.monitoring_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor resource usage and trigger optimizations if needed."""
        try:
            # Check memory usage
            memory_usage = self.get_resource_usage(ResourceType.MEMORY)
            
            if memory_usage.percentage_used > memory_usage.critical_threshold:
                self.logger.warning(f"Critical memory usage: {memory_usage.percentage_used:.1f}%")
                self.optimize_memory(force=True)
            elif memory_usage.percentage_used > memory_usage.warning_threshold:
                self.logger.info(f"High memory usage: {memory_usage.percentage_used:.1f}%")
                # Schedule optimization
                self.executor.submit(self.optimize_memory)
            
            # Update resource usage cache
            self.resource_usage[ResourceType.MEMORY] = memory_usage
            self.resource_usage[ResourceType.NETWORK] = self.get_resource_usage(ResourceType.NETWORK)
            self.resource_usage[ResourceType.API_CALLS] = self.get_resource_usage(ResourceType.API_CALLS)
            
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {e}")
    
    def _cleanup_weak_references(self) -> int:
        """Clean up dead weak references."""
        cleaned_count = 0
        alive_refs = []
        
        for ref in self.weak_references:
            if ref() is None:
                cleaned_count += 1
            else:
                alive_refs.append(ref)
        
        self.weak_references = alive_refs
        return cleaned_count
    
    def _create_connection(self) -> Optional[Any]:
        """Create a new connection (placeholder implementation)."""
        # This would create actual network connections
        # For now, return a placeholder object
        return {"connection_id": time.time(), "created_at": datetime.now()}
    
    def _validate_connection(self, connection: Any) -> bool:
        """Validate that connection is still usable."""
        # This would check if the connection is still valid
        # For now, return True for placeholder connections
        return connection is not None
    
    def _close_connection(self, connection: Any) -> None:
        """Close a connection."""
        # This would close actual network connections
        # For placeholder, just log
        self.logger.debug(f"Closed connection: {connection}")
    
    def _get_health_status(self) -> Dict[str, str]:
        """Get overall health status."""
        try:
            memory_usage = self.get_resource_usage(ResourceType.MEMORY)
            network_usage = self.get_resource_usage(ResourceType.NETWORK)
            api_usage = self.get_resource_usage(ResourceType.API_CALLS)
            
            # Determine overall health
            issues = []
            
            if memory_usage.percentage_used > 90:
                issues.append("Critical memory usage")
            elif memory_usage.percentage_used > 80:
                issues.append("High memory usage")
            
            if network_usage.percentage_used > 90:
                issues.append("Connection pool exhausted")
            
            if api_usage.percentage_used > 90:
                issues.append("API rate limit approaching")
            
            if not issues:
                status = "Healthy"
                message = "All resources within normal limits"
            elif len(issues) == 1:
                status = "Warning"
                message = issues[0]
            else:
                status = "Critical"
                message = f"Multiple issues: {', '.join(issues)}"
            
            return {
                'status': status,
                'message': message,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'Error',
                'message': f'Health check failed: {e}',
                'issues': ['Health monitoring error']
            }


# Global resource manager instance
_resource_manager = None

def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager