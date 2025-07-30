"""
Ownership-Based Memory Model for Arena Bot AI Helper v2.

This module implements a comprehensive ownership-based memory management system
with reference counting, automatic cleanup, and memory pool management to prevent
memory leaks and ensure efficient resource utilization.

Features:
- P0.11.1: Ownership-Based Memory Model implementation
- Reference counting with automatic cleanup
- Memory pool management for object reuse
- GC pressure monitoring with idle collection
- Thread-safe memory tracking
- Weak reference support for avoiding circular references
- Memory debugging and leak detection

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import gc
import sys
import time
import threading
import weakref
import logging
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Set, Type, TypeVar, Generic, 
    Callable, Iterator, Union, Protocol
)
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import uuid

from .exceptions import AIHelperMemoryError, AIHelperResourceError
from .monitoring import get_performance_monitor


T = TypeVar('T')


class ObjectState(Enum):
    """States in the object lifecycle"""
    CREATED = "created"
    ACTIVE = "active"
    REFERENCED = "referenced" 
    UNREFERENCED = "unreferenced"
    CLEANUP_PENDING = "cleanup_pending"
    DESTROYED = "destroyed"


class OwnershipType(Enum):
    """Types of ownership relationships"""
    OWNED = "owned"          # Full ownership, responsible for cleanup
    SHARED = "shared"        # Shared ownership through reference counting
    BORROWED = "borrowed"    # Temporary reference, no cleanup responsibility
    WEAK = "weak"           # Weak reference, doesn't prevent cleanup


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_objects: int = 0
    active_objects: int = 0
    referenced_objects: int = 0
    memory_pools_active: int = 0
    total_allocated_mb: float = 0.0
    peak_allocated_mb: float = 0.0
    gc_collections: int = 0
    cleanup_operations: int = 0
    leaked_objects: int = 0


@dataclass
class ObjectInfo:
    """Information about a managed object"""
    object_id: str
    object_type: str
    creation_time: float
    reference_count: int
    state: ObjectState
    ownership_type: OwnershipType
    size_bytes: int = 0
    creator_info: Optional[str] = None
    cleanup_handlers: List[Callable] = field(default_factory=list)


class ManagedObject(Protocol):
    """Protocol for objects that can be managed by the memory model"""
    
    def cleanup(self) -> None:
        """Cleanup method called when object is being destroyed"""
        ...
    
    def get_memory_size(self) -> int:
        """Get approximate memory size in bytes"""
        ...


class ReferenceCountedMixin:
    """Mixin class that adds reference counting to objects"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ref_count = 1
        self._ref_lock = threading.RLock()
        self._object_id = str(uuid.uuid4())
        self._creation_time = time.time()
        self._cleanup_handlers: List[Callable] = []
        
        # Register with memory manager
        memory_manager = get_memory_manager()
        memory_manager._register_object(self)
    
    def add_ref(self) -> 'ReferenceCountedMixin':
        """Add reference to object"""
        with self._ref_lock:
            self._ref_count += 1
            return self
    
    def release(self) -> int:
        """Release reference to object"""
        with self._ref_lock:
            self._ref_count -= 1
            ref_count = self._ref_count
            
            if ref_count == 0:
                self._cleanup()
            elif ref_count < 0:
                raise AIHelperMemoryError(f"Reference count went negative for object {self._object_id}")
            
            return ref_count
    
    def get_ref_count(self) -> int:
        """Get current reference count"""
        with self._ref_lock:
            return self._ref_count
    
    def add_cleanup_handler(self, handler: Callable) -> None:
        """Add cleanup handler to be called when object is destroyed"""
        self._cleanup_handlers.append(handler)
    
    def _cleanup(self) -> None:
        """Internal cleanup method"""
        try:
            # Call cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    logging.error(f"Cleanup handler failed for object {self._object_id}: {e}")
            
            # Call object's cleanup method if it exists
            if hasattr(self, 'cleanup') and callable(self.cleanup):
                self.cleanup()
            
            # Unregister from memory manager
            memory_manager = get_memory_manager()
            memory_manager._unregister_object(self)
            
        except Exception as e:
            logging.error(f"Cleanup failed for object {self._object_id}: {e}")


class MemoryPool(Generic[T]):
    """
    Memory pool for object reuse to reduce allocation overhead
    """
    
    def __init__(self, object_factory: Callable[[], T], max_size: int = 100):
        """
        Initialize memory pool.
        
        Args:
            object_factory: Function to create new objects
            max_size: Maximum number of objects to pool
        """
        self.object_factory = object_factory
        self.max_size = max_size
        self._pool: deque = deque()
        self._lock = threading.Lock()
        self._created_count = 0
        self._reused_count = 0
        
        self.logger = logging.getLogger(__name__ + ".memory_pool")
    
    def acquire(self) -> T:
        """Acquire object from pool or create new one"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._reused_count += 1
                self.logger.debug(f"Reused object from pool (reused: {self._reused_count})")
                return obj
            else:
                obj = self.object_factory()
                self._created_count += 1
                self.logger.debug(f"Created new object (created: {self._created_count})")
                return obj
    
    def release(self, obj: T) -> None:
        """Release object back to pool"""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset') and callable(obj.reset):
                    try:
                        obj.reset()
                    except Exception as e:
                        self.logger.error(f"Failed to reset object for pool: {e}")
                        return  # Don't add to pool if reset failed
                
                self._pool.append(obj)
                self.logger.debug(f"Object returned to pool (pooled: {len(self._pool)})")
            else:
                # Pool is full, let object be garbage collected
                self.logger.debug("Pool full, object will be garbage collected")
    
    def clear(self) -> None:
        """Clear all objects from pool"""
        with self._lock:
            self._pool.clear()
            self.logger.info("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                'created': self._created_count,
                'reused': self._reused_count,
                'pooled': len(self._pool),
                'max_size': self.max_size
            }


class WeakReferenceManager:
    """
    Manages weak references to avoid circular reference issues
    """
    
    def __init__(self):
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + ".weak_refs")
    
    def create_weak_ref(self, obj: Any, callback: Optional[Callable] = None) -> str:
        """Create weak reference to object"""
        with self._lock:
            ref_id = str(uuid.uuid4())
            
            def cleanup_callback(ref):
                self._cleanup_weak_ref(ref_id)
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Weak reference callback failed: {e}")
            
            weak_ref = weakref.ref(obj, cleanup_callback)
            self._weak_refs[ref_id] = weak_ref
            
            self.logger.debug(f"Created weak reference {ref_id}")
            return ref_id
    
    def get_object(self, ref_id: str) -> Optional[Any]:
        """Get object from weak reference"""
        with self._lock:
            weak_ref = self._weak_refs.get(ref_id)
            if weak_ref:
                obj = weak_ref()
                if obj is None:
                    # Object was garbage collected
                    self._cleanup_weak_ref(ref_id)
                return obj
            return None
    
    def add_cleanup_callback(self, ref_id: str, callback: Callable) -> None:
        """Add callback to be called when weak reference is cleaned up"""
        with self._lock:
            self._callbacks[ref_id].append(callback)
    
    def _cleanup_weak_ref(self, ref_id: str) -> None:
        """Clean up weak reference"""
        with self._lock:
            if ref_id in self._weak_refs:
                del self._weak_refs[ref_id]
            
            # Call cleanup callbacks
            for callback in self._callbacks.get(ref_id, []):
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Weak reference cleanup callback failed: {e}")
            
            if ref_id in self._callbacks:
                del self._callbacks[ref_id]
            
            self.logger.debug(f"Cleaned up weak reference {ref_id}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get weak reference statistics"""
        with self._lock:
            return {
                'active_refs': len(self._weak_refs),
                'pending_callbacks': sum(len(callbacks) for callbacks in self._callbacks.values())
            }


class MemoryManager:
    """
    P0.11.1: Main Ownership-Based Memory Manager
    
    Provides comprehensive memory management with reference counting,
    automatic cleanup, and memory pool management.
    """
    
    def __init__(self):
        """Initialize memory manager"""
        self.logger = logging.getLogger(__name__)
        
        # Object tracking
        self._objects: Dict[str, ObjectInfo] = {}
        self._objects_lock = threading.RLock()
        
        # Memory pools
        self._memory_pools: Dict[Type, MemoryPool] = {}
        self._pools_lock = threading.Lock()
        
        # Weak reference manager
        self._weak_ref_manager = WeakReferenceManager()
        
        # Statistics
        self._stats = MemoryStats()
        self._stats_lock = threading.Lock()
        
        # GC monitoring
        self._gc_threshold = 0.8  # Start aggressive GC at 80% memory usage
        self._gc_monitoring_active = False
        self._gc_thread: Optional[threading.Thread] = None
        
        # Performance monitor integration
        try:
            self._performance_monitor = get_performance_monitor()
        except Exception:
            self._performance_monitor = None
        
        self.logger.info("Memory manager initialized")
    
    def _register_object(self, obj: Any) -> None:
        """Register object with memory manager"""
        if not hasattr(obj, '_object_id'):
            return
        
        with self._objects_lock:
            object_info = ObjectInfo(
                object_id=obj._object_id,
                object_type=type(obj).__name__,
                creation_time=getattr(obj, '_creation_time', time.time()),
                reference_count=getattr(obj, '_ref_count', 1),
                state=ObjectState.CREATED,
                ownership_type=OwnershipType.OWNED,
                size_bytes=self._estimate_object_size(obj),
                creator_info=self._get_creator_info()
            )
            
            self._objects[obj._object_id] = object_info
            
            with self._stats_lock:
                self._stats.total_objects += 1
                self._stats.active_objects += 1
        
        self.logger.debug(f"Registered object {obj._object_id} ({type(obj).__name__})")
    
    def _unregister_object(self, obj: Any) -> None:
        """Unregister object from memory manager"""
        if not hasattr(obj, '_object_id'):
            return
        
        with self._objects_lock:
            if obj._object_id in self._objects:
                object_info = self._objects[obj._object_id]
                object_info.state = ObjectState.DESTROYED
                del self._objects[obj._object_id]
                
                with self._stats_lock:
                    self._stats.active_objects -= 1
                    self._stats.cleanup_operations += 1
        
        self.logger.debug(f"Unregistered object {obj._object_id}")
    
    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            if hasattr(obj, 'get_memory_size') and callable(obj.get_memory_size):
                return obj.get_memory_size()
            else:
                return sys.getsizeof(obj)
        except Exception:
            return 0
    
    def _get_creator_info(self) -> str:
        """Get information about who created the object"""
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if frame.f_code.co_filename != __file__:
                    return f"{frame.f_code.co_filename}:{frame.f_lineno}"
                frame = frame.f_back
            return "unknown"
        except Exception:
            return "unknown"
    
    def create_memory_pool(self, object_type: Type[T], factory: Callable[[], T], max_size: int = 100) -> MemoryPool[T]:
        """Create memory pool for specific object type"""
        with self._pools_lock:
            if object_type in self._memory_pools:
                return self._memory_pools[object_type]
            
            pool = MemoryPool(factory, max_size)
            self._memory_pools[object_type] = pool
            
            with self._stats_lock:
                self._stats.memory_pools_active += 1
            
            self.logger.info(f"Created memory pool for {object_type.__name__} (max_size: {max_size})")
            return pool
    
    def get_memory_pool(self, object_type: Type[T]) -> Optional[MemoryPool[T]]:
        """Get existing memory pool for object type"""
        with self._pools_lock:
            return self._memory_pools.get(object_type)
    
    def create_weak_reference(self, obj: Any, callback: Optional[Callable] = None) -> str:
        """Create weak reference to object"""
        return self._weak_ref_manager.create_weak_ref(obj, callback)
    
    def get_weak_reference(self, ref_id: str) -> Optional[Any]:
        """Get object from weak reference"""
        return self._weak_ref_manager.get_object(ref_id)
    
    def start_gc_monitoring(self) -> None:
        """Start garbage collection monitoring"""
        if self._gc_monitoring_active:
            return
        
        self._gc_monitoring_active = True
        self._gc_thread = threading.Thread(
            target=self._gc_monitoring_loop,
            daemon=True,
            name="MemoryGCMonitor"
        )
        self._gc_thread.start()
        
        self.logger.info("GC monitoring started")
    
    def stop_gc_monitoring(self) -> None:
        """Stop garbage collection monitoring"""
        self._gc_monitoring_active = False
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=1.0)
        
        self.logger.info("GC monitoring stopped")
    
    def _gc_monitoring_loop(self) -> None:
        """Main GC monitoring loop"""
        while self._gc_monitoring_active:
            try:
                # Check memory usage
                if self._performance_monitor:
                    memory_usage = self._performance_monitor.get_memory_usage()
                    if memory_usage > self._gc_threshold:
                        self._trigger_idle_collection()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"GC monitoring error: {e}")
                time.sleep(60)  # Back off on error
    
    def _trigger_idle_collection(self) -> None:
        """Trigger idle garbage collection"""
        try:
            self.logger.info("Triggering idle garbage collection")
            
            # Force garbage collection
            collected = gc.collect()
            
            with self._stats_lock:
                self._stats.gc_collections += 1
            
            self.logger.info(f"Idle GC collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Idle GC failed: {e}")
    
    def force_cleanup(self) -> Dict[str, int]:
        """Force cleanup of all unused objects"""
        cleanup_stats = {
            'objects_cleaned': 0,
            'pools_cleared': 0,
            'gc_collected': 0
        }
        
        try:
            # Clear memory pools
            with self._pools_lock:
                for pool in self._memory_pools.values():
                    pool.clear()
                cleanup_stats['pools_cleared'] = len(self._memory_pools)
            
            # Force garbage collection
            cleanup_stats['gc_collected'] = gc.collect()
            
            with self._stats_lock:
                self._stats.gc_collections += 1
            
            self.logger.info(f"Force cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            self.logger.error(f"Force cleanup failed: {e}")
        
        return cleanup_stats
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks"""
        leaks = []
        current_time = time.time()
        
        with self._objects_lock:
            for obj_id, obj_info in self._objects.items():
                # Objects that have been alive for more than 10 minutes
                # and are not in active state might be leaks
                age = current_time - obj_info.creation_time
                if age > 600 and obj_info.state != ObjectState.ACTIVE:  # 10 minutes
                    leaks.append({
                        'object_id': obj_id,
                        'object_type': obj_info.object_type,
                        'age_seconds': age,
                        'reference_count': obj_info.reference_count,
                        'state': obj_info.state.value,
                        'creator_info': obj_info.creator_info
                    })
        
        if leaks:
            with self._stats_lock:
                self._stats.leaked_objects = len(leaks)
            
            self.logger.warning(f"Detected {len(leaks)} potential memory leaks")
        
        return leaks
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        with self._stats_lock:
            stats = MemoryStats(
                total_objects=self._stats.total_objects,
                active_objects=self._stats.active_objects,
                referenced_objects=self._stats.referenced_objects,
                memory_pools_active=self._stats.memory_pools_active,
                total_allocated_mb=self._stats.total_allocated_mb,
                peak_allocated_mb=self._stats.peak_allocated_mb,
                gc_collections=self._stats.gc_collections,
                cleanup_operations=self._stats.cleanup_operations,
                leaked_objects=self._stats.leaked_objects
            )
        
        return stats
    
    def get_object_info(self, object_id: str) -> Optional[ObjectInfo]:
        """Get information about specific object"""
        with self._objects_lock:
            return self._objects.get(object_id)
    
    def get_all_objects(self) -> List[ObjectInfo]:
        """Get information about all tracked objects"""
        with self._objects_lock:
            return list(self._objects.values())
    
    @contextmanager
    def managed_scope(self):
        """Context manager for automatic cleanup of scope-local objects"""
        scope_objects = []
        
        def track_object(obj):
            if hasattr(obj, '_object_id'):
                scope_objects.append(obj)
            return obj
        
        try:
            yield track_object
        finally:
            # Cleanup all objects created in this scope
            for obj in scope_objects:
                if hasattr(obj, 'release') and callable(obj.release):
                    try:
                        obj.release()
                    except Exception as e:
                        self.logger.error(f"Failed to release scoped object: {e}")
    
    def shutdown(self) -> None:
        """Shutdown memory manager"""
        self.stop_gc_monitoring()
        
        # Force final cleanup
        self.force_cleanup()
        
        # Clear all pools
        with self._pools_lock:
            for pool in self._memory_pools.values():
                pool.clear()
            self._memory_pools.clear()
        
        self.logger.info("Memory manager shutdown completed")


# Global instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


# Convenience functions and decorators

def managed_object(cls):
    """Class decorator to add reference counting to a class"""
    if not issubclass(cls, ReferenceCountedMixin):
        # Create a new class that inherits from both the original class and ReferenceCountedMixin
        class ManagedClass(ReferenceCountedMixin, cls):
            pass
        
        ManagedClass.__name__ = cls.__name__
        ManagedClass.__qualname__ = cls.__qualname__
        return ManagedClass
    return cls


def create_memory_pool(object_type: Type[T], factory: Callable[[], T], max_size: int = 100) -> MemoryPool[T]:
    """Create memory pool for object type"""
    return get_memory_manager().create_memory_pool(object_type, factory, max_size)


def create_weak_ref(obj: Any, callback: Optional[Callable] = None) -> str:
    """Create weak reference to object"""
    return get_memory_manager().create_weak_reference(obj, callback)


def get_memory_stats() -> MemoryStats:
    """Get memory statistics"""
    return get_memory_manager().get_memory_stats()


# Export main components
__all__ = [
    # Core Classes
    'MemoryManager',
    'ReferenceCountedMixin', 
    'MemoryPool',
    'WeakReferenceManager',
    
    # Enums
    'ObjectState',
    'OwnershipType',
    
    # Data Classes  
    'MemoryStats',
    'ObjectInfo',
    
    # Protocols
    'ManagedObject',
    
    # Factory Functions
    'get_memory_manager',
    
    # Convenience Functions
    'managed_object',
    'create_memory_pool',
    'create_weak_ref',
    'get_memory_stats'
]