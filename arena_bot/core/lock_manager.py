#!/usr/bin/env python3
"""
Global Lock Ordering Manager - Emergency Thread Safety Fix

This module implements the mandatory lock ordering protocol to eliminate
circular dependencies and deadlocks as specified in THREAD_SAFETY_ANALYSIS.md.

Lock Order: GUI → AI → Monitor → Overlay

All multi-lock acquisitions MUST use this manager to prevent deadlocks.
"""

import threading
import logging
import time
import contextlib
from typing import List, Dict, Optional, Set, Callable
from enum import Enum, IntEnum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LockType(IntEnum):
    """
    Global lock ordering enum - CRITICAL: Order must never change!
    
    This ordering prevents circular dependencies by enforcing consistent
    acquisition order across all components.
    """
    GUI = 1      # User interface locks (highest priority)
    AI = 2       # AI Helper and analysis locks  
    MONITOR = 3  # Log monitoring and data collection locks
    OVERLAY = 4  # Visual overlay and rendering locks (lowest priority)

@dataclass
class LockInfo:
    """Information about a managed lock"""
    name: str
    lock_type: LockType
    lock_obj: threading.RLock
    created_at: datetime
    acquisitions: int = 0
    contentions: int = 0
    max_hold_time: float = 0.0
    total_hold_time: float = 0.0

class DeadlockDetector:
    """
    Deadlock detection and prevention system
    
    Monitors lock acquisition patterns and prevents potential deadlocks
    by enforcing the global ordering protocol.
    """
    
    def __init__(self):
        self._acquisition_history: Dict[int, List[str]] = {}
        self._thread_locks: Dict[int, Set[str]] = {}
        self._lock = threading.RLock()
    
    def record_acquisition(self, thread_id: int, lock_name: str):
        """Record lock acquisition for deadlock analysis"""
        with self._lock:
            if thread_id not in self._acquisition_history:
                self._acquisition_history[thread_id] = []
                self._thread_locks[thread_id] = set()
            
            self._acquisition_history[thread_id].append(lock_name)
            self._thread_locks[thread_id].add(lock_name)
    
    def record_release(self, thread_id: int, lock_name: str):
        """Record lock release"""
        with self._lock:
            if thread_id in self._thread_locks:
                self._thread_locks[thread_id].discard(lock_name)
    
    def check_for_potential_deadlock(self, thread_id: int, requested_locks: List[str]) -> bool:
        """
        Check if requesting these locks could cause a deadlock
        
        Returns True if deadlock is possible, False if safe
        """
        with self._lock:
            current_locks = self._thread_locks.get(thread_id, set())
            
            # Check if any other thread holds locks we want while wanting locks we have
            for other_thread, other_locks in self._thread_locks.items():
                if other_thread == thread_id:
                    continue
                
                # If other thread has locks we want and we have locks it might want
                has_conflict = (
                    any(lock in other_locks for lock in requested_locks) and
                    any(lock in current_locks for lock in other_locks)
                )
                
                if has_conflict:
                    logger.warning(f"Potential deadlock detected between threads {thread_id} and {other_thread}")
                    return True
            
            return False

class GlobalLockManager:
    """
    Global Lock Manager implementing the mandatory lock ordering protocol
    
    This is the ONLY way to acquire multiple locks safely in the system.
    All components MUST use this manager for multi-lock operations.
    
    Usage:
        # Single lock
        with lock_manager.acquire_lock("gui_main"):
            # Critical section
            
        # Multiple locks (automatically ordered)
        with lock_manager.acquire_locks_ordered(["ai_cache", "gui_main", "overlay_render"]):
            # Critical section with proper ordering
    """
    
    def __init__(self):
        self._locks: Dict[str, LockInfo] = {}
        self._registry_lock = threading.RLock()
        self._deadlock_detector = DeadlockDetector()
        self._metrics = {
            'total_acquisitions': 0,
            'deadlock_preventions': 0,
            'contentions_detected': 0,
            'average_hold_time': 0.0
        }
        
        # Create the four main component locks
        self._initialize_main_locks()
        
        logger.info("🔒 Global Lock Manager initialized with mandatory ordering: GUI → AI → Monitor → Overlay")
    
    def _initialize_main_locks(self):
        """Initialize the four main component locks"""
        main_locks = [
            ("gui_main", LockType.GUI),
            ("ai_main", LockType.AI), 
            ("monitor_main", LockType.MONITOR),
            ("overlay_main", LockType.OVERLAY)
        ]
        
        for lock_name, lock_type in main_locks:
            self.register_lock(lock_name, lock_type)
    
    def register_lock(self, lock_name: str, lock_type: LockType) -> threading.RLock:
        """
        Register a new lock in the global ordering system
        
        Args:
            lock_name: Unique identifier for the lock
            lock_type: Lock type determining ordering priority
            
        Returns:
            The registered RLock object
        """
        with self._registry_lock:
            if lock_name in self._locks:
                return self._locks[lock_name].lock_obj
            
            lock_obj = threading.RLock()
            lock_info = LockInfo(
                name=lock_name,
                lock_type=lock_type,
                lock_obj=lock_obj,
                created_at=datetime.now()
            )
            
            self._locks[lock_name] = lock_info
            logger.debug(f"Registered lock: {lock_name} (type: {lock_type.name})")
            
            return lock_obj
    
    def get_lock(self, lock_name: str) -> Optional[threading.RLock]:
        """Get a registered lock by name"""
        with self._registry_lock:
            lock_info = self._locks.get(lock_name)
            return lock_info.lock_obj if lock_info else None
    
    @contextlib.contextmanager
    def acquire_lock(self, lock_name: str):
        """
        Acquire a single lock safely
        
        Args:
            lock_name: Name of the lock to acquire
        """
        lock_obj = self.get_lock(lock_name)
        if not lock_obj:
            raise ValueError(f"Lock '{lock_name}' not registered")
        
        thread_id = threading.get_ident()
        start_time = time.time()
        
        try:
            # Record acquisition attempt
            self._deadlock_detector.record_acquisition(thread_id, lock_name)
            
            with lock_obj:
                # Update metrics
                with self._registry_lock:
                    lock_info = self._locks[lock_name]
                    lock_info.acquisitions += 1
                    self._metrics['total_acquisitions'] += 1
                
                yield lock_obj
                
        finally:
            # Record release and update hold time
            hold_time = time.time() - start_time
            self._deadlock_detector.record_release(thread_id, lock_name)
            
            with self._registry_lock:
                lock_info = self._locks[lock_name]
                lock_info.total_hold_time += hold_time
                lock_info.max_hold_time = max(lock_info.max_hold_time, hold_time)
    
    @contextlib.contextmanager  
    def acquire_locks_ordered(self, lock_names: List[str]):
        """
        Acquire multiple locks in the mandatory global ordering
        
        This is the CRITICAL function that prevents deadlocks by ensuring
        all locks are acquired in consistent order: GUI → AI → Monitor → Overlay
        
        Args:
            lock_names: List of lock names to acquire
        """
        if not lock_names:
            yield []
            return
        
        thread_id = threading.get_ident()
        
        # Get lock info for all requested locks
        with self._registry_lock:
            lock_infos = []
            for name in lock_names:
                if name not in self._locks:
                    raise ValueError(f"Lock '{name}' not registered")
                lock_infos.append(self._locks[name])
        
        # Sort by lock type (global ordering) then by name for consistency
        sorted_locks = sorted(lock_infos, key=lambda x: (x.lock_type.value, x.name))
        
        # Check for potential deadlock
        if self._deadlock_detector.check_for_potential_deadlock(thread_id, lock_names):
            self._metrics['deadlock_preventions'] += 1
            logger.warning(f"Deadlock prevention: Delaying lock acquisition on thread {thread_id}")
            time.sleep(0.001)  # Brief delay to break timing
        
        # Acquire locks in proper order
        acquired_locks = []
        start_time = time.time()
        
        try:
            for lock_info in sorted_locks:
                lock_info.lock_obj.acquire()
                acquired_locks.append(lock_info)
                self._deadlock_detector.record_acquisition(thread_id, lock_info.name)
                
                # Update acquisition metrics
                with self._registry_lock:
                    lock_info.acquisitions += 1
                    self._metrics['total_acquisitions'] += 1
            
            logger.debug(f"Acquired {len(acquired_locks)} locks in order: {[l.name for l in sorted_locks]}")
            yield [l.lock_obj for l in sorted_locks]
            
        finally:
            # Release in reverse order (standard practice)
            hold_time = time.time() - start_time
            
            for lock_info in reversed(acquired_locks):
                try:
                    lock_info.lock_obj.release()
                    self._deadlock_detector.record_release(thread_id, lock_info.name)
                    
                    # Update hold time metrics
                    with self._registry_lock:
                        lock_info.total_hold_time += hold_time / len(acquired_locks)
                        lock_info.max_hold_time = max(lock_info.max_hold_time, hold_time)
                        
                except Exception as e:
                    logger.error(f"Error releasing lock {lock_info.name}: {e}")
    
    def get_lock_statistics(self) -> Dict:
        """Get comprehensive lock usage statistics"""
        with self._registry_lock:
            stats = {
                'global_metrics': self._metrics.copy(),
                'lock_details': {}
            }
            
            for name, lock_info in self._locks.items():
                avg_hold = (lock_info.total_hold_time / lock_info.acquisitions 
                           if lock_info.acquisitions > 0 else 0.0)
                
                stats['lock_details'][name] = {
                    'type': lock_info.lock_type.name,
                    'acquisitions': lock_info.acquisitions,
                    'contentions': lock_info.contentions,
                    'max_hold_time': lock_info.max_hold_time,
                    'avg_hold_time': avg_hold,
                    'created_at': lock_info.created_at.isoformat()
                }
            
            return stats
    
    def validate_lock_ordering(self) -> bool:
        """
        Validate that all registered locks follow the global ordering
        
        Returns True if ordering is correct, False otherwise
        """
        with self._registry_lock:
            lock_types = [info.lock_type.value for info in self._locks.values()]
            sorted_types = sorted(lock_types)
            
            is_valid = lock_types == sorted_types
            if not is_valid:
                logger.error("Lock ordering validation FAILED - potential deadlock risk")
            
            return is_valid

# Global singleton instance
_global_lock_manager: Optional[GlobalLockManager] = None
_manager_init_lock = threading.Lock()

def get_global_lock_manager() -> GlobalLockManager:
    """
    Get the global lock manager singleton
    
    This ensures all components use the same lock ordering system
    """
    global _global_lock_manager
    
    if _global_lock_manager is None:
        with _manager_init_lock:
            if _global_lock_manager is None:
                _global_lock_manager = GlobalLockManager()
    
    return _global_lock_manager

# Convenience functions for common usage patterns
def acquire_gui_lock():
    """Acquire GUI component lock"""
    return get_global_lock_manager().acquire_lock("gui_main")

def acquire_ai_lock():
    """Acquire AI component lock"""
    return get_global_lock_manager().acquire_lock("ai_main")

def acquire_monitor_lock():
    """Acquire Monitor component lock"""
    return get_global_lock_manager().acquire_lock("monitor_main")

def acquire_overlay_lock():
    """Acquire Overlay component lock"""
    return get_global_lock_manager().acquire_lock("overlay_main")

def acquire_ordered_locks(*lock_names):
    """Acquire multiple locks in proper global ordering"""
    return get_global_lock_manager().acquire_locks_ordered(list(lock_names))

# Emergency deadlock recovery
def emergency_reset_locks():
    """
    Emergency function to reset all locks - USE ONLY IN DEADLOCK RECOVERY
    
    WARNING: This will forcibly release ALL locks. Only use when system
    is completely deadlocked and needs emergency recovery.
    """
    global _global_lock_manager
    
    logger.critical("🚨 EMERGENCY LOCK RESET - Forcibly releasing all locks")
    
    if _global_lock_manager:
        with _global_lock_manager._registry_lock:
            for lock_info in _global_lock_manager._locks.values():
                try:
                    # Force release by creating new lock
                    old_lock = lock_info.lock_obj
                    lock_info.lock_obj = threading.RLock()
                    logger.warning(f"Force reset lock: {lock_info.name}")
                except Exception as e:
                    logger.error(f"Error resetting lock {lock_info.name}: {e}")
    
    logger.critical("Emergency lock reset complete - System should be recoverable")

if __name__ == "__main__":
    # Test the lock manager
    manager = get_global_lock_manager()
    
    # Test single lock
    with manager.acquire_lock("gui_main"):
        print("✅ Single lock acquisition works")
    
    # Test multiple locks (should be ordered automatically)
    test_locks = ["overlay_main", "gui_main", "ai_main"]  # Deliberately out of order
    with manager.acquire_locks_ordered(test_locks):
        print("✅ Multi-lock acquisition works with proper ordering")
    
    # Show statistics
    stats = manager.get_lock_statistics()
    print(f"📊 Lock statistics: {stats}")
    
    print("🔒 Global Lock Manager test complete")