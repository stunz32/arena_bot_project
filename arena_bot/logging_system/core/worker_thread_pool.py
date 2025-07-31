"""
Worker Thread Pool for S-Tier Logging System.

This module implements a dynamic thread pool for handling all I/O operations
in dedicated threads, ensuring the main application threads are never blocked
by logging operations.

Components:
- SinkWorker: Individual worker thread for processing messages
- HealthMonitor: Worker health monitoring and recovery
- WorkerThreadPool: Dynamic thread pool management

Features:
- Dynamic scaling (2-8 threads) based on queue depth
- Health monitoring with automatic thread recovery
- Graceful shutdown with message flushing
- Comprehensive performance tracking
- Load balancing across available sinks
"""

import os
import sys
import time
import threading
import logging
import uuid
import traceback
import weakref
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from queue import Queue, Empty, Full
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum

# Import from our queue system
from .hybrid_async_queue import HybridAsyncQueue, LogMessage


class WorkerState(Enum):
    """Worker thread states."""
    STARTING = "starting"
    IDLE = "idle" 
    PROCESSING = "processing"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerStats:
    """Statistics for a worker thread."""
    worker_id: str
    state: WorkerState
    messages_processed: int
    processing_time_total: float
    errors_count: int
    last_activity: float
    current_batch_size: int
    average_processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'worker_id': self.worker_id,
            'state': self.state.value,
            'messages_processed': self.messages_processed,
            'processing_time_total': self.processing_time_total,
            'errors_count': self.errors_count,
            'last_activity': self.last_activity,
            'current_batch_size': self.current_batch_size,
            'average_processing_time_ms': self.average_processing_time_ms,
            'uptime_seconds': time.time() - (self.last_activity - self.processing_time_total) if self.processing_time_total > 0 else 0
        }


class SinkWorker:
    """
    Individual worker thread for processing log messages.
    
    Each worker runs in its own thread and processes messages from the
    queue, sending them to the appropriate sinks. Includes health monitoring,
    error recovery, and performance tracking.
    """
    
    def __init__(self, 
                 worker_id: str,
                 queue: HybridAsyncQueue,
                 sinks: List[Any],
                 batch_size: int = 10,
                 batch_timeout: float = 0.1):
        """
        Initialize sink worker.
        
        Args:
            worker_id: Unique identifier for this worker
            queue: Message queue to process
            sinks: List of sink objects to send messages to  
            batch_size: Maximum messages per batch
            batch_timeout: Maximum time to wait for batch
        """
        self.worker_id = worker_id
        self.queue = queue
        self.sinks = sinks
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state = WorkerState.STARTING
        self._state_lock = threading.RLock()
        
        # Performance tracking
        self._messages_processed = 0
        self._processing_time_total = 0.0
        self._errors_count = 0
        self._last_activity = time.time()
        self._current_batch_size = 0
        self._processing_times = deque(maxlen=100)  # Last 100 processing times
        
        # Error handling
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._last_error_time = 0.0
        self._error_backoff_base = 1.0  # Base backoff time in seconds
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.SinkWorker.{worker_id}")
        
        # Health monitoring
        self._health_check_interval = 30.0  # Check health every 30 seconds
        self._last_health_check = time.time()
        
        self._logger.info(f"SinkWorker {worker_id} initialized")
    
    def start(self) -> bool:
        """
        Start the worker thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self._thread and self._thread.is_alive():
                self._logger.warning(f"Worker {self.worker_id} already running")
                return True
            
            self._stop_event.clear()
            self._state = WorkerState.STARTING
            
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"LogWorker-{self.worker_id}",
                daemon=True
            )
            self._thread.start()
            
            # Wait for worker to start (max 5 seconds)
            start_time = time.time()
            while self._state == WorkerState.STARTING and time.time() - start_time < 5.0:
                time.sleep(0.01)
            
            if self._state == WorkerState.IDLE:
                self._logger.info(f"Worker {self.worker_id} started successfully")
                return True
            else:
                self._logger.error(f"Worker {self.worker_id} failed to start")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to start worker {self.worker_id}: {e}")
            self._state = WorkerState.ERROR
            return False
    
    def stop(self, timeout: float = 30.0) -> bool:
        """
        Stop the worker thread gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            True if stopped cleanly, False if forced
        """
        try:
            if not self._thread or not self._thread.is_alive():
                self._state = WorkerState.STOPPED
                return True
            
            self._logger.info(f"Stopping worker {self.worker_id}")
            self._state = WorkerState.STOPPING
            self._stop_event.set()
            
            # Wait for thread to finish
            self._thread.join(timeout)
            
            if self._thread.is_alive():
                self._logger.warning(f"Worker {self.worker_id} did not stop cleanly")
                self._state = WorkerState.ERROR
                return False
            else:
                self._state = WorkerState.STOPPED
                self._logger.info(f"Worker {self.worker_id} stopped cleanly")
                return True
                
        except Exception as e:
            self._logger.error(f"Error stopping worker {self.worker_id}: {e}")
            self._state = WorkerState.ERROR
            return False
    
    def _worker_loop(self) -> None:
        """Main worker loop - runs in dedicated thread."""
        try:
            self._state = WorkerState.IDLE
            self._logger.info(f"Worker {self.worker_id} started")
            
            while not self._stop_event.is_set():
                try:
                    # Process batch of messages
                    batch_processed = self._process_message_batch()
                    
                    if batch_processed == 0:
                        # No messages available, brief sleep
                        self._state = WorkerState.IDLE
                        time.sleep(0.01)  # 10ms sleep when idle
                    else:
                        self._state = WorkerState.PROCESSING
                    
                    # Periodic health check
                    self._perform_health_check()
                    
                    # Reset error backoff on successful processing
                    if batch_processed > 0:
                        self._consecutive_errors = 0
                    
                except Exception as e:
                    self._handle_worker_error(e)
                    
        except Exception as e:
            self._logger.critical(f"Worker {self.worker_id} loop crashed: {e}")
            self._state = WorkerState.ERROR
        finally:
            self._logger.info(f"Worker {self.worker_id} loop ended")
    
    def _process_message_batch(self) -> int:
        """
        Process a batch of messages from the queue.
        
        Returns:
            Number of messages processed
        """
        start_time = time.perf_counter()
        
        try:
            # Get batch of messages
            messages = self.queue.get(timeout=self.batch_timeout, batch_size=self.batch_size)
            
            if not messages:
                return 0
            
            # Handle single message vs batch
            if not isinstance(messages, list):
                messages = [messages]
            
            if not messages:
                return 0
            
            self._current_batch_size = len(messages)
            
            # Process each message through all sinks
            processed_count = 0
            for message in messages:
                if self._stop_event.is_set():
                    break
                    
                try:
                    self._process_single_message(message)
                    processed_count += 1
                    
                except Exception as e:
                    self._logger.error(f"Failed to process message {message.sequence_number}: {e}")
                    # Continue processing other messages
            
            # Update statistics
            elapsed_time = time.perf_counter() - start_time
            self._messages_processed += processed_count
            self._processing_time_total += elapsed_time
            self._processing_times.append(elapsed_time)
            self._last_activity = time.time()
            self._current_batch_size = 0
            
            return processed_count
            
        except Exception as e:
            self._logger.error(f"Batch processing failed: {e}")
            return 0
    
    def _process_single_message(self, message: LogMessage) -> None:
        """
        Process a single message through all sinks.
        
        Args:
            message: LogMessage to process
        """
        for sink in self.sinks:
            try:
                # Check if sink is available and healthy
                if hasattr(sink, 'is_healthy') and not sink.is_healthy():
                    continue
                
                # Send message to sink
                sink.write(message)
                
            except Exception as e:
                self._logger.warning(f"Sink {type(sink).__name__} failed to process message: {e}")
                # Continue with other sinks
    
    def _handle_worker_error(self, error: Exception) -> None:
        """
        Handle worker errors with backoff and recovery.
        
        Args:
            error: Exception that occurred
        """
        self._errors_count += 1
        self._consecutive_errors += 1
        self._last_error_time = time.time()
        
        self._logger.error(f"Worker {self.worker_id} error #{self._consecutive_errors}: {error}")
        
        # Check if we should degrade
        if self._consecutive_errors >= self._max_consecutive_errors:
            self._state = WorkerState.DEGRADED
            self._logger.warning(f"Worker {self.worker_id} entering degraded mode")
        
        # Apply exponential backoff
        backoff_time = min(
            self._error_backoff_base * (2 ** min(self._consecutive_errors - 1, 5)),
            30.0  # Max 30 second backoff
        )
        
        self._logger.info(f"Worker {self.worker_id} backing off for {backoff_time:.1f}s")
        time.sleep(backoff_time)
    
    def _perform_health_check(self) -> None:
        """Perform periodic health check."""
        current_time = time.time()
        
        if current_time - self._last_health_check > self._health_check_interval:
            self._last_health_check = current_time
            
            # Check for stalled worker
            if current_time - self._last_activity > 300:  # 5 minutes
                self._logger.warning(f"Worker {self.worker_id} appears stalled")
                self._state = WorkerState.DEGRADED
            
            # Check sink health
            unhealthy_sinks = []
            for sink in self.sinks:
                try:
                    if hasattr(sink, 'is_healthy') and not sink.is_healthy():
                        unhealthy_sinks.append(type(sink).__name__)
                except Exception as e:
                    self._logger.warning(f"Health check failed for sink {type(sink).__name__}: {e}")
            
            if unhealthy_sinks:
                self._logger.warning(f"Unhealthy sinks detected: {unhealthy_sinks}")
    
    def get_stats(self) -> WorkerStats:
        """Get current worker statistics."""
        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times else 0.0
        )
        
        return WorkerStats(
            worker_id=self.worker_id,
            state=self._state,
            messages_processed=self._messages_processed,
            processing_time_total=self._processing_time_total,
            errors_count=self._errors_count,
            last_activity=self._last_activity,
            current_batch_size=self._current_batch_size,
            average_processing_time_ms=avg_processing_time * 1000
        )
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return (
            self._state in (WorkerState.IDLE, WorkerState.PROCESSING) and
            time.time() - self._last_activity < 300 and  # Active within 5 minutes
            self._consecutive_errors < self._max_consecutive_errors
        )


class HealthMonitor:
    """
    Health monitoring for worker threads.
    
    Monitors worker health and automatically recovers failed threads.
    Provides comprehensive health reporting and alerting.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        
        # Monitoring state
        self._workers: Dict[str, SinkWorker] = {}
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Health tracking
        self._health_history = deque(maxlen=100)
        self._last_check_time = 0.0
        self._recovery_attempts = defaultdict(int)
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.HealthMonitor")
    
    def register_worker(self, worker: SinkWorker) -> None:
        """
        Register a worker for monitoring.
        
        Args:
            worker: SinkWorker to monitor
        """
        with self._lock:
            self._workers[worker.worker_id] = worker
            self._logger.info(f"Registered worker {worker.worker_id} for health monitoring")
    
    def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker from monitoring.
        
        Args:
            worker_id: ID of worker to unregister
        """
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                self._logger.info(f"Unregistered worker {worker_id} from health monitoring")
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        with self._lock:
            if self._monitoring_active:
                return
            
            self._monitoring_active = True
            self._stop_event.clear()
            
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="WorkerHealthMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
            
            self._logger.info("Worker health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        with self._lock:
            if not self._monitoring_active:
                return
            
            self._monitoring_active = False
            self._stop_event.set()
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            self._logger.info("Worker health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
                
            except Exception as e:
                self._logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _perform_health_check(self) -> None:
        """Perform comprehensive health check on all workers."""
        current_time = time.time()
        self._last_check_time = current_time
        
        health_report = {
            'timestamp': current_time,
            'total_workers': 0,
            'healthy_workers': 0,
            'degraded_workers': 0,
            'failed_workers': 0,
            'recoveries_attempted': 0,
            'worker_details': {}
        }
        
        with self._lock:
            workers_copy = dict(self._workers)  # Copy to avoid lock issues
        
        for worker_id, worker in workers_copy.items():
            try:
                stats = worker.get_stats()
                is_healthy = worker.is_healthy()
                
                health_report['total_workers'] += 1
                health_report['worker_details'][worker_id] = stats.to_dict()
                
                if is_healthy:
                    health_report['healthy_workers'] += 1
                elif stats.state == WorkerState.DEGRADED:
                    health_report['degraded_workers'] += 1
                    self._attempt_worker_recovery(worker)
                    health_report['recoveries_attempted'] += 1
                else:
                    health_report['failed_workers'] += 1
                    self._attempt_worker_recovery(worker)
                    health_report['recoveries_attempted'] += 1
                
            except Exception as e:
                self._logger.error(f"Health check failed for worker {worker_id}: {e}")
                health_report['failed_workers'] += 1
        
        # Store health history
        self._health_history.append(health_report)
        
        # Log summary if there are issues
        if health_report['degraded_workers'] > 0 or health_report['failed_workers'] > 0:
            self._logger.warning(
                f"Worker health issues: {health_report['degraded_workers']} degraded, "
                f"{health_report['failed_workers']} failed out of {health_report['total_workers']} total"
            )
    
    def _attempt_worker_recovery(self, worker: SinkWorker) -> bool:
        """
        Attempt to recover a failed worker.
        
        Args:
            worker: Worker to recover
            
        Returns:
            True if recovery attempted, False if skipped
        """
        worker_id = worker.worker_id
        
        # Limit recovery attempts
        if self._recovery_attempts[worker_id] >= 3:
            self._logger.error(f"Max recovery attempts reached for worker {worker_id}")
            return False
        
        try:
            self._recovery_attempts[worker_id] += 1
            self._logger.info(f"Attempting recovery for worker {worker_id} (attempt #{self._recovery_attempts[worker_id]})")
            
            # Stop the worker
            worker.stop(timeout=10.0)
            
            # Wait briefly
            time.sleep(1.0)
            
            # Restart the worker
            if worker.start():
                self._logger.info(f"Successfully recovered worker {worker_id}")
                self._recovery_attempts[worker_id] = 0  # Reset on successful recovery
                return True
            else:
                self._logger.error(f"Failed to recover worker {worker_id}")
                return False
                
        except Exception as e:
            self._logger.error(f"Worker recovery failed for {worker_id}: {e}")
            return False
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self._health_history:
            return {'status': 'no_data', 'message': 'No health data available'}
        
        latest_report = self._health_history[-1]
        
        # Calculate health trends
        if len(self._health_history) >= 2:
            previous_report = self._health_history[-2]
            trend = {
                'healthy_change': latest_report['healthy_workers'] - previous_report['healthy_workers'],
                'degraded_change': latest_report['degraded_workers'] - previous_report['degraded_workers'],
                'failed_change': latest_report['failed_workers'] - previous_report['failed_workers']
            }
        else:
            trend = {'healthy_change': 0, 'degraded_change': 0, 'failed_change': 0}
        
        # Overall health status
        total_workers = latest_report['total_workers']
        healthy_workers = latest_report['healthy_workers']
        
        if total_workers == 0:
            overall_status = 'no_workers'
        elif healthy_workers == total_workers:
            overall_status = 'healthy'
        elif healthy_workers >= total_workers * 0.7:
            overall_status = 'degraded'
        else:
            overall_status = 'critical'
        
        return {
            'overall_status': overall_status,
            'latest_report': latest_report,
            'trend': trend,
            'last_check_time': self._last_check_time,
            'monitoring_active': self._monitoring_active,
            'recovery_attempts': dict(self._recovery_attempts)
        }


class WorkerThreadPool:
    """
    Dynamic thread pool for logging worker threads.
    
    Manages a pool of worker threads that process messages from the queue.
    Provides dynamic scaling, health monitoring, and graceful shutdown.
    
    Features:
    - Dynamic scaling (2-8 threads) based on queue depth
    - Health monitoring with automatic recovery
    - Load balancing across sinks
    - Graceful shutdown with message flushing
    - Comprehensive performance tracking
    """
    
    def __init__(self,
                 queue: HybridAsyncQueue,
                 sinks: List[Any],
                 min_workers: int = 2,
                 max_workers: int = 8,
                 scale_up_threshold: int = 1000,
                 scale_down_threshold: int = 100,
                 scaling_check_interval: float = 10.0):
        """
        Initialize worker thread pool.
        
        Args:
            queue: Message queue to process
            sinks: List of sink objects
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads
            scale_up_threshold: Queue size to trigger scale up
            scale_down_threshold: Queue size to trigger scale down
            scaling_check_interval: How often to check scaling needs
        """
        self.queue = queue
        self.sinks = sinks
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_check_interval = scaling_check_interval
        
        # Worker management
        self._workers: Dict[str, SinkWorker] = {}
        self._target_worker_count = min_workers
        self._worker_id_counter = 0
        self._lock = threading.RLock()
        
        # Health monitoring
        self.health_monitor = HealthMonitor()
        
        # Scaling management
        self._scaling_active = False
        self._scaling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self._start_time = time.time()
        self._scaling_events = deque(maxlen=50)
        
        # Create logger
        self._logger = logging.getLogger(f"{__name__}.WorkerThreadPool")
        
        self._logger.info(f"WorkerThreadPool initialized: {min_workers}-{max_workers} workers")
    
    def start(self) -> bool:
        """
        Start the worker thread pool.
        
        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                # Start health monitoring
                self.health_monitor.start_monitoring()
                
                # Create initial workers
                for _ in range(self.min_workers):
                    self._create_worker()
                
                # Start scaling management
                self._scaling_active = True
                self._stop_event.clear()
                
                self._scaling_thread = threading.Thread(
                    target=self._scaling_loop,
                    name="WorkerPoolScaling",
                    daemon=True
                )
                self._scaling_thread.start()
                
                self._logger.info(f"WorkerThreadPool started with {len(self._workers)} workers")
                return True
                
        except Exception as e:
            self._logger.error(f"Failed to start WorkerThreadPool: {e}")
            return False
    
    def shutdown(self, timeout: float = 60.0) -> bool:
        """
        Shutdown worker thread pool gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            True if shutdown cleanly
        """
        try:
            self._logger.info("WorkerThreadPool shutting down")
            
            # Stop scaling
            self._scaling_active = False
            self._stop_event.set()
            
            if self._scaling_thread and self._scaling_thread.is_alive():
                self._scaling_thread.join(timeout=5.0)
            
            # Stop all workers
            with self._lock:
                workers_copy = dict(self._workers)
            
            worker_timeout = timeout / len(workers_copy) if workers_copy else 5.0
            
            for worker_id, worker in workers_copy.items():
                try:
                    self._logger.info(f"Stopping worker {worker_id}")
                    worker.stop(timeout=worker_timeout)
                    self.health_monitor.unregister_worker(worker_id)
                except Exception as e:
                    self._logger.error(f"Error stopping worker {worker_id}: {e}")
            
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
            
            # Clear workers
            with self._lock:
                self._workers.clear()
            
            self._logger.info("WorkerThreadPool shutdown complete")
            return True
            
        except Exception as e:
            self._logger.error(f"WorkerThreadPool shutdown error: {e}")
            return False
    
    def _create_worker(self) -> Optional[SinkWorker]:
        """
        Create a new worker thread.
        
        Returns:
            SinkWorker instance if created successfully
        """
        try:
            self._worker_id_counter += 1
            worker_id = f"worker-{self._worker_id_counter:03d}"
            
            worker = SinkWorker(
                worker_id=worker_id,
                queue=self.queue,
                sinks=self.sinks,
                batch_size=10,
                batch_timeout=0.1
            )
            
            if worker.start():
                with self._lock:
                    self._workers[worker_id] = worker
                
                self.health_monitor.register_worker(worker)
                self._logger.info(f"Created worker {worker_id}")
                return worker
            else:
                self._logger.error(f"Failed to start worker {worker_id}")
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to create worker: {e}")
            return None
    
    def _remove_worker(self, worker_id: str) -> bool:
        """
        Remove a worker thread.
        
        Args:
            worker_id: ID of worker to remove
            
        Returns:
            True if removed successfully
        """
        try:
            with self._lock:
                if worker_id not in self._workers:
                    return False
                
                worker = self._workers[worker_id]
                
            # Stop the worker
            if worker.stop(timeout=10.0):
                with self._lock:
                    del self._workers[worker_id]
                
                self.health_monitor.unregister_worker(worker_id)
                self._logger.info(f"Removed worker {worker_id}")
                return True
            else:
                self._logger.error(f"Failed to stop worker {worker_id}")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to remove worker {worker_id}: {e}")
            return False
    
    def _scaling_loop(self) -> None:
        """Main scaling management loop."""
        while not self._stop_event.is_set():
            try:
                self._check_scaling_needs()
                time.sleep(self.scaling_check_interval)
                
            except Exception as e:
                self._logger.error(f"Scaling loop error: {e}")
                time.sleep(self.scaling_check_interval)
    
    def _check_scaling_needs(self) -> None:
        """Check if we need to scale up or down."""
        try:
            queue_size = self.queue.size()
            current_worker_count = len(self._workers)
            
            # Determine target worker count
            if queue_size > self.scale_up_threshold and current_worker_count < self.max_workers:
                # Scale up
                self._target_worker_count = min(
                    current_worker_count + 1,
                    self.max_workers
                )
                self._scale_to_target()
                
                self._scaling_events.append({
                    'timestamp': time.time(),
                    'action': 'scale_up',
                    'queue_size': queue_size,
                    'old_count': current_worker_count,
                    'new_count': self._target_worker_count
                })
                
            elif queue_size < self.scale_down_threshold and current_worker_count > self.min_workers:
                # Scale down (but be conservative)
                self._target_worker_count = max(
                    current_worker_count - 1,
                    self.min_workers
                )
                self._scale_to_target()
                
                self._scaling_events.append({
                    'timestamp': time.time(),
                    'action': 'scale_down',
                    'queue_size': queue_size,
                    'old_count': current_worker_count,
                    'new_count': self._target_worker_count
                })
                
        except Exception as e:
            self._logger.error(f"Scaling check failed: {e}")
    
    def _scale_to_target(self) -> None:
        """Scale workers to target count."""
        try:
            current_count = len(self._workers)
            
            if self._target_worker_count > current_count:
                # Add workers
                for _ in range(self._target_worker_count - current_count):
                    if not self._create_worker():
                        break
                        
            elif self._target_worker_count < current_count:
                # Remove workers (remove oldest/least active)
                with self._lock:
                    workers_to_remove = list(self._workers.keys())[:current_count - self._target_worker_count]
                
                for worker_id in workers_to_remove:
                    self._remove_worker(worker_id)
                    
        except Exception as e:
            self._logger.error(f"Scaling to target failed: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive thread pool statistics."""
        try:
            with self._lock:
                workers_copy = dict(self._workers)
            
            # Collect worker stats
            worker_stats = {}
            total_messages = 0
            total_errors = 0
            
            for worker_id, worker in workers_copy.items():
                try:
                    stats = worker.get_stats()
                    worker_stats[worker_id] = stats.to_dict()
                    total_messages += stats.messages_processed
                    total_errors += stats.errors_count
                except Exception as e:
                    self._logger.warning(f"Failed to get stats for worker {worker_id}: {e}")
            
            # Health report
            health_report = self.health_monitor.get_health_report()
            
            # Recent scaling events
            recent_scaling = list(self._scaling_events)[-10:]  # Last 10 events
            
            return {
                'pool_status': {
                    'active_workers': len(workers_copy),
                    'target_workers': self._target_worker_count,
                    'min_workers': self.min_workers,
                    'max_workers': self.max_workers,
                    'uptime_seconds': time.time() - self._start_time
                },
                'performance': {
                    'total_messages_processed': total_messages,
                    'total_errors': total_errors,
                    'error_rate_percent': (total_errors / total_messages * 100) if total_messages > 0 else 0,
                    'queue_size': self.queue.size(),
                    'scaling_events_count': len(self._scaling_events)
                },
                'workers': worker_stats,
                'health': health_report,
                'recent_scaling': recent_scaling,
                'configuration': {
                    'scale_up_threshold': self.scale_up_threshold,
                    'scale_down_threshold': self.scale_down_threshold,
                    'scaling_check_interval': self.scaling_check_interval
                }
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get comprehensive stats: {e}")
            return {'error': str(e)}


# Module exports
__all__ = [
    'WorkerState',
    'WorkerStats',
    'SinkWorker',
    'HealthMonitor', 
    'WorkerThreadPool'
]