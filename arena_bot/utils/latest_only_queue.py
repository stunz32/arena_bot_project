"""
Latest-only delivery system for high-frequency producer scenarios.

Provides thread-safe queuing that automatically drops stale items when a
faster producer overwhelms a slower consumer, ensuring only the latest
data is processed.
"""

import threading
import time
from typing import TypeVar, Generic, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class FrameSnapshot:
    """Immutable snapshot of frame data with monotonic ID."""
    frame_id: int
    timestamp: float
    data: Any
    producer_thread_id: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Ensure immutability by freezing after creation."""
        # Note: For true immutability, data should be frozen/copied by caller
        object.__setattr__(self, '_immutable', True)
    
    def __setattr__(self, name, value):
        """Prevent modification after creation."""
        if hasattr(self, '_immutable') and self._immutable:
            raise AttributeError(f"Cannot modify immutable FrameSnapshot.{name}")
        super().__setattr__(name, value)


class LatestOnlyQueue(Generic[T]):
    """
    Thread-safe queue that maintains only the latest item.
    
    When a producer is faster than consumer, old items are automatically
    dropped to prevent backpressure and ensure real-time processing.
    """
    
    def __init__(self, name: str = "unknown"):
        """
        Initialize latest-only queue.
        
        Args:
            name: Queue name for debugging and logging
        """
        self.name = name
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Current state
        self._current_snapshot: Optional[FrameSnapshot] = None
        self._last_frame_id = 0
        self._is_closed = False
        
        # Statistics
        self._stats = {
            'items_queued': 0,
            'items_dropped': 0,
            'items_consumed': 0,
            'producer_threads': set(),
            'consumer_threads': set(),
            'total_wait_time': 0.0,
            'max_wait_time': 0.0
        }
        
        logger.debug(f"Initialized LatestOnlyQueue: {name}")
    
    def put_latest(self, data: T, producer_id: Optional[str] = None) -> int:
        """
        Put new data, replacing any existing unconsumed data.
        
        Args:
            data: Data to queue (should be immutable or will be copied)
            producer_id: Optional producer identifier for debugging
            
        Returns:
            Frame ID of the queued snapshot
        """
        with self._condition:
            if self._is_closed:
                raise ValueError(f"Queue {self.name} is closed")
            
            # Generate monotonic frame ID
            self._last_frame_id += 1
            frame_id = self._last_frame_id
            
            # Create immutable snapshot
            snapshot = FrameSnapshot(
                frame_id=frame_id,
                timestamp=time.time(),
                data=data,
                producer_thread_id=threading.get_ident()
            )
            
            # Track if we're dropping an existing item
            dropped_frame = None
            if self._current_snapshot is not None:
                dropped_frame = self._current_snapshot.frame_id
                self._stats['items_dropped'] += 1
            
            # Replace current snapshot
            self._current_snapshot = snapshot
            self._stats['items_queued'] += 1
            self._stats['producer_threads'].add(threading.get_ident())
            
            # Notify waiting consumers
            self._condition.notify_all()
            
            if dropped_frame is not None:
                logger.debug(f"Queue {self.name}: frame {frame_id} replaced frame {dropped_frame}")
            else:
                logger.debug(f"Queue {self.name}: queued frame {frame_id}")
            
            return frame_id
    
    def get_latest(self, timeout: Optional[float] = None) -> Optional[FrameSnapshot]:
        """
        Get the latest available snapshot, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait for data (None = wait forever)
            
        Returns:
            Latest snapshot, or None if timeout or queue closed
        """
        start_time = time.time()
        
        with self._condition:
            # Wait for data if none available
            while self._current_snapshot is None and not self._is_closed:
                if not self._condition.wait(timeout=timeout):
                    # Timeout occurred
                    return None
            
            # Queue was closed
            if self._is_closed:
                return None
            
            # Get and clear current snapshot
            snapshot = self._current_snapshot
            self._current_snapshot = None
            
            # Update statistics
            wait_time = time.time() - start_time
            self._stats['items_consumed'] += 1
            self._stats['consumer_threads'].add(threading.get_ident())
            self._stats['total_wait_time'] += wait_time
            self._stats['max_wait_time'] = max(self._stats['max_wait_time'], wait_time)
            
            logger.debug(f"Queue {self.name}: consumed frame {snapshot.frame_id} "
                        f"(waited {wait_time*1000:.1f}ms)")
            
            return snapshot
    
    def peek_latest(self) -> Optional[FrameSnapshot]:
        """
        Peek at latest snapshot without consuming it.
        
        Returns:
            Latest snapshot, or None if queue empty
        """
        with self._condition:
            return self._current_snapshot
    
    def has_data(self) -> bool:
        """Check if queue has unconsumed data."""
        with self._condition:
            return self._current_snapshot is not None
    
    def get_frame_id(self) -> int:
        """Get the next frame ID that will be assigned."""
        with self._condition:
            return self._last_frame_id + 1
    
    def close(self):
        """Close the queue and wake up any waiting consumers."""
        with self._condition:
            self._is_closed = True
            self._condition.notify_all()
            logger.debug(f"Queue {self.name} closed")
    
    def clear(self):
        """Clear any pending data."""
        with self._condition:
            if self._current_snapshot is not None:
                dropped_frame = self._current_snapshot.frame_id
                self._current_snapshot = None
                self._stats['items_dropped'] += 1
                logger.debug(f"Queue {self.name}: cleared frame {dropped_frame}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._condition:
            avg_wait_time = (
                self._stats['total_wait_time'] / self._stats['items_consumed']
                if self._stats['items_consumed'] > 0 else 0.0
            )
            
            return {
                'name': self.name,
                'current_frame_id': self._last_frame_id,
                'has_pending_data': self._current_snapshot is not None,
                'is_closed': self._is_closed,
                'statistics': {
                    **{k: v for k, v in self._stats.items() 
                       if k not in ['producer_threads', 'consumer_threads']},
                    'producer_thread_count': len(self._stats['producer_threads']),
                    'consumer_thread_count': len(self._stats['consumer_threads']),
                    'avg_wait_time': avg_wait_time,
                    'drop_rate': (
                        self._stats['items_dropped'] / self._stats['items_queued']
                        if self._stats['items_queued'] > 0 else 0.0
                    )
                }
            }


class PipelineFrameManager:
    """
    Manages frame flow through detection pipeline with latest-only delivery.
    
    Coordinates multiple latest-only queues for different pipeline stages
    and ensures no OpenCV operations occur on UI thread.
    """
    
    def __init__(self, name: str = "pipeline"):
        """
        Initialize pipeline frame manager.
        
        Args:
            name: Pipeline name for debugging
        """
        self.name = name
        self._queues: Dict[str, LatestOnlyQueue] = {}
        self._lock = threading.Lock()
        self._ui_thread_id: Optional[int] = None
        
        logger.debug(f"Initialized PipelineFrameManager: {name}")
    
    def register_ui_thread(self, thread_id: Optional[int] = None):
        """
        Register the UI thread ID to prevent OpenCV operations on it.
        
        Args:
            thread_id: Thread ID, or None to use current thread
        """
        with self._lock:
            self._ui_thread_id = thread_id or threading.get_ident()
            logger.debug(f"Registered UI thread: {self._ui_thread_id}")
    
    def check_opencv_safety(self):
        """
        Check if it's safe to perform OpenCV operations on current thread.
        
        Raises:
            RuntimeError: If called from UI thread
        """
        current_thread = threading.get_ident()
        if self._ui_thread_id and current_thread == self._ui_thread_id:
            raise RuntimeError(
                f"OpenCV operations not allowed on UI thread {current_thread}. "
                f"Use background thread for image processing."
            )
    
    def get_queue(self, stage_name: str) -> LatestOnlyQueue:
        """
        Get or create a latest-only queue for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Latest-only queue for the stage
        """
        with self._lock:
            if stage_name not in self._queues:
                queue_name = f"{self.name}_{stage_name}"
                self._queues[stage_name] = LatestOnlyQueue(queue_name)
                logger.debug(f"Created queue for stage: {stage_name}")
            
            return self._queues[stage_name]
    
    def put_frame(self, stage_name: str, frame_data: Any) -> int:
        """
        Put frame data into a stage queue.
        
        Args:
            stage_name: Pipeline stage name
            frame_data: Frame data (should be immutable)
            
        Returns:
            Frame ID
        """
        queue = self.get_queue(stage_name)
        return queue.put_latest(frame_data)
    
    def get_frame(self, stage_name: str, timeout: Optional[float] = None) -> Optional[FrameSnapshot]:
        """
        Get latest frame from a stage queue.
        
        Args:
            stage_name: Pipeline stage name
            timeout: Maximum wait time
            
        Returns:
            Latest frame snapshot, or None if timeout
        """
        queue = self.get_queue(stage_name)
        return queue.get_latest(timeout=timeout)
    
    def close_all(self):
        """Close all stage queues."""
        with self._lock:
            for queue in self._queues.values():
                queue.close()
            logger.debug(f"Closed all queues for pipeline: {self.name}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics for all pipeline stages."""
        with self._lock:
            return {
                'pipeline_name': self.name,
                'ui_thread_id': self._ui_thread_id,
                'stage_count': len(self._queues),
                'stages': {name: queue.get_stats() for name, queue in self._queues.items()}
            }


# Global pipeline manager instance
_default_pipeline = PipelineFrameManager("default")


def get_default_pipeline() -> PipelineFrameManager:
    """Get the default pipeline frame manager."""
    return _default_pipeline


def ensure_opencv_thread_safety():
    """Ensure current thread is safe for OpenCV operations."""
    _default_pipeline.check_opencv_safety()