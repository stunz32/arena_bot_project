"""
Thread-safe bounded queue with drop-oldest policy.

Provides a queue implementation that caps size and drops the oldest items
when full, ensuring latest-only delivery for high-frequency producers.
"""

import threading
from collections import deque
from typing import Any, Optional


class BoundedQueue:
    """
    Thread-safe bounded queue with drop-oldest policy.
    
    When the queue is full and a new item is added, the oldest item
    is automatically dropped. This ensures that consumers always get
    the most recent items when producers are faster than consumers.
    
    Usage:
        queue = BoundedQueue(maxsize=5)
        queue.put("item1")
        queue.put("item2")
        item = queue.get()  # Returns "item1"
    """
    
    def __init__(self, maxsize: int = 10):
        """
        Initialize bounded queue.
        
        Args:
            maxsize: Maximum number of items in queue
        """
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        
        self._maxsize = maxsize
        self._queue = deque(maxlen=maxsize)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
    
    def put(self, item: Any) -> Optional[Any]:
        """
        Put an item into the queue.
        
        If the queue is full, the oldest item is dropped and returned.
        
        Args:
            item: Item to add to queue
            
        Returns:
            The dropped item if queue was full, None otherwise
        """
        dropped_item = None
        
        with self._lock:
            # Check if we'll drop an item
            if len(self._queue) == self._maxsize:
                dropped_item = self._queue[0]  # Will be dropped by append
            
            # Add new item (deque handles size limit automatically)
            self._queue.append(item)
            
            # Notify waiting consumers
            self._not_empty.notify()
        
        return dropped_item
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """
        Get an item from the queue.
        
        Args:
            timeout: Maximum time to wait for an item (None = wait forever)
            
        Returns:
            The next item in the queue
            
        Raises:
            TimeoutError: If timeout expires before an item is available
        """
        with self._not_empty:
            # Wait for item to be available
            while len(self._queue) == 0:
                if not self._not_empty.wait(timeout):
                    raise TimeoutError("Queue get() timed out")
            
            # Get the oldest item
            return self._queue.popleft()
    
    def get_nowait(self) -> Any:
        """
        Get an item from the queue without blocking.
        
        Returns:
            The next item in the queue
            
        Raises:
            ValueError: If the queue is empty
        """
        with self._lock:
            if len(self._queue) == 0:
                raise ValueError("Queue is empty")
            return self._queue.popleft()
    
    def put_latest_only(self, item: Any) -> int:
        """
        Put an item, clearing the queue first for latest-only behavior.
        
        This ensures consumers always get the most recent item by
        clearing any pending items before adding the new one.
        
        Args:
            item: Item to add to queue
            
        Returns:
            Number of items that were dropped
        """
        with self._lock:
            dropped_count = len(self._queue)
            self._queue.clear()
            self._queue.append(item)
            self._not_empty.notify()
            return dropped_count
    
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._queue)
    
    def maxsize(self) -> int:
        """Get maximum queue size"""
        return self._maxsize
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._queue) == 0
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        with self._lock:
            return len(self._queue) == self._maxsize
    
    def clear(self) -> int:
        """
        Clear all items from queue.
        
        Returns:
            Number of items that were removed
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count