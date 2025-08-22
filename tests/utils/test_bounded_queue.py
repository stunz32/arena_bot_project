"""
Tests for BoundedQueue utility - latest-only delivery validation

Tests the bounded queue implementation to ensure proper thread-safe behavior
and latest-only delivery when producers are faster than consumers.
"""

import threading
import time
import queue
from unittest import TestCase

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.utils.bounded_queue import BoundedQueue


class TestBoundedQueue(TestCase):
    """Test BoundedQueue implementation"""
    
    def test_basic_put_get(self):
        """Test basic put and get operations"""
        q = BoundedQueue(maxsize=3)
        
        # Initially empty
        self.assertTrue(q.is_empty())
        self.assertFalse(q.is_full())
        self.assertEqual(q.size(), 0)
        
        # Add items
        q.put("item1")
        q.put("item2")
        
        self.assertEqual(q.size(), 2)
        self.assertFalse(q.is_empty())
        self.assertFalse(q.is_full())
        
        # Get items
        item1 = q.get_nowait()
        item2 = q.get_nowait()
        
        self.assertEqual(item1, "item1")
        self.assertEqual(item2, "item2")
        self.assertTrue(q.is_empty())
    
    def test_overflow_behavior(self):
        """Test that queue drops oldest items when full"""
        q = BoundedQueue(maxsize=3)
        
        # Fill the queue
        q.put("item1")
        q.put("item2")
        q.put("item3")
        
        self.assertTrue(q.is_full())
        self.assertEqual(q.size(), 3)
        
        # Add one more - should drop oldest
        dropped = q.put("item4")
        
        self.assertEqual(dropped, "item1")
        self.assertEqual(q.size(), 3)
        self.assertTrue(q.is_full())
        
        # Verify remaining items
        items = []
        while not q.is_empty():
            items.append(q.get_nowait())
        
        self.assertEqual(items, ["item2", "item3", "item4"])
    
    def test_put_latest_only(self):
        """Test put_latest_only clears queue first"""
        q = BoundedQueue(maxsize=5)
        
        # Add several items
        q.put("old1")
        q.put("old2")
        q.put("old3")
        
        self.assertEqual(q.size(), 3)
        
        # Use put_latest_only
        dropped_count = q.put_latest_only("latest")
        
        self.assertEqual(dropped_count, 3)
        self.assertEqual(q.size(), 1)
        
        # Only latest item should remain
        item = q.get_nowait()
        self.assertEqual(item, "latest")
        self.assertTrue(q.is_empty())
    
    def test_get_timeout(self):
        """Test get with timeout"""
        q = BoundedQueue(maxsize=3)
        
        # Should timeout on empty queue
        with self.assertRaises(TimeoutError):
            q.get(timeout=0.1)
        
        # Put item in another thread after delay
        def delayed_put():
            time.sleep(0.1)
            q.put("delayed_item")
        
        thread = threading.Thread(target=delayed_put)
        thread.start()
        
        # Should get the item
        item = q.get(timeout=0.5)
        self.assertEqual(item, "delayed_item")
        
        thread.join()
    
    def test_get_nowait_empty(self):
        """Test get_nowait on empty queue raises exception"""
        q = BoundedQueue(maxsize=3)
        
        with self.assertRaises(ValueError):
            q.get_nowait()
    
    def test_latest_only_delivery_under_pressure(self):
        """
        Test latest-only behavior with fast producer and slow consumer.
        
        This is the key test for the "latest-only" delivery requirement:
        - Queue size should never exceed maxsize
        - Consumer should ultimately observe the most recent value
        """
        q = BoundedQueue(maxsize=5)
        
        # Shared state
        producer_done = threading.Event()
        consumer_results = []
        last_produced_value = None
        
        def fast_producer():
            """Produce items rapidly"""
            nonlocal last_produced_value
            
            for i in range(50):  # Produce 50 items rapidly
                value = f"item_{i:03d}"
                q.put(value)
                last_produced_value = value
                time.sleep(0.001)  # Very fast producer
            
            producer_done.set()
        
        def slow_consumer():
            """Consume items slowly"""
            while not producer_done.is_set() or not q.is_empty():
                try:
                    item = q.get(timeout=0.1)
                    consumer_results.append(item)
                    time.sleep(0.05)  # Slow consumer
                except TimeoutError:
                    if producer_done.is_set():
                        break
        
        # Start threads
        producer_thread = threading.Thread(target=fast_producer)
        consumer_thread = threading.Thread(target=slow_consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        # Monitor queue size during execution
        max_observed_size = 0
        while not producer_done.is_set():
            current_size = q.size()
            max_observed_size = max(max_observed_size, current_size)
            time.sleep(0.01)
        
        # Wait for completion
        producer_thread.join()
        consumer_thread.join()
        
        # Verify queue size never exceeded limit
        self.assertLessEqual(max_observed_size, q.maxsize(),
                           f"Queue size {max_observed_size} exceeded maxsize {q.maxsize()}")
        
        # Verify consumer got some items
        self.assertGreater(len(consumer_results), 0, "Consumer should have received items")
        
        # Verify last consumed item is recent (not necessarily the very last due to timing)
        # We check it's in the last 20% of produced items
        if consumer_results:
            last_consumed = consumer_results[-1]
            last_consumed_num = int(last_consumed.split('_')[1])
            self.assertGreater(last_consumed_num, 40,  # Last 20% of 50 items
                             f"Last consumed item {last_consumed} should be recent")
        
        # Final queue should be empty or have only recent items
        remaining_items = []
        while not q.is_empty():
            remaining_items.append(q.get_nowait())
        
        for item in remaining_items:
            item_num = int(item.split('_')[1])
            self.assertGreater(item_num, 40, f"Remaining item {item} should be recent")
    
    def test_thread_safety(self):
        """Test thread safety with multiple producers and consumers"""
        q = BoundedQueue(maxsize=10)
        
        items_produced = []
        items_consumed = []
        production_done = threading.Event()
        
        def producer(producer_id):
            for i in range(20):
                item = f"producer_{producer_id}_item_{i}"
                q.put(item)
                items_produced.append(item)
                time.sleep(0.01)
        
        def consumer(consumer_id):
            consumed = []
            while not production_done.is_set() or not q.is_empty():
                try:
                    item = q.get(timeout=0.1)
                    consumed.append(item)
                    time.sleep(0.015)
                except TimeoutError:
                    if production_done.is_set():
                        break
            items_consumed.extend(consumed)
        
        # Start multiple producers and consumers
        producers = [threading.Thread(target=producer, args=(i,)) for i in range(3)]
        consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(2)]
        
        for p in producers:
            p.start()
        for c in consumers:
            c.start()
        
        # Wait for producers to finish
        for p in producers:
            p.join()
        
        production_done.set()
        
        # Wait for consumers to finish
        for c in consumers:
            c.join()
        
        # Verify no items were lost due to race conditions
        # (Some may be dropped due to overflow, but all consumed items should be valid)
        for consumed_item in items_consumed:
            self.assertIn(consumed_item, items_produced,
                         f"Consumed item {consumed_item} was not produced")
    
    def test_clear_queue(self):
        """Test clearing the queue"""
        q = BoundedQueue(maxsize=5)
        
        # Add items
        q.put("item1")
        q.put("item2") 
        q.put("item3")
        
        self.assertEqual(q.size(), 3)
        
        # Clear queue
        cleared_count = q.clear()
        
        self.assertEqual(cleared_count, 3)
        self.assertEqual(q.size(), 0)
        self.assertTrue(q.is_empty())
    
    def test_invalid_maxsize(self):
        """Test that invalid maxsize raises error"""
        with self.assertRaises(ValueError):
            BoundedQueue(maxsize=0)
        
        with self.assertRaises(ValueError):
            BoundedQueue(maxsize=-1)