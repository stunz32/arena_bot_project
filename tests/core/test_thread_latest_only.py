"""
Thread safety and latest-only delivery tests.

Tests the latest-only queue system with faster producer and slower consumer
scenarios to ensure stale frames are dropped and no races occur.
"""

import os
import sys
import pytest
import time
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import random

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.utils.latest_only_queue import (
    LatestOnlyQueue, FrameSnapshot, PipelineFrameManager,
    get_default_pipeline, ensure_opencv_thread_safety
)


class TestLatestOnlyQueue:
    """Test latest-only queue functionality."""
    
    @pytest.fixture
    def queue(self):
        """Create a test queue."""
        return LatestOnlyQueue("test_queue")
    
    def test_basic_queue_operations(self, queue):
        """Test basic put and get operations."""
        # Queue should start empty
        assert not queue.has_data()
        assert queue.peek_latest() is None
        
        # Put some data
        frame_id = queue.put_latest("test_data")
        assert frame_id == 1
        assert queue.has_data()
        
        # Peek without consuming
        snapshot = queue.peek_latest()
        assert snapshot is not None
        assert snapshot.frame_id == 1
        assert snapshot.data == "test_data"
        assert queue.has_data()  # Still has data after peek
        
        # Get and consume
        snapshot = queue.get_latest(timeout=1.0)
        assert snapshot is not None
        assert snapshot.frame_id == 1
        assert snapshot.data == "test_data"
        assert not queue.has_data()  # No data after consumption
        
        print("✅ Basic queue operations work correctly")
    
    def test_latest_only_behavior(self, queue):
        """Test that only the latest item is kept."""
        # Put multiple items rapidly
        frame_ids = []
        for i in range(5):
            frame_id = queue.put_latest(f"data_{i}")
            frame_ids.append(frame_id)
            time.sleep(0.001)  # Small delay to ensure different timestamps
        
        # Should have only the latest item
        assert queue.has_data()
        snapshot = queue.get_latest(timeout=1.0)
        
        assert snapshot is not None
        assert snapshot.frame_id == frame_ids[-1]  # Last frame ID
        assert snapshot.data == "data_4"  # Last data
        
        # Queue should be empty now
        assert not queue.has_data()
        
        # Check statistics
        stats = queue.get_stats()
        assert stats['statistics']['items_queued'] == 5
        assert stats['statistics']['items_dropped'] == 4  # 4 items were dropped
        assert stats['statistics']['items_consumed'] == 1
        
        print(f"✅ Latest-only behavior: {stats['statistics']['items_dropped']} items dropped as expected")
    
    def test_immutable_snapshots(self, queue):
        """Test that snapshots are immutable after creation."""
        # Put test data
        test_data = {"mutable": "original"}
        frame_id = queue.put_latest(test_data)
        
        # Get snapshot
        snapshot = queue.get_latest(timeout=1.0)
        assert snapshot is not None
        
        # Attempt to modify snapshot should fail
        with pytest.raises(AttributeError):
            snapshot.frame_id = 999
        
        with pytest.raises(AttributeError):
            snapshot.data = "modified"
        
        # Original data reference can still be modified (caller responsibility)
        test_data["mutable"] = "modified"
        # But snapshot should maintain reference to original object
        assert snapshot.data["mutable"] == "modified"  # This shows why caller should copy
        
        print("✅ Snapshot immutability enforced")
    
    def test_monotonic_frame_ids(self, queue):
        """Test that frame IDs are strictly monotonic."""
        frame_ids = []
        
        # Put items from multiple threads
        def producer_thread(thread_id: int):
            for i in range(10):
                frame_id = queue.put_latest(f"thread_{thread_id}_item_{i}")
                frame_ids.append(frame_id)
                time.sleep(0.001)
        
        # Run multiple producer threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=producer_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Frame IDs should be strictly monotonic
        sorted_frame_ids = sorted(frame_ids)
        assert frame_ids != sorted_frame_ids or len(set(frame_ids)) == len(frame_ids), \
            "Frame IDs should be unique and generated in thread-safe manner"
        
        # All frame IDs should be unique
        assert len(set(frame_ids)) == len(frame_ids), "All frame IDs should be unique"
        
        # Frame IDs should be sequential (1, 2, 3, ...)
        expected_ids = list(range(1, len(frame_ids) + 1))
        assert sorted(frame_ids) == expected_ids, "Frame IDs should be sequential"
        
        print(f"✅ Monotonic frame IDs: {len(frame_ids)} unique IDs generated")
    
    def test_faster_producer_slower_consumer(self, queue):
        """Test scenario with faster producer and slower consumer."""
        consumed_frames = []
        producer_count = 0
        stop_event = threading.Event()
        
        def fast_producer():
            """Produce frames rapidly."""
            nonlocal producer_count
            while not stop_event.is_set():
                producer_count += 1
                queue.put_latest(f"frame_{producer_count}")
                time.sleep(0.001)  # 1ms between frames = 1000 FPS
        
        def slow_consumer():
            """Consume frames slowly."""
            while not stop_event.is_set():
                snapshot = queue.get_latest(timeout=0.1)
                if snapshot:
                    consumed_frames.append(snapshot)
                time.sleep(0.05)  # 50ms between consumption = 20 FPS
        
        # Start producer and consumer
        producer_thread = threading.Thread(target=fast_producer)
        consumer_thread = threading.Thread(target=slow_consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        # Let them run for a short time
        time.sleep(0.5)
        stop_event.set()
        
        # Wait for threads to finish
        producer_thread.join(timeout=1.0)
        consumer_thread.join(timeout=1.0)
        
        # Get final statistics
        stats = queue.get_stats()
        
        # Validate behavior
        assert producer_count > len(consumed_frames), \
            f"Producer should be faster: produced {producer_count}, consumed {len(consumed_frames)}"
        
        assert stats['statistics']['items_dropped'] > 0, \
            "Should have dropped frames due to fast producer"
        
        # Consumed frames should have increasing frame IDs (latest-only)
        if len(consumed_frames) > 1:
            frame_ids = [s.frame_id for s in consumed_frames]
            assert all(frame_ids[i] < frame_ids[i+1] for i in range(len(frame_ids)-1)), \
                f"Consumed frame IDs should be increasing: {frame_ids}"
        
        drop_rate = stats['statistics']['drop_rate']
        print(f"✅ Fast producer scenario: {producer_count} produced, "
              f"{len(consumed_frames)} consumed, {drop_rate:.1%} drop rate")
    
    def test_thread_safety_stress(self, queue):
        """Stress test thread safety with multiple producers and consumers."""
        results = {"produced": 0, "consumed": 0, "errors": []}
        lock = threading.Lock()
        stop_event = threading.Event()
        
        def producer(producer_id: int):
            """Producer thread."""
            try:
                count = 0
                while not stop_event.is_set():
                    queue.put_latest(f"producer_{producer_id}_frame_{count}")
                    count += 1
                    with lock:
                        results["produced"] += 1
                    time.sleep(random.uniform(0.001, 0.005))  # 1-5ms
            except Exception as e:
                results["errors"].append(f"Producer {producer_id}: {e}")
        
        def consumer(consumer_id: int):
            """Consumer thread."""
            try:
                while not stop_event.is_set():
                    snapshot = queue.get_latest(timeout=0.01)
                    if snapshot:
                        with lock:
                            results["consumed"] += 1
                        # Simulate processing time
                        time.sleep(random.uniform(0.005, 0.01))  # 5-10ms
            except Exception as e:
                results["errors"].append(f"Consumer {consumer_id}: {e}")
        
        # Create multiple producers and consumers
        threads = []
        
        # 3 producers
        for i in range(3):
            thread = threading.Thread(target=producer, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 2 consumers
        for i in range(2):
            thread = threading.Thread(target=consumer, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Let stress test run
        time.sleep(1.0)
        stop_event.set()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Check for errors
        assert len(results["errors"]) == 0, f"Thread safety errors: {results['errors']}"
        
        # Validate results
        assert results["produced"] > 0, "Should have produced some frames"
        assert results["consumed"] > 0, "Should have consumed some frames"
        
        stats = queue.get_stats()
        print(f"✅ Thread safety stress test: {results['produced']} produced, "
              f"{results['consumed']} consumed, {stats['statistics']['drop_rate']:.1%} drop rate")
    
    def test_queue_close_behavior(self, queue):
        """Test queue closing and cleanup."""
        # Put some data
        queue.put_latest("test_data")
        assert queue.has_data()
        
        # Close queue
        queue.close()
        
        # Should not be able to put new data
        with pytest.raises(ValueError):
            queue.put_latest("new_data")
        
        # Getting from closed queue should return None
        snapshot = queue.get_latest(timeout=0.1)
        assert snapshot is None
        
        stats = queue.get_stats()
        assert stats['is_closed'] == True
        
        print("✅ Queue close behavior works correctly")


class TestPipelineFrameManager:
    """Test pipeline frame manager functionality."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline manager."""
        return PipelineFrameManager("test_pipeline")
    
    def test_pipeline_stage_management(self, pipeline):
        """Test pipeline stage creation and management."""
        # Get queues for different stages
        coord_queue = pipeline.get_queue("coordinates")
        match_queue = pipeline.get_queue("matching")
        
        assert coord_queue is not pipeline.get_queue("matching")
        assert coord_queue is pipeline.get_queue("coordinates")  # Same instance
        
        # Put data in different stages
        coord_frame_id = pipeline.put_frame("coordinates", {"x": 100, "y": 200})
        match_frame_id = pipeline.put_frame("matching", {"card_id": "fireball"})
        
        assert coord_frame_id == 1
        assert match_frame_id == 1  # Different queues have independent frame IDs
        
        # Get data from stages
        coord_snapshot = pipeline.get_frame("coordinates", timeout=0.1)
        match_snapshot = pipeline.get_frame("matching", timeout=0.1)
        
        assert coord_snapshot is not None
        assert coord_snapshot.data == {"x": 100, "y": 200}
        
        assert match_snapshot is not None  
        assert match_snapshot.data == {"card_id": "fireball"}
        
        print("✅ Pipeline stage management works correctly")
    
    def test_opencv_thread_safety_check(self, pipeline):
        """Test OpenCV thread safety enforcement."""
        # Register current thread as UI thread
        pipeline.register_ui_thread()
        
        # Should raise error when checking from UI thread
        with pytest.raises(RuntimeError, match="OpenCV operations not allowed on UI thread"):
            pipeline.check_opencv_safety()
        
        # Should work from different thread
        def background_check():
            pipeline.check_opencv_safety()  # Should not raise
        
        thread = threading.Thread(target=background_check)
        thread.start()
        thread.join()
        
        print("✅ OpenCV thread safety check works correctly")
    
    def test_pipeline_statistics(self, pipeline):
        """Test pipeline statistics collection."""
        # Create some activity
        pipeline.put_frame("stage1", "data1")
        pipeline.put_frame("stage2", "data2")
        pipeline.put_frame("stage1", "data1_updated")  # Should drop previous
        
        pipeline.get_frame("stage1", timeout=0.1)
        
        # Get statistics
        stats = pipeline.get_pipeline_stats()
        
        assert stats['pipeline_name'] == "test_pipeline"
        assert stats['stage_count'] == 2
        assert 'stage1' in stats['stages']
        assert 'stage2' in stats['stages']
        
        stage1_stats = stats['stages']['stage1']
        assert stage1_stats['statistics']['items_queued'] == 2
        assert stage1_stats['statistics']['items_dropped'] == 1
        assert stage1_stats['statistics']['items_consumed'] == 1
        
        print(f"✅ Pipeline statistics: {stats['stage_count']} stages tracked")


class TestGlobalPipelineIntegration:
    """Test global pipeline integration."""
    
    def test_default_pipeline_access(self):
        """Test default pipeline access and OpenCV safety."""
        default_pipeline = get_default_pipeline()
        assert default_pipeline is not None
        
        # Should work by default (no UI thread registered)
        ensure_opencv_thread_safety()  # Should not raise
        
        # Register current thread as UI
        default_pipeline.register_ui_thread()
        
        # Now should fail
        with pytest.raises(RuntimeError):
            ensure_opencv_thread_safety()
        
        print("✅ Default pipeline integration works correctly")
    
    def test_multi_threaded_pipeline_usage(self):
        """Test pipeline usage across multiple threads."""
        pipeline = get_default_pipeline()
        results = {"processed": 0, "errors": []}
        lock = threading.Lock()
        
        def processing_thread(thread_id: int):
            """Simulate processing thread."""
            try:
                # Should be safe on background thread
                ensure_opencv_thread_safety()
                
                # Process some frames
                for i in range(10):
                    frame_data = {"thread": thread_id, "frame": i, "data": f"processed_{i}"}
                    pipeline.put_frame(f"thread_{thread_id}", frame_data)
                    
                    # Simulate processing delay
                    time.sleep(0.001)
                    
                    with lock:
                        results["processed"] += 1
                        
            except Exception as e:
                results["errors"].append(f"Thread {thread_id}: {e}")
        
        # Create multiple processing threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=processing_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
        assert results["processed"] == 30, f"Expected 30 processed, got {results['processed']}"
        
        # Check pipeline statistics
        stats = pipeline.get_pipeline_stats()
        assert stats['stage_count'] >= 3  # At least 3 thread stages
        
        print(f"✅ Multi-threaded pipeline: {results['processed']} frames processed across "
              f"{len(threads)} threads")


# Integration test
def test_realistic_pipeline_scenario():
    """
    Integration test simulating realistic detection pipeline scenario.
    
    Tests the complete flow: UI thread → coordinate detection → card matching → UI update
    """
    pipeline = PipelineFrameManager("detection_pipeline")
    
    # Register main thread as UI thread (simulate GUI)
    pipeline.register_ui_thread()
    
    results = {
        "screenshots": 0,
        "coordinates": 0,
        "matches": 0,
        "ui_updates": 0,
        "errors": []
    }
    
    def screenshot_producer():
        """Simulate screenshot capture (would run on timer)."""
        try:
            for i in range(20):
                # Simulate screenshot data
                screenshot_data = {
                    "image_id": i,
                    "timestamp": time.time(),
                    "width": 1920,
                    "height": 1080
                }
                pipeline.put_frame("screenshots", screenshot_data)
                results["screenshots"] += 1
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            results["errors"].append(f"Screenshot producer: {e}")
    
    def coordinate_detector():
        """Simulate coordinate detection (background thread)."""
        try:
            while True:
                snapshot = pipeline.get_frame("screenshots", timeout=1.0)
                if snapshot is None:
                    break
                
                # Simulate OpenCV work (should be safe on background thread)
                ensure_opencv_thread_safety()
                
                # Simulate coordinate detection
                coords = {
                    "frame_id": snapshot.frame_id,
                    "cards": [
                        {"x": 100, "y": 200, "w": 150, "h": 200},
                        {"x": 300, "y": 200, "w": 150, "h": 200},
                        {"x": 500, "y": 200, "w": 150, "h": 200}
                    ]
                }
                pipeline.put_frame("coordinates", coords)
                results["coordinates"] += 1
                time.sleep(0.01)  # 10ms processing time
        except Exception as e:
            results["errors"].append(f"Coordinate detector: {e}")
    
    def card_matcher():
        """Simulate card matching (background thread).""" 
        try:
            while True:
                snapshot = pipeline.get_frame("coordinates", timeout=1.0)
                if snapshot is None:
                    break
                
                # Simulate card matching
                matches = {
                    "frame_id": snapshot.frame_id,
                    "matches": [
                        {"card_id": "fireball", "confidence": 0.95},
                        {"card_id": "polymorph", "confidence": 0.87},
                        {"card_id": "flamestrike", "confidence": 0.92}
                    ]
                }
                pipeline.put_frame("matches", matches)
                results["matches"] += 1
                time.sleep(0.05)  # 50ms processing time
        except Exception as e:
            results["errors"].append(f"Card matcher: {e}")
    
    def ui_updater():
        """Simulate UI updates (main thread)."""
        try:
            while True:
                snapshot = pipeline.get_frame("matches", timeout=1.0)
                if snapshot is None:
                    break
                
                # UI updates should NOT do OpenCV operations
                # This would fail: ensure_opencv_thread_safety()
                
                # Simulate UI update
                results["ui_updates"] += 1
                time.sleep(0.016)  # 60 FPS UI updates
        except Exception as e:
            results["errors"].append(f"UI updater: {e}")
    
    # Start background threads
    coord_thread = threading.Thread(target=coordinate_detector)
    match_thread = threading.Thread(target=card_matcher)
    
    coord_thread.start()
    match_thread.start()
    
    # Start screenshot production (simulate main thread timer)
    screenshot_thread = threading.Thread(target=screenshot_producer)
    screenshot_thread.start()
    
    # Simulate UI updates on main thread
    ui_thread = threading.Thread(target=ui_updater)
    ui_thread.start()
    
    # Wait for screenshot production to finish
    screenshot_thread.join()
    
    # Give processing time to catch up
    time.sleep(0.5)
    
    # Close pipeline to signal threads to stop
    pipeline.close_all()
    
    # Wait for all threads
    coord_thread.join(timeout=2.0)
    match_thread.join(timeout=2.0)
    ui_thread.join(timeout=2.0)
    
    # Validate results
    assert len(results["errors"]) == 0, f"Pipeline errors: {results['errors']}"
    assert results["screenshots"] > 0, "Should have captured screenshots"
    assert results["coordinates"] > 0, "Should have detected coordinates"
    assert results["matches"] > 0, "Should have matched cards"
    assert results["ui_updates"] > 0, "Should have updated UI"
    
    # Get final statistics
    stats = pipeline.get_pipeline_stats()
    
    print(f"✅ Realistic pipeline scenario:")
    print(f"   Screenshots: {results['screenshots']}")
    print(f"   Coordinates: {results['coordinates']}")
    print(f"   Matches: {results['matches']}")
    print(f"   UI Updates: {results['ui_updates']}")
    print(f"   Pipeline stages: {stats['stage_count']}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])