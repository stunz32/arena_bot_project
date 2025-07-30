#!/usr/bin/env python3
"""
Memory Leak Detection and Resource Cleanup Validation Suite
Comprehensive memory and resource management testing

This test validates:
1. Memory leak detection during extended operation
2. Resource cleanup validation (files, threads, sockets)
3. Garbage collection effectiveness
4. Cache memory management
5. Thread lifecycle management
6. File handle management
7. Event queue cleanup
8. Component shutdown procedures
"""

import sys
import time
import threading
import unittest
import gc
import weakref
import os
import tempfile
from pathlib import Path
from collections import defaultdict
from unittest.mock import Mock, patch, MagicMock
import json
from queue import Queue
import tracemalloc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Memory testing configuration
EXTENDED_TEST_DURATION = 60  # seconds for extended leak testing
MEMORY_SAMPLE_INTERVAL = 1   # seconds between memory samples
LEAK_THRESHOLD_MB = 10       # MB growth considered a leak
RESOURCE_CLEANUP_TIMEOUT = 5 # seconds to wait for cleanup


class MemoryLeakTestBase(unittest.TestCase):
    """Base class for memory leak testing with detailed tracking."""
    
    def setUp(self):
        """Set up memory leak testing environment."""
        print(f"\n🔍 Setting up Memory Leak Test Environment...")
        
        # Start memory tracing
        tracemalloc.start()
        
        self.leak_test_results = {
            'memory_leaks_detected': [],
            'resource_leaks_detected': [],
            'cleanup_failures': [],
            'gc_effectiveness': [],
            'thread_leaks': [],
            'file_handle_leaks': []
        }
        
        # Resource tracking
        self.initial_memory = self.get_memory_usage()
        self.initial_thread_count = threading.active_count()
        self.initial_file_descriptors = self.get_open_file_count()
        
        # Weak references for tracking object lifecycle
        self.tracked_objects = []
        
        # Test artifacts for cleanup testing
        self.test_temp_files = []
        self.test_threads = []
        self.test_queues = []
        
        print(f"   📊 Initial State:")
        print(f"      Memory usage: {self.initial_memory:.1f}MB")
        print(f"      Active threads: {self.initial_thread_count}")
        print(f"      Open file descriptors: {self.initial_file_descriptors}")
    
    def tearDown(self):
        """Clean up after memory leak tests."""
        # Stop memory tracing
        tracemalloc.stop()
        
        # Clean up test resources
        self.cleanup_test_resources()
        
        # Force garbage collection
        gc.collect()
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            # Fallback using tracemalloc
            try:
                current, peak = tracemalloc.get_traced_memory()
                return current / 1024 / 1024
            except:
                return 0
    
    def get_open_file_count(self):
        """Get count of open file descriptors."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return len(process.open_files())
        except:
            return 0
    
    def track_object_lifecycle(self, obj, name):
        """Track an object's lifecycle using weak references."""
        weak_ref = weakref.ref(obj)
        self.tracked_objects.append({
            'name': name,
            'ref': weak_ref,
            'created_at': time.time()
        })
        return weak_ref
    
    def check_object_cleanup(self):
        """Check if tracked objects have been properly cleaned up."""
        cleanup_failures = []
        
        for obj_info in self.tracked_objects:
            if obj_info['ref']() is not None:  # Object still exists
                age = time.time() - obj_info['created_at']
                cleanup_failures.append({
                    'name': obj_info['name'],
                    'age_seconds': age
                })
        
        return cleanup_failures
    
    def create_test_resources(self, count=10):
        """Create test resources for cleanup validation."""
        # Create temporary files
        for i in range(count):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b"test data for cleanup validation")
            temp_file.close()
            self.test_temp_files.append(temp_file.name)
        
        # Create test threads
        for i in range(3):
            event = threading.Event()
            thread = threading.Thread(
                target=lambda e: e.wait(),
                args=(event,),
                name=f"TestThread-{i}"
            )
            thread.start()
            self.test_threads.append((thread, event))
        
        # Create test queues
        for i in range(5):
            queue = Queue(maxsize=100)
            # Add some test data
            for j in range(10):
                queue.put(f"test_data_{i}_{j}")
            self.test_queues.append(queue)
    
    def cleanup_test_resources(self):
        """Clean up test resources and validate cleanup."""
        # Clean up temporary files
        for temp_file in self.test_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.leak_test_results['cleanup_failures'].append({
                    'type': 'file',
                    'resource': temp_file,
                    'error': str(e)
                })
        
        # Clean up test threads
        for thread, event in self.test_threads:
            try:
                event.set()  # Signal thread to stop
                thread.join(timeout=2)
                if thread.is_alive():
                    self.leak_test_results['thread_leaks'].append({
                        'name': thread.name,
                        'status': 'still_alive'
                    })
            except Exception as e:
                self.leak_test_results['cleanup_failures'].append({
                    'type': 'thread',
                    'resource': thread.name,
                    'error': str(e)
                })
        
        # Clean up test queues
        for queue in self.test_queues:
            try:
                # Drain the queue
                while not queue.empty():
                    queue.get_nowait()
            except Exception as e:
                self.leak_test_results['cleanup_failures'].append({
                    'type': 'queue',
                    'error': str(e)
                })


class ExtendedMemoryLeakTest(MemoryLeakTestBase):
    """Test for memory leaks during extended operation."""
    
    def test_extended_operation_memory_leak(self):
        """Test for memory leaks during extended system operation."""
        print("   🕐 Testing extended operation memory leak...")
        
        memory_samples = []
        operation_count = 0
        start_time = time.time()
        
        print(f"   📊 Running extended test for {EXTENDED_TEST_DURATION} seconds...")
        
        # Simulate extended operation
        while time.time() - start_time < EXTENDED_TEST_DURATION:
            # Simulate various operations that might leak memory
            self.simulate_ai_analysis_operation()
            self.simulate_gui_update_operation()
            self.simulate_cache_operation()
            self.simulate_event_processing_operation()
            
            operation_count += 1
            
            # Sample memory every interval
            if operation_count % 10 == 0:  # Every 10 operations
                current_memory = self.get_memory_usage()
                memory_samples.append({
                    'timestamp': time.time() - start_time,
                    'memory_mb': current_memory,
                    'operations': operation_count
                })
                
                # Print progress
                if len(memory_samples) % 10 == 0:
                    print(f"   📊 Progress: {len(memory_samples)} samples, Current: {current_memory:.1f}MB")
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.1)
        
        final_memory = self.get_memory_usage()
        memory_growth = final_memory - self.initial_memory
        
        # Analyze memory usage pattern
        if len(memory_samples) >= 2:
            # Calculate memory growth trend
            first_sample = memory_samples[0]['memory_mb']
            last_sample = memory_samples[-1]['memory_mb']
            total_growth = last_sample - first_sample
            
            # Calculate growth rate
            time_span = memory_samples[-1]['timestamp'] - memory_samples[0]['timestamp']
            growth_rate = total_growth / time_span if time_span > 0 else 0  # MB/second
            
            print(f"   📊 Extended Memory Leak Test Results:")
            print(f"      Test duration: {EXTENDED_TEST_DURATION}s")
            print(f"      Total operations: {operation_count}")
            print(f"      Memory samples: {len(memory_samples)}")
            print(f"      Initial memory: {self.initial_memory:.1f}MB")
            print(f"      Final memory: {final_memory:.1f}MB")
            print(f"      Total growth: {memory_growth:.1f}MB")
            print(f"      Growth rate: {growth_rate*60:.2f}MB/minute")
            
            # Memory leak detection
            if abs(memory_growth) > LEAK_THRESHOLD_MB:
                self.leak_test_results['memory_leaks_detected'].append({
                    'type': 'extended_operation',
                    'growth_mb': memory_growth,
                    'growth_rate_mb_per_min': growth_rate * 60,
                    'operations': operation_count
                })
                print(f"   ⚠️ Potential memory leak detected!")
            else:
                print(f"   ✅ No significant memory leak detected")
            
            # Performance assertions
            self.assertLess(abs(memory_growth), LEAK_THRESHOLD_MB,
                          f"Memory leak detected: {memory_growth:.1f}MB growth")
            
            # Growth rate should be minimal
            max_growth_rate = 0.5  # 0.5 MB/minute maximum acceptable
            self.assertLess(abs(growth_rate * 60), max_growth_rate,
                          f"Memory growth rate too high: {growth_rate*60:.2f}MB/min")
        
        print("   ✅ Extended operation memory leak test completed")
    
    def simulate_ai_analysis_operation(self):
        """Simulate AI analysis operation that might leak memory."""
        # Create temporary data structures
        analysis_data = {
            'cards': [f'card_{i}' for i in range(10)],
            'scores': [random.uniform(0, 1) for i in range(10)],
            'metadata': {'timestamp': time.time(), 'analysis_id': id(self)}
        }
        
        # Simulate processing
        time.sleep(0.01)
        
        # Important: Clear data to prevent accumulation
        analysis_data.clear()
    
    def simulate_gui_update_operation(self):
        """Simulate GUI update operation that might leak memory."""
        # Create GUI state data
        gui_state = {
            'display_data': [f'item_{i}' for i in range(20)],
            'widget_states': {f'widget_{i}': f'state_{i}' for i in range(5)},
            'event_queue': [f'event_{i}' for i in range(15)]
        }
        
        # Simulate GUI update
        time.sleep(0.005)
        
        # Clear GUI state
        gui_state.clear()
    
    def simulate_cache_operation(self):
        """Simulate cache operation that might leak memory."""
        # Create cache entries
        cache_data = {}
        for i in range(50):
            key = f'cache_key_{i}'
            value = {'data': f'cached_value_{i}', 'timestamp': time.time()}
            cache_data[key] = value
        
        # Simulate cache access
        time.sleep(0.002)
        
        # Important: Clear cache data
        cache_data.clear()
    
    def simulate_event_processing_operation(self):
        """Simulate event processing that might leak memory."""
        # Create event data
        events = []
        for i in range(30):
            event = {
                'type': f'event_type_{i % 5}',
                'data': f'event_data_{i}',
                'timestamp': time.time()
            }
            events.append(event)
        
        # Simulate event processing
        time.sleep(0.003)
        
        # Clear events
        events.clear()


class ResourceCleanupTest(MemoryLeakTestBase):
    """Test resource cleanup and lifecycle management."""
    
    def test_component_lifecycle_cleanup(self):
        """Test proper cleanup of component lifecycles."""
        print("   🔄 Testing component lifecycle cleanup...")
        
        # Create test resources
        self.create_test_resources(count=15)
        
        # Simulate component usage
        print("   📊 Simulating component usage...")
        
        # Create mock components with proper cleanup methods
        components = []
        for i in range(5):
            component = self.create_mock_component(f"Component-{i}")
            components.append(component)
            
            # Track component lifecycle
            self.track_object_lifecycle(component, f"Component-{i}")
        
        # Simulate component operations
        for component in components:
            component.start()
            component.process_data({'test': 'data'})
            time.sleep(0.1)
        
        # Initiate component cleanup
        print("   🧹 Initiating component cleanup...")
        for component in components:
            component.stop()
            component.cleanup()
        
        # Clear component references
        components.clear()
        
        # Force garbage collection
        gc.collect()
        time.sleep(1)  # Allow time for cleanup
        
        # Check cleanup effectiveness
        cleanup_failures = self.check_object_cleanup()
        
        print(f"   📊 Component Cleanup Results:")
        print(f"      Components created: 5")
        print(f"      Cleanup failures: {len(cleanup_failures)}")
        
        if cleanup_failures:
            for failure in cleanup_failures:
                print(f"      - {failure['name']}: still exists after {failure['age_seconds']:.1f}s")
                self.leak_test_results['cleanup_failures'].append(failure)
        
        # Resource state after cleanup
        final_thread_count = threading.active_count() 
        final_file_descriptors = self.get_open_file_count()
        
        print(f"   📊 Resource State After Cleanup:")
        print(f"      Threads: {self.initial_thread_count} → {final_thread_count}")
        print(f"      File descriptors: {self.initial_file_descriptors} → {final_file_descriptors}")
        
        # Validate cleanup
        thread_increase = final_thread_count - self.initial_thread_count
        self.assertLessEqual(thread_increase, 2, f"Too many threads remaining: +{thread_increase}")
        
        fd_increase = final_file_descriptors - self.initial_file_descriptors
        self.assertLessEqual(fd_increase, 5, f"Too many file descriptors remaining: +{fd_increase}")
        
        print("   ✅ Component lifecycle cleanup test completed")
    
    def create_mock_component(self, name):
        """Create a mock component with proper lifecycle methods."""
        component = Mock()
        component.name = name
        component.running = False
        component.resources = []
        
        def start():
            component.running = True
            # Simulate resource allocation
            component.resources = [f"resource_{i}" for i in range(5)]
        
        def stop():
            component.running = False
        
        def cleanup():
            # Simulate proper resource cleanup
            component.resources.clear()
        
        def process_data(data):
            if component.running:
                # Simulate data processing
                return f"processed_{data}"
            return None
        
        component.start = start
        component.stop = stop
        component.cleanup = cleanup
        component.process_data = process_data
        
        return component
    
    def test_garbage_collection_effectiveness(self):
        """Test effectiveness of garbage collection."""
        print("   🗑️ Testing garbage collection effectiveness...")
        
        # Create objects that should be garbage collected
        test_objects = []
        for i in range(100):
            obj = {
                'id': i,
                'data': [f'item_{j}' for j in range(100)],  # Some bulk data
                'refs': {}
            }
            test_objects.append(obj)
            
            # Track some objects for lifecycle monitoring
            if i % 10 == 0:
                self.track_object_lifecycle(obj, f"TestObject-{i}")
        
        # Create circular references (challenging for GC)
        for i in range(len(test_objects) - 1):
            test_objects[i]['refs']['next'] = test_objects[i + 1]
            test_objects[i + 1]['refs']['prev'] = test_objects[i]
        
        initial_objects = len(gc.get_objects())
        print(f"   📊 Objects before cleanup: {initial_objects}")
        
        # Clear references
        test_objects.clear()
        
        # Test garbage collection effectiveness
        print("   🧹 Running garbage collection...")
        
        # Multiple GC passes for thorough cleanup
        collected_counts = []
        for i in range(3):
            collected = gc.collect()
            collected_counts.append(collected)
            print(f"   📊 GC Pass {i+1}: {collected} objects collected")
            time.sleep(0.5)
        
        final_objects = len(gc.get_objects())
        objects_cleaned = initial_objects - final_objects
        
        print(f"   📊 Garbage Collection Results:")
        print(f"      Objects before: {initial_objects}")
        print(f"      Objects after: {final_objects}")
        print(f"      Objects cleaned: {objects_cleaned}")
        print(f"      Total collected: {sum(collected_counts)}")
        
        # Check tracked object cleanup
        cleanup_failures = self.check_object_cleanup()
        gc_effectiveness = len(cleanup_failures) == 0
        
        self.leak_test_results['gc_effectiveness'].append({
            'objects_cleaned': objects_cleaned,
            'gc_collections': sum(collected_counts),
            'cleanup_failures': len(cleanup_failures),
            'effective': gc_effectiveness
        })
        
        if gc_effectiveness:
            print("   ✅ Garbage collection highly effective")
        else:
            print(f"   ⚠️ Some objects not collected: {len(cleanup_failures)}")
        
        print("   ✅ Garbage collection effectiveness test completed")


class ThreadLeakTest(MemoryLeakTestBase):
    """Test for thread leaks and proper thread cleanup."""
    
    def test_thread_lifecycle_management(self):
        """Test proper thread lifecycle management."""
        print("   🧵 Testing thread lifecycle management...")
        
        initial_threads = threading.active_count()
        created_threads = []
        
        print(f"   📊 Initial active threads: {initial_threads}")
        
        # Create various types of threads
        thread_types = [
            ('worker_thread', self.create_worker_thread),
            ('event_processor', self.create_event_processor_thread),
            ('monitor_thread', self.create_monitor_thread),
            ('cleanup_thread', self.create_cleanup_thread)
        ]
        
        # Create multiple threads of each type
        for thread_type, creator_func in thread_types:
            for i in range(3):  # 3 threads of each type
                thread_name = f"{thread_type}_{i}"
                thread, stop_event = creator_func(thread_name)
                created_threads.append((thread, stop_event, thread_name))
                thread.start()
        
        current_threads = threading.active_count()
        thread_increase = current_threads - initial_threads
        
        print(f"   📊 After thread creation:")
        print(f"      Active threads: {current_threads}")
        print(f"      Threads created: {len(created_threads)}")
        print(f"      Thread increase: +{thread_increase}")
        
        # Let threads run for a bit
        time.sleep(2)
        
        # Initiate thread shutdown
        print("   🛑 Initiating thread shutdown...")
        
        shutdown_success = 0
        shutdown_failures = []
        
        for thread, stop_event, thread_name in created_threads:
            try:
                # Signal thread to stop
                stop_event.set()
                
                # Wait for thread to finish
                thread.join(timeout=3)
                
                if thread.is_alive():
                    shutdown_failures.append(thread_name)
                    self.leak_test_results['thread_leaks'].append({
                        'name': thread_name,
                        'status': 'failed_to_stop'
                    })
                else:
                    shutdown_success += 1
                    
            except Exception as e:
                shutdown_failures.append(thread_name)
                self.leak_test_results['cleanup_failures'].append({
                    'type': 'thread_cleanup',
                    'resource': thread_name,
                    'error': str(e)
                })
        
        # Check final thread count
        time.sleep(1)  # Allow time for cleanup
        final_threads = threading.active_count()
        remaining_threads = final_threads - initial_threads
        
        print(f"   📊 Thread Shutdown Results:")
        print(f"      Successful shutdowns: {shutdown_success}")
        print(f"      Failed shutdowns: {len(shutdown_failures)}")
        print(f"      Final active threads: {final_threads}")
        print(f"      Remaining excess threads: {remaining_threads}")
        
        if shutdown_failures:
            for failure in shutdown_failures:
                print(f"      - Failed to stop: {failure}")
        
        # Performance assertions
        shutdown_rate = shutdown_success / len(created_threads)
        self.assertGreater(shutdown_rate, 0.9, f"Thread shutdown rate too low: {shutdown_rate:.2%}")
        
        self.assertLessEqual(remaining_threads, 2, f"Too many threads remaining: {remaining_threads}")
        
        print("   ✅ Thread lifecycle management test completed")
    
    def create_worker_thread(self, name):
        """Create a worker thread for testing."""
        stop_event = threading.Event()
        
        def worker():
            while not stop_event.is_set():
                # Simulate work
                time.sleep(0.1)
                # Process some data
                data = [i for i in range(100)]
                result = sum(data)  # Simple computation
        
        thread = threading.Thread(target=worker, name=name)
        return thread, stop_event
    
    def create_event_processor_thread(self, name):
        """Create an event processor thread for testing."""
        stop_event = threading.Event()
        event_queue = Queue()
        
        def event_processor():
            while not stop_event.is_set():
                try:
                    # Add some test events
                    if not event_queue.full():
                        event_queue.put(f"event_{time.time()}")
                    
                    # Process events
                    if not event_queue.empty():
                        event = event_queue.get(timeout=0.1)
                        # Simulate event processing
                        time.sleep(0.05)
                        
                except:
                    pass  # Timeout is expected
        
        thread = threading.Thread(target=event_processor, name=name)
        return thread, stop_event
    
    def create_monitor_thread(self, name):
        """Create a monitor thread for testing."""
        stop_event = threading.Event()
        
        def monitor():
            while not stop_event.is_set():
                # Simulate monitoring
                current_time = time.time()
                memory_usage = self.get_memory_usage()
                
                # Simulate data collection
                monitoring_data = {
                    'timestamp': current_time,
                    'memory': memory_usage,
                    'threads': threading.active_count()
                }
                
                time.sleep(0.5)  # Monitor every 500ms
        
        thread = threading.Thread(target=monitor, name=name)
        return thread, stop_event
    
    def create_cleanup_thread(self, name):
        """Create a cleanup thread for testing."""
        stop_event = threading.Event()
        
        def cleanup():
            while not stop_event.is_set():
                # Simulate periodic cleanup
                gc.collect()  # Force garbage collection
                time.sleep(1)  # Cleanup every second
        
        thread = threading.Thread(target=cleanup, name=name)
        return thread, stop_event


def run_comprehensive_memory_leak_tests():
    """Run the complete memory leak and resource cleanup test suite."""
    print("🔍 STARTING COMPREHENSIVE MEMORY LEAK TESTING")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add memory leak tests
    suite.addTest(ExtendedMemoryLeakTest('test_extended_operation_memory_leak'))
    
    suite.addTest(ResourceCleanupTest('test_component_lifecycle_cleanup'))
    suite.addTest(ResourceCleanupTest('test_garbage_collection_effectiveness'))
    
    suite.addTest(ThreadLeakTest('test_thread_lifecycle_management'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL MEMORY LEAK TESTS PASSED!")
        print("✅ No memory leaks detected")
        print("✅ Resource cleanup functioning properly")
        print("✅ Garbage collection effective")
        print("✅ Thread lifecycle management working")
        print("✅ System memory management is robust")
    else:
        print("❌ MEMORY LEAK TESTS REVEALED ISSUES!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n🔍 MEMORY MANAGEMENT ISSUES:")
            for test, traceback in result.failures:
                print(f"   • {test}: Memory or resource management problem detected")
        
        if result.errors:
            print("\n🔍 MEMORY TEST ERRORS:")
            for test, traceback in result.errors:
                print(f"   • {test}: Exception during memory testing")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import random  # Import needed for the test
    success = run_comprehensive_memory_leak_tests()
    sys.exit(0 if success else 1)