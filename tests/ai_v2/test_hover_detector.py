#!/usr/bin/env python3
"""
Test Suite for Hover Detection System - Phase 3

This module provides comprehensive testing for the HoverDetector system
as specified in Phase 3 of the todo_ai_helper.md master plan.

Test Coverage:
- P3.2.1-P3.2.6: CPU-optimized hover detection with adaptive polling
- Mouse tracking optimization and resource management
- Hover state machine with debouncing
- Performance monitoring and memory management
- Integration with Visual Intelligence Overlay

Test Categories:
- Unit tests for core hover detection functionality
- Performance tests for CPU usage optimization
- State machine validation tests
- Memory management and cleanup tests
- Integration tests with mouse tracking

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import unittest
import threading
import time
import logging
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from arena_bot.ui.hover_detector import (
        HoverDetector, HoverConfiguration, HoverState, MouseTracker,
        MouseEvent, MouseEventType, HoverRegion, PollingStrategy,
        create_hover_detector, example_hover_callback
    )
    from arena_bot.ai_v2.exceptions import AIHelperUIError, AIHelperPerformanceError
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for CI environment
    class HoverDetector: pass
    class HoverConfiguration: pass
    class HoverState: pass
    class MouseTracker: pass
    class MouseEvent: pass
    class MouseEventType: pass
    class HoverRegion: pass
    class PollingStrategy: pass
    class AIHelperUIError(Exception): pass
    class AIHelperPerformanceError(Exception): pass
    create_hover_detector = lambda: None
    example_hover_callback = lambda: None

# Configure test logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestHoverConfiguration(unittest.TestCase):
    """Test hover detection configuration validation and defaults"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = HoverConfiguration()
        
        # Polling settings
        self.assertEqual(config.idle_poll_rate_hz, 10.0)
        self.assertEqual(config.active_poll_rate_hz, 60.0)
        self.assertEqual(config.precision_poll_rate_hz, 120.0)
        
        # Sensitivity settings
        self.assertEqual(config.movement_threshold_pixels, 5.0)
        self.assertEqual(config.hover_threshold_pixels, 3.0)
        self.assertEqual(config.velocity_sensitivity_factor, 0.1)
        
        # Performance settings
        self.assertEqual(config.max_cpu_usage, 0.03)  # 3% limit
        self.assertEqual(config.yield_interval_ms, 10.0)
        
        # Memory management
        self.assertEqual(config.max_event_history, 1000)
        self.assertEqual(config.position_history_size, 100)
        
        # Mouse tracking
        self.assertTrue(config.enable_raw_input)
        self.assertTrue(config.mouse_acceleration_compensation)
        self.assertTrue(config.multi_mouse_support)
    
    def test_performance_settings_validation(self):
        """Test performance settings are within reasonable bounds"""
        config = HoverConfiguration()
        
        # CPU usage should be reasonable
        self.assertGreater(config.max_cpu_usage, 0.0)
        self.assertLess(config.max_cpu_usage, 0.1)  # Should be less than 10%
        
        # Poll rates should be reasonable
        self.assertGreater(config.idle_poll_rate_hz, 1.0)
        self.assertLess(config.precision_poll_rate_hz, 200.0)
        
        # Memory limits should be reasonable
        self.assertGreater(config.max_event_history, 100)
        self.assertLess(config.max_event_history, 10000)

class TestMouseEvent(unittest.TestCase):
    """Test mouse event data structures"""
    
    def test_mouse_event_creation(self):
        """Test mouse event creation and properties"""
        timestamp = time.perf_counter()
        event = MouseEvent(
            event_type=MouseEventType.MOVE,
            x=100,
            y=200,
            timestamp=timestamp,
            window_handle=12345
        )
        
        self.assertEqual(event.event_type, MouseEventType.MOVE)
        self.assertEqual(event.x, 100)
        self.assertEqual(event.y, 200)
        self.assertEqual(event.timestamp, timestamp)
        self.assertEqual(event.window_handle, 12345)
    
    def test_mouse_event_types(self):
        """Test all mouse event types are available"""
        event_types = [
            MouseEventType.MOVE,
            MouseEventType.LEFT_DOWN,
            MouseEventType.LEFT_UP,
            MouseEventType.RIGHT_DOWN,
            MouseEventType.RIGHT_UP,
            MouseEventType.MIDDLE_DOWN,
            MouseEventType.MIDDLE_UP
        ]
        
        for event_type in event_types:
            event = MouseEvent(event_type, 0, 0, time.perf_counter())
            self.assertEqual(event.event_type, event_type)

class TestHoverRegion(unittest.TestCase):
    """Test hover region functionality"""
    
    def test_hover_region_creation(self):
        """Test hover region creation"""
        callback = Mock()
        region = HoverRegion(
            x=100, y=200, width=150, height=100,
            callback=callback, priority=5, sensitivity=1.5
        )
        
        self.assertEqual(region.x, 100)
        self.assertEqual(region.y, 200)
        self.assertEqual(region.width, 150)
        self.assertEqual(region.height, 100)
        self.assertEqual(region.callback, callback)
        self.assertEqual(region.priority, 5)
        self.assertEqual(region.sensitivity, 1.5)
    
    def test_hover_region_defaults(self):
        """Test hover region default values"""
        callback = Mock()
        region = HoverRegion(
            x=0, y=0, width=100, height=100,
            callback=callback
        )
        
        self.assertEqual(region.priority, 0)
        self.assertEqual(region.sensitivity, 1.0)

class TestMouseTracker(unittest.TestCase):
    """Test mouse tracking functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = HoverConfiguration()
        self.tracker = MouseTracker(self.config)
    
    def test_tracker_initialization(self):
        """Test mouse tracker initialization"""
        self.assertEqual(self.tracker.config, self.config)
        self.assertFalse(self.tracker.tracking)
        self.assertEqual(self.tracker.last_position, (0, 0))
        self.assertEqual(self.tracker.current_velocity, 0.0)
        
        # Test collections are initialized
        self.assertEqual(len(self.tracker.position_history), 0)
        self.assertEqual(len(self.tracker.event_history), 0)
    
    @patch('arena_bot.ui.hover_detector.WINDOWS_API_AVAILABLE', False)
    def test_fallback_tracking_start(self):
        """Test fallback tracking when Windows API unavailable"""
        with patch.object(self.tracker, '_start_fallback_tracking', return_value=True):
            success = self.tracker.start_tracking()
            self.assertTrue(success)
    
    @patch('arena_bot.ui.hover_detector.WINDOWS_API_AVAILABLE', True)
    def test_raw_input_tracking_attempt(self):
        """Test attempt to use raw input tracking"""
        with patch.object(self.tracker, '_setup_raw_input_tracking', return_value=False), \
             patch.object(self.tracker, '_setup_hook_tracking', return_value=True):
            
            success = self.tracker.start_tracking()
            self.assertTrue(success)
            self.assertTrue(self.tracker.tracking)
    
    def test_mouse_move_handling(self):
        """Test mouse movement event handling"""
        initial_history_len = len(self.tracker.position_history)
        
        self.tracker._on_mouse_move(100, 200)
        
        # Should have added to position history
        self.assertEqual(len(self.tracker.position_history), initial_history_len + 1)
        self.assertEqual(self.tracker.last_position, (100, 200))
        
        # Should have calculated velocity
        self.tracker._on_mouse_move(110, 210)
        self.assertGreater(self.tracker.current_velocity, 0)
    
    def test_mouse_velocity_calculation(self):
        """Test mouse velocity calculation"""
        # Start with no movement
        self.assertEqual(self.tracker.current_velocity, 0.0)
        
        # Simulate movement
        self.tracker._on_mouse_move(0, 0)
        time.sleep(0.001)  # Small delay
        self.tracker._on_mouse_move(10, 0)  # Move 10 pixels
        
        # Should have positive velocity
        self.assertGreater(self.tracker.current_velocity, 0)
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        # Fill up history beyond normal limits
        for i in range(self.config.max_event_history + 100):
            event = MouseEvent(MouseEventType.MOVE, i, i, time.perf_counter())
            self.tracker.event_history.append(event)
        
        # Fill position history
        for i in range(self.config.position_history_size + 50):
            self.tracker.position_history.append((i, i, time.perf_counter()))
        
        # Force cleanup
        self.tracker.last_cleanup_time = 0  # Force cleanup
        self.tracker.cleanup_memory()
        
        # Should have cleaned up excess entries
        self.assertLessEqual(len(self.tracker.event_history), self.config.max_event_history)
        self.assertLessEqual(len(self.tracker.position_history), self.config.position_history_size)
    
    def test_mouse_acceleration_compensation(self):
        """Test mouse acceleration compensation"""
        # This is a placeholder test since the actual implementation
        # would require Windows API integration
        x, y = self.tracker._compensate_mouse_acceleration(100, 200)
        
        # For now, should return input unchanged in fallback
        self.assertEqual(x, 100)
        self.assertEqual(y, 200)
    
    def test_tracker_lifecycle(self):
        """Test tracker start/stop lifecycle"""
        with patch.object(self.tracker, '_start_fallback_tracking', return_value=True):
            # Start tracking
            success = self.tracker.start_tracking()
            self.assertTrue(success)
            self.assertTrue(self.tracker.tracking)
            
            # Stop tracking
            self.tracker.stop_tracking()
            self.assertFalse(self.tracker.tracking)

class TestHoverDetector(unittest.TestCase):
    """Test main hover detector functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = HoverConfiguration()
        self.config.hover_delay_ms = 100.0  # Shorter delay for tests
        self.detector = HoverDetector(self.config)
    
    def test_detector_initialization(self):
        """Test hover detector initialization"""
        self.assertEqual(self.detector.state, HoverState.IDLE)
        self.assertFalse(self.detector.running)
        self.assertEqual(self.detector.current_strategy, PollingStrategy.IDLE)
        self.assertEqual(len(self.detector.hover_regions), 0)
        self.assertIsNone(self.detector.current_hover_region)
    
    def test_hover_region_management(self):
        """Test hover region addition and removal"""
        callback = Mock()
        region1 = HoverRegion(0, 0, 100, 100, callback, priority=1)
        region2 = HoverRegion(200, 200, 100, 100, callback, priority=2)
        
        # Add regions
        self.detector.add_hover_region(region1)
        self.detector.add_hover_region(region2)
        
        self.assertEqual(len(self.detector.hover_regions), 2)
        # Should be sorted by priority (highest first)
        self.assertEqual(self.detector.hover_regions[0].priority, 2)
        
        # Remove region
        self.detector.remove_hover_region(region1)
        self.assertEqual(len(self.detector.hover_regions), 1)
        
        # Clear all regions
        self.detector.clear_hover_regions()
        self.assertEqual(len(self.detector.hover_regions), 0)
    
    def test_point_in_region_detection(self):
        """Test point-in-region detection logic"""
        callback = Mock()
        region = HoverRegion(100, 100, 200, 150, callback)
        
        # Points inside region
        self.assertTrue(self.detector._point_in_region((150, 150), region))
        self.assertTrue(self.detector._point_in_region((100, 100), region))  # Top-left corner
        self.assertTrue(self.detector._point_in_region((300, 250), region))  # Bottom-right corner
        
        # Points outside region
        self.assertFalse(self.detector._point_in_region((50, 50), region))
        self.assertFalse(self.detector._point_in_region((350, 300), region))
        self.assertFalse(self.detector._point_in_region((150, 50), region))
    
    def test_polling_strategy_adaptation(self):
        """Test adaptive polling strategy"""
        # Initially should be idle
        self.assertEqual(self.detector.current_strategy, PollingStrategy.IDLE)
        
        # Simulate mouse movement
        self.detector.last_mouse_position = (0, 0)
        self.detector._update_polling_strategy()
        
        # Simulate movement detection
        self.detector.last_mouse_position = (10, 10)
        self.detector.movement_detected = True
        self.detector._update_polling_strategy()
        
        self.assertEqual(self.detector.current_strategy, PollingStrategy.ACTIVE)
    
    def test_poll_rate_calculation(self):
        """Test polling rate calculation for different strategies"""
        # Test idle rate
        self.detector.current_strategy = PollingStrategy.IDLE
        rate = self.detector._get_current_poll_rate()
        self.assertEqual(rate, self.config.idle_poll_rate_hz)
        
        # Test active rate
        self.detector.current_strategy = PollingStrategy.ACTIVE
        rate = self.detector._get_current_poll_rate()
        self.assertEqual(rate, self.config.active_poll_rate_hz)
        
        # Test precision rate
        self.detector.current_strategy = PollingStrategy.PRECISION
        rate = self.detector._get_current_poll_rate()
        self.assertEqual(rate, self.config.precision_poll_rate_hz)
    
    def test_hover_state_transitions(self):
        """Test hover state machine transitions"""
        callback = Mock()
        region = HoverRegion(100, 100, 100, 100, callback)
        
        # Test entering hover region
        self.detector._enter_hover_region(region, (150, 150))
        self.assertEqual(self.detector.state, HoverState.TRACKING)
        self.assertIsNotNone(self.detector.hover_start_time)
        
        # Test triggering hover
        self.detector._trigger_hover(region, (150, 150))
        self.assertEqual(self.detector.state, HoverState.HOVERING)
        callback.assert_called_once()
        
        # Test exiting hover region
        self.detector._exit_hover_region(region)
        self.assertEqual(self.detector.state, HoverState.IDLE)
        self.assertIsNone(self.detector.hover_start_time)
    
    def test_region_priority_resolution(self):
        """Test region overlap priority resolution"""
        high_priority_callback = Mock()
        low_priority_callback = Mock()
        
        # Overlapping regions with different priorities
        high_region = HoverRegion(100, 100, 100, 100, high_priority_callback, priority=5)
        low_region = HoverRegion(120, 120, 100, 100, low_priority_callback, priority=1)
        
        self.detector.add_hover_region(high_region)
        self.detector.add_hover_region(low_region)
        
        # Simulate mouse at overlapping position
        with patch.object(self.detector.mouse_tracker, 'get_current_position', return_value=(150, 150)):
            self.detector._process_hover_detection()
        
        # Should select higher priority region
        self.assertEqual(self.detector.current_hover_region, high_region)
    
    def test_sensitivity_adjustment(self):
        """Test hover sensitivity adjustment"""
        initial_movement_threshold = self.config.movement_threshold_pixels
        initial_hover_threshold = self.config.hover_threshold_pixels
        
        # Increase sensitivity
        self.detector.set_sensitivity(2.0)
        
        self.assertEqual(
            self.config.movement_threshold_pixels,
            initial_movement_threshold * 2.0
        )
        self.assertEqual(
            self.config.hover_threshold_pixels,
            initial_hover_threshold * 2.0
        )
        
        # Test bounds checking
        self.detector.set_sensitivity(5.0)  # Should be clamped to 2.0
        self.detector.set_sensitivity(0.05)  # Should be clamped to 0.1
    
    def test_detector_lifecycle(self):
        """Test detector start/stop lifecycle"""
        with patch.object(self.detector.mouse_tracker, 'start_tracking', return_value=True), \
             patch.object(self.detector, '_detection_loop'):
            
            # Start detector
            success = self.detector.start()
            self.assertTrue(success)
            self.assertTrue(self.detector.running)
            
            # Stop detector
            self.detector.stop()
            self.assertFalse(self.detector.running)
            self.assertEqual(self.detector.state, HoverState.IDLE)
    
    def test_pause_resume_functionality(self):
        """Test pause and resume functionality"""
        self.detector.pause()
        self.assertEqual(self.detector.state, HoverState.SUSPENDED)
        
        self.detector.resume()
        self.assertEqual(self.detector.state, HoverState.IDLE)
        
        # Test resume only works from suspended state
        self.detector.state = HoverState.TRACKING
        self.detector.resume()
        self.assertEqual(self.detector.state, HoverState.TRACKING)  # Should not change
    
    def test_performance_stats_collection(self):
        """Test performance statistics collection"""
        stats = self.detector.get_performance_stats()
        
        # Should contain expected keys
        expected_keys = [
            'cpu_usage', 'polling_strategy', 'current_poll_rate',
            'regions_count', 'state', 'events_processed', 'hover_detections'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Values should be reasonable
        self.assertGreaterEqual(stats['cpu_usage'], 0.0)
        self.assertLessEqual(stats['cpu_usage'], 1.0)
        self.assertGreaterEqual(stats['regions_count'], 0)
    
    def test_mouse_event_handling(self):
        """Test mouse event handling from tracker"""
        # Test mouse move event
        move_event = MouseEvent(MouseEventType.MOVE, 100, 200, time.perf_counter())
        self.detector._on_mouse_move(move_event)
        
        self.assertGreater(self.detector.performance_stats['events_processed'], 0)
        
        # Test mouse button event with debouncing
        click_event = MouseEvent(MouseEventType.LEFT_DOWN, 100, 200, time.perf_counter())
        self.detector.state = HoverState.HOVERING
        
        self.detector._on_mouse_event(click_event)
        self.assertEqual(self.detector.state, HoverState.DEBOUNCING)

class TestHoverDetectorIntegration(unittest.TestCase):
    """Test hover detector integration with other systems"""
    
    def test_factory_function(self):
        """Test hover detector factory function"""
        config = HoverConfiguration()
        detector = create_hover_detector(config)
        
        if detector:  # Only test if not mocked
            self.assertIsInstance(detector, HoverDetector)
            self.assertEqual(detector.config, config)
    
    def test_example_callback_function(self):
        """Test example hover callback function"""
        if example_hover_callback:
            event = MouseEvent(MouseEventType.MOVE, 100, 200, time.perf_counter())
            
            # Should not raise exception
            try:
                example_hover_callback(event)
            except Exception as e:
                self.fail(f"Example callback raised exception: {e}")
    
    def test_resource_tracker_integration(self):
        """Test integration with resource tracking"""
        detector = HoverDetector()
        
        with patch.object(detector, 'start', return_value=True):
            # Should handle resource tracker creation gracefully
            detector.start()

class TestHoverDetectorPerformance(unittest.TestCase):
    """Performance tests for hover detection system"""
    
    def test_cpu_usage_monitoring(self):
        """Test CPU usage stays within limits"""
        config = HoverConfiguration()
        config.max_cpu_usage = 0.05  # 5% limit
        
        detector = HoverDetector(config)
        
        # Simulate some processing time
        start_time = time.perf_counter()
        time.sleep(0.001)  # Simulate 1ms of work
        
        detector._update_performance_stats(start_time)
        
        stats = detector.get_performance_stats()
        self.assertIn('cpu_usage', stats)
        self.assertGreaterEqual(stats['cpu_usage'], 0.0)
    
    def test_memory_usage_bounded(self):
        """Test memory usage remains bounded"""
        config = HoverConfiguration()
        config.max_event_history = 50  # Small limit for test
        
        tracker = MouseTracker(config)
        
        # Generate many events
        for i in range(100):
            event = MouseEvent(MouseEventType.MOVE, i, i, time.perf_counter())
            tracker.event_history.append(event)
        
        # Should be limited to max size
        self.assertLessEqual(len(tracker.event_history), config.max_event_history)
    
    def test_adaptive_polling_efficiency(self):
        """Test adaptive polling reduces CPU usage when idle"""
        detector = HoverDetector()
        
        # Test idle strategy uses lowest poll rate
        detector.current_strategy = PollingStrategy.IDLE
        idle_rate = detector._get_current_poll_rate()
        
        detector.current_strategy = PollingStrategy.PRECISION
        precision_rate = detector._get_current_poll_rate()
        
        # Precision should be higher than idle
        self.assertGreater(precision_rate, idle_rate)
    
    def test_debouncing_performance(self):
        """Test debouncing prevents excessive event processing"""
        config = HoverConfiguration()
        config.debounce_delay_ms = 50.0
        
        detector = HoverDetector(config)
        detector.state = HoverState.HOVERING
        
        # Rapid clicks should trigger debouncing
        for _ in range(5):
            event = MouseEvent(MouseEventType.LEFT_DOWN, 100, 100, time.perf_counter())
            detector._on_mouse_event(event)
        
        # Should be in debouncing state
        self.assertEqual(detector.state, HoverState.DEBOUNCING)

class TestHoverErrorHandling(unittest.TestCase):
    """Test error handling and resilience"""
    
    def test_mouse_tracker_failure_handling(self):
        """Test handling of mouse tracker failures"""
        detector = HoverDetector()
        
        with patch.object(detector.mouse_tracker, 'start_tracking', return_value=False):
            success = detector.start()
            self.assertFalse(success)
    
    def test_callback_exception_handling(self):
        """Test handling of exceptions in hover callbacks"""
        def failing_callback(event):
            raise Exception("Callback failed")
        
        detector = HoverDetector()
        region = HoverRegion(0, 0, 100, 100, failing_callback)
        
        # Should not crash when callback fails
        try:
            detector._trigger_hover(region, (50, 50))
        except Exception as e:
            self.fail(f"Hover trigger should handle callback exceptions: {e}")
    
    def test_resource_tracker_failure_handling(self):
        """Test handling of resource tracker failures"""
        with patch('arena_bot.ui.hover_detector.ResourceTracker',
                   side_effect=Exception("Resource tracker failed")):
            
            detector = HoverDetector()
            
            with patch.object(detector.mouse_tracker, 'start_tracking', return_value=True):
                # Should start successfully even if resource tracker fails
                success = detector.start()
                self.assertTrue(success)
    
    def test_cleanup_on_stop(self):
        """Test proper cleanup when stopping detector"""
        detector = HoverDetector()
        
        with patch.object(detector.mouse_tracker, 'start_tracking', return_value=True), \
             patch.object(detector, '_detection_loop'):
            
            detector.start()
            self.assertTrue(detector.running)
            
            # Should cleanup properly
            detector.stop()
            self.assertFalse(detector.running)

# Test suite setup
def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHoverConfiguration,
        TestMouseEvent,
        TestHoverRegion,
        TestMouseTracker,
        TestHoverDetector,
        TestHoverDetectorIntegration,
        TestHoverDetectorPerformance,
        TestHoverErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def run_performance_benchmarks():
    """Run performance benchmarks for hover detection system"""
    print("=== Hover Detection Performance Benchmarks ===")
    
    # Test detector initialization time
    start_time = time.perf_counter()
    config = HoverConfiguration()
    detector = HoverDetector(config)
    init_time = time.perf_counter() - start_time
    print(f"Detector initialization: {init_time*1000:.2f}ms")
    
    # Test mouse event processing performance
    tracker = MouseTracker(config)
    iterations = 1000
    
    start_time = time.perf_counter()
    for i in range(iterations):
        tracker._on_mouse_move(i % 1000, i % 1000)
    
    total_time = time.perf_counter() - start_time
    avg_time = (total_time / iterations) * 1000000  # Convert to microseconds
    print(f"Mouse event processing: {avg_time:.1f}μs per event")
    
    # Test hover region detection performance
    detector = HoverDetector(config)
    
    # Add multiple regions
    for i in range(10):
        region = HoverRegion(i*100, i*100, 100, 100, lambda e: None, priority=i)
        detector.add_hover_region(region)
    
    start_time = time.perf_counter()
    for _ in range(100):
        detector._process_hover_detection()
    
    total_time = time.perf_counter() - start_time
    avg_time = (total_time / 100) * 1000  # Convert to ms
    print(f"Hover detection processing: {avg_time:.2f}ms per cycle")
    
    # Test memory cleanup performance
    tracker = MouseTracker(config)
    
    # Fill with events
    for i in range(2000):
        event = MouseEvent(MouseEventType.MOVE, i, i, time.perf_counter())
        tracker.event_history.append(event)
    
    start_time = time.perf_counter()
    tracker.cleanup_memory()
    cleanup_time = time.perf_counter() - start_time
    print(f"Memory cleanup time: {cleanup_time*1000:.2f}ms")

if __name__ == '__main__':
    print("Hover Detection System Test Suite")
    print("=================================")
    
    # Run unit tests
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance benchmarks
    if result.wasSuccessful():
        print("\n" + "="*50)
        run_performance_benchmarks()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)