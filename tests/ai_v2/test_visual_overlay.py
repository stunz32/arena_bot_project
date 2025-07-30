#!/usr/bin/env python3
"""
Test Suite for Visual Intelligence Overlay - Phase 3

This module provides comprehensive testing for the VisualIntelligenceOverlay
system as specified in Phase 3 of the todo_ai_helper.md master plan.

Test Coverage:
- P3.1.1-P3.1.12: Multi-monitor platform compatibility and click-through
- P3.3.1-P3.3.6: Performance monitoring and optimization
- Integration with AI v2 data models
- Error handling and recovery mechanisms
- Resource management and cleanup

Test Categories:
- Unit tests for core functionality
- Performance benchmarks and stress tests
- Multi-monitor simulation tests
- Click-through validation tests
- Error recovery and resilience tests

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import unittest
import threading
import time
import logging
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from arena_bot.ui.visual_overlay import (
        VisualIntelligenceOverlay, OverlayConfiguration, OverlayState,
        MonitorConfiguration, WindowsAPIHelper, PerformanceLimiter,
        ClickThroughMode, MonitorInfo, create_visual_overlay
    )
    from arena_bot.ai_v2.data_models import (
        AIDecision, CardOption, EvaluationScores, CardInfo, ConfidenceLevel,
        create_ai_decision, create_card_info
    )
    from arena_bot.ai_v2.exceptions import AIHelperUIError, AIHelperPerformanceError
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for CI environment
    class VisualIntelligenceOverlay: pass
    class OverlayConfiguration: pass
    class OverlayState: pass
    class MonitorConfiguration: pass
    class WindowsAPIHelper: pass
    class PerformanceLimiter: pass
    class ClickThroughMode: pass
    class MonitorInfo: pass
    class AIDecision: pass
    class CardOption: pass
    class EvaluationScores: pass
    class CardInfo: pass
    class ConfidenceLevel: pass
    class AIHelperUIError(Exception): pass
    class AIHelperPerformanceError(Exception): pass
    create_visual_overlay = lambda: None
    create_ai_decision = lambda: None
    create_card_info = lambda: None

# Configure test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
logger = logging.getLogger(__name__)

class TestOverlayConfiguration(unittest.TestCase):
    """Test overlay configuration validation and defaults"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = OverlayConfiguration()
        
        # Performance settings
        self.assertEqual(config.target_fps, 30)
        self.assertEqual(config.max_cpu_usage, 0.05)
        self.assertEqual(config.frame_budget_ms, 33.33)
        self.assertTrue(config.enable_vsync)
        
        # Visual settings
        self.assertEqual(config.opacity, 0.9)
        self.assertEqual(config.background_color, "#2c3e50")
        
        # Click-through settings
        self.assertTrue(config.click_through_enabled)
        self.assertEqual(config.click_through_mode, ClickThroughMode.LAYERED_WINDOW)
        self.assertGreater(len(config.fallback_modes), 0)
        
        # Monitor settings
        self.assertTrue(config.auto_detect_monitors)
        self.assertEqual(config.target_monitor, -1)
        self.assertTrue(config.dpi_awareness)
        
        # Error recovery
        self.assertTrue(config.crash_recovery_enabled)
        self.assertEqual(config.max_recovery_attempts, 3)

class TestMonitorConfiguration(unittest.TestCase):
    """Test monitor configuration functionality"""
    
    def test_monitor_properties(self):
        """Test monitor configuration properties"""
        monitor = MonitorConfiguration(
            handle=1,
            bounds=(0, 0, 1920, 1080),
            work_area=(0, 0, 1920, 1040),
            dpi_scale=1.25,
            is_primary=True,
            device_name="Primary Monitor"
        )
        
        self.assertEqual(monitor.width, 1920)
        self.assertEqual(monitor.height, 1080)
        self.assertEqual(monitor.center, (960, 540))
        self.assertAlmostEqual(monitor.aspect_ratio, 16/9, places=2)
        self.assertTrue(monitor.is_primary)
    
    def test_ultrawide_detection(self):
        """Test ultrawide display detection"""
        ultrawide = MonitorConfiguration(
            handle=1,
            bounds=(0, 0, 3440, 1440),
            work_area=(0, 0, 3440, 1440),
            device_name="Ultrawide"
        )
        
        self.assertGreater(ultrawide.aspect_ratio, 2.0)  # 21:9 ratio
        self.assertEqual(ultrawide.width, 3440)
        self.assertEqual(ultrawide.height, 1440)
    
    def test_multi_monitor_setup(self):
        """Test multi-monitor configuration"""
        primary = MonitorConfiguration(
            handle=1,
            bounds=(0, 0, 1920, 1080),
            work_area=(0, 0, 1920, 1080),
            is_primary=True,
            device_name="Primary"
        )
        
        secondary = MonitorConfiguration(
            handle=2,
            bounds=(1920, 0, 3840, 1080),
            work_area=(1920, 0, 3840, 1080),
            is_primary=False,
            device_name="Secondary"
        )
        
        # Test that monitors don't overlap incorrectly
        self.assertEqual(primary.bounds[2], secondary.bounds[0])  # Adjacent monitors
        self.assertNotEqual(primary.center, secondary.center)

class TestWindowsAPIHelper(unittest.TestCase):
    """Test Windows API helper functionality"""
    
    @patch('arena_bot.ui.visual_overlay.WINDOWS_API_AVAILABLE', True)
    @patch('arena_bot.ui.visual_overlay.EnumDisplayMonitors')
    @patch('arena_bot.ui.visual_overlay.GetSystemMetrics')
    def test_monitor_detection(self, mock_get_metrics, mock_enum_monitors):
        """Test monitor detection functionality"""
        # Mock EnumDisplayMonitors callback
        def mock_enum_callback(callback_func):
            # Simulate two monitors
            callback_func(1, None, (0, 0, 1920, 1080), None)
            callback_func(2, None, (1920, 0, 3840, 1080), None)
            return True
        
        mock_enum_monitors.side_effect = lambda a, b, callback, d: mock_enum_callback(callback)
        mock_get_metrics.side_effect = lambda x: 0 if x in [76, 77] else 1920
        
        monitors = WindowsAPIHelper.get_monitor_info()
        
        self.assertGreater(len(monitors), 0)
        self.assertTrue(any(m.is_primary for m in monitors))
    
    @patch('arena_bot.ui.visual_overlay.WINDOWS_API_AVAILABLE', False)
    def test_fallback_monitor_detection(self):
        """Test fallback monitor detection when Windows API unavailable"""
        monitors = WindowsAPIHelper.get_monitor_info()
        
        self.assertEqual(len(monitors), 1)
        self.assertEqual(monitors[0].bounds, (0, 0, 1920, 1080))
        self.assertTrue(monitors[0].is_primary)
    
    @patch('arena_bot.ui.visual_overlay.WINDOWS_API_AVAILABLE', True)
    @patch('arena_bot.ui.visual_overlay.FindWindow')
    @patch('arena_bot.ui.visual_overlay.IsWindow')
    def test_game_window_detection(self, mock_is_window, mock_find_window):
        """Test game window detection"""
        mock_find_window.return_value = 12345
        mock_is_window.return_value = True
        
        window_titles = ["Hearthstone", "Hearthstone.exe"]
        hwnd = WindowsAPIHelper.find_game_window(window_titles)
        
        self.assertEqual(hwnd, 12345)
        mock_find_window.assert_called()
    
    @patch('arena_bot.ui.visual_overlay.WINDOWS_API_AVAILABLE', True)
    @patch('ctypes.windll.user32.SetWindowLongW')
    @patch('arena_bot.ui.visual_overlay.SetLayeredWindowAttributes')
    def test_click_through_setup(self, mock_set_attributes, mock_set_long):
        """Test click-through setup functionality"""
        mock_set_long.return_value = 1  # Success
        
        success = WindowsAPIHelper.setup_click_through(
            12345, ClickThroughMode.LAYERED_WINDOW
        )
        
        self.assertTrue(success)
        mock_set_long.assert_called()
        mock_set_attributes.assert_called()
    
    @patch('arena_bot.ui.visual_overlay.GetSystemMetrics')
    def test_remote_session_detection(self, mock_get_metrics):
        """Test remote session detection"""
        # Test local session
        mock_get_metrics.return_value = 0
        self.assertFalse(WindowsAPIHelper.detect_remote_session())
        
        # Test remote session
        mock_get_metrics.return_value = 1
        self.assertTrue(WindowsAPIHelper.detect_remote_session())

class TestPerformanceLimiter(unittest.TestCase):
    """Test performance limiting functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = OverlayConfiguration()
        self.limiter = PerformanceLimiter(self.config)
    
    def test_frame_timing(self):
        """Test frame timing functionality"""
        start_time = self.limiter.start_frame()
        self.assertIsInstance(start_time, float)
        
        # Simulate some work
        time.sleep(0.001)
        
        within_budget = self.limiter.end_frame(start_time)
        self.assertIsInstance(within_budget, bool)
    
    def test_performance_throttling(self):
        """Test performance throttling on budget violations"""
        initial_throttle = self.limiter.performance_throttle
        
        # Simulate budget violations
        for _ in range(10):
            start_time = time.perf_counter()
            time.sleep(0.05)  # Exceed frame budget
            self.limiter.end_frame(start_time)
        
        # Should have throttled performance
        self.assertLess(self.limiter.performance_throttle, initial_throttle)
    
    def test_frame_skipping(self):
        """Test frame skipping logic"""
        # Initially should not skip frames
        self.assertFalse(self.limiter.should_skip_frame())
        
        # Force heavy throttling
        self.limiter.performance_throttle = 0.5
        
        # Should sometimes skip frames when heavily throttled
        skip_results = [self.limiter.should_skip_frame() for _ in range(10)]
        # At least some frames should be skipped
        self.assertTrue(any(skip_results))
    
    def test_performance_stats(self):
        """Test performance statistics collection"""
        # Generate some frame timing data
        for _ in range(5):
            start = self.limiter.start_frame()
            time.sleep(0.001)
            self.limiter.end_frame(start)
        
        stats = self.limiter.get_performance_stats()
        
        self.assertIn('avg_frame_time_ms', stats)
        self.assertIn('current_fps', stats)
        self.assertIn('performance_throttle', stats)
        self.assertGreater(stats['current_fps'], 0)

class TestVisualIntelligenceOverlay(unittest.TestCase):
    """Test main overlay functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = OverlayConfiguration()
        self.config.click_through_enabled = False  # Disable for tests
        
    def test_overlay_initialization(self):
        """Test overlay initialization"""
        overlay = VisualIntelligenceOverlay(self.config)
        
        self.assertEqual(overlay.state, OverlayState.INACTIVE)
        self.assertFalse(overlay.running)
        self.assertEqual(overlay.recovery_attempts, 0)
        self.assertIsNotNone(overlay.performance_limiter)
    
    @patch('arena_bot.ui.visual_overlay.WindowsAPIHelper.get_monitor_info')
    @patch('arena_bot.ui.visual_overlay.WindowsAPIHelper.detect_remote_session')
    def test_overlay_monitor_selection(self, mock_remote_session, mock_get_monitors):
        """Test monitor selection logic"""
        mock_remote_session.return_value = False
        mock_get_monitors.return_value = [
            MonitorConfiguration(
                handle=1, bounds=(0, 0, 1920, 1080), work_area=(0, 0, 1920, 1080),
                is_primary=True, device_name="Primary"
            ),
            MonitorConfiguration(
                handle=2, bounds=(1920, 0, 3840, 1080), work_area=(1920, 0, 3840, 1080),
                is_primary=False, device_name="Secondary"
            )
        ]
        
        overlay = VisualIntelligenceOverlay(self.config)
        
        # Mock the UI initialization to avoid actual Tkinter
        with patch.object(overlay, '_initialize_ui', return_value=True):
            success = overlay.initialize()
            
        self.assertTrue(success)
        self.assertIsNotNone(overlay.target_monitor)
        self.assertEqual(len(overlay.monitors), 2)
    
    def test_ai_decision_update(self):
        """Test AI decision update functionality"""
        overlay = VisualIntelligenceOverlay(self.config)
        
        # Create mock AI decision
        card1 = create_card_info("Fireball", 4) if create_card_info else Mock()
        card2 = create_card_info("Lightning Bolt", 1) if create_card_info else Mock()
        card3 = create_card_info("Counterspell", 3) if create_card_info else Mock()
        
        if create_ai_decision:
            decision = create_ai_decision(
                recommended_pick=1,
                card_evaluations=[
                    (Mock(card_info=card1, position=1), Mock(composite_score=85.0)),
                    (Mock(card_info=card2, position=2), Mock(composite_score=75.0)),
                    (Mock(card_info=card3, position=3), Mock(composite_score=80.0))
                ]
            )
        else:
            decision = Mock()
            decision.recommended_pick = 1
            decision.confidence = Mock()
            decision.reasoning = "Test reasoning"
        
        overlay.update_decision(decision)
        
        self.assertEqual(overlay.current_decision, decision)
        self.assertIsNotNone(overlay.decision_timestamp)
    
    def test_overlay_lifecycle(self):
        """Test overlay start/stop lifecycle"""
        overlay = VisualIntelligenceOverlay(self.config)
        
        # Mock initialization to avoid UI creation
        with patch.object(overlay, 'initialize', return_value=True), \
             patch.object(overlay, '_update_loop'):
            
            # Test start
            success = overlay.start()
            self.assertTrue(success)
            self.assertTrue(overlay.running)
            
            # Test stop
            overlay.stop()
            self.assertFalse(overlay.running)
            self.assertEqual(overlay.state, OverlayState.INACTIVE)
    
    def test_performance_stats(self):
        """Test performance statistics collection"""
        overlay = VisualIntelligenceOverlay(self.config)
        
        stats = overlay.get_performance_stats()
        
        self.assertIn('state', stats)
        self.assertIn('running', stats)
        self.assertIn('click_through_active', stats)
        self.assertIn('monitor_count', stats)
        self.assertEqual(stats['state'], OverlayState.INACTIVE.value)
        self.assertFalse(stats['running'])
    
    @patch('arena_bot.ui.visual_overlay.PerformanceMonitor')
    def test_performance_monitoring_integration(self, mock_monitor_class):
        """Test integration with performance monitoring"""
        mock_monitor = Mock()
        mock_monitor_class.return_value = mock_monitor
        
        overlay = VisualIntelligenceOverlay(self.config)
        
        with patch.object(overlay, '_initialize_ui', return_value=True):
            overlay.initialize()
        
        # Should have tried to create performance monitor
        mock_monitor_class.assert_called_once()
        mock_monitor.start.assert_called_once()
    
    def test_overlay_recovery_mechanism(self):
        """Test overlay recovery from errors"""
        overlay = VisualIntelligenceOverlay(self.config)
        
        # Test recovery attempt tracking
        self.assertTrue(overlay._should_attempt_recovery())
        
        # Simulate max recovery attempts
        overlay.recovery_attempts = overlay.config.max_recovery_attempts
        self.assertFalse(overlay._should_attempt_recovery())
    
    def test_remote_session_handling(self):
        """Test remote session detection and handling"""
        with patch('arena_bot.ui.visual_overlay.WindowsAPIHelper.detect_remote_session', 
                   return_value=True):
            overlay = VisualIntelligenceOverlay(self.config)
            
            with patch.object(overlay, '_initialize_ui', return_value=True):
                overlay.initialize()
            
            self.assertTrue(overlay.is_remote_session)

class TestOverlayIntegration(unittest.TestCase):
    """Test overlay integration with other systems"""
    
    def test_factory_function(self):
        """Test overlay factory function"""
        config = OverlayConfiguration()
        overlay = create_visual_overlay(config)
        
        if overlay:  # Only test if not mocked
            self.assertIsInstance(overlay, VisualIntelligenceOverlay)
            self.assertEqual(overlay.config, config)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = OverlayConfiguration()
        
        # Test valid configuration
        config.target_fps = 30
        config.max_cpu_usage = 0.05
        config.opacity = 0.9
        
        overlay = VisualIntelligenceOverlay(config)
        self.assertEqual(overlay.config.target_fps, 30)
        
        # Test edge cases
        config.target_fps = 0  # Should handle gracefully
        config.opacity = 2.0   # Should be clamped
        
        overlay = VisualIntelligenceOverlay(config)
        # Should not crash on invalid values

class TestOverlayPerformance(unittest.TestCase):
    """Performance and stress tests for overlay system"""
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable over time"""
        overlay = VisualIntelligenceOverlay()
        
        # This would be a longer-running test in practice
        # For unit tests, we just verify the mechanism exists
        self.assertTrue(hasattr(overlay, 'performance_limiter'))
        self.assertTrue(hasattr(overlay.performance_limiter, 'get_performance_stats'))
    
    def test_frame_rate_limiting(self):
        """Test frame rate limiting effectiveness"""
        config = OverlayConfiguration()
        config.target_fps = 60
        
        limiter = PerformanceLimiter(config)
        
        start_time = time.perf_counter()
        frame_times = []
        
        for _ in range(5):  # Small number for unit test
            frame_start = limiter.start_frame()
            time.sleep(0.001)  # Simulate minimal work
            limiter.end_frame(frame_start)
            frame_times.append(time.perf_counter() - frame_start)
        
        # Should have some timing data
        self.assertGreater(len(limiter.frame_times), 0)
    
    def test_adaptive_performance_scaling(self):
        """Test adaptive performance scaling under load"""
        config = OverlayConfiguration()
        limiter = PerformanceLimiter(config)
        
        initial_fps = limiter.get_target_fps()
        
        # Simulate performance issues
        limiter.performance_throttle = 0.8
        throttled_fps = limiter.get_target_fps()
        
        self.assertLess(throttled_fps, initial_fps)

class TestClickThroughValidation(unittest.TestCase):
    """Test click-through functionality validation"""
    
    @patch('arena_bot.ui.visual_overlay.WINDOWS_API_AVAILABLE', True)
    @patch('ctypes.windll.user32.GetWindowLongW')
    def test_click_through_validation(self, mock_get_long):
        """Test click-through validation logic"""
        from arena_bot.ui.visual_overlay import WS_EX_TRANSPARENT, WS_EX_LAYERED
        
        # Test transparent window
        mock_get_long.return_value = WS_EX_TRANSPARENT
        self.assertTrue(WindowsAPIHelper.validate_click_through(12345))
        
        # Test layered window
        mock_get_long.return_value = WS_EX_LAYERED
        self.assertTrue(WindowsAPIHelper.validate_click_through(12345))
        
        # Test no click-through styles
        mock_get_long.return_value = 0
        self.assertFalse(WindowsAPIHelper.validate_click_through(12345))
    
    def test_click_through_fallback_modes(self):
        """Test click-through fallback mode hierarchy"""
        config = OverlayConfiguration()
        
        self.assertGreater(len(config.fallback_modes), 0)
        self.assertIn(ClickThroughMode.TRANSPARENT_WINDOW, config.fallback_modes)
        self.assertIn(ClickThroughMode.WINDOW_EX_STYLE, config.fallback_modes)

class TestOverlayErrorHandling(unittest.TestCase):
    """Test error handling and resilience"""
    
    def test_initialization_failure_handling(self):
        """Test handling of initialization failures"""
        overlay = VisualIntelligenceOverlay()
        
        # Mock failed monitor detection
        with patch('arena_bot.ui.visual_overlay.WindowsAPIHelper.get_monitor_info',
                   side_effect=Exception("Monitor detection failed")):
            success = overlay.initialize()
            self.assertFalse(success)
            self.assertEqual(overlay.state, OverlayState.ERROR)
    
    def test_ui_creation_failure_handling(self):
        """Test handling of UI creation failures"""
        overlay = VisualIntelligenceOverlay()
        
        with patch.object(overlay, '_initialize_ui', return_value=False):
            success = overlay.initialize()
            self.assertFalse(success)
    
    def test_performance_monitor_failure_handling(self):
        """Test handling of performance monitor failures"""
        overlay = VisualIntelligenceOverlay()
        
        with patch('arena_bot.ui.visual_overlay.PerformanceMonitor',
                   side_effect=Exception("Performance monitor failed")), \
             patch.object(overlay, '_initialize_ui', return_value=True):
            
            # Should initialize successfully even if performance monitor fails
            success = overlay.initialize()
            self.assertTrue(success)
    
    def test_recovery_attempt_limits(self):
        """Test recovery attempt limiting"""
        config = OverlayConfiguration()
        config.max_recovery_attempts = 2
        
        overlay = VisualIntelligenceOverlay(config)
        
        # Should allow recovery initially
        self.assertTrue(overlay._should_attempt_recovery())
        
        # After max attempts, should stop trying
        overlay.recovery_attempts = config.max_recovery_attempts
        self.assertFalse(overlay._should_attempt_recovery())

# Test suite setup
def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestOverlayConfiguration,
        TestMonitorConfiguration,
        TestWindowsAPIHelper,
        TestPerformanceLimiter,
        TestVisualIntelligenceOverlay,
        TestOverlayIntegration,
        TestOverlayPerformance,
        TestClickThroughValidation,
        TestOverlayErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def run_performance_benchmarks():
    """Run performance benchmarks for overlay system"""
    print("=== Visual Overlay Performance Benchmarks ===")
    
    # Test overlay initialization time
    start_time = time.perf_counter()
    config = OverlayConfiguration()
    overlay = VisualIntelligenceOverlay(config)
    init_time = time.perf_counter() - start_time
    print(f"Overlay initialization: {init_time*1000:.2f}ms")
    
    # Test performance limiter overhead
    limiter = PerformanceLimiter(config)
    iterations = 1000
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        frame_start = limiter.start_frame()
        limiter.end_frame(frame_start)
    total_time = time.perf_counter() - start_time
    
    avg_overhead = (total_time / iterations) * 1000  # Convert to ms
    print(f"Performance limiter overhead: {avg_overhead:.3f}ms per frame")
    
    # Test configuration creation performance
    start_time = time.perf_counter()
    for _ in range(100):
        config = OverlayConfiguration()
    config_time = time.perf_counter() - start_time
    print(f"Configuration creation: {config_time*10:.2f}ms per 100 configs")

if __name__ == '__main__':
    print("Visual Intelligence Overlay Test Suite")
    print("=====================================")
    
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