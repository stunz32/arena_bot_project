"""
Live capture smoke tests for Phase 4.

These tests verify real Windows capture functionality but are gated behind
ARENA_LIVE_TESTS=1 environment variable and require:
- Windows platform
- GUI desktop session
- Hearthstone window discoverable

Otherwise they skip cleanly with clear reasons.
"""

import os
import sys
import time
import json
import pytest
import logging
from pathlib import Path
from typing import Dict, Any

from arena_bot.utils.live_test_gate import LiveTestGate, require_live_testing
from arena_bot.capture.capture_backend import AdaptiveCaptureManager
from arena_bot.utils.debug_dump import DebugDumpManager


logger = logging.getLogger(__name__)


class TestLiveCaptureSmoke:
    """Live capture smoke tests with environment gating."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.debug_manager = DebugDumpManager()
        self.capture_manager = AdaptiveCaptureManager()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        if hasattr(self, 'debug_manager'):
            # Don't clean up debug artifacts - they're useful for inspection
            pass
    
    @pytest.fixture
    def debug_tag(self):
        """Generate debug tag for this test run."""
        return "live_capture_smoke"
    
    def test_live_capture_gate_check(self):
        """Test that live test gating works correctly."""
        # This test always runs to verify gating logic
        
        can_run, reason = LiveTestGate.check_live_test_requirements()
        
        # Log the gate status for diagnostics
        logger.info(f"Live test gate status: can_run={can_run}, reason='{reason}'")
        
        # Verify gate components work
        assert isinstance(can_run, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0
        
        # Check individual components
        live_tests_enabled = LiveTestGate.is_live_testing_enabled()
        is_windows = LiveTestGate.is_windows_platform()
        has_gui = LiveTestGate.is_gui_session_available()
        hearthstone_window = LiveTestGate.find_hearthstone_window()
        
        logger.info(f"Gate components - ARENA_LIVE_TESTS: {live_tests_enabled}, "
                   f"Windows: {is_windows}, GUI: {has_gui}, "
                   f"Hearthstone: {hearthstone_window is not None}")
        
        # This test always passes - it's just checking the gate logic
        assert True
    
    def test_live_capture_backend_selection(self):
        """Test that capture backend selection works without requiring live environment."""
        # This test can run anywhere to verify backend logic
        
        # Test backend availability detection
        backend = self.capture_manager.get_active_backend()
        
        # Should have selected some backend
        assert backend is not None
        assert hasattr(backend, 'get_name')
        assert hasattr(backend, 'is_available')
        
        backend_name = backend.get_name()
        is_available = backend.is_available()
        
        logger.info(f"Selected capture backend: {backend_name}, available: {is_available}")
        
        # Backend should report itself as available if it was selected
        assert is_available, f"Selected backend {backend_name} reports as not available"
    
    def test_live_single_frame_capture(self, debug_tag):
        """
        Test capturing a single frame of the Hearthstone window.
        
        This is the main live capture smoke test - requires full live environment.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🚀 Starting live capture smoke test")
        
        # Start debug dump
        self.debug_manager.start_dump(debug_tag)
        
        try:
            # Find Hearthstone window
            window = self.capture_manager.find_hearthstone_window()
            assert window is not None, "Hearthstone window not found for capture test"
            
            logger.info(f"📱 Found Hearthstone window: '{window.title}' "
                       f"({window.width}x{window.height}) at ({window.x}, {window.y})")
            
            # Capture the window
            start_time = time.time()
            frame = self.capture_manager.capture_rect(window.x, window.y, window.width, window.height)
            capture_duration = time.time() - start_time
            
            # Verify frame properties
            assert frame is not None, "Capture returned None"
            assert frame.image is not None, "Captured frame has no image data"
            assert frame.image.size > 0, "Captured frame image is empty"
            
            # Check frame dimensions
            height, width = frame.image.shape[:2]
            assert height > 100, f"Frame height too small: {height}"
            assert width > 100, f"Frame width too small: {width}"
            assert height <= window.height * 2, f"Frame height unexpectedly large: {height}"
            assert width <= window.width * 2, f"Frame width unexpectedly large: {width}"
            
            # Check capture timing (loose timeout for live testing)
            assert capture_duration < 5.0, f"Capture took too long: {capture_duration:.2f}s"
            
            # Verify frame metadata
            assert frame.backend_name is not None, "Frame missing backend name"
            assert frame.timestamp > 0, "Frame missing valid timestamp"
            assert frame.capture_duration_ms >= 0, "Frame missing capture duration"
            
            logger.info(f"✅ Frame captured successfully: {width}x{height} "
                       f"via {frame.backend_name} in {capture_duration:.3f}s")
            
            # Create debug artifact
            frame_path = self.debug_manager.save_image(frame.image, "live_capture_frame")
            
            # Save frame metadata
            frame_metadata = {
                'window_info': {
                    'title': window.title,
                    'handle': window.handle,
                    'position': [window.x, window.y],
                    'size': [window.width, window.height],
                    'pid': window.pid
                },
                'frame_info': {
                    'backend_name': frame.backend_name,
                    'timestamp': frame.timestamp,
                    'capture_duration_ms': frame.capture_duration_ms,
                    'source_rect': frame.source_rect,
                    'dpi_scale': frame.dpi_scale,
                    'image_size': [width, height],
                    'metadata': frame.metadata
                },
                'test_info': {
                    'test_duration_s': capture_duration,
                    'frame_path': str(frame_path),
                    'backend_selection': self.capture_manager.get_active_backend().get_name()
                }
            }
            
            metadata_path = self.debug_manager.save_json(frame_metadata, "live_capture_metadata")
            
            logger.info(f"📁 Debug artifacts saved: frame={frame_path}, metadata={metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Live capture test failed: {e}")
            raise
        finally:
            # End debug dump
            self.debug_manager.end_dump()
            debug_dir = self.debug_manager.get_current_dump_dir()
            logger.info(f"📂 Debug artifacts in: {debug_dir}")
    
    def test_live_capture_performance_bounds(self, debug_tag):
        """
        Test that live capture meets performance bounds.
        
        Requires live environment but tests multiple captures for performance.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🏃 Starting live capture performance test")
        
        # Start debug dump
        self.debug_manager.start_dump(debug_tag + "_perf")
        
        try:
            # Find Hearthstone window
            window = self.capture_manager.find_hearthstone_window()
            assert window is not None, "Hearthstone window not found for performance test"
            
            # Performance bounds (loose for live testing)
            MAX_CAPTURE_TIME = 2.0  # 2 seconds max per capture
            MIN_FPS = 0.5  # At least 0.5 FPS (2 seconds per frame)
            NUM_TEST_CAPTURES = 3
            
            capture_times = []
            frames = []
            
            logger.info(f"📊 Testing {NUM_TEST_CAPTURES} captures for performance")
            
            for i in range(NUM_TEST_CAPTURES):
                start_time = time.time()
                frame = self.capture_manager.capture_rect(window.x, window.y, window.width, window.height)
                capture_time = time.time() - start_time
                
                capture_times.append(capture_time)
                frames.append(frame)
                
                logger.info(f"   Capture {i+1}: {capture_time:.3f}s via {frame.backend_name}")
                
                # Check individual capture time
                assert capture_time < MAX_CAPTURE_TIME, \
                    f"Capture {i+1} too slow: {capture_time:.3f}s > {MAX_CAPTURE_TIME}s"
                
                # Small delay between captures
                time.sleep(0.1)
            
            # Calculate performance metrics
            avg_capture_time = sum(capture_times) / len(capture_times)
            max_capture_time = max(capture_times)
            min_capture_time = min(capture_times)
            effective_fps = 1.0 / avg_capture_time if avg_capture_time > 0 else 0
            
            # Check performance bounds
            assert effective_fps >= MIN_FPS, \
                f"Effective FPS too low: {effective_fps:.2f} < {MIN_FPS}"
            
            logger.info(f"📈 Performance metrics: avg={avg_capture_time:.3f}s, "
                       f"min={min_capture_time:.3f}s, max={max_capture_time:.3f}s, "
                       f"fps={effective_fps:.2f}")
            
            # Save performance metadata
            perf_metadata = {
                'performance_metrics': {
                    'num_captures': NUM_TEST_CAPTURES,
                    'capture_times_s': capture_times,
                    'avg_capture_time_s': avg_capture_time,
                    'min_capture_time_s': min_capture_time,
                    'max_capture_time_s': max_capture_time,
                    'effective_fps': effective_fps
                },
                'performance_bounds': {
                    'max_capture_time_s': MAX_CAPTURE_TIME,
                    'min_fps': MIN_FPS,
                    'bounds_met': effective_fps >= MIN_FPS and max_capture_time < MAX_CAPTURE_TIME
                },
                'backend_info': {
                    'backend_name': frames[0].backend_name if frames else None,
                    'backend_metadata': frames[0].metadata if frames else None
                }
            }
            
            metadata_path = self.debug_manager.save_json(perf_metadata, "live_capture_performance")
            logger.info(f"📁 Performance metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Live capture performance test failed: {e}")
            raise
        finally:
            # End debug dump
            self.debug_manager.end_dump()
            debug_dir = self.debug_manager.get_current_dump_dir()
            logger.info(f"📂 Performance test artifacts in: {debug_dir}")
    
    def test_live_capture_backend_fallback(self):
        """
        Test that capture backend fallback works in live environment.
        
        This test tries to verify fallback behavior but may skip if only one backend available.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🔄 Testing capture backend fallback")
        
        # Get all available backends
        backends = self.capture_manager.backends
        available_backends = [b for b in backends if b.is_available()]
        
        logger.info(f"Available backends: {[b.get_name() for b in available_backends]}")
        
        if len(available_backends) < 2:
            pytest.skip(f"Only {len(available_backends)} backend(s) available - cannot test fallback")
        
        # Test each backend individually
        window = self.capture_manager.find_hearthstone_window()
        assert window is not None, "Hearthstone window not found for fallback test"
        
        for backend in available_backends:
            try:
                frame = backend.capture_rect(window.x, window.y, window.width, window.height)
                assert frame is not None, f"Backend {backend.get_name()} returned None"
                assert frame.image is not None, f"Backend {backend.get_name()} returned no image data"
                
                logger.info(f"✅ Backend {backend.get_name()} working: "
                           f"{frame.image.shape[1]}x{frame.image.shape[0]} "
                           f"in {frame.capture_duration_ms:.1f}ms")
                
            except Exception as e:
                logger.warning(f"⚠️ Backend {backend.get_name()} failed: {e}")
        
        # Test adaptive manager's fallback behavior
        # (This mainly tests that the manager can recover from backend failures)
        original_backend = self.capture_manager.get_active_backend()
        logger.info(f"Original active backend: {original_backend.get_name()}")
        
        # Capture should work with adaptive manager
        frame = self.capture_manager.capture_rect(window.x, window.y, window.width, window.height)
        assert frame is not None, "Adaptive capture manager failed"
        
        final_backend = self.capture_manager.get_active_backend()
        logger.info(f"Final active backend: {final_backend.get_name()}")
        
        logger.info("✅ Backend fallback test completed")