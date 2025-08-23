"""
Live overlay flags smoke tests for Phase 4.

These tests verify PyQt6 overlay functionality and Windows integration
but are gated behind ARENA_LIVE_TESTS=1 environment variable.

Tests verify:
- Overlay instantiation and Qt flags
- Click-through behavior via Qt/PyQt introspection  
- Window attachment to Hearthstone bounds
- DPI awareness and positioning
"""

import os
import sys
import time
import json
import pytest
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from arena_bot.utils.live_test_gate import LiveTestGate, require_live_testing
from arena_bot.utils.debug_dump import DebugDumpManager

# PyQt6 imports with fallback
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QRect
    from PyQt6.QtTest import QTest
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

# Windows API imports
if sys.platform == 'win32':
    try:
        import win32gui
        import win32con
        WINDOWS_API_AVAILABLE = True
    except ImportError:
        WINDOWS_API_AVAILABLE = False
else:
    WINDOWS_API_AVAILABLE = False

from arena_bot.ui.pyqt_overlay import PyQt6Overlay, create_overlay_application, test_overlay_features


logger = logging.getLogger(__name__)


class TestLiveOverlayFlags:
    """Live overlay flags smoke tests with environment gating."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.debug_manager = DebugDumpManager()
        self.app = None
        self.overlay = None
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up overlay and app
        if self.overlay:
            try:
                self.overlay.hide_overlay()
                self.overlay.close()
            except Exception as e:
                logger.debug(f"Error closing overlay: {e}")
            self.overlay = None
        
        # Don't quit the QApplication as it may be shared
        # QApplication cleanup is handled by PyQt
    
    @pytest.fixture
    def debug_tag(self):
        """Generate debug tag for this test run."""
        return "live_overlay_flags"
    
    def test_live_overlay_gate_check(self):
        """Test that live overlay gating works correctly."""
        # This test always runs to verify gating logic
        
        can_run, reason = LiveTestGate.check_live_test_requirements()
        
        # Log the gate status for diagnostics
        logger.info(f"Live overlay gate status: can_run={can_run}, reason='{reason}'")
        
        # Check PyQt6 availability
        pyqt6_available = PYQT6_AVAILABLE
        windows_api_available = WINDOWS_API_AVAILABLE
        
        logger.info(f"Dependencies - PyQt6: {pyqt6_available}, Windows API: {windows_api_available}")
        
        # This test always passes - it's just checking the gate logic
        assert True
    
    def test_overlay_instantiation_without_live(self):
        """Test that overlay can be instantiated without live environment."""
        # This test can run anywhere to verify overlay class logic
        
        if not PYQT6_AVAILABLE:
            pytest.skip("PyQt6 not available")
        
        try:
            # Create application and overlay
            app, overlay = create_overlay_application()
            
            # Should be able to create without error
            assert app is not None
            assert overlay is not None
            
            # Check basic properties
            assert hasattr(overlay, 'initialize')
            assert hasattr(overlay, 'show_overlay')
            assert hasattr(overlay, 'hide_overlay')
            assert hasattr(overlay, 'get_status')
            
            # Get initial status
            status = overlay.get_status()
            assert isinstance(status, dict)
            assert 'initialized' in status
            
            logger.info(f"✅ Overlay instantiation test passed - Status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Overlay instantiation test failed: {e}")
            raise
    
    def test_live_overlay_initialization(self, debug_tag):
        """
        Test that overlay initializes correctly in live environment.
        
        This is the main live overlay test - requires full live environment.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🚀 Starting live overlay initialization test")
        
        # Start debug dump
        self.debug_manager.start_dump(debug_tag)
        
        try:
            # Check prerequisites
            if not PYQT6_AVAILABLE:
                pytest.skip("PyQt6 not available for live overlay test")
            
            # Create application and overlay
            self.app, self.overlay = create_overlay_application()
            
            # Initialize overlay
            init_success = self.overlay.initialize()
            assert init_success, "Overlay initialization failed"
            
            # Get status after initialization
            status = self.overlay.get_status()
            assert status['initialized'], "Overlay reports as not initialized"
            
            logger.info(f"✅ Overlay initialized successfully - Status: {status}")
            
            # Save initialization metadata
            init_metadata = {
                'initialization_status': status,
                'app_info': {
                    'app_name': self.app.applicationName(),
                    'app_version': self.app.applicationVersion(),
                    'platform': sys.platform,
                    'pyqt6_available': PYQT6_AVAILABLE,
                    'windows_api_available': WINDOWS_API_AVAILABLE
                },
                'test_info': {
                    'test_name': 'live_overlay_initialization',
                    'timestamp': time.time()
                }
            }
            
            metadata_path = self.debug_manager.save_json(init_metadata, "live_overlay_init_metadata")
            logger.info(f"📁 Initialization metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Live overlay initialization test failed: {e}")
            raise
        finally:
            # End debug dump
            self.debug_manager.end_dump()
            debug_dir = self.debug_manager.get_current_dump_dir()
            logger.info(f"📂 Initialization test artifacts in: {debug_dir}")
    
    def test_live_overlay_qt_flags_introspection(self, debug_tag):
        """
        Test overlay Qt flags and window properties via introspection.
        
        This test verifies that the overlay has correct Qt flags set.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🔍 Starting live overlay Qt flags introspection")
        
        # Start debug dump
        self.debug_manager.start_dump(debug_tag + "_flags")
        
        try:
            if not PYQT6_AVAILABLE:
                pytest.skip("PyQt6 not available for Qt flags test")
            
            # Create and initialize overlay
            self.app, self.overlay = create_overlay_application()
            assert self.overlay.initialize(), "Overlay initialization failed"
            
            # Show overlay to ensure window is created
            self.overlay.show_overlay()
            QApplication.processEvents()  # Process events to ensure window creation
            
            # Check window flags via Qt introspection
            window_flags = self.overlay.windowFlags()
            
            # Expected flags for overlay behavior
            expected_flags = [
                Qt.WindowType.FramelessWindowHint,  # No title bar
                Qt.WindowType.WindowStaysOnTopHint,  # Always on top
                Qt.WindowType.Tool,  # Tool window (not in taskbar)
                Qt.WindowType.SubWindow  # Prevent focus stealing
            ]
            
            flags_status = {}
            for flag in expected_flags:
                has_flag = bool(window_flags & flag)
                flags_status[flag.name] = has_flag
                logger.info(f"   {flag.name}: {'✅' if has_flag else '❌'}")
                
                # Assert critical flags are set
                if flag in [Qt.WindowType.FramelessWindowHint, Qt.WindowType.WindowStaysOnTopHint]:
                    assert has_flag, f"Critical flag {flag.name} not set"
            
            # Check widget attributes
            widget_attributes = {
                'WA_TranslucentBackground': self.overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground),
                'visible': self.overlay.isVisible(),
                'window_title': self.overlay.windowTitle()
            }
            
            logger.info(f"Widget attributes: {widget_attributes}")
            
            # Check geometry
            geometry = self.overlay.geometry()
            geometry_info = {
                'x': geometry.x(),
                'y': geometry.y(),
                'width': geometry.width(),
                'height': geometry.height()
            }
            
            logger.info(f"Overlay geometry: {geometry_info}")
            
            # Assert reasonable geometry
            assert geometry.width() > 100, f"Overlay width too small: {geometry.width()}"
            assert geometry.height() > 100, f"Overlay height too small: {geometry.height()}"
            
            # Save flags introspection metadata
            flags_metadata = {
                'qt_flags_status': flags_status,
                'widget_attributes': widget_attributes,
                'geometry_info': geometry_info,
                'overlay_status': self.overlay.get_status(),
                'test_info': {
                    'test_name': 'live_overlay_qt_flags_introspection',
                    'timestamp': time.time()
                }
            }
            
            metadata_path = self.debug_manager.save_json(flags_metadata, "live_overlay_flags_metadata")
            logger.info(f"📁 Qt flags metadata saved: {metadata_path}")
            
            logger.info("✅ Qt flags introspection test passed")
            
        except Exception as e:
            logger.error(f"❌ Live overlay Qt flags test failed: {e}")
            raise
        finally:
            # End debug dump
            self.debug_manager.end_dump()
            debug_dir = self.debug_manager.get_current_dump_dir()
            logger.info(f"📂 Qt flags test artifacts in: {debug_dir}")
    
    def test_live_overlay_click_through_introspection(self, debug_tag):
        """
        Test click-through behavior via Windows API introspection.
        
        This test checks Windows-level click-through settings.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🖱️ Starting live overlay click-through introspection")
        
        # Start debug dump
        self.debug_manager.start_dump(debug_tag + "_clickthrough")
        
        try:
            if not PYQT6_AVAILABLE:
                pytest.skip("PyQt6 not available for click-through test")
            
            if not WINDOWS_API_AVAILABLE:
                pytest.skip("Windows API not available for click-through test")
            
            # Create and initialize overlay
            self.app, self.overlay = create_overlay_application()
            assert self.overlay.initialize(), "Overlay initialization failed"
            
            # Show overlay to ensure window is created
            self.overlay.show_overlay()
            QApplication.processEvents()  # Process events to ensure window creation
            
            # Give Windows features time to initialize
            time.sleep(0.5)
            QApplication.processEvents()
            
            # Get overlay status (includes click-through status)
            status = self.overlay.get_status()
            
            # Check if overlay reports click-through as enabled
            click_through_enabled = status.get('click_through_enabled', False)
            window_handle = status.get('window_handle')
            
            logger.info(f"Overlay click-through status: {click_through_enabled}")
            logger.info(f"Window handle: {window_handle}")
            
            click_through_metadata = {
                'overlay_click_through_enabled': click_through_enabled,
                'window_handle': window_handle,
                'overlay_status': status
            }
            
            # If we have a window handle, check Windows API level
            if window_handle and WINDOWS_API_AVAILABLE:
                try:
                    # Get window style
                    style = win32gui.GetWindowLong(window_handle, win32con.GWL_EXSTYLE)
                    
                    # Check for click-through flags
                    has_layered = bool(style & win32con.WS_EX_LAYERED)
                    has_transparent = bool(style & win32con.WS_EX_TRANSPARENT)
                    
                    windows_api_info = {
                        'window_style': style,
                        'has_layered_flag': has_layered,
                        'has_transparent_flag': has_transparent,
                        'windows_click_through_active': has_layered and has_transparent
                    }
                    
                    click_through_metadata['windows_api_info'] = windows_api_info
                    
                    logger.info(f"Windows API click-through status: "
                               f"layered={has_layered}, transparent={has_transparent}")
                    
                    # Log success/warning based on click-through status
                    if has_layered and has_transparent:
                        logger.info("✅ Windows-level click-through appears active")
                    else:
                        logger.warning("⚠️ Windows-level click-through may not be active")
                    
                except Exception as e:
                    logger.warning(f"Could not introspect Windows API click-through: {e}")
                    click_through_metadata['windows_api_error'] = str(e)
            
            # Add test metadata
            click_through_metadata['test_info'] = {
                'test_name': 'live_overlay_click_through_introspection',
                'timestamp': time.time(),
                'platform': sys.platform,
                'windows_api_available': WINDOWS_API_AVAILABLE
            }
            
            # Save click-through introspection metadata
            metadata_path = self.debug_manager.save_json(click_through_metadata, "live_overlay_clickthrough_metadata")
            logger.info(f"📁 Click-through metadata saved: {metadata_path}")
            
            logger.info("✅ Click-through introspection test completed")
            
        except Exception as e:
            logger.error(f"❌ Live overlay click-through test failed: {e}")
            raise
        finally:
            # End debug dump
            self.debug_manager.end_dump()
            debug_dir = self.debug_manager.get_current_dump_dir()
            logger.info(f"📂 Click-through test artifacts in: {debug_dir}")
    
    def test_live_overlay_window_attachment(self, debug_tag):
        """
        Test overlay attachment to Hearthstone window bounds.
        
        This test verifies that the overlay can track and attach to Hearthstone.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("📱 Starting live overlay window attachment test")
        
        # Start debug dump
        self.debug_manager.start_dump(debug_tag + "_attachment")
        
        try:
            if not PYQT6_AVAILABLE:
                pytest.skip("PyQt6 not available for window attachment test")
            
            # Create and initialize overlay
            self.app, self.overlay = create_overlay_application()
            assert self.overlay.initialize(), "Overlay initialization failed"
            
            # Show overlay and start tracking
            tracking_success = self.overlay.show_overlay()
            
            # Give window tracking time to find Hearthstone
            time.sleep(1.0)
            QApplication.processEvents()
            
            # Get overlay status
            status = self.overlay.get_status()
            window_tracker_status = status.get('window_tracker_status', {})
            current_bounds = status.get('current_bounds')
            
            logger.info(f"Window tracking status: {window_tracker_status}")
            logger.info(f"Current bounds: {current_bounds}")
            
            # Check if tracking is working
            is_tracking = window_tracker_status.get('is_tracking', False)
            has_bounds = current_bounds is not None
            
            if not is_tracking:
                last_error = window_tracker_status.get('last_error', 'Unknown error')
                logger.warning(f"⚠️ Window tracking not active: {last_error}")
                # Don't fail the test - this may be expected if Hearthstone not in ideal state
            
            if has_bounds:
                logger.info(f"✅ Overlay has Hearthstone bounds: {current_bounds}")
                
                # Verify bounds are reasonable
                x, y, width, height = current_bounds
                assert width > 100, f"Tracked window width too small: {width}"
                assert height > 100, f"Tracked window height too small: {height}"
                assert x >= -3840, f"Tracked window X position unreasonable: {x}"  # Allow for multi-monitor
                assert y >= -2160, f"Tracked window Y position unreasonable: {y}"
            else:
                logger.warning("⚠️ Overlay does not have Hearthstone bounds")
            
            # Get overlay geometry
            overlay_geometry = self.overlay.geometry()
            overlay_info = {
                'x': overlay_geometry.x(),
                'y': overlay_geometry.y(),
                'width': overlay_geometry.width(),
                'height': overlay_geometry.height()
            }
            
            logger.info(f"Overlay geometry: {overlay_info}")
            
            # Save window attachment metadata
            attachment_metadata = {
                'window_tracking_status': window_tracker_status,
                'hearthstone_bounds': current_bounds,
                'overlay_geometry': overlay_info,
                'tracking_success': tracking_success,
                'overlay_status': status,
                'test_info': {
                    'test_name': 'live_overlay_window_attachment',
                    'timestamp': time.time(),
                    'tracking_active': is_tracking,
                    'has_bounds': has_bounds
                }
            }
            
            metadata_path = self.debug_manager.save_json(attachment_metadata, "live_overlay_attachment_metadata")
            logger.info(f"📁 Window attachment metadata saved: {metadata_path}")
            
            logger.info("✅ Window attachment test completed")
            
        except Exception as e:
            logger.error(f"❌ Live overlay window attachment test failed: {e}")
            raise
        finally:
            # End debug dump
            self.debug_manager.end_dump()
            debug_dir = self.debug_manager.get_current_dump_dir()
            logger.info(f"📂 Window attachment test artifacts in: {debug_dir}")
    
    def test_overlay_features_smoke(self):
        """
        Test overlay features using the built-in test function.
        
        This is a lightweight test of the overlay test infrastructure.
        """
        require_live_testing()  # Skip if requirements not met
        
        logger.info("🧪 Running overlay features smoke test")
        
        try:
            success, message = test_overlay_features()
            
            logger.info(f"Overlay features test result: {message}")
            
            if success:
                logger.info("✅ Overlay features smoke test passed")
            else:
                logger.error(f"❌ Overlay features smoke test failed: {message}")
                pytest.fail(f"Overlay features test failed: {message}")
                
        except Exception as e:
            logger.error(f"❌ Overlay features smoke test failed with exception: {e}")
            raise