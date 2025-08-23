"""
Live test gating utilities for Phase 4.

Provides environment-based gating for live tests that require:
- Windows platform
- GUI desktop session
- Hearthstone window discovery
- ARENA_LIVE_TESTS=1 environment variable
"""

import os
import sys
import platform
from typing import Tuple, Optional


class LiveTestGate:
    """Environment gate for live tests that require actual Windows desktop."""
    
    @staticmethod
    def is_live_testing_enabled() -> bool:
        """Check if ARENA_LIVE_TESTS environment variable is set to '1'."""
        return os.environ.get('ARENA_LIVE_TESTS', '0') == '1'
    
    @staticmethod
    def is_windows_platform() -> bool:
        """Check if running on Windows platform."""
        return sys.platform == 'win32' or platform.system() == 'Windows'
    
    @staticmethod
    def is_gui_session_available() -> bool:
        """Check if a GUI desktop session is available."""
        if not LiveTestGate.is_windows_platform():
            return False
            
        try:
            # Try to import PyQt6 and create a QApplication
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QCoreApplication
            
            # Check if QApplication can be created (indicates GUI available)
            app = QCoreApplication.instance()
            if app is None:
                # Create a temporary app to test GUI availability
                test_app = QApplication([])
                test_app.quit()
                return True
            else:
                # Already have an app instance
                return True
                
        except Exception:
            return False
    
    @staticmethod
    def find_hearthstone_window() -> Optional[Tuple[str, int]]:
        """
        Try to find Hearthstone window.
        
        Returns:
            Tuple of (window_title, window_handle) if found, None otherwise
        """
        if not LiveTestGate.is_windows_platform():
            return None
            
        try:
            import win32gui
            import win32con
            
            hearthstone_windows = []
            
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if 'hearthstone' in window_title.lower():
                        # Check if window is of reasonable size (not minimized)
                        rect = win32gui.GetWindowRect(hwnd)
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        if width > 100 and height > 100:  # Minimum reasonable size
                            windows.append((window_title, hwnd))
                return True
            
            win32gui.EnumWindows(enum_windows_callback, hearthstone_windows)
            
            if hearthstone_windows:
                return hearthstone_windows[0]  # Return first found
            else:
                return None
                
        except ImportError:
            # win32gui not available, try alternative method
            return None
        except Exception:
            return None
    
    @staticmethod
    def check_live_test_requirements() -> Tuple[bool, str]:
        """
        Check all requirements for live testing.
        
        Returns:
            Tuple of (can_run_live_tests, reason_if_not)
        """
        if not LiveTestGate.is_live_testing_enabled():
            return False, "ARENA_LIVE_TESTS environment variable not set to '1'"
        
        if not LiveTestGate.is_windows_platform():
            return False, f"Not running on Windows (platform: {platform.system()})"
        
        if not LiveTestGate.is_gui_session_available():
            return False, "GUI desktop session not available"
        
        hearthstone_window = LiveTestGate.find_hearthstone_window()
        if hearthstone_window is None:
            return False, "Hearthstone window not found - please launch Hearthstone in windowed/borderless mode"
        
        return True, f"Live testing ready - found Hearthstone window: '{hearthstone_window[0]}'"


def require_live_testing():
    """
    Decorator/utility function for tests that require live testing.
    Raises pytest.skip if requirements not met.
    """
    try:
        import pytest
        can_run, reason = LiveTestGate.check_live_test_requirements()
        if not can_run:
            pytest.skip(f"Live testing not available: {reason}")
    except ImportError:
        # pytest not available, just check and raise regular exception
        can_run, reason = LiveTestGate.check_live_test_requirements()
        if not can_run:
            raise RuntimeError(f"Live testing not available: {reason}")