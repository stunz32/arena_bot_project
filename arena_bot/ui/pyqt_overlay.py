"""
PyQt6-based overlay for Windows with advanced click-through and DPI support.

This overlay provides hardened Windows integration with:
- OS-level click-through using WS_EX_LAYERED and WS_EX_TRANSPARENT
- DPI awareness and multi-monitor support
- Window bounds tracking and attachment to Hearthstone
- Never steal focus, stays on top
"""

import sys
import logging
import platform
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import time

# PyQt6 imports
try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
    )
    from PyQt6.QtCore import (
        Qt, QTimer, QRect, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
    )
    from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

# Windows API imports
if platform.system() == 'Windows':
    try:
        import win32api
        import win32con
        import win32gui
        WINDOWS_API_AVAILABLE = True
    except ImportError:
        WINDOWS_API_AVAILABLE = False
else:
    WINDOWS_API_AVAILABLE = False

from arena_bot.utils.window_tracker import WindowTracker, NormalizedBounds


logger = logging.getLogger(__name__)


class OverlayTheme:
    """Theme configuration for the overlay."""
    
    def __init__(self):
        self.background_color = QColor(0, 0, 0, 180)  # Semi-transparent black
        self.text_color = QColor(255, 255, 255, 255)  # White
        self.highlight_color = QColor(255, 215, 0, 255)  # Gold
        self.success_color = QColor(46, 204, 113, 255)  # Green
        self.warning_color = QColor(241, 196, 15, 255)  # Yellow
        self.error_color = QColor(231, 76, 60, 255)  # Red
        self.border_color = QColor(52, 152, 219, 255)  # Blue
        
        # Fonts
        self.title_font = QFont("Arial", 12, QFont.Weight.Bold)
        self.text_font = QFont("Arial", 10)
        self.small_font = QFont("Arial", 8)


class PyQt6Overlay(QWidget):
    """
    PyQt6-based overlay with Windows hardening and click-through support.
    
    Features:
    - OS-level click-through using Windows API
    - DPI-aware positioning and scaling
    - Automatic attachment to Hearthstone window bounds
    - Never steals focus from game
    """
    
    # Signals
    overlay_closed = pyqtSignal()
    bounds_changed = pyqtSignal(QRect)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.theme = OverlayTheme()
        
        # State
        self._is_initialized = False
        self._click_through_enabled = False
        self._window_handle = None
        self._dpi_scale = 1.0
        self._inset_pixels = 5
        
        # Window tracking
        self._window_tracker = WindowTracker()
        self._current_bounds = None
        
        # Timers
        self._update_timer = QTimer()
        self._position_timer = QTimer()
        
        # AI decision data
        self._current_decision = None
        self._last_update_time = 0
        
        # Performance tracking
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0
        
        self.logger.info("PyQt6Overlay initialized")
    
    def initialize(self) -> bool:
        """Initialize the overlay window and Windows-specific features."""
        if self._is_initialized:
            return True
        
        try:
            # Check prerequisites
            if not PYQT6_AVAILABLE:
                self.logger.error("PyQt6 not available")
                return False
            
            if not self._check_windows_support():
                self.logger.warning("Windows API support limited - some features may not work")
            
            # Setup window properties
            self._setup_window_properties()
            
            # Create UI layout
            self._create_ui()
            
            # Setup Windows-specific features
            if WINDOWS_API_AVAILABLE:
                self._setup_windows_features()
            
            # Setup window tracking
            self._setup_window_tracking()
            
            # Setup timers
            self._setup_timers()
            
            self._is_initialized = True
            self.logger.info("PyQt6Overlay initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PyQt6Overlay: {e}")
            return False
    
    def _check_windows_support(self) -> bool:
        """Check if Windows API features are available."""
        if platform.system() != 'Windows':
            self.logger.info("Not running on Windows - some features will be limited")
            return False
        
        if not WINDOWS_API_AVAILABLE:
            self.logger.warning("Windows API modules not available - click-through may not work")
            return False
        
        return True
    
    def _setup_window_properties(self):
        """Setup basic window properties for overlay behavior."""
        # Window flags for overlay behavior
        flags = (
            Qt.WindowType.FramelessWindowHint |  # No title bar
            Qt.WindowType.WindowStaysOnTopHint |  # Always on top
            Qt.WindowType.Tool |  # Tool window (doesn't appear in taskbar)
            Qt.WindowType.SubWindow  # Prevent focus stealing
        )
        
        self.setWindowFlags(flags)
        
        # Make window translucent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set initial size and position
        self.resize(400, 200)
        
        # Don't show in taskbar
        self.setWindowTitle("Arena Bot Overlay")
        
        self.logger.info("Window properties configured")
    
    def _create_ui(self):
        """Create the UI layout and widgets."""
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        self.title_label = QLabel("🎯 Arena Bot")
        self.title_label.setFont(self.theme.title_font)
        self.title_label.setStyleSheet(f"color: {self.theme.text_color.name()};")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Status
        self.status_label = QLabel("🔍 Waiting for draft...")
        self.status_label.setFont(self.theme.text_font)
        self.status_label.setStyleSheet(f"color: {self.theme.text_color.name()};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Recommendation area
        self.recommendation_widget = QWidget()
        self.recommendation_layout = QVBoxLayout(self.recommendation_widget)
        layout.addWidget(self.recommendation_widget)
        
        # Performance info (small text at bottom)
        self.perf_label = QLabel("")
        self.perf_label.setFont(self.theme.small_font)
        self.perf_label.setStyleSheet(f"color: {self.theme.text_color.name()}; opacity: 0.7;")
        self.perf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.perf_label)
        
        self.setLayout(layout)
        self.logger.info("UI layout created")
    
    def _setup_windows_features(self):
        """Setup Windows-specific features like click-through."""
        if not WINDOWS_API_AVAILABLE:
            return
        
        # Get window handle after widget is shown
        QApplication.processEvents()  # Ensure window is created
        self._window_handle = int(self.winId())
        
        if self._window_handle:
            self._enable_click_through()
            self._set_window_dpi_awareness()
            self.logger.info(f"Windows features configured for handle: {self._window_handle}")
        else:
            self.logger.warning("Could not get window handle for Windows features")
    
    def _enable_click_through(self):
        """Enable OS-level click-through using Windows API."""
        if not self._window_handle or not WINDOWS_API_AVAILABLE:
            return
        
        try:
            # Get current window style
            style = win32gui.GetWindowLong(self._window_handle, win32con.GWL_EXSTYLE)
            
            # Add layered and transparent styles
            new_style = style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            
            # Apply new style
            result = win32gui.SetWindowLong(self._window_handle, win32con.GWL_EXSTYLE, new_style)
            
            if result == 0:
                # Check for error
                error = win32api.GetLastError()
                if error != 0:
                    self.logger.warning(f"SetWindowLong failed with error: {error}")
                    return
            
            # Verify click-through is working
            if self._validate_click_through():
                self._click_through_enabled = True
                self.logger.info("✅ OS-level click-through enabled")
            else:
                self.logger.warning("⚠️ Click-through validation failed")
                
        except Exception as e:
            self.logger.error(f"Failed to enable click-through: {e}")
    
    def _validate_click_through(self) -> bool:
        """Validate that click-through is working."""
        if not self._window_handle or not WINDOWS_API_AVAILABLE:
            return False
        
        try:
            # Check if window has the expected styles
            style = win32gui.GetWindowLong(self._window_handle, win32con.GWL_EXSTYLE)
            has_layered = bool(style & win32con.WS_EX_LAYERED)
            has_transparent = bool(style & win32con.WS_EX_TRANSPARENT)
            
            self.logger.debug(f"Window style validation - Layered: {has_layered}, Transparent: {has_transparent}")
            return has_layered and has_transparent
            
        except Exception as e:
            self.logger.error(f"Click-through validation failed: {e}")
            return False
    
    def _set_window_dpi_awareness(self):
        """Set DPI awareness for proper scaling."""
        try:
            # Get DPI scale factor
            if hasattr(self, 'devicePixelRatio'):
                self._dpi_scale = self.devicePixelRatio()
            else:
                self._dpi_scale = QApplication.instance().devicePixelRatio()
            
            self.logger.info(f"DPI scale factor: {self._dpi_scale}")
            
        except Exception as e:
            self.logger.warning(f"Could not determine DPI scale: {e}")
            self._dpi_scale = 1.0
    
    def _setup_window_tracking(self):
        """Setup automatic window tracking for Hearthstone."""
        try:
            # Add callback for bounds changes
            self._window_tracker.add_bounds_callback(self._on_hearthstone_bounds_changed)
            
            self.logger.info("Window tracking configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup window tracking: {e}")
    
    def _setup_timers(self):
        """Setup update and positioning timers."""
        # Main update timer (30 FPS)
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(33)  # ~30 FPS
        
        # Position update timer (5 FPS)
        self._position_timer.timeout.connect(self._update_position)
        self._position_timer.start(200)  # 5 FPS
        
        self.logger.info("Timers configured")
    
    def start_tracking(self) -> bool:
        """Start tracking Hearthstone window."""
        try:
            if self._window_tracker.start_tracking():
                self.logger.info("✅ Window tracking started")
                return True
            else:
                error = self._window_tracker.get_last_error()
                self.logger.error(f"❌ Failed to start window tracking: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start window tracking: {e}")
            return False
    
    def stop_tracking(self):
        """Stop tracking Hearthstone window."""
        try:
            self._window_tracker.stop_tracking()
            self.logger.info("Window tracking stopped")
        except Exception as e:
            self.logger.error(f"Error stopping window tracking: {e}")
    
    def _on_hearthstone_bounds_changed(self, bounds: NormalizedBounds):
        """Handle Hearthstone window bounds changes."""
        self._current_bounds = bounds
        self.logger.debug(f"Hearthstone bounds updated: {bounds.get_rect()}")
        
        # Update overlay position on next position timer tick
    
    def _update_position(self):
        """Update overlay position based on Hearthstone window."""
        if not self._current_bounds:
            return
        
        try:
            # Get inset bounds
            inset_rect = self._current_bounds.get_inset_rect(self._inset_pixels)
            x, y, width, height = inset_rect
            
            # Calculate overlay position (top-right corner of game window)
            overlay_width = 400
            overlay_height = 200
            
            overlay_x = x + width - overlay_width - 10  # 10px margin from right edge
            overlay_y = y + 10  # 10px margin from top edge
            
            # Apply DPI scaling
            if self._dpi_scale != 1.0:
                overlay_x = int(overlay_x * self._dpi_scale)
                overlay_y = int(overlay_y * self._dpi_scale)
                overlay_width = int(overlay_width * self._dpi_scale)
                overlay_height = int(overlay_height * self._dpi_scale)
            
            # Update geometry
            new_rect = QRect(overlay_x, overlay_y, overlay_width, overlay_height)
            current_rect = self.geometry()
            
            # Only update if position changed significantly (avoid jitter)
            if (abs(new_rect.x() - current_rect.x()) > 5 or 
                abs(new_rect.y() - current_rect.y()) > 5):
                
                self.setGeometry(new_rect)
                self.bounds_changed.emit(new_rect)
                
                self.logger.debug(f"Overlay position updated: ({overlay_x}, {overlay_y})")
            
        except Exception as e:
            self.logger.error(f"Error updating overlay position: {e}")
    
    def _update_display(self):
        """Update the overlay display (main rendering loop)."""
        try:
            # Update FPS counter
            self._frame_count += 1
            current_time = time.time()
            if current_time - self._last_fps_time >= 1.0:
                self._current_fps = self._frame_count / (current_time - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = current_time
            
            # Update performance info
            if self._window_tracker:
                status = self._window_tracker.get_status()
                perf_text = f"FPS: {self._current_fps:.1f} | "
                
                if status['is_tracking']:
                    bounds = status['current_bounds']
                    if bounds:
                        perf_text += f"Tracked: {bounds[2]}x{bounds[3]} | "
                    perf_text += f"✅ Active"
                    if self._click_through_enabled:
                        perf_text += " | 🖱️ Click-through"
                else:
                    perf_text += "❌ No tracking"
                    if status['last_error']:
                        perf_text += f" ({status['last_error'][:30]}...)"
                
                self.perf_label.setText(perf_text)
            
            # Update recommendation display
            self._update_recommendations()
            
        except Exception as e:
            self.logger.error(f"Error in display update: {e}")
    
    def _update_recommendations(self):
        """Update the AI recommendations display."""
        if not self._current_decision:
            self.status_label.setText("🔍 Waiting for draft...")
            # Clear recommendation area
            for i in reversed(range(self.recommendation_layout.count())):
                child = self.recommendation_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            return
        
        # Update status
        self.status_label.setText("✅ Draft detected")
        
        # TODO(claude): Implement recommendation display based on AI decision format
        # For now, show placeholder
        if self.recommendation_layout.count() == 0:
            placeholder = QLabel("🎯 AI recommendations will appear here")
            placeholder.setFont(self.theme.text_font)
            placeholder.setStyleSheet(f"color: {self.theme.highlight_color.name()};")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.recommendation_layout.addWidget(placeholder)
    
    def update_decision(self, decision_data: Dict[str, Any]):
        """Update the overlay with new AI decision data."""
        self._current_decision = decision_data
        self._last_update_time = time.time()
        
        self.logger.info("AI decision data updated")
    
    def show_overlay(self):
        """Show the overlay and start tracking."""
        if not self._is_initialized:
            if not self.initialize():
                self.logger.error("Failed to initialize overlay")
                return False
        
        # Show the widget
        self.show()
        
        # Ensure Windows features are applied after show
        if WINDOWS_API_AVAILABLE and not self._click_through_enabled:
            QTimer.singleShot(100, self._setup_windows_features)  # Delay to ensure window is ready
        
        # Start window tracking
        success = self.start_tracking()
        
        self.logger.info(f"Overlay shown - tracking: {'✅' if success else '❌'}")
        return success
    
    def hide_overlay(self):
        """Hide the overlay and stop tracking."""
        self.hide()
        self.stop_tracking()
        self.logger.info("Overlay hidden")
    
    def paintEvent(self, event):
        """Custom paint event for overlay background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw semi-transparent background
        painter.fillRect(self.rect(), self.theme.background_color)
        
        # Draw border
        pen = QPen(self.theme.border_color, 2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        super().paintEvent(event)
    
    def closeEvent(self, event):
        """Handle overlay close event."""
        self.stop_tracking()
        self._update_timer.stop()
        self._position_timer.stop()
        self.overlay_closed.emit()
        super().closeEvent(event)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current overlay status."""
        return {
            'initialized': self._is_initialized,
            'visible': self.isVisible(),
            'click_through_enabled': self._click_through_enabled,
            'dpi_scale': self._dpi_scale,
            'window_handle': self._window_handle,
            'fps': self._current_fps,
            'window_tracker_status': self._window_tracker.get_status() if self._window_tracker else None,
            'current_bounds': self._current_bounds.get_rect() if self._current_bounds else None
        }


def create_overlay_application() -> Tuple[QApplication, PyQt6Overlay]:
    """
    Create QApplication and overlay instance.
    
    Returns:
        Tuple of (QApplication, PyQt6Overlay)
    """
    if not PYQT6_AVAILABLE:
        raise RuntimeError("PyQt6 not available")
    
    # Create or get existing QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Arena Bot Overlay")
        app.setApplicationVersion("4.0")
        app.setQuitOnLastWindowClosed(False)  # Don't quit when overlay closes
    
    # Create overlay
    overlay = PyQt6Overlay()
    
    return app, overlay


# Test function for validation
def test_overlay_features():
    """Test overlay features for live smoke tests."""
    logger.info("Testing PyQt6 overlay features...")
    
    try:
        app, overlay = create_overlay_application()
        
        # Initialize
        if not overlay.initialize():
            return False, "Failed to initialize overlay"
        
        # Test show/hide
        overlay.show_overlay()
        QApplication.processEvents()
        
        # Check status
        status = overlay.get_status()
        
        # Hide overlay
        overlay.hide_overlay()
        
        return True, f"Overlay test passed - Status: {status}"
        
    except Exception as e:
        return False, f"Overlay test failed: {e}"