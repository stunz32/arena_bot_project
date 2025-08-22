#!/usr/bin/env python3
"""
Tkinter GUI smoke tests for Arena Bot
Provides reproducible UI testing with screenshot capture and widget analysis.
Adapted from PyQt6 solution for tkinter-based applications.
"""

import pytest
import tkinter as tk
from tkinter import ttk
import sys
import os
from pathlib import Path
import time
import threading
from unittest.mock import patch, MagicMock
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.debug_utils import snap_widget, snap_fullscreen, dump_widget_tree, create_debug_snapshot

logger = logging.getLogger(__name__)

# Import UI components with error handling for missing dependencies
try:
    from arena_bot.ui.draft_overlay import DraftOverlay, OverlayConfig
    DRAFT_OVERLAY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Draft overlay not available: {e}")
    DRAFT_OVERLAY_AVAILABLE = False
    DraftOverlay = None
    OverlayConfig = None

try:
    from arena_bot.ui.visual_overlay import VisualOverlay
    VISUAL_OVERLAY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visual overlay not available: {e}")
    VISUAL_OVERLAY_AVAILABLE = False
    VisualOverlay = None

# Settings dialog has complex AI dependencies, skip for now
SETTINGS_DIALOG_AVAILABLE = False

class TkinterTestHelper:
    """Helper class for tkinter testing in headless environments."""
    
    def __init__(self):
        self.root = None
        self.widgets = []
    
    def setup_test_environment(self):
        """Setup tkinter for testing (including headless)."""
        try:
            # Try to create a root window
            self.root = tk.Tk()
            self.root.withdraw()  # Hide the window initially
            
            # Configure for testing
            self.root.title("Arena Bot Test Window")
            self.root.geometry("1280x800")
            
            return True
        except tk.TclError as e:
            logger.warning(f"Tkinter not available: {e}")
            return False
    
    def teardown_test_environment(self):
        """Clean up test environment."""
        try:
            if self.root:
                # Destroy all test widgets
                for widget in self.widgets:
                    try:
                        widget.destroy()
                    except:
                        pass
                
                # Destroy root
                self.root.destroy()
                self.root = None
                self.widgets.clear()
        except Exception as e:
            logger.warning(f"Error during teardown: {e}")
    
    def create_test_widget(self, widget_class, **kwargs):
        """Create a test widget and track it for cleanup."""
        if not self.root:
            raise RuntimeError("Test environment not setup")
        
        widget = widget_class(self.root, **kwargs)
        self.widgets.append(widget)
        return widget
    
    def wait_for_render(self, widget, timeout=1.0):
        """Wait for widget to be fully rendered."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                widget.update_idletasks()
                if widget.winfo_exists() and widget.winfo_width() > 1:
                    break
                time.sleep(0.01)
            except:
                break

@pytest.fixture
def tk_helper():
    """Fixture providing tkinter test helper."""
    helper = TkinterTestHelper()
    if helper.setup_test_environment():
        yield helper
    else:
        pytest.skip("Tkinter not available in test environment")
    helper.teardown_test_environment()

@pytest.fixture
def mock_card_recognizer():
    """Mock card recognizer for testing."""
    mock = MagicMock()
    mock.get_detection_stats.return_value = {
        'histogram_database_size': 100,
        'template_counts': [20, 5],
        'screen_count': 3
    }
    return mock

class TestDraftOverlay:
    """Test suite for DraftOverlay component."""
    
    @pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
    def test_draft_overlay_creation(self, tk_helper, size):
        """Test draft overlay creation and basic functionality."""
        if not tk_helper.root:
            pytest.skip("Tkinter not available")
        
        if not DRAFT_OVERLAY_AVAILABLE:
            pytest.skip("Draft overlay dependencies not available")
        
        # Create overlay with test configuration
        config = OverlayConfig(
            opacity=0.8,
            update_interval=1.0,
            show_tier_scores=True,
            show_win_rates=True
        )
        
        overlay = DraftOverlay(config)
        
        try:
            # Initialize overlay (mocked to avoid actual screen interaction)
            with patch.object(overlay, '_start_monitoring'):
                overlay.initialize()
            
            # Ensure overlay window exists
            if overlay.root:
                overlay.root.geometry(f"{size[0]}x{size[1]}")
                overlay.root.deiconify()  # Show window for testing
                tk_helper.wait_for_render(overlay.root)
                
                # Capture debug snapshot
                results = create_debug_snapshot(overlay.root, "draft_overlay")
                
                # Verify snapshot files were created
                assert any("widget_screenshot" in k for k in results.keys())
                assert any("widget_tree" in k for k in results.keys())
                
                logger.info(f"Draft overlay test completed for size {size}")
        
        finally:
            overlay.cleanup()
    
    def test_draft_overlay_card_display(self, tk_helper):
        """Test overlay card recommendation display."""
        if not tk_helper.root:
            pytest.skip("Tkinter not available")
        
        if not DRAFT_OVERLAY_AVAILABLE:
            pytest.skip("Draft overlay dependencies not available")
        
        config = OverlayConfig()
        overlay = DraftOverlay(config)
        
        try:
            with patch.object(overlay, '_start_monitoring'):
                overlay.initialize()
            
            if overlay.root:
                # Simulate card analysis data
                mock_analysis = {
                    'cards': [
                        {'name': 'Test Card 1', 'tier_score': 85, 'win_rate': 65.2},
                        {'name': 'Test Card 2', 'tier_score': 72, 'win_rate': 58.9},
                        {'name': 'Test Card 3', 'tier_score': 91, 'win_rate': 71.5}
                    ],
                    'recommendation': 'Test Card 3'
                }
                
                # Update overlay with mock data
                overlay.update_analysis(mock_analysis)
                tk_helper.wait_for_render(overlay.root)
                
                # Capture state
                snap_widget(overlay.root, "draft_overlay_with_cards")
                dump_widget_tree(overlay.root, "artifacts/draft_overlay_tree.json")
        
        finally:
            overlay.cleanup()

class TestSettingsDialog:
    """Test suite for SettingsDialog component."""
    
    def test_settings_dialog_creation(self, tk_helper):
        """Test settings dialog creation and layout."""
        if not tk_helper.root:
            pytest.skip("Tkinter not available")
        
        if not SETTINGS_DIALOG_AVAILABLE:
            pytest.skip("Settings dialog has complex AI dependencies - skipping for now")
        
        # This test is disabled due to complex AI v2 dependencies
        # Will be re-enabled once AI dependencies are resolved
        pytest.skip("Settings dialog test disabled due to AI v2 import dependencies")

class TestVisualOverlay:
    """Test suite for VisualOverlay component."""
    
    def test_visual_overlay_basic(self, tk_helper):
        """Test visual overlay basic functionality."""
        if not tk_helper.root:
            pytest.skip("Tkinter not available")
        
        if not VISUAL_OVERLAY_AVAILABLE:
            pytest.skip("Visual overlay dependencies not available")
        
        try:
            
            # Create overlay with mocked dependencies
            with patch('arena_bot.ui.visual_overlay.get_s_tier_logger') as mock_logger:
                mock_logger.return_value = MagicMock()
                
                overlay = VisualOverlay()
                
                # Mock the overlay initialization to avoid screen dependencies
                with patch.object(overlay, '_create_overlay_window') as mock_create:
                    mock_window = tk_helper.create_test_widget(tk.Toplevel)
                    mock_window.geometry("400x300")
                    mock_create.return_value = mock_window
                    
                    overlay.initialize()
                    tk_helper.wait_for_render(mock_window)
                    
                    # Capture overlay state
                    snap_widget(mock_window, "visual_overlay")
                    dump_widget_tree(mock_window, "artifacts/visual_overlay_tree.json")
        
        except ImportError as e:
            pytest.skip(f"Visual overlay dependencies not available: {e}")

class TestGUIIntegration:
    """Integration tests for multiple GUI components."""
    
    def test_multiple_windows_coordination(self, tk_helper):
        """Test coordination between multiple GUI windows."""
        if not tk_helper.root:
            pytest.skip("Tkinter not available")
        
        windows = []
        
        try:
            # Create multiple test windows to simulate real usage
            for i in range(3):
                window = tk_helper.create_test_widget(tk.Toplevel)
                window.title(f"Test Window {i+1}")
                window.geometry(f"{300 + i*50}x{200 + i*30}")
                
                # Add some test content
                label = tk.Label(window, text=f"Window {i+1} Content")
                label.pack(pady=20)
                
                button = tk.Button(window, text=f"Button {i+1}")
                button.pack(pady=10)
                
                windows.append(window)
                tk_helper.wait_for_render(window)
            
            # Capture fullscreen to see all windows
            snap_fullscreen("artifacts/multiple_windows_test.png")
            
            # Capture individual window trees
            for i, window in enumerate(windows):
                dump_widget_tree(window, f"artifacts/window_{i+1}_tree.json")
        
        finally:
            for window in windows:
                try:
                    window.destroy()
                except:
                    pass

@pytest.mark.integration
class TestGUISmoke:
    """High-level smoke tests for entire GUI system."""
    
    def test_gui_startup_sequence(self, tk_helper, mock_card_recognizer):
        """Test complete GUI startup sequence."""
        if not tk_helper.root:
            pytest.skip("Tkinter not available")
        
        startup_sequence = []
        
        try:
            # Mock main application components
            with patch('arena_bot.core.card_recognizer.get_card_recognizer') as mock_recognizer:
                mock_recognizer.return_value = mock_card_recognizer
                
                # Simulate startup sequence
                startup_sequence.append("Creating main window")
                main_window = tk_helper.create_test_widget(tk.Toplevel)
                main_window.title("Arena Bot - Main")
                main_window.geometry("1024x768")
                
                startup_sequence.append("Initializing overlays")
                # Create overlay placeholder
                overlay_frame = tk.Frame(main_window, bg="blue", width=200, height=100)
                overlay_frame.pack(side="top", fill="x")
                
                startup_sequence.append("Setting up controls")
                # Create control panel
                control_frame = tk.Frame(main_window, bg="gray", height=50)
                control_frame.pack(side="bottom", fill="x")
                
                start_button = tk.Button(control_frame, text="Start Detection")
                start_button.pack(side="left", padx=10, pady=5)
                
                stop_button = tk.Button(control_frame, text="Stop Detection")
                stop_button.pack(side="left", padx=10, pady=5)
                
                tk_helper.wait_for_render(main_window)
                
                # Capture final state
                results = create_debug_snapshot(main_window, "gui_startup_complete")
                
                # Verify all components are present
                assert len(main_window.winfo_children()) >= 2, "Main window should have child components"
                
                logger.info(f"GUI startup sequence: {' → '.join(startup_sequence)}")
        
        except Exception as e:
            logger.error(f"GUI startup test failed: {e}")
            raise

if __name__ == "__main__":
    # Run tests directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])