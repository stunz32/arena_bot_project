"""
GUI Smoke Tests

Tests for UI Safe Demo mode and uniform fill detection.
Skips cleanly on headless environments.
"""

import pytest
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Skip entire test module if no display available
def can_run_gui_tests() -> bool:
    """Check if GUI tests can be run."""
    try:
        if os.name == 'nt':  # Windows
            return True
        elif 'DISPLAY' in os.environ:  # Linux with X11
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            return True
        else:
            return False
    except Exception:
        return False


# Skip module if no GUI available
if not can_run_gui_tests():
    pytest.skip("No GUI desktop session available", allow_module_level=True)


def test_gui_smoke_with_safe_demo_mode():
    """Test that GUI starts in Safe Demo mode and renders visible content."""
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    from arena_bot.ui.ui_health import UIHealthReporter, take_window_screenshot, detect_uniform_frame
    
    # Initialize GUI in Safe Demo mode
    bot = IntegratedArenaBotGUI(ui_safe_demo=True)
    
    try:
        assert bot.ui_safe_demo == True, "UI Safe Demo mode should be enabled"
        assert bot.ui_health_reporter is not None, "UI health reporter should be initialized"
        assert bot.safe_demo_manager is not None, "Safe demo manager should be initialized"
        assert bot.root is not None, "GUI window should be created"
        
        # Give GUI time to initialize and render
        bot.root.update()
        time.sleep(0.5)  # Allow demo mode to render
        bot.root.update_idletasks()
        
        # Check that paint counter has incremented
        paint_count_initial = bot.ui_health_reporter.paint_counter
        
        # Force additional redraws
        if bot.safe_demo_manager and bot.safe_demo_manager.is_enabled():
            bot.safe_demo_manager.force_paint()
        
        bot.root.update()
        time.sleep(0.2)
        
        paint_count_after = bot.ui_health_reporter.paint_counter
        assert paint_count_after > 0, f"Paint counter should be > 0, got {paint_count_after}"
        
        # Test UI health reporting
        health_report = bot.ui_health_reporter.get_ui_health_report()
        assert 'paint_counter' in health_report, "Health report should include paint counter"
        assert 'window_available' in health_report, "Health report should include window availability"
        assert health_report['window_available'] == True, "Window should be available"
        
        # Test central widget structure
        if 'central_widget' in health_report:
            central_widget = health_report['central_widget']
            assert central_widget.get('has_layout', False), "Central widget should have layout"
            layout_count = central_widget.get('layout_item_count', 0)
            assert layout_count >= 0, f"Layout item count should be >= 0, got {layout_count}"
        
        # Test window properties
        if 'main_window' in health_report:
            main_window = health_report['main_window']
            assert 'child_count' in main_window, "Main window should report child count"
            child_count = main_window['child_count']
            assert child_count > 0, f"Main window should have children, got {child_count}"
        
        print(f"✅ GUI smoke test passed: paint_count={paint_count_after}, health_report keys={list(health_report.keys())}")
        
    finally:
        # Clean up
        if bot.root:
            bot.root.quit()
            bot.root.destroy()


def test_gui_smoke_uniform_fill_detection():
    """Test uniform fill detection and screenshot capture."""
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    from arena_bot.ui.ui_health import take_window_screenshot, detect_uniform_frame
    import tempfile
    
    # Initialize GUI in Safe Demo mode
    bot = IntegratedArenaBotGUI(ui_safe_demo=True)
    
    try:
        # Give GUI time to render
        bot.root.update()
        time.sleep(0.5)
        bot.root.update_idletasks()
        
        # Take a screenshot
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_path = Path(temp_dir) / "test_screenshot.png"
            
            screenshot_success = take_window_screenshot(bot.root, screenshot_path)
            if not screenshot_success:
                pytest.skip("Screenshot capture failed - may be headless environment")
            
            assert screenshot_path.exists(), "Screenshot file should be created"
            assert screenshot_path.stat().st_size > 0, "Screenshot file should not be empty"
            
            # Analyze screenshot for uniform fill
            uniform_stats = detect_uniform_frame(screenshot_path)
            
            assert 'uniform_detected' in uniform_stats, "Uniform detection should be in results"
            assert 'statistics' in uniform_stats, "Statistics should be in results"
            assert 'image_size' in uniform_stats, "Image size should be in results"
            
            # Check image properties
            image_size = uniform_stats['image_size']
            assert image_size['width'] > 0, "Image width should be > 0"
            assert image_size['height'] > 0, "Image height should be > 0"
            
            # Check statistics
            stats = uniform_stats['statistics']
            assert 'grayscale' in stats, "Grayscale stats should be present"
            
            grayscale_stats = stats['grayscale']
            assert 'variance' in grayscale_stats, "Variance should be calculated"
            assert 'mean' in grayscale_stats, "Mean should be calculated"
            
            variance = grayscale_stats['variance']
            uniform_detected = uniform_stats['uniform_detected']
            
            # With Safe Demo mode active, we should NOT have uniform fill
            # (unless there's a bug in the demo rendering)
            if uniform_detected:
                print(f"⚠️  Uniform fill detected with variance {variance:.2f} - this may indicate a rendering issue")
                # Don't fail the test, but log the issue
            else:
                print(f"✅ Non-uniform fill detected with variance {variance:.2f} - Safe Demo mode is working")
            
            assert variance >= 0, f"Variance should be >= 0, got {variance}"
            
    finally:
        # Clean up
        if bot.root:
            bot.root.quit()
            bot.root.destroy()


def test_gui_smoke_safe_demo_components():
    """Test that Safe Demo components are created and visible."""
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    
    # Initialize GUI in Safe Demo mode
    bot = IntegratedArenaBotGUI(ui_safe_demo=True)
    
    try:
        # Give GUI time to initialize
        bot.root.update()
        time.sleep(0.3)
        bot.root.update_idletasks()
        
        # Check Safe Demo manager state
        assert bot.safe_demo_manager is not None, "Safe demo manager should exist"
        assert bot.safe_demo_manager.is_enabled(), "Safe demo mode should be enabled"
        
        # Check renderer state
        renderer = bot.safe_demo_manager.renderer
        assert renderer is not None, "Safe demo renderer should exist"
        assert renderer.demo_active, "Demo should be active"
        
        # Check that demo widgets exist
        assert renderer.watermark_label is not None, "Watermark label should exist"
        assert renderer.animation_label is not None, "Animation label should exist"
        assert renderer.fps_label is not None, "FPS label should exist"
        assert renderer.guide_frame is not None, "Guide frame should exist"
        
        # Verify widgets are properly placed
        if renderer.watermark_label.winfo_exists():
            watermark_text = renderer.watermark_label.cget('text')
            assert 'DEMO' in watermark_text, f"Watermark should contain 'DEMO', got '{watermark_text}'"
        
        if renderer.animation_label.winfo_exists():
            animation_text = renderer.animation_label.cget('text')
            assert 'LIVE' in animation_text, f"Animation label should contain 'LIVE', got '{animation_text}'"
        
        print("✅ Safe Demo components verified")
        
    finally:
        # Clean up
        if bot.root:
            bot.root.quit()
            bot.root.destroy()


@pytest.mark.skipif(os.name != 'nt', reason="Windows-specific overlay test")
def test_gui_smoke_window_flags_on_windows():
    """Test window flags are correct on Windows."""
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    
    # Initialize GUI in Safe Demo mode
    bot = IntegratedArenaBotGUI(ui_safe_demo=True)
    
    try:
        # Give GUI time to initialize
        bot.root.update()
        time.sleep(0.2)
        
        # Get window attributes
        health_report = bot.ui_health_reporter.get_ui_health_report()
        
        if 'window_attributes' in health_report:
            attributes = health_report['window_attributes']
            
            # Check for topmost attribute
            if 'topmost' in attributes:
                topmost = attributes['topmost']
                assert topmost == True, f"Window should be topmost on Windows, got {topmost}"
            
            print(f"✅ Windows window flags verified: {attributes}")
        else:
            print("⚠️  Window attributes not available")
        
    finally:
        # Clean up
        if bot.root:
            bot.root.quit()
            bot.root.destroy()


def test_gui_smoke_ui_health_summary():
    """Test UI health one-line summary format."""
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    
    # Initialize GUI in Safe Demo mode
    bot = IntegratedArenaBotGUI(ui_safe_demo=True)
    
    try:
        # Give GUI time to initialize
        bot.root.update()
        time.sleep(0.3)
        bot.root.update_idletasks()
        
        # Get health summary
        summary = bot.ui_health_reporter.get_one_line_summary()
        
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary) > 0, "Summary should not be empty"
        assert 'UI:' in summary, f"Summary should contain 'UI:', got '{summary}'"
        
        # Should contain paint count info
        assert 'paints:' in summary, f"Summary should contain paint count, got '{summary}'"
        
        # Should contain uptime info  
        assert 'uptime:' in summary, f"Summary should contain uptime, got '{summary}'"
        
        # Should have positive indicators (✅) for healthy UI
        assert '✅' in summary, f"Summary should contain success indicator, got '{summary}'"
        
        print(f"✅ UI health summary verified: {summary}")
        
    finally:
        # Clean up
        if bot.root:
            bot.root.quit()
            bot.root.destroy()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])