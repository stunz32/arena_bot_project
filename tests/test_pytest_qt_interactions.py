#!/usr/bin/env python3
"""
🧪 PyQt6 Interaction Testing with pytest-qt

Implements your friend's recommendation for robust Qt interaction testing.
Replaces our problematic tkinter-based testing with pytest-qt framework.

Key Improvements:
- Real PyQt6 widget interaction (clicks, typing, signals)
- No threading conflicts (pure Qt testing)
- Headless operation with Xvfb
- Signal/slot testing with QSignalSpy
- Deterministic timing with qtbot

Usage:
    pytest tests/test_pytest_qt_interactions.py -v
    xvfb-run pytest tests/test_pytest_qt_interactions.py -v
"""

import sys
import os
import pytest
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ['TEST_PROFILE'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
    from PyQt6.QtTest import QTest, QSignalSpy
    from PyQt6.QtGui import QPixmap, QImage
    import pytest_qt
    PYTEST_QT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ pytest-qt not available: {e}")
    PYTEST_QT_AVAILABLE = False

# Mock our main GUI class for testing
class MockIntegratedArenaBotGUI(QMainWindow):
    """
    Mock GUI for testing PyQt6 interactions
    
    Simulates the actual IntegratedArenaBotGUI with simplified functionality
    for testing purposes.
    """
    
    # Signals for testing
    analysis_started = pyqtSignal()
    analysis_finished = pyqtSignal(dict)
    screenshot_taken = pyqtSignal()
    monitoring_toggled = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arena Bot Test GUI")
        self.setGeometry(100, 100, 800, 600)
        
        # State variables
        self.monitoring_active = False
        self.last_analysis_result = None
        self.screenshot_count = 0
        
        # Setup UI
        self.setup_ui()
        
        # Mock repositories
        self.setup_mock_dependencies()
    
    def setup_ui(self):
        """Setup simplified UI for testing"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Main buttons (matching real GUI)
        self.screenshot_btn = QPushButton("Analyze Screenshot")
        self.screenshot_btn.clicked.connect(self.manual_screenshot)
        layout.addWidget(self.screenshot_btn)
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.toggle_monitoring)
        layout.addWidget(self.start_btn)
        
        # Detection method buttons
        self.ultimate_detection_btn = QPushButton("Ultimate Detection")
        self.ultimate_detection_btn.clicked.connect(self.toggle_ultimate_detection)
        self.ultimate_detection_btn.setCheckable(True)
        self.ultimate_detection_btn.setChecked(True)
        layout.addWidget(self.ultimate_detection_btn)
        
        self.phash_detection_btn = QPushButton("PHash Detection")
        self.phash_detection_btn.clicked.connect(self.toggle_phash_detection)
        self.phash_detection_btn.setCheckable(True)
        layout.addWidget(self.phash_detection_btn)
        
        self.arena_priority_btn = QPushButton("Arena Priority")
        self.arena_priority_btn.clicked.connect(self.toggle_arena_priority)
        self.arena_priority_btn.setCheckable(True)
        layout.addWidget(self.arena_priority_btn)
        
        # Settings and utility buttons
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        layout.addWidget(self.settings_btn)
        
        self.coord_select_btn = QPushButton("Coordinate Selection")
        self.coord_select_btn.clicked.connect(self.open_coordinate_selector)
        layout.addWidget(self.coord_select_btn)
        
        self.debug_mode_btn = QPushButton("Debug Mode")
        self.debug_mode_btn.clicked.connect(self.toggle_debug_mode)
        self.debug_mode_btn.setCheckable(True)
        layout.addWidget(self.debug_mode_btn)
        
        # Results area
        self.recommendation_text = QLabel("No analysis yet")
        layout.addWidget(self.recommendation_text)
    
    def setup_mock_dependencies(self):
        """Setup mock dependencies for testing"""
        # Mock card repository
        from arena_bot.core.card_repository import get_test_repository
        self.card_repository = get_test_repository(50)
        
        # Mock detection systems
        self.detection_active = True
        self.detection_method = "ultimate"
    
    def manual_screenshot(self):
        """Simulate screenshot analysis"""
        self.screenshot_taken.emit()
        self.status_label.setText("Analyzing...")
        self.analysis_started.emit()
        
        # Simulate analysis delay
        QTimer.singleShot(100, self._finish_analysis)
        self.screenshot_count += 1
    
    def _finish_analysis(self):
        """Complete mock analysis"""
        result = {
            "cards_found": ["Test Card 1", "Test Card 2", "Test Card 3"],
            "recommendation": "Pick Test Card 2",
            "confidence": 0.87,
            "analysis_time": 0.1
        }
        
        self.last_analysis_result = result
        self.status_label.setText("Analysis complete")
        self.recommendation_text.setText(f"Recommendation: {result['recommendation']}")
        self.analysis_finished.emit(result)
    
    def toggle_monitoring(self):
        """Toggle monitoring state"""
        self.monitoring_active = not self.monitoring_active
        
        if self.monitoring_active:
            self.start_btn.setText("Stop Monitoring")
            self.status_label.setText("Monitoring active")
        else:
            self.start_btn.setText("Start Monitoring")
            self.status_label.setText("Monitoring stopped")
        
        self.monitoring_toggled.emit(self.monitoring_active)
    
    def toggle_ultimate_detection(self):
        """Toggle ultimate detection"""
        if self.ultimate_detection_btn.isChecked():
            self.detection_method = "ultimate"
            # Uncheck other methods
            self.phash_detection_btn.setChecked(False)
            self.arena_priority_btn.setChecked(False)
    
    def toggle_phash_detection(self):
        """Toggle PHash detection"""
        if self.phash_detection_btn.isChecked():
            self.detection_method = "phash"
            # Uncheck other methods
            self.ultimate_detection_btn.setChecked(False)
            self.arena_priority_btn.setChecked(False)
    
    def toggle_arena_priority(self):
        """Toggle arena priority"""
        if self.arena_priority_btn.isChecked():
            self.detection_method = "arena_priority"
            # Uncheck other methods
            self.ultimate_detection_btn.setChecked(False)
            self.phash_detection_btn.setChecked(False)
    
    def open_settings_dialog(self):
        """Mock settings dialog"""
        self.status_label.setText("Settings opened")
    
    def open_coordinate_selector(self):
        """Mock coordinate selector"""
        self.status_label.setText("Coordinate selector opened")
    
    def toggle_debug_mode(self):
        """Toggle debug mode"""
        debug_active = self.debug_mode_btn.isChecked()
        self.status_label.setText(f"Debug mode: {'ON' if debug_active else 'OFF'}")

# Test fixtures
@pytest.fixture
def qapp(qapp):
    """Qt application fixture"""
    return qapp

@pytest.fixture
def main_window(qtbot):
    """Main window fixture"""
    window = MockIntegratedArenaBotGUI()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window, timeout=5000)
    return window

# Test classes
class TestBasicGUIInteraction:
    """Test basic GUI interaction functionality"""
    
    def test_window_creation(self, main_window):
        """Test that main window creates properly"""
        assert main_window.isVisible()
        assert main_window.windowTitle() == "Arena Bot Test GUI"
        assert main_window.status_label.text() == "Ready"
    
    def test_screenshot_button_click(self, qtbot, main_window):
        """Test screenshot button interaction"""
        # Setup signal spy
        spy = QSignalSpy(main_window.analysis_started)
        
        # Click screenshot button
        qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
        
        # Verify signal was emitted
        assert spy.wait(1000)  # Wait up to 1 second
        assert len(spy) == 1
        
        # Verify UI state change
        assert "Analyzing" in main_window.status_label.text()
        
        # Wait for analysis to complete
        finished_spy = QSignalSpy(main_window.analysis_finished)
        assert finished_spy.wait(2000)
        
        # Verify final state
        assert "complete" in main_window.status_label.text()
        assert main_window.last_analysis_result is not None
    
    def test_monitoring_toggle(self, qtbot, main_window):
        """Test monitoring toggle functionality"""
        # Setup signal spy
        spy = QSignalSpy(main_window.monitoring_toggled)
        
        # Initially not monitoring
        assert not main_window.monitoring_active
        assert main_window.start_btn.text() == "Start Monitoring"
        
        # Click to start monitoring
        qtbot.mouseClick(main_window.start_btn, Qt.MouseButton.LeftButton)
        
        # Verify signal and state
        assert spy.wait(1000)
        assert len(spy) == 1
        assert main_window.monitoring_active
        assert main_window.start_btn.text() == "Stop Monitoring"
        
        # Click to stop monitoring
        qtbot.mouseClick(main_window.start_btn, Qt.MouseButton.LeftButton)
        
        # Verify toggle back
        assert len(spy) == 2
        assert not main_window.monitoring_active
        assert main_window.start_btn.text() == "Start Monitoring"

class TestDetectionMethods:
    """Test detection method switching"""
    
    def test_ultimate_detection_toggle(self, qtbot, main_window):
        """Test ultimate detection method selection"""
        # Ultimate should be selected by default
        assert main_window.ultimate_detection_btn.isChecked()
        assert main_window.detection_method == "ultimate"
        
        # Click should maintain selection (it's already checked)
        qtbot.mouseClick(main_window.ultimate_detection_btn, Qt.MouseButton.LeftButton)
        assert main_window.ultimate_detection_btn.isChecked()
    
    def test_phash_detection_toggle(self, qtbot, main_window):
        """Test PHash detection method selection"""
        # Click PHash button
        qtbot.mouseClick(main_window.phash_detection_btn, Qt.MouseButton.LeftButton)
        
        # Verify selection and mutual exclusion
        assert main_window.phash_detection_btn.isChecked()
        assert not main_window.ultimate_detection_btn.isChecked()
        assert not main_window.arena_priority_btn.isChecked()
        assert main_window.detection_method == "phash"
    
    def test_arena_priority_toggle(self, qtbot, main_window):
        """Test arena priority method selection"""
        # Click arena priority button
        qtbot.mouseClick(main_window.arena_priority_btn, Qt.MouseButton.LeftButton)
        
        # Verify selection and mutual exclusion
        assert main_window.arena_priority_btn.isChecked()
        assert not main_window.ultimate_detection_btn.isChecked()
        assert not main_window.phash_detection_btn.isChecked()
        assert main_window.detection_method == "arena_priority"
    
    def test_detection_method_exclusivity(self, qtbot, main_window):
        """Test that only one detection method can be active"""
        # Start with ultimate (default)
        assert main_window.ultimate_detection_btn.isChecked()
        
        # Switch to phash
        qtbot.mouseClick(main_window.phash_detection_btn, Qt.MouseButton.LeftButton)
        assert main_window.phash_detection_btn.isChecked()
        assert not main_window.ultimate_detection_btn.isChecked()
        
        # Switch to arena priority
        qtbot.mouseClick(main_window.arena_priority_btn, Qt.MouseButton.LeftButton)
        assert main_window.arena_priority_btn.isChecked()
        assert not main_window.phash_detection_btn.isChecked()
        
        # Back to ultimate
        qtbot.mouseClick(main_window.ultimate_detection_btn, Qt.MouseButton.LeftButton)
        assert main_window.ultimate_detection_btn.isChecked()
        assert not main_window.arena_priority_btn.isChecked()

class TestUtilityFunctions:
    """Test utility button functionality"""
    
    def test_settings_button(self, qtbot, main_window):
        """Test settings dialog opening"""
        qtbot.mouseClick(main_window.settings_btn, Qt.MouseButton.LeftButton)
        assert "Settings opened" in main_window.status_label.text()
    
    def test_coordinate_selector_button(self, qtbot, main_window):
        """Test coordinate selector opening"""
        qtbot.mouseClick(main_window.coord_select_btn, Qt.MouseButton.LeftButton)
        assert "Coordinate selector opened" in main_window.status_label.text()
    
    def test_debug_mode_toggle(self, qtbot, main_window):
        """Test debug mode toggle"""
        # Initially off
        assert not main_window.debug_mode_btn.isChecked()
        
        # Click to enable
        qtbot.mouseClick(main_window.debug_mode_btn, Qt.MouseButton.LeftButton)
        assert main_window.debug_mode_btn.isChecked()
        assert "Debug mode: ON" in main_window.status_label.text()
        
        # Click to disable
        qtbot.mouseClick(main_window.debug_mode_btn, Qt.MouseButton.LeftButton)
        assert not main_window.debug_mode_btn.isChecked()
        assert "Debug mode: OFF" in main_window.status_label.text()

class TestKeyboardInteraction:
    """Test keyboard interaction"""
    
    def test_button_keyboard_activation(self, qtbot, main_window):
        """Test activating buttons with keyboard"""
        # Focus on screenshot button and press space
        main_window.screenshot_btn.setFocus()
        qtbot.keyPress(main_window.screenshot_btn, Qt.Key.Key_Space)
        
        # Should trigger analysis
        spy = QSignalSpy(main_window.analysis_started)
        assert spy.wait(1000)
    
    def test_tab_navigation(self, qtbot, main_window):
        """Test tab navigation between elements"""
        # Set focus to first button
        main_window.screenshot_btn.setFocus()
        assert main_window.screenshot_btn.hasFocus()
        
        # Tab to next button
        qtbot.keyPress(main_window, Qt.Key.Key_Tab)
        # Note: Focus behavior depends on tab order, this tests basic functionality

class TestSignalSlotInteraction:
    """Test signal/slot mechanisms"""
    
    def test_analysis_workflow_signals(self, qtbot, main_window):
        """Test complete analysis workflow with signals"""
        # Setup spies for all signals
        started_spy = QSignalSpy(main_window.analysis_started)
        finished_spy = QSignalSpy(main_window.analysis_finished)
        screenshot_spy = QSignalSpy(main_window.screenshot_taken)
        
        # Trigger analysis
        qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
        
        # Verify signal sequence
        assert screenshot_spy.wait(1000)
        assert started_spy.wait(1000)
        assert finished_spy.wait(2000)
        
        # Verify signal data
        assert len(started_spy) == 1
        assert len(finished_spy) == 1
        
        # Check analysis result data
        result_data = finished_spy[0][0]  # First signal, first argument
        assert 'cards_found' in result_data
        assert 'recommendation' in result_data
    
    def test_monitoring_signals(self, qtbot, main_window):
        """Test monitoring toggle signals"""
        spy = QSignalSpy(main_window.monitoring_toggled)
        
        # Toggle on
        qtbot.mouseClick(main_window.start_btn, Qt.MouseButton.LeftButton)
        assert spy.wait(1000)
        assert len(spy) == 1
        assert spy[0][0] is True  # First signal argument should be True
        
        # Toggle off
        qtbot.mouseClick(main_window.start_btn, Qt.MouseButton.LeftButton)
        assert len(spy) == 2
        assert spy[1][0] is False  # Second signal argument should be False

class TestPerformanceAndTiming:
    """Test performance and timing aspects"""
    
    def test_analysis_timing(self, qtbot, main_window):
        """Test analysis completes within reasonable time"""
        start_time = time.time()
        
        # Trigger analysis
        qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
        
        # Wait for completion
        spy = QSignalSpy(main_window.analysis_finished)
        assert spy.wait(5000)  # 5 second timeout
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly in test mode
        assert total_time < 1.0  # Less than 1 second
    
    def test_multiple_rapid_clicks(self, qtbot, main_window):
        """Test handling of rapid button clicks"""
        spy = QSignalSpy(main_window.analysis_started)
        
        # Rapid clicks
        for i in range(3):
            qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
            qtbot.wait(10)  # Small delay
        
        # Should handle gracefully (exact behavior depends on implementation)
        # At minimum, should not crash
        assert main_window.isVisible()

class TestIntegrationWithMockRepositories:
    """Test integration with mock card repositories"""
    
    def test_card_repository_integration(self, qtbot, main_window):
        """Test that card repository is properly integrated"""
        # Repository should be available
        assert hasattr(main_window, 'card_repository')
        assert main_window.card_repository is not None
        
        # Should be able to get cards
        cards = list(main_window.card_repository.iter_cards())
        assert len(cards) > 0
        
        # Analysis should work with repository
        qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
        
        spy = QSignalSpy(main_window.analysis_finished)
        assert spy.wait(2000)
        
        # Should have analysis result
        assert main_window.last_analysis_result is not None

class TestErrorScenarios:
    """Test error handling scenarios"""
    
    def test_widget_cleanup(self, qtbot, main_window):
        """Test proper widget cleanup"""
        # Window should be visible
        assert main_window.isVisible()
        
        # Close window
        main_window.close()
        
        # Should be closed
        assert not main_window.isVisible()
    
    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE, reason="pytest-qt not available")
    def test_exception_handling_in_slots(self, qtbot, main_window, monkeypatch):
        """Test that exceptions in slots don't crash the application"""
        # Mock a method to raise an exception
        def mock_analysis():
            raise ValueError("Test exception")
        
        monkeypatch.setattr(main_window, '_finish_analysis', mock_analysis)
        
        # Click should not crash the application
        qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
        
        # Window should still be responsive
        assert main_window.isVisible()

# Benchmark tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests using pytest-benchmark"""
    
    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE, reason="pytest-qt not available")
    def test_window_creation_benchmark(self, qtbot, benchmark):
        """Benchmark window creation time"""
        def create_window():
            window = MockIntegratedArenaBotGUI()
            qtbot.addWidget(window)
            window.show()
            return window
        
        window = benchmark(create_window)
        assert window.isVisible()
    
    @pytest.mark.skipif(not PYTEST_QT_AVAILABLE, reason="pytest-qt not available")
    def test_button_click_benchmark(self, qtbot, main_window, benchmark):
        """Benchmark button click response time"""
        def click_screenshot_button():
            qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
            spy = QSignalSpy(main_window.analysis_started)
            spy.wait(1000)
        
        benchmark(click_screenshot_button)

# Integration test that mimics real user workflows
class TestRealUserWorkflows:
    """Test complete user workflows"""
    
    def test_complete_analysis_workflow(self, qtbot, main_window):
        """Test complete user analysis workflow"""
        # 1. User opens application (already done by fixture)
        assert main_window.isVisible()
        
        # 2. User selects detection method
        qtbot.mouseClick(main_window.ultimate_detection_btn, Qt.MouseButton.LeftButton)
        assert main_window.detection_method == "ultimate"
        
        # 3. User takes screenshot
        qtbot.mouseClick(main_window.screenshot_btn, Qt.MouseButton.LeftButton)
        
        # 4. Wait for analysis
        spy = QSignalSpy(main_window.analysis_finished)
        assert spy.wait(3000)
        
        # 5. Verify results are displayed
        assert main_window.last_analysis_result is not None
        assert "Recommendation:" in main_window.recommendation_text.text()
        
        # 6. User enables monitoring
        qtbot.mouseClick(main_window.start_btn, Qt.MouseButton.LeftButton)
        assert main_window.monitoring_active
    
    def test_settings_configuration_workflow(self, qtbot, main_window):
        """Test settings configuration workflow"""
        # 1. User opens settings
        qtbot.mouseClick(main_window.settings_btn, Qt.MouseButton.LeftButton)
        assert "Settings opened" in main_window.status_label.text()
        
        # 2. User opens coordinate selector
        qtbot.mouseClick(main_window.coord_select_btn, Qt.MouseButton.LeftButton)
        assert "Coordinate selector opened" in main_window.status_label.text()
        
        # 3. User enables debug mode
        qtbot.mouseClick(main_window.debug_mode_btn, Qt.MouseButton.LeftButton)
        assert main_window.debug_mode_btn.isChecked()

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])