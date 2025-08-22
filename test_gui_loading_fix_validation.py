#!/usr/bin/env python3
"""
Comprehensive validation tests for GUI loading fix in integrated_arena_bot_gui.py

Tests validate:
1. GUI initialization with and without display server
2. Graceful fallback to command-line mode when GUI fails
3. Thread safety of background operations
4. Error handling and user messaging
5. Performance improvements (startup time)

This test suite ensures the fix works reliably across all environments
(with/without GUI support) and provides proper fallback mechanisms.
"""

import pytest
import unittest.mock as mock
import threading
import time
import os
import sys
import tempfile
import queue
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules under test
try:
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
except ImportError as e:
    pytest.skip(f"Could not import IntegratedArenaBotGUI: {e}", allow_module_level=True)


class TestGUILoadingFix:
    """Test suite for GUI loading fix validation."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment and cleanup after each test."""
        # Store original environment
        self.original_display = os.environ.get('DISPLAY')
        
        # Mock dependencies to avoid actual GUI creation during tests
        self.mock_patches = [
            patch('tkinter.Tk'),
            patch('arena_bot.utils.asset_loader.AssetLoader'),
            patch('arena_bot.core.card_recognizer.CardRecognizer'),
            patch('logging_compatibility.get_logger'),
            patch('arena_bot.ai.draft_advisor.DraftAdvisor'),
        ]
        
        self.mocks = {}
        for patcher in self.mock_patches:
            mock_obj = patcher.start()
            # Store mocks by their target name for easy access
            target_name = patcher.attribute or patcher.new_callable.__name__
            self.mocks[target_name] = mock_obj
        
        yield
        
        # Cleanup patches
        for patcher in self.mock_patches:
            patcher.stop()
        
        # Restore environment
        if self.original_display is not None:
            os.environ['DISPLAY'] = self.original_display
        elif 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']

    def test_gui_initialization_success(self):
        """Test successful GUI initialization when display server is available."""
        # Mock successful tkinter.Tk creation
        mock_root = MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Mock logger to capture initialization messages
        mock_logger = MagicMock()
        self.mocks['get_logger'].return_value = mock_logger
        
        # Set display environment to simulate GUI availability
        os.environ['DISPLAY'] = ':0'
        
        with patch('builtins.print') as mock_print:
            # Create bot instance - this should succeed
            bot = IntegratedArenaBotGUI()
            
            # Verify GUI was initialized properly
            assert bot.root is not None
            assert bot.root == mock_root
            
            # Verify setup_gui was called and root was created
            self.mocks['Tk'].assert_called_once()
            
            # Verify success message was printed
            mock_print.assert_any_call("DEBUG: Root window created successfully")

    def test_gui_initialization_failure_fallback(self):
        """Test graceful fallback to command-line mode when GUI initialization fails."""
        # Mock tkinter.Tk to raise exception (simulating no display server)
        self.mocks['Tk'].side_effect = Exception("No display server available")
        
        # Mock logger
        mock_logger = MagicMock()
        self.mocks['get_logger'].return_value = mock_logger
        
        # Remove display environment to simulate headless environment
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
        
        with patch('builtins.print') as mock_print:
            # Create bot instance - should fallback gracefully
            bot = IntegratedArenaBotGUI()
            
            # Verify GUI fallback occurred
            assert bot.root is None
            
            # Verify error messages were displayed
            error_calls = [call for call in mock_print.call_args_list 
                          if any("GUI not available" in str(arg) for arg in call[0])]
            assert len(error_calls) > 0
            
            # Verify helpful instructions were provided
            instruction_calls = [call for call in mock_print.call_args_list 
                               if any("command-line mode" in str(arg) for arg in call[0])]
            assert len(instruction_calls) > 0

    def test_event_polling_gui_available(self):
        """Test event polling starts correctly when GUI is available."""
        # Mock successful GUI setup
        mock_root = MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Create bot and setup
        bot = IntegratedArenaBotGUI()
        bot.event_polling_active = False
        bot.event_queue = queue.Queue()
        
        # Call _start_event_polling
        bot._start_event_polling()
        
        # Verify event polling was activated
        assert bot.event_polling_active is True
        
        # Verify root.after was called to schedule event checking
        mock_root.after.assert_called_once_with(50, bot._check_for_events)

    def test_event_polling_gui_unavailable(self):
        """Test event polling handles GUI unavailable gracefully."""
        # Setup bot without GUI
        bot = IntegratedArenaBotGUI()
        bot.root = None  # Simulate GUI unavailable
        bot.event_polling_active = False
        bot.event_queue = queue.Queue()
        
        with patch('builtins.print') as mock_print:
            # Call _start_event_polling
            bot._start_event_polling()
            
            # Verify event polling was marked active but no scheduling occurred
            assert bot.event_polling_active is True
            
            # Verify informational message was displayed
            info_calls = [call for call in mock_print.call_args_list 
                         if any("Event polling disabled" in str(arg) for arg in call[0])]
            assert len(info_calls) > 0

    def test_run_method_gui_mode(self):
        """Test run method with GUI available."""
        # Mock successful GUI setup
        mock_root = MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Create bot
        bot = IntegratedArenaBotGUI()
        
        # Mock mainloop to prevent actual GUI loop
        mock_root.mainloop = MagicMock()
        
        # Call run method
        bot.run()
        
        # Verify mainloop was called
        mock_root.mainloop.assert_called_once()

    def test_run_method_command_line_fallback(self):
        """Test run method falls back to command-line mode when GUI unavailable."""
        # Setup bot without GUI
        bot = IntegratedArenaBotGUI()
        bot.root = None  # Simulate GUI unavailable
        
        # Mock run_command_line method
        with patch.object(bot, 'run_command_line') as mock_run_cli, \
             patch.object(bot, 'log_text') as mock_log_text:
            
            # Call run method
            bot.run()
            
            # Verify command-line mode was started
            mock_run_cli.assert_called_once()
            
            # Verify error message was logged
            mock_log_text.assert_called_with("❌ GUI not available, running in command-line mode")

    def test_background_database_loading_thread_safety(self):
        """Test that background database loading is thread-safe and doesn't block GUI."""
        # Mock asset loader
        mock_asset_loader = MagicMock()
        self.mocks['AssetLoader'].return_value = mock_asset_loader
        
        # Create temporary cards directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            cards_dir = Path(temp_dir) / "cards"
            cards_dir.mkdir()
            
            # Create test card files
            for i in range(5):
                card_file = cards_dir / f"test_card_{i}.png"
                card_file.write_bytes(b"fake_png_data")
            
            mock_asset_loader.get_cards_directory.return_value = cards_dir
            
            # Mock cv2.imread to return test images
            with patch('cv2.imread') as mock_imread:
                mock_imread.return_value = MagicMock()  # Mock image array
                
                # Create bot
                bot = IntegratedArenaBotGUI()
                bot.asset_loader = mock_asset_loader
                
                # Track thread creation
                original_thread_init = threading.Thread.__init__
                thread_creation_count = 0
                
                def track_thread_creation(thread_self, *args, **kwargs):
                    nonlocal thread_creation_count
                    thread_creation_count += 1
                    return original_thread_init(thread_self, *args, **kwargs)
                
                with patch.object(threading.Thread, '__init__', side_effect=track_thread_creation):
                    # Start background loading
                    bot._start_card_database_loading()
                    
                    # Wait briefly for thread to start
                    time.sleep(0.1)
                    
                    # Verify background thread was created
                    assert thread_creation_count > 0

    def test_phash_background_loading_thread_safety(self):
        """Test that pHash background loading doesn't block the main thread."""
        # Mock pHash matcher
        mock_phash_matcher = MagicMock()
        
        # Create bot and setup pHash matcher
        bot = IntegratedArenaBotGUI()
        bot.phash_matcher = mock_phash_matcher
        bot.root = MagicMock()  # Mock GUI available
        
        # Mock cache manager
        with patch('arena_bot.detection.phash_cache_manager.get_phash_cache_manager') as mock_cache_mgr:
            mock_cache_instance = MagicMock()
            mock_cache_mgr.return_value = mock_cache_instance
            
            # Mock empty cache to trigger background computation
            mock_cache_instance.load_phashes.return_value = None
            
            # Track thread creation
            thread_created = threading.Event()
            original_thread_start = threading.Thread.start
            
            def track_thread_start(thread_self):
                thread_created.set()
                # Don't actually start the thread to avoid complexity
                pass
            
            with patch.object(threading.Thread, 'start', side_effect=track_thread_start):
                # Start pHash background loading
                card_images = {"test_card": MagicMock()}
                bot._start_phash_background_loading(card_images)
                
                # Verify thread creation was initiated
                assert thread_created.is_set()

    def test_startup_performance_benchmarks(self):
        """Test that startup performance is within acceptable bounds."""
        # Mock dependencies for fast startup
        mock_root = MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        mock_asset_loader = MagicMock()
        self.mocks['AssetLoader'].return_value = mock_asset_loader
        
        # Mock empty cards directory for fast loading
        with tempfile.TemporaryDirectory() as temp_dir:
            cards_dir = Path(temp_dir) / "cards"
            cards_dir.mkdir()
            mock_asset_loader.get_cards_directory.return_value = cards_dir
            
            # Measure initialization time
            start_time = time.time()
            
            # Create bot instance
            bot = IntegratedArenaBotGUI()
            
            initialization_time = time.time() - start_time
            
            # Verify initialization is reasonably fast (under 5 seconds)
            assert initialization_time < 5.0, f"Initialization took {initialization_time:.3f}s, expected < 5.0s"
            
            # Verify GUI was created
            assert bot.root is not None

    def test_command_line_mode_functionality(self):
        """Test command-line mode provides proper functionality."""
        # Setup bot without GUI
        bot = IntegratedArenaBotGUI()
        bot.root = None
        bot.running = False
        
        # Mock input/output for command-line interaction
        with patch('builtins.input', side_effect=['start', 'stop', 'quit']), \
             patch('builtins.print') as mock_print, \
             patch.object(bot, 'toggle_monitoring') as mock_toggle, \
             patch.object(bot, 'stop') as mock_stop:
            
            # Run command-line mode
            bot.run_command_line()
            
            # Verify commands were processed
            assert mock_toggle.call_count == 2  # start and stop
            mock_stop.assert_called_once()
            
            # Verify help text was displayed
            help_calls = [call for call in mock_print.call_args_list 
                         if any("Commands:" in str(arg) for arg in call[0])]
            assert len(help_calls) > 0

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling throughout the initialization process."""
        # Test various failure scenarios
        
        # Scenario 1: Tkinter import failure
        with patch('tkinter.Tk', side_effect=ImportError("No tkinter module")):
            with patch('builtins.print') as mock_print:
                bot = IntegratedArenaBotGUI()
                assert bot.root is None
                
                error_calls = [call for call in mock_print.call_args_list 
                             if any("GUI not available" in str(arg) for arg in call[0])]
                assert len(error_calls) > 0

        # Scenario 2: Display server connection failure
        with patch('tkinter.Tk', side_effect=Exception("Can't connect to display server")):
            with patch('builtins.print') as mock_print:
                bot = IntegratedArenaBotGUI()
                assert bot.root is None
                
                instruction_calls = [call for call in mock_print.call_args_list 
                                   if any("X server" in str(arg) for arg in call[0])]
                assert len(instruction_calls) > 0

    def test_gui_setup_graceful_degradation(self):
        """Test that GUI setup degrades gracefully when components fail."""
        # Mock partial GUI failure
        mock_root = MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Make some GUI operations fail
        mock_root.geometry.side_effect = Exception("Geometry setting failed")
        
        # Bot should still initialize successfully
        bot = IntegratedArenaBotGUI()
        
        # Root should still be available even if some setup failed
        assert bot.root is not None

    def test_background_operations_isolation(self):
        """Test that background operations don't interfere with main thread."""
        # Create bot with mocked dependencies
        bot = IntegratedArenaBotGUI()
        bot.asset_loader = MagicMock()
        bot.phash_matcher = MagicMock()
        
        # Mock cards directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cards_dir = Path(temp_dir) / "cards" 
            cards_dir.mkdir()
            bot.asset_loader.get_cards_directory.return_value = cards_dir
            
            # Start multiple background operations
            main_thread_id = threading.get_ident()
            
            # Track if background operations run in different threads
            background_thread_ids = set()
            
            def track_thread_execution():
                background_thread_ids.add(threading.get_ident())
                time.sleep(0.1)  # Simulate work
            
            with patch.object(bot, '_load_card_database', side_effect=track_thread_execution), \
                 patch.object(bot, '_load_phash_database_background', side_effect=track_thread_execution):
                
                # Start background operations
                bot._start_card_database_loading()
                bot._start_phash_background_loading({"test": MagicMock()})
                
                # Wait for background operations to complete
                time.sleep(0.5)
                
                # Verify operations ran in background threads
                assert len(background_thread_ids) > 0
                assert main_thread_id not in background_thread_ids


class TestPerformanceMetrics:
    """Performance validation tests for GUI loading improvements."""
    
    def test_startup_time_improvement(self):
        """Test that startup time is improved with background loading."""
        with patch('integrated_arena_bot_gui.IntegratedArenaBotGUI._start_card_database_loading'), \
             patch('integrated_arena_bot_gui.IntegratedArenaBotGUI._start_phash_background_loading'), \
             patch('tkinter.Tk') as mock_tk:
            
            mock_root = MagicMock()
            mock_tk.return_value = mock_root
            
            # Measure initialization time with background loading
            start_time = time.time()
            bot = IntegratedArenaBotGUI()
            end_time = time.time()
            
            startup_time = end_time - start_time
            
            # Startup should be fast since database loading is backgrounded
            assert startup_time < 2.0, f"Startup took {startup_time:.3f}s, expected < 2.0s"

    def test_memory_usage_optimization(self):
        """Test that memory usage is optimized during initialization."""
        # This is a basic test - in practice you'd use memory profiling tools
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create bot instance
        with patch('tkinter.Tk') as mock_tk:
            mock_tk.return_value = MagicMock()
            bot = IntegratedArenaBotGUI()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 100MB for basic initialization)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB, expected < 100MB"


class TestEnvironmentCompatibility:
    """Test compatibility across different environments."""
    
    @pytest.mark.parametrize("display_value,expected_gui", [
        (":0", True),           # Standard X11 display
        (":1.0", True),         # Secondary display
        ("localhost:10.0", True), # SSH forwarding
        (None, False),          # No display set
        ("", False),            # Empty display
    ])
    def test_display_environment_handling(self, display_value, expected_gui):
        """Test handling of different DISPLAY environment values."""
        # Setup environment
        if display_value is not None:
            os.environ['DISPLAY'] = display_value
        elif 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
        
        # Mock tkinter behavior based on display availability
        if expected_gui:
            mock_root = MagicMock()
            with patch('tkinter.Tk', return_value=mock_root):
                bot = IntegratedArenaBotGUI()
                assert bot.root is not None
        else:
            with patch('tkinter.Tk', side_effect=Exception("No display")):
                bot = IntegratedArenaBotGUI()
                assert bot.root is None

    def test_wsl_environment_handling(self):
        """Test proper handling of WSL environment without X11."""
        # Simulate WSL environment
        original_platform = sys.platform
        try:
            sys.platform = "linux"
            if 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']
            
            with patch('tkinter.Tk', side_effect=Exception("No X11 forwarding")), \
                 patch('builtins.print') as mock_print:
                
                bot = IntegratedArenaBotGUI()
                
                # Verify WSL-specific guidance was provided
                wsl_calls = [call for call in mock_print.call_args_list 
                           if any("WSL" in str(arg) for arg in call[0])]
                assert len(wsl_calls) > 0
                
        finally:
            sys.platform = original_platform


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])