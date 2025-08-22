#!/usr/bin/env python3
"""
Specific validation tests for the exact GUI loading fixes implemented.

This file tests the specific methods and logic that were fixed:
1. setup_gui() - Enhanced error handling with graceful fallback
2. _start_event_polling() - Thread-safe event polling with GUI availability checks  
3. run() - Proper fallback to command-line mode
4. Background loading methods - Thread safety and non-blocking operation

These tests focus on the exact code changes made to fix the blank screen issue.
"""

import pytest
import unittest.mock as mock
import threading
import time
import os
import sys
import queue
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestSpecificGUIFixes:
    """Test suite focused on specific GUI fix methods."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup comprehensive mocks for testing."""
        self.mock_patches = [
            mock.patch('tkinter.Tk'),
            mock.patch('arena_bot.utils.asset_loader.AssetLoader'),
            mock.patch('arena_bot.core.card_recognizer.CardRecognizer'),
            mock.patch('logging_compatibility.get_logger'),
            mock.patch('arena_bot.ai.draft_advisor.DraftAdvisor'),
            mock.patch('cv2.imread'),
        ]
        
        self.mocks = {}
        for patcher in self.mock_patches:
            mock_obj = patcher.start()
            target_name = patcher.attribute or 'unknown'
            self.mocks[target_name] = mock_obj
        
        yield
        
        for patcher in self.mock_patches:
            patcher.stop()
    
    def test_setup_gui_success_path(self):
        """Test setup_gui() success path with proper GUI creation."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock successful tkinter.Tk creation
        mock_root = mock.MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        with mock.patch('builtins.print') as mock_print:
            # Create bot instance
            bot = IntegratedArenaBotGUI()
            
            # Verify setup_gui completed successfully
            assert bot.root is not None
            assert bot.root == mock_root
            
            # Verify GUI was configured properly
            mock_root.title.assert_called_with("🎯 Integrated Arena Bot - Complete GUI")
            mock_root.geometry.assert_called_with("1800x1200")
            mock_root.minsize.assert_called_with(1200, 800)
            mock_root.maxsize.assert_called_with(2560, 1600)
            mock_root.configure.assert_called_with(bg='#2C3E50')
            mock_root.attributes.assert_called_with('-topmost', True)
            
            # Verify success message
            mock_print.assert_any_call("DEBUG: Root window created successfully")
    
    def test_setup_gui_failure_fallback(self):
        """Test setup_gui() graceful fallback when GUI creation fails."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock tkinter.Tk to raise exception
        self.mocks['Tk'].side_effect = Exception("Cannot connect to X server")
        
        with mock.patch('builtins.print') as mock_print:
            # Create bot instance
            bot = IntegratedArenaBotGUI()
            
            # Verify graceful fallback occurred
            assert bot.root is None
            
            # Verify all expected error messages were displayed
            expected_messages = [
                "⚠️ GUI not available: Cannot connect to X server",
                "🔧 This is common in WSL/Linux environments without X11 display server",
                "💡 Application will continue in command-line mode",
                "💡 To enable GUI on WSL: Install X server (VcXsrv/Xming) and set DISPLAY variable"
            ]
            
            print_calls = [str(call) for call in mock_print.call_args_list]
            for message in expected_messages:
                assert any(message in call for call in print_calls), f"Missing message: {message}"
    
    def test_start_event_polling_with_gui(self):
        """Test _start_event_polling() when GUI is available."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock successful GUI setup
        mock_root = mock.MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Create bot and setup required attributes
        bot = IntegratedArenaBotGUI()
        bot.event_polling_active = False
        bot.event_queue = queue.Queue()
        
        with mock.patch('builtins.print') as mock_print:
            # Call the method under test
            bot._start_event_polling()
            
            # Verify event polling was activated
            assert bot.event_polling_active is True
            
            # Verify GUI event scheduling was called
            mock_root.after.assert_called_once_with(50, bot._check_for_events)
            
            # Verify status message
            mock_print.assert_any_call("🔄 Starting event-driven architecture with 50ms polling")
    
    def test_start_event_polling_without_gui(self):
        """Test _start_event_polling() when GUI is not available."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock GUI failure
        self.mocks['Tk'].side_effect = Exception("No display")
        
        # Create bot (will have root=None due to GUI failure)
        bot = IntegratedArenaBotGUI()
        bot.event_polling_active = False
        bot.event_queue = queue.Queue()
        
        # Verify bot has no GUI
        assert bot.root is None
        
        with mock.patch('builtins.print') as mock_print:
            # Call the method under test
            bot._start_event_polling()
            
            # Verify event polling was marked active but no GUI scheduling
            assert bot.event_polling_active is True
            
            # Verify informational message
            mock_print.assert_any_call("ℹ️ Event polling disabled - GUI not available (command-line mode)")
    
    def test_check_for_events_gui_safety(self):
        """Test _check_for_events() safely handles GUI availability."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock successful GUI setup
        mock_root = mock.MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Create bot and setup
        bot = IntegratedArenaBotGUI()
        bot.event_polling_active = True
        bot.event_queue = queue.Queue()
        
        # Test with GUI available
        with mock.patch('builtins.print'):
            bot._check_for_events()
            
            # Verify GUI event scheduling continues
            # Note: after() gets called multiple times during initialization and polling
            assert mock_root.after.called
        
        # Test with GUI suddenly unavailable (edge case)
        bot.root = None
        with mock.patch('builtins.print'):
            # This should not crash
            bot._check_for_events()
    
    def test_run_method_gui_mode(self):
        """Test run() method when GUI is available."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock successful GUI setup
        mock_root = mock.MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Create bot
        bot = IntegratedArenaBotGUI()
        
        # Verify GUI is available
        assert hasattr(bot, 'root')
        assert bot.root is not None
        
        # Call run method
        bot.run()
        
        # Verify mainloop was called
        mock_root.mainloop.assert_called_once()
    
    def test_run_method_command_line_fallback(self):
        """Test run() method falls back to command-line when GUI unavailable."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock GUI failure
        self.mocks['Tk'].side_effect = Exception("No display")
        
        # Create bot (will have root=None)
        bot = IntegratedArenaBotGUI()
        
        # Verify no GUI
        assert not hasattr(bot, 'root') or bot.root is None
        
        # Mock the methods that run() will call
        with mock.patch.object(bot, 'log_text') as mock_log_text, \
             mock.patch.object(bot, 'run_command_line') as mock_run_cli:
            
            # Call run method
            bot.run()
            
            # Verify fallback occurred
            mock_log_text.assert_called_with("❌ GUI not available, running in command-line mode")
            mock_run_cli.assert_called_once()
    
    def test_background_card_database_loading_thread_creation(self):
        """Test _start_card_database_loading() creates background thread."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock asset loader with test data
        mock_asset_loader = mock.MagicMock()
        self.mocks['AssetLoader'].return_value = mock_asset_loader
        
        # Create bot
        bot = IntegratedArenaBotGUI()
        bot.asset_loader = mock_asset_loader
        
        # Track thread creation
        thread_created = threading.Event()
        original_thread_init = threading.Thread.__init__
        
        def track_thread_creation(thread_self, *args, **kwargs):
            if 'Card Database Loader' in str(kwargs.get('name', '')):
                thread_created.set()
            return original_thread_init(thread_self, *args, **kwargs)
        
        with mock.patch.object(threading.Thread, '__init__', side_effect=track_thread_creation), \
             mock.patch('builtins.print'):
            
            # Call the method
            bot._start_card_database_loading()
            
            # Verify thread was created
            assert thread_created.is_set(), "Background thread was not created"
    
    def test_phash_background_loading_thread_safety(self):
        """Test _start_phash_background_loading() thread safety."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Create bot with mocked GUI
        mock_root = mock.MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        bot = IntegratedArenaBotGUI()
        bot.phash_matcher = mock.MagicMock()
        
        # Mock cache manager to trigger background computation
        with mock.patch('arena_bot.detection.phash_cache_manager.get_phash_cache_manager') as mock_cache_mgr:
            mock_cache_instance = mock.MagicMock()
            mock_cache_mgr.return_value = mock_cache_instance
            mock_cache_instance.load_phashes.return_value = None  # No cache, trigger background
            
            # Track thread creation
            thread_created = threading.Event()
            
            def track_thread_start(thread_self):
                if hasattr(thread_self, '_target') and 'phash' in str(thread_self._target):
                    thread_created.set()
                # Don't actually start to avoid complexity
            
            with mock.patch.object(threading.Thread, 'start', side_effect=track_thread_start), \
                 mock.patch('builtins.print'):
                
                # Call the method
                card_images = {"test_card": mock.MagicMock()}
                bot._start_phash_background_loading(card_images)
                
                # Verify background thread was initiated
                assert thread_created.is_set(), "pHash background thread was not created"
    
    def test_initialization_order_and_thread_safety(self):
        """Test that initialization happens in correct order and is thread-safe."""
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Mock successful GUI
        mock_root = mock.MagicMock()
        self.mocks['Tk'].return_value = mock_root
        
        # Track initialization order
        init_order = []
        
        def track_setup_gui(self):
            init_order.append('setup_gui')
            # Mock the original setup_gui behavior
            self.root = mock_root
        
        def track_start_card_loading(self):
            init_order.append('card_loading')
        
        def track_start_phash_loading(self, card_images):
            init_order.append('phash_loading')
        
        with mock.patch('integrated_arena_bot_gui.IntegratedArenaBotGUI.setup_gui', track_setup_gui), \
             mock.patch('integrated_arena_bot_gui.IntegratedArenaBotGUI._start_card_database_loading', track_start_card_loading), \
             mock.patch('integrated_arena_bot_gui.IntegratedArenaBotGUI._start_phash_background_loading', track_start_phash_loading), \
             mock.patch('builtins.print'):
            
            # Create bot
            bot = IntegratedArenaBotGUI()
            
            # Verify initialization order
            assert 'setup_gui' in init_order, "setup_gui was not called"
            assert 'card_loading' in init_order, "card loading was not started"
            
            # Verify GUI was set up before background loading
            setup_gui_index = init_order.index('setup_gui')
            card_loading_index = init_order.index('card_loading')
            assert setup_gui_index < card_loading_index, "GUI setup should happen before background loading"


class TestErrorHandlingSpecifics:
    """Test specific error handling improvements."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup mocks for error handling tests."""
        self.mock_patches = [
            mock.patch('arena_bot.utils.asset_loader.AssetLoader'),
            mock.patch('arena_bot.core.card_recognizer.CardRecognizer'),
            mock.patch('logging_compatibility.get_logger'),
        ]
        
        self.mocks = {}
        for patcher in self.mock_patches:
            mock_obj = patcher.start()
            self.mocks[patcher.attribute or 'unknown'] = mock_obj
        
        yield
        
        for patcher in self.mock_patches:
            patcher.stop()
    
    def test_tkinter_import_error_handling(self):
        """Test handling of tkinter import errors."""
        # Mock tkinter import failure
        with mock.patch('tkinter.Tk', side_effect=ImportError("No module named 'tkinter'")):
            with mock.patch('builtins.print') as mock_print:
                from integrated_arena_bot_gui import IntegratedArenaBotGUI
                
                bot = IntegratedArenaBotGUI()
                
                # Verify graceful fallback
                assert bot.root is None
                
                # Verify error message mentions the specific error
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("GUI not available" in call for call in print_calls)
    
    def test_display_connection_error_handling(self):
        """Test handling of display connection errors."""
        # Mock display connection failure
        with mock.patch('tkinter.Tk', side_effect=Exception("Can't connect to display")):
            with mock.patch('builtins.print') as mock_print:
                from integrated_arena_bot_gui import IntegratedArenaBotGUI
                
                bot = IntegratedArenaBotGUI()
                
                # Verify graceful fallback
                assert bot.root is None
                
                # Verify helpful error messages
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("WSL/Linux environments" in call for call in print_calls)
                assert any("Install X server" in call for call in print_calls)
    
    def test_partial_gui_failure_resilience(self):
        """Test resilience to partial GUI setup failures."""
        # Mock partial GUI failure
        mock_root = mock.MagicMock()
        mock_root.geometry.side_effect = Exception("Geometry setting failed")
        
        with mock.patch('tkinter.Tk', return_value=mock_root):
            with mock.patch('builtins.print'):
                from integrated_arena_bot_gui import IntegratedArenaBotGUI
                
                # This should not crash despite partial failure
                bot = IntegratedArenaBotGUI()
                
                # Root should still be available
                assert bot.root is not None


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=long"])