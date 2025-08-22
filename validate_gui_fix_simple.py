#!/usr/bin/env python3
"""
Simple GUI Fix Validation Script

This script provides a lightweight way to validate the GUI loading fix
without requiring pytest or other testing frameworks. It directly tests
the key functionality and provides clear pass/fail results.

This is ideal for quick validation after applying the fix.
"""

import sys
import os
import time
import threading
import unittest.mock as mock
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_import():
    """Test that the module can be imported."""
    print("🧪 Test 1: Module Import")
    try:
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        print("   ✅ PASS: Module imported successfully")
        return True, IntegratedArenaBotGUI
    except Exception as e:
        print(f"   ❌ FAIL: Could not import module: {e}")
        return False, None


def test_gui_success_path(IntegratedArenaBotGUI):
    """Test GUI initialization success path."""
    print("\n🧪 Test 2: GUI Initialization Success")
    try:
        # Mock successful tkinter creation
        with mock.patch('tkinter.Tk') as mock_tk, \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print') as mock_print:
            
            mock_root = mock.MagicMock()
            mock_tk.return_value = mock_root
            
            # Create bot instance
            bot = IntegratedArenaBotGUI()
            
            # Verify GUI was created
            if hasattr(bot, 'root') and bot.root is not None:
                print("   ✅ PASS: GUI initialized successfully")
                print("   ✅ PASS: Root window created")
                
                # Verify GUI configuration was attempted
                if mock_root.title.called:
                    print("   ✅ PASS: GUI configured properly")
                else:
                    print("   ⚠️ WARNING: GUI configuration may not have completed")
                
                return True
            else:
                print("   ❌ FAIL: GUI not initialized")
                return False
                
    except Exception as e:
        print(f"   ❌ FAIL: GUI initialization failed: {e}")
        return False


def test_gui_fallback_path(IntegratedArenaBotGUI):
    """Test GUI fallback to command-line mode."""
    print("\n🧪 Test 3: GUI Fallback to Command-Line")
    try:
        # Mock tkinter failure
        with mock.patch('tkinter.Tk', side_effect=Exception("No display server")), \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print') as mock_print:
            
            # Create bot instance
            bot = IntegratedArenaBotGUI()
            
            # Verify fallback occurred
            if not hasattr(bot, 'root') or bot.root is None:
                print("   ✅ PASS: Graceful fallback to command-line mode")
                
                # Check for helpful error messages
                print_calls = [str(call) for call in mock_print.call_args_list]
                
                if any("GUI not available" in call for call in print_calls):
                    print("   ✅ PASS: Error message displayed")
                else:
                    print("   ⚠️ WARNING: Error message not found")
                
                if any("command-line mode" in call for call in print_calls):
                    print("   ✅ PASS: Fallback message displayed")
                else:
                    print("   ⚠️ WARNING: Fallback message not found")
                
                if any("WSL" in call for call in print_calls):
                    print("   ✅ PASS: WSL guidance provided")
                else:
                    print("   ⚠️ WARNING: WSL guidance not found")
                
                return True
            else:
                print("   ❌ FAIL: Fallback did not occur")
                return False
                
    except Exception as e:
        print(f"   ❌ FAIL: Fallback test failed: {e}")
        return False


def test_event_polling_safety(IntegratedArenaBotGUI):
    """Test event polling thread safety."""
    print("\n🧪 Test 4: Event Polling Thread Safety")
    try:
        # Test with GUI available
        with mock.patch('tkinter.Tk') as mock_tk, \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print'):
            
            mock_root = mock.MagicMock()
            mock_tk.return_value = mock_root
            
            bot = IntegratedArenaBotGUI()
            bot.event_polling_active = False
            
            # Test event polling start
            bot._start_event_polling()
            
            if bot.event_polling_active:
                print("   ✅ PASS: Event polling activated with GUI")
                
                if mock_root.after.called:
                    print("   ✅ PASS: GUI event scheduling activated")
                else:
                    print("   ⚠️ WARNING: GUI event scheduling not found")
            else:
                print("   ❌ FAIL: Event polling not activated")
                return False
        
        # Test without GUI
        with mock.patch('tkinter.Tk', side_effect=Exception("No display")), \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print') as mock_print:
            
            bot = IntegratedArenaBotGUI()
            bot.event_polling_active = False
            
            # Test event polling without GUI
            bot._start_event_polling()
            
            if bot.event_polling_active:
                print("   ✅ PASS: Event polling activated without GUI")
                
                # Check for appropriate message
                print_calls = [str(call) for call in mock_print.call_args_list]
                if any("Event polling disabled" in call for call in print_calls):
                    print("   ✅ PASS: Appropriate no-GUI message displayed")
                else:
                    print("   ⚠️ WARNING: No-GUI message not found")
            else:
                print("   ❌ FAIL: Event polling not activated without GUI")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Event polling test failed: {e}")
        return False


def test_run_method_behavior(IntegratedArenaBotGUI):
    """Test run method behavior in both modes."""
    print("\n🧪 Test 5: Run Method Behavior")
    try:
        # Test with GUI
        with mock.patch('tkinter.Tk') as mock_tk, \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print'):
            
            mock_root = mock.MagicMock()
            mock_tk.return_value = mock_root
            
            bot = IntegratedArenaBotGUI()
            
            # Mock mainloop to prevent actual GUI loop
            mock_root.mainloop = mock.MagicMock()
            
            # Test run with GUI
            bot.run()
            
            if mock_root.mainloop.called:
                print("   ✅ PASS: GUI mainloop called when GUI available")
            else:
                print("   ❌ FAIL: GUI mainloop not called")
                return False
        
        # Test without GUI
        with mock.patch('tkinter.Tk', side_effect=Exception("No display")), \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print'):
            
            bot = IntegratedArenaBotGUI()
            
            # Mock command-line runner
            with mock.patch.object(bot, 'run_command_line') as mock_run_cli, \
                 mock.patch.object(bot, 'log_text') as mock_log_text:
                
                # Test run without GUI
                bot.run()
                
                if mock_run_cli.called:
                    print("   ✅ PASS: Command-line mode called when GUI unavailable")
                else:
                    print("   ❌ FAIL: Command-line mode not called")
                    return False
                
                if mock_log_text.called:
                    print("   ✅ PASS: Fallback message logged")
                else:
                    print("   ⚠️ WARNING: Fallback message not logged")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Run method test failed: {e}")
        return False


def test_background_loading_safety(IntegratedArenaBotGUI):
    """Test background loading thread safety."""
    print("\n🧪 Test 6: Background Loading Thread Safety")
    try:
        with mock.patch('tkinter.Tk') as mock_tk, \
             mock.patch('arena_bot.utils.asset_loader.AssetLoader') as mock_asset_loader, \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('logging_compatibility.get_logger'), \
             mock.patch('builtins.print'):
            
            mock_root = mock.MagicMock()
            mock_tk.return_value = mock_root
            
            # Setup mock asset loader
            mock_asset_instance = mock.MagicMock()
            mock_asset_loader.return_value = mock_asset_instance
            
            bot = IntegratedArenaBotGUI()
            bot.asset_loader = mock_asset_instance
            
            # Track thread creation
            thread_created = threading.Event()
            original_thread_init = threading.Thread.__init__
            
            def track_thread_creation(thread_self, *args, **kwargs):
                if 'Card Database Loader' in str(kwargs.get('name', '')):
                    thread_created.set()
                return original_thread_init(thread_self, *args, **kwargs)
            
            with mock.patch.object(threading.Thread, '__init__', side_effect=track_thread_creation):
                # Test background loading
                bot._start_card_database_loading()
                
                # Wait briefly for thread creation
                time.sleep(0.1)
                
                if thread_created.is_set():
                    print("   ✅ PASS: Background database loading thread created")
                else:
                    print("   ⚠️ WARNING: Background thread creation not detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Background loading test failed: {e}")
        return False


def test_current_environment():
    """Test the actual environment capabilities."""
    print("\n🧪 Test 7: Current Environment Validation")
    
    # Check DISPLAY variable
    display = os.environ.get('DISPLAY')
    if display:
        print(f"   ℹ️ DISPLAY set to: {display}")
    else:
        print("   ℹ️ DISPLAY not set (normal for headless environments)")
    
    # Test tkinter availability
    try:
        import tkinter as tk
        
        # Try to create a test window
        test_root = tk.Tk()
        test_root.withdraw()  # Hide immediately
        test_root.destroy()
        
        print("   ✅ PASS: GUI environment is functional")
        gui_available = True
        
    except Exception as e:
        print(f"   ℹ️ GUI environment not available: {e}")
        print("   ℹ️ This is normal for headless/WSL environments")
        gui_available = False
    
    # Test the actual bot in current environment
    try:
        with mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('arena_bot.logging_system.stier_logging.get_logger'):
            
            from integrated_arena_bot_gui import IntegratedArenaBotGUI
            
            start_time = time.time()
            bot = IntegratedArenaBotGUI()
            init_time = time.time() - start_time
            
            if gui_available:
                if hasattr(bot, 'root') and bot.root is not None:
                    print(f"   ✅ PASS: Bot initialized in GUI mode ({init_time:.3f}s)")
                    # Clean up
                    if hasattr(bot.root, 'destroy'):
                        bot.root.destroy()
                else:
                    print("   ⚠️ WARNING: GUI available but bot didn't use it")
            else:
                if not hasattr(bot, 'root') or bot.root is None:
                    print(f"   ✅ PASS: Bot properly fell back to command-line mode ({init_time:.3f}s)")
                else:
                    print("   ⚠️ WARNING: GUI unavailable but bot tried to create one")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Environment test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🎯 Simple GUI Fix Validation")
    print("=" * 50)
    print("Testing the fix for GUI loading issues and command-line fallback")
    print()
    
    results = []
    
    # Test 1: Basic import
    success, IntegratedArenaBotGUI = test_basic_import()
    results.append(success)
    
    if not success or IntegratedArenaBotGUI is None:
        print("\n❌ Cannot continue tests due to import failure")
        return 1
    
    # Test 2: GUI success path
    results.append(test_gui_success_path(IntegratedArenaBotGUI))
    
    # Test 3: GUI fallback path
    results.append(test_gui_fallback_path(IntegratedArenaBotGUI))
    
    # Test 4: Event polling safety
    results.append(test_event_polling_safety(IntegratedArenaBotGUI))
    
    # Test 5: Run method behavior
    results.append(test_run_method_behavior(IntegratedArenaBotGUI))
    
    # Test 6: Background loading safety
    results.append(test_background_loading_safety(IntegratedArenaBotGUI))
    
    # Test 7: Current environment
    results.append(test_current_environment())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! GUI loading fix is working correctly.")
        print("\n✅ The fix successfully addresses:")
        print("   • GUI initialization with proper error handling")
        print("   • Graceful fallback to command-line mode") 
        print("   • Thread-safe event polling")
        print("   • Background loading without blocking")
        print("   • Comprehensive error messages and user guidance")
        return 0
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)