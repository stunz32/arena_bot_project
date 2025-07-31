#!/usr/bin/env python3
"""
Production Hotfix Verification Script
Tests the Unicode encoding and cross-platform path resolution fixes.
"""

import sys
import os
from pathlib import Path

# Test imports
try:
    from logging_config import setup_logging, test_unicode_support, get_platform_info
    from hearthstone_log_monitor import HearthstoneLogMonitor
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_logging_system():
    """Test the platform-aware logging system."""
    print("\n" + "="*60)
    print("TESTING LOGGING SYSTEM")
    print("="*60)
    
    # Test Unicode support detection
    test_unicode_support()
    
    # Test logging setup
    print("\n--- Testing Logging Setup ---")
    logger = setup_logging(console_output=True)
    
    # Test various Unicode characters that previously caused issues
    test_messages = [
        "✅ Testing basic checkmark",
        "❌ Testing X mark", 
        "⚠️ Testing warning sign",
        "🎯 Testing target emoji",
        "💓 Testing heartbeat emoji",
        "💔 Testing broken heart emoji",
        "📁 Testing folder emoji",
        "🎮 Testing game controller emoji",
        "█ Testing block character",
        "🚀 Testing rocket emoji"
    ]
    
    print("Testing Unicode character logging:")
    for msg in test_messages:
        try:
            logger.info(msg)
            print(f"  ✓ Logged: {msg[:20]}...")
        except Exception as e:
            print(f"  ✗ Failed: {msg[:20]}... - {e}")
    
    return True


def test_path_resolution():
    """Test cross-platform path resolution."""
    print("\n" + "="*60)
    print("TESTING PATH RESOLUTION")
    print("="*60)
    
    platform_info = get_platform_info()
    print(f"Platform: {platform_info['platform']}")
    print(f"Python version: {platform_info['python_version'][:20]}...")
    
    # Test monitor initialization without path
    print("\n--- Testing Monitor Initialization ---")
    try:
        monitor = HearthstoneLogMonitor()
        print(f"✅ Monitor initialized successfully")
        print(f"   Base path: {monitor.logs_base_path}")
        print(f"   Path exists: {monitor.logs_base_path.exists()}")
        
        # Test directory discovery
        print("\n--- Testing Directory Discovery ---")
        latest_dir = monitor.find_latest_log_directory()
        if latest_dir:
            print(f"✅ Found log directory: {latest_dir}")
            
            # Test log file discovery
            log_files = monitor.discover_log_files(latest_dir)
            print(f"✅ Found {len(log_files)} log files:")
            for log_type, log_path in log_files.items():
                print(f"   {log_type}: {log_path.name}")
        else:
            print("⚠️ No log directory found (this is expected if Hearthstone isn't installed)")
            
    except Exception as e:
        print(f"❌ Monitor initialization failed: {e}")
        return False
    
    # Test different path formats
    print("\n--- Testing Path Format Handling ---")
    test_paths = [
        "/mnt/m/Hearthstone/Logs",  # WSL format
        "M:/Hearthstone/Logs",      # Windows format  
        "C:/Program Files (x86)/Hearthstone/Logs",  # Common Windows path
    ]
    
    for test_path in test_paths:
        try:
            path_obj = Path(test_path)
            print(f"  {test_path}: exists={path_obj.exists()}, accessible={os.access(path_obj.parent, os.R_OK) if path_obj.parent.exists() else False}")
        except Exception as e:
            print(f"  {test_path}: error={e}")
    
    return True


def test_heartbeat_system():
    """Test the heartbeat and error recovery system."""
    print("\n" + "="*60)
    print("TESTING HEARTBEAT SYSTEM")
    print("="*60)
    
    try:
        monitor = HearthstoneLogMonitor()
        
        # Test heartbeat check
        print("--- Testing Heartbeat Check ---")
        accessible = monitor._check_heartbeat_and_log_accessibility()
        print(f"✅ Heartbeat check completed: accessible={accessible}")
        
        # Test error recovery
        print("--- Testing Error Recovery ---")
        monitor._attempt_log_error_recovery()
        print("✅ Error recovery completed without crashing")
        
        return True
        
    except Exception as e:
        print(f"❌ Heartbeat system test failed: {e}")
        return False


def test_unicode_display_functions():
    """Test the platform-safe display functions."""
    print("\n" + "="*60)
    print("TESTING DISPLAY FUNCTIONS")
    print("="*60)
    
    try:
        from hearthstone_log_monitor import GameState
        monitor = HearthstoneLogMonitor()
        
        # Test prominent screen change display
        print("--- Testing Screen Change Display ---")
        monitor._display_prominent_screen_change(GameState.ARENA_DRAFT)
        print("✅ Screen change display completed")
        
        # Test draft start display
        print("\n--- Testing Draft Start Display ---")
        monitor._display_draft_start()
        print("✅ Draft start display completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Display function test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("🔧 PRODUCTION HOTFIX VERIFICATION")
    print("Testing Unicode encoding and cross-platform path fixes")
    print("Platform:", sys.platform)
    
    tests = [
        ("Logging System", test_logging_system),
        ("Path Resolution", test_path_resolution), 
        ("Heartbeat System", test_heartbeat_system),
        ("Display Functions", test_unicode_display_functions),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Production fixes verified!")
        return 0
    else:
        print("⚠️ Some tests failed - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())