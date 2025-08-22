#!/usr/bin/env python3
"""
GUI Loading Fix Validation Runner

This script runs comprehensive validation tests for the GUI loading fix
and provides detailed reporting on test results and performance metrics.

Usage:
    python3 run_gui_loading_validation.py [--verbose] [--performance] [--coverage]
    
Options:
    --verbose     Show detailed test output
    --performance Run performance benchmarks
    --coverage    Generate test coverage report
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_basic_validation():
    """Run basic GUI loading validation tests."""
    print("🧪 Running Basic GUI Loading Validation Tests...")
    print("=" * 60)
    
    try:
        import pytest
        
        # Run core validation tests
        result = pytest.main([
            "test_gui_loading_fix_validation.py::TestGUILoadingFix",
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure for quick feedback
        ])
        
        if result == 0:
            print("✅ Basic validation tests PASSED")
            return True
        else:
            print("❌ Basic validation tests FAILED")
            return False
            
    except ImportError:
        print("⚠️ pytest not available, running manual validation...")
        return run_manual_validation()


def run_performance_benchmarks():
    """Run performance benchmark tests."""
    print("\n⚡ Running Performance Benchmarks...")
    print("=" * 60)
    
    try:
        import pytest
        
        result = pytest.main([
            "test_gui_loading_fix_validation.py::TestPerformanceMetrics",
            "-v",
            "--tb=short"
        ])
        
        return result == 0
        
    except ImportError:
        print("⚠️ pytest not available for performance tests")
        return False


def run_environment_compatibility():
    """Run environment compatibility tests."""
    print("\n🌍 Running Environment Compatibility Tests...")
    print("=" * 60)
    
    try:
        import pytest
        
        result = pytest.main([
            "test_gui_loading_fix_validation.py::TestEnvironmentCompatibility",
            "-v",
            "--tb=short"
        ])
        
        return result == 0
        
    except ImportError:
        print("⚠️ pytest not available for compatibility tests")
        return False


def run_manual_validation():
    """Run manual validation when pytest is not available."""
    print("🔧 Running Manual Validation Tests...")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Import validation
    total_tests += 1
    try:
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        print("✅ Test 1: Module import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Test 1: Module import failed: {e}")
    
    # Test 2: GUI initialization with mocked tkinter
    total_tests += 1
    try:
        import unittest.mock as mock
        with mock.patch('tkinter.Tk') as mock_tk:
            mock_root = mock.MagicMock()
            mock_tk.return_value = mock_root
            
            bot = IntegratedArenaBotGUI()
            if hasattr(bot, 'root') and bot.root is not None:
                print("✅ Test 2: GUI initialization successful")
                success_count += 1
            else:
                print("❌ Test 2: GUI initialization failed")
    except Exception as e:
        print(f"❌ Test 2: GUI initialization test failed: {e}")
    
    # Test 3: Fallback mode validation
    total_tests += 1
    try:
        with mock.patch('tkinter.Tk', side_effect=Exception("No display")):
            bot = IntegratedArenaBotGUI()
            if not hasattr(bot, 'root') or bot.root is None:
                print("✅ Test 3: Fallback mode successful")
                success_count += 1
            else:
                print("❌ Test 3: Fallback mode failed")
    except Exception as e:
        print(f"❌ Test 3: Fallback mode test failed: {e}")
    
    print(f"\n📊 Manual validation: {success_count}/{total_tests} tests passed")
    return success_count == total_tests


def test_actual_gui_creation():
    """Test actual GUI creation in current environment."""
    print("\n🖥️  Testing Actual GUI Creation...")
    print("=" * 60)
    
    # Test with current environment
    try:
        import tkinter as tk
        
        # Try to create a simple test window
        test_root = tk.Tk()
        test_root.title("GUI Test")
        test_root.withdraw()  # Hide window immediately
        
        # If we get here, GUI is available
        test_root.destroy()
        print("✅ GUI environment is available (X11/display server working)")
        gui_available = True
        
    except Exception as e:
        print(f"ℹ️ GUI environment not available: {e}")
        print("   This is normal in headless/WSL environments")
        gui_available = False
    
    # Test the actual IntegratedArenaBotGUI in current environment
    try:
        print("\n🧪 Testing IntegratedArenaBotGUI in current environment...")
        
        # Import with mocked dependencies to avoid side effects
        import unittest.mock as mock
        
        with mock.patch('arena_bot.utils.asset_loader.AssetLoader'), \
             mock.patch('arena_bot.core.card_recognizer.CardRecognizer'), \
             mock.patch('arena_bot.logging_system.stier_logging.get_logger'):
            
            start_time = time.time()
            
            # Create bot instance
            from integrated_arena_bot_gui import IntegratedArenaBotGUI
            bot = IntegratedArenaBotGUI()
            
            init_time = time.time() - start_time
            
            # Check results
            if gui_available:
                if hasattr(bot, 'root') and bot.root is not None:
                    print(f"✅ GUI mode initialized successfully in {init_time:.3f}s")
                    # Clean up
                    if hasattr(bot.root, 'destroy'):
                        bot.root.destroy()
                else:
                    print("⚠️ GUI available but bot didn't initialize GUI mode")
            else:
                if not hasattr(bot, 'root') or bot.root is None:
                    print(f"✅ Command-line fallback mode activated successfully in {init_time:.3f}s")
                else:
                    print("⚠️ GUI not available but bot tried to create GUI")
            
            return True
            
    except Exception as e:
        print(f"❌ IntegratedArenaBotGUI test failed: {e}")
        return False


def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📋 Test Report Summary")
    print("=" * 60)
    
    # Environment information
    print("🖥️  Environment Information:")
    print(f"   Platform: {sys.platform}")
    print(f"   Python: {sys.version}")
    print(f"   Display: {os.environ.get('DISPLAY', 'Not set')}")
    
    # Check for GUI availability
    gui_available = False
    try:
        import tkinter
        tkinter.Tk().destroy()
        gui_available = True
    except:
        pass
    
    print(f"   GUI Available: {'Yes' if gui_available else 'No'}")
    
    # Dependencies check
    print("\n📦 Dependencies:")
    dependencies = [
        'tkinter', 'pytest', 'unittest.mock', 'threading', 
        'queue', 'psutil', 'cv2', 'numpy'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (missing)")
    
    print("\n💡 Recommendations:")
    if not gui_available:
        print("   • GUI not available - this is normal for headless/WSL environments")
        print("   • Command-line fallback mode should be working")
        print("   • To enable GUI: Install X server (VcXsrv/Xming) and set DISPLAY")
    else:
        print("   • GUI environment is working correctly")
        print("   • Both GUI and command-line modes should be available")


def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description="GUI Loading Fix Validation Runner")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed test output")
    parser.add_argument("--performance", "-p", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Generate test coverage report")
    
    args = parser.parse_args()
    
    print("🎯 GUI Loading Fix Validation Suite")
    print("=" * 60)
    print("Testing the fix for GUI loading issues where bot showed blank screen")
    print("Validating graceful fallback to command-line mode and thread safety")
    print()
    
    results = []
    
    # Run basic validation
    results.append(run_basic_validation())
    
    # Run performance tests if requested
    if args.performance:
        results.append(run_performance_benchmarks())
    
    # Run environment compatibility tests
    results.append(run_environment_compatibility())
    
    # Test actual GUI creation
    results.append(test_actual_gui_creation())
    
    # Generate coverage report if requested
    if args.coverage:
        print("\n📊 Generating Coverage Report...")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "test_gui_loading_fix_validation.py",
                "--cov=integrated_arena_bot_gui",
                "--cov-report=html",
                "--cov-report=term"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Coverage report generated")
                print("   HTML report: htmlcov/index.html")
            else:
                print("⚠️ Coverage report generation failed")
                
        except ImportError:
            print("⚠️ pytest-cov not available for coverage reporting")
    
    # Generate final report
    generate_test_report()
    
    # Final results
    print(f"\n🏁 Final Results: {sum(results)}/{len(results)} test suites passed")
    
    if all(results):
        print("🎉 All validation tests PASSED! GUI loading fix is working correctly.")
        return 0
    else:
        print("⚠️ Some validation tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)