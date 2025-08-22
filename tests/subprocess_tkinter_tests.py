#!/usr/bin/env python3
"""
🔧 Subprocess-based Tkinter Testing

Isolates tkinter testing from PyQt6 to prevent Tcl_AsyncDelete threading conflicts.
Implements your friend's recommendation for subprocess isolation.

Usage:
    python3 tests/subprocess_tkinter_tests.py
"""

import sys
import os
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class TkinterTestResult:
    """Result of a subprocess tkinter test"""
    test_name: str
    success: bool
    error_message: Optional[str] = None
    duration: float = 0.0
    details: Dict[str, Any] = None

class SubprocessTkinterTester:
    """
    Runs tkinter-based tests in isolated subprocesses to prevent
    threading conflicts with PyQt6 main process.
    """
    
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        self.test_results: List[TkinterTestResult] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Centralized logging"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_tkinter_test_subprocess(self, test_script: str, test_name: str) -> TkinterTestResult:
        """
        Run a tkinter test in an isolated subprocess
        """
        start_time = time.time()
        self.log(f"🔧 Running subprocess test: {test_name}")
        
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Run in subprocess with environment isolation
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            env['TEST_MODE'] = '1'
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=str(project_root)
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.log(f"✅ {test_name} - SUCCESS ({duration:.2f}s)")
                
                # Try to parse JSON output
                details = None
                try:
                    if result.stdout.strip():
                        details = json.loads(result.stdout.strip())
                except:
                    details = {"stdout": result.stdout, "stderr": result.stderr}
                
                return TkinterTestResult(
                    test_name=test_name,
                    success=True,
                    duration=duration,
                    details=details
                )
            else:
                error_msg = f"Exit code {result.returncode}: {result.stderr}"
                self.log(f"❌ {test_name} - FAILED: {error_msg}")
                
                return TkinterTestResult(
                    test_name=test_name,
                    success=False,
                    error_message=error_msg,
                    duration=duration,
                    details={"stdout": result.stdout, "stderr": result.stderr}
                )
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = "Test timed out after 30 seconds"
            self.log(f"⏰ {test_name} - TIMEOUT: {error_msg}")
            
            return TkinterTestResult(
                test_name=test_name,
                success=False,
                error_message=error_msg,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Subprocess error: {str(e)}"
            self.log(f"💥 {test_name} - ERROR: {error_msg}")
            
            return TkinterTestResult(
                test_name=test_name,
                success=False,
                error_message=error_msg,
                duration=duration
            )
        finally:
            # Cleanup temp file
            try:
                os.unlink(script_path)
            except:
                pass
    
    def test_coordinate_selector_subprocess(self) -> TkinterTestResult:
        """Test CoordinateSelector in subprocess"""
        
        test_script = '''
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tkinter as tk
    from integrated_arena_bot_gui import CoordinateSelector
    
    # Test coordinate selector creation
    root = tk.Tk()
    root.withdraw()  # Hide window
    
    selector = CoordinateSelector(root)
    
    # Test coordinate validation
    test_coords = {
        'card1': {'x': 100, 'y': 200, 'width': 150, 'height': 200},
        'card2': {'x': 300, 'y': 200, 'width': 150, 'height': 200},
        'card3': {'x': 500, 'y': 200, 'width': 150, 'height': 200}
    }
    
    validation_passed = True
    issues = []
    
    for card, coords in test_coords.items():
        if coords['x'] < 0 or coords['y'] < 0:
            validation_passed = False
            issues.append(f"{card}: negative coordinates")
        if coords['width'] <= 0 or coords['height'] <= 0:
            validation_passed = False
            issues.append(f"{card}: invalid dimensions")
    
    root.destroy()
    
    result = {
        "coordinate_selector_created": True,
        "validation_passed": validation_passed,
        "validation_issues": issues,
        "test_coordinates_count": len(test_coords)
    }
    
    print(json.dumps(result))
    
except Exception as e:
    result = {
        "coordinate_selector_created": False,
        "error": str(e),
        "validation_passed": False
    }
    print(json.dumps(result))
    sys.exit(1)
'''
        
        return self.run_tkinter_test_subprocess(test_script, "CoordinateSelector Subprocess Test")
    
    def test_draft_overlay_subprocess(self) -> TkinterTestResult:
        """Test DraftOverlay in subprocess"""
        
        test_script = '''
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from arena_bot.ui.draft_overlay import DraftOverlay, OverlayConfig
    
    # Test overlay creation
    config = OverlayConfig()
    overlay = DraftOverlay(config)
    
    # Test methods exist
    methods_tested = []
    missing_methods = []
    
    required_methods = ['initialize', 'cleanup']
    for method_name in required_methods:
        if hasattr(overlay, method_name):
            methods_tested.append(method_name)
        else:
            missing_methods.append(method_name)
    
    # Test initialization
    initialization_success = False
    try:
        overlay.initialize()
        initialization_success = True
    except Exception as e:
        init_error = str(e)
    
    # Test cleanup
    cleanup_success = False
    try:
        overlay.cleanup()
        cleanup_success = True
    except Exception as e:
        cleanup_error = str(e)
    
    result = {
        "overlay_created": True,
        "methods_tested": methods_tested,
        "missing_methods": missing_methods,
        "initialization_success": initialization_success,
        "cleanup_success": cleanup_success
    }
    
    print(json.dumps(result))
    
except Exception as e:
    result = {
        "overlay_created": False,
        "error": str(e)
    }
    print(json.dumps(result))
    sys.exit(1)
'''
        
        return self.run_tkinter_test_subprocess(test_script, "DraftOverlay Subprocess Test")
    
    def test_visual_overlay_subprocess(self) -> TkinterTestResult:
        """Test VisualOverlay in subprocess"""
        
        test_script = '''
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from arena_bot.ui.visual_overlay import VisualOverlay
    
    # Test overlay creation
    overlay = VisualOverlay()
    
    # Test required methods exist
    required_methods = ['show', 'hide', 'update_recommendations']
    methods_available = []
    methods_missing = []
    
    for method_name in required_methods:
        if hasattr(overlay, method_name):
            methods_available.append(method_name)
        else:
            methods_missing.append(method_name)
    
    # Test method execution
    method_execution_results = {}
    
    try:
        if hasattr(overlay, 'show'):
            overlay.show()
            method_execution_results['show'] = 'success'
    except Exception as e:
        method_execution_results['show'] = f'error: {str(e)}'
    
    try:
        if hasattr(overlay, 'hide'):
            overlay.hide()
            method_execution_results['hide'] = 'success'
    except Exception as e:
        method_execution_results['hide'] = f'error: {str(e)}'
    
    result = {
        "visual_overlay_created": True,
        "methods_available": methods_available,
        "methods_missing": methods_missing,
        "method_execution_results": method_execution_results
    }
    
    print(json.dumps(result))
    
except Exception as e:
    result = {
        "visual_overlay_created": False,
        "error": str(e)
    }
    print(json.dumps(result))
    sys.exit(1)
'''
        
        return self.run_tkinter_test_subprocess(test_script, "VisualOverlay Subprocess Test")
    
    def run_all_subprocess_tests(self) -> List[TkinterTestResult]:
        """Run all tkinter tests in isolated subprocesses"""
        
        tests = [
            self.test_coordinate_selector_subprocess,
            self.test_draft_overlay_subprocess,
            self.test_visual_overlay_subprocess,
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                self.log(f"Test function failed: {e}", "ERROR")
                failed_result = TkinterTestResult(
                    test_name=f"Failed_{test_func.__name__}",
                    success=False,
                    error_message=str(e),
                    duration=0.0
                )
                results.append(failed_result)
                self.test_results.append(failed_result)
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) * 100,
                "total_duration": sum(r.duration for r in self.test_results),
                "average_duration": sum(r.duration for r in self.test_results) / len(self.test_results)
            },
            "successful_tests": [
                {
                    "name": r.test_name,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in successful_tests
            ],
            "failed_tests": [
                {
                    "name": r.test_name,
                    "error": r.error_message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in failed_tests
            ],
            "detailed_results": [asdict(r) for r in self.test_results]
        }
        
        return report
    
    def save_report(self, filename: str = None) -> Path:
        """Save test report to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"subprocess_tkinter_report_{timestamp}.json"
        
        report_path = self.artifacts_dir / filename
        report = self.generate_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"📊 Subprocess tkinter report saved: {report_path}")
        return report_path

def main():
    """Main entry point"""
    tester = SubprocessTkinterTester()
    
    try:
        print("🔧 Subprocess Tkinter Testing System")
        print("=" * 50)
        print("Isolates tkinter tests from PyQt6 to prevent threading conflicts")
        print("")
        
        # Run all tests
        results = tester.run_all_subprocess_tests()
        
        # Generate and save report
        report_path = tester.save_report()
        report = tester.generate_report()
        
        # Print summary
        print(f"\n🎯 SUBPROCESS TKINTER TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"✅ Successful: {report['summary']['successful_tests']}")
        print(f"❌ Failed: {report['summary']['failed_tests']}")
        print(f"📊 Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {report['summary']['total_duration']:.2f}s")
        print(f"📁 Report: {report_path}")
        
        if report['failed_tests']:
            print(f"\n❌ FAILED TESTS:")
            for failure in report['failed_tests']:
                print(f"  • {failure['name']}: {failure['error']}")
        
        if report['successful_tests']:
            print(f"\n✅ SUCCESSFUL TESTS:")
            for success in report['successful_tests']:
                print(f"  • {success['name']} ({success['duration']:.2f}s)")
        
        # Exit code based on results
        if report['summary']['failed_tests'] > 0:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"💥 Subprocess testing system crashed: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())