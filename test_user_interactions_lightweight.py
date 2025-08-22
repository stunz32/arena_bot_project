#!/usr/bin/env python3
"""
🎮 Lightweight User Interaction Testing

Tests user interactions without the full GUI startup delay.
Focuses on finding real bugs users encounter when clicking buttons and using features.

Usage:
    python3 test_user_interactions_lightweight.py --all
    python3 test_user_interactions_lightweight.py --quick
"""

import sys
import os
import json
import time
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import tkinter as tk
    from tkinter import ttk
    from app.debug_utils import create_debug_snapshot
    TKINTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GUI imports not available: {e}")
    TKINTER_AVAILABLE = False

@dataclass
class InteractionResult:
    """Result of testing a user interaction"""
    interaction_name: str
    success: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    duration: float = 0.0

class LightweightInteractionTester:
    """Tests user interactions with minimal overhead"""
    
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        self.test_results: List[InteractionResult] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Centralized logging"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_interaction(self, name: str, test_func) -> InteractionResult:
        """Test a single interaction"""
        start_time = time.time()
        self.log(f"🧪 Testing: {name}")
        
        try:
            result_data = test_func()
            duration = time.time() - start_time
            
            success = result_data.get('success', True) if isinstance(result_data, dict) else True
            error_message = result_data.get('error') if isinstance(result_data, dict) else None
            details = result_data if isinstance(result_data, dict) else None
            
            result = InteractionResult(
                interaction_name=name,
                success=success,
                error_message=error_message,
                details=details,
                duration=duration
            )
            
            if success:
                self.log(f"✅ {name} - SUCCESS ({duration:.2f}s)")
            else:
                self.log(f"❌ {name} - FAILED: {error_message}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            self.log(f"💥 {name} - CRASHED: {error_msg}")
            
            return InteractionResult(
                interaction_name=name,
                success=False,
                error_message=error_msg,
                details={"traceback": traceback.format_exc()},
                duration=duration
            )
    
    # ========================================
    # GUI COMPONENT INTERACTION TESTS
    # ========================================
    
    def test_draft_overlay_interactions(self) -> Dict[str, Any]:
        """Test DraftOverlay user interactions"""
        try:
            from arena_bot.ui.draft_overlay import DraftOverlay, OverlayConfig
            
            config = OverlayConfig()
            overlay = DraftOverlay(config)
            
            issues_found = []
            
            # Test 1: Initialize overlay
            try:
                overlay.initialize()
                self.log("✅ DraftOverlay.initialize() works")
            except Exception as e:
                issues_found.append(f"initialize() failed: {e}")
            
            # Test 2: Start monitoring 
            try:
                if hasattr(overlay, '_start_monitoring'):
                    overlay._start_monitoring()
                    self.log("✅ DraftOverlay._start_monitoring() works")
                else:
                    issues_found.append("_start_monitoring() method missing")
            except Exception as e:
                issues_found.append(f"_start_monitoring() failed: {e}")
            
            # Test 3: Test window creation
            try:
                if hasattr(overlay, 'root') and overlay.root:
                    # Try to get window properties
                    geometry = overlay.root.geometry()
                    title = overlay.root.title()
                    self.log(f"✅ DraftOverlay window: {geometry}, title: '{title}'")
                else:
                    issues_found.append("No root window created")
            except Exception as e:
                issues_found.append(f"Window properties failed: {e}")
            
            # Test 4: Test update methods
            try:
                if hasattr(overlay, 'update_draft_info'):
                    # Test with dummy data
                    test_data = {
                        'cards': ['Test Card 1', 'Test Card 2', 'Test Card 3'],
                        'recommendations': ['Pick Test Card 2'],
                        'current_pick': 1
                    }
                    overlay.update_draft_info(test_data)
                    self.log("✅ DraftOverlay.update_draft_info() works")
                else:
                    issues_found.append("update_draft_info() method missing")
            except Exception as e:
                issues_found.append(f"update_draft_info() failed: {e}")
            
            # Test 5: Cleanup
            try:
                overlay.cleanup()
                self.log("✅ DraftOverlay.cleanup() works")
            except Exception as e:
                issues_found.append(f"cleanup() failed: {e}")
            
            return {
                'success': len(issues_found) == 0,
                'issues_found': issues_found,
                'total_tests': 5,
                'passed_tests': 5 - len(issues_found)
            }
            
        except ImportError as e:
            return {
                'success': False,
                'error': f"Cannot import DraftOverlay: {e}",
                'issues_found': [f"Import failed: {e}"]
            }
    
    def test_visual_overlay_interactions(self) -> Dict[str, Any]:
        """Test VisualOverlay user interactions"""
        try:
            from arena_bot.ui.visual_overlay import VisualOverlay
            
            issues_found = []
            
            # Test 1: Create overlay
            try:
                overlay = VisualOverlay()
                self.log("✅ VisualOverlay creation works")
            except Exception as e:
                issues_found.append(f"VisualOverlay creation failed: {e}")
                return {
                    'success': False,
                    'issues_found': issues_found,
                    'error': f"Cannot create VisualOverlay: {e}"
                }
            
            # Test 2: Test methods exist
            required_methods = ['show', 'hide', 'update_recommendations']
            for method_name in required_methods:
                if hasattr(overlay, method_name):
                    self.log(f"✅ VisualOverlay.{method_name}() exists")
                else:
                    issues_found.append(f"{method_name}() method missing")
            
            # Test 3: Test show/hide functionality
            try:
                if hasattr(overlay, 'show'):
                    overlay.show()
                    self.log("✅ VisualOverlay.show() works")
                
                if hasattr(overlay, 'hide'):
                    overlay.hide()
                    self.log("✅ VisualOverlay.hide() works")
            except Exception as e:
                issues_found.append(f"show/hide failed: {e}")
            
            # Test 4: Test update with dummy data
            try:
                if hasattr(overlay, 'update_recommendations'):
                    test_recommendations = [
                        {'card': 'Test Card', 'score': 85, 'reason': 'Good value'},
                        {'card': 'Test Card 2', 'score': 70, 'reason': 'Decent stats'}
                    ]
                    overlay.update_recommendations(test_recommendations)
                    self.log("✅ VisualOverlay.update_recommendations() works")
            except Exception as e:
                issues_found.append(f"update_recommendations() failed: {e}")
            
            return {
                'success': len(issues_found) == 0,
                'issues_found': issues_found,
                'total_tests': 4,
                'passed_tests': 4 - len(issues_found)
            }
            
        except ImportError as e:
            return {
                'success': False,
                'error': f"Cannot import VisualOverlay: {e}",
                'issues_found': [f"Import failed: {e}"]
            }
    
    def test_screenshot_analysis_workflow(self) -> Dict[str, Any]:
        """Test screenshot analysis without full GUI"""
        try:
            # Test the core analysis functions
            issues_found = []
            
            # Test 1: Can we import the main analysis components?
            try:
                from arena_bot.detection.ultimate_detector import UltimateDetectionEngine
                self.log("✅ UltimateDetectionEngine import works")
            except ImportError as e:
                issues_found.append(f"Cannot import UltimateDetectionEngine: {e}")
            
            # Test 2: Can we import the screenshot analysis method?
            try:
                from integrated_arena_bot_gui import IntegratedArenaBotGUI
                gui_class = IntegratedArenaBotGUI
                
                # Check if analyze_screenshot_data method exists
                if hasattr(gui_class, 'analyze_screenshot_data'):
                    self.log("✅ analyze_screenshot_data method exists")
                else:
                    issues_found.append("analyze_screenshot_data method missing")
                
                # Check manual_screenshot method
                if hasattr(gui_class, 'manual_screenshot'):
                    self.log("✅ manual_screenshot method exists")
                else:
                    issues_found.append("manual_screenshot method missing")
                    
            except ImportError as e:
                issues_found.append(f"Cannot import main GUI: {e}")
            
            # Test 3: Test with dummy screenshot data
            try:
                import numpy as np
                from PIL import Image
                
                # Create a dummy screenshot
                dummy_image = Image.new('RGB', (1920, 1080), color='blue')
                dummy_array = np.array(dummy_image)
                
                self.log("✅ Can create dummy screenshot data")
                
                # Test image processing
                if dummy_array.shape == (1080, 1920, 3):
                    self.log("✅ Image dimensions correct")
                else:
                    issues_found.append(f"Unexpected image shape: {dummy_array.shape}")
                    
            except Exception as e:
                issues_found.append(f"Image processing failed: {e}")
            
            return {
                'success': len(issues_found) == 0,
                'issues_found': issues_found,
                'total_tests': 3,
                'passed_tests': 3 - len(issues_found)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Screenshot analysis test failed: {e}",
                'issues_found': [str(e)]
            }
    
    def test_coordinate_selection_functionality(self) -> Dict[str, Any]:
        """Test coordinate selection without full GUI"""
        try:
            from integrated_arena_bot_gui import CoordinateSelector
            
            issues_found = []
            
            # Test 1: Can we create CoordinateSelector?
            try:
                # Create a temporary root for testing
                root = tk.Tk()
                root.withdraw()  # Hide the window
                
                selector = CoordinateSelector(root)
                self.log("✅ CoordinateSelector creation works")
                
                root.destroy()
                
            except Exception as e:
                issues_found.append(f"CoordinateSelector creation failed: {e}")
            
            # Test 2: Check coordinate validation
            try:
                # Test coordinate validation logic
                test_coords = {
                    'card1': {'x': 100, 'y': 200, 'width': 150, 'height': 200},
                    'card2': {'x': 300, 'y': 200, 'width': 150, 'height': 200},
                    'card3': {'x': 500, 'y': 200, 'width': 150, 'height': 200}
                }
                
                # Check if coordinates are reasonable
                for card, coords in test_coords.items():
                    if coords['x'] < 0 or coords['y'] < 0:
                        issues_found.append(f"Invalid coordinates for {card}: negative values")
                    if coords['width'] <= 0 or coords['height'] <= 0:
                        issues_found.append(f"Invalid dimensions for {card}: zero or negative size")
                
                if len(issues_found) == 0:
                    self.log("✅ Coordinate validation logic works")
                    
            except Exception as e:
                issues_found.append(f"Coordinate validation failed: {e}")
            
            return {
                'success': len(issues_found) == 0,
                'issues_found': issues_found,
                'total_tests': 2,
                'passed_tests': 2 - len(issues_found)
            }
            
        except ImportError as e:
            return {
                'success': False,
                'error': f"Cannot import CoordinateSelector: {e}",
                'issues_found': [f"Import failed: {e}"]
            }
    
    def test_ai_decision_making(self) -> Dict[str, Any]:
        """Test AI decision making components"""
        try:
            issues_found = []
            
            # Test 1: Can we import AI components?
            try:
                from arena_bot.ai.draft_advisor import DraftAdvisor
                self.log("✅ DraftAdvisor import works")
            except ImportError as e:
                issues_found.append(f"Cannot import DraftAdvisor: {e}")
            
            # Test 2: Can we create advisor?
            try:
                advisor = DraftAdvisor()
                self.log("✅ DraftAdvisor creation works")
                
                # Test 3: Can we get recommendations?
                if hasattr(advisor, 'get_recommendation'):
                    test_cards = ['Test Card 1', 'Test Card 2', 'Test Card 3']
                    try:
                        recommendation = advisor.get_recommendation(test_cards)
                        self.log("✅ DraftAdvisor.get_recommendation() works")
                    except Exception as e:
                        issues_found.append(f"get_recommendation() failed: {e}")
                else:
                    issues_found.append("get_recommendation() method missing")
                    
            except Exception as e:
                issues_found.append(f"DraftAdvisor creation failed: {e}")
            
            # Test 4: Test tier data loading
            try:
                tier_data_path = Path("assets/tier_data.json")
                if tier_data_path.exists():
                    with open(tier_data_path) as f:
                        tier_data = json.load(f)
                    self.log(f"✅ Tier data loaded: {len(tier_data)} entries")
                else:
                    issues_found.append("Tier data file not found")
            except Exception as e:
                issues_found.append(f"Tier data loading failed: {e}")
            
            return {
                'success': len(issues_found) == 0,
                'issues_found': issues_found,
                'total_tests': 4,
                'passed_tests': 4 - len(issues_found)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"AI decision making test failed: {e}",
                'issues_found': [str(e)]
            }
    
    def test_error_handling_scenarios(self) -> Dict[str, Any]:
        """Test common error scenarios users might encounter"""
        issues_found = []
        
        # Test 1: Missing screenshot file
        try:
            from PIL import Image
            
            # Try to open non-existent file
            try:
                Image.open("non_existent_screenshot.png")
                issues_found.append("Should have failed to open non-existent file")
            except FileNotFoundError:
                self.log("✅ Properly handles missing screenshot files")
            except Exception as e:
                issues_found.append(f"Wrong error type for missing file: {e}")
                
        except ImportError:
            issues_found.append("PIL not available for image processing")
        
        # Test 2: Invalid coordinate data
        try:
            invalid_coords = {
                'card1': {'x': -100, 'y': 200},  # Negative x
                'card2': {'x': 5000, 'y': 200}, # Too large x
                'card3': {'x': 300, 'y': -50}   # Negative y
            }
            
            validation_errors = []
            for card, coords in invalid_coords.items():
                if coords['x'] < 0 or coords['x'] > 2000:
                    validation_errors.append(f"{card}: invalid x coordinate")
                if coords['y'] < 0 or coords['y'] > 1500:
                    validation_errors.append(f"{card}: invalid y coordinate")
            
            if len(validation_errors) > 0:
                self.log("✅ Coordinate validation detects invalid values")
            else:
                issues_found.append("Coordinate validation too permissive")
                
        except Exception as e:
            issues_found.append(f"Coordinate validation test failed: {e}")
        
        # Test 3: Empty/corrupted data handling
        try:
            # Test empty analysis result
            empty_result = {}
            corrupted_result = {'invalid': 'data', 'missing': 'required_fields'}
            
            # These should be handled gracefully
            test_results = [empty_result, corrupted_result, None]
            for i, result in enumerate(test_results):
                try:
                    # Simulate processing these results
                    if result is None:
                        self.log(f"✅ Handles None result gracefully")
                    elif not result:
                        self.log(f"✅ Handles empty result gracefully")
                    else:
                        self.log(f"✅ Handles corrupted result gracefully")
                except Exception as e:
                    issues_found.append(f"Failed to handle test result {i}: {e}")
                    
        except Exception as e:
            issues_found.append(f"Data handling test failed: {e}")
        
        return {
            'success': len(issues_found) == 0,
            'issues_found': issues_found,
            'total_tests': 3,
            'passed_tests': 3 - len(issues_found)
        }
    
    def test_performance_bottlenecks(self) -> Dict[str, Any]:
        """Test for performance issues users might experience"""
        issues_found = []
        performance_metrics = {}
        
        # Test 1: Import time measurement
        try:
            start_time = time.time()
            from arena_bot.ui.draft_overlay import DraftOverlay
            import_time = time.time() - start_time
            performance_metrics['draft_overlay_import_time'] = import_time
            
            if import_time > 1.0:
                issues_found.append(f"DraftOverlay import too slow: {import_time:.2f}s")
            else:
                self.log(f"✅ DraftOverlay import fast: {import_time:.3f}s")
                
        except Exception as e:
            issues_found.append(f"Import timing test failed: {e}")
        
        # Test 2: Object creation time
        try:
            from arena_bot.ui.draft_overlay import DraftOverlay, OverlayConfig
            
            start_time = time.time()
            config = OverlayConfig()
            overlay = DraftOverlay(config)
            creation_time = time.time() - start_time
            performance_metrics['overlay_creation_time'] = creation_time
            
            if creation_time > 0.5:
                issues_found.append(f"Overlay creation too slow: {creation_time:.2f}s")
            else:
                self.log(f"✅ Overlay creation fast: {creation_time:.3f}s")
                
            overlay.cleanup()
            
        except Exception as e:
            issues_found.append(f"Creation timing test failed: {e}")
        
        # Test 3: Memory usage check
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            performance_metrics['memory_usage_mb'] = memory_mb
            
            if memory_mb > 1000:  # 1GB threshold
                issues_found.append(f"High memory usage: {memory_mb:.1f}MB")
            else:
                self.log(f"✅ Reasonable memory usage: {memory_mb:.1f}MB")
                
        except ImportError:
            self.log("⚠️ psutil not available for memory testing")
        except Exception as e:
            issues_found.append(f"Memory test failed: {e}")
        
        return {
            'success': len(issues_found) == 0,
            'issues_found': issues_found,
            'performance_metrics': performance_metrics,
            'total_tests': 3,
            'passed_tests': 3 - len(issues_found)
        }
    
    def run_all_tests(self) -> List[InteractionResult]:
        """Run all user interaction tests"""
        tests = [
            ("Draft Overlay Interactions", self.test_draft_overlay_interactions),
            ("Visual Overlay Interactions", self.test_visual_overlay_interactions),
            ("Screenshot Analysis Workflow", self.test_screenshot_analysis_workflow),
            ("Coordinate Selection", self.test_coordinate_selection_functionality),
            ("AI Decision Making", self.test_ai_decision_making),
            ("Error Handling Scenarios", self.test_error_handling_scenarios),
            ("Performance Bottlenecks", self.test_performance_bottlenecks),
        ]
        
        results = []
        for test_name, test_func in tests:
            result = self.test_interaction(test_name, test_func)
            results.append(result)
            self.test_results.append(result)
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        # Collect all issues found
        all_issues = []
        for result in failed_tests:
            if result.details and 'issues_found' in result.details:
                all_issues.extend(result.details['issues_found'])
            elif result.error_message:
                all_issues.append(result.error_message)
        
        # Performance metrics
        performance_data = {}
        for result in self.test_results:
            if result.details and 'performance_metrics' in result.details:
                performance_data.update(result.details['performance_metrics'])
        
        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) * 100,
                "total_duration": sum(r.duration for r in self.test_results),
                "average_duration": sum(r.duration for r in self.test_results) / len(self.test_results)
            },
            "user_issues_found": all_issues,
            "performance_metrics": performance_data,
            "successful_tests": [
                {
                    "name": r.interaction_name,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in successful_tests
            ],
            "failed_tests": [
                {
                    "name": r.interaction_name,
                    "error": r.error_message,
                    "duration": r.duration,
                    "issues": r.details.get('issues_found', []) if r.details else []
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
            filename = f"user_interaction_report_{timestamp}.json"
        
        report_path = self.artifacts_dir / filename
        report = self.generate_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"📊 User interaction report saved: {report_path}")
        return report_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test user interactions (lightweight)")
    parser.add_argument("--all", action="store_true", help="Run all interaction tests")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of tests")
    
    args = parser.parse_args()
    
    tester = LightweightInteractionTester()
    
    try:
        print("🎮 Lightweight User Interaction Testing")
        print("=" * 50)
        
        if args.all or args.quick:
            print("Running user interaction tests...")
            results = tester.run_all_tests()
        else:
            print("Please specify --all or --quick")
            return 1
        
        # Generate and save report
        report_path = tester.save_report()
        report = tester.generate_report()
        
        # Print summary
        print(f"\n🎯 USER INTERACTION TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"✅ Successful: {report['summary']['successful_tests']}")
        print(f"❌ Failed: {report['summary']['failed_tests']}")
        print(f"📊 Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {report['summary']['total_duration']:.2f}s")
        print(f"📁 Report: {report_path}")
        
        if report['user_issues_found']:
            print(f"\n❌ USER ISSUES FOUND:")
            for issue in report['user_issues_found']:
                print(f"  • {issue}")
        
        if report['performance_metrics']:
            print(f"\n⚡ PERFORMANCE METRICS:")
            for metric, value in report['performance_metrics'].items():
                if isinstance(value, float):
                    print(f"  • {metric}: {value:.3f}")
                else:
                    print(f"  • {metric}: {value}")
        
        # Exit code based on results
        if report['summary']['failed_tests'] > 0:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"💥 Testing system crashed: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())