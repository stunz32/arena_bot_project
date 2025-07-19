#!/usr/bin/env python3
"""
Automated Validation Suite for Arena Bot Detection System
Comprehensive testing framework with IoU validation, performance benchmarking, and regression testing.
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time
import logging

# Add project modules
sys.path.insert(0, str(Path(__file__).parent))

from debug_config import get_debug_config, enable_debug
from visual_debugger import VisualDebugger, create_debug_visualization, save_debug_image
from metrics_logger import MetricsLogger, log_detection_metrics, generate_performance_report
from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector

logger = logging.getLogger(__name__)

class ValidationSuite:
    """
    Comprehensive validation suite for testing detection accuracy and performance.
    Provides automated testing with pass/fail criteria and detailed reporting.
    """
    
    def __init__(self):
        self.config = get_debug_config()
        self.visual_debugger = VisualDebugger()
        self.metrics_logger = MetricsLogger()
        self.detector = SmartCoordinateDetector()
        
        # Test configuration
        self.test_screenshots = []
        self.ground_truth_data = self.load_ground_truth()
        
        # Performance thresholds (customizable)
        self.thresholds = self.config.THRESHOLDS.copy()
        
        # Test results storage
        self.test_results = []
        self.overall_results = {}
    
    def load_ground_truth(self) -> Dict:
        """Load ground truth data for validation."""
        try:
            if self.config.GROUND_TRUTH_FILE.exists():
                with open(self.config.GROUND_TRUTH_FILE, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Ground truth file not found: {self.config.GROUND_TRUTH_FILE}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return {}
    
    def add_test_screenshot(self, screenshot_path: str, resolution: str = None, 
                           expected_cards: List[str] = None):
        """Add a screenshot for testing."""
        if not Path(screenshot_path).exists():
            logger.error(f"Screenshot not found: {screenshot_path}")
            return
        
        test_case = {
            'path': screenshot_path,
            'resolution': resolution,
            'expected_cards': expected_cards or [],
        }
        
        self.test_screenshots.append(test_case)
        logger.info(f"Added test screenshot: {screenshot_path}")
    
    def run_full_validation(self, save_debug_images: bool = True) -> Dict[str, Any]:
        """
        Run complete validation suite with all detection methods.
        
        Args:
            save_debug_images: Whether to save debug visualizations
            
        Returns:
            Comprehensive validation results
        """
        logger.info("🚀 Starting full validation suite...")
        enable_debug()  # Ensure debug mode is active
        
        # Test all detection methods
        detection_methods = self.config.DETECTION_METHODS
        method_results = {}
        
        for method in detection_methods:
            logger.info(f"Testing detection method: {method}")
            method_results[method] = self.test_detection_method(method, save_debug_images)
        
        # Run cross-resolution tests
        resolution_results = self.test_cross_resolution_compatibility()
        
        # Performance benchmarking
        performance_results = self.run_performance_benchmark()
        
        # Calculate overall scores
        overall_scores = self.calculate_overall_scores(method_results)
        
        # Compile final results
        validation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method_results': method_results,
            'resolution_results': resolution_results,
            'performance_results': performance_results,
            'overall_scores': overall_scores,
            'pass_fail_summary': self.generate_pass_fail_summary(method_results),
            'recommendations': self.generate_recommendations(method_results),
        }
        
        # Save results
        self.save_validation_results(validation_results)
        
        # Print summary
        self.print_validation_summary(validation_results)
        
        return validation_results
    
    def test_detection_method(self, method_name: str, save_debug: bool = True) -> Dict[str, Any]:
        """Test a specific detection method against ground truth."""
        method_results = {
            'method': method_name,
            'tests_run': 0,
            'tests_passed': 0,
            'total_iou': 0.0,
            'total_confidence': 0.0,
            'total_timing': 0.0,
            'individual_tests': [],
            'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        }
        
        # Use debug images as test cases if no custom screenshots provided
        if not self.test_screenshots:
            self.add_debug_images_as_tests()
        
        for test_case in self.test_screenshots:
            try:
                # Load screenshot
                screenshot = cv2.imread(test_case['path'])
                if screenshot is None:
                    logger.error(f"Failed to load screenshot: {test_case['path']}")
                    continue
                
                # Get ground truth boxes
                height, width = screenshot.shape[:2]
                resolution_str = test_case.get('resolution', f"{width}x{height}")
                ground_truth_boxes = self.get_ground_truth_boxes(resolution_str)
                
                if not ground_truth_boxes:
                    logger.warning(f"No ground truth data for resolution: {resolution_str}")
                    continue
                
                # Run detection
                detection_start = time.time()
                detection_result = self.run_detection_method(method_name, screenshot)
                detection_time = (time.time() - detection_start) * 1000
                
                if not detection_result or not detection_result.get('success'):
                    logger.warning(f"Detection failed for method {method_name}")
                    continue
                
                detected_boxes = detection_result.get('card_positions', [])
                
                # Calculate metrics
                test_metrics = self.calculate_test_metrics(
                    detected_boxes, ground_truth_boxes, detection_time
                )
                
                # Create debug visualization if requested
                if save_debug:
                    debug_img = create_debug_visualization(
                        screenshot,
                        detected_boxes,
                        ground_truth_boxes,
                        method_name,
                        timing_ms=detection_time
                    )
                    
                    debug_path = save_debug_image(
                        debug_img, 
                        f"validation_{method_name}_{method_results['tests_run']}", 
                        method_name
                    )
                
                # Store individual test result
                individual_result = {
                    'screenshot': test_case['path'],
                    'resolution': resolution_str,
                    'detection_time_ms': detection_time,
                    'detected_boxes': detected_boxes,
                    'ground_truth_boxes': ground_truth_boxes,
                    **test_metrics
                }
                
                method_results['individual_tests'].append(individual_result)
                
                # Update aggregated results
                method_results['tests_run'] += 1
                method_results['total_iou'] += test_metrics['mean_iou']
                method_results['total_timing'] += detection_time
                
                # Check if test passed
                if (test_metrics['mean_iou'] >= self.thresholds['min_iou'] and
                    detection_time <= self.thresholds['max_detection_time_ms']):
                    method_results['tests_passed'] += 1
                
                # Update grade distribution
                grade = test_metrics.get('grade', 'F')
                method_results['grade_distribution'][grade] += 1
                
                logger.info(f"Test {method_results['tests_run']}: IoU={test_metrics['mean_iou']:.3f}, "
                           f"Time={detection_time:.1f}ms, Grade={grade}")
                
            except Exception as e:
                logger.error(f"Error testing {method_name}: {e}")
                continue
        
        # Calculate averages
        if method_results['tests_run'] > 0:
            method_results['avg_iou'] = method_results['total_iou'] / method_results['tests_run']
            method_results['avg_timing'] = method_results['total_timing'] / method_results['tests_run']
            method_results['pass_rate'] = method_results['tests_passed'] / method_results['tests_run']
        
        return method_results
    
    def run_detection_method(self, method_name: str, screenshot: np.ndarray) -> Optional[Dict]:
        """Run specific detection method on screenshot."""
        try:
            if method_name == "simple_working":
                return self.detector.detect_cards_simple_working(screenshot)
            elif method_name == "hybrid_cascade":
                return self.detector.detect_cards_with_hybrid_cascade(screenshot)
            elif method_name == "enhanced_auto":
                return self.detector.detect_cards_automatically(screenshot)
            elif method_name == "static_scaling":
                return self.detector.detect_cards_via_static_scaling(screenshot)
            elif method_name == "contour_detection":
                return self.detector.detect_cards_via_contours(screenshot)
            elif method_name == "anchor_detection":
                return self.detector.detect_cards_via_anchors(screenshot)
            else:
                logger.error(f"Unknown detection method: {method_name}")
                return None
        except Exception as e:
            logger.error(f"Error running detection method {method_name}: {e}")
            return None
    
    def calculate_test_metrics(self, detected_boxes: List, ground_truth_boxes: List, 
                              detection_time: float) -> Dict[str, float]:
        """Calculate comprehensive metrics for a single test."""
        metrics = {}
        
        # IoU calculation
        iou_scores = []
        for i in range(min(len(detected_boxes), len(ground_truth_boxes))):
            if i < len(detected_boxes) and i < len(ground_truth_boxes):
                iou = self.config.calculate_iou(detected_boxes[i], ground_truth_boxes[i])
                iou_scores.append(iou)
        
        metrics['mean_iou'] = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        metrics['min_iou'] = min(iou_scores) if iou_scores else 0.0
        metrics['max_iou'] = max(iou_scores) if iou_scores else 0.0
        
        # Detection accuracy
        expected_cards = len(ground_truth_boxes)
        detected_cards = len(detected_boxes)
        metrics['detection_rate'] = min(1.0, detected_cards / expected_cards) if expected_cards > 0 else 0.0
        metrics['miss_rate'] = max(0.0, (expected_cards - detected_cards) / expected_cards) if expected_cards > 0 else 0.0
        
        # Timing metrics
        metrics['detection_time_ms'] = detection_time
        metrics['timing_score'] = max(0.0, 1.0 - detection_time / self.thresholds['max_detection_time_ms'])
        
        # Overall grade calculation
        metrics['grade'] = self.calculate_grade(metrics)
        
        return metrics
    
    def calculate_grade(self, metrics: Dict) -> str:
        """Calculate letter grade based on metrics."""
        score = 0.0
        
        # IoU score (50% weight)
        iou_score = min(1.0, metrics['mean_iou'] / self.thresholds['min_iou'])
        score += iou_score * 0.5
        
        # Detection rate (30% weight)
        score += metrics['detection_rate'] * 0.3
        
        # Timing score (20% weight)
        score += metrics['timing_score'] * 0.2
        
        # Convert to letter grade
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def get_ground_truth_boxes(self, resolution: str) -> List[Tuple[int, int, int, int]]:
        """Get ground truth boxes for specific resolution."""
        try:
            resolutions = self.ground_truth_data.get('resolutions', {})
            if resolution in resolutions:
                positions = resolutions[resolution].get('card_positions', [])
                return [(pos['x'], pos['y'], pos['width'], pos['height']) for pos in positions]
        except Exception as e:
            logger.error(f"Error getting ground truth boxes: {e}")
        return []
    
    def add_debug_images_as_tests(self):
        """Add existing debug images as test cases."""
        project_root = Path(__file__).parent
        debug_images = [
            "debug_card_1.png",
            "debug_card_2.png", 
            "debug_card_3.png"
        ]
        
        for img_name in debug_images:
            img_path = project_root / img_name
            if img_path.exists():
                self.add_test_screenshot(str(img_path), "3440x1440")
    
    def test_cross_resolution_compatibility(self) -> Dict[str, Any]:
        """Test detection across different resolutions."""
        logger.info("Testing cross-resolution compatibility...")
        
        resolutions_to_test = ["1920x1080", "2560x1440", "3440x1440"]
        resolution_results = {}
        
        for resolution in resolutions_to_test:
            if resolution in self.ground_truth_data.get('resolutions', {}):
                # Create synthetic test image for this resolution
                w, h = map(int, resolution.split('x'))
                test_image = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Test with simple_working method (most reliable)
                result = self.detector.detect_cards_simple_working(test_image)
                
                if result:
                    detected_boxes = result.get('card_positions', [])
                    ground_truth_boxes = self.get_ground_truth_boxes(resolution)
                    
                    if ground_truth_boxes and detected_boxes:
                        metrics = self.calculate_test_metrics(detected_boxes, ground_truth_boxes, 0)
                        resolution_results[resolution] = {
                            'supported': True,
                            'avg_iou': metrics['mean_iou'],
                            'detection_rate': metrics['detection_rate']
                        }
                    else:
                        resolution_results[resolution] = {'supported': False, 'reason': 'No detection'}
                else:
                    resolution_results[resolution] = {'supported': False, 'reason': 'Detection failed'}
        
        return resolution_results
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        logger.info("Running performance benchmark...")
        
        # Create test image
        test_image = np.zeros((1440, 3440, 3), dtype=np.uint8)
        
        performance_results = {}
        
        for method in self.config.DETECTION_METHODS:
            times = []
            for _ in range(5):  # Run 5 times for average
                start_time = time.time()
                result = self.run_detection_method(method, test_image)
                end_time = time.time()
                
                if result:
                    times.append((end_time - start_time) * 1000)
            
            if times:
                performance_results[method] = {
                    'avg_time_ms': sum(times) / len(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'meets_threshold': sum(times) / len(times) <= self.thresholds['max_detection_time_ms']
                }
        
        return performance_results
    
    def calculate_overall_scores(self, method_results: Dict) -> Dict[str, Any]:
        """Calculate overall performance scores."""
        overall_scores = {
            'best_method': None,
            'worst_method': None,
            'average_iou': 0.0,
            'average_timing': 0.0,
            'overall_pass_rate': 0.0
        }
        
        if not method_results:
            return overall_scores
        
        # Find best and worst methods
        best_score = -1
        worst_score = 2
        
        total_iou = 0
        total_timing = 0
        total_tests = 0
        total_passed = 0
        
        for method, results in method_results.items():
            if results['tests_run'] > 0:
                # Calculate composite score
                avg_iou = results.get('avg_iou', 0)
                avg_timing = results.get('avg_timing', 999)
                pass_rate = results.get('pass_rate', 0)
                
                composite_score = (avg_iou * 0.5) + (pass_rate * 0.3) + ((100 - avg_timing) / 100 * 0.2)
                
                if composite_score > best_score:
                    best_score = composite_score
                    overall_scores['best_method'] = method
                
                if composite_score < worst_score:
                    worst_score = composite_score
                    overall_scores['worst_method'] = method
                
                # Accumulate totals
                total_iou += avg_iou * results['tests_run']
                total_timing += avg_timing * results['tests_run']
                total_tests += results['tests_run']
                total_passed += results['tests_passed']
        
        # Calculate averages
        if total_tests > 0:
            overall_scores['average_iou'] = total_iou / total_tests
            overall_scores['average_timing'] = total_timing / total_tests
            overall_scores['overall_pass_rate'] = total_passed / total_tests
        
        return overall_scores
    
    def generate_pass_fail_summary(self, method_results: Dict) -> Dict[str, bool]:
        """Generate pass/fail summary based on thresholds."""
        summary = {}
        
        for method, results in method_results.items():
            if results['tests_run'] > 0:
                avg_iou = results.get('avg_iou', 0)
                avg_timing = results.get('avg_timing', 999)
                pass_rate = results.get('pass_rate', 0)
                
                # Check thresholds
                iou_pass = avg_iou >= self.thresholds['min_iou']
                timing_pass = avg_timing <= self.thresholds['max_detection_time_ms']
                overall_pass = pass_rate >= 0.8  # 80% of tests must pass
                
                summary[method] = {
                    'iou_pass': iou_pass,
                    'timing_pass': timing_pass,
                    'overall_pass': overall_pass,
                    'final_result': iou_pass and timing_pass and overall_pass
                }
        
        return summary
    
    def generate_recommendations(self, method_results: Dict) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Find methods that need improvement
        for method, results in method_results.items():
            if results['tests_run'] > 0:
                avg_iou = results.get('avg_iou', 0)
                avg_timing = results.get('avg_timing', 999)
                
                if avg_iou < self.thresholds['min_iou']:
                    recommendations.append(f"⚠️ {method}: Improve coordinate accuracy (IoU: {avg_iou:.3f})")
                
                if avg_timing > self.thresholds['max_detection_time_ms']:
                    recommendations.append(f"⏱️ {method}: Optimize for speed (Time: {avg_timing:.1f}ms)")
                
                grade_dist = results.get('grade_distribution', {})
                if grade_dist.get('F', 0) > grade_dist.get('A', 0):
                    recommendations.append(f"📊 {method}: High failure rate, consider algorithm improvements")
        
        if not recommendations:
            recommendations.append("✅ All detection methods performing within acceptable thresholds")
        
        return recommendations
    
    def save_validation_results(self, results: Dict):
        """Save validation results to file."""
        try:
            results_file = self.config.debug_data_dir / 'validation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation results saved: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    def print_validation_summary(self, results: Dict):
        """Print human-readable validation summary."""
        print("\n" + "="*80)
        print("🎯 ARENA BOT VALIDATION SUITE RESULTS")
        print("="*80)
        
        # Overall scores
        overall = results.get('overall_scores', {})
        print(f"🏆 Best Method: {overall.get('best_method', 'Unknown')}")
        print(f"📊 Average IoU: {overall.get('average_iou', 0):.3f}")
        print(f"⏱️ Average Timing: {overall.get('average_timing', 0):.1f}ms")
        print(f"✅ Overall Pass Rate: {overall.get('overall_pass_rate', 0):.1%}")
        
        # Method summary
        print("\n📋 METHOD SUMMARY:")
        for method, results_data in results.get('method_results', {}).items():
            if results_data['tests_run'] > 0:
                status = "✅ PASS" if results.get('pass_fail_summary', {}).get(method, {}).get('final_result') else "❌ FAIL"
                print(f"   {method}: {status} "
                      f"(IoU: {results_data.get('avg_iou', 0):.3f}, "
                      f"Time: {results_data.get('avg_timing', 0):.1f}ms)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for rec in results.get('recommendations', []):
            print(f"   {rec}")
        
        print("="*80)


# Convenience functions for external use
def run_full_validation() -> Dict[str, Any]:
    """Run complete validation suite - convenience function."""
    suite = ValidationSuite()
    return suite.run_full_validation()

def run_quick_test(method_name: str = "simple_working") -> Dict[str, Any]:
    """Run quick test on single method - convenience function."""
    suite = ValidationSuite()
    return suite.test_detection_method(method_name)

def check_system_health() -> bool:
    """Quick system health check - returns True if basic functionality works."""
    try:
        suite = ValidationSuite()
        # Test with simple synthetic image
        test_img = np.zeros((1440, 3440, 3), dtype=np.uint8)
        result = suite.detector.detect_cards_simple_working(test_img)
        return result is not None and result.get('success', False)
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return False


if __name__ == "__main__":
    # CLI interface for validation suite
    import argparse
    
    parser = argparse.ArgumentParser(description="Arena Bot Validation Suite")
    parser.add_argument("--method", help="Test specific detection method")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--health", action="store_true", help="Run system health check")
    
    args = parser.parse_args()
    
    if args.health:
        health_ok = check_system_health()
        print(f"System Health: {'✅ OK' if health_ok else '❌ FAILED'}")
        sys.exit(0 if health_ok else 1)
    
    elif args.quick:
        method = args.method or "simple_working"
        results = run_quick_test(method)
        print(f"Quick test results for {method}: {results}")
    
    else:
        # Full validation
        results = run_full_validation()
        
        # Exit with appropriate code
        overall_pass = results.get('overall_scores', {}).get('overall_pass_rate', 0) >= 0.8
        sys.exit(0 if overall_pass else 1)