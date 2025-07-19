#!/usr/bin/env python3
"""
Coordinate Detection Regression Test

This test ensures that coordinate detection accuracy never drops below 
the established threshold of 92% IoU for the calibrated resolution 2574x1339.

Created: 2025-07-15
Purpose: Guard against regressions in coordinate detection accuracy
"""

import cv2
import numpy as np
import sys
import os
from typing import List, Tuple, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector


class CoordinateRegressionTest:
    """Regression test for coordinate detection accuracy"""
    
    # Test configuration
    MIN_IOU_THRESHOLD = 0.92  # Minimum acceptable IoU
    MIN_PERFECT_CARDS = 2     # Minimum number of cards that must be perfect
    TEST_SCREENSHOT = "debug_frames/Screenshot 2025-07-05 085410.png"
    
    # Ground truth coordinates (CORRECTED to actual visible cards 2025-07-15)
    GROUND_TRUTH = [
        (580, 100, 185, 260),   # Card 1: Funhouse Mirror - actual visible card
        (785, 100, 185, 260),   # Card 2: Holy Nova - actual visible card
        (990, 100, 185, 260)    # Card 3: Mystified To'cha - actual visible card
    ]
    
    def __init__(self):
        self.detector = SmartCoordinateDetector()
        self.results = {}
        
    def calculate_iou(self, detected: Tuple[int, int, int, int], 
                     truth: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between detected and ground truth boxes"""
        dx1, dy1, dw, dh = detected
        dx2, dy2 = dx1 + dw, dy1 + dh
        
        tx1, ty1, tw, th = truth
        tx2, ty2 = tx1 + tw, ty1 + th
        
        # Intersection
        ix1, iy1 = max(dx1, tx1), max(dy1, ty1)
        ix2, iy2 = min(dx2, tx2), min(dy2, ty2)
        
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (dw * dh) + (tw * th) - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def load_test_screenshot(self) -> np.ndarray:
        """Load the test screenshot"""
        if not os.path.exists(self.TEST_SCREENSHOT):
            raise FileNotFoundError(f"Test screenshot not found: {self.TEST_SCREENSHOT}")
            
        screenshot = cv2.imread(self.TEST_SCREENSHOT)
        if screenshot is None:
            raise ValueError(f"Could not load screenshot: {self.TEST_SCREENSHOT}")
            
        return screenshot
        
    def run_detection(self, screenshot: np.ndarray) -> Dict:
        """Run coordinate detection on the screenshot"""
        result = self.detector.detect_cards_via_static_scaling(screenshot)
        
        if not result or not result.get('success'):
            raise RuntimeError("Detection failed to find cards")
            
        if len(result['card_positions']) != len(self.GROUND_TRUTH):
            raise RuntimeError(f"Expected {len(self.GROUND_TRUTH)} cards, got {len(result['card_positions'])}")
            
        return result
        
    def validate_accuracy(self, detected_positions: List[Tuple[int, int, int, int]]) -> Dict:
        """Validate detection accuracy against ground truth"""
        results = {
            'card_ious': [],
            'perfect_cards': 0,
            'average_iou': 0.0,
            'passed': False
        }
        
        total_iou = 0.0
        
        for i, (detected, truth) in enumerate(zip(detected_positions, self.GROUND_TRUTH)):
            iou = self.calculate_iou(detected, truth)
            results['card_ious'].append({
                'card_id': i + 1,
                'iou': iou,
                'detected': detected,
                'truth': truth,
                'perfect': iou >= self.MIN_IOU_THRESHOLD
            })
            
            total_iou += iou
            if iou >= self.MIN_IOU_THRESHOLD:
                results['perfect_cards'] += 1
                
        results['average_iou'] = total_iou / len(detected_positions)
        results['passed'] = (
            results['average_iou'] >= self.MIN_IOU_THRESHOLD and 
            results['perfect_cards'] >= self.MIN_PERFECT_CARDS
        )
        
        return results
        
    def print_results(self, results: Dict) -> None:
        """Print detailed test results"""
        print("=" * 60)
        print("🔍 COORDINATE DETECTION REGRESSION TEST")
        print("=" * 60)
        print(f"Screenshot: {self.TEST_SCREENSHOT}")
        print(f"Resolution: 2574x1339 (calibrated)")
        print(f"Min IoU Threshold: {self.MIN_IOU_THRESHOLD}")
        print()
        
        print("📊 INDIVIDUAL CARD RESULTS:")
        for card in results['card_ious']:
            status = "✅ PERFECT" if card['perfect'] else "❌ FAILED"
            print(f"  Card {card['card_id']}: IoU={card['iou']:.3f} {status}")
            print(f"    Detected: {card['detected']}")
            print(f"    Truth:    {card['truth']}")
            print()
            
        print("🎯 SUMMARY:")
        print(f"  Average IoU: {results['average_iou']:.3f}")
        print(f"  Perfect cards: {results['perfect_cards']}/{len(self.GROUND_TRUTH)}")
        print(f"  Min threshold: {self.MIN_IOU_THRESHOLD}")
        print()
        
        if results['passed']:
            print("🏆 TEST PASSED - Coordinate detection accuracy maintained!")
        else:
            print("❌ TEST FAILED - Coordinate detection accuracy degraded!")
            print("   This indicates a regression in the detection algorithm.")
            print("   Please review recent changes and recalibrate if necessary.")
            
    def run_test(self) -> bool:
        """Run the complete regression test"""
        try:
            # Load test data
            screenshot = self.load_test_screenshot()
            print(f"📸 Loaded test screenshot: {screenshot.shape[1]}x{screenshot.shape[0]}")
            
            # Run detection
            detection_result = self.run_detection(screenshot)
            print(f"🔍 Detection method: {detection_result['detection_method']}")
            
            # Validate accuracy
            validation_results = self.validate_accuracy(detection_result['card_positions'])
            
            # Print results
            self.print_results(validation_results)
            
            return validation_results['passed']
            
        except Exception as e:
            print(f"❌ REGRESSION TEST ERROR: {e}")
            return False


def main():
    """Main entry point for regression test"""
    test = CoordinateRegressionTest()
    passed = test.run_test()
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()