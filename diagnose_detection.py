#!/usr/bin/env python3
"""
Diagnose card detection issues and test against known correct cards.
Implements Arena Tracker's exact histogram computation and matching techniques.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def diagnose_card_detection(screenshot_path: str, correct_cards: list):
    """Diagnose why detection is failing and test specific improvements."""
    print("🔬 CARD DETECTION DIAGNOSIS")
    print("=" * 80)
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print("❌ Failed to load screenshot")
            return False
        
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        from arena_bot.core.window_detector import get_window_detector
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        window_detector = get_window_detector()
        window_detector.initialize()
        
        # Load the correct cards specifically
        print(f"🎯 Loading target cards: {correct_cards}")
        target_images = {}
        for card_code in correct_cards:
            normal = asset_loader.load_card_image(card_code, premium=False)
            premium = asset_loader.load_card_image(card_code, premium=True)
            if normal is not None:
                target_images[card_code] = normal
            if premium is not None:
                target_images[f"{card_code}_premium"] = premium
        
        print(f"✅ Loaded {len(target_images)} target card images")
        
        # Also load a broader set for comparison
        print("📚 Loading comparison database...")
        available_cards = asset_loader.get_available_cards()
        
        # Load cards from the same sets as correct cards for comparison
        comparison_cards = []
        for card_code in available_cards:
            if (card_code.startswith('TOY_') or 
                card_code.startswith('ULD_') or 
                card_code.startswith('TTN_') or
                card_code.startswith('ONY_') or
                card_code.startswith('EX1_')):
                comparison_cards.append(card_code)
        
        print(f"   Loading {len(comparison_cards)} cards from relevant sets...")
        comparison_images = {}
        for card_code in comparison_cards[:200]:  # Limit for speed
            normal = asset_loader.load_card_image(card_code, premium=False)
            if normal is not None:
                comparison_images[card_code] = normal
        
        all_images = {**target_images, **comparison_images}
        histogram_matcher.load_card_database(all_images)
        print(f"✅ Loaded {histogram_matcher.get_database_size()} histograms for analysis")
        
        # Get regions from auto-detection
        ui_elements = window_detector.auto_detect_arena_cards(screenshot)
        if ui_elements is None:
            print("❌ Failed to detect arena interface")
            return False
        
        print(f"🎯 Detected {len(ui_elements.card_regions)} card regions")
        
        # Analyze each region in detail
        for i, (x, y, w, h) in enumerate(ui_elements.card_regions):
            print(f"\n{'='*60}")
            print(f"🔍 ANALYZING CARD {i+1} (Expected: {correct_cards[i] if i < len(correct_cards) else 'Unknown'})")
            print(f"Region: ({x}, {y}, {w}, {h})")
            
            # Extract card region
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            w = min(w, width - x)
            h = min(h, height - y)
            
            card_image = screenshot[y:y+h, x:x+w]
            
            # Save for inspection
            debug_path = f"debug_card_{i+1}.png"
            cv2.imwrite(debug_path, card_image)
            print(f"💾 Saved region to {debug_path}")
            
            # Test different histogram approaches
            print("\n📊 HISTOGRAM ANALYSIS:")
            
            # Current approach
            current_hist = histogram_matcher.compute_histogram(card_image)
            if current_hist is not None:
                print(f"   Current histogram shape: {current_hist.shape}")
                print(f"   Histogram range: {current_hist.min():.3f} to {current_hist.max():.3f}")
                print(f"   Histogram sum: {current_hist.sum():.3f}")
            
            # Test against target card directly
            if i < len(correct_cards):
                expected_card = correct_cards[i]
                if expected_card in target_images:
                    expected_image = target_images[expected_card]
                    expected_hist = histogram_matcher.compute_histogram(expected_image)
                    
                    if current_hist is not None and expected_hist is not None:
                        distance = histogram_matcher.compare_histograms(current_hist, expected_hist)
                        print(f"   Direct comparison with {expected_card}: distance = {distance:.4f}")
                        
                        # Also test with premium version
                        premium_key = f"{expected_card}_premium"
                        if premium_key in target_images:
                            premium_image = target_images[premium_key]
                            premium_hist = histogram_matcher.compute_histogram(premium_image)
                            premium_distance = histogram_matcher.compare_histograms(current_hist, premium_hist)
                            print(f"   Direct comparison with {premium_key}: distance = {premium_distance:.4f}")
            
            # Get top matches from database
            print("\n🏆 TOP MATCHES:")
            if current_hist is not None:
                matches = histogram_matcher.find_best_matches(current_hist, max_candidates=10)
                for j, match in enumerate(matches[:5]):
                    marker = "⭐" if match.card_code in correct_cards else "  "
                    print(f"   {j+1}. {marker} {match.card_code} (dist: {match.distance:.4f}, conf: {match.confidence:.3f})")
            
            # Test different preprocessing approaches
            print("\n🔧 TESTING PREPROCESSING IMPROVEMENTS:")
            
            # Arena Tracker uses the full card image for histogram
            # Let's test with different regions and preprocessing
            
            # Test 1: Different crop regions
            test_regions = [
                ("Full card", card_image),
                ("Center 80%", card_image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]),
                ("Lower 60%", card_image[int(h*0.4):h, :]),
                ("Upper 60%", card_image[0:int(h*0.6), :])
            ]
            
            for region_name, region_image in test_regions:
                if region_image.size > 0:
                    test_hist = histogram_matcher.compute_histogram(region_image)
                    if test_hist is not None and i < len(correct_cards):
                        expected_card = correct_cards[i]
                        if expected_card in target_images:
                            expected_image = target_images[expected_card]
                            expected_hist = histogram_matcher.compute_histogram(expected_image)
                            distance = histogram_matcher.compare_histograms(test_hist, expected_hist)
                            print(f"   {region_name}: distance to {expected_card} = {distance:.4f}")
            
            # Test 2: Different histogram parameters
            print("\n🎛️  TESTING HISTOGRAM PARAMETERS:")
            
            # Test different bin counts (Arena Tracker uses 50x60)
            bin_tests = [
                (30, 40, "30x40 bins"),
                (50, 60, "50x60 bins (current)"),
                (70, 80, "70x80 bins")
            ]
            
            for h_bins, s_bins, name in bin_tests:
                try:
                    # Convert to HSV
                    hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
                    
                    # Compute histogram with different parameters
                    hist = cv2.calcHist(
                        [hsv], [0, 1], None, 
                        [h_bins, s_bins], 
                        [0, 180, 0, 256]
                    )
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    
                    if i < len(correct_cards):
                        expected_card = correct_cards[i]
                        if expected_card in target_images:
                            expected_image = target_images[expected_card]
                            expected_hsv = cv2.cvtColor(expected_image, cv2.COLOR_BGR2HSV)
                            expected_hist = cv2.calcHist(
                                [expected_hsv], [0, 1], None,
                                [h_bins, s_bins],
                                [0, 180, 0, 256]
                            )
                            cv2.normalize(expected_hist, expected_hist, 0, 1, cv2.NORM_MINMAX)
                            
                            distance = cv2.compareHist(hist, expected_hist, cv2.HISTCMP_BHATTACHARYYA)
                            print(f"   {name}: distance to {expected_card} = {distance:.4f}")
                            
                except Exception as e:
                    print(f"   {name}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    correct_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    return diagnose_card_detection(screenshot_path, correct_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)