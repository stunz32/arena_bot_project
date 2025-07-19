#!/usr/bin/env python3
"""
Simple auto-detection test using our proven methods.
"""

import cv2
import numpy as np
import logging
from arena_bot.detection.histogram_matcher import get_histogram_matcher
from arena_bot.utils.asset_loader import get_asset_loader
from arena_bot.core.surf_detector import get_surf_detector

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def compute_arena_tracker_histogram(image: np.ndarray) -> np.ndarray:
    """Compute Arena Tracker's exact histogram."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    ranges = [0, 180, 0, 256]
    channels = [0, 1]
    hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def main():
    """Test auto detection."""
    print("=== Simple Auto Detection Test ===")
    
    # Load screenshot
    screenshot = cv2.imread("screenshot.png")
    print(f"Screenshot: {screenshot.shape}")
    
    # Use SURF detector to find interface
    surf_detector = get_surf_detector()
    interface_rect = surf_detector.detect_arena_interface(screenshot)
    
    if interface_rect is None:
        print("✗ Interface detection failed")
        return
    
    print(f"✓ Interface detected: {interface_rect}")
    
    # Calculate card positions
    card_positions = surf_detector.calculate_card_positions(interface_rect)
    print(f"✓ Card positions: {card_positions}")
    
    # Load histogram matcher with small database
    histogram_matcher = get_histogram_matcher()
    asset_loader = get_asset_loader()
    
    # Load a small subset of cards for testing
    test_cards = ['TOY_380', 'ULD_309', 'TTN_042']  # Our target cards
    card_hists = {}
    
    for card_code in test_cards:
        for is_premium in [False, True]:
            card_image = asset_loader.load_card_image(card_code, premium=is_premium)
            if card_image is not None:
                # Use same region extraction as Arena Tracker
                if is_premium:
                    x, y, w, h = 57, 71, 80, 80
                else:
                    x, y, w, h = 60, 71, 80, 80
                
                if (card_image.shape[1] >= x + w) and (card_image.shape[0] >= y + h):
                    region = card_image[y:y+h, x:x+w]
                    hist = compute_arena_tracker_histogram(region)
                    hist_key = f"{card_code}{'_premium' if is_premium else ''}"
                    card_hists[hist_key] = hist
    
    histogram_matcher.load_card_database(card_hists)
    print(f"✓ Loaded {len(card_hists)} test histograms")
    
    # Extract and test each card
    for i, (x, y, w, h) in enumerate(card_positions):
        print(f"\nCard {i+1}: ({x}, {y}, {w}, {h})")
        
        # Extract card image
        card_image = screenshot[y:y+h, x:x+w]
        
        # Use center crop strategy like in final_success_test.py
        card_h, card_w = card_image.shape[:2]
        if card_h >= 60 and card_w >= 60:
            processed_region = card_image[30:card_h-30, 30:card_w-30]
        else:
            processed_region = card_image
        
        # Resize to 80x80
        if processed_region.shape[:2] != (80, 80):
            resized = cv2.resize(processed_region, (80, 80), interpolation=cv2.INTER_AREA)
        else:
            resized = processed_region
        
        # Compute histogram and find best match
        screen_hist = compute_arena_tracker_histogram(resized)
        match_result = histogram_matcher.find_best_match_with_confidence(screen_hist)
        
        if match_result:
            print(f"  ✓ Detected: {match_result['card_code']} (confidence: {match_result['confidence']:.3f})")
        else:
            print(f"  ✗ No match found")

if __name__ == "__main__":
    main()