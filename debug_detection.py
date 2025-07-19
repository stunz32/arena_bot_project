#!/usr/bin/env python3
"""Debug the card detection process in detail"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def debug_detection():
    """Debug card detection step by step"""
    
    # Initialize components
    from arena_bot.detection.histogram_matcher import get_histogram_matcher
    from arena_bot.utils.asset_loader import get_asset_loader
    
    asset_loader = get_asset_loader()
    histogram_matcher = get_histogram_matcher()
    
    # Load card database
    cards_dir = asset_loader.assets_dir / "cards"
    card_images = {}
    card_count = 0
    
    print("Loading card database...")
    for card_file in cards_dir.glob("*.png"):
        try:
            image = cv2.imread(str(card_file))
            if image is not None:
                card_code = card_file.stem
                card_images[card_code] = image
                card_count += 1
        except Exception as e:
            continue
    
    histogram_matcher.load_card_database(card_images)
    print(f"✅ Loaded {card_count} card images")
    
    # Load screenshot
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    screenshot = cv2.imread(screenshot_path)
    print(f"✅ Screenshot loaded: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    # Test each card region
    card_regions = [
        (410, 120, 300, 250),  # Left card (focused)
        (855, 120, 300, 250),  # Middle card (focused)
        (1300, 120, 300, 250), # Right card (focused)
    ]
    
    for i, (x, y, w, h) in enumerate(card_regions):
        print(f"\n=== TESTING CARD {i+1} ===")
        
        # Extract card region
        card_region = screenshot[y:y+h, x:x+w]
        print(f"Card region size: {card_region.shape[1]}x{card_region.shape[0]}")
        
        # Try with very low confidence threshold
        print("Testing with confidence threshold 0.1...")
        match = histogram_matcher.match_card(card_region, confidence_threshold=0.1)
        
        if match:
            print(f"✅ MATCH FOUND!")
            print(f"   Card: {match.card_code}")
            print(f"   Confidence: {match.confidence:.3f}")
            print(f"   Distance: {match.distance:.3f}")
            print(f"   Premium: {match.is_premium}")
        else:
            print("❌ No match found even with 0.1 threshold")
            
            # Try to get the top candidates regardless of threshold
            query_hist = histogram_matcher.compute_histogram(card_region)
            if query_hist is not None:
                matches = histogram_matcher.find_best_matches(query_hist, max_candidates=3)
                print(f"Top 3 candidates:")
                for j, candidate in enumerate(matches[:3]):
                    print(f"   {j+1}. {candidate.card_code}: conf={candidate.confidence:.3f}, dist={candidate.distance:.3f}")
            else:
                print("❌ Failed to compute histogram")

if __name__ == "__main__":
    debug_detection()