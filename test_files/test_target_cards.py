#!/usr/bin/env python3
"""Test detection specifically for our target cards"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_target_cards():
    """Test detection specifically for TOY_380, ULD_309, TTN_042"""
    
    # Initialize components
    from arena_bot.detection.histogram_matcher import get_histogram_matcher
    from arena_bot.utils.asset_loader import get_asset_loader
    from arena_bot.data.cards_json_loader import get_cards_json_loader
    
    asset_loader = get_asset_loader()
    histogram_matcher = get_histogram_matcher()
    cards_loader = get_cards_json_loader()
    
    # Load card database
    cards_dir = asset_loader.assets_dir / "cards"
    card_images = {}
    
    for card_file in cards_dir.glob("*.png"):
        try:
            image = cv2.imread(str(card_file))
            if image is not None:
                card_code = card_file.stem
                card_images[card_code] = image
        except Exception as e:
            continue
    
    histogram_matcher.load_card_database(card_images)
    print(f"✅ Loaded {len(card_images)} card images")
    
    # Target cards we want to detect
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    print("Target card info:")
    for card_id in target_cards:
        name = cards_loader.get_card_name(card_id)
        print(f"  {card_id}: '{name}'")
    
    # Load screenshot
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    screenshot = cv2.imread(screenshot_path)
    
    # Test the extracted card regions
    card_regions = [
        (410, 120, 300, 250),  # Left card (focused)
        (855, 120, 300, 250),  # Middle card (focused)
        (1300, 120, 300, 250), # Right card (focused)
    ]
    
    print(f"\nTesting card regions against target cards...")
    
    for i, (x, y, w, h) in enumerate(card_regions):
        print(f"\n=== CARD REGION {i+1} ===")
        
        # Extract card region
        card_region = screenshot[y:y+h, x:x+w]
        
        # Get histogram for this region
        query_hist = histogram_matcher.compute_histogram(card_region)
        if query_hist is None:
            print("❌ Failed to compute histogram")
            continue
        
        # Test against each target card specifically
        print("Testing against target cards:")
        for target_id in target_cards:
            if target_id in histogram_matcher.card_histograms:
                target_hist = histogram_matcher.card_histograms[target_id]
                distance = histogram_matcher.compare_histograms(query_hist, target_hist)
                confidence = 1.0 - distance
                target_name = cards_loader.get_card_name(target_id)
                
                print(f"  {target_id} ('{target_name}'): conf={confidence:.3f}, dist={distance:.3f}")
                
                if confidence > 0.1:  # Very lenient
                    print(f"    ✅ POTENTIAL MATCH!")
            else:
                print(f"  {target_id}: Not in histogram database")
        
        # Also get top general candidates
        matches = histogram_matcher.find_best_matches(query_hist, max_candidates=3)
        print("Top 3 general candidates:")
        for j, candidate in enumerate(matches[:3]):
            name = cards_loader.get_card_name(candidate.card_code)
            print(f"  {j+1}. {candidate.card_code} ('{name}'): conf={candidate.confidence:.3f}")

if __name__ == "__main__":
    test_target_cards()