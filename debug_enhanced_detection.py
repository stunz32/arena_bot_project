#!/usr/bin/env python3
"""
Debug the enhanced detection system to understand why target cards aren't being matched.
"""

import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def debug_target_card_matching():
    """Debug why target cards aren't being matched properly."""
    print("🔍 DEBUGGING TARGET CARD MATCHING")
    print("=" * 80)
    
    # Initialize components
    from arena_bot.detection.histogram_matcher import get_histogram_matcher
    from arena_bot.data.cards_json_loader import get_cards_json_loader
    from arena_bot.utils.asset_loader import get_asset_loader
    
    histogram_matcher = get_histogram_matcher()
    cards_loader = get_cards_json_loader()
    asset_loader = get_asset_loader()
    
    # Load just the target cards and a few others for comparison
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    test_cards = target_cards + ["EX1_339", "TOY_356"]  # Add cards that were detected
    
    print(f"Loading test cards: {test_cards}")
    
    card_images = {}
    for card_code in test_cards:
        try:
            image = cv2.imread(str(asset_loader.assets_dir / "cards" / f"{card_code}.png"))
            if image is not None:
                card_images[card_code] = image
                print(f"✅ Loaded {card_code}: {image.shape}")
            else:
                print(f"❌ Failed to load {card_code}")
        except Exception as e:
            print(f"❌ Error loading {card_code}: {e}")
    
    # Load into histogram matcher
    histogram_matcher.load_card_database(card_images)
    print(f"\n📚 Loaded {len(card_images)} cards into matcher")
    
    def extract_arena_tracker_region(card_image, is_premium=False):
        """Extract Arena Tracker's 80x80 region."""
        if is_premium:
            x, y, w, h = 57, 71, 80, 80
        else:
            x, y, w, h = 60, 71, 80, 80
        
        if (card_image.shape[1] < x + w) or (card_image.shape[0] < y + h):
            # Fallback: resize and try again
            resized = cv2.resize(card_image, (218, 300), interpolation=cv2.INTER_AREA)
            return resized[y:y+h, x:x+w]
        
        return card_image[y:y+h, x:x+w]
    
    def compute_arena_tracker_histogram(image):
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
    
    # Test each extracted card region against target cards
    for i in range(1, 4):
        print(f"\n{'='*60}")
        print(f"🔍 TESTING EXTRACTED CARD {i}")
        print(f"{'='*60}")
        
        # Load the extracted card
        extracted_path = f"/home/marcco/arena_bot_project/enhanced_card_{i}.png"
        extracted_card = cv2.imread(extracted_path)
        
        if extracted_card is None:
            print(f"❌ Could not load {extracted_path}")
            continue
        
        print(f"📸 Extracted card shape: {extracted_card.shape}")
        
        # Test different region extraction strategies
        strategies = [
            ("full_card_80x80", cv2.resize(extracted_card, (80, 80), interpolation=cv2.INTER_AREA)),
            ("center_crop", extracted_card[30:extracted_card.shape[0]-30, 30:extracted_card.shape[1]-30] if extracted_card.shape[0] >= 60 and extracted_card.shape[1] >= 60 else extracted_card),
            ("upper_70_percent", extracted_card[0:int(extracted_card.shape[0]*0.7), :]),
        ]
        
        for strategy_name, processed_region in strategies:
            if processed_region.size == 0:
                continue
                
            print(f"\n📊 Strategy: {strategy_name}")
            
            # Resize to 80x80 for comparison
            if processed_region.shape[:2] != (80, 80):
                resized_region = cv2.resize(processed_region, (80, 80), interpolation=cv2.INTER_AREA)
            else:
                resized_region = processed_region
            
            # Save processed region for inspection
            debug_path = f"debug_card_{i}_{strategy_name}.png"
            cv2.imwrite(debug_path, resized_region)
            
            # Compute histogram
            screen_hist = compute_arena_tracker_histogram(resized_region)
            
            # Compare with each target card directly
            print(f"   Direct comparisons:")
            for target_code in target_cards:
                if target_code in card_images:
                    # Extract reference region
                    ref_region = extract_arena_tracker_region(card_images[target_code])
                    if ref_region is not None:
                        ref_hist = compute_arena_tracker_histogram(ref_region)
                        distance = cv2.compareHist(screen_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
                        confidence = 1.0 - distance
                        target_name = cards_loader.get_card_name(target_code)
                        
                        print(f"      {target_code} ({target_name}): dist={distance:.4f}, conf={confidence:.3f}")
                        
                        if confidence > 0.1:
                            print(f"         🎯 POTENTIAL MATCH!")
            
            # Also test with histogram matcher
            matches = histogram_matcher.find_best_matches(screen_hist, max_candidates=10)
            print(f"   Top 10 histogram matches:")
            for j, match in enumerate(matches):
                marker = "🎯" if match.card_code in target_cards else "  "
                name = cards_loader.get_card_name(match.card_code)
                print(f"      {j+1:2d}. {marker} {match.card_code} ({name}): conf={match.confidence:.3f}")
    
    # Test target card histograms against each other
    print(f"\n{'='*80}")
    print("🔍 TARGET CARD CROSS-COMPARISON")
    print(f"{'='*80}")
    
    target_hists = {}
    for target_code in target_cards:
        if target_code in card_images:
            ref_region = extract_arena_tracker_region(card_images[target_code])
            if ref_region is not None:
                target_hists[target_code] = compute_arena_tracker_histogram(ref_region)
                cv2.imwrite(f"target_reference_{target_code}.png", ref_region)
    
    print("Cross-comparison matrix:")
    for code1 in target_hists:
        for code2 in target_hists:
            if code1 != code2:
                distance = cv2.compareHist(target_hists[code1], target_hists[code2], cv2.HISTCMP_BHATTACHARYYA)
                print(f"   {code1} vs {code2}: distance={distance:.4f}")


def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    debug_target_card_matching()


if __name__ == "__main__":
    main()