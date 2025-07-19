#!/usr/bin/env python3
"""
Deep diagnosis to understand why TOY_380, ULD_309, TTN_042 aren't being detected.
Check if they exist in database and analyze their histograms vs detected cards.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def deep_diagnosis(screenshot_path: str, target_cards: list):
    """Deep dive into why specific cards aren't being detected."""
    print("🔬 DEEP DIAGNOSIS: Card Detection Analysis")
    print("=" * 80)
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
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
        
        # Load full database
        print("📚 Loading FULL card database (all available cards)...")
        available_cards = asset_loader.get_available_cards()
        all_images = {}
        
        # Track if we find target cards
        target_cards_found = []
        
        print(f"🔍 Searching through {len(available_cards)} cards for targets: {target_cards}")
        
        for i, card_code in enumerate(available_cards):
            # Check if this is one of our target cards
            if any(card_code.startswith(target) for target in target_cards):
                print(f"✅ Found target card: {card_code}")
                target_cards_found.append(card_code)
            
            normal = asset_loader.load_card_image(card_code, premium=False)
            premium = asset_loader.load_card_image(card_code, premium=True)
            if normal is not None:
                all_images[card_code] = normal
            if premium is not None:
                all_images[f"{card_code}_premium"] = premium
            
            # Progress indicator
            if i % 1000 == 0:
                print(f"   Processed {i}/{len(available_cards)} cards...")
        
        print(f"\n🎯 Target cards found in database: {target_cards_found}")
        if len(target_cards_found) != len(target_cards):
            print("⚠️  Some target cards missing from database!")
            missing = [card for card in target_cards if not any(found.startswith(card) for found in target_cards_found)]
            print(f"   Missing: {missing}")
        
        histogram_matcher.load_card_database(all_images)
        print(f"✅ Loaded {histogram_matcher.get_database_size()} histograms")
        
        # Get card regions
        ui_elements = window_detector.auto_detect_arena_cards(screenshot)
        if ui_elements is None:
            print("❌ Failed to detect arena interface")
            return False
        
        regions = ui_elements.card_regions
        print(f"🎯 Analyzing {len(regions)} card regions")
        
        # For each region, do deep analysis
        for i, region in enumerate(regions):
            if i >= len(target_cards):
                break
                
            expected_card = target_cards[i]
            print(f"\n{'='*60}")
            print(f"🔍 DEEP ANALYSIS - Card {i+1} (Expected: {expected_card})")
            print(f"{'='*60}")
            
            # Extract card image
            x, y, w, h = region
            card_image = screenshot[y:y+h, x:x+w]
            
            # Save extracted card for inspection
            debug_path = f"deep_diagnosis_card_{i+1}.png"
            cv2.imwrite(debug_path, card_image)
            print(f"💾 Saved extracted card: {debug_path}")
            
            # Compute histogram
            hist = histogram_matcher.compute_histogram(card_image)
            if hist is None:
                print("❌ Failed to compute histogram")
                continue
            
            # Get TOP 20 matches to see where target card ranks
            candidates = histogram_matcher.find_best_matches(hist, max_candidates=50)
            print(f"\n📊 TOP 20 MATCHES:")
            
            target_found_rank = None
            for j, candidate in enumerate(candidates[:20]):
                is_target = any(candidate.card_code.startswith(target) for target in target_cards)
                is_expected = candidate.card_code.startswith(expected_card)
                
                marker = "🎯" if is_expected else ("⭐" if is_target else "  ")
                
                print(f"   {j+1:2d}. {marker} {candidate.card_code:15s} (dist: {candidate.distance:.4f}, conf: {candidate.confidence:.3f})")
                
                if is_expected and target_found_rank is None:
                    target_found_rank = j + 1
            
            if target_found_rank:
                print(f"\n✅ Expected card found at rank {target_found_rank}")
            else:
                print(f"\n❌ Expected card NOT in top 20")
                
                # Look deeper - check if it exists at all
                print(f"🔍 Searching entire database for {expected_card}...")
                for j, candidate in enumerate(candidates):
                    if candidate.card_code.startswith(expected_card):
                        print(f"   Found {candidate.card_code} at rank {j+1} (dist: {candidate.distance:.4f})")
                        break
                else:
                    print(f"   {expected_card} not found in any matches!")
            
            # Compare top match vs expected match
            if candidates:
                top_match = candidates[0]
                print(f"\n📈 COMPARISON:")
                print(f"   Top match: {top_match.card_code} (dist: {top_match.distance:.4f})")
                
                # Find the expected card's score
                expected_match = None
                for candidate in candidates:
                    if candidate.card_code.startswith(expected_card):
                        expected_match = candidate
                        break
                
                if expected_match:
                    distance_diff = expected_match.distance - top_match.distance
                    print(f"   Expected:  {expected_match.card_code} (dist: {expected_match.distance:.4f})")
                    print(f"   Difference: {distance_diff:.4f} (expected is {'worse' if distance_diff > 0 else 'better'})")
                    
                    if distance_diff > 0:
                        print(f"   💡 Expected card is {distance_diff:.4f} points worse than top match")
                    else:
                        print(f"   ⚠️  Expected card is actually better but not selected!")
                else:
                    print(f"   Expected card not found in database matches")
        
        # Summary of findings
        print(f"\n{'='*80}")
        print("🔬 DIAGNOSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Target cards in database: {len(target_cards_found)}/{len(target_cards)}")
        print(f"Database size: {histogram_matcher.get_database_size()} histograms")
        print(f"Cards analyzed: {len(target_cards_found)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deep diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return deep_diagnosis(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)