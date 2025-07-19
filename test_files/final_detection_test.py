#!/usr/bin/env python3
"""Final detection test - check if target cards are in top candidates"""

import sys
import cv2
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def final_detection_test():
    """Test if target cards appear in top candidates"""
    
    from arena_bot.detection.histogram_matcher import get_histogram_matcher
    from arena_bot.utils.asset_loader import get_asset_loader
    from arena_bot.data.cards_json_loader import get_cards_json_loader
    
    asset_loader = get_asset_loader()
    histogram_matcher = get_histogram_matcher()
    cards_loader = get_cards_json_loader()
    
    # Load filtered card database (same as integrated bot)
    cards_dir = asset_loader.assets_dir / "cards"
    card_images = {}
    
    for card_file in cards_dir.glob("*.png"):
        try:
            image = cv2.imread(str(card_file))
            if image is not None:
                card_code = card_file.stem
                # Filter out non-draftable cards (HERO, BG, etc.)
                if not any(card_code.startswith(prefix) for prefix in ['HERO_', 'BG_', 'TB_', 'KARA_']):
                    card_images[card_code] = image
        except Exception as e:
            continue
    
    histogram_matcher.load_card_database(card_images)
    print(f"✅ Loaded {len(card_images)} card images")
    
    # Load screenshot and test regions
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    screenshot = cv2.imread(screenshot_path)
    
    card_regions = [
        (410, 120, 300, 250),  # Left card
        (855, 120, 300, 250),  # Middle card
        (1300, 120, 300, 250), # Right card
    ]
    
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    target_names = [cards_loader.get_card_name(card_id) for card_id in target_cards]
    
    print(f"Target cards: {', '.join(target_names)}")
    print(f"Looking for: {', '.join(target_cards)}")
    
    for i, (x, y, w, h) in enumerate(card_regions):
        print(f"\n=== CARD REGION {i+1} ===")
        
        card_region = screenshot[y:y+h, x:x+w]
        query_hist = histogram_matcher.compute_histogram(card_region)
        
        if query_hist is not None:
            # Get top 10 candidates
            matches = histogram_matcher.find_best_matches(query_hist, max_candidates=10)
            
            print(f"Top 10 candidates:")
            found_target = False
            for j, match in enumerate(matches):
                name = cards_loader.get_card_name(match.card_code)
                marker = "🎯" if match.card_code in target_cards else "  "
                if match.card_code in target_cards:
                    found_target = True
                print(f"{marker} {j+1:2d}. {match.card_code} ('{name}') - {match.confidence:.3f}")
            
            if found_target:
                print("✅ TARGET CARD FOUND IN TOP 10!")
            else:
                print("❌ No target cards in top 10")

if __name__ == "__main__":
    final_detection_test()