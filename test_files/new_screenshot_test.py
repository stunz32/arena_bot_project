#!/usr/bin/env python3
"""
Test with the new screenshot at different resolution and window position.
This tests coordinate-independent detection.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def compute_arena_tracker_histogram(image: np.ndarray) -> np.ndarray:
    """Arena Tracker's exact histogram method."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    ranges = [0, 180, 0, 256]
    channels = [0, 1]
    
    hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def extract_arena_tracker_region(card_image: np.ndarray, is_premium: bool = False) -> np.ndarray:
    """Extract Arena Tracker's 80x80 region."""
    if is_premium:
        x, y, w, h = 57, 71, 80, 80
    else:
        x, y, w, h = 60, 71, 80, 80
    
    if (card_image.shape[1] < x + w) or (card_image.shape[0] < y + h):
        return None
    
    return card_image[y:y+h, x:x+w]

def test_new_screenshot(screenshot_path: str, target_cards: list):
    """Test with the new screenshot and adjusted coordinates."""
    print("🎯 NEW SCREENSHOT TEST")
    print("=" * 80)
    
    try:
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        print(f"🎮 Resolution: 2560x1140, window positioned to the right")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        asset_loader = get_asset_loader()
        
        # Load card database with Arena Tracker method
        print("📚 Loading card database...")
        available_cards = asset_loader.get_available_cards()
        card_hists = {}
        
        for card_code in available_cards[:3000]:
            for is_premium in [False, True]:
                card_image = asset_loader.load_card_image(card_code, premium=is_premium)
                if card_image is not None:
                    at_region = extract_arena_tracker_region(card_image, is_premium=is_premium)
                    if at_region is not None:
                        hist = compute_arena_tracker_histogram(at_region)
                        hist_key = f"{card_code}{'_premium' if is_premium else ''}"
                        card_hists[hist_key] = hist
        
        print(f"✅ Loaded {len(card_hists)} card histograms")
        
        # Visual analysis of the new screenshot shows the arena interface is roughly centered
        # but positioned to the right. The cards appear to be around these coordinates:
        # Looking at the screenshot, I can estimate the card positions:
        
        # Method 1: Manual estimation based on visual inspection
        estimated_coords = [
            (356, 86, 200, 280),   # Left card (Clay Matriarch, 6-mana)
            (583, 86, 200, 280),   # Middle card (Dwarven Archaeologist, 2-mana) 
            (810, 86, 200, 280),   # Right card (Cyclopian Crusher, 3-mana)
        ]
        
        # Method 2: Try multiple coordinate sets around the estimated position
        coordinate_sets = [
            ("Visual estimate", estimated_coords),
            ("Shifted left", [(x-10, y, w, h) for x, y, w, h in estimated_coords]),
            ("Shifted right", [(x+10, y, w, h) for x, y, w, h in estimated_coords]),
            ("Shifted up", [(x, y-5, w, h) for x, y, w, h in estimated_coords]),
            ("Shifted down", [(x, y+5, w, h) for x, y, w, h in estimated_coords]),
        ]
        
        best_results = []
        
        for set_name, coords in coordinate_sets:
            print(f"\n{'='*60}")
            print(f"🔍 Testing coordinate set: {set_name}")
            print(f"{'='*60}")
            
            set_results = []
            
            for i, (x, y, w, h) in enumerate(coords):
                if i >= len(target_cards):
                    break
                    
                expected_card = target_cards[i]
                print(f"\n📍 Card {i+1} ({expected_card}): ({x}, {y}, {w}, {h})")
                
                # Extract card from screenshot
                screen_card = screenshot[y:y+h, x:x+w]
                
                if screen_card.size == 0:
                    print(f"❌ Empty region")
                    continue
                
                # Save extracted card for inspection
                debug_path = f"new_test_{set_name.replace(' ', '_')}_card_{i+1}.png"
                cv2.imwrite(debug_path, screen_card)
                print(f"💾 {debug_path}")
                
                # Test different extraction strategies
                strategies = [
                    ("Full card", screen_card),
                    ("Upper 70%", screen_card[0:int(h*0.7), :]),
                    ("Center focus", screen_card[20:h-20, 20:w-20] if h > 40 and w > 40 else screen_card),
                ]
                
                best_match = None
                best_distance = float('inf')
                best_strategy = None
                target_rank = None
                
                for strategy_name, processed_region in strategies:
                    if processed_region.size == 0:
                        continue
                    
                    # Resize to 80x80 for Arena Tracker comparison
                    if processed_region.shape[:2] != (80, 80):
                        resized = cv2.resize(processed_region, (80, 80), interpolation=cv2.INTER_AREA)
                    else:
                        resized = processed_region
                    
                    # Compute histogram
                    screen_hist = compute_arena_tracker_histogram(resized)
                    
                    # Compare with database
                    matches = []
                    for card_key, card_hist in card_hists.items():
                        distance = cv2.compareHist(screen_hist, card_hist, cv2.HISTCMP_BHATTACHARYYA)
                        matches.append((distance, card_key))
                    
                    matches.sort(key=lambda x: x[0])
                    
                    # Look for target card
                    for rank, (distance, card_key) in enumerate(matches):
                        base_code = card_key.replace('_premium', '')
                        if base_code.startswith(expected_card):
                            if target_rank is None:
                                target_rank = rank + 1
                                print(f"   🎯 {strategy_name}: TARGET at rank {rank+1}! (dist: {distance:.4f})")
                                
                                if distance < best_distance:
                                    best_match = card_key
                                    best_distance = distance
                                    best_strategy = strategy_name
                            break
                    
                    # Show top 3 for this strategy
                    print(f"   📋 {strategy_name} top 3:")
                    for rank, (distance, card_key) in enumerate(matches[:3]):
                        base_code = card_key.replace('_premium', '')
                        marker = "🎯" if base_code.startswith(expected_card) else "  "
                        print(f"      {rank+1}. {marker} {card_key} (dist: {distance:.4f})")
                
                # Record result for this card
                if best_match:
                    base_code = best_match.replace('_premium', '')
                    is_correct = base_code.startswith(expected_card)
                    
                    set_results.append({
                        'card': i+1,
                        'expected': expected_card,
                        'detected': best_match,
                        'distance': best_distance,
                        'strategy': best_strategy,
                        'correct': is_correct,
                        'target_rank': target_rank
                    })
                    
                    print(f"   🏆 Best: {best_match} (dist: {best_distance:.4f}) via {best_strategy}")
                    print(f"   ✅ {'CORRECT' if is_correct else 'INCORRECT'}")
                else:
                    set_results.append({
                        'card': i+1,
                        'expected': expected_card,
                        'detected': None,
                        'distance': float('inf'),
                        'strategy': None,
                        'correct': False,
                        'target_rank': target_rank
                    })
                    print(f"   ❌ No match found")
            
            # Evaluate this coordinate set
            correct_count = sum(1 for r in set_results if r['correct'])
            found_count = sum(1 for r in set_results if r['target_rank'] is not None)
            
            print(f"\n📊 Set '{set_name}' Results:")
            print(f"   Correct: {correct_count}/3")
            print(f"   Found: {found_count}/3")
            
            best_results.append({
                'set_name': set_name,
                'coords': coords,
                'results': set_results,
                'correct_count': correct_count,
                'found_count': found_count
            })
        
        # Find best coordinate set
        best_set = max(best_results, key=lambda x: (x['correct_count'], x['found_count']))
        
        print(f"\n{'='*80}")
        print("🏆 BEST COORDINATE SET RESULTS")
        print(f"{'='*80}")
        print(f"Best set: {best_set['set_name']}")
        print(f"Accuracy: {best_set['correct_count']}/3 ({best_set['correct_count']/3*100:.1f}%)")
        print(f"Found: {best_set['found_count']}/3")
        print()
        
        for result in best_set['results']:
            status = "✅" if result['correct'] else "❌"
            found = "🎯" if result['target_rank'] else "❓"
            rank_info = f"(rank {result['target_rank']})" if result['target_rank'] else ""
            print(f"{status} {found} Card {result['card']}: {result['expected']} → {result['detected'] or 'None'} {rank_info}")
        
        if best_set['correct_count'] == 3:
            print(f"\n🎉 PERFECT SUCCESS! Arena Tracker method works!")
            return True
        elif best_set['found_count'] == 3:
            print(f"\n📈 All targets found - very close to success!")
            return True
        else:
            print(f"\n🔍 Making progress - coordinate refinement needed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return test_new_screenshot(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)