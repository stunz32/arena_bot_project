#!/usr/bin/env python3
"""
Test with the correct Hearthstone interface coordinates found by red area detection.
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

def test_correct_coordinates(screenshot_path: str, target_cards: list):
    """Test with the correct coordinates found by red area detection."""
    print("🎯 TESTING WITH CORRECT COORDINATES")
    print("=" * 80)
    
    try:
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # The red area detection found the Hearthstone interface at (1333, 180, 1197, 704)
        # This means the interface starts at x=1333, y=180
        # Within this interface, the 3 cards are positioned horizontally
        
        interface_x = 1333
        interface_y = 180
        interface_w = 1197
        interface_h = 704
        
        print(f"🎮 Hearthstone interface: ({interface_x}, {interface_y}, {interface_w}, {interface_h})")
        
        # Extract the interface for inspection
        hearthstone_interface = screenshot[interface_y:interface_y+interface_h, interface_x:interface_x+interface_w]
        cv2.imwrite("hearthstone_interface.png", hearthstone_interface)
        print(f"💾 Saved interface: hearthstone_interface.png")
        
        # Now calculate card positions within the interface
        # Based on the extracted interface image, the 3 cards appear to be positioned roughly:
        # - Each card is approximately 218 pixels wide
        # - Cards start around y=90 within the interface
        # - Cards are spaced evenly across the interface width
        
        # Calculate card positions relative to the full screenshot
        card_y_offset = 90  # Y position within interface
        card_height = 300
        card_width = 218
        
        # Distribute 3 cards across the interface width
        card_spacing = interface_w // 4  # Divide into 4 sections, cards in positions 1, 2, 3
        
        card_coords = []
        for i in range(3):
            card_x = interface_x + card_spacing * (i + 1) - card_width // 2
            card_y = interface_y + card_y_offset
            card_coords.append((card_x, card_y, card_width, card_height))
        
        print(f"📍 Calculated card coordinates:")
        for i, (x, y, w, h) in enumerate(card_coords):
            print(f"   Card {i+1}: ({x}, {y}, {w}, {h})")
        
        # Initialize Arena Tracker components
        from arena_bot.utils.asset_loader import get_asset_loader
        asset_loader = get_asset_loader()
        
        # Load card database
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
        
        # Test with calculated coordinates
        results = []
        
        for i, (x, y, w, h) in enumerate(card_coords):
            if i >= len(target_cards):
                break
                
            expected_card = target_cards[i]
            print(f"\n{'='*60}")
            print(f"🔍 TESTING CARD {i+1}")
            print(f"Expected: {expected_card}")
            print(f"Coordinates: ({x}, {y}, {w}, {h})")
            print(f"{'='*60}")
            
            # Extract card from screenshot
            screen_card = screenshot[y:y+h, x:x+w]
            
            if screen_card.size == 0:
                print(f"❌ Empty region")
                continue
            
            # Save extracted card
            debug_path = f"correct_coords_card_{i+1}.png"
            cv2.imwrite(debug_path, screen_card)
            print(f"💾 Saved: {debug_path}")
            
            # Test different extraction strategies
            strategies = [
                ("Full card", screen_card),
                ("Upper 70%", screen_card[0:int(h*0.7), :]),
                ("Card art region", screen_card[20:150, 20:w-20] if h >= 170 and w >= 40 else screen_card),
                ("Center crop", screen_card[30:h-30, 30:w-30] if h >= 60 and w >= 60 else screen_card),
            ]
            
            best_match = None
            best_distance = float('inf')
            best_strategy = None
            target_rank = None
            
            for strategy_name, processed_region in strategies:
                if processed_region.size == 0:
                    print(f"\n📊 {strategy_name}: ❌ Empty region")
                    continue
                
                print(f"\n📊 Strategy: {strategy_name} {processed_region.shape}")
                
                # Save processed region
                processed_path = f"correct_coords_card_{i+1}_{strategy_name.replace(' ', '_').lower()}.png"
                cv2.imwrite(processed_path, processed_region)
                
                # Resize to 80x80 for Arena Tracker comparison
                if processed_region.shape[:2] != (80, 80):
                    resized = cv2.resize(processed_region, (80, 80), interpolation=cv2.INTER_AREA)
                else:
                    resized = processed_region
                
                # Compute Arena Tracker histogram
                screen_hist = compute_arena_tracker_histogram(resized)
                
                # Compare with database using Bhattacharyya distance
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
                            print(f"   🎯 TARGET FOUND at rank {rank+1}! Distance: {distance:.4f}")
                            
                            if distance < best_distance:
                                best_match = card_key
                                best_distance = distance
                                best_strategy = strategy_name
                        break
                
                # Show top 5 matches
                print(f"   📋 Top 5:")
                for rank, (distance, card_key) in enumerate(matches[:5]):
                    base_code = card_key.replace('_premium', '')
                    marker = "🎯" if base_code.startswith(expected_card) else "  "
                    print(f"      {rank+1}. {marker} {card_key:20s} (dist: {distance:.4f})")
                
                # Consider top match if very good
                if matches and matches[0][0] < 0.3:  # Very good match threshold
                    top_distance, top_card = matches[0]
                    if top_distance < best_distance:
                        best_match = top_card
                        best_distance = top_distance
                        best_strategy = strategy_name
            
            # Record result
            if best_match:
                base_code = best_match.replace('_premium', '')
                is_correct = base_code.startswith(expected_card)
                
                print(f"\n🏆 RESULT:")
                print(f"   Strategy: {best_strategy}")
                print(f"   Detected: {best_match}")
                print(f"   Distance: {best_distance:.4f}")
                print(f"   Expected: {expected_card}")
                print(f"   Target rank: {target_rank}")
                print(f"   Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': best_match,
                    'distance': best_distance,
                    'strategy': best_strategy,
                    'correct': is_correct,
                    'target_rank': target_rank
                })
            else:
                print(f"\n❌ NO MATCH")
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': None,
                    'distance': float('inf'),
                    'strategy': None,
                    'correct': False,
                    'target_rank': target_rank
                })
        
        # Final results
        print(f"\n{'='*80}")
        print("🎯 FINAL RESULTS WITH CORRECT COORDINATES")
        print(f"{'='*80}")
        
        correct_count = sum(1 for r in results if r['correct'])
        found_count = sum(1 for r in results if r['target_rank'] is not None)
        total_count = len(results)
        
        print(f"✅ Correct detections: {correct_count}/{total_count}")
        print(f"🎯 Target cards found: {found_count}/{total_count}")
        print(f"📊 Success rate: {correct_count/total_count*100:.1f}%")
        print()
        
        for result in results:
            status = "✅" if result['correct'] else "❌"
            found = "🎯" if result['target_rank'] else "❓"
            rank_info = f"(rank {result['target_rank']})" if result['target_rank'] else ""
            print(f"{status} {found} Card {result['position']}: {result['expected']} → {result['detected'] or 'None'} {rank_info}")
            if result['detected']:
                print(f"      Distance {result['distance']:.4f} via {result['strategy']}")
        
        if correct_count == total_count:
            print(f"\n🎉 PERFECT SUCCESS! Arena Tracker method working perfectly!")
            return True
        elif found_count == total_count:
            print(f"\n📈 All targets found - very close to perfect!")
            return True
        else:
            print(f"\n🔍 Making progress - continue refining")
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
    
    return test_correct_coordinates(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)