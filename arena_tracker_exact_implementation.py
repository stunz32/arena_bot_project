#!/usr/bin/env python3
"""
Implementation of Arena Tracker's exact card detection method.
Based on analysis of their C++ source code.
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
    """
    Compute histogram exactly like Arena Tracker does.
    Based on their getHist function:
    - Convert to HSV
    - Use 50 bins for hue, 60 for saturation
    - Only use H and S channels (ignore V)
    - Normalize with MINMAX
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Arena Tracker parameters
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    
    # Range parameters (exactly like Arena Tracker)
    h_ranges = [0, 180]  # Hue: 0-179
    s_ranges = [0, 256]  # Saturation: 0-255
    ranges = h_ranges + s_ranges
    
    # Use only H and S channels (channels 0 and 1)
    channels = [0, 1]
    
    # Calculate histogram
    hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
    
    # Normalize exactly like Arena Tracker: NORM_MINMAX to range 0-1
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist

def extract_arena_tracker_card_region(card_image: np.ndarray, is_premium: bool = False) -> np.ndarray:
    """
    Extract the exact 80x80 region that Arena Tracker uses for card comparison.
    Based on their code:
    - Normal cards: cv::Rect(60,71,80,80)
    - Premium cards: cv::Rect(57,71,80,80)
    """
    if is_premium:
        # Premium cards: start at (57,71), extract 80x80
        x, y, w, h = 57, 71, 80, 80
    else:
        # Normal cards: start at (60,71), extract 80x80
        x, y, w, h = 60, 71, 80, 80
    
    # Check bounds
    if (card_image.shape[1] < x + w) or (card_image.shape[0] < y + h):
        print(f"   ⚠️  Card too small for extraction: {card_image.shape} vs required {x+w}x{y+h}")
        return None
    
    # Extract the region
    region = card_image[y:y+h, x:x+w]
    return region

def test_arena_tracker_exact_method(screenshot_path: str, target_cards: list):
    """Test using Arena Tracker's exact histogram calculation method."""
    print("🎯 ARENA TRACKER EXACT METHOD TEST")
    print("=" * 80)
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        
        asset_loader = get_asset_loader()
        
        # Load target card reference images to see what they look like
        print("🔍 Loading target card reference images:")
        reference_hists = {}
        
        for card_code in target_cards:
            card_image = asset_loader.load_card_image(card_code, premium=False)
            if card_image is not None:
                print(f"   ✅ {card_code}: {card_image.shape}")
                
                # Extract Arena Tracker region
                at_region = extract_arena_tracker_card_region(card_image, is_premium=False)
                if at_region is not None:
                    # Save for inspection
                    cv2.imwrite(f"at_reference_{card_code}_region.png", at_region)
                    
                    # Compute Arena Tracker histogram
                    hist = compute_arena_tracker_histogram(at_region)
                    reference_hists[card_code] = hist
                    print(f"   📊 Histogram computed: {hist.shape}")
                else:
                    print(f"   ❌ Failed to extract region")
            else:
                print(f"   ❌ {card_code}: Not found")
        
        # Load ALL cards with Arena Tracker method for comparison
        print(f"\n📚 Loading card database with Arena Tracker method...")
        available_cards = asset_loader.get_available_cards()
        card_hists = {}
        
        for i, card_code in enumerate(available_cards[:2000]):  # Limit for speed
            # Load both normal and premium versions
            for is_premium in [False, True]:
                card_image = asset_loader.load_card_image(card_code, premium=is_premium)
                if card_image is not None:
                    at_region = extract_arena_tracker_card_region(card_image, is_premium=is_premium)
                    if at_region is not None:
                        hist = compute_arena_tracker_histogram(at_region)
                        hist_key = f"{card_code}{'_premium' if is_premium else ''}"
                        card_hists[hist_key] = hist
            
            if i % 500 == 0:
                print(f"   Processed {i}/{2000} cards...")
        
        print(f"✅ Loaded {len(card_hists)} card histograms")
        
        # Now test with precise arena draft coordinates
        # Based on visual analysis, try these coordinates for the 3 cards
        draft_coords = [
            (186, 85, 218, 295),   # Left card
            (438, 85, 218, 295),   # Middle card  
            (690, 85, 218, 295),   # Right card
        ]
        
        results = []
        
        for i, (x, y, w, h) in enumerate(draft_coords):
            if i >= len(target_cards):
                break
                
            expected_card = target_cards[i]
            print(f"\n{'='*60}")
            print(f"🔍 ARENA TRACKER METHOD - Card {i+1}")
            print(f"Expected: {expected_card}")
            print(f"Region: ({x}, {y}, {w}, {h})")
            print(f"{'='*60}")
            
            # Extract card from screenshot
            screen_card = screenshot[y:y+h, x:x+w]
            
            if screen_card.size == 0:
                print(f"❌ Empty screen region")
                continue
            
            # Save extracted card
            debug_path = f"at_method_card_{i+1}.png"
            cv2.imwrite(debug_path, screen_card)
            print(f"💾 Saved: {debug_path}")
            
            # Try extracting Arena Tracker region from this screen capture
            # We need to find the right sub-region within the screen capture that corresponds 
            # to the 80x80 area that Arena Tracker would use
            
            # For arena draft cards, the useful region is likely in the center/upper portion
            strategies = [
                ("Full extracted", screen_card),
                ("Upper 60%", screen_card[0:int(h*0.6), :]),
                ("Center 80x80", screen_card[int(h*0.3):int(h*0.3)+80, int(w*0.5)-40:int(w*0.5)+40] if h >= 110 and w >= 80 else None),
                ("Art region", screen_card[20:100, 20:100] if h >= 100 and w >= 100 else None),
            ]
            
            best_match = None
            best_distance = float('inf')
            best_strategy = None
            target_found = False
            
            for strategy_name, processed_region in strategies:
                if processed_region is None or processed_region.size == 0:
                    print(f"\n📊 {strategy_name}: ❌ Invalid region")
                    continue
                
                print(f"\n📊 Testing: {strategy_name} ({processed_region.shape})")
                
                # Save processed region
                processed_path = f"at_method_card_{i+1}_{strategy_name.lower().replace(' ', '_')}.png"
                cv2.imwrite(processed_path, processed_region)
                print(f"   💾 {processed_path}")
                
                # Resize to 80x80 if needed (Arena Tracker uses 80x80)
                if processed_region.shape[:2] != (80, 80):
                    resized = cv2.resize(processed_region, (80, 80), interpolation=cv2.INTER_AREA)
                else:
                    resized = processed_region
                
                # Compute histogram using Arena Tracker method
                screen_hist = compute_arena_tracker_histogram(resized)
                
                # Compare with all card histograms using Bhattacharyya distance
                matches = []
                for card_key, card_hist in card_hists.items():
                    # Use Arena Tracker's method: compareHist with parameter 3 (Bhattacharyya)
                    distance = cv2.compareHist(screen_hist, card_hist, cv2.HISTCMP_BHATTACHARYYA)
                    matches.append((distance, card_key))
                
                # Sort by distance (lower is better)
                matches.sort(key=lambda x: x[0])
                
                # Show top 10 matches
                print(f"   📋 Top 10 matches:")
                for j, (distance, card_key) in enumerate(matches[:10]):
                    base_code = card_key.replace('_premium', '')
                    is_target = base_code.startswith(expected_card)
                    marker = "🎯" if is_target else "  "
                    print(f"      {j+1:2d}. {marker} {card_key:20s} (dist: {distance:.4f})")
                    
                    if is_target and not target_found:
                        target_found = True
                        print(f"   ✅ TARGET FOUND at rank {j+1}!")
                        
                        if distance < best_distance:
                            best_match = card_key
                            best_distance = distance
                            best_strategy = strategy_name
                
                # Also check if top match is better
                if matches and matches[0][0] < best_distance:
                    top_distance, top_card = matches[0]
                    # Only update if it's reasonable (distance < 0.5)
                    if top_distance < 0.5:
                        best_match = top_card
                        best_distance = top_distance
                        best_strategy = strategy_name
            
            # Record result
            if best_match:
                base_code = best_match.replace('_premium', '')
                is_correct = base_code.startswith(expected_card)
                
                print(f"\n🏆 BEST RESULT:")
                print(f"   Strategy: {best_strategy}")
                print(f"   Detected: {best_match}")
                print(f"   Distance: {best_distance:.4f}")
                print(f"   Expected: {expected_card}")
                print(f"   Target found: {'✅ Yes' if target_found else '❌ No'}")
                print(f"   Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': best_match,
                    'distance': best_distance,
                    'strategy': best_strategy,
                    'correct': is_correct,
                    'target_found': target_found
                })
            else:
                print(f"\n❌ NO MATCH FOUND")
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': None,
                    'distance': float('inf'),
                    'strategy': None,
                    'correct': False,
                    'target_found': target_found
                })
        
        # Final summary
        print(f"\n{'='*80}")
        print("🎯 ARENA TRACKER METHOD RESULTS")
        print(f"{'='*80}")
        
        correct_count = sum(1 for r in results if r['correct'])
        found_count = sum(1 for r in results if r['target_found'])
        total_count = len(results)
        
        print(f"✅ Correct detections: {correct_count}/{total_count}")
        print(f"🎯 Target cards found: {found_count}/{total_count}")
        print(f"📊 Accuracy: {correct_count/total_count*100:.1f}%")
        print()
        
        for result in results:
            status = "✅" if result['correct'] else "❌"
            found = "🎯" if result['target_found'] else "❓"
            print(f"{status} {found} Card {result['position']}: {result['expected']} → {result['detected'] or 'None'}")
            if result['detected']:
                print(f"      Distance: {result['distance']:.4f} via {result['strategy']}")
        
        if correct_count == total_count:
            print(f"\n🎉 PERFECT SUCCESS - Arena Tracker method works!")
        elif found_count == total_count:
            print(f"\n📈 All targets found - need better region extraction")
        elif correct_count > 0:
            print(f"\n📈 Partial success - method is working")
        else:
            print(f"\n⚠️  Need further investigation")
        
        return correct_count > 0
        
    except Exception as e:
        print(f"❌ Arena Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return test_arena_tracker_exact_method(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)