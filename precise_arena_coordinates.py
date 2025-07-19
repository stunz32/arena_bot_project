#!/usr/bin/env python3
"""
Precise Arena draft card coordinate detection.
Fine-tune coordinates based on visual analysis of the screenshot.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_precise_coordinates(screenshot_path: str, target_cards: list):
    """Test with precisely measured Arena draft card coordinates."""
    print("🎯 PRECISE ARENA DRAFT COORDINATES TEST")
    print("=" * 80)
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Looking at the screenshot more carefully:
        # - The cards are in a red-bordered area
        # - Each card appears to be roughly 218x300 pixels
        # - Cards are positioned with some spacing between them
        
        # More precise coordinates based on visual inspection:
        precise_regions = [
            (186, 82, 218, 300),   # Left card (purple 6-mana)
            (438, 82, 218, 300),   # Middle card (orange 3-mana) 
            (690, 82, 218, 300),   # Right card (blue 3-mana)
        ]
        
        print(f"🎯 Using precise coordinates: {len(precise_regions)} cards")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        
        # Load database - focus on cards that might match
        print("📚 Loading card database...")
        available_cards = asset_loader.get_available_cards()
        all_images = {}
        
        for card_code in available_cards:
            normal = asset_loader.load_card_image(card_code, premium=False)
            if normal is not None:
                all_images[card_code] = normal
        
        histogram_matcher.load_card_database(all_images)
        print(f"✅ Loaded {histogram_matcher.get_database_size()} histograms")
        
        results = []
        
        for i, region in enumerate(precise_regions):
            if i >= len(target_cards):
                break
                
            expected_card = target_cards[i]
            x, y, w, h = region
            
            print(f"\n{'='*60}")
            print(f"🔍 PRECISE CARD {i+1} (Expected: {expected_card})")
            print(f"📍 Coordinates: x={x}, y={y}, w={w}, h={h}")
            print(f"{'='*60}")
            
            # Extract with precise coordinates
            card_image = screenshot[y:y+h, x:x+w]
            
            if card_image.size == 0:
                print(f"❌ Empty card region")
                continue
            
            # Save for inspection
            debug_path = f"precise_card_{i+1}.png"
            cv2.imwrite(debug_path, card_image)
            print(f"💾 Saved: {debug_path}")
            
            # Test various processing strategies
            strategies = [
                ("Raw extraction", card_image),
                ("Card art only (upper 60%)", card_image[0:int(h*0.6), :]),
                ("No borders (10% crop)", card_image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]),
                ("Center focus", card_image[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]),
            ]
            
            best_match = None
            best_confidence = 0
            best_strategy = None
            target_found = False
            
            for strategy_name, processed_image in strategies:
                print(f"\n📊 Testing: {strategy_name}")
                
                if processed_image.size == 0:
                    print(f"   ❌ Empty processed image")
                    continue
                
                # Save processed version
                processed_path = f"precise_card_{i+1}_{strategy_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}.png"
                cv2.imwrite(processed_path, processed_image)
                print(f"   💾 Saved: {processed_path}")
                
                # Compute histogram and find matches
                hist = histogram_matcher.compute_histogram(processed_image)
                if hist is not None:
                    candidates = histogram_matcher.find_best_matches(hist, max_candidates=50)
                    
                    # Check if target card is in top candidates
                    print(f"   📋 Top 10 matches:")
                    for j, candidate in enumerate(candidates[:10]):
                        is_target = candidate.card_code.startswith(expected_card)
                        marker = "🎯" if is_target else "  "
                        print(f"      {j+1:2d}. {marker} {candidate.card_code:15s} (dist: {candidate.distance:.4f})")
                        
                        if is_target and not target_found:
                            target_found = True
                            print(f"   ✅ FOUND TARGET at rank {j+1}!")
                            
                            if candidate.confidence > best_confidence:
                                best_match = candidate
                                best_confidence = candidate.confidence  
                                best_strategy = strategy_name
                    
                    # Also check the top match regardless
                    if candidates and candidates[0].confidence > best_confidence:
                        top_candidate = candidates[0]
                        if not best_match or top_candidate.confidence > best_confidence:
                            match = histogram_matcher.match_card(processed_image, confidence_threshold=0.5)
                            if match:
                                best_match = match
                                best_confidence = match.confidence
                                best_strategy = strategy_name
            
            # Record result
            if best_match:
                is_correct = best_match.card_code.startswith(expected_card)
                print(f"\n🏆 BEST RESULT for Card {i+1}:")
                print(f"   Strategy: {best_strategy}")
                print(f"   Detected: {best_match.card_code}")
                print(f"   Confidence: {best_confidence:.3f}")
                print(f"   Expected: {expected_card}")
                print(f"   Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': best_match.card_code,
                    'confidence': best_confidence,
                    'strategy': best_strategy,
                    'correct': is_correct,
                    'target_found': target_found
                })
            else:
                print(f"\n❌ NO MATCH found for Card {i+1}")
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': None,
                    'confidence': 0,
                    'strategy': None,
                    'correct': False,
                    'target_found': target_found
                })
        
        # Final summary
        print(f"\n{'='*80}")
        print("🎯 PRECISE COORDINATES RESULTS")
        print(f"{'='*80}")
        
        correct_count = sum(1 for r in results if r['correct'])
        target_found_count = sum(1 for r in results if r['target_found'])
        total_count = len(results)
        
        print(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
        print(f"Target cards found in database: {target_found_count}/{total_count}")
        print()
        
        for result in results:
            status = "✅" if result['correct'] else "❌"
            target_status = "🎯" if result['target_found'] else "❓"
            print(f"{status} {target_status} Card {result['position']}: {result['expected']} → {result['detected'] or 'None'}")
            if result['detected']:
                print(f"      Confidence: {result['confidence']:.3f}, Strategy: {result['strategy']}")
        
        if correct_count == total_count:
            print(f"\n🎉 PERFECT ACCURACY ACHIEVED!")
        elif target_found_count == total_count:
            print(f"\n📈 All target cards found in database - need to improve ranking")
        elif target_found_count > 0:
            print(f"\n🔍 Some target cards found - continue improving extraction")
        else:
            print(f"\n⚠️  No target cards found - may need to check card database")
        
        return correct_count > 0
        
    except Exception as e:
        print(f"❌ Precise coordinate testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return test_precise_coordinates(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)