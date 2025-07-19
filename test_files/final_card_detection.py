#!/usr/bin/env python3
"""
Final attempt at accurate card detection with precise coordinates.
Based on visual analysis of screenshot and reference cards.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def final_card_test(screenshot_path: str, target_cards: list):
    """Final test with most accurate coordinates possible."""
    print("🎯 FINAL CARD DETECTION TEST")
    print("=" * 80)
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # After careful visual inspection of the screenshot:
        # The arena draft cards are positioned in a 3x1 grid within the red border
        # Each card appears to be approximately 218 pixels wide and starts at these x positions:
        # Left card: ~186, Middle card: ~438, Right card: ~690
        # Y position: ~82, Height: ~300
        
        # But let me try slightly different positioning based on the partial successes:
        final_coords = [
            (186, 85, 218, 295),   # Left card (TOY_380 - Clay Matriarch)
            (438, 85, 218, 295),   # Middle card (ULD_309 - Dwarven Archaeologist) 
            (690, 85, 218, 295),   # Right card (TTN_042 - Cyclopian Crusher)
        ]
        
        print(f"🎯 Using final coordinates: {len(final_coords)} cards")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        
        # Load database
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
        
        for i, region in enumerate(final_coords):
            if i >= len(target_cards):
                break
                
            expected_card = target_cards[i]
            x, y, w, h = region
            
            print(f"\n{'='*60}")
            print(f"🔍 FINAL TEST - Card {i+1} (Expected: {expected_card})")
            print(f"📍 Region: x={x}, y={y}, w={w}, h={h}")
            print(f"{'='*60}")
            
            # Extract card
            card_image = screenshot[y:y+h, x:x+w]
            
            if card_image.size == 0:
                print(f"❌ Empty region")
                continue
            
            # Save extraction
            debug_path = f"final_card_{i+1}.png"
            cv2.imwrite(debug_path, card_image)
            print(f"💾 Saved: {debug_path}")
            
            # Test multiple processing approaches
            processing_strategies = [
                ("Original", card_image),
                ("Upper 75%", card_image[0:int(h*0.75), :]),
                ("Slight crop", card_image[10:h-10, 10:w-10]),
                ("Art focus", card_image[30:int(h*0.7), 20:w-20]),
            ]
            
            best_match = None
            best_confidence = 0
            best_strategy = None
            found_target = False
            
            for strategy_name, processed in processing_strategies:
                print(f"\n📊 Strategy: {strategy_name}")
                
                if processed.size == 0:
                    print(f"   ❌ Empty")
                    continue
                
                # Save processed version
                processed_path = f"final_card_{i+1}_{strategy_name.lower().replace(' ', '_')}.png"
                cv2.imwrite(processed_path, processed)
                print(f"   💾 {processed_path}")
                
                # Get matches
                hist = histogram_matcher.compute_histogram(processed)
                if hist is not None:
                    candidates = histogram_matcher.find_best_matches(hist, max_candidates=50)
                    
                    # Look for target card
                    target_rank = None
                    for j, candidate in enumerate(candidates):
                        if candidate.card_code.startswith(expected_card):
                            target_rank = j + 1
                            found_target = True
                            print(f"   🎯 FOUND {expected_card} at rank {j+1}! (dist: {candidate.distance:.4f})")
                            
                            if candidate.confidence > best_confidence:
                                best_match = candidate
                                best_confidence = candidate.confidence
                                best_strategy = strategy_name
                            break
                    
                    # Show top 5 anyway
                    print(f"   📋 Top 5:")
                    for j, candidate in enumerate(candidates[:5]):
                        marker = "🎯" if candidate.card_code.startswith(expected_card) else "  "
                        print(f"      {j+1}. {marker} {candidate.card_code} (dist: {candidate.distance:.4f})")
                    
                    # Also consider top match if confidence is good
                    if candidates and not best_match:
                        top = candidates[0]
                        match = histogram_matcher.match_card(processed, confidence_threshold=0.6)
                        if match and match.confidence > best_confidence:
                            best_match = match
                            best_confidence = match.confidence
                            best_strategy = strategy_name
            
            # Record result
            if best_match:
                is_correct = best_match.card_code.startswith(expected_card)
                print(f"\n🏆 BEST RESULT:")
                print(f"   Strategy: {best_strategy}")
                print(f"   Detected: {best_match.card_code}")
                print(f"   Confidence: {best_confidence:.3f}")
                print(f"   Expected: {expected_card}")
                print(f"   Target found: {'✅ Yes' if found_target else '❌ No'}")
                print(f"   Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': best_match.card_code,
                    'confidence': best_confidence,
                    'strategy': best_strategy,
                    'correct': is_correct,
                    'target_found': found_target
                })
            else:
                print(f"\n❌ NO CONFIDENT MATCH")
                print(f"   Target found: {'✅ Yes' if found_target else '❌ No'}")
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': None,
                    'confidence': 0,
                    'strategy': None,
                    'correct': False,
                    'target_found': found_target
                })
        
        # Summary
        print(f"\n{'='*80}")
        print("🎯 FINAL RESULTS")
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
                print(f"      {result['confidence']:.3f} confidence via {result['strategy']}")
        
        success_level = "PERFECT" if correct_count == total_count else "PARTIAL" if correct_count > 0 else "FAILED"
        print(f"\n🏁 Result: {success_level}")
        
        if found_count == total_count and correct_count < total_count:
            print("💡 All target cards found in database - need to improve ranking/confidence")
        elif found_count < total_count:
            print("⚠️  Some target cards not found - check extraction coordinates")
        
        return correct_count > 0
        
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return final_card_test(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)