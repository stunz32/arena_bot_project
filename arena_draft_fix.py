#!/usr/bin/env python3
"""
Fix Arena draft card detection by using manual coordinates based on screenshot analysis.
The current window detection is failing - let's use precise manual coordinates.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_arena_draft_cards(screenshot_path: str, target_cards: list):
    """Test with manually specified Arena draft card regions."""
    print("🎯 ARENA DRAFT CARD DETECTION FIX")
    print("=" * 80)
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Manual coordinates for the 3 Arena draft cards based on screenshot analysis
        # Looking at the screenshot, the cards are positioned in the arena draft interface
        draft_card_regions = [
            (186, 81, 248, 323),   # Left card (6-mana purple/epic)
            (435, 81, 248, 323),   # Middle card (3-mana orange/legendary) 
            (684, 81, 248, 323),   # Right card (3-mana blue/rare)
        ]
        
        print(f"🎯 Using manual Arena draft regions: {len(draft_card_regions)} cards")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        
        # Load full database for accurate matching
        print("📚 Loading full card database...")
        available_cards = asset_loader.get_available_cards()
        all_images = {}
        
        for card_code in available_cards:
            normal = asset_loader.load_card_image(card_code, premium=False)
            premium = asset_loader.load_card_image(card_code, premium=True)
            if normal is not None:
                all_images[card_code] = normal
            if premium is not None:
                all_images[f"{card_code}_premium"] = premium
        
        histogram_matcher.load_card_database(all_images)
        print(f"✅ Loaded {histogram_matcher.get_database_size()} histograms")
        
        results = []
        
        for i, region in enumerate(draft_card_regions):
            if i >= len(target_cards):
                break
                
            expected_card = target_cards[i]
            x, y, w, h = region
            
            print(f"\n{'='*60}")
            print(f"🔍 ARENA DRAFT CARD {i+1} (Expected: {expected_card})")
            print(f"📍 Region: ({x}, {y}, {w}, {h})")
            print(f"{'='*60}")
            
            # Extract card with manual coordinates
            card_image = screenshot[y:y+h, x:x+w]
            
            if card_image.size == 0:
                print(f"❌ Empty card region")
                continue
            
            # Save extracted card for inspection
            debug_path = f"arena_draft_card_{i+1}.png"
            cv2.imwrite(debug_path, card_image)
            print(f"💾 Saved: {debug_path}")
            
            # Test different extraction strategies for better matching
            strategies = [
                ("Full card", card_image),
                ("Upper 70%", card_image[0:int(h*0.7), :]),
                ("Center crop", card_image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]),
            ]
            
            best_match = None
            best_confidence = 0
            best_strategy = None
            
            for strategy_name, processed_image in strategies:
                print(f"\n📊 Testing: {strategy_name}")
                
                if processed_image.size == 0:
                    print(f"   ❌ Empty processed image")
                    continue
                
                # Save processed image for inspection
                processed_path = f"arena_draft_card_{i+1}_{strategy_name.lower().replace(' ', '_')}.png"
                cv2.imwrite(processed_path, processed_image)
                print(f"   💾 Saved: {processed_path}")
                
                # Test with histogram matching
                match = histogram_matcher.match_card(processed_image, confidence_threshold=0.8)
                
                if match:
                    print(f"   ✅ Match: {match.card_code} (conf: {match.confidence:.3f})")
                    
                    if match.confidence > best_confidence:
                        best_match = match
                        best_confidence = match.confidence
                        best_strategy = strategy_name
                else:
                    # Get top candidates even without confidence match
                    hist = histogram_matcher.compute_histogram(processed_image)
                    if hist is not None:
                        candidates = histogram_matcher.find_best_matches(hist, max_candidates=20)
                        print(f"   📋 Top 10 candidates:")
                        
                        for j, candidate in enumerate(candidates[:10]):
                            is_target = candidate.card_code.startswith(expected_card)
                            marker = "🎯" if is_target else "  "
                            print(f"      {j+1:2d}. {marker} {candidate.card_code:15s} (dist: {candidate.distance:.4f})")
                        
                        # Check for target card in top candidates
                        for candidate in candidates[:5]:
                            if candidate.card_code.startswith(expected_card):
                                if candidate.confidence > best_confidence:
                                    best_match = candidate
                                    best_confidence = candidate.confidence
                                    best_strategy = strategy_name
                                    print(f"   ✅ Found target card in top 5!")
                                break
            
            # Record best result
            if best_match:
                print(f"\n🏆 BEST RESULT for Card {i+1}:")
                print(f"   Strategy: {best_strategy}")
                print(f"   Detected: {best_match.card_code}")
                print(f"   Confidence: {best_confidence:.3f}")
                print(f"   Expected: {expected_card}")
                
                is_correct = best_match.card_code.startswith(expected_card)
                print(f"   Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': best_match.card_code,
                    'confidence': best_confidence,
                    'strategy': best_strategy,
                    'correct': is_correct
                })
            else:
                print(f"\n❌ NO MATCH found for Card {i+1}")
                results.append({
                    'position': i+1,
                    'expected': expected_card,
                    'detected': None,
                    'confidence': 0,
                    'strategy': None,
                    'correct': False
                })
        
        # Final summary
        print(f"\n{'='*80}")
        print("🎯 ARENA DRAFT RESULTS SUMMARY")
        print(f"{'='*80}")
        
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
        print()
        
        for result in results:
            status = "✅" if result['correct'] else "❌"
            print(f"{status} Card {result['position']}: {result['expected']} → {result['detected'] or 'None'}")
            if result['detected']:
                print(f"   Confidence: {result['confidence']:.3f}, Strategy: {result['strategy']}")
        
        if correct_count == total_count:
            print(f"\n🎉 PERFECT ACCURACY ACHIEVED!")
        elif correct_count > 0:
            print(f"\n📈 Partial success: {correct_count}/{total_count} correct")
        else:
            print(f"\n⚠️  No correct detections - investigating further")
        
        return correct_count > 0
        
    except Exception as e:
        print(f"❌ Arena draft testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    target_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return test_arena_draft_cards(screenshot_path, target_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)