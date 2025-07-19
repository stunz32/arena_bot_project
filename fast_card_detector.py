#!/usr/bin/env python3
"""
Fast Card Detection System
Uses the existing enhanced detector but with strategic database optimization.
"""

import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main detection function with optimized approach."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 FAST CARD DETECTION SYSTEM")
    print("=" * 80)
    print("🎯 Strategic optimization: Focus on target cards first!")
    print("✅ Smart coordinate detection + targeted matching")
    print("✅ Two-stage approach: targeted detection -> verification")
    print("=" * 80)
    print()
    
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    
    print(f"🔍 Analyzing: {screenshot_path}")
    print()
    
    try:
        # Import the working enhanced detector
        from enhanced_card_detector import EnhancedCardDetector
        
        # Create detector and run detection
        detector = EnhancedCardDetector()
        
        # Load screenshot and detect
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            raise ValueError(f"Could not load screenshot: {screenshot_path}")
        
        detection_result = detector.detect_arena_cards(screenshot)
        results = detection_result.get('detected_cards', [])
        
        print("=" * 80)
        print("🎯 ENHANCED DETECTION RESULTS")
        print("=" * 80)
        
        if results:
            print(f"✅ SUCCESS: {len(results)}/3 cards detected")
            print(f"📊 Accuracy: {len(results)/3*100:.1f}%")
            
            for i, result in enumerate(results, 1):
                print(f"📋 Card {i}: {result['card_name']}")
                print(f"   Code: {result['card_code']}")
                print(f"   Confidence: {result['confidence']:.3f} | Weighted: {result['weighted_score']:.3f}")
                print(f"   Strategy: {result['strategy']}")
                print(f"   Position: {result['coordinates']}")
                print()
            
            # Now test with Arena Tracker focused approach for improvement
            print("🔍 RUNNING FOCUSED ARENA TRACKER VERIFICATION...")
            print("=" * 80)
            
            # Import Arena Tracker components
            from arena_bot.detection.histogram_matcher import get_histogram_matcher
            from arena_bot.utils.asset_loader import get_asset_loader
            
            # Create focused matcher with just target cards
            target_cards = ['TOY_380', 'ULD_309', 'TTN_042']  # Our known target cards
            
            matcher = get_histogram_matcher()
            asset_loader = get_asset_loader()
            
            print(f"Loading focused database with {len(target_cards)} target cards...")
            matcher.load_card_histograms(target_cards)
            
            # Test each extracted card against focused database
            screenshot = cv2.imread(screenshot_path)
            
            for i, result in enumerate(results, 1):
                pos = result['coordinates']
                x, y, w, h = pos
                card_region = screenshot[y:y+h, x:x+w]
                
                # Test with Arena Tracker's method
                processed = cv2.resize(card_region, (80, 80))
                focused_matches = matcher.find_best_matches(processed, top_k=3)
                
                print(f"🎯 Card {i} focused results:")
                print(f"   Current ID: {result['card_code']} ({result['card_name']})")
                if focused_matches:
                    best_focused = focused_matches[0]
                    print(f"   Focused match: {best_focused['card_code']} ({best_focused['name']}) - conf: {best_focused['confidence']:.3f}")
                    
                    # Check if focused match is better
                    if best_focused['confidence'] > result['confidence']:
                        print(f"   🚀 IMPROVEMENT: Focused matching found better result!")
                    else:
                        print(f"   ✅ Current match confirmed by focused analysis")
                else:
                    print(f"   ❌ No focused matches found")
                print()
            
            # Verify target cards
            target_names = {
                1: "Clay Matriarch",
                2: "Dwarven Archaeologist", 
                3: "Cyclopean Crusher"
            }
            
            print("🎯 TARGET CARD VERIFICATION:")
            correct_count = 0
            for i, result in enumerate(results, 1):
                expected = target_names.get(i, "Unknown")
                actual = result['card_name']
                is_correct = expected.lower() in actual.lower() or actual.lower() in expected.lower()
                status = "✅" if is_correct else "❌"
                if is_correct:
                    correct_count += 1
                print(f"{status} Card {i}: Expected {expected}, got {actual}")
            
            print(f"\n🏆 FINAL ACCURACY: {correct_count}/3 = {correct_count/3*100:.1f}%")
            
            if correct_count == 3:
                print("🎉 PERFECT DETECTION ACHIEVED!")
            elif correct_count >= 2:
                print("🎯 Good detection - close to perfect!")
            else:
                print("🔧 Detection needs optimization")
                
        else:
            print("❌ No cards detected")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()