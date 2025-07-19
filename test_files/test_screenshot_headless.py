#!/usr/bin/env python3
"""
Headless screenshot testing that bypasses Qt dependencies.
Tests card recognition directly without screen capture components.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_screenshot_detection(screenshot_path: str):
    """Test card detection with a screenshot file, bypassing Qt components."""
    print(f"🖼️  Testing screenshot: {screenshot_path}")
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    try:
        # Load the screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print("❌ Failed to load screenshot")
            return False
        
        height, width = screenshot.shape[:2]
        print(f"✅ Screenshot loaded: {width}x{height}")
        
        # Initialize components manually (bypassing card_recognizer to avoid Qt)
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        from arena_bot.detection.template_matcher import get_template_matcher
        
        print("🔄 Initializing detection components...")
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        template_matcher = get_template_matcher()
        
        # Initialize template matcher with templates
        if not template_matcher.initialize():
            print("⚠️  Warning: Template matcher initialization failed")
        else:
            mana_count, rarity_count = template_matcher.get_template_counts()
            print(f"✅ Template matcher loaded: {mana_count} mana + {rarity_count} rarity templates")
        
        # Load more cards for better matching (500 should give good coverage)
        print("📚 Loading card database (first 500 cards for testing)...")
        available_cards = asset_loader.get_available_cards()
        test_cards = available_cards[:500]
        
        card_images = {}
        for card_code in test_cards:
            image = asset_loader.load_card_image(card_code)
            if image is not None:
                card_images[card_code] = image
        
        histogram_matcher.load_card_database(card_images)
        print(f"✅ Loaded {histogram_matcher.get_database_size()} card histograms")
        
        # Define card regions for ultrawide (3440x1440) or standard resolutions
        if width > 3000:  # Ultrawide
            # Positions for 3440x1440 - cards are more centered
            card_regions = [
                (int(width * 0.25 - 100), int(height * 0.35), 200, 280),   # Left card
                (int(width * 0.50 - 100), int(height * 0.35), 200, 280),   # Middle card  
                (int(width * 0.75 - 100), int(height * 0.35), 200, 280)    # Right card
            ]
        else:  # Standard resolution
            scale_x = width / 1920.0
            scale_y = height / 1080.0
            card_regions = [
                (int(480 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y)),   # Left card
                (int(860 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y)),   # Middle card  
                (int(1240 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y))   # Right card
            ]
        
        print(f"🔍 Testing card detection in 3 regions...")
        results = []
        
        for i, (x, y, w, h) in enumerate(card_regions):
            print(f"  Testing region {i+1}: ({x}, {y}, {w}, {h})")
            
            # Extract card region with bounds checking
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w > 50 and h > 50:  # Ensure reasonable size
                card_image = screenshot[y:y+h, x:x+w]
                
                # Save extracted region for debugging
                debug_path = f"card_region_{i+1}.png"
                cv2.imwrite(debug_path, card_image)
                print(f"    💾 Saved region to {debug_path}")
                
                # Test histogram matching with lower threshold
                match = histogram_matcher.match_card(card_image, confidence_threshold=0.8)  # Lower threshold
                
                # Also get the best matches even if below threshold for debugging
                hist = histogram_matcher.compute_histogram(card_image)
                if hist is not None:
                    best_matches = histogram_matcher.find_best_matches(hist, max_candidates=3)
                    if best_matches:
                        second_match = best_matches[1].card_code if len(best_matches) > 1 else 'N/A'
                        second_dist = best_matches[1].distance if len(best_matches) > 1 else 0.0
                        print(f"    🔍 Best matches: {best_matches[0].card_code} ({best_matches[0].distance:.3f}), "
                              f"{second_match} ({second_dist:.3f})")
                
                if match:
                    print(f"    ✅ Card {i+1}: {match.card_code} (confidence: {match.confidence:.3f})")
                    
                    # Test template matching on specific subregions of the card
                    # Mana cost is typically in the top-left corner
                    h, w = card_image.shape[:2]
                    mana_region = card_image[0:int(h*0.3), 0:int(w*0.3)]  # Top-left 30%
                    rarity_region = card_image[int(h*0.7):h, int(w*0.4):int(w*0.6)]  # Bottom-center
                    
                    # Save subregions for debugging
                    cv2.imwrite(f"mana_region_{i+1}.png", mana_region)
                    cv2.imwrite(f"rarity_region_{i+1}.png", rarity_region)
                    
                    mana_cost = template_matcher.detect_mana_cost(mana_region)
                    rarity = template_matcher.detect_rarity(rarity_region)
                    
                    print(f"    📊 Mana: {mana_cost}, Rarity: {rarity}")
                    
                    results.append({
                        'position': i+1,
                        'card_code': match.card_code,
                        'confidence': match.confidence,
                        'mana_cost': mana_cost,
                        'rarity': rarity,
                        'is_premium': match.is_premium
                    })
                else:
                    print(f"    ❌ Card {i+1}: No confident match found")
            else:
                print(f"    ⚠️  Card {i+1}: Region too small or out of bounds")
        
        print(f"\n📊 Detection Summary:")
        print(f"   Screenshot: {width}x{height}")
        print(f"   Database: {histogram_matcher.get_database_size()} cards")
        print(f"   Detected: {len(results)}/3 cards")
        
        if results:
            print("\n🎯 Detection Results:")
            for result in results:
                print(f"   Card {result['position']}: {result['card_code']} "
                      f"(conf: {result['confidence']:.3f}, mana: {result['mana_cost']}, "
                      f"rarity: {result['rarity']})")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main testing function."""
    print("🎮 Arena Bot - Headless Screenshot Testing")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    # Check for screenshot argument or find screenshot.png
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
    else:
        screenshot_path = "screenshot.png"
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        print("\n📝 Usage:")
        print("   python3 test_screenshot_headless.py screenshot.png")
        print("   python3 test_screenshot_headless.py /path/to/screenshot.png")
        return False
    
    success = test_screenshot_detection(screenshot_path)
    
    if success:
        print("\n🎉 Testing completed successfully!")
        print("💡 Check the saved card_region_*.png files to see the extracted regions")
    else:
        print("\n⚠️  Testing completed with issues")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)