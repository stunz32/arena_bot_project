#!/usr/bin/env python3
"""
Verify that target cards exist in database and see what they look like.
Also create a systematic coordinate finder.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_target_cards():
    """Verify target cards exist and show them."""
    print("🔍 VERIFYING TARGET CARDS IN DATABASE")
    print("=" * 80)
    
    try:
        from arena_bot.utils.asset_loader import get_asset_loader
        
        asset_loader = get_asset_loader()
        target_cards = ["TOY_380", "ULD_309", "TTN_042"]
        
        for card_code in target_cards:
            print(f"\n📋 Checking {card_code}:")
            
            # Load the reference image
            card_image = asset_loader.load_card_image(card_code, premium=False)
            if card_image is not None:
                print(f"   ✅ Found: {card_image.shape}")
                # Save for inspection
                cv2.imwrite(f"reference_{card_code}.png", card_image)
                print(f"   💾 Saved: reference_{card_code}.png")
            else:
                print(f"   ❌ Not found")
                
            # Also try premium version
            premium_image = asset_loader.load_card_image(card_code, premium=True)
            if premium_image is not None:
                print(f"   ✅ Premium found: {premium_image.shape}")
                cv2.imwrite(f"reference_{card_code}_premium.png", premium_image)
                print(f"   💾 Saved: reference_{card_code}_premium.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_coordinate_grid(screenshot_path: str):
    """Create a systematic coordinate finder using visual inspection."""
    print(f"\n🎯 COORDINATE GRID ANALYSIS")
    print("=" * 80)
    
    try:
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Based on visual inspection, let's try different coordinate systems
        # The cards appear to be in positions around these approximate areas
        
        coordinate_sets = [
            "Manual estimate v1",
            [(195, 85, 200, 290), (445, 85, 200, 290), (695, 85, 200, 290)],
            
            "Manual estimate v2", 
            [(200, 90, 180, 270), (450, 90, 180, 270), (700, 90, 180, 270)],
            
            "Manual estimate v3",
            [(210, 95, 160, 250), (460, 95, 160, 250), (710, 95, 160, 250)],
        ]
        
        for i in range(0, len(coordinate_sets), 2):
            set_name = coordinate_sets[i]
            coords = coordinate_sets[i+1]
            
            print(f"\n📊 Testing {set_name}:")
            
            for j, (x, y, w, h) in enumerate(coords):
                card_image = screenshot[y:y+h, x:x+w]
                
                if card_image.size > 0:
                    filename = f"coord_test_{set_name.replace(' ', '_').lower()}_{j+1}.png"
                    cv2.imwrite(filename, card_image)
                    print(f"   Card {j+1}: {x},{y},{w},{h} → {filename}")
                else:
                    print(f"   Card {j+1}: {x},{y},{w},{h} → EMPTY")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinate analysis failed: {e}")
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    # First verify cards exist
    verify_target_cards()
    
    # Then test coordinates
    create_coordinate_grid("screenshot.png")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)