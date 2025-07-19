#!/usr/bin/env python3
"""
Systematic coordinate finder to locate the exact arena draft card positions.
"""

import os
import cv2
import numpy as np

def find_precise_card_coordinates(screenshot_path: str):
    """Find precise coordinates by systematic testing."""
    print("🔍 PRECISE COORDINATE FINDER")
    print("=" * 60)
    
    screenshot = cv2.imread(screenshot_path)
    height, width = screenshot.shape[:2]
    print(f"Screenshot: {width}x{height}")
    
    # Based on visual inspection of the screenshot, the arena draft area appears to be:
    # - Within the wooden/red-bordered interface
    # - Cards are roughly 200 pixels wide each
    # - Positioned horizontally across the screen
    # - Y position around 80-400 range
    
    # Let me test a grid of positions around where I think the cards are
    base_y = 85  # Starting Y position
    card_width = 200
    card_height = 300
    
    # Test different starting X positions for the 3 cards
    x_positions = [
        [180, 430, 680],  # Option 1
        [190, 440, 690],  # Option 2  
        [200, 450, 700],  # Option 3
        [210, 460, 710],  # Option 4
    ]
    
    for opt_idx, x_pos in enumerate(x_positions):
        print(f"\n📍 Testing option {opt_idx + 1}: X positions {x_pos}")
        
        for card_idx, x in enumerate(x_pos):
            # Test multiple Y positions and sizes
            test_configs = [
                (x, base_y, card_width, card_height),
                (x, base_y + 5, card_width, card_height - 10),
                (x, base_y + 10, card_width - 20, card_height - 20),
                (x + 10, base_y, card_width - 20, card_height),
            ]
            
            for config_idx, (test_x, test_y, test_w, test_h) in enumerate(test_configs):
                if test_x + test_w > width or test_y + test_h > height:
                    continue
                
                extracted = screenshot[test_y:test_y+test_h, test_x:test_x+test_w]
                
                if extracted.size > 0:
                    filename = f"coord_finder_opt{opt_idx+1}_card{card_idx+1}_config{config_idx+1}.png"
                    cv2.imwrite(filename, extracted)
                    print(f"   Card {card_idx+1} Config {config_idx+1}: ({test_x},{test_y},{test_w},{test_h}) → {filename}")
    
    # Also try to find cards by looking for the specific visual patterns
    # The cards should have distinct visual characteristics
    print(f"\n🎨 Looking for visual patterns...")
    
    # Try to detect regions with high color variance (likely to be card art)
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    # Look for rectangular regions that might be cards
    # Cards should have defined borders and varying content
    
    # Test specific regions based on the screenshot layout
    promising_regions = [
        # Based on visual inspection, cards appear to be approximately here:
        (195, 90, 190, 280),   # Left card
        (445, 90, 190, 280),   # Middle card  
        (695, 90, 190, 280),   # Right card
    ]
    
    print(f"\n🎯 Testing promising regions:")
    for i, (x, y, w, h) in enumerate(promising_regions):
        extracted = screenshot[y:y+h, x:x+w]
        filename = f"promising_card_{i+1}.png"
        cv2.imwrite(filename, extracted)
        print(f"   Card {i+1}: ({x},{y},{w},{h}) → {filename}")
        
        # Also extract smaller sub-regions to see the card art
        if h >= 150 and w >= 100:
            # Try to extract just the card art portion (upper center)
            art_region = extracted[20:120, 20:w-20]
            art_filename = f"promising_card_{i+1}_art.png"
            cv2.imwrite(art_filename, art_region)
            print(f"     Art region: {art_filename}")

def main():
    """Main function."""
    find_precise_card_coordinates("screenshot.png")

if __name__ == "__main__":
    main()