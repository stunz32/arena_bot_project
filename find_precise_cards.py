#!/usr/bin/env python3
"""Find precise card coordinates by analyzing the draft interface"""

import cv2
import numpy as np

def find_card_coordinates():
    """Find the exact card positions in the screenshot"""
    
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    screenshot = cv2.imread(screenshot_path)
    
    print(f"Screenshot size: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    # Based on visual inspection of your screenshot:
    # The draft cards appear to be in a red/brown wooden frame
    # They look smaller than our current regions
    # Let me try more precise coordinates focusing just on the card art
    
    # Try smaller, more focused regions that just capture the card art
    test_regions = [
        # Left card - more centered on actual card
        (410, 120, 300, 250),   
        # Middle card - more centered
        (855, 120, 300, 250),   
        # Right card - more centered  
        (1300, 120, 300, 250),  
    ]
    
    print("Testing smaller, more focused card regions:")
    for i, (x, y, w, h) in enumerate(test_regions):
        print(f"  Card {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # Check bounds
        if (y + h <= screenshot.shape[0] and x + w <= screenshot.shape[1] and 
            x >= 0 and y >= 0):
            
            card_region = screenshot[y:y+h, x:x+w]
            
            # Save debug image
            debug_path = f"/home/marcco/arena_bot_project/focused_card_{i+1}.png"
            cv2.imwrite(debug_path, card_region)
            print(f"    ✅ Saved: {debug_path}")
            print(f"    📏 Size: {card_region.shape[1]}x{card_region.shape[0]}")
            
        else:
            print(f"    ❌ Invalid coordinates")

if __name__ == "__main__":
    find_card_coordinates()