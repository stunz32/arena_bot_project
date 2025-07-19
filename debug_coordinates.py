#!/usr/bin/env python3
"""Debug coordinate extraction for your specific screenshot"""

import cv2
import numpy as np

def debug_card_coordinates():
    """Debug card coordinate extraction"""
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    
    # Load screenshot
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"❌ Could not load screenshot: {screenshot_path}")
        return
    
    print(f"✅ Screenshot loaded: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    # Based on your screenshot, the cards appear to be around these areas:
    # Looking at the image, the cards seem to be in the center area
    # Your resolution is 3408x1250
    
    # Try coordinates based on visual inspection of your screenshot
    card_regions = [
        (370, 70, 380, 320),   # Left card - adjusted
        (815, 70, 380, 320),   # Middle card - adjusted  
        (1260, 70, 380, 320),  # Right card - adjusted
    ]
    
    print("Testing card regions:")
    for i, (x, y, w, h) in enumerate(card_regions):
        print(f"  Card {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # Check if coordinates are valid
        if (y + h <= screenshot.shape[0] and x + w <= screenshot.shape[1] and 
            x >= 0 and y >= 0):
            
            card_region = screenshot[y:y+h, x:x+w]
            
            # Save debug image to see what we're extracting
            debug_path = f"/home/marcco/arena_bot_project/debug_card_{i+1}.png"
            cv2.imwrite(debug_path, card_region)
            print(f"    ✅ Saved debug image: {debug_path}")
            print(f"    📏 Region size: {card_region.shape[1]}x{card_region.shape[0]}")
            
        else:
            print(f"    ❌ Invalid coordinates for card {i+1}")
            print(f"       Screenshot: {screenshot.shape[1]}x{screenshot.shape[0]}")
            print(f"       Trying: x={x} y={y} w={w} h={h}")
            print(f"       Max x+w: {x+w}, Max y+h: {y+h}")

if __name__ == "__main__":
    debug_card_coordinates()