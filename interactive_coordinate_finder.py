#!/usr/bin/env python3
"""
Interactive Coordinate Finder
Helps determine exact card positions through iterative refinement
"""

import cv2
import numpy as np

def find_cards_interactively(screenshot_path):
    """Interactive tool to find card coordinates"""
    
    # Load screenshot
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"❌ Could not load {screenshot_path}")
        return
        
    height, width = screenshot.shape[:2]
    print(f"📸 Screenshot: {width}x{height}")
    
    # Starting estimates (rough center area where cards should be)
    estimates = [
        {"name": "Card 1 (Funhouse Mirror)", "x": 700, "y": 200, "w": 200, "h": 280},
        {"name": "Card 2 (Holy Nova)", "x": 950, "y": 200, "w": 200, "h": 280},
        {"name": "Card 3 (Mystified To'cha)", "x": 1200, "y": 200, "w": 200, "h": 280}
    ]
    
    print("\n🔍 INTERACTIVE COORDINATE FINDING")
    print("Testing initial estimates...")
    
    for i, card in enumerate(estimates):
        print(f"\n📋 {card['name']}")
        print(f"   Testing: x={card['x']}, y={card['y']}, w={card['w']}, h={card['h']}")
        
        # Extract test region
        x, y, w, h = card['x'], card['y'], card['w'], card['h']
        
        # Bounds check
        if x + w > width or y + h > height or x < 0 or y < 0:
            print(f"   ❌ Out of bounds!")
            continue
            
        cutout = screenshot[y:y+h, x:x+w]
        test_path = f"debug_frames/INTERACTIVE_TEST_Card{i+1}.png"
        cv2.imwrite(test_path, cutout)
        print(f"   💾 Saved: {test_path}")
    
    print(f"\n📋 REFINEMENT INSTRUCTIONS:")
    print(f"1. Examine the test cutouts above")
    print(f"2. If they show the actual cards, we're done!")
    print(f"3. If not, adjust the coordinates below and re-run")
    print(f"4. Look for:")
    print(f"   - Complete card artwork")
    print(f"   - Card name at bottom")
    print(f"   - Mana cost in top-left")
    print(f"   - Stats in bottom corners (if creature)")
    
    return estimates

if __name__ == "__main__":
    screenshot_path = "debug_frames/Screenshot 2025-07-05 085410.png"
    estimates = find_cards_interactively(screenshot_path)
    
    print(f"\n🎯 CURRENT ESTIMATES:")
    for i, card in enumerate(estimates):
        print(f"   Card {i+1}: ({card['x']}, {card['y']}, {card['w']}, {card['h']})")