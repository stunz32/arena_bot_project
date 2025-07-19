#!/usr/bin/env python3
"""
Find the exact Hearthstone window position in the new screenshot.
"""

import cv2
import numpy as np

def find_hearthstone_window(screenshot_path: str):
    """Analyze the screenshot to find Hearthstone window boundaries."""
    print("🔍 FINDING HEARTHSTONE WINDOW")
    print("=" * 60)
    
    screenshot = cv2.imread(screenshot_path)
    height, width = screenshot.shape[:2]
    print(f"Full screenshot: {width}x{height}")
    
    # Looking at the screenshot visually, I can see the Hearthstone interface
    # appears to be in the center-right area. Let me systematically find it.
    
    # The screenshot shows the arena draft interface with the wooden/red border
    # I can see the "Draft a new card for your deck (6/30)" text and 3 cards below
    
    # From visual inspection of the screenshot, the Hearthstone interface appears to be roughly:
    # - Starting around x=400-500 and extending to around x=1400-1500
    # - Y position around 20-50 for the top, extending down to around 600-700
    
    # Let me test a grid of positions to find the actual cards
    print("\n📍 Testing grid positions to locate cards...")
    
    # Based on the visual layout, the 3 cards appear to be positioned roughly here:
    # (looking at the red-bordered draft area in the screenshot)
    
    test_regions = [
        # Format: (description, x, y, w, h)
        ("Left area test", 500, 80, 150, 250),
        ("Center-left test", 600, 80, 150, 250),
        ("Center test", 700, 80, 150, 250),
        ("Center-right test", 800, 80, 150, 250),
        ("Right test", 900, 80, 150, 250),
        
        # Try different Y positions too
        ("Left Y+20", 500, 100, 150, 250),
        ("Center Y+20", 700, 100, 150, 250),
        ("Right Y+20", 900, 100, 150, 250),
        
        # Try wider regions
        ("Wide left", 450, 80, 200, 300),
        ("Wide center", 650, 80, 200, 300),
        ("Wide right", 850, 80, 200, 300),
    ]
    
    for desc, x, y, w, h in test_regions:
        # Check bounds
        if x + w > width or y + h > height:
            print(f"❌ {desc}: Out of bounds")
            continue
        
        # Extract region
        region = screenshot[y:y+h, x:x+w]
        filename = f"window_finder_{desc.replace(' ', '_').lower()}.png"
        cv2.imwrite(filename, region)
        print(f"📍 {desc}: ({x}, {y}, {w}, {h}) → {filename}")
    
    # Also try to detect the red border area which should contain the cards
    print("\n🎨 Looking for red border area...")
    
    # Convert to HSV to look for red colors
    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    
    # Define range for red colors (Hearthstone UI has distinctive red borders)
    # Red in HSV can be in two ranges due to hue wraparound
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red areas
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Save the mask to see red areas
    cv2.imwrite("red_areas_mask.png", red_mask)
    print("🔴 Saved red areas mask: red_areas_mask.png")
    
    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for large rectangular contours that might be the draft area
    print("\n📦 Large red rectangular areas found:")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 5000:  # Filter for reasonably large areas
            x, y, w, h = cv2.boundingRect(contour)
            print(f"   Area {i+1}: ({x}, {y}, {w}, {h}) - Area: {area:.0f}")
            
            # Extract this region for inspection
            if x + w <= width and y + h <= height:
                region = screenshot[y:y+h, x:x+w]
                filename = f"red_area_{i+1}.png"
                cv2.imwrite(filename, region)
                print(f"      Saved: {filename}")
    
    print("\n💡 Manual inspection suggestions:")
    print("1. Look at the extracted regions to see which ones contain cards")
    print("2. Check red_areas_mask.png to see detected red UI elements")
    print("3. The correct coordinates should show the 3 arena draft cards clearly")

def main():
    find_hearthstone_window("screenshot.png")

if __name__ == "__main__":
    main()