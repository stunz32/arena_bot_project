#!/usr/bin/env python3

import cv2
import numpy as np
from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector

def debug_color_guided_crop():
    """Debug the Color-Guided Adaptive Crop implementation."""
    
    # Load screenshot
    screenshot_path = "/mnt/d/cursor bots/arena_bot_project/debug_frames/Hearthstone Screenshot 07-11-25 17.33.10.png"
    screenshot = cv2.imread(screenshot_path)
    
    # Get coarse positions
    detector = SmartCoordinateDetector()
    result = detector.detect_cards_automatically(screenshot)
    
    if not result or not result['success']:
        print("Failed to detect cards")
        return
    
    coarse_positions = result['card_positions']
    
    # Debug each card
    for i, (x, y, w, h) in enumerate(coarse_positions):
        print(f"\n=== DEBUGGING CARD {i+1} ===")
        
        # Extract ROI
        roi_image = screenshot[y:y+h, x:x+w]
        roi_height, roi_width = roi_image.shape[:2]
        
        print(f"ROI size: {roi_width}x{roi_height}")
        
        # Save original ROI for inspection
        cv2.imwrite(f"/mnt/d/cursor bots/arena_bot_project/debug_frames/DEBUG_ROI_Card{i+1}.png", roi_image)
        
        # PHASE 1: Color detection
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # Create color mask to find blue mana gem
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Save blue mask for inspection
        cv2.imwrite(f"/mnt/d/cursor bots/arena_bot_project/debug_frames/DEBUG_BLUE_MASK_Card{i+1}.png", blue_mask)
        
        # Find blue contours
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(blue_contours)} blue contours")
        
        crop_y = int(roi_height * 0.15)  # Default fallback
        
        if blue_contours:
            # Find largest blue contour
            largest_blue_contour = max(blue_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_blue_contour)
            
            # Get bounding box
            gem_x, gem_y, gem_w, gem_h = cv2.boundingRect(largest_blue_contour)
            
            print(f"Largest blue contour: area={area}, bbox=({gem_x}, {gem_y}, {gem_w}, {gem_h})")
            
            # Calculate crop line
            crop_y = gem_y + int(gem_h * 0.5)
            crop_y = max(int(roi_height * 0.05), min(crop_y, int(roi_height * 0.3)))
            
            print(f"Calculated crop_y: {crop_y} (gem_y={gem_y}, gem_h={gem_h})")
            
            # Draw debug visualization
            debug_img = roi_image.copy()
            cv2.drawContours(debug_img, [largest_blue_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(debug_img, (gem_x, gem_y), (gem_x + gem_w, gem_y + gem_h), (255, 0, 0), 2)
            cv2.line(debug_img, (0, crop_y), (roi_width, crop_y), (0, 0, 255), 2)
            
            cv2.imwrite(f"/mnt/d/cursor bots/arena_bot_project/debug_frames/DEBUG_MANA_GEM_Card{i+1}.png", debug_img)
        else:
            print(f"No blue contours found, using default crop_y: {crop_y}")
        
        # PHASE 2: Create mask
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        cv2.rectangle(mask, (0, crop_y), (roi_width, roi_height), 255, -1)
        
        # Save mask for inspection
        cv2.imwrite(f"/mnt/d/cursor bots/arena_bot_project/debug_frames/DEBUG_MASK_Card{i+1}.png", mask)
        
        # Apply mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        processed_image = cv2.bitwise_and(roi_image, mask_3channel)
        
        # Save processed image
        cv2.imwrite(f"/mnt/d/cursor bots/arena_bot_project/debug_frames/DEBUG_PROCESSED_Card{i+1}.png", processed_image)
        
        # PHASE 3: Contour detection
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours after processing")
        
        # Save binary image
        cv2.imwrite(f"/mnt/d/cursor bots/arena_bot_project/debug_frames/DEBUG_BINARY_Card{i+1}.png", binary)

if __name__ == "__main__":
    debug_color_guided_crop()