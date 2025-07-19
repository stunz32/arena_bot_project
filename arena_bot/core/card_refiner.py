import cv2
import numpy as np
from typing import Tuple, Optional


class CardRefiner:
    """
    Stage 2 of the two-stage pipeline: Takes coarse ROI from SmartCoordinateDetector
    and finds pixel-perfect card boundaries within that region.
    """
    
    @staticmethod
    def refine_card_region(roi_image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Refines a coarse ROI to find pixel-perfect card boundaries using Color-Guided Adaptive Crop.
        Finds the mana gem anchor to calculate precise crop line, then applies contour detection.
        
        Args:
            roi_image: Coarse card cutout from Stage 1 detection
            
        Returns:
            Tuple of (x, y, width, height) for tight card bounding box
            relative to the input ROI image
        """
        if roi_image is None or roi_image.size == 0:
            return (0, 0, 0, 0)
        
        roi_height, roi_width = roi_image.shape[:2]
        
        # PHASE 1: Color-Guided Adaptive Crop - Find Mana Gem Anchor
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # Create color mask to find blue mana gem (blue/white gem in top-left)
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours of blue shapes
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crop_y = int(roi_height * 0.15)  # Default fallback to 15% crop
        
        if blue_contours:
            # Find the largest blue contour (most likely the mana gem)
            largest_blue_contour = max(blue_contours, key=cv2.contourArea)
            
            # Get bounding box of mana gem
            gem_x, gem_y, gem_w, gem_h = cv2.boundingRect(largest_blue_contour)
            
            # Calculate adaptive crop line: halfway through mana gem
            crop_y = gem_y + int(gem_h * 0.5)
            
            # Ensure crop_y is within reasonable bounds
            crop_y = max(int(roi_height * 0.05), min(crop_y, int(roi_height * 0.3)))
        
        # PHASE 2: Create and Apply Intelligent Mask
        # Create completely black mask image
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        
        # Draw solid white rectangle from crop_y to bottom (preserve clean card area)
        cv2.rectangle(mask, (0, crop_y), (roi_width, roi_height), 255, -1)
        
        # Apply mask to original ROI using bitwise_and
        # This blacks out everything above crop_y, preserves clean card below
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        processed_image = cv2.bitwise_and(roi_image, mask_3channel)
        
        # PHASE 3: Simple Contour Detection on Cleaned Image
        # Convert to grayscale
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive threshold to create binary image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours (external only)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours found: return entire ROI
            return (0, 0, roi_width, roi_height)
        
        # Find best card contour optimized for masked images
        best_contour = CardRefiner._find_best_card_contour_masked(contours, processed_image.shape, crop_y)
        
        if best_contour is None:
            # No suitable contour found: return entire ROI
            return (0, 0, roi_width, roi_height)
        
        # Get bounding rectangle of best contour
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Ensure bounds are within ROI
        x = max(0, min(x, roi_width - 1))
        y = max(0, min(y, roi_height - 1))
        w = min(w, roi_width - x)
        h = min(h, roi_height - y)
        
        # Sanity check: If the refined area is less than 30% of the original, it's likely garbage
        coarse_area = roi_height * roi_width
        refined_area = w * h
        
        if refined_area < (coarse_area * 0.3):
            # Return original dimensions instead of tiny garbage region
            return (0, 0, roi_width, roi_height)
        
        return (x, y, w, h)
    
    @staticmethod
    def _find_best_card_contour(contours, image_shape) -> Optional[np.ndarray]:
        """
        Finds the contour most likely to represent a Hearthstone card.
        
        Args:
            contours: List of detected contours
            image_shape: Shape of the source image (height, width, channels)
            
        Returns:
            Best contour or None if no suitable contour found
        """
        min_area = (image_shape[0] * image_shape[1]) * 0.1  # At least 10% of ROI
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip too small contours
            if area < min_area:
                continue
            
            # Get bounding rectangle to check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            
            if h == 0:  # Avoid division by zero
                continue
                
            aspect_ratio = w / h
            
            # Hearthstone cards have aspect ratio ~0.7-0.8 (width/height)
            if 0.65 <= aspect_ratio <= 0.85:
                # Score based on area (larger is better) and aspect ratio closeness to 0.75
                aspect_score = 1.0 - abs(aspect_ratio - 0.75) / 0.1  # Normalized to [0,1]
                area_score = area / (image_shape[0] * image_shape[1])  # Normalized area
                
                total_score = area_score * 0.7 + aspect_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_contour = contour
        
        return best_contour
    
    @staticmethod
    def _find_best_card_contour_masked(contours, image_shape, crop_y) -> Optional[np.ndarray]:
        """
        Finds the contour most likely to represent a Hearthstone card in a masked image.
        Optimized for images where the top portion has been masked out.
        
        Args:
            contours: List of detected contours
            image_shape: Shape of the source image (height, width, channels)
            crop_y: Y coordinate where masking was applied
            
        Returns:
            Best contour or None if no suitable contour found
        """
        # Calculate available area (below crop line)
        available_height = image_shape[0] - crop_y
        available_area = image_shape[1] * available_height
        min_area = available_area * 0.05  # At least 5% of available area
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip too small contours
            if area < min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip contours that are entirely above crop line
            if y + h <= crop_y:
                continue
            
            if h == 0:  # Avoid division by zero
                continue
                
            aspect_ratio = w / h
            
            # For masked images, be more flexible with aspect ratios
            # Cards may appear more square when top is cropped
            if 0.5 <= aspect_ratio <= 1.2:
                # Score based on area and position (prefer larger, lower contours)
                area_score = area / available_area  # Normalized to available area
                position_score = (y - crop_y) / available_height  # Prefer contours below crop
                aspect_score = 1.0 - abs(aspect_ratio - 0.75) / 0.25  # Still prefer ~0.75 ratio
                
                # Weighted combination favoring area and position
                total_score = area_score * 0.6 + position_score * 0.3 + aspect_score * 0.1
                
                if total_score > best_score:
                    best_score = total_score
                    best_contour = contour
        
        return best_contour
    
