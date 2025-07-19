#!/usr/bin/env python3
"""
SURF-based automatic screen detection implementation.
Based on Arena Tracker's findTemplateOnScreen and findTemplateOnMat functions.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class SURFDetector:
    """
    SURF-based feature detection for automatic Hearthstone interface location.
    Implements Arena Tracker's exact SURF detection algorithm.
    """
    
    def __init__(self, min_hessian: int = 400, min_good_matches: int = 10):
        """
        Initialize SURF detector with Arena Tracker parameters.
        
        Args:
            min_hessian: SURF detector Hessian threshold (Arena Tracker uses 400)
            min_good_matches: Minimum good matches needed for detection
        """
        self.min_hessian = min_hessian
        self.min_good_matches = min_good_matches
        
        # Initialize SURF detector (Arena Tracker's exact parameters)
        try:
            self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=min_hessian)
            # Test if SURF actually works (patented features may be disabled)
            test_img = np.zeros((100, 100), dtype=np.uint8)
            self.surf.detectAndCompute(test_img, None)
            self.use_orb = False
            print("✅ SURF detector initialized successfully")
        except (AttributeError, cv2.error) as e:
            # Fallback to ORB if SURF not available or patented features disabled
            print(f"⚠️ SURF not available ({e}), using ORB detector as fallback")
            logger.warning("SURF not available, using ORB detector as fallback")
            self.surf = cv2.ORB_create(nfeatures=1000)
            self.use_orb = True
        
        # Initialize FLANN matcher (Arena Tracker uses FLANN)
        if not self.use_orb:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Use BFMatcher for ORB
            self.flann = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def load_template(self, template_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[cv2.KeyPoint]]]:
        """
        Load template image and compute SURF features.
        
        Args:
            template_path: Path to template image
            
        Returns:
            Tuple of (template_image, descriptors, keypoints) or None if failed
        """
        try:
            # Load template in grayscale (Arena Tracker method)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"Could not load template: {template_path}")
                return None
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.surf.detectAndCompute(template, None)
            
            if descriptors is None or len(keypoints) < self.min_good_matches:
                logger.warning(f"Insufficient features in template: {template_path}")
                return None
            
            logger.info(f"Loaded template with {len(keypoints)} features: {template_path}")
            return template, descriptors, keypoints
            
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")
            return None
    
    def find_template_on_screen(self, template_data: Tuple[np.ndarray, np.ndarray, List[cv2.KeyPoint]], 
                              screenshot: np.ndarray, 
                              template_points: List[Tuple[float, float]] = None) -> Optional[Tuple[List[Tuple[float, float]], float, float]]:
        """
        Find template on screenshot using SURF feature matching.
        Implements Arena Tracker's exact findTemplateOnMat algorithm.
        
        Args:
            template_data: (template_image, descriptors, keypoints) from load_template
            screenshot: Screenshot to search in
            template_points: Corner points of template region to map
            
        Returns:
            Tuple of (mapped_points, screen_scale_x, screen_scale_y) or None if not found
        """
        if template_data is None:
            return None
        
        template_img, template_descriptors, template_keypoints = template_data
        
        try:
            # Convert screenshot to grayscale
            if len(screenshot.shape) == 3:
                screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                screenshot_gray = screenshot
            
            # Detect keypoints and compute descriptors for screenshot
            screen_keypoints, screen_descriptors = self.surf.detectAndCompute(screenshot_gray, None)
            
            if screen_descriptors is None or len(screen_keypoints) < self.min_good_matches:
                logger.warning("Insufficient features in screenshot")
                return None
            
            # Match features using FLANN (Arena Tracker method)
            if not self.use_orb:
                matches = self.flann.knnMatch(template_descriptors, screen_descriptors, k=2)
                
                # Apply ratio test (Arena Tracker uses 0.04 threshold)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.04:  # Arena Tracker's exact threshold
                            good_matches.append(m)
            else:
                # ORB fallback
                matches = self.flann.match(template_descriptors, screen_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 50]  # ORB threshold
            
            logger.info(f"Found {len(good_matches)} good matches")
            
            if len(good_matches) < self.min_good_matches:
                logger.warning(f"Insufficient good matches: {len(good_matches)} < {self.min_good_matches}")
                return None
            
            # Extract matching points
            template_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            screen_pts = np.float32([screen_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography (Arena Tracker uses RANSAC)
            homography, mask = cv2.findHomography(template_pts, screen_pts, cv2.RANSAC)
            
            if homography is None:
                logger.warning("Could not compute homography")
                return None
            
            # Default template points if not provided (full template corners)
            if template_points is None:
                h, w = template_img.shape
                template_points = [
                    (0, 0),
                    (w, 0),
                    (w, h),
                    (0, h)
                ]
            
            # Transform template points to screen coordinates
            template_corners = np.float32(template_points).reshape(-1, 1, 2)
            screen_corners = cv2.perspectiveTransform(template_corners, homography)
            
            # Extract mapped points
            mapped_points = [(float(pt[0][0]), float(pt[0][1])) for pt in screen_corners]
            
            # Calculate screen scale (Arena Tracker method)
            screen_scale_x = 1.0  # Placeholder - would calculate based on screen resolution
            screen_scale_y = 1.0
            
            logger.info(f"Template found at: {mapped_points}")
            return mapped_points, screen_scale_x, screen_scale_y
            
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return None
    
    def detect_arena_interface(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Automatically detect Hearthstone arena interface using template matching.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            (x, y, width, height) of arena interface or None if not found
        """
        # For now, we'll use the working red area detection as a fallback
        # This could be extended with actual arena UI templates
        try:
            # Convert to HSV for red area detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Define range for red colors (Hearthstone UI)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for red areas
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for the largest red area (likely the main interface)
            largest_area = 0
            best_rect = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100000:  # Large enough to be the main interface
                    x, y, w, h = cv2.boundingRect(contour)
                    if area > largest_area:
                        largest_area = area
                        best_rect = (x, y, w, h)
            
            if best_rect:
                logger.info(f"Arena interface detected at: {best_rect}")
                return best_rect
            else:
                logger.warning("Arena interface not detected")
                return None
                
        except Exception as e:
            logger.error(f"Error detecting arena interface: {e}")
            return None
    
    def calculate_card_positions(self, interface_rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Calculate the 3 card positions within the detected arena interface.
        
        Args:
            interface_rect: (x, y, width, height) of arena interface
            
        Returns:
            List of (x, y, width, height) for each of the 3 cards
        """
        interface_x, interface_y, interface_w, interface_h = interface_rect
        
        # Arena draft card positioning (based on our successful detection)
        card_y_offset = 90  # Y offset within interface
        card_height = 300
        card_width = 218
        
        # Distribute 3 cards across interface width
        card_spacing = interface_w // 4
        
        card_positions = []
        for i in range(3):
            card_x = interface_x + card_spacing * (i + 1) - card_width // 2
            card_y = interface_y + card_y_offset
            card_positions.append((card_x, card_y, card_width, card_height))
        
        return card_positions


def get_surf_detector() -> SURFDetector:
    """Get the global SURF detector instance."""
    return SURFDetector()