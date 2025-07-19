"""
Template matching system for mana cost and rarity detection.

Direct port of Arena Tracker's template matching algorithms using L2 distance.
Includes adaptive grid search for optimal template positioning.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TemplateMatch:
    """Container for template match results."""
    template_id: int
    distance: float
    position: Tuple[int, int]
    confidence: float


class TemplateMatcher:
    """
    Template matching system using Arena Tracker's proven L2 distance method.
    
    Implements adaptive grid search for optimal template positioning.
    """
    
    def __init__(self):
        """Initialize template matcher."""
        self.logger = logging.getLogger(__name__)
        
        # Adjusted thresholds for better detection (original Arena Tracker values were too strict)
        self.MANA_L2_THRESHOLD = 10.0  # Increased from 4.5
        self.RARITY_L2_THRESHOLD = 20.0  # Increased from 9.0
        
        # Template databases
        self.mana_templates: Dict[int, np.ndarray] = {}
        self.rarity_templates: Dict[int, np.ndarray] = {}
        
        self.logger.info("TemplateMatcher initialized with Arena Tracker's parameters")
        self.logger.info(f"Mana threshold: {self.MANA_L2_THRESHOLD}")
        self.logger.info(f"Rarity threshold: {self.RARITY_L2_THRESHOLD}")
    
    def initialize(self) -> bool:
        """
        Initialize template matcher by loading all templates.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            from ..utils.asset_loader import get_asset_loader
            
            asset_loader = get_asset_loader()
            
            # Load mana templates (0-9)
            mana_templates = asset_loader.load_mana_templates()
            if mana_templates:
                self.load_mana_templates(mana_templates)
            else:
                self.logger.warning("No mana templates loaded")
            
            # Load rarity templates (0-3)
            rarity_templates = asset_loader.load_rarity_templates()
            if rarity_templates:
                self.load_rarity_templates(rarity_templates)
            else:
                self.logger.warning("No rarity templates loaded")
            
            template_count = len(self.mana_templates) + len(self.rarity_templates)
            if template_count > 0:
                self.logger.info(f"TemplateMatcher initialized with {len(self.mana_templates)} mana + {len(self.rarity_templates)} rarity templates")
                return True
            else:
                self.logger.error("No templates loaded")
                return False
                
        except Exception as e:
            self.logger.error(f"TemplateMatcher initialization failed: {e}")
            return False
    
    def compute_l2_distance(self, sample: np.ndarray, template: np.ndarray) -> float:
        """
        Compute normalized L2 distance between sample and template.
        
        Direct port of Arena Tracker's getL2Mat() function.
        
        Args:
            sample: Sample image region
            template: Template image
            
        Returns:
            Normalized L2 distance
        """
        try:
            # Ensure both images are the same size
            if sample.shape != template.shape:
                # Resize sample to match template
                sample = cv2.resize(sample, (template.shape[1], template.shape[0]))
            
            # Compute L2 norm (Euclidean distance)
            l2_distance = cv2.norm(sample, template, cv2.NORM_L2)
            
            # Normalize by image size (Arena Tracker's method)
            normalized_distance = l2_distance / (template.shape[0] * template.shape[1])
            
            return normalized_distance
            
        except Exception as e:
            self.logger.error(f"L2 distance computation failed: {e}")
            return float('inf')
    
    def adaptive_grid_search(self, source_region: np.ndarray, template: np.ndarray,
                           search_area: int = 8) -> Tuple[float, int, int]:
        """
        Perform adaptive grid search for optimal template position.
        
        Port of Arena Tracker's getBestN() adaptive search algorithm.
        
        Args:
            source_region: Region to search in
            template: Template to match
            search_area: Maximum search offset
            
        Returns:
            Tuple of (best_distance, best_x, best_y)
        """
        best_distance = float('inf')
        best_x = 0
        best_y = 0
        
        # Initial parameters
        init_jump = 1
        jump = init_jump
        center_x = 0
        center_y = 0
        max_jump_reached = False
        
        # Initial center search
        center_distance = self._search_at_position(source_region, template, center_x, center_y)
        best_distance = center_distance
        center_best = center_distance
        
        # Adaptive grid search loop
        while (jump > 0 and 
               abs(center_x) < init_jump * 16 and 
               abs(center_y) < init_jump * 16):
            
            # Calculate search bounds
            start_x = center_x - jump * 2
            start_y = center_y - jump * 2
            end_x = center_x + jump * 2
            end_y = center_y + jump * 2
            
            # Search in grid pattern
            for x in range(start_x, end_x + 1, jump):
                for y in range(start_y, end_y + 1, jump):
                    # Skip center (already searched)
                    if x == center_x and y == center_y:
                        continue
                    
                    # Search at this position
                    distance = self._search_at_position(source_region, template, x, y)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_x = x
                        best_y = y
            
            # Adaptive step size adjustment (Arena Tracker's logic)
            if not max_jump_reached and best_distance > 3.5 and jump == init_jump * 4:
                jump = jump // 2
                max_jump_reached = True
            elif not max_jump_reached and best_distance > 3.5:
                jump = jump * 2
            elif center_best == best_distance:
                jump = jump // 2
            else:
                # Move to new best position
                center_x = best_x
                center_y = best_y
                center_best = best_distance
                
                if max_jump_reached and jump == init_jump * 2:
                    jump = jump // 2
        
        return best_distance, best_x, best_y
    
    def _search_at_position(self, source_region: np.ndarray, template: np.ndarray,
                          offset_x: int, offset_y: int) -> float:
        """
        Search for template at specific position with offset.
        
        Args:
            source_region: Source image region
            template: Template to match
            offset_x: X offset from center
            offset_y: Y offset from center
            
        Returns:
            L2 distance at this position
        """
        try:
            # Calculate sample region with offset
            center_x = source_region.shape[1] // 2
            center_y = source_region.shape[0] // 2
            
            sample_x = center_x + offset_x
            sample_y = center_y + offset_y
            
            # Extract sample region
            sample_region = self._extract_template_region(
                source_region, sample_x, sample_y, 
                template.shape[1], template.shape[0]
            )
            
            if sample_region is None:
                return float('inf')
            
            # Compute L2 distance
            return self.compute_l2_distance(sample_region, template)
            
        except Exception as e:
            self.logger.error(f"Position search failed: {e}")
            return float('inf')
    
    def _extract_template_region(self, source: np.ndarray, x: int, y: int,
                                width: int, height: int) -> Optional[np.ndarray]:
        """
        Extract a template-sized region from source image.
        
        Args:
            source: Source image
            x: Center X coordinate
            y: Center Y coordinate
            width: Template width
            height: Template height
            
        Returns:
            Extracted region or None if out of bounds
        """
        try:
            # Calculate region bounds
            left = max(0, x - width // 2)
            top = max(0, y - height // 2)
            right = min(source.shape[1], left + width)
            bottom = min(source.shape[0], top + height)
            
            # Check if region is valid
            if right - left < width // 2 or bottom - top < height // 2:
                return None
            
            # Extract region
            region = source[top:bottom, left:right]
            
            # Resize if needed
            if region.shape[:2] != (height, width):
                region = cv2.resize(region, (width, height))
            
            return region
            
        except Exception as e:
            self.logger.error(f"Region extraction failed: {e}")
            return None
    
    def load_mana_templates(self, mana_templates: Dict[int, np.ndarray]):
        """
        Load mana cost templates.
        
        Args:
            mana_templates: Dictionary mapping mana cost to template image
        """
        self.mana_templates = mana_templates.copy()
        self.logger.info(f"Loaded {len(self.mana_templates)} mana templates")
    
    def load_rarity_templates(self, rarity_templates: Dict[int, np.ndarray]):
        """
        Load rarity templates.
        
        Args:
            rarity_templates: Dictionary mapping rarity to template image
        """
        self.rarity_templates = rarity_templates.copy()
        self.logger.info(f"Loaded {len(self.rarity_templates)} rarity templates")
    
    def detect_mana_cost(self, mana_region: np.ndarray) -> Optional[int]:
        """
        Detect mana cost in a card region.
        
        Args:
            mana_region: Image region containing mana cost
            
        Returns:
            Detected mana cost or None if not found
        """
        if not self.mana_templates:
            self.logger.warning("No mana templates loaded")
            return None
        
        best_distance = float('inf')
        best_mana = None
        
        for mana_cost, template in self.mana_templates.items():
            # Use adaptive grid search
            distance, _, _ = self.adaptive_grid_search(mana_region, template)
            
            if distance < best_distance:
                best_distance = distance
                best_mana = mana_cost
        
        # Check threshold (Arena Tracker's approach)
        if best_distance <= self.MANA_L2_THRESHOLD:
            self.logger.debug(f"Mana cost detected: {best_mana} (distance: {best_distance:.3f})")
            return best_mana
        else:
            self.logger.debug(f"No mana cost detected (best distance: {best_distance:.3f})")
            return None
    
    def detect_rarity(self, rarity_region: np.ndarray) -> Optional[int]:
        """
        Detect rarity in a card region.
        
        Args:
            rarity_region: Image region containing rarity gem
            
        Returns:
            Detected rarity or None if not found
        """
        if not self.rarity_templates:
            self.logger.warning("No rarity templates loaded")
            return None
        
        best_distance = float('inf')
        best_rarity = None
        
        for rarity, template in self.rarity_templates.items():
            # Use adaptive grid search
            distance, _, _ = self.adaptive_grid_search(rarity_region, template)
            
            if distance < best_distance:
                best_distance = distance
                best_rarity = rarity
        
        # Check threshold (Arena Tracker's approach)
        if best_distance <= self.RARITY_L2_THRESHOLD:
            self.logger.debug(f"Rarity detected: {best_rarity} (distance: {best_distance:.3f})")
            return best_rarity
        else:
            self.logger.debug(f"No rarity detected (best distance: {best_distance:.3f})")
            return None
    
    def get_template_counts(self) -> Tuple[int, int]:
        """
        Get the number of loaded templates.
        
        Returns:
            Tuple of (mana_count, rarity_count)
        """
        return len(self.mana_templates), len(self.rarity_templates)
    
    def find_mana_crystals(self, screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find mana crystal positions in a screenshot using basic color detection.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            List of (x, y) positions of detected mana crystals
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Mana crystal color ranges (blue/cyan)
            mana_ranges = [
                ([100, 50, 50], [130, 255, 255]),   # Blue range
                ([85, 50, 50], [105, 255, 255])     # Cyan range
            ]
            
            mana_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in mana_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mana_mask = cv2.bitwise_or(mana_mask, mask)
            
            # Clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mana_mask = cv2.morphologyEx(mana_mask, cv2.MORPH_CLOSE, kernel)
            mana_mask = cv2.morphologyEx(mana_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mana_positions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Mana crystals are small circular/oval shapes
                if 100 < area < 2000:  # Reasonable mana crystal size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (should be roughly circular)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.7 < aspect_ratio < 1.4:  # Roughly circular
                        center_x = x + w // 2
                        center_y = y + h // 2
                        mana_positions.append((center_x, center_y))
            
            # Sort by x position (left to right) and take up to 3
            mana_positions.sort(key=lambda pos: pos[0])
            return mana_positions[:3]
            
        except Exception as e:
            self.logger.error(f"Error in mana crystal detection: {e}")
            return []


# Global template matcher instance
_template_matcher = None


def get_template_matcher() -> TemplateMatcher:
    """
    Get the global template matcher instance.
    
    Returns:
        TemplateMatcher instance
    """
    global _template_matcher
    if _template_matcher is None:
        _template_matcher = TemplateMatcher()
    return _template_matcher