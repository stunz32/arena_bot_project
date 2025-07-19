#!/usr/bin/env python3
"""
Smart Coordinate Detection System
Automatically detects Hearthstone Arena interface and calculates precise card positions.
Based on the proven red area detection method with multiple fallback strategies.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SmartCoordinateDetector:
    """
    Intelligent coordinate detection system that automatically finds the Hearthstone
    arena interface and calculates precise card positions.
    
    Uses a multi-strategy approach:
    1. Red area detection (primary - proven to work)
    2. Template matching (fallback)
    3. Edge detection (fallback)
    4. Manual positioning (last resort)
    """
    
    def __init__(self):
        """Initialize the smart coordinate detector."""
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters - tuned based on successful tests
        self.min_interface_area = 800000  # Minimum interface size
        self.expected_interface_ratio = 1.7  # Width/height ratio (1197/704 ≈ 1.7)
        self.card_y_offset = 90  # Y offset within interface
        
        # Arena Helper-style reference coordinates (1920×1080 base)
        self.reference_resolution = (1920, 1080)
        self.reference_card_size = (250, 370)  # Base card size for scaling
        
        # Red area detection parameters (proven successful)
        self.red_hsv_ranges = [
            ([0, 50, 50], [10, 255, 255]),    # Lower red range
            ([170, 50, 50], [180, 255, 255])  # Upper red range
        ]
        
        # Auto-calibrated offsets for specific resolutions (discovered through calibration)
        self.resolution_calibrations = {
            # Removed complex calibration - testing base algorithm first
        }
    
    def calculate_optimal_card_size(self, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Calculate optimal card size using Arena Helper-style dynamic scaling.
        
        Based on your successful detection (Card 3: 357×534), we target similar sizes
        for ultrawide displays while maintaining compatibility across resolutions.
        
        Args:
            screen_width: Current screen width
            screen_height: Current screen height
            
        Returns:
            (optimal_width, optimal_height) for card regions
        """
        # Calculate scaling factors from reference resolution
        scale_x = screen_width / self.reference_resolution[0]
        scale_y = screen_height / self.reference_resolution[1]
        
        # Use the SMALLER scale to prevent out-of-bounds regions
        # This ensures regions fit within the screen bounds
        scale = min(scale_x, scale_y)
        
        # Calculate optimal dimensions
        base_width, base_height = self.reference_card_size
        optimal_width = int(base_width * scale)
        optimal_height = int(base_height * scale)
        
        # Ensure minimum sizes for detection algorithm compatibility
        # pHash needs 300×420+, histogram needs 250×350+
        min_width, min_height = 300, 420
        optimal_width = max(optimal_width, min_width)
        optimal_height = max(optimal_height, min_height)
        
        # Cap maximum sizes to prevent excessive memory usage
        max_width, max_height = 500, 700
        optimal_width = min(optimal_width, max_width)
        optimal_height = min(optimal_height, max_height)
        
        self.logger.info(f"Dynamic card sizing: {screen_width}×{screen_height} → {optimal_width}×{optimal_height}")
        self.logger.info(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}, used={scale:.3f}")
        
        return optimal_width, optimal_height
    
    def _assess_region_quality(self, region: np.ndarray) -> float:
        """
        Assess the quality of a card region for detection algorithms.
        
        Args:
            region: Card region image array
            
        Returns:
            Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        try:
            if region.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Quality metrics
            # 1. Brightness analysis (should not be too dark or too bright)
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 127) / 127  # Optimal around 127
            brightness_score = max(0.0, min(1.0, brightness_score))
            
            # 2. Contrast analysis (should have good variation)
            contrast = np.std(gray)
            contrast_score = min(contrast / 50.0, 1.0)  # Good contrast > 50
            
            # 3. Edge density (cards should have defined edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density / 0.15, 1.0)  # Target 15% edge pixels
            
            # 4. Not uniform color (should have texture/details)
            unique_values = len(np.unique(gray))
            texture_score = min(unique_values / 100.0, 1.0)  # More variety is better
            
            # Weighted combination
            quality_score = (brightness_score * 0.3 + 
                           contrast_score * 0.3 + 
                           edge_score * 0.2 + 
                           texture_score * 0.2)
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error assessing region quality: {e}")
            return 0.0
    
    def validate_card_contour(self, contour: np.ndarray) -> bool:
        """
        Validate if a contour represents a Hearthstone card using Magic Card Detector approach.
        
        Args:
            contour: OpenCV contour to validate
            
        Returns:
            True if contour is card-like, False otherwise
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio validation (Hearthstone cards are taller than wide)
            aspect_ratio = w / h if h > 0 else 0
            valid_aspect = 0.60 < aspect_ratio < 0.75  # Hearthstone card ratio ~0.67
            
            # Area validation (reasonable card size)
            area = cv2.contourArea(contour)
            min_area = 15000   # Minimum card area
            max_area = 200000  # Maximum card area  
            valid_area = min_area < area < max_area
            
            # Size validation (minimum dimensions for detection)
            min_width, min_height = 200, 280
            valid_size = w > min_width and h > min_height
            
            # Contour complexity (should not be too simple)
            perimeter = cv2.arcLength(contour, True)
            complexity_ratio = area / (perimeter ** 2) if perimeter > 0 else 0
            valid_complexity = 0.04 < complexity_ratio < 0.25  # Reasonable shape complexity
            
            is_valid = valid_aspect and valid_area and valid_size and valid_complexity
            
            if not is_valid:
                self.logger.debug(f"Contour rejected: aspect={aspect_ratio:.3f}, area={area}, "
                                f"size={w}×{h}, complexity={complexity_ratio:.3f}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating contour: {e}")
            return False
    
    def validate_card_region(self, x: int, y: int, w: int, h: int) -> bool:
        """
        Validate if a region represents a valid Hearthstone card.
        
        Args:
            x, y, w, h: Region coordinates and dimensions
            
        Returns:
            True if region is card-like, False otherwise
        """
        try:
            # Aspect ratio validation (Hearthstone cards are taller than wide)
            aspect_ratio = w / h if h > 0 else 0
            valid_aspect = 0.60 < aspect_ratio < 0.75  # Hearthstone card ratio ~0.67
            
            # Area validation (reasonable card size)
            area = w * h
            min_area = 15000   # Minimum card area
            max_area = 200000  # Maximum card area  
            valid_area = min_area < area < max_area
            
            # Size validation (minimum dimensions for detection)
            min_width, min_height = 200, 280
            valid_size = w > min_width and h > min_height
            
            is_valid = valid_aspect and valid_area and valid_size
            
            if not is_valid:
                self.logger.debug(f"Region rejected: aspect={aspect_ratio:.3f}, area={area}, size={w}×{h}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating region: {e}")
            return False
    
    def score_card_region(self, x: int, y: int, w: int, h: int) -> float:
        """
        Score a card region for quality and card-likeness.
        
        Args:
            x, y, w, h: Region coordinates and dimensions
            
        Returns:
            Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        try:
            # Aspect ratio scoring (target Hearthstone ratio ~0.67)
            aspect_ratio = w / h if h > 0 else 0
            target_aspect = 0.67
            aspect_score = 1.0 - abs(aspect_ratio - target_aspect) / target_aspect
            aspect_score = max(0.0, min(1.0, aspect_score))
            
            # Size scoring (larger regions generally better for detection)
            area = w * h
            target_area = 100000  # ~316×316 pixel region
            size_score = min(area / target_area, 1.0)
            
            # Position scoring (prefer center regions)
            # This will be enhanced when we have screen dimensions
            position_score = 1.0  # Placeholder for now
            
            # Weighted combination
            total_score = (aspect_score * 0.5 + 
                          size_score * 0.3 + 
                          position_score * 0.2)
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error scoring region: {e}")
            return 0.0
    
    def detect_cards_via_mana_anchors(self, screenshot: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect card positions using mana crystal anchors (Arena Helper method).
        
        Uses template matching to find mana crystals, then calculates full card
        regions from anchor positions for sub-pixel accuracy.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            List of (x, y, width, height) for detected card regions
        """
        try:
            height, width = screenshot.shape[:2]
            self.logger.info("Attempting mana crystal anchor detection...")
            
            # Try to use existing template matcher for mana crystals
            try:
                from ..detection.template_matcher import TemplateMatcher
                template_matcher = TemplateMatcher()
                mana_positions = template_matcher.find_mana_crystals(screenshot)
                self.logger.info(f"Template matcher found {len(mana_positions)} mana crystals")
            except ImportError:
                self.logger.warning("Template matcher not available, using basic detection")
                mana_positions = self._detect_mana_crystals_basic(screenshot)
            
            if not mana_positions:
                self.logger.info("No mana crystals detected")
                return []
            
            # Calculate optimal card size for current resolution
            card_width, card_height = self.calculate_optimal_card_size(width, height)
            
            # Calculate card regions from mana crystal positions
            card_regions = []
            for i, (mana_x, mana_y) in enumerate(mana_positions[:3]):  # Max 3 cards
                # Calculate full card region from mana crystal position
                # Mana crystal is typically at top-left of card with some offset
                card_x = mana_x - 40  # Offset left from mana crystal
                card_y = mana_y - 20  # Offset up from mana crystal
                
                # Ensure region is within screenshot bounds
                card_x = max(0, min(card_x, width - card_width))
                card_y = max(0, min(card_y, height - card_height))
                
                card_regions.append((card_x, card_y, card_width, card_height))
                self.logger.info(f"Card {i+1} from mana anchor: ({card_x}, {card_y}, {card_width}, {card_height})")
            
            # Validate regions for reasonable spacing and non-overlap
            validated_regions = self._validate_anchor_regions(card_regions, screenshot)
            
            self.logger.info(f"Mana anchor detection: {len(validated_regions)}/3 cards positioned")
            return validated_regions
            
        except Exception as e:
            self.logger.error(f"Error in mana anchor detection: {e}")
            return []
    
    def _detect_mana_crystals_basic(self, screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """
        Basic mana crystal detection using color and shape analysis.
        Fallback when template matcher is not available.
        
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
            self.logger.error(f"Error in basic mana detection: {e}")
            return []
    
    def _validate_anchor_regions(self, regions: List[Tuple[int, int, int, int]], 
                                screenshot: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Validate card regions detected via anchor positioning.
        
        Args:
            regions: List of candidate card regions
            screenshot: Full screen screenshot
            
        Returns:
            List of validated card regions
        """
        try:
            if not regions:
                return []
            
            validated = []
            height, width = screenshot.shape[:2]
            
            for i, (x, y, w, h) in enumerate(regions):
                # Bounds checking
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    self.logger.warning(f"Anchor region {i+1} out of bounds: ({x}, {y}, {w}, {h})")
                    continue
                
                # Quality assessment
                region = screenshot[y:y+h, x:x+w]
                quality = self._assess_region_quality(region)
                
                if quality > 0.2:  # Basic quality threshold
                    validated.append((x, y, w, h))
                    self.logger.info(f"Anchor region {i+1} validated (quality: {quality:.3f})")
                else:
                    self.logger.warning(f"Anchor region {i+1} poor quality: {quality:.3f}")
            
            # Check for reasonable spacing between cards
            if len(validated) >= 2:
                validated = self._ensure_reasonable_spacing(validated)
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Error validating anchor regions: {e}")
            return regions  # Return original if validation fails
    
    def _ensure_reasonable_spacing(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Ensure card regions have reasonable spacing (not overlapping or too close).
        
        Args:
            regions: List of card regions
            
        Returns:
            List of regions with reasonable spacing
        """
        try:
            if len(regions) < 2:
                return regions
            
            # Sort by x position
            sorted_regions = sorted(regions, key=lambda r: r[0])
            
            validated_regions = [sorted_regions[0]]  # Always include first
            
            for current in sorted_regions[1:]:
                cx, cy, cw, ch = current
                
                # Check spacing with previously validated regions
                valid_spacing = True
                for prev in validated_regions:
                    px, py, pw, ph = prev
                    
                    # Calculate horizontal distance between regions
                    distance = cx - (px + pw)  # Distance from end of prev to start of current
                    
                    # Minimum spacing should be at least 20 pixels
                    if distance < 20:
                        self.logger.warning(f"Region spacing too small: {distance} pixels")
                        valid_spacing = False
                        break
                    
                    # Maximum spacing check (cards shouldn't be too far apart)
                    if distance > 500:  # Reasonable maximum for ultrawide
                        self.logger.warning(f"Region spacing too large: {distance} pixels")
                        valid_spacing = False
                        break
                
                if valid_spacing:
                    validated_regions.append(current)
            
            self.logger.info(f"Spacing validation: {len(validated_regions)}/{len(regions)} regions kept")
            return validated_regions
            
        except Exception as e:
            self.logger.error(f"Error checking spacing: {e}")
            return regions
    
    def optimize_region_for_phash(self, x: int, y: int, w: int, h: int, 
                                 max_width: int, max_height: int) -> Tuple[int, int, int, int]:
        """
        Optimize a card region for pHash detection performance.
        
        pHash works best with regions 300×420+ pixels for sub-millisecond detection.
        This method ensures regions meet these requirements.
        
        Args:
            x, y, w, h: Original region coordinates
            max_width, max_height: Screenshot bounds
            
        Returns:
            Optimized (x, y, width, height) for pHash
        """
        try:
            # pHash optimal dimensions (based on research and testing)
            min_width, min_height = 300, 420
            optimal_width, optimal_height = 350, 500  # Target for best performance
            
            # Calculate current region quality for pHash
            current_area = w * h
            target_area = optimal_width * optimal_height
            
            if current_area >= target_area * 0.8:  # Already good size
                self.logger.debug(f"Region already pHash-optimal: {w}×{h}")
                return (x, y, w, h)
            
            # Calculate scale factor to reach optimal size
            scale_w = optimal_width / w
            scale_h = optimal_height / h
            scale = min(scale_w, scale_h)  # Use smaller scale to maintain aspect ratio
            
            # Apply scaling
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Ensure minimums
            new_w = max(new_w, min_width)
            new_h = max(new_h, min_height)
            
            # Center the expansion
            expansion_w = new_w - w
            expansion_h = new_h - h
            new_x = x - expansion_w // 2
            new_y = y - expansion_h // 2
            
            # Ensure bounds
            new_x = max(0, min(new_x, max_width - new_w))
            new_y = max(0, min(new_y, max_height - new_h))
            new_w = min(new_w, max_width - new_x)
            new_h = min(new_h, max_height - new_y)
            
            self.logger.info(f"pHash optimization: {w}×{h} → {new_w}×{new_h} (scale: {scale:.3f})")
            return (new_x, new_y, new_w, new_h)
            
        except Exception as e:
            self.logger.error(f"Error optimizing region for pHash: {e}")
            return (x, y, w, h)  # Return original on error
    
    def optimize_region_for_histogram(self, x: int, y: int, w: int, h: int,
                                     max_width: int, max_height: int) -> Tuple[int, int, int, int]:
        """
        Optimize a card region for histogram matching performance.
        
        Arena Tracker's histogram matching works best with 250×350+ regions.
        
        Args:
            x, y, w, h: Original region coordinates
            max_width, max_height: Screenshot bounds
            
        Returns:
            Optimized (x, y, width, height) for histogram matching
        """
        try:
            # Histogram matching optimal dimensions
            min_width, min_height = 250, 350
            optimal_width, optimal_height = 280, 400
            
            # Check if already optimal
            if w >= optimal_width and h >= optimal_height:
                return (x, y, w, h)
            
            # Scale to reach optimal size
            scale_w = optimal_width / w if w < optimal_width else 1.0
            scale_h = optimal_height / h if h < optimal_height else 1.0
            scale = max(scale_w, scale_h)  # Ensure both dimensions meet minimums
            
            # Don't oversample too much
            scale = min(scale, 1.5)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Center expansion
            new_x = x - (new_w - w) // 2
            new_y = y - (new_h - h) // 2
            
            # Bounds checking
            new_x = max(0, min(new_x, max_width - new_w))
            new_y = max(0, min(new_y, max_height - new_h))
            new_w = min(new_w, max_width - new_x)
            new_h = min(new_h, max_height - new_y)
            
            self.logger.debug(f"Histogram optimization: {w}×{h} → {new_w}×{new_h}")
            return (new_x, new_y, new_w, new_h)
            
        except Exception as e:
            self.logger.error(f"Error optimizing region for histogram: {e}")
            return (x, y, w, h)
    
    def optimize_region_for_ultimate_detection(self, x: int, y: int, w: int, h: int,
                                             max_width: int, max_height: int) -> Tuple[int, int, int, int]:
        """
        Optimize a card region for Ultimate Detection preprocessing.
        
        Ultimate Detection benefits from larger regions for CLAHE and bilateral filtering.
        
        Args:
            x, y, w, h: Original region coordinates 
            max_width, max_height: Screenshot bounds
            
        Returns:
            Optimized (x, y, width, height) for Ultimate Detection
        """
        try:
            # Ultimate Detection preprocessing optimal dimensions
            min_width, min_height = 280, 400
            max_width_limit, max_height_limit = 450, 650
            
            # Check if region needs adjustment
            needs_expansion = w < min_width or h < min_height
            needs_reduction = w > max_width_limit or h > max_height_limit
            
            if not needs_expansion and not needs_reduction:
                return (x, y, w, h)
            
            if needs_expansion:
                # Expand to minimum size
                scale = max(min_width / w, min_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Center expansion
                new_x = x - (new_w - w) // 2
                new_y = y - (new_h - h) // 2
                
            else:  # needs_reduction
                # Reduce to maximum size
                scale = min(max_width_limit / w, max_height_limit / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Center reduction
                new_x = x + (w - new_w) // 2
                new_y = y + (h - new_h) // 2
            
            # Bounds checking
            new_x = max(0, min(new_x, max_width - new_w))
            new_y = max(0, min(new_y, max_height - new_h))
            new_w = min(new_w, max_width - new_x)
            new_h = min(new_h, max_height - new_y)
            
            self.logger.debug(f"Ultimate Detection optimization: {w}×{h} → {new_w}×{new_h}")
            return (new_x, new_y, new_w, new_h)
            
        except Exception as e:
            self.logger.error(f"Error optimizing region for Ultimate Detection: {e}")
            return (x, y, w, h)
    
    def assess_region_for_detection_method(self, region: np.ndarray) -> Dict[str, float]:
        """
        Assess which detection method would work best for a given region.
        
        Args:
            region: Card region image array
            
        Returns:
            Dict with confidence scores for each detection method
        """
        try:
            if region.size == 0:
                return {"phash": 0.0, "histogram": 0.0, "ultimate": 0.0, "basic": 0.2}
            
            h, w = region.shape[:2]
            area = w * h
            
            # Quality assessment
            quality_score = self._assess_region_quality(region)
            
            # Method suitability scores
            scores = {}
            
            # pHash suitability (needs high quality and good size)
            phash_size_score = min(area / 105000, 1.0)  # 300×350 = 105,000
            phash_quality_threshold = 0.7  # High quality needed
            scores["phash"] = (phash_size_score * 0.6 + 
                             min(quality_score / phash_quality_threshold, 1.0) * 0.4)
            
            # Histogram suitability (works with medium quality and size)
            hist_size_score = min(area / 87500, 1.0)   # 250×350 = 87,500
            hist_quality_threshold = 0.5  # Medium quality acceptable
            scores["histogram"] = (hist_size_score * 0.5 + 
                                 min(quality_score / hist_quality_threshold, 1.0) * 0.5)
            
            # Ultimate Detection suitability (good for poor quality regions)
            ultimate_size_score = min(area / 112000, 1.0)  # 280×400 = 112,000
            ultimate_quality_boost = 1.0 - quality_score  # Better for poor quality
            scores["ultimate"] = (ultimate_size_score * 0.4 + 
                                ultimate_quality_boost * 0.4 + 
                                quality_score * 0.2)
            
            # Basic fallback (always works)
            scores["basic"] = 0.5 + quality_score * 0.3
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error assessing detection methods: {e}")
            return {"phash": 0.0, "histogram": 0.3, "ultimate": 0.3, "basic": 0.5}
    
    def recommend_optimal_detection_method(self, region: np.ndarray) -> Tuple[str, float]:
        """
        Recommend the optimal detection method for a region.
        
        Args:
            region: Card region image array
            
        Returns:
            (method_name, confidence_score) tuple
        """
        try:
            scores = self.assess_region_for_detection_method(region)
            
            # Find best method
            best_method = max(scores.items(), key=lambda item: item[1])
            method_name, confidence = best_method
            
            self.logger.debug(f"Detection method scores: {scores}")
            self.logger.info(f"Recommended method: {method_name} (confidence: {confidence:.3f})")
            
            return method_name, confidence
            
        except Exception as e:
            self.logger.error(f"Error recommending detection method: {e}")
            return "basic", 0.5
    
    def detect_hearthstone_interface(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Automatically detect the Hearthstone arena interface using red area detection.
        This method has proven successful in previous tests.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            (x, y, width, height) of interface region or None if not found
        """
        try:
            self.logger.info(f"Detecting Hearthstone interface in {screenshot.shape[1]}x{screenshot.shape[0]} screenshot")
            
            # Strategy 1: Red area detection (primary method)
            interface_rect = self._detect_red_interface(screenshot)
            if interface_rect:
                self.logger.info(f"✅ Red area detection successful: {interface_rect}")
                return interface_rect
            
            # Strategy 2: Large dark red region detection
            interface_rect = self._detect_dark_red_interface(screenshot)
            if interface_rect:
                self.logger.info(f"✅ Dark red detection successful: {interface_rect}")
                return interface_rect
            
            # Strategy 3: Contour-based detection
            interface_rect = self._detect_contour_interface(screenshot)
            if interface_rect:
                self.logger.info(f"✅ Contour detection successful: {interface_rect}")
                return interface_rect
            
            # Strategy 4: Fallback to manual positioning (based on successful coordinates)
            interface_rect = self._estimate_interface_position(screenshot)
            if interface_rect:
                self.logger.info(f"⚠️ Using estimated interface position: {interface_rect}")
                return interface_rect
            
            self.logger.warning("❌ All interface detection methods failed")
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting interface: {e}")
            return None
    
    def _detect_red_interface(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect interface using red color areas (proven method)."""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Create mask for red areas
            red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in self.red_hsv_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                red_mask = cv2.bitwise_or(red_mask, mask)
            
            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours of red areas
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for large rectangular red areas (interface background)
            best_rect = None
            best_score = 0
            valid_contours = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Must be large enough to be the main interface
                if area < self.min_interface_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enhanced validation using Magic Card Detector approach
                # Check aspect ratio (interface should be wider than tall)
                ratio = w / h if h > 0 else 0
                if ratio < 1.2 or ratio > 2.5:  # Interface should be reasonably wide
                    continue
                
                # Additional geometric validation
                # Interface should have reasonable proportions
                if w < 800 or h < 400:  # Minimum interface size
                    continue
                
                valid_contours += 1
                
                # Enhanced scoring with multiple factors
                ratio_score = 1.0 - abs(ratio - self.expected_interface_ratio) / self.expected_interface_ratio
                area_score = min(area / self.min_interface_area, 3.0) / 3.0
                
                # Size preference score (larger interfaces are generally better)
                size_preference = min((w * h) / 1000000, 1.0)  # Prefer up to 1M pixel interfaces
                
                # Convexity score (interfaces should be relatively convex)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                convexity_score = area / hull_area if hull_area > 0 else 0
                
                total_score = (ratio_score * 0.4 + 
                             area_score * 0.3 + 
                             size_preference * 0.2 + 
                             convexity_score * 0.1)
                
                if total_score > best_score:
                    best_score = total_score
                    best_rect = (x, y, w, h)
            
            self.logger.info(f"Red detection: Found {len(contours)} contours, {valid_contours} valid interfaces")
            
            if best_rect and best_score > 0.5:
                self.logger.info(f"Red interface detected with score {best_score:.3f}")
                return best_rect
                
            return None
            
        except Exception as e:
            self.logger.error(f"Red detection error: {e}")
            return None
    
    def _detect_dark_red_interface(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect interface using darker red tones."""
        try:
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Darker red ranges for interface background
            dark_red_ranges = [
                ([0, 30, 30], [15, 255, 200]),
                ([165, 30, 30], [180, 255, 200])
            ]
            
            red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in dark_red_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                red_mask = cv2.bitwise_or(red_mask, mask)
            
            # Find largest contour
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > self.min_interface_area:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return (x, y, w, h)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Dark red detection error: {e}")
            return None
    
    def _detect_contour_interface(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect interface using edge detection and contours with Magic Card Detector validation."""
        try:
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply morphological operations to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Score and validate contours
            best_interface = None
            best_score = 0
            valid_candidates = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_interface_area:
                    continue
                    
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic interface validation (should be wider than tall)
                ratio = w / h if h > 0 else 0
                if not (1.2 < ratio < 2.5):  # Interface aspect ratio
                    continue
                
                # Size validation
                if w < 800 or h < 400:  # Minimum interface dimensions
                    continue
                
                valid_candidates += 1
                
                # Approximate the contour for shape analysis
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Enhanced scoring with multiple geometric factors
                # 1. Shape complexity (should be roughly rectangular)
                vertex_score = 1.0 if 4 <= len(approx) <= 8 else 0.5
                
                # 2. Aspect ratio match to expected interface
                ratio_score = 1.0 - abs(ratio - self.expected_interface_ratio) / self.expected_interface_ratio
                ratio_score = max(0.0, min(1.0, ratio_score))
                
                # 3. Area score (larger is generally better for interfaces)
                area_score = min(area / self.min_interface_area, 3.0) / 3.0
                
                # 4. Contour solidity (filled vs convex hull ratio)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity_score = area / hull_area if hull_area > 0 else 0
                
                # 5. Extent (contour area vs bounding rectangle area)
                rect_area = w * h
                extent_score = area / rect_area if rect_area > 0 else 0
                
                # Weighted combination of scores
                total_score = (vertex_score * 0.2 + 
                             ratio_score * 0.3 + 
                             area_score * 0.2 + 
                             solidity_score * 0.15 + 
                             extent_score * 0.15)
                
                if total_score > best_score:
                    best_score = total_score
                    best_interface = (x, y, w, h)
            
            self.logger.info(f"Contour detection: Found {len(contours)} contours, {valid_candidates} valid candidates")
            
            if best_interface and best_score > 0.6:  # Higher threshold for contour detection
                self.logger.info(f"Contour interface detected with score {best_score:.3f}")
                return best_interface
            
            return None
            
        except Exception as e:
            self.logger.error(f"Contour detection error: {e}")
            return None
    
    def _estimate_interface_position(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Estimate interface position based on successful test coordinates.
        This is a fallback when automatic detection fails.
        """
        try:
            height, width = screenshot.shape[:2]
            self.logger.info(f"Estimating interface for {width}x{height} screenshot")
            
            # Based on successful test: interface was at (1333, 180, 1197, 704) for a 3408x1250 screenshot
            # Calculate proportional positioning
            
            # Known working coordinates from test_correct_coordinates.py
            ref_width, ref_height = 3408, 1250
            ref_interface = (1333, 180, 1197, 704)
            
            if width >= 2000 and height >= 1000:  # Large screen
                # Scale the reference coordinates
                scale_x = width / ref_width
                scale_y = height / ref_height
                
                est_x = int(ref_interface[0] * scale_x)
                est_y = int(ref_interface[1] * scale_y)
                est_w = int(ref_interface[2] * scale_x)
                est_h = int(ref_interface[3] * scale_y)
                
                # Ensure coordinates are within bounds
                est_x = max(0, min(est_x, width - est_w))
                est_y = max(0, min(est_y, height - est_h))
                est_w = min(est_w, width - est_x)
                est_h = min(est_h, height - est_y)
                
                return (est_x, est_y, est_w, est_h)
            
            elif width >= 1920 and height >= 1080:  # Standard HD
                # Center the interface for standard resolutions
                est_w = int(width * 0.6)  # Interface takes ~60% of width
                est_h = int(height * 0.5)  # Interface takes ~50% of height
                est_x = (width - est_w) // 2
                est_y = (height - est_h) // 2
                
                return (est_x, est_y, est_w, est_h)
            
            else:  # Smaller resolutions
                # Assume interface covers most of the screen
                est_w = int(width * 0.8)
                est_h = int(height * 0.7)
                est_x = (width - est_w) // 2
                est_y = (height - est_h) // 2
                
                return (est_x, est_y, est_w, est_h)
            
        except Exception as e:
            self.logger.error(f"Estimation error: {e}")
            return None
    
    def apply_calibration_offsets(self, card_positions: List[Tuple[int, int, int, int]], 
                                 screen_width: int, screen_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Apply auto-calibrated offsets for specific resolutions to improve accuracy.
        
        Args:
            card_positions: Original card positions
            screen_width: Screen width
            screen_height: Screen height
            
        Returns:
            Calibrated card positions with improved accuracy
        """
        resolution_key = f"{screen_width}x{screen_height}"
        
        if resolution_key in self.resolution_calibrations:
            calibration = self.resolution_calibrations[resolution_key]
            self.logger.info(f"Applying calibration for {resolution_key}: {calibration['description']}")
            
            calibrated_positions = []
            for x, y, w, h in card_positions:
                # Apply offsets and scaling
                new_x = x + calibration["x_offset"]
                new_y = y + calibration["y_offset"] 
                new_w = int(w * calibration["width_scale"])
                new_h = int(h * calibration["height_scale"])
                
                # Ensure bounds
                new_x = max(0, min(new_x, screen_width - new_w))
                new_y = max(0, min(new_y, screen_height - new_h))
                
                calibrated_positions.append((new_x, new_y, new_w, new_h))
            
            self.logger.info(f"Calibration applied: x_offset={calibration['x_offset']}, y_offset={calibration['y_offset']}")
            self.logger.info(f"Scale factors: width={calibration['width_scale']}, height={calibration['height_scale']}")
            return calibrated_positions
        else:
            self.logger.info(f"No calibration available for {resolution_key}, using original positions")
            return card_positions

    def calculate_card_positions(self, interface_rect: Tuple[int, int, int, int], 
                                screen_width: int = None, screen_height: int = None) -> List[Tuple[int, int, int, int]]:
        """
        Calculate the 3 card positions within the detected interface using dynamic sizing.
        Uses Arena Helper-style scaling for optimal detection accuracy.
        
        Args:
            interface_rect: (x, y, width, height) of detected interface
            screen_width: Screen width for dynamic sizing (optional)
            screen_height: Screen height for dynamic sizing (optional)
            
        Returns:
            List of (x, y, width, height) for each of the 3 cards
        """
        interface_x, interface_y, interface_w, interface_h = interface_rect
        
        # Calculate optimal card dimensions if screen size provided
        if screen_width and screen_height:
            card_width, card_height = self.calculate_optimal_card_size(screen_width, screen_height)
        else:
            # Fallback to estimated sizing based on interface
            scale_factor = interface_w / 1197  # Reference interface width
            card_width = int(250 * scale_factor)
            card_height = int(370 * scale_factor)
            # Ensure minimums for detection
            card_width = max(card_width, 300)
            card_height = max(card_height, 420)
        
        self.logger.info(f"Calculating card positions for interface: {interface_rect}")
        self.logger.info(f"Using dynamic card size: {card_width}×{card_height}")
        
        # Distribute 3 cards across the interface width
        # Based on successful test: cards are positioned in 4 sections (1, 2, 3)
        card_spacing = interface_w // 4
        
        card_positions = []
        for i in range(3):
            # Calculate card position
            card_x = interface_x + card_spacing * (i + 1) - card_width // 2
            card_y = interface_y + self.card_y_offset
            
            # Ensure card is within screenshot bounds
            card_x = max(0, card_x)
            card_y = max(0, card_y)
            
            card_positions.append((card_x, card_y, card_width, card_height))
        
        # Apply auto-calibration if available for this resolution
        if screen_width and screen_height:
            card_positions = self.apply_calibration_offsets(card_positions, screen_width, screen_height)
        
        self.logger.info(f"Final calculated {len(card_positions)} card positions:")
        for i, pos in enumerate(card_positions):
            self.logger.info(f"  Card {i+1}: {pos}")
        
        return card_positions
    
    def detect_cards_automatically(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Automatically detect card positions using the proven red area detection method.
        Returns coarse, unrefined card positions from interface detection.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            Dict with detection results or None if failed
        """
        try:
            height, width = screenshot.shape[:2]
            self.logger.info(f"Starting automatic detection for {width}×{height} screenshot")
            
            # Detect the Hearthstone interface
            interface_rect = self.detect_hearthstone_interface(screenshot)
            if not interface_rect:
                self.logger.error("Failed to detect Hearthstone interface")
                return None
            
            # Calculate card positions with dynamic sizing
            card_positions = self.calculate_card_positions(interface_rect, width, height)
            
            # Assess quality and generate recommendations
            region_qualities = []
            method_recommendations = []
            optimized_regions = {}
            
            for i, (x, y, w, h) in enumerate(card_positions):
                # Check if region is within screenshot bounds
                if (x + w <= width and y + h <= height and x >= 0 and y >= 0):
                    # Extract region for quality assessment
                    region = screenshot[y:y+h, x:x+w]
                    quality_score = self._assess_region_quality(region)
                    region_qualities.append(quality_score)
                    
                    # Get detection method recommendation
                    recommended_method, method_confidence = self.recommend_optimal_detection_method(region)
                    method_recommendations.append((recommended_method, method_confidence))
                    
                    # Generate optimized regions for each detection method
                    card_optimizations = {}
                    
                    # pHash optimization
                    phash_region = self.optimize_region_for_phash(x, y, w, h, width, height)
                    card_optimizations["phash"] = phash_region
                    
                    # Histogram optimization
                    hist_region = self.optimize_region_for_histogram(x, y, w, h, width, height)
                    card_optimizations["histogram"] = hist_region
                    
                    # Ultimate Detection optimization
                    ultimate_region = self.optimize_region_for_ultimate_detection(x, y, w, h, width, height)
                    card_optimizations["ultimate"] = ultimate_region
                    
                    # Basic region (original)
                    card_optimizations["basic"] = (x, y, w, h)
                    
                    optimized_regions[f"card_{i+1}"] = card_optimizations
                    
                    self.logger.info(f"✅ Card {i+1}: ({x},{y},{w},{h}), quality: {quality_score:.3f}, method: {recommended_method}")
                else:
                    self.logger.warning(f"⚠️ Card {i+1} region out of bounds: {(x, y, w, h)}")
                    region_qualities.append(0.0)
                    method_recommendations.append(("basic", 0.2))
            
            # Calculate overall confidence
            overall_confidence = np.mean(region_qualities) if region_qualities else 0.0
            method_confidences = [conf for _, conf in method_recommendations]
            overall_method_confidence = np.mean(method_confidences) if method_confidences else 0.0
            
            # Compile results
            detection_result = {
                'interface_rect': interface_rect,
                'card_positions': card_positions,
                'detection_method': 'red_area_detection',
                'success': len(card_positions) >= 2,
                'confidence': overall_confidence,
                'region_qualities': region_qualities,
                'card_size_used': self.calculate_optimal_card_size(width, height),
                'method_recommendations': method_recommendations,
                'method_confidence': overall_method_confidence,
                'optimized_regions': optimized_regions,
                'optimization_available': True,
                'stats': {
                    'cards_detected': len(card_positions),
                    'average_quality': overall_confidence,
                    'average_method_confidence': overall_method_confidence,
                    'recommended_methods': [method for method, _ in method_recommendations],
                    'phash_ready_regions': sum(1 for _, conf in method_recommendations if conf > 0.7),
                    'histogram_ready_regions': sum(1 for _, conf in method_recommendations if conf > 0.5),
                }
            }
            
            self.logger.info(f"✅ Detection complete: {len(card_positions)}/3 cards positioned")
            self.logger.info(f"Overall confidence: {overall_confidence:.3f}")
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Error in automatic detection: {e}")
            return None
    
    def save_debug_images(self, screenshot: np.ndarray, detection_result: Dict[str, Any], 
                         output_dir: str = "/home/marcco/arena_bot_project") -> None:
        """Save debug images showing the detection results."""
        try:
            output_path = Path(output_dir)
            
            # Save interface region
            if 'interface_rect' in detection_result:
                ix, iy, iw, ih = detection_result['interface_rect']
                interface_img = screenshot[iy:iy+ih, ix:ix+iw]
                cv2.imwrite(str(output_path / "smart_detected_interface.png"), interface_img)
            
            # Save card regions
            if 'card_positions' in detection_result:
                for i, (x, y, w, h) in enumerate(detection_result['card_positions']):
                    card_img = screenshot[y:y+h, x:x+w]
                    cv2.imwrite(str(output_path / f"smart_card_{i+1}.png"), card_img)
            
            self.logger.info(f"Debug images saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug images: {e}")
    
    def detect_cards_with_hybrid_cascade(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        4-stage hybrid cascade detection system: contour→anchor→red_area→static_fallback
        
        FIXED CASCADE ORDER - Visual analysis methods first, static scaling as last resort:
        1. Contour detection (Magic Card Detector method) - analyzes actual screen content
        2. Anchor positioning (Template-based mana crystals) - precise visual anchoring
        3. Red area detection (Proven automated method) - finds game interface automatically
        4. Static positioning (Arena Helper method) - hardcoded fallback only
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            Dict with best detection results from cascade
        """
        try:
            height, width = screenshot.shape[:2]
            self.logger.info(f"🔄 Starting 4-stage hybrid cascade detection for {width}×{height}")
            
            # Stage 1: Contour Detection (Magic Card Detector method) - VISUAL ANALYSIS FIRST
            self.logger.info("🔍 Stage 1: Contour detection (Magic Card Detector method)")
            contour_result = self.detect_cards_via_contours(screenshot)
            if contour_result and contour_result.get('success') and contour_result.get('confidence', 0) > 0.7:
                # Validate coordinates before accepting result
                if self._validate_coordinate_plausibility(contour_result.get('card_positions', []), width, height):
                    self.logger.info("✅ Stage 1 SUCCESS: Good confidence contour detection")
                    contour_result['cascade_stage'] = 'contour'
                    contour_result['cascade_confidence'] = contour_result.get('confidence', 0)
                    return contour_result
                else:
                    self.logger.warning("⚠️ Stage 1 REJECTED: Contour detection coordinates implausible")
            
            # Stage 2: Anchor Positioning (Template-based mana crystals) - PRECISE VISUAL ANCHORING
            self.logger.info("⚓ Stage 2: Anchor positioning (mana crystal templates)")
            anchor_result = self.detect_cards_via_anchors(screenshot)
            if anchor_result and anchor_result.get('success') and anchor_result.get('confidence', 0) > 0.6:
                # Validate coordinates before accepting result
                if self._validate_coordinate_plausibility(anchor_result.get('card_positions', []), width, height):
                    self.logger.info("✅ Stage 2 SUCCESS: Adequate confidence anchor detection")
                    anchor_result['cascade_stage'] = 'anchor'
                    anchor_result['cascade_confidence'] = anchor_result.get('confidence', 0)
                    return anchor_result
                else:
                    self.logger.warning("⚠️ Stage 2 REJECTED: Anchor detection coordinates implausible")
            
            # Stage 3: Red Area Detection (Proven automated method) - INTERFACE ANALYSIS
            self.logger.info("🚨 Stage 3: Red area detection (proven automated method)")
            red_area_result = self.detect_cards_automatically(screenshot)
            if red_area_result and red_area_result.get('success') and red_area_result.get('confidence', 0) > 0.4:
                # Validate coordinates before accepting result
                if self._validate_coordinate_plausibility(red_area_result.get('card_positions', []), width, height):
                    self.logger.info("✅ Stage 3 SUCCESS: Automated interface detection")
                    red_area_result['cascade_stage'] = 'red_area'
                    red_area_result['cascade_confidence'] = red_area_result.get('confidence', 0)
                    return red_area_result
                else:
                    self.logger.warning("⚠️ Stage 3 REJECTED: Red area detection coordinates implausible")
            
            # Stage 4: Static Positioning (Arena Helper method) - ABSOLUTE LAST RESORT
            self.logger.info("🎯 Stage 4: Static positioning (hardcoded fallback - LAST RESORT)")
            static_result = self.detect_cards_via_static_scaling(screenshot)
            if static_result:
                self.logger.info("⚠️ Stage 4 FALLBACK: Using static scaling as last resort")
                static_result['cascade_stage'] = 'static_fallback'
                static_result['cascade_confidence'] = static_result.get('confidence', 0)
                return static_result
            
            self.logger.error("❌ All 4 cascade stages failed")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in hybrid cascade detection: {e}")
            return None
    
    def detect_cards_via_static_scaling(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Stage 1: Arena Helper static scaling method
        Uses predetermined positions scaled for current resolution with auto-calibration
        """
        try:
            height, width = screenshot.shape[:2]
            
            # Calculate optimal card size for current resolution
            card_width, card_height = self.calculate_optimal_card_size(width, height)
            
            # Standard Arena Helper static positions (scaled from 1920×1080)
            scale_x = width / 1920
            scale_y = height / 1080
            
            # Reference positions from Arena Helper (x, y only - use dynamic card size)
            base_positions = [
                (393, 175),  # Card 1
                (673, 175),  # Card 2  
                (953, 175),  # Card 3
            ]
            
            scaled_positions = []
            for x, y in base_positions:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                # Use calculated optimal card size instead of hardcoded dimensions
                scaled_positions.append((scaled_x, scaled_y, card_width, card_height))
            
            # Apply auto-calibration for this resolution
            calibrated_positions = self.apply_calibration_offsets(scaled_positions, width, height)
            
            # Validate positions are within bounds
            valid_positions = []
            for x, y, w, h in calibrated_positions:
                if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                    valid_positions.append((x, y, w, h))
            
            confidence = len(valid_positions) / 3.0  # 3 expected cards
            
            return {
                'card_positions': valid_positions,
                'detection_method': 'arena_helper_static_scaling_CALIBRATED',
                'success': len(valid_positions) >= 2,
                'confidence': confidence,
                'interface_rect': (0, 0, width, height),  # Full screen
                'card_size_used': (card_width, card_height),  # Include size info for logging
            }
            
        except Exception as e:
            self.logger.error(f"Error in static scaling detection: {e}")
            return None
    
    def detect_cards_via_contours(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Stage 2: Magic Card Detector contour method
        Finds card-shaped regions with aspect ratio validation
        """
        try:
            height, width = screenshot.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Create mask for card-like regions (detect gold borders)
            lower_gold = np.array([15, 50, 100])
            upper_gold = np.array([35, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Find contours
            contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            card_candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Magic Card Detector validation
                if self.validate_card_region(x, y, w, h):
                    score = self.score_card_region(x, y, w, h)
                    card_candidates.append((x, y, w, h, score))
            
            # Sort by score and take top 3
            card_candidates.sort(key=lambda x: x[4], reverse=True)
            card_positions = [(x, y, w, h) for x, y, w, h, _ in card_candidates[:3]]
            
            confidence = min(1.0, len(card_positions) / 3.0)
            
            return {
                'card_positions': card_positions,
                'detection_method': 'magic_card_detector_contours',
                'success': len(card_positions) >= 2,
                'confidence': confidence,
                'interface_rect': (0, 0, width, height),
            }
            
        except Exception as e:
            self.logger.error(f"Error in contour detection: {e}")
            return None
    
    def detect_cards_via_anchors(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Stage 3: Template-based anchor positioning
        Uses mana crystal templates to position cards precisely
        """
        try:
            # Use existing mana anchor detection
            anchor_positions = self.detect_cards_via_mana_anchors(screenshot)
            
            if not anchor_positions:
                return None
            
            height, width = screenshot.shape[:2]
            confidence = min(1.0, len(anchor_positions) / 3.0)
            
            return {
                'card_positions': anchor_positions,
                'detection_method': 'template_anchor_positioning',
                'success': len(anchor_positions) >= 2,
                'confidence': confidence,
                'interface_rect': (0, 0, width, height),
            }
            
        except Exception as e:
            self.logger.error(f"Error in anchor detection: {e}")
            return None
    
    def detect_cards_simple_working(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Simple, reliable detection method that actually works.
        Uses your exact successful coordinates from debug images.
        """
        try:
            height, width = screenshot.shape[:2]
            self.logger.info(f"Simple detection for {width}×{height} resolution")
            
            # For 3440×1440 ultrawide (your resolution)
            if width >= 3440:
                # Based on your successful detection logs - these coordinates work
                card_positions = [
                    (704, 233, 447, 493),   # Card 1 
                    (1205, 233, 447, 493),  # Card 2
                    (1707, 233, 447, 493),  # Card 3  
                ]
            else:
                # Scale for other resolutions  
                scale_x = width / 3440
                scale_y = height / 1440
                
                card_positions = [
                    (int(704 * scale_x), int(233 * scale_y), int(447 * scale_x), int(493 * scale_y)),
                    (int(1205 * scale_x), int(233 * scale_y), int(447 * scale_x), int(493 * scale_y)),
                    (int(1707 * scale_x), int(233 * scale_y), int(447 * scale_x), int(493 * scale_y)),
                ]
            
            # Validate positions are within bounds
            valid_positions = []
            for x, y, w, h in card_positions:
                if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                    valid_positions.append((x, y, w, h))
            
            confidence = len(valid_positions) / 3.0
            
            # Generate optimization info for each card (crucial for pHash performance)
            optimized_regions = {}
            for i, (x, y, w, h) in enumerate(valid_positions):
                card_optimizations = {}
                
                # pHash optimization: ensure 300×420+ minimum
                phash_w = max(w, 350)  # Target 350×500 for optimal pHash
                phash_h = max(h, 500)
                phash_x = max(0, x - (phash_w - w) // 2)
                phash_y = max(0, y - (phash_h - h) // 2)
                phash_x = min(phash_x, width - phash_w)
                phash_y = min(phash_y, height - phash_h)
                card_optimizations["phash"] = (phash_x, phash_y, phash_w, phash_h)
                
                # Other method optimizations
                card_optimizations["histogram"] = (x, y, w, h)  # Original region fine for histogram
                card_optimizations["ultimate"] = (x, y, w, h)   # Original region fine for ultimate
                card_optimizations["basic"] = (x, y, w, h)      # Original region
                
                optimized_regions[f"card_{i+1}"] = card_optimizations
            
            return {
                'card_positions': valid_positions,
                'detection_method': 'simple_working_method',
                'success': len(valid_positions) >= 2,
                'confidence': confidence,
                'interface_rect': (0, 0, width, height),
                'card_size_used': (447, 493),  # Known working size
                'optimization_available': True,
                'optimized_regions': optimized_regions,
                'method_recommendations': [("phash", 0.9)] * len(valid_positions),
                'method_confidence': 0.9,
                'stats': {
                    'phash_ready_regions': len(valid_positions),
                    'recommended_methods': ["phash"] * len(valid_positions),
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in simple working detection: {e}")
            return None
    
    def _validate_coordinate_plausibility(self, card_positions: List[Tuple[int, int, int, int]], 
                                        screen_width: int, screen_height: int) -> bool:
        """
        Validate that detected card positions are plausible for Hearthstone Arena.
        
        More robust validation that works with windowed mode and different resolutions.
        
        Args:
            card_positions: List of (x, y, w, h) card positions
            screen_width: Screen width
            screen_height: Screen height
            
        Returns:
            True if coordinates are plausible, False otherwise
        """
        if not card_positions or len(card_positions) < 2:
            return False  # Must find at least 2 cards

        # Check 1: All cards must be within the screen bounds and have a valid size
        for x, y, w, h in card_positions:
            if not (0 <= x < screen_width and 0 <= y < screen_height and 
                   x + w <= screen_width and y + h <= screen_height and w > 100 and h > 100):
                self.logger.warning(f"Plausibility check failed: Position ({x},{y},{w},{h}) is out of bounds or too small.")
                return False

        # Check 2: Cards should be roughly horizontally aligned
        avg_y = sum(p[1] for p in card_positions) / len(card_positions)
        if any(abs(p[1] - avg_y) > (screen_height * 0.1) for p in card_positions):
            self.logger.warning(f"Plausibility check failed: Cards are not horizontally aligned.")
            return False

        # Check 3: Cards should not overlap
        sorted_pos = sorted(card_positions, key=lambda p: p[0])
        for i in range(len(sorted_pos) - 1):
            if sorted_pos[i][0] + sorted_pos[i][2] > sorted_pos[i+1][0]:
                self.logger.warning(f"Plausibility check failed: Cards are overlapping.")
                return False
        
        self.logger.info("✅ Coordinate plausibility check PASSED.")
        return True


def benchmark_detection_methods(screenshot_path: str = None) -> Dict[str, Any]:
    """
    Benchmark all detection methods for performance and accuracy comparison.
    
    Args:
        screenshot_path: Path to test screenshot (optional)
        
    Returns:
        Dict with benchmark results for all methods
    """
    import time
    import cv2
    
    detector = SmartCoordinateDetector()
    
    # Use debug screenshot if available, otherwise create test data
    if screenshot_path and Path(screenshot_path).exists():
        screenshot = cv2.imread(screenshot_path)
    else:
        # Create test screenshot (3440x1440 ultrawide)
        screenshot = np.zeros((1440, 3440, 3), dtype=np.uint8)
    
    methods_to_test = [
        ("Enhanced Auto", "detect_cards_automatically"),
        ("Hybrid Cascade", "detect_cards_with_hybrid_cascade"),
        ("Static Scaling", "detect_cards_via_static_scaling"),
        ("Contour Detection", "detect_cards_via_contours"),
        ("Anchor Detection", "detect_cards_via_anchors"),
    ]
    
    benchmark_results = {
        'test_resolution': f"{screenshot.shape[1]}x{screenshot.shape[0]}",
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'methods': {}
    }
    
    for method_name, method_func in methods_to_test:
        try:
            # Time the detection
            start_time = time.time()
            
            method = getattr(detector, method_func)
            result = method(screenshot)
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Analyze results
            success = result and result.get('success', False)
            confidence = result.get('confidence', 0.0) if result else 0.0
            cards_detected = len(result.get('card_positions', [])) if result else 0
            cascade_stage = result.get('cascade_stage', 'N/A') if result else 'N/A'
            
            benchmark_results['methods'][method_name] = {
                'execution_time_ms': round(execution_time, 2),
                'success': success,
                'confidence': round(confidence, 3),
                'cards_detected': cards_detected,
                'cascade_stage': cascade_stage,
                'method_function': method_func
            }
            
            print(f"✅ {method_name}: {execution_time:.2f}ms, success={success}, confidence={confidence:.3f}")
            
        except Exception as e:
            benchmark_results['methods'][method_name] = {
                'execution_time_ms': 0,
                'success': False,
                'confidence': 0.0,
                'cards_detected': 0,
                'cascade_stage': 'ERROR',
                'error': str(e),
                'method_function': method_func
            }
            print(f"❌ {method_name}: Error - {e}")
    
    # Calculate performance summary
    successful_methods = [m for m in benchmark_results['methods'].values() if m['success']]
    if successful_methods:
        avg_time = sum(m['execution_time_ms'] for m in successful_methods) / len(successful_methods)
        avg_confidence = sum(m['confidence'] for m in successful_methods) / len(successful_methods)
        
        benchmark_results['summary'] = {
            'successful_methods': len(successful_methods),
            'total_methods': len(methods_to_test),
            'success_rate': len(successful_methods) / len(methods_to_test),
            'average_execution_time_ms': round(avg_time, 2),
            'average_confidence': round(avg_confidence, 3),
            'fastest_method': min(successful_methods, key=lambda x: x['execution_time_ms']),
            'highest_confidence_method': max(successful_methods, key=lambda x: x['confidence'])
        }
    
    return benchmark_results


def get_smart_coordinate_detector() -> SmartCoordinateDetector:
    """Get the global smart coordinate detector instance."""
    return SmartCoordinateDetector()