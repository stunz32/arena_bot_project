"""
Card recognition system combining all detection methods.

Integrates screen capture, histogram matching, template matching, and validation.
Based on Arena Tracker's proven 3-card detection pipeline.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .screen_detector import get_screen_detector
from ..detection.histogram_matcher import get_histogram_matcher, CardMatch
from ..detection.template_matcher import get_template_matcher
from ..detection.validation_engine import get_validation_engine, ValidationResult
from ..utils.asset_loader import get_asset_loader


@dataclass
class CardDetectionResult:
    """Container for card detection results."""
    card_code: Optional[str]
    confidence: float
    mana_cost: Optional[int]
    rarity: Optional[int]
    is_premium: bool
    validation_result: Optional[ValidationResult]
    position: int  # 0, 1, 2 for left, center, right


class CardRecognizer:
    """
    Main card recognition system.
    
    Combines screen capture, histogram matching, template matching,
    and validation using Arena Tracker's proven approach.
    """
    
    def __init__(self):
        """Initialize card recognizer."""
        self.logger = logging.getLogger(__name__)
        
        # Component instances
        self.screen_detector = get_screen_detector()
        self.histogram_matcher = get_histogram_matcher()
        self.template_matcher = get_template_matcher()
        self.validation_engine = get_validation_engine()
        self.asset_loader = get_asset_loader()
        
        # Arena Tracker's card positions (percentage of screen)
        self.CARD_POSITIONS = [
            {"x": 0.139, "y": 0.251, "width": 0.194, "height": 0.498},  # Left
            {"x": 0.403, "y": 0.251, "width": 0.194, "height": 0.498},  # Center
            {"x": 0.667, "y": 0.251, "width": 0.194, "height": 0.498}   # Right
        ]
        
        # Mana cost regions (relative to card)
        self.MANA_REGIONS = [
            {"x": 0.05, "y": 0.08, "width": 0.15, "height": 0.15},
            {"x": 0.05, "y": 0.08, "width": 0.15, "height": 0.15},
            {"x": 0.05, "y": 0.08, "width": 0.15, "height": 0.15}
        ]
        
        # Rarity regions (relative to card)
        self.RARITY_REGIONS = [
            {"x": 0.4, "y": 0.85, "width": 0.2, "height": 0.1},
            {"x": 0.4, "y": 0.85, "width": 0.2, "height": 0.1},
            {"x": 0.4, "y": 0.85, "width": 0.2, "height": 0.1}
        ]
        
        # Detection state
        self.is_initialized = False
        self.last_detection_results: List[CardDetectionResult] = []
        
        self.logger.info("CardRecognizer initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the card recognition system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load card database for histogram matching
            self.logger.info("Loading card database...")
            available_cards = self.asset_loader.get_available_cards()
            
            if not available_cards:
                self.logger.error("No card images found in assets")
                return False
            
            # Load a subset of cards for testing (load all later)
            card_images = {}
            for i, card_code in enumerate(available_cards[:100]):  # Load first 100 for testing
                image = self.asset_loader.load_card_image(card_code)
                if image is not None:
                    card_images[card_code] = image
                
                # Also load premium version if available
                premium_image = self.asset_loader.load_card_image(card_code, premium=True)
                if premium_image is not None:
                    card_images[f"{card_code}_premium"] = premium_image
            
            # Load histograms
            self.histogram_matcher.load_card_database(card_images)
            self.logger.info(f"Loaded {len(card_images)} card images")
            
            # Load templates
            mana_templates = self.asset_loader.load_mana_templates()
            rarity_templates = self.asset_loader.load_rarity_templates()
            
            self.template_matcher.load_mana_templates(mana_templates)
            self.template_matcher.load_rarity_templates(rarity_templates)
            
            mana_count, rarity_count = self.template_matcher.get_template_counts()
            self.logger.info(f"Loaded {mana_count} mana templates, {rarity_count} rarity templates")
            
            self.is_initialized = True
            self.logger.info("Card recognition system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize card recognition system: {e}")
            return False
    
    def extract_card_regions(self, screen_image: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Extract 3 card regions from screen capture.
        
        Args:
            screen_image: Full screen capture
            
        Returns:
            List of 3 card images (or None if extraction failed)
        """
        card_regions = []
        screen_height, screen_width = screen_image.shape[:2]
        
        for i, pos in enumerate(self.CARD_POSITIONS):
            try:
                # Calculate absolute coordinates
                x = int(pos["x"] * screen_width)
                y = int(pos["y"] * screen_height)
                width = int(pos["width"] * screen_width)
                height = int(pos["height"] * screen_height)
                
                # Extract region with bounds checking
                x = max(0, min(x, screen_width - width))
                y = max(0, min(y, screen_height - height))
                
                card_region = screen_image[y:y+height, x:x+width]
                
                if card_region.size > 0:
                    card_regions.append(card_region)
                    self.logger.debug(f"Extracted card {i}: {card_region.shape} at ({x}, {y})")
                else:
                    card_regions.append(None)
                    self.logger.warning(f"Failed to extract card {i}")
                    
            except Exception as e:
                self.logger.error(f"Error extracting card {i}: {e}")
                card_regions.append(None)
        
        return card_regions
    
    def extract_mana_rarity_regions(self, card_image: np.ndarray, 
                                   position: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract mana cost and rarity regions from card image.
        
        Args:
            card_image: Card image
            position: Card position (0, 1, 2)
            
        Returns:
            Tuple of (mana_region, rarity_region)
        """
        if position >= len(self.MANA_REGIONS):
            return None, None
        
        card_height, card_width = card_image.shape[:2]
        
        # Extract mana region
        mana_region = None
        try:
            mana_pos = self.MANA_REGIONS[position]
            mx = int(mana_pos["x"] * card_width)
            my = int(mana_pos["y"] * card_height)
            mw = int(mana_pos["width"] * card_width)
            mh = int(mana_pos["height"] * card_height)
            
            mana_region = card_image[my:my+mh, mx:mx+mw]
            
        except Exception as e:
            self.logger.error(f"Error extracting mana region: {e}")
        
        # Extract rarity region
        rarity_region = None
        try:
            rarity_pos = self.RARITY_REGIONS[position]
            rx = int(rarity_pos["x"] * card_width)
            ry = int(rarity_pos["y"] * card_height)
            rw = int(rarity_pos["width"] * card_width)
            rh = int(rarity_pos["height"] * card_height)
            
            rarity_region = card_image[ry:ry+rh, rx:rx+rw]
            
        except Exception as e:
            self.logger.error(f"Error extracting rarity region: {e}")
        
        return mana_region, rarity_region
    
    def detect_cards(self, screen_index: int = None) -> List[CardDetectionResult]:
        """
        Detect cards from screen capture.
        
        Args:
            screen_index: Screen index to capture from
            
        Returns:
            List of CardDetectionResult objects
        """
        if not self.is_initialized:
            self.logger.error("Card recognition system not initialized")
            return []
        
        # Capture screen
        screen_image = self.screen_detector.capture_screen(screen_index)
        if screen_image is None:
            self.logger.error("Failed to capture screen")
            return []
        
        # Extract card regions
        card_regions = self.extract_card_regions(screen_image)
        
        # Process each card
        results = []
        for i, card_image in enumerate(card_regions):
            if card_image is None:
                results.append(CardDetectionResult(
                    card_code=None,
                    confidence=0.0,
                    mana_cost=None,
                    rarity=None,
                    is_premium=False,
                    validation_result=None,
                    position=i
                ))
                continue
            
            # Detect card using histogram matching
            card_match = self.histogram_matcher.match_card(card_image)
            
            if card_match is None:
                results.append(CardDetectionResult(
                    card_code=None,
                    confidence=0.0,
                    mana_cost=None,
                    rarity=None,
                    is_premium=False,
                    validation_result=None,
                    position=i
                ))
                continue
            
            # Extract mana and rarity regions
            mana_region, rarity_region = self.extract_mana_rarity_regions(card_image, i)
            
            # Validate with template matching
            validation_result = self.validation_engine.validate_card_detection(
                card_match, mana_region, rarity_region
            )
            
            result = CardDetectionResult(
                card_code=card_match.card_code,
                confidence=card_match.confidence,
                mana_cost=validation_result.mana_cost,
                rarity=validation_result.rarity,
                is_premium=card_match.is_premium,
                validation_result=validation_result,
                position=i
            )
            
            results.append(result)
            self.logger.debug(f"Card {i} detected: {result.card_code} (confidence: {result.confidence:.3f})")
        
        self.last_detection_results = results
        return results
    
    def get_detection_stats(self) -> Dict[str, any]:
        """
        Get detection system statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        return {
            "is_initialized": self.is_initialized,
            "histogram_database_size": self.histogram_matcher.get_database_size(),
            "template_counts": self.template_matcher.get_template_counts(),
            "last_detection_count": len(self.last_detection_results),
            "screen_count": self.screen_detector.get_screen_count()
        }


# Global card recognizer instance
_card_recognizer = None


def get_card_recognizer() -> CardRecognizer:
    """
    Get the global card recognizer instance.
    
    Returns:
        CardRecognizer instance
    """
    global _card_recognizer
    if _card_recognizer is None:
        _card_recognizer = CardRecognizer()
    return _card_recognizer