#!/usr/bin/env python3
"""
Automatic arena detection system combining all methods.
Provides a unified interface for automatic card detection.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from .surf_detector import get_surf_detector
from ..detection.histogram_matcher import get_histogram_matcher
from ..utils.asset_loader import get_asset_loader

logger = logging.getLogger(__name__)

class AutoDetector:
    """
    Unified automatic detection system for Hearthstone arena drafts.
    Combines SURF-based interface detection with histogram-based card recognition.
    """
    
    def __init__(self):
        """Initialize the auto detector with all necessary components."""
        self.surf_detector = get_surf_detector()
        self.histogram_matcher = get_histogram_matcher()
        self.asset_loader = get_asset_loader()
        
        # Load card database on initialization
        self.card_database_loaded = False
        self._load_card_database()
    
    def _load_card_database(self):
        """Load the complete card database for recognition."""
        try:
            logger.info("Loading card database...")
            available_cards = self.asset_loader.get_available_cards()
            
            # Filter out HERO cards which don't exist as premium
            actual_cards = [card for card in available_cards if not card.startswith('HERO_')]
            logger.info(f"Filtering {len(available_cards)} cards -> {len(actual_cards)} non-hero cards")
            
            card_hists = {}
            
            for card_code in actual_cards:
                for is_premium in [False, True]:
                    card_image = self.asset_loader.load_card_image(card_code, premium=is_premium)
                    if card_image is not None:
                        # Extract Arena Tracker's 80x80 region
                        at_region = self._extract_arena_tracker_region(card_image, is_premium)
                        if at_region is not None:
                            hist = self._compute_arena_tracker_histogram(at_region)
                            hist_key = f"{card_code}{'_premium' if is_premium else ''}"
                            card_hists[hist_key] = hist
            
            self.histogram_matcher.load_card_database(card_hists)
            self.card_database_loaded = True
            logger.info(f"Loaded {len(card_hists)} card histograms")
            
        except Exception as e:
            logger.error(f"Error loading card database: {e}")
            self.card_database_loaded = False
    
    def _extract_arena_tracker_region(self, card_image: np.ndarray, is_premium: bool = False) -> Optional[np.ndarray]:
        """Extract Arena Tracker's 80x80 region from card image."""
        if is_premium:
            x, y, w, h = 57, 71, 80, 80
        else:
            x, y, w, h = 60, 71, 80, 80
        
        if (card_image.shape[1] < x + w) or (card_image.shape[0] < y + h):
            return None
        
        return card_image[y:y+h, x:x+w]
    
    def _compute_arena_tracker_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute Arena Tracker's exact histogram."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        h_bins = 50
        s_bins = 60
        hist_size = [h_bins, s_bins]
        ranges = [0, 180, 0, 256]
        channels = [0, 1]
        
        hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def detect_arena_cards(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Automatically detect arena interface and identify the 3 draft cards.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            Dict containing detection results or None if failed
        """
        if not self.card_database_loaded:
            logger.error("Card database not loaded")
            return None
        
        try:
            # Step 1: Detect arena interface automatically
            interface_rect = self.surf_detector.detect_arena_interface(screenshot)
            if interface_rect is None:
                logger.warning("Could not detect arena interface")
                return None
            
            logger.info(f"Arena interface detected: {interface_rect}")
            
            # Step 2: Calculate card positions within interface
            card_positions = self.surf_detector.calculate_card_positions(interface_rect)
            logger.info(f"Calculated {len(card_positions)} card positions")
            
            # Step 3: Extract and identify each card
            detected_cards = []
            
            for i, (x, y, w, h) in enumerate(card_positions):
                logger.info(f"Processing card {i+1} at ({x}, {y}, {w}, {h})")
                
                # Extract card from screenshot
                card_image = screenshot[y:y+h, x:x+w]
                
                if card_image.size == 0:
                    logger.warning(f"Empty card region {i+1}")
                    continue
                
                # Identify the card using our proven center crop strategy
                card_result = self._identify_card(card_image, position=i+1)
                if card_result:
                    card_result['position'] = i + 1
                    card_result['coordinates'] = (x, y, w, h)
                    detected_cards.append(card_result)
                else:
                    logger.warning(f"Could not identify card {i+1}")
            
            # Compile results
            detection_result = {
                'interface_rect': interface_rect,
                'card_positions': card_positions,
                'detected_cards': detected_cards,
                'success': len(detected_cards) > 0,
                'accuracy': len(detected_cards) / len(card_positions) if card_positions else 0
            }
            
            logger.info(f"Detection complete: {len(detected_cards)}/{len(card_positions)} cards identified")
            return detection_result
            
        except Exception as e:
            logger.error(f"Error in auto detection: {e}")
            return None
    
    def _identify_card(self, card_image: np.ndarray, position: int) -> Optional[Dict[str, Any]]:
        """
        Identify a single card using histogram matching.
        
        Args:
            card_image: Extracted card image
            position: Card position (1, 2, or 3)
            
        Returns:
            Dict with card identification results or None
        """
        try:
            # Use center crop strategy (proven to work best)
            h, w = card_image.shape[:2]
            if h >= 60 and w >= 60:
                processed_region = card_image[30:h-30, 30:w-30]
            else:
                processed_region = card_image
            
            # Resize to 80x80 for Arena Tracker comparison
            if processed_region.shape[:2] != (80, 80):
                resized = cv2.resize(processed_region, (80, 80), interpolation=cv2.INTER_AREA)
            else:
                resized = processed_region
            
            # Compute histogram
            screen_hist = self._compute_arena_tracker_histogram(resized)
            
            # Find best match using histogram matcher
            match_result = self.histogram_matcher.find_best_match_with_confidence(screen_hist)
            
            if match_result:
                return {
                    'card_code': match_result['card_code'],
                    'confidence': match_result['confidence'],
                    'distance': match_result['distance'],
                    'is_premium': match_result['is_premium'],
                    'processing_strategy': 'center_crop'
                }
            else:
                logger.warning(f"No confident match for card {position}")
                return None
                
        except Exception as e:
            logger.error(f"Error identifying card {position}: {e}")
            return None
    
    def detect_single_screenshot(self, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """
        Convenience method to detect cards from a screenshot file.
        
        Args:
            screenshot_path: Path to screenshot file
            
        Returns:
            Detection results or None
        """
        try:
            screenshot = cv2.imread(screenshot_path)
            if screenshot is None:
                logger.error(f"Could not load screenshot: {screenshot_path}")
                return None
            
            return self.detect_arena_cards(screenshot)
            
        except Exception as e:
            logger.error(f"Error processing screenshot {screenshot_path}: {e}")
            return None


def get_auto_detector() -> AutoDetector:
    """Get the global auto detector instance."""
    return AutoDetector()