"""
Card recognition system combining all detection methods.

Integrates screen capture, histogram matching, template matching, and validation.
Based on Arena Tracker's proven 3-card detection pipeline.
"""

import cv2
import numpy as np
import logging
import time
import asyncio
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# Import S-Tier logging compatibility layer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from logging_compatibility import get_logger, get_async_logger

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
        """Initialize card recognizer with S-Tier logging."""
        # Initialize S-Tier compatible logger
        self.logger = get_logger(__name__)
        self.async_logger = None  # Will be initialized in async context
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_detection_time_ms': 0.0,
            'peak_detection_time_ms': 0.0
        }
        
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
        
        # Log initialization with rich context
        self.logger.info("🎯 CardRecognizer initialized with S-Tier logging", extra={
            'initialization_context': {
                'card_positions': len(self.CARD_POSITIONS),
                'mana_regions': len(self.MANA_REGIONS),
                'rarity_regions': len(self.RARITY_REGIONS),
                'components_loaded': [
                    'screen_detector',
                    'histogram_matcher', 
                    'template_matcher',
                    'validation_engine',
                    'asset_loader'
                ]
            }
        })
    
    def initialize(self) -> bool:
        """
        Initialize the card recognition system with enhanced S-Tier logging.
        
        Returns:
            True if successful, False otherwise
        """
        initialization_start = time.perf_counter()
        
        try:
            # Load card database for histogram matching
            self.logger.info("📂 Loading card database for recognition system", extra={
                'loading_context': {
                    'phase': 'card_database_loading',
                    'start_time': initialization_start
                }
            })
            
            available_cards = self.asset_loader.get_available_cards()
            
            if not available_cards:
                self.logger.error("❌ No card images found in assets", extra={
                    'error_context': {
                        'asset_loader_available': hasattr(self, 'asset_loader'),
                        'initialization_failed': True
                    }
                })
                return False
            
            # Load a subset of cards for testing (load all later)
            card_images = {}
            loading_start = time.perf_counter()
            
            for i, card_code in enumerate(available_cards[:100]):  # Load first 100 for testing
                image = self.asset_loader.load_card_image(card_code)
                if image is not None:
                    card_images[card_code] = image
                
                # Also load premium version if available
                premium_image = self.asset_loader.load_card_image(card_code, premium=True)
                if premium_image is not None:
                    card_images[f"{card_code}_premium"] = premium_image
            
            loading_time = (time.perf_counter() - loading_start) * 1000
            
            # Load histograms
            histogram_start = time.perf_counter()
            self.histogram_matcher.load_card_database(card_images)
            histogram_time = (time.perf_counter() - histogram_start) * 1000
            
            self.logger.info("📊 Card database loaded successfully", extra={
                'database_context': {
                    'total_cards_available': len(available_cards),
                    'cards_loaded': len(card_images),
                    'loading_time_ms': loading_time,
                    'histogram_processing_time_ms': histogram_time,
                    'load_percentage': (len(card_images) / len(available_cards)) * 100
                }
            })
            
            # Load templates
            template_start = time.perf_counter()
            mana_templates = self.asset_loader.load_mana_templates()
            rarity_templates = self.asset_loader.load_rarity_templates()
            
            self.template_matcher.load_mana_templates(mana_templates)
            self.template_matcher.load_rarity_templates(rarity_templates)
            
            mana_count, rarity_count = self.template_matcher.get_template_counts()
            template_time = (time.perf_counter() - template_start) * 1000
            
            self.logger.info("🔧 Templates loaded successfully", extra={
                'template_context': {
                    'mana_templates': mana_count,
                    'rarity_templates': rarity_count,
                    'template_loading_time_ms': template_time
                }
            })
            
            # Calculate total initialization time
            total_initialization_time = (time.perf_counter() - initialization_start) * 1000
            
            self.is_initialized = True
            
            self.logger.info("✅ Card recognition system initialized successfully", extra={
                'initialization_summary': {
                    'total_time_ms': total_initialization_time,
                    'cards_loaded': len(card_images),
                    'mana_templates': mana_count,
                    'rarity_templates': rarity_count,
                    'performance_target_met': total_initialization_time < 5000,  # 5 second target
                    'system_ready': True
                }
            })
            
            return True
            
        except Exception as e:
            initialization_time = (time.perf_counter() - initialization_start) * 1000
            
            self.logger.error("❌ Failed to initialize card recognition system", extra={
                'initialization_error': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'initialization_time_ms': initialization_time,
                    'available_cards_count': len(available_cards) if 'available_cards' in locals() else 0,
                    'cards_loaded_count': len(card_images) if 'card_images' in locals() else 0
                }
            }, exc_info=True)
            
            return False
    
    async def initialize_async(self) -> bool:
        """
        Initialize the card recognition system with async S-Tier logging.
        
        Returns:
            True if successful, False otherwise
        """
        # Initialize async logger
        if self.async_logger is None:
            self.async_logger = await get_async_logger(__name__)
        
        initialization_start = time.perf_counter()
        
        try:
            await self.async_logger.ainfo("🚀 Starting async card recognition system initialization", extra={
                'async_initialization_context': {
                    'phase': 'async_startup',
                    'start_time': initialization_start,
                    'logging_system': 's_tier_async'
                }
            })
            
            # Use sync initialization but with async logging
            success = self.initialize()
            
            total_time = (time.perf_counter() - initialization_start) * 1000
            
            if success:
                await self.async_logger.ainfo("✅ Async card recognition system ready", extra={
                    'async_initialization_result': {
                        'success': True,
                        'total_time_ms': total_time,
                        'async_logging_active': True,
                        'performance_mode': 'enterprise'
                    }
                })
            else:
                await self.async_logger.aerror("❌ Async card recognition system initialization failed", extra={
                    'async_initialization_result': {
                        'success': False,
                        'total_time_ms': total_time,
                        'async_logging_active': True
                    }
                })
            
            return success
            
        except Exception as e:
            await self.async_logger.aerror("💥 Async initialization crashed", extra={
                'async_crash_context': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'initialization_time_ms': (time.perf_counter() - initialization_start) * 1000
                }
            }, exc_info=True)
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