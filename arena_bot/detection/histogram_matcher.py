"""
Histogram-based card matching system.

Direct port of Arena Tracker's proven histogram computation and matching algorithms.
Uses HSV color space and Bhattacharyya distance for robust card recognition.
Enhanced with high-performance caching and tiered loading for arena optimization.
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from ..utils.histogram_cache import get_histogram_cache_manager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


@dataclass
class CardMatch:
    """Container for card match results."""
    card_code: str
    distance: float
    is_premium: bool
    confidence: float


class HistogramMatcher:
    """
    Histogram-based card matching using Arena Tracker's proven method.
    
    Uses HSV color space with 50x60 bins and Bhattacharyya distance comparison.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize histogram matcher.
        
        Args:
            use_cache: Enable histogram caching for fast loading
        """
        self.logger = logging.getLogger(__name__)
        
        # Arena Tracker's exact histogram parameters
        self.H_BINS = 50      # Hue bins (0-180 degrees)
        self.S_BINS = 60      # Saturation bins (0-255)
        self.hist_size = [self.H_BINS, self.S_BINS]
        
        # HSV ranges (flattened format for newer OpenCV)
        self.h_ranges = [0, 180]
        self.s_ranges = [0, 256]
        self.ranges = self.h_ranges + self.s_ranges
        
        # Use H and S channels only (ignore V for illumination invariance)
        self.channels = [0, 1]
        
        # Card histogram database
        self.card_histograms: Dict[str, np.ndarray] = {}
        
        # Caching system
        self.use_cache = use_cache and CACHE_AVAILABLE
        self.cache_manager = None
        
        if self.use_cache:
            try:
                self.cache_manager = get_histogram_cache_manager()
                self.logger.info("✅ Cache system enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ Cache system failed to initialize: {e}")
                self.use_cache = False
        
        # Tier tracking
        self.loaded_tiers: Set[str] = set()
        self.tier_load_times: Dict[str, float] = {}
        
        self.logger.info("HistogramMatcher initialized with Arena Tracker's parameters")
        self.logger.info(f"Histogram bins: {self.H_BINS}x{self.S_BINS}")
        self.logger.info(f"Cache enabled: {self.use_cache}")
    
    def compute_histogram(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute HSV histogram for an image.
        
        Direct port of Arena Tracker's getHist() function.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Normalized histogram or None if failed
        """
        try:
            # Convert BGR to HSV (Arena Tracker's method)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Compute 2D histogram using Arena Tracker's parameters
            hist = cv2.calcHist(
                [hsv],                    # Images
                self.channels,            # Channels [0, 1] (H, S)
                None,                     # Mask
                self.hist_size,           # Histogram size [50, 60]
                self.ranges,              # Ranges [[0, 180], [0, 256]]
                accumulate=False
            )
            
            # Normalize histogram (Arena Tracker's method)
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Histogram computation failed: {e}")
            return None
    
    def compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compare two histograms using Bhattacharyya distance.
        
        Direct port of Arena Tracker's comparison method.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            Bhattacharyya distance (0 = identical, 1 = completely different)
        """
        try:
            # Use Bhattacharyya distance (Arena Tracker's method)
            distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            return distance
            
        except Exception as e:
            self.logger.error(f"Histogram comparison failed: {e}")
            return 1.0  # Return maximum distance on error
    
    def add_card_histogram(self, card_code: str, image: np.ndarray, is_premium: bool = False):
        """
        Add a card histogram to the database.
        
        Args:
            card_code: Hearthstone card code
            image: Card image
            is_premium: Whether this is a premium (golden) card
        """
        hist = self.compute_histogram(image)
        
        if hist is not None:
            key = f"{card_code}_premium" if is_premium else card_code
            self.card_histograms[key] = hist
            self.logger.debug(f"Added histogram for {key}")
        else:
            self.logger.warning(f"Failed to compute histogram for {card_code}")
    
    def load_card_database(self, card_images: Dict[str, np.ndarray]):
        """
        Load card images and compute histograms.
        
        Args:
            card_images: Dictionary mapping card codes to images
        """
        self.logger.info(f"Loading card database with {len(card_images)} cards")
        
        for card_code, image in card_images.items():
            is_premium = card_code.endswith("_premium")
            base_code = card_code.replace("_premium", "")
            
            self.add_card_histogram(base_code, image, is_premium)
        
        self.logger.info(f"Card database loaded with {len(self.card_histograms)} histograms")
    
    def load_card_database_from_histograms(self, card_histograms: Dict[str, np.ndarray]):
        """
        Load card database from pre-computed histograms.
        
        This method is used for arena priority functionality where histograms
        are pre-computed from arena-eligible cards only.
        
        Args:
            card_histograms: Dictionary mapping card codes to pre-computed histograms
        """
        self.logger.info(f"Loading card database from pre-computed histograms: {len(card_histograms)} cards")
        
        self.card_histograms.clear()
        
        for card_code, histogram in card_histograms.items():
            is_premium = card_code.endswith("_premium")
            base_code = card_code.replace("_premium", "")
            
            # Store histogram directly (already computed)
            key = (base_code, is_premium)
            self.card_histograms[key] = histogram
        
        self.logger.info(f"✅ Card database loaded from histograms: {len(self.card_histograms)} total")
    
    def find_best_matches(self, query_histogram: np.ndarray, 
                         max_candidates: int = 15) -> List[CardMatch]:
        """
        Find best matching cards for a query histogram.
        
        Port of Arena Tracker's mapBestMatchingCodes() function.
        
        Args:
            query_histogram: Histogram to match against
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of CardMatch objects sorted by distance (best first)
        """
        matches = []
        
        for card_key, card_hist in self.card_histograms.items():
            # Calculate Bhattacharyya distance
            distance = self.compare_histograms(query_histogram, card_hist)
            
            # Parse card code and premium status
            is_premium = card_key.endswith("_premium")
            card_code = card_key.replace("_premium", "")
            
            # Calculate confidence (inverse of distance)
            confidence = 1.0 - distance
            
            match = CardMatch(
                card_code=card_code,
                distance=distance,
                is_premium=is_premium,
                confidence=confidence
            )
            
            matches.append(match)
        
        # Sort by distance (best matches first)
        matches.sort(key=lambda x: x.distance)
        
        # Limit to max candidates (Arena Tracker's approach)
        matches = matches[:max_candidates]
        
        if matches:
            self.logger.debug(f"Found {len(matches)} matches, best distance: {matches[0].distance:.4f}")
        else:
            self.logger.debug("No matches found")
        
        return matches
    
    def match_card(self, image: np.ndarray, confidence_threshold: float = 0.35) -> Optional[CardMatch]:
        """
        Match a single card image against the database.
        
        Args:
            image: Card image to match
            confidence_threshold: Minimum confidence for valid match
            
        Returns:
            Best CardMatch or None if no good match found
        """
        # Compute histogram for the query image
        query_hist = self.compute_histogram(image)
        
        if query_hist is None:
            self.logger.warning("Failed to compute histogram for query image")
            return None
        
        # Find best matches
        matches = self.find_best_matches(query_hist, max_candidates=5)
        
        if not matches:
            self.logger.warning("No matches found in database")
            return None
        
        best_match = matches[0]
        
        # Check if match meets confidence threshold (Arena Tracker's approach)
        if best_match.confidence >= confidence_threshold:
            self.logger.debug(f"Card matched: {best_match.card_code} (distance: {best_match.distance:.4f})")
            return best_match
        else:
            self.logger.debug(f"No confident match found (best distance: {best_match.distance:.4f})")
            return None
    
    def get_histogram_params(self) -> Dict[str, Any]:
        """
        Get current histogram parameters for cache compatibility.
        
        Returns:
            Dictionary with histogram computation parameters
        """
        return {
            'h_bins': self.H_BINS,
            's_bins': self.S_BINS,
            'h_ranges': self.h_ranges,
            's_ranges': self.s_ranges,
            'channels': self.channels,
            'algorithm': 'arena_tracker_hsv_bhattacharyya'
        }
    
    def load_card_database_cached(self, card_images: Dict[str, np.ndarray], 
                                 tier: str = "default", force_recompute: bool = False):
        """
        Load card database with cache-first strategy.
        
        Args:
            card_images: Dictionary mapping card codes to images  
            tier: Cache tier (arena/safety/full)
            force_recompute: Force recomputation even if cached
        """
        start_time = time.time()
        self.logger.info(f"Loading card database with cache (tier: {tier}, cards: {len(card_images)})")
        
        if not self.use_cache:
            # Fallback to regular loading
            self.load_card_database(card_images)
            return
        
        # Get cached card IDs for this tier
        cached_card_ids = self.cache_manager.get_cached_card_ids(tier)
        
        # Separate cards into cached and uncached
        card_ids_to_load = list(card_images.keys())
        cached_available = [cid for cid in card_ids_to_load if cid.replace('_premium', '') in cached_card_ids]
        needs_computation = [cid for cid in card_ids_to_load if cid.replace('_premium', '') not in cached_card_ids]
        
        if force_recompute:
            needs_computation = card_ids_to_load
            cached_available = []
        
        self.logger.info(f"Cache status: {len(cached_available)} cached, {len(needs_computation)} need computation")
        
        # Load cached histograms in batch
        if cached_available:
            cache_card_ids = [cid.replace('_premium', '') for cid in cached_available]
            cached_histograms = self.cache_manager.batch_load_histograms(cache_card_ids, tier)
            
            # Add to database
            for card_id in cached_available:
                base_id = card_id.replace('_premium', '')
                if base_id in cached_histograms:
                    self.card_histograms[card_id] = cached_histograms[base_id]
        
        # Compute histograms for uncached cards
        if needs_computation:
            new_histograms = {}
            
            for card_id in needs_computation:
                image = card_images[card_id]
                hist = self.compute_histogram(image)
                
                if hist is not None:
                    self.card_histograms[card_id] = hist
                    # Store for caching (without _premium suffix)
                    base_id = card_id.replace('_premium', '')
                    new_histograms[base_id] = hist
            
            # Save new histograms to cache
            if new_histograms:
                self.cache_manager.batch_save_histograms(new_histograms, tier)
        
        # Track tier loading
        load_time = time.time() - start_time
        self.loaded_tiers.add(tier)
        self.tier_load_times[tier] = load_time
        
        self.logger.info(f"✅ Cache loading completed: {len(self.card_histograms)} histograms ({load_time:.2f}s)")
        
        cache_hit_rate = len(cached_available) / len(card_ids_to_load) * 100 if card_ids_to_load else 0
        self.logger.info(f"Cache hit rate: {cache_hit_rate:.1f}%")
    
    def load_tier_histograms(self, tier: str, max_cards: Optional[int] = None) -> bool:
        """
        Load histograms for a specific tier (full tier only now).
        
        Args:
            tier: Tier to load ("full" is the only supported option)
            max_cards: Maximum number of cards to load (None for all)
            
        Returns:
            True if tier loaded successfully
        """
        if tier in self.loaded_tiers:
            self.logger.info(f"Tier '{tier}' already loaded")
            return True
        
        start_time = time.time()
        
        try:
            if tier == "full":
                return self._load_full_tier(max_cards)
            else:
                self.logger.warning(f"Tier '{tier}' not supported. Use 'full' tier.")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load tier '{tier}': {e}")
            return False
    
    
    
    def _load_full_tier(self, max_cards: Optional[int] = None) -> bool:
        """
        Load full card database.
        
        Args:
            max_cards: Maximum cards to load
            
        Returns:
            True if loaded successfully
        """
        self.logger.info("🌍 Loading full tier (all available cards)")
        
        # Get all cached card IDs
        all_cached_ids = self.cache_manager.get_cached_card_ids("full")
        
        if max_cards:
            all_cached_ids = list(all_cached_ids)[:max_cards]
        else:
            all_cached_ids = list(all_cached_ids)
        
        self.logger.info(f"Loading {len(all_cached_ids)} cards from full cache")
        
        # Load in batch
        cached_histograms = self.cache_manager.batch_load_histograms(all_cached_ids, "full")
        
        # Add to database (replace existing)
        self.card_histograms.update(cached_histograms)
        
        self.loaded_tiers.add("full")
        self.tier_load_times["full"] = time.time() - time.time()
        
        cache_hit_rate = len(cached_histograms) / len(all_cached_ids) * 100 if all_cached_ids else 0
        self.logger.info(f"✅ Full tier loaded: {len(cached_histograms)}/{len(all_cached_ids)} cards ({cache_hit_rate:.1f}% cached)")
        
        return True
    
    def get_tier_status(self) -> Dict[str, Any]:
        """
        Get status information about loaded tiers.
        
        Returns:
            Dictionary with tier loading information
        """
        return {
            'loaded_tiers': list(self.loaded_tiers),
            'tier_load_times': dict(self.tier_load_times),
            'total_cards_loaded': len(self.card_histograms),
            'cache_enabled': self.use_cache
        }
    
    
    def get_database_size(self) -> int:
        """Get the number of cards in the histogram database."""
        return len(self.card_histograms)
    
    def clear_database(self):
        """Clear the card histogram database."""
        self.card_histograms.clear()
        self.loaded_tiers.clear()
        self.tier_load_times.clear()
        self.logger.info("Card histogram database cleared")


# Global histogram matcher instance
_histogram_matcher = None


def get_histogram_matcher() -> HistogramMatcher:
    """
    Get the global histogram matcher instance.
    
    Returns:
        HistogramMatcher instance
    """
    global _histogram_matcher
    if _histogram_matcher is None:
        _histogram_matcher = HistogramMatcher()
    return _histogram_matcher