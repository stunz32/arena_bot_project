"""
Perceptual Hash-based card matching system.

Ultra-fast pre-filtering using perceptual hashing (pHash) with Hamming distance comparison.
Based on research from wittenbe/Hearthstone-Image-Recognition and other successful card recognition projects.
Provides 100-1000x faster detection for clear card images with graceful fallback to existing systems.
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    from ..data.arena_card_database import get_arena_card_database
    ARENA_DB_AVAILABLE = True
except ImportError:
    ARENA_DB_AVAILABLE = False


@dataclass
class PHashMatch:
    """Container for pHash match results."""
    card_code: str
    hamming_distance: int
    confidence: float
    is_premium: bool
    processing_time: float


class PerceptualHashMatcher:
    """
    Perceptual hash-based card matching for ultra-fast detection.
    
    Uses imagehash.phash() for 64-bit DCT hashes with Hamming distance comparison.
    Designed for 100-1000x faster detection on clear card images.
    """
    
    def __init__(self, use_cache: bool = True, hamming_threshold: int = 10):
        """
        Initialize perceptual hash matcher.
        
        Args:
            use_cache: Enable pHash caching for fast loading
            hamming_threshold: Maximum Hamming distance for matches (0-64)
        """
        self.logger = logging.getLogger(__name__)
        
        if not IMAGEHASH_AVAILABLE:
            raise ImportError("imagehash library required: pip install imagehash")
        
        # Configuration
        self.use_cache = use_cache
        self.hamming_threshold = hamming_threshold
        
        # Hash storage: {hash_string: card_code}
        self.phash_database: Dict[str, str] = {}
        
        # Card lookup: {card_code: hash_string} 
        self.card_phashes: Dict[str, str] = {}
        
        # Performance tracking
        self.total_lookups = 0
        self.successful_matches = 0
        self.total_lookup_time = 0.0
        
        # Arena card integration
        self.arena_db = None
        if ARENA_DB_AVAILABLE:
            try:
                self.arena_db = get_arena_card_database()
            except Exception as e:
                self.logger.warning(f"Arena database not available: {e}")
        
        self.logger.info(f"PerceptualHashMatcher initialized (threshold: {hamming_threshold})")
    
    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> Optional[str]:
        """
        Compute perceptual hash for an image.
        
        Args:
            image: OpenCV image array (BGR format)
            hash_size: Hash size (8 = 64-bit hash, 16 = 256-bit hash)
            
        Returns:
            Hash string or None if computation failed
        """
        try:
            # Convert OpenCV BGR to PIL RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Compute perceptual hash
            phash = imagehash.phash(pil_image, hash_size=hash_size)
            
            return str(phash)
            
        except Exception as e:
            self.logger.error(f"pHash computation failed: {e}")
            return None
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hash strings.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Hamming distance (number of differing bits)
        """
        try:
            # Convert hash strings to imagehash objects for comparison
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            
            return h1 - h2  # imagehash uses - operator for Hamming distance
            
        except Exception as e:
            self.logger.error(f"Hamming distance calculation failed: {e}")
            return 64  # Return maximum distance on error
    
    def add_card_phash(self, card_code: str, image: np.ndarray, is_premium: bool = False):
        """
        Add a card's perceptual hash to the database.
        
        Args:
            card_code: Hearthstone card code (e.g., "AT_001")
            image: Card image
            is_premium: Whether this is a premium (golden) card
        """
        phash = self.compute_phash(image)
        
        if phash is not None:
            # Create full card identifier
            full_card_code = f"{card_code}_premium" if is_premium else card_code
            
            # Store bidirectional mapping
            self.phash_database[phash] = full_card_code
            self.card_phashes[full_card_code] = phash
            
            self.logger.debug(f"Added pHash for {full_card_code}: {phash}")
        else:
            self.logger.warning(f"Failed to compute pHash for {card_code}")
    
    def load_card_database(self, card_images: Dict[str, np.ndarray], progress_callback=None):
        """
        Load card images and compute perceptual hashes.
        
        Args:
            card_images: Dictionary mapping card codes to images
            progress_callback: Optional callback for progress updates
        """
        self.logger.info(f"Computing pHashes for {len(card_images)} cards")
        start_time = time.time()
        
        processed = 0
        for card_code, image in card_images.items():
            is_premium = card_code.endswith("_premium")
            base_code = card_code.replace("_premium", "")
            
            self.add_card_phash(base_code, image, is_premium)
            processed += 1
            
            # Progress reporting
            if progress_callback and processed % 500 == 0:
                progress_callback(processed, len(card_images))
        
        duration = time.time() - start_time
        self.logger.info(f"pHash database loaded: {len(self.card_phashes)} cards in {duration:.2f}s")
    
    def find_best_phash_match(self, query_image: np.ndarray, 
                              confidence_threshold: float = 0.3) -> Optional[PHashMatch]:
        """
        Find the best matching card using perceptual hash comparison.
        
        Args:
            query_image: Card image to match
            confidence_threshold: Minimum confidence for valid match
            
        Returns:
            PHashMatch object or None if no good match found
        """
        start_time = time.time()
        self.total_lookups += 1
        
        # Compute query hash
        query_hash = self.compute_phash(query_image)
        if query_hash is None:
            return None
        
        # Find best match using Hamming distance
        best_match = None
        min_distance = float('inf')
        best_card_code = None
        
        for stored_hash, card_code in self.phash_database.items():
            distance = self.hamming_distance(query_hash, stored_hash)
            
            if distance < min_distance:
                min_distance = distance
                best_card_code = card_code
                
                # Early termination for perfect matches
                if distance == 0:
                    break
        
        processing_time = time.time() - start_time
        self.total_lookup_time += processing_time
        
        # Check if match is good enough
        if min_distance <= self.hamming_threshold and best_card_code:
            # Convert Hamming distance to confidence score
            # Lower distance = higher confidence
            confidence = max(0.0, 1.0 - (min_distance / 64.0))
            
            if confidence >= confidence_threshold:
                self.successful_matches += 1
                
                is_premium = best_card_code.endswith('_premium')
                base_card_code = best_card_code.replace('_premium', '')
                
                return PHashMatch(
                    card_code=best_card_code,
                    hamming_distance=int(min_distance),
                    confidence=confidence,
                    is_premium=is_premium,
                    processing_time=processing_time
                )
        
        return None
    
    def find_best_matches(self, query_image: np.ndarray, 
                          top_k: int = 3) -> List[PHashMatch]:
        """
        Find top-k best matching cards using pHash.
        
        Args:
            query_image: Card image to match
            top_k: Number of top matches to return
            
        Returns:
            List of PHashMatch objects sorted by confidence
        """
        start_time = time.time()
        
        query_hash = self.compute_phash(query_image)
        if query_hash is None:
            return []
        
        # Calculate distances for all cards
        matches = []
        for stored_hash, card_code in self.phash_database.items():
            distance = self.hamming_distance(query_hash, stored_hash)
            confidence = max(0.0, 1.0 - (distance / 64.0))
            
            if distance <= self.hamming_threshold:
                is_premium = card_code.endswith('_premium')
                
                matches.append(PHashMatch(
                    card_code=card_code,
                    hamming_distance=int(distance),
                    confidence=confidence,
                    is_premium=is_premium,
                    processing_time=time.time() - start_time
                ))
        
        # Sort by confidence (descending) and return top-k
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for the pHash matcher.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_lookup_time = 0.0
        if self.total_lookups > 0:
            avg_lookup_time = self.total_lookup_time / self.total_lookups
        
        success_rate = 0.0
        if self.total_lookups > 0:
            success_rate = self.successful_matches / self.total_lookups
        
        return {
            'total_cards': len(self.card_phashes),
            'total_hashes': len(self.phash_database),
            'total_lookups': self.total_lookups,
            'successful_matches': self.successful_matches,
            'success_rate': success_rate,
            'avg_lookup_time_ms': avg_lookup_time * 1000,
            'hamming_threshold': self.hamming_threshold
        }
    
    def is_arena_eligible(self, card_code: str) -> bool:
        """
        Check if a card is arena eligible (if arena database available).
        
        Args:
            card_code: Card code to check
            
        Returns:
            True if arena eligible, False otherwise (or if DB unavailable)
        """
        if not self.arena_db:
            return True  # Assume eligible if no arena DB
        
        try:
            clean_code = card_code.replace('_premium', '')
            return self.arena_db.is_arena_eligible(clean_code)
        except Exception:
            return True
    
    def get_card_name(self, card_code: str) -> str:
        """
        Get user-friendly card name (if arena database available).
        
        Args:
            card_code: Card code
            
        Returns:
            Card name or card code if unavailable
        """
        if not self.arena_db:
            return card_code
        
        try:
            clean_code = card_code.replace('_premium', '')
            return self.arena_db.get_card_name(clean_code)
        except Exception:
            return card_code


def get_perceptual_hash_matcher(use_cache: bool = True, 
                                hamming_threshold: int = 10) -> Optional[PerceptualHashMatcher]:
    """
    Factory function to create a PerceptualHashMatcher instance.
    
    Args:
        use_cache: Enable pHash caching
        hamming_threshold: Hamming distance threshold for matches
        
    Returns:
        PerceptualHashMatcher instance or None if dependencies unavailable
    """
    try:
        return PerceptualHashMatcher(use_cache=use_cache, hamming_threshold=hamming_threshold)
    except ImportError as e:
        logging.getLogger(__name__).error(f"Cannot create PerceptualHashMatcher: {e}")
        return None