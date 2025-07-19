"""
Patent-Free Feature Detection Ensemble

Implements ORB, BRISK, AKAZE, and SIFT feature detection algorithms
for enhanced card recognition. All algorithms are patent-free for commercial use.
Part of the Zero-Cost Detection Enhancement Plan.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Import feature cache manager
from .feature_cache_manager import FeatureCacheManager


@dataclass
class FeatureMatch:
    """Container for feature detection match results."""
    card_code: str
    distance: float
    confidence: float
    feature_count: int
    algorithm: str
    processing_time: float


@dataclass
class FeatureDescriptor:
    """Container for computed feature descriptors."""
    card_code: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    algorithm: str


class PatentFreeFeatureDetector:
    """
    Individual feature detector for a specific algorithm.
    
    Supports ORB, BRISK, AKAZE, and SIFT (patent expired 2020).
    """
    
    def __init__(self, algorithm: str = "ORB", use_cache: bool = True):
        """
        Initialize feature detector for specified algorithm.
        
        Args:
            algorithm: Algorithm name ("ORB", "BRISK", "AKAZE", "SIFT")
            use_cache: Whether to use feature caching (default: True)
        """
        self.algorithm = algorithm.upper()
        self.logger = logging.getLogger(f"{__name__}.{self.algorithm}")
        self.use_cache = use_cache
        
        # Initialize detector based on algorithm
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        
        # Feature database
        self.feature_database: Dict[str, FeatureDescriptor] = {}
        
        # Initialize cache manager if caching is enabled
        if self.use_cache:
            try:
                self.cache_manager = FeatureCacheManager()
                self.logger.info(f"{self.algorithm} feature detector initialized with caching")
            except Exception as e:
                self.logger.warning(f"Failed to initialize cache manager: {e}")
                self.use_cache = False
                self.cache_manager = None
                self.logger.info(f"{self.algorithm} feature detector initialized without caching")
        else:
            self.cache_manager = None
            self.logger.info(f"{self.algorithm} feature detector initialized without caching")
    
    def _create_detector(self):
        """Create the appropriate feature detector."""
        try:
            if self.algorithm == "ORB":
                # ORB: Patent-free, very fast
                return cv2.ORB_create(
                    nfeatures=2000,      # More features for better matching
                    scaleFactor=1.2,     # Scale pyramid factor
                    nlevels=8,           # Number of pyramid levels
                    edgeThreshold=31,    # Border edge threshold
                    firstLevel=0,        # First level in pyramid
                    WTA_K=2,            # Number of points for WTA hash
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,        # Patch size for descriptor
                    fastThreshold=20     # FAST threshold
                )
            
            elif self.algorithm == "BRISK":
                # BRISK: Patent-free, confirmed by authors
                return cv2.BRISK_create(
                    thresh=30,           # Feature detection threshold
                    octaves=3,           # Number of octaves
                    patternScale=1.0     # Pattern scale factor
                )
            
            elif self.algorithm == "AKAZE":
                # AKAZE: Not subject to patents, good performance
                return cv2.AKAZE_create(
                    descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                    descriptor_size=0,
                    descriptor_channels=3,
                    threshold=0.001,     # Feature detection threshold
                    nOctaves=4,          # Number of octaves
                    nOctaveLayers=4,     # Layers per octave
                    diffusivity=cv2.KAZE_DIFF_PM_G2
                )
            
            elif self.algorithm == "SIFT":
                # SIFT: Patent expired March 2020, now free to use
                return cv2.SIFT_create(
                    nfeatures=2000,      # Maximum features
                    nOctaveLayers=3,     # Layers in each octave
                    contrastThreshold=0.04,  # Contrast threshold
                    edgeThreshold=10,    # Edge threshold
                    sigma=1.6            # Gaussian sigma
                )
            
            else:
                self.logger.error(f"Unsupported algorithm: {self.algorithm}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create {self.algorithm} detector: {e}")
            return None
    
    def _create_matcher(self):
        """Create appropriate matcher for the descriptor type."""
        try:
            if self.algorithm in ["ORB", "BRISK", "AKAZE"]:
                # Binary descriptors use Hamming distance
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            elif self.algorithm == "SIFT":
                # Float descriptors use L2 distance
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create matcher for {self.algorithm}: {e}")
            return None
    
    def compute_features(self, image: np.ndarray) -> Optional[Tuple[List[cv2.KeyPoint], np.ndarray]]:
        """
        Compute features for an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (keypoints, descriptors) or None if failed
        """
        if self.detector is None:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            
            if descriptors is None or len(keypoints) == 0:
                self.logger.debug(f"No features detected with {self.algorithm}")
                return None
            
            self.logger.debug(f"{self.algorithm} detected {len(keypoints)} features")
            return keypoints, descriptors
            
        except Exception as e:
            self.logger.error(f"Feature computation failed with {self.algorithm}: {e}")
            return None
    
    def add_card_features(self, card_code: str, image: np.ndarray) -> bool:
        """
        Add card features to the database.
        
        Args:
            card_code: Hearthstone card code
            image: Card image
            
        Returns:
            True if features were successfully added
        """
        try:
            # Check cache first if caching is enabled
            if self.use_cache and self.cache_manager:
                cached_features = self.cache_manager.load_features(card_code, self.algorithm)
                if cached_features:
                    keypoints = cached_features['keypoints']
                    descriptors = cached_features['descriptors']
                    
                    # Store in database
                    self.feature_database[card_code] = FeatureDescriptor(
                        card_code=card_code,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        algorithm=self.algorithm
                    )
                    
                    self.logger.debug(f"Loaded {len(keypoints)} cached features for {card_code}")
                    return True
            
            # Cache miss or caching disabled - compute features
            result = self.compute_features(image)
            if result is None:
                return False
            
            keypoints, descriptors = result
            
            # Store in database
            self.feature_database[card_code] = FeatureDescriptor(
                card_code=card_code,
                keypoints=keypoints,
                descriptors=descriptors,
                algorithm=self.algorithm
            )
            
            # Cache the computed features if caching is enabled
            if self.use_cache and self.cache_manager:
                try:
                    self.cache_manager.save_features(card_code, self.algorithm, keypoints, descriptors)
                    self.logger.debug(f"Cached {len(keypoints)} features for {card_code}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache features for {card_code}: {e}")
            
            self.logger.debug(f"Added {len(keypoints)} features for {card_code}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add features for {card_code}: {e}")
            return False
    
    def match_features(self, query_image: np.ndarray, max_candidates: int = 5) -> List[FeatureMatch]:
        """
        Match query image against feature database.
        
        Args:
            query_image: Query image to match
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of FeatureMatch objects sorted by confidence
        """
        if self.matcher is None or len(self.feature_database) == 0:
            return []
        
        start_time = time.time()
        
        try:
            # Compute features for query image
            query_result = self.compute_features(query_image)
            if query_result is None:
                return []
            
            query_keypoints, query_descriptors = query_result
            
            matches = []
            
            # Match against each card in database
            for card_code, card_features in self.feature_database.items():
                try:
                    # Perform matching
                    raw_matches = self.matcher.match(query_descriptors, card_features.descriptors)
                    
                    if len(raw_matches) < 4:  # Need minimum matches for confidence
                        continue
                    
                    # Calculate match quality
                    distances = [m.distance for m in raw_matches]
                    avg_distance = np.mean(distances)
                    match_count = len(raw_matches)
                    
                    # Calculate confidence based on match count and distance
                    confidence = self._calculate_confidence(match_count, avg_distance, len(query_keypoints))
                    
                    matches.append(FeatureMatch(
                        card_code=card_code,
                        distance=avg_distance,
                        confidence=confidence,
                        feature_count=match_count,
                        algorithm=self.algorithm,
                        processing_time=time.time() - start_time
                    ))
                    
                except Exception as e:
                    self.logger.debug(f"Matching failed for {card_code}: {e}")
                    continue
            
            # Sort by confidence and limit results
            matches.sort(key=lambda x: x.confidence, reverse=True)
            return matches[:max_candidates]
            
        except Exception as e:
            self.logger.error(f"Feature matching failed: {e}")
            return []
    
    def _calculate_confidence(self, match_count: int, avg_distance: float, query_features: int) -> float:
        """
        Calculate confidence score based on match quality.
        
        Args:
            match_count: Number of feature matches
            avg_distance: Average distance of matches
            query_features: Total features in query image
            
        Returns:
            Confidence score (0-1)
        """
        # Normalize match count (more matches = higher confidence)
        match_ratio = min(1.0, match_count / max(50, query_features * 0.1))
        
        # Normalize distance (lower distance = higher confidence)
        if self.algorithm in ["ORB", "BRISK", "AKAZE"]:
            # Binary descriptors: distance typically 0-100
            distance_score = max(0.0, 1.0 - (avg_distance / 100.0))
        else:
            # SIFT uses different distance scale
            distance_score = max(0.0, 1.0 - (avg_distance / 200.0))
        
        # Combine scores with weighting
        confidence = (match_ratio * 0.6) + (distance_score * 0.4)
        
        return min(1.0, confidence)
    
    def get_database_size(self) -> int:
        """Get number of cards in feature database."""
        return len(self.feature_database)
    
    def clear_database(self):
        """Clear feature database."""
        self.feature_database.clear()
        self.logger.info(f"{self.algorithm} feature database cleared")


class FreeAlgorithmEnsemble:
    """
    Ensemble of patent-free feature detection algorithms.
    
    Combines ORB, BRISK, AKAZE, and SIFT for robust card detection.
    """
    
    def __init__(self, algorithms: List[str] = None, use_cache: bool = True):
        """
        Initialize ensemble with specified algorithms.
        
        Args:
            algorithms: List of algorithm names to use
            use_cache: Whether to use feature caching (default: True)
        """
        if algorithms is None:
            algorithms = ["ORB", "BRISK", "AKAZE", "SIFT"]
        
        self.logger = logging.getLogger(__name__)
        self.detectors: Dict[str, PatentFreeFeatureDetector] = {}
        self.primary_algorithm = "ORB"  # Fastest, used as primary
        self.use_cache = use_cache
        
        # Initialize each detector
        for algorithm in algorithms:
            try:
                detector = PatentFreeFeatureDetector(algorithm, use_cache=use_cache)
                if detector.detector is not None:
                    self.detectors[algorithm] = detector
                    cache_status = "with cache" if use_cache else "without cache"
                    self.logger.info(f"Successfully initialized {algorithm} detector {cache_status}")
                else:
                    self.logger.warning(f"Failed to initialize {algorithm} detector")
            except Exception as e:
                self.logger.error(f"Failed to create {algorithm} detector: {e}")
        
        if not self.detectors:
            self.logger.error("No feature detectors were successfully initialized")
        else:
            self.logger.info(f"FreeAlgorithmEnsemble initialized with {len(self.detectors)} detectors")
    
    def load_card_database(self, card_images: Dict[str, np.ndarray]):
        """
        Load card images and compute features for all algorithms.
        
        Args:
            card_images: Dictionary mapping card codes to images
        """
        self.logger.info(f"Loading feature database with {len(card_images)} cards")
        
        for card_code, image in card_images.items():
            for algorithm, detector in self.detectors.items():
                try:
                    detector.add_card_features(card_code, image)
                except Exception as e:
                    self.logger.debug(f"Failed to add features for {card_code} with {algorithm}: {e}")
        
        # Log database sizes
        for algorithm, detector in self.detectors.items():
            size = detector.get_database_size()
            self.logger.info(f"{algorithm} feature database: {size} cards")
    
    def detect_card(self, query_image: np.ndarray, max_candidates: int = 5) -> List[FeatureMatch]:
        """
        Detect card using ensemble of feature algorithms.
        
        Args:
            query_image: Query image to match
            max_candidates: Maximum candidates per algorithm
            
        Returns:
            List of all matches from all algorithms
        """
        all_matches = []
        
        for algorithm, detector in self.detectors.items():
            try:
                matches = detector.match_features(query_image, max_candidates)
                all_matches.extend(matches)
                self.logger.debug(f"{algorithm} found {len(matches)} matches")
            except Exception as e:
                self.logger.debug(f"{algorithm} detection failed: {e}")
                continue
        
        # Sort all matches by confidence
        all_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_matches
    
    def get_best_match(self, query_image: np.ndarray) -> Optional[FeatureMatch]:
        """
        Get single best match across all algorithms.
        
        Args:
            query_image: Query image to match
            
        Returns:
            Best FeatureMatch or None if no good match found
        """
        matches = self.detect_card(query_image, max_candidates=3)
        
        if matches and matches[0].confidence > 0.3:
            return matches[0]
        
        return None
    
    def get_algorithm_consensus(self, query_image: np.ndarray, min_agreements: int = 2) -> Optional[str]:
        """
        Get card code that multiple algorithms agree on.
        
        Args:
            query_image: Query image to match
            min_agreements: Minimum number of algorithms that must agree
            
        Returns:
            Card code with consensus or None
        """
        matches = self.detect_card(query_image, max_candidates=1)
        
        # Count votes for each card
        votes = {}
        for match in matches:
            if match.confidence > 0.3:  # Only count confident votes
                votes[match.card_code] = votes.get(match.card_code, 0) + 1
        
        # Find cards with enough agreements
        for card_code, vote_count in votes.items():
            if vote_count >= min_agreements:
                self.logger.debug(f"Consensus reached for {card_code} with {vote_count} votes")
                return card_code
        
        return None
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of successfully initialized algorithms."""
        return list(self.detectors.keys())
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database size statistics for each algorithm."""
        return {algorithm: detector.get_database_size() 
                for algorithm, detector in self.detectors.items()}


# Global ensemble instance
_feature_ensemble = None


def get_feature_ensemble(use_cache: bool = True) -> FreeAlgorithmEnsemble:
    """
    Get the global FreeAlgorithmEnsemble instance.
    
    Args:
        use_cache: Whether to use feature caching (default: True)
        
    Returns:
        FreeAlgorithmEnsemble instance
    """
    global _feature_ensemble
    if _feature_ensemble is None:
        _feature_ensemble = FreeAlgorithmEnsemble(use_cache=use_cache)
    return _feature_ensemble