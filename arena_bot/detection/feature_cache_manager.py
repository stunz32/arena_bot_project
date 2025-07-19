#!/usr/bin/env python3
"""
FEATURE CACHE MANAGER
Persistent binary cache for Ultimate Detection Engine feature descriptors.
Eliminates the one-time analysis freeze by caching complex feature data.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

class FeatureCacheManager:
    """
    Manages persistent binary cache for feature descriptors (ORB, SIFT, BRISK, AKAZE).
    
    Cache Structure:
    assets/cache/features/
    ├── orb/
    │   ├── AT_001.pkl
    │   ├── AT_002.pkl
    │   └── ...
    ├── sift/
    │   ├── AT_001.pkl
    │   └── ...
    ├── brisk/
    └── akaze/
    
    Cache Metadata:
    assets/cache/features/cache_metadata.json
    """
    
    def __init__(self, cache_dir: Path = None):
        """Initialize the feature cache manager."""
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "features"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for cache statistics and validation
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Supported algorithms
        self.supported_algorithms = ['orb', 'sift', 'brisk', 'akaze']
        
        # Create algorithm subdirectories
        for algo in self.supported_algorithms:
            algo_dir = self.cache_dir / algo.lower()
            algo_dir.mkdir(exist_ok=True)
        
        logger.info(f"FeatureCacheManager initialized: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata or create default."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        # Default metadata
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "algorithms": {},
            "total_cached_cards": 0
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_cache_path(self, card_code: str, algorithm: str) -> Path:
        """Get the cache file path for a specific card and algorithm."""
        if algorithm.lower() not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {self.supported_algorithms}")
        
        algo_dir = self.cache_dir / algorithm.lower()
        return algo_dir / f"{card_code}.pkl"
    
    def save_features(self, card_code: str, algorithm: str, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray):
        """
        Save feature keypoints and descriptors to cache.
        
        Args:
            card_code: Card identifier (e.g., "AT_001")
            algorithm: Feature algorithm name (orb, sift, brisk, akaze)
            keypoints: OpenCV KeyPoint objects
            descriptors: NumPy array of feature descriptors
        """
        try:
            cache_path = self.get_cache_path(card_code, algorithm)
            
            # Convert KeyPoint objects to serializable format
            serializable_keypoints = []
            if keypoints:
                for kp in keypoints:
                    serializable_keypoints.append({
                        'x': kp.pt[0],
                        'y': kp.pt[1],
                        'size': kp.size,
                        'angle': kp.angle,
                        'response': kp.response,
                        'octave': kp.octave,
                        'class_id': kp.class_id
                    })
            
            # Prepare data for serialization
            cache_data = {
                'card_code': card_code,
                'algorithm': algorithm,
                'keypoints': serializable_keypoints,
                'descriptors': descriptors,
                'cached_at': datetime.now().isoformat(),
                'keypoint_count': len(keypoints) if keypoints else 0,
                'descriptor_shape': descriptors.shape if descriptors is not None else None
            }
            
            # Save to pickle file
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            algo_key = algorithm.lower()
            if algo_key not in self.metadata["algorithms"]:
                self.metadata["algorithms"][algo_key] = {"cached_cards": 0}
            
            self.metadata["algorithms"][algo_key]["cached_cards"] += 1
            self.metadata["total_cached_cards"] += 1
            
            logger.debug(f"Cached {algorithm} features for {card_code}: {len(keypoints)} keypoints")
            
        except Exception as e:
            logger.error(f"Failed to save features for {card_code}/{algorithm}: {e}")
            raise
    
    def load_features(self, card_code: str, algorithm: str) -> Optional[Dict]:
        """
        Load feature keypoints and descriptors from cache.
        
        Args:
            card_code: Card identifier (e.g., "AT_001")
            algorithm: Feature algorithm name (orb, sift, brisk, akaze)
            
        Returns:
            Dictionary with 'keypoints' and 'descriptors' keys, or None if not cached
        """
        try:
            cache_path = self.get_cache_path(card_code, algorithm)
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data
            if not isinstance(cache_data, dict) or 'keypoints' not in cache_data or 'descriptors' not in cache_data:
                logger.warning(f"Invalid cache data for {card_code}/{algorithm}")
                return None
            
            # Convert serializable keypoints back to OpenCV KeyPoint objects
            keypoints = []
            if cache_data['keypoints']:
                for kp_data in cache_data['keypoints']:
                    keypoint = cv2.KeyPoint(
                        x=kp_data['x'],
                        y=kp_data['y'],
                        size=kp_data['size'],
                        angle=kp_data['angle'],
                        response=kp_data['response'],
                        octave=kp_data['octave'],
                        class_id=kp_data['class_id']
                    )
                    keypoints.append(keypoint)
            
            logger.debug(f"Loaded {algorithm} features for {card_code}: {len(keypoints)} keypoints")
            
            return {
                'keypoints': keypoints,
                'descriptors': cache_data['descriptors']
            }
            
        except Exception as e:
            logger.error(f"Failed to load features for {card_code}/{algorithm}: {e}")
            return None
    
    def is_cached(self, card_code: str, algorithm: str) -> bool:
        """
        Check if features are cached for a specific card and algorithm.
        
        Args:
            card_code: Card identifier (e.g., "AT_001")
            algorithm: Feature algorithm name (orb, sift, brisk, akaze)
            
        Returns:
            True if features are cached, False otherwise
        """
        try:
            cache_path = self.get_cache_path(card_code, algorithm)
            return cache_path.exists() and cache_path.stat().st_size > 0
        except Exception as e:
            logger.error(f"Failed to check cache for {card_code}/{algorithm}: {e}")
            return False
    
    def get_cached_algorithms(self, card_code: str) -> List[str]:
        """Get list of algorithms that have cached features for a card."""
        cached_algorithms = []
        for algo in self.supported_algorithms:
            if self.is_cached(card_code, algo):
                cached_algorithms.append(algo)
        return cached_algorithms
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        stats = {
            'total_cached_cards': 0,
            'algorithms': {},
            'cache_size_mb': 0.0,
            'cache_directory': str(self.cache_dir)
        }
        
        # Calculate cache size and file counts
        total_size = 0
        for algo in self.supported_algorithms:
            algo_dir = self.cache_dir / algo.lower()
            algo_files = list(algo_dir.glob("*.pkl"))
            algo_size = sum(f.stat().st_size for f in algo_files)
            
            stats['algorithms'][algo] = {
                'cached_cards': len(algo_files),
                'size_mb': algo_size / (1024 * 1024)
            }
            
            total_size += algo_size
            stats['total_cached_cards'] = max(stats['total_cached_cards'], len(algo_files))
        
        stats['cache_size_mb'] = total_size / (1024 * 1024)
        
        return stats
    
    def is_cache_valid_for_ensemble(self, algorithms: List[str]) -> bool:
        """
        Check if the cache is present and complete for all required algorithms.
        
        Args:
            algorithms: List of algorithm names to check (e.g., ['orb', 'sift', 'brisk', 'akaze'])
            
        Returns:
            True if cache is valid and complete for all algorithms, False otherwise
        """
        if not self.cache_dir.exists():
            logger.info("Feature cache directory does not exist")
            return False
        
        for algo in algorithms:
            algo_dir = self.cache_dir / algo.lower()
            if not algo_dir.exists():
                logger.warning(f"Feature cache is incomplete. Missing directory for: {algo}")
                return False
            
            # Check if directory has any cache files
            cache_files = list(algo_dir.glob("*.pkl"))
            if not cache_files:
                logger.warning(f"Feature cache is incomplete. Empty directory for: {algo}")
                return False
        
        logger.info("Found valid, complete feature cache for all ensemble algorithms")
        return True
    
    def clear_cache(self, algorithm: str = None):
        """
        Clear the feature cache.
        
        Args:
            algorithm: If specified, only clear cache for this algorithm.
                      If None, clear all cached features.
        """
        try:
            if algorithm:
                # Clear specific algorithm
                algo_dir = self.cache_dir / algorithm.lower()
                if algo_dir.exists():
                    for cache_file in algo_dir.glob("*.pkl"):
                        cache_file.unlink()
                    logger.info(f"Cleared {algorithm} feature cache")
            else:
                # Clear all algorithms
                for algo in self.supported_algorithms:
                    algo_dir = self.cache_dir / algo.lower()
                    if algo_dir.exists():
                        for cache_file in algo_dir.glob("*.pkl"):
                            cache_file.unlink()
                logger.info("Cleared all feature caches")
            
            # Reset metadata
            self.metadata = self._load_metadata()
            self.metadata["algorithms"] = {}
            self.metadata["total_cached_cards"] = 0
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
    
    def validate_cache(self, card_code: str, algorithm: str) -> bool:
        """
        Validate that cached features are properly formatted.
        
        Args:
            card_code: Card identifier
            algorithm: Feature algorithm name
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            cache_data = self.load_features(card_code, algorithm)
            if not cache_data:
                return False
            
            # Check keypoints
            keypoints = cache_data['keypoints']
            if not isinstance(keypoints, list):
                return False
            
            # Check descriptors
            descriptors = cache_data['descriptors']
            if descriptors is not None and not isinstance(descriptors, np.ndarray):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Cache validation failed for {card_code}/{algorithm}: {e}")
            return False
    
    def __del__(self):
        """Save metadata when the manager is destroyed."""
        try:
            self._save_metadata()
        except:
            pass  # Ignore errors during cleanup