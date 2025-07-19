"""
Perceptual Hash Cache Manager

High-performance binary caching system for perceptual hashes.
Implements sub-millisecond loading strategy for ultra-fast card detection.
"""

import logging
import json
import time
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import threading

try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


@dataclass
class PHashCacheMetadata:
    """Metadata for pHash cache files."""
    version: str
    created_at: str
    hash_size: int  # 8 for 64-bit, 16 for 256-bit
    hamming_threshold: int
    card_count: int
    total_size_bytes: int
    compression: bool
    checksum: str
    
    def is_compatible(self, hash_size: int, hamming_threshold: int) -> bool:
        """Check if cached pHashes are compatible with current parameters."""
        return (self.hash_size == hash_size and 
                self.hamming_threshold == hamming_threshold)


@dataclass
class PHashCacheStats:
    """Statistics for pHash cache operations."""
    cache_hits: int = 0
    cache_misses: int = 0
    load_time_ms: float = 0.0
    save_time_ms: float = 0.0
    total_cards_cached: int = 0
    cache_size_kb: float = 0.0  # pHashes are much smaller than histograms
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


class PHashCacheManager:
    """
    High-performance perceptual hash cache manager.
    
    Provides binary caching of pHash strings with compression,
    integrity checking, and batch operations for optimal performance.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize pHash cache manager.
        
        Args:
            cache_dir: Cache directory (uses default if None)
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "phashes"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.metadata_file = self.cache_dir / "metadata.json"
        self.phashes_file = self.cache_dir / "phashes.bin"
        
        # Cache data
        self.cached_phashes: Dict[str, str] = {}  # {card_code: hash_string}
        self.reverse_lookup: Dict[str, str] = {}  # {hash_string: card_code}
        
        # Statistics
        self.stats = PHashCacheStats()
        self._lock = threading.Lock()
        
        self.logger.info(f"PHashCacheManager initialized (cache_dir: {self.cache_dir})")
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum for cache integrity."""
        return hashlib.sha256(data).hexdigest()
    
    def _save_metadata(self, metadata: PHashCacheMetadata):
        """Save cache metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _load_metadata(self) -> Optional[PHashCacheMetadata]:
        """Load cache metadata from JSON file."""
        try:
            if not self.metadata_file.exists():
                return None
            
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return PHashCacheMetadata(**data)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None
    
    def save_phashes(self, phash_data: Dict[str, str], 
                     hash_size: int = 8, 
                     hamming_threshold: int = 10) -> bool:
        """
        Save pHash data to cache.
        
        Args:
            phash_data: Dictionary mapping card codes to hash strings
            hash_size: Hash size used for computation
            hamming_threshold: Hamming distance threshold
            
        Returns:
            True if saved successfully, False otherwise
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Prepare data for serialization
                cache_data = {
                    'card_phashes': phash_data,
                    'reverse_lookup': {v: k for k, v in phash_data.items()}
                }
                
                # Serialize to binary format
                binary_data = pickle.dumps(cache_data, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Compress if LZ4 available
                compression_used = False
                if LZ4_AVAILABLE:
                    try:
                        compressed_data = lz4.compress(binary_data)
                        if len(compressed_data) < len(binary_data):
                            binary_data = compressed_data
                            compression_used = True
                    except Exception as e:
                        self.logger.warning(f"LZ4 compression failed, using uncompressed: {e}")
                
                # Compute checksum
                checksum = self._compute_checksum(binary_data)
                
                # Save binary data
                with open(self.phashes_file, 'wb') as f:
                    f.write(binary_data)
                
                # Create and save metadata
                metadata = PHashCacheMetadata(
                    version="1.0",
                    created_at=datetime.now().isoformat(),
                    hash_size=hash_size,
                    hamming_threshold=hamming_threshold,
                    card_count=len(phash_data),
                    total_size_bytes=len(binary_data),
                    compression=compression_used,
                    checksum=checksum
                )
                
                self._save_metadata(metadata)
                
                # Update internal cache
                self.cached_phashes = phash_data.copy()
                self.reverse_lookup = {v: k for k, v in phash_data.items()}
                
                # Update statistics
                save_time = (time.time() - start_time) * 1000
                self.stats.save_time_ms = save_time
                self.stats.total_cards_cached = len(phash_data)
                self.stats.cache_size_kb = len(binary_data) / 1024
                
                compression_info = f" (LZ4 compressed)" if compression_used else ""
                self.logger.info(f"Saved {len(phash_data)} pHashes to cache in {save_time:.1f}ms{compression_info}")
                self.logger.info(f"Cache size: {self.stats.cache_size_kb:.1f} KB")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save pHash cache: {e}")
            return False
    
    def load_phashes(self, hash_size: int = 8, 
                     hamming_threshold: int = 10) -> Optional[Dict[str, str]]:
        """
        Load pHash data from cache.
        
        Args:
            hash_size: Expected hash size
            hamming_threshold: Expected hamming threshold
            
        Returns:
            Dictionary mapping card codes to hash strings, or None if cache invalid
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Check if cache files exist
                if not self.phashes_file.exists() or not self.metadata_file.exists():
                    self.stats.cache_misses += 1
                    return None
                
                # Load and validate metadata
                metadata = self._load_metadata()
                if not metadata:
                    self.stats.cache_misses += 1
                    return None
                
                # Check compatibility
                if not metadata.is_compatible(hash_size, hamming_threshold):
                    self.logger.info(f"Cache incompatible (size: {metadata.hash_size} vs {hash_size}, "
                                   f"threshold: {metadata.hamming_threshold} vs {hamming_threshold})")
                    self.stats.cache_misses += 1
                    return None
                
                # Load binary data
                with open(self.phashes_file, 'rb') as f:
                    binary_data = f.read()
                
                # Verify checksum
                actual_checksum = self._compute_checksum(binary_data)
                if actual_checksum != metadata.checksum:
                    self.logger.error("Cache checksum mismatch - cache corrupted")
                    self.stats.cache_misses += 1
                    return None
                
                # Decompress if needed
                if metadata.compression and LZ4_AVAILABLE:
                    try:
                        binary_data = lz4.decompress(binary_data)
                    except Exception as e:
                        self.logger.error(f"Failed to decompress cache: {e}")
                        self.stats.cache_misses += 1
                        return None
                
                # Deserialize data
                cache_data = pickle.loads(binary_data)
                phash_data = cache_data['card_phashes']
                
                # Update internal cache
                self.cached_phashes = phash_data.copy()
                self.reverse_lookup = cache_data.get('reverse_lookup', {})
                
                # Update statistics
                load_time = (time.time() - start_time) * 1000
                self.stats.load_time_ms = load_time
                self.stats.cache_hits += 1
                self.stats.total_cards_cached = len(phash_data)
                self.stats.cache_size_kb = metadata.total_size_bytes / 1024
                
                compression_info = f" (LZ4 compressed)" if metadata.compression else ""
                self.logger.info(f"Loaded {len(phash_data)} pHashes from cache in {load_time:.1f}ms{compression_info}")
                
                return phash_data
                
        except Exception as e:
            self.logger.error(f"Failed to load pHash cache: {e}")
            self.stats.cache_misses += 1
            return None
    
    def get_cached_phash(self, card_code: str) -> Optional[str]:
        """
        Get cached pHash for a specific card.
        
        Args:
            card_code: Card code to lookup
            
        Returns:
            Hash string or None if not cached
        """
        return self.cached_phashes.get(card_code)
    
    def get_card_by_hash(self, hash_string: str) -> Optional[str]:
        """
        Get card code by hash string (reverse lookup).
        
        Args:
            hash_string: Hash string to lookup
            
        Returns:
            Card code or None if not found
        """
        return self.reverse_lookup.get(hash_string)
    
    def is_cached(self, card_code: str) -> bool:
        """Check if a card's pHash is cached."""
        return card_code in self.cached_phashes
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        metadata = self._load_metadata()
        
        return {
            'cache_exists': self.phashes_file.exists(),
            'metadata_exists': self.metadata_file.exists(),
            'cached_cards': len(self.cached_phashes),
            'cache_size_kb': self.stats.cache_size_kb,
            'last_load_time_ms': self.stats.load_time_ms,
            'last_save_time_ms': self.stats.save_time_ms,
            'hit_rate_percent': self.stats.get_hit_rate(),
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'metadata': asdict(metadata) if metadata else None
        }
    
    def clear_cache(self):
        """Clear all cached data and files."""
        try:
            with self._lock:
                # Clear memory cache
                self.cached_phashes.clear()
                self.reverse_lookup.clear()
                
                # Remove cache files
                if self.phashes_file.exists():
                    self.phashes_file.unlink()
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                
                # Reset statistics
                self.stats = PHashCacheStats()
                
                self.logger.info("pHash cache cleared")
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def validate_cache(self) -> bool:
        """
        Validate cache integrity.
        
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            metadata = self._load_metadata()
            if not metadata:
                return False
            
            if not self.phashes_file.exists():
                return False
            
            # Check checksum
            with open(self.phashes_file, 'rb') as f:
                binary_data = f.read()
            
            actual_checksum = self._compute_checksum(binary_data)
            return actual_checksum == metadata.checksum
            
        except Exception as e:
            self.logger.error(f"Cache validation failed: {e}")
            return False


# Global cache manager instance
_phash_cache_manager = None
_cache_lock = threading.Lock()


def get_phash_cache_manager(cache_dir: Optional[Path] = None) -> PHashCacheManager:
    """
    Get singleton pHash cache manager instance.
    
    Args:
        cache_dir: Cache directory (uses default if None)
        
    Returns:
        PHashCacheManager instance
    """
    global _phash_cache_manager
    
    with _cache_lock:
        if _phash_cache_manager is None:
            _phash_cache_manager = PHashCacheManager(cache_dir)
        
        return _phash_cache_manager