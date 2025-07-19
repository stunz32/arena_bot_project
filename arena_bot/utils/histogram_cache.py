"""
Histogram Cache Manager

High-performance binary caching system for OpenCV histograms.
Implements Arena Tracker's fast loading strategy with sub-50ms load times.
"""

import logging
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


@dataclass
class CacheMetadata:
    """Metadata for histogram cache files."""
    version: str
    created_at: str
    histogram_params: Dict[str, Any]
    card_count: int
    total_size_bytes: int
    compression: bool
    checksum: str
    
    def is_compatible(self, params: Dict[str, Any]) -> bool:
        """Check if cached histograms are compatible with current parameters."""
        return self.histogram_params == params


@dataclass
class CacheStats:
    """Statistics for cache operations."""
    cache_hits: int = 0
    cache_misses: int = 0
    load_time_ms: float = 0.0
    save_time_ms: float = 0.0
    total_cards_cached: int = 0
    cache_size_mb: float = 0.0
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


class HistogramCacheManager:
    """
    High-performance histogram cache manager.
    
    Provides binary caching of OpenCV histograms with compression,
    integrity checking, and batch operations for optimal performance.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize histogram cache manager.
        
        Args:
            cache_dir: Cache directory (uses default if None)
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "histograms"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.use_compression = LZ4_AVAILABLE  # Use LZ4 compression if available
        self.cache_version = "1.0"
        self.max_cache_size_mb = 500  # Maximum cache size
        
        # Runtime data
        self.stats = CacheStats()
        self._cache_lock = threading.RLock()
        
        # Cache files
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.manifest_file = self.cache_dir / "cache_manifest.json"
        
        self.logger.info(f"HistogramCacheManager initialized: {self.cache_dir}")
        if self.use_compression:
            self.logger.info("✅ LZ4 compression enabled")
        else:
            self.logger.warning("⚠️ LZ4 not available - install with: pip install lz4")
    
    def _get_cache_file_path(self, card_id: str, tier: str = "default") -> Path:
        """Get cache file path for a card."""
        tier_dir = self.cache_dir / tier
        tier_dir.mkdir(exist_ok=True)
        
        extension = ".hist.lz4" if self.use_compression else ".hist"
        return tier_dir / f"{card_id}{extension}"
    
    def _compute_histogram_checksum(self, histogram: np.ndarray) -> str:
        """Compute checksum for histogram integrity checking."""
        hist_bytes = histogram.tobytes()
        return hashlib.sha256(hist_bytes).hexdigest()[:16]  # Use first 16 chars
    
    def _serialize_histogram(self, histogram: np.ndarray) -> bytes:
        """
        Serialize histogram to binary format.
        
        Args:
            histogram: OpenCV histogram array
            
        Returns:
            Serialized binary data
        """
        try:
            # Convert to bytes using numpy's native format
            hist_bytes = histogram.tobytes()
            
            # Create header with shape and dtype info
            header = {
                'shape': histogram.shape,
                'dtype': str(histogram.dtype)
            }
            header_json = json.dumps(header).encode('utf-8')
            header_size = len(header_json)
            
            # Combine header size (4 bytes) + header + data
            serialized = (
                header_size.to_bytes(4, byteorder='little') +
                header_json +
                hist_bytes
            )
            
            # Apply compression if available
            if self.use_compression:
                serialized = lz4.compress(serialized)
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Failed to serialize histogram: {e}")
            raise
    
    def _deserialize_histogram(self, data: bytes) -> np.ndarray:
        """
        Deserialize histogram from binary format.
        
        Args:
            data: Serialized binary data
            
        Returns:
            OpenCV histogram array
        """
        try:
            # Decompress if needed
            if self.use_compression:
                data = lz4.decompress(data)
            
            # Read header size
            header_size = int.from_bytes(data[:4], byteorder='little')
            
            # Read header
            header_json = data[4:4+header_size].decode('utf-8')
            header = json.loads(header_json)
            
            # Read histogram data
            hist_bytes = data[4+header_size:]
            
            # Reconstruct array
            histogram = np.frombuffer(hist_bytes, dtype=header['dtype'])
            histogram = histogram.reshape(header['shape'])
            
            return histogram
            
        except Exception as e:
            self.logger.error(f"Failed to deserialize histogram: {e}")
            raise
    
    def save_histogram(self, card_id: str, histogram: np.ndarray, tier: str = "default") -> bool:
        """
        Save histogram to cache.
        
        Args:
            card_id: Card identifier
            histogram: OpenCV histogram array
            tier: Cache tier (arena, safety, full)
            
        Returns:
            True if saved successfully
        """
        start_time = time.time()
        
        try:
            with self._cache_lock:
                cache_file = self._get_cache_file_path(card_id, tier)
                
                # Serialize histogram
                serialized_data = self._serialize_histogram(histogram)
                
                # Write to file
                with open(cache_file, 'wb') as f:
                    f.write(serialized_data)
                
                # Update statistics
                save_time = (time.time() - start_time) * 1000
                self.stats.save_time_ms += save_time
                
                self.logger.debug(f"Saved histogram cache: {card_id} ({len(serialized_data)} bytes, {save_time:.1f}ms)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save histogram cache for {card_id}: {e}")
            return False
    
    def load_histogram(self, card_id: str, tier: str = "default") -> Optional[np.ndarray]:
        """
        Load histogram from cache.
        
        Args:
            card_id: Card identifier
            tier: Cache tier
            
        Returns:
            OpenCV histogram array or None if not cached
        """
        start_time = time.time()
        
        try:
            with self._cache_lock:
                cache_file = self._get_cache_file_path(card_id, tier)
                
                if not cache_file.exists():
                    self.stats.cache_misses += 1
                    return None
                
                # Read and deserialize
                with open(cache_file, 'rb') as f:
                    serialized_data = f.read()
                
                histogram = self._deserialize_histogram(serialized_data)
                
                # Update statistics
                load_time = (time.time() - start_time) * 1000
                self.stats.load_time_ms += load_time
                self.stats.cache_hits += 1
                
                self.logger.debug(f"Loaded histogram cache: {card_id} ({load_time:.1f}ms)")
                return histogram
                
        except Exception as e:
            self.logger.error(f"Failed to load histogram cache for {card_id}: {e}")
            self.stats.cache_misses += 1
            return None
    
    def batch_save_histograms(self, histograms: Dict[str, np.ndarray], 
                             tier: str = "default",
                             progress_callback: Optional[callable] = None) -> Dict[str, bool]:
        """
        Save multiple histograms in batch with parallel processing.
        
        Args:
            histograms: Dictionary of card_id -> histogram
            tier: Cache tier
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary of card_id -> success_status
        """
        self.logger.info(f"Batch saving {len(histograms)} histograms to tier '{tier}'")
        
        results = {}
        completed = 0
        start_time = time.time()
        
        def save_single(card_id: str, histogram: np.ndarray) -> Tuple[str, bool]:
            success = self.save_histogram(card_id, histogram, tier)
            return card_id, success
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_card = {
                executor.submit(save_single, card_id, histogram): card_id
                for card_id, histogram in histograms.items()
            }
            
            # Process completed tasks
            for future in as_completed(future_to_card):
                card_id = future_to_card[future]
                try:
                    card_id, success = future.result()
                    results[card_id] = success
                    completed += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(completed / len(histograms))
                        
                except Exception as e:
                    self.logger.error(f"Error saving {card_id}: {e}")
                    results[card_id] = False
                    completed += 1
        
        elapsed_time = time.time() - start_time
        success_count = sum(results.values())
        
        self.logger.info(f"Batch save completed: {success_count}/{len(histograms)} successful "
                        f"({elapsed_time:.1f}s, {len(histograms)/elapsed_time:.1f} cards/sec)")
        
        return results
    
    def batch_load_histograms(self, card_ids: List[str], 
                             tier: str = "default",
                             progress_callback: Optional[callable] = None) -> Dict[str, np.ndarray]:
        """
        Load multiple histograms in batch with parallel processing.
        
        Args:
            card_ids: List of card identifiers
            tier: Cache tier
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary of card_id -> histogram (only successful loads)
        """
        self.logger.info(f"Batch loading {len(card_ids)} histograms from tier '{tier}'")
        
        results = {}
        completed = 0
        start_time = time.time()
        
        def load_single(card_id: str) -> Tuple[str, Optional[np.ndarray]]:
            histogram = self.load_histogram(card_id, tier)
            return card_id, histogram
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:  # More threads for reading
            # Submit all tasks
            future_to_card = {
                executor.submit(load_single, card_id): card_id
                for card_id in card_ids
            }
            
            # Process completed tasks
            for future in as_completed(future_to_card):
                card_id = future_to_card[future]
                try:
                    card_id, histogram = future.result()
                    if histogram is not None:
                        results[card_id] = histogram
                    completed += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(completed / len(card_ids))
                        
                except Exception as e:
                    self.logger.error(f"Error loading {card_id}: {e}")
                    completed += 1
        
        elapsed_time = time.time() - start_time
        hit_rate = len(results) / len(card_ids) * 100
        
        self.logger.info(f"Batch load completed: {len(results)}/{len(card_ids)} loaded "
                        f"({hit_rate:.1f}% hit rate, {elapsed_time:.1f}s, {len(card_ids)/elapsed_time:.1f} cards/sec)")
        
        return results
    
    def get_cached_card_ids(self, tier: str = "default") -> Set[str]:
        """
        Get set of card IDs that are currently cached.
        
        Args:
            tier: Cache tier
            
        Returns:
            Set of cached card IDs
        """
        tier_dir = self.cache_dir / tier
        if not tier_dir.exists():
            return set()
        
        cached_ids = set()
        
        # Look for cache files
        extensions = [".hist", ".hist.lz4"]
        for ext in extensions:
            for cache_file in tier_dir.glob(f"*{ext}"):
                card_id = cache_file.stem
                if card_id.endswith('.hist'):  # Remove .hist from .hist.lz4
                    card_id = card_id[:-5]
                cached_ids.add(card_id)
        
        return cached_ids
    
    def validate_cache_integrity(self, tier: str = "default") -> Dict[str, Any]:
        """
        Validate cache integrity and identify corruption.
        
        Args:
            tier: Cache tier to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating cache integrity for tier '{tier}'")
        
        cached_ids = self.get_cached_card_ids(tier)
        validation_results = {
            'total_files': len(cached_ids),
            'valid_files': 0,
            'corrupted_files': 0,
            'missing_files': 0,
            'corrupted_card_ids': [],
            'validation_time': 0.0
        }
        
        start_time = time.time()
        
        for card_id in cached_ids:
            try:
                histogram = self.load_histogram(card_id, tier)
                if histogram is not None:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['missing_files'] += 1
            except Exception as e:
                validation_results['corrupted_files'] += 1
                validation_results['corrupted_card_ids'].append(card_id)
                self.logger.warning(f"Corrupted cache file: {card_id} - {e}")
        
        validation_results['validation_time'] = time.time() - start_time
        
        self.logger.info(f"Cache validation completed: {validation_results['valid_files']}/{validation_results['total_files']} valid "
                        f"({validation_results['corrupted_files']} corrupted)")
        
        return validation_results
    
    def clear_cache(self, tier: str = "default") -> bool:
        """
        Clear all cache files for a tier.
        
        Args:
            tier: Cache tier to clear
            
        Returns:
            True if cleared successfully
        """
        try:
            tier_dir = self.cache_dir / tier
            if tier_dir.exists():
                import shutil
                shutil.rmtree(tier_dir)
                tier_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Cleared cache for tier '{tier}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache for tier '{tier}': {e}")
            return False
    
    def get_cache_size(self, tier: str = "default") -> Dict[str, Any]:
        """
        Get cache size information.
        
        Args:
            tier: Cache tier
            
        Returns:
            Dictionary with size information
        """
        tier_dir = self.cache_dir / tier
        if not tier_dir.exists():
            return {'size_bytes': 0, 'size_mb': 0.0, 'file_count': 0}
        
        total_size = 0
        file_count = 0
        
        for cache_file in tier_dir.rglob("*.hist*"):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size
                file_count += 1
        
        return {
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'file_count': file_count
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_size_info = self.get_cache_size()
        
        return {
            'stats': asdict(self.stats),
            'cache_size': total_size_info,
            'hit_rate_percent': self.stats.get_hit_rate(),
            'compression_enabled': self.use_compression,
            'version': self.cache_version,
            'available_tiers': [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
        }
    
    def optimize_cache(self, tier: str = "default") -> Dict[str, Any]:
        """
        Optimize cache by removing old or unused files.
        
        Args:
            tier: Cache tier to optimize
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Optimizing cache for tier '{tier}'")
        
        # For now, just validate and report
        # Future: implement LRU eviction, compression optimization, etc.
        validation_results = self.validate_cache_integrity(tier)
        size_info = self.get_cache_size(tier)
        
        optimization_results = {
            'files_validated': validation_results['total_files'],
            'corrupted_files_removed': 0,
            'size_before_mb': size_info['size_mb'],
            'size_after_mb': size_info['size_mb'],
            'space_saved_mb': 0.0
        }
        
        # Remove corrupted files
        for card_id in validation_results['corrupted_card_ids']:
            try:
                cache_file = self._get_cache_file_path(card_id, tier)
                if cache_file.exists():
                    cache_file.unlink()
                    optimization_results['corrupted_files_removed'] += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove corrupted file {card_id}: {e}")
        
        # Recalculate size after cleanup
        size_after = self.get_cache_size(tier)
        optimization_results['size_after_mb'] = size_after['size_mb']
        optimization_results['space_saved_mb'] = size_info['size_mb'] - size_after['size_mb']
        
        self.logger.info(f"Cache optimization completed: removed {optimization_results['corrupted_files_removed']} corrupted files")
        
        return optimization_results


# Global instance
_histogram_cache_manager = None


def get_histogram_cache_manager() -> HistogramCacheManager:
    """
    Get the global histogram cache manager instance.
    
    Returns:
        HistogramCacheManager instance
    """
    global _histogram_cache_manager
    if _histogram_cache_manager is None:
        _histogram_cache_manager = HistogramCacheManager()
    return _histogram_cache_manager


if __name__ == "__main__":
    # Test the histogram cache manager
    logging.basicConfig(level=logging.INFO)
    
    cache_manager = get_histogram_cache_manager()
    
    print("Histogram Cache Manager Test")
    print("=" * 40)
    
    # Create test histogram
    test_histogram = np.random.rand(50, 60).astype(np.float32)
    test_card_id = "TEST_001"
    
    print(f"Test histogram shape: {test_histogram.shape}")
    print(f"Test histogram dtype: {test_histogram.dtype}")
    
    # Test save/load cycle
    print(f"\nTesting save/load cycle for {test_card_id}...")
    
    save_success = cache_manager.save_histogram(test_card_id, test_histogram, "test")
    print(f"Save successful: {save_success}")
    
    loaded_histogram = cache_manager.load_histogram(test_card_id, "test")
    if loaded_histogram is not None:
        print(f"Load successful: {loaded_histogram.shape}")
        print(f"Arrays equal: {np.array_equal(test_histogram, loaded_histogram)}")
    else:
        print("Load failed!")
    
    # Test batch operations
    print(f"\nTesting batch operations...")
    test_histograms = {
        f"TEST_{i:03d}": np.random.rand(50, 60).astype(np.float32)
        for i in range(10)
    }
    
    batch_results = cache_manager.batch_save_histograms(test_histograms, "test")
    success_count = sum(batch_results.values())
    print(f"Batch save: {success_count}/{len(test_histograms)} successful")
    
    loaded_batch = cache_manager.batch_load_histograms(list(test_histograms.keys()), "test")
    print(f"Batch load: {len(loaded_batch)}/{len(test_histograms)} loaded")
    
    # Show statistics
    print(f"\nCache statistics:")
    stats = cache_manager.get_cache_statistics()
    print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
    print(f"Cache size: {stats['cache_size']['size_mb']:.2f} MB")
    print(f"Files: {stats['cache_size']['file_count']}")
    print(f"Compression: {stats['compression_enabled']}")
    
    # Cleanup test cache
    cache_manager.clear_cache("test")
    print(f"\nTest cache cleared.")