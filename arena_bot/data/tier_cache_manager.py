"""
Tier Cache Manager

Manages binary caching and efficient storage of HearthArena tier data.
Provides fast access to tier information with optimized serialization.
"""

import json
import logging
import pickle
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import time

try:
    from .heartharena_tier_manager import get_heartharena_tier_manager, TierData
    from .cards_json_loader import get_cards_json_loader
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent))
    from heartharena_tier_manager import get_heartharena_tier_manager, TierData
    from cards_json_loader import get_cards_json_loader


@dataclass
class TierCacheInfo:
    """Container for tier cache metadata."""
    total_entries: int
    classes_cached: int
    cache_size_bytes: int
    last_updated: str
    source_hash: str
    compression_ratio: float


@dataclass
class TierCacheStats:
    """Statistics for tier cache performance."""
    load_time_ms: float
    save_time_ms: float
    binary_size_bytes: int
    json_size_bytes: int
    compression_efficiency: float


class TierCacheManager:
    """
    Manages efficient binary caching of HearthArena tier data.
    
    Provides optimized storage, fast loading, and automatic cache management
    for tier information with both binary and JSON fallback formats.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize tier cache manager.
        
        Args:
            cache_dir: Directory for cached tier data
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "tier_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.binary_cache_file = self.cache_dir / "tier_data.bin"
        self.json_cache_file = self.cache_dir / "tier_data.json"
        self.metadata_file = self.cache_dir / "cache_info.json"
        self.stats_file = self.cache_dir / "performance_stats.json"
        
        # Configuration
        self.max_cache_age_hours = 24  # Refresh daily
        self.compression_level = 9  # Maximum compression
        self.use_binary_format = True  # Prefer binary for speed
        
        # Binary format version for compatibility
        self.binary_format_version = 1
        self.magic_header = b'HATC'  # HearthArena Tier Cache
        
        # Runtime data
        self.cached_data: Optional[Dict[str, Dict[str, TierData]]] = None
        self.cache_info: Optional[TierCacheInfo] = None
        self.tier_manager = get_heartharena_tier_manager()
        self.cards_loader = get_cards_json_loader()
        
        # Performance tracking
        self.last_stats: Optional[TierCacheStats] = None
        
        self.logger.info("TierCacheManager initialized with binary caching")
    
    def get_cache_age_hours(self) -> float:
        """Get age of cached data in hours."""
        if not self.cache_info or not self.cache_info.last_updated:
            return float('inf')
        
        try:
            last_update = datetime.fromisoformat(self.cache_info.last_updated)
            age = datetime.now() - last_update
            return age.total_seconds() / 3600  # Convert to hours
        except Exception:
            return float('inf')
    
    def is_cache_stale(self) -> bool:
        """Check if cache needs updating."""
        return self.get_cache_age_hours() > self.max_cache_age_hours
    
    def _create_binary_header(self, data_size: int) -> bytes:
        """Create binary cache file header."""
        header = struct.pack(
            '<4sII',  # magic, version, data_size
            self.magic_header,
            self.binary_format_version,
            data_size
        )
        return header
    
    def _validate_binary_header(self, data: bytes) -> Tuple[bool, int]:
        """Validate binary cache file header."""
        if len(data) < 12:  # Header size
            return False, 0
        
        try:
            magic, version, data_size = struct.unpack('<4sII', data[:12])
            
            if magic != self.magic_header:
                self.logger.warning("Invalid magic header in binary cache")
                return False, 0
            
            if version != self.binary_format_version:
                self.logger.warning(f"Unsupported binary format version: {version}")
                return False, 0
            
            return True, data_size
            
        except struct.error:
            return False, 0
    
    def save_tier_data_binary(self, tier_data: Dict[str, Dict[str, TierData]]) -> bool:
        """
        Save tier data in optimized binary format.
        
        Args:
            tier_data: Tier data to cache
            
        Returns:
            True if saved successfully
        """
        start_time = time.time()
        
        try:
            self.logger.info("💾 Saving tier data in binary format...")
            
            # Convert TierData objects to serializable format
            serializable_data = {}
            for class_name, class_tiers in tier_data.items():
                serializable_data[class_name] = {}
                for card_name, tier_info in class_tiers.items():
                    serializable_data[class_name][card_name] = {
                        'tier': tier_info.tier,
                        'tier_index': tier_info.tier_index,
                        'confidence': tier_info.confidence
                    }
            
            # Serialize with pickle for maximum efficiency
            pickled_data = pickle.dumps(serializable_data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create header
            header = self._create_binary_header(len(pickled_data))
            
            # Write binary file
            with open(self.binary_cache_file, 'wb') as f:
                f.write(header)
                f.write(pickled_data)
            
            # Also save JSON fallback
            json_start = time.time()
            with open(self.json_cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            json_time = (time.time() - json_start) * 1000
            
            # Calculate statistics
            binary_size = self.binary_cache_file.stat().st_size
            json_size = self.json_cache_file.stat().st_size
            save_time = (time.time() - start_time) * 1000
            
            # Update cache info
            total_entries = sum(len(class_tiers) for class_tiers in tier_data.values())
            source_content = json.dumps(sorted(tier_data.keys()))
            source_hash = hashlib.md5(source_content.encode()).hexdigest()[:8]
            
            self.cache_info = TierCacheInfo(
                total_entries=total_entries,
                classes_cached=len(tier_data),
                cache_size_bytes=binary_size,
                last_updated=datetime.now().isoformat(),
                source_hash=source_hash,
                compression_ratio=json_size / binary_size if binary_size > 0 else 1.0
            )
            
            # Save cache metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.cache_info), f, indent=2)
            
            # Update performance stats
            self.last_stats = TierCacheStats(
                load_time_ms=0.0,  # Not applicable for save
                save_time_ms=save_time,
                binary_size_bytes=binary_size,
                json_size_bytes=json_size,
                compression_efficiency=(json_size - binary_size) / json_size * 100
            )
            
            # Save performance stats
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.last_stats), f, indent=2)
            
            self.cached_data = tier_data
            
            self.logger.info(f"✅ Binary tier cache saved successfully:")
            self.logger.info(f"   Total entries: {total_entries}")
            self.logger.info(f"   Binary size: {binary_size:,} bytes")
            self.logger.info(f"   JSON size: {json_size:,} bytes")
            self.logger.info(f"   Compression: {self.cache_info.compression_ratio:.1f}x")
            self.logger.info(f"   Save time: {save_time:.1f}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save binary tier cache: {e}")
            return False
    
    def load_tier_data_binary(self) -> bool:
        """
        Load tier data from binary cache.
        
        Returns:
            True if loaded successfully
        """
        start_time = time.time()
        
        try:
            # Try binary format first
            if self.binary_cache_file.exists() and self.use_binary_format:
                self.logger.info("📂 Loading tier data from binary cache...")
                
                with open(self.binary_cache_file, 'rb') as f:
                    data = f.read()
                
                # Validate header
                header_valid, data_size = self._validate_binary_header(data)
                if not header_valid:
                    self.logger.warning("Invalid binary cache header, falling back to JSON")
                    return self._load_tier_data_json()
                
                # Extract and deserialize data
                pickled_data = data[12:]  # Skip header
                if len(pickled_data) != data_size:
                    self.logger.warning("Binary cache size mismatch, falling back to JSON")
                    return self._load_tier_data_json()
                
                serializable_data = pickle.loads(pickled_data)
                
            else:
                # Fallback to JSON
                return self._load_tier_data_json()
            
            # Convert back to TierData objects
            tier_data = {}
            for class_name, class_tiers in serializable_data.items():
                tier_data[class_name] = {}
                for card_name, tier_info in class_tiers.items():
                    tier_data[class_name][card_name] = TierData(
                        tier=tier_info['tier'],
                        tier_index=tier_info['tier_index'],
                        confidence=tier_info['confidence']
                    )
            
            # Load cache metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.cache_info = TierCacheInfo(**metadata)
            
            # Calculate load performance
            load_time = (time.time() - start_time) * 1000
            
            self.cached_data = tier_data
            
            cache_age = self.get_cache_age_hours()
            total_entries = sum(len(class_tiers) for class_tiers in tier_data.values())
            
            self.logger.info(f"✅ Binary tier cache loaded successfully:")
            self.logger.info(f"   Total entries: {total_entries}")
            self.logger.info(f"   Classes: {len(tier_data)}")
            self.logger.info(f"   Cache age: {cache_age:.1f} hours")
            self.logger.info(f"   Load time: {load_time:.1f}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load binary tier cache: {e}")
            return self._load_tier_data_json()
    
    def _load_tier_data_json(self) -> bool:
        """Fallback method to load tier data from JSON."""
        try:
            if not self.json_cache_file.exists():
                self.logger.info("No JSON tier cache found")
                return False
            
            self.logger.info("📂 Loading tier data from JSON fallback...")
            
            with open(self.json_cache_file, 'r', encoding='utf-8') as f:
                serializable_data = json.load(f)
            
            # Convert back to TierData objects
            tier_data = {}
            for class_name, class_tiers in serializable_data.items():
                tier_data[class_name] = {}
                for card_name, tier_info in class_tiers.items():
                    tier_data[class_name][card_name] = TierData(
                        tier=tier_info['tier'],
                        tier_index=tier_info['tier_index'],
                        confidence=tier_info['confidence']
                    )
            
            self.cached_data = tier_data
            self.logger.info(f"✅ JSON tier cache loaded as fallback")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load JSON tier cache: {e}")
            return False
    
    def update_tier_cache(self, force: bool = False) -> bool:
        """
        Update tier cache from HearthArena.
        
        Args:
            force: Force update even if cache is fresh
            
        Returns:
            True if update successful
        """
        if not force and not self.is_cache_stale():
            self.logger.info("Tier cache is fresh, skipping update")
            return True
        
        self.logger.info("🚀 Starting tier cache update...")
        
        # Update tier data through manager
        tier_update_success = self.tier_manager.update_tier_data(force=force)
        if not tier_update_success:
            self.logger.error("❌ Failed to update tier data from HearthArena")
            return False
        
        # Get updated tier data
        tier_stats = self.tier_manager.get_tier_statistics()
        if tier_stats['status'] != 'loaded':
            self.logger.error("❌ No tier data available after update")
            return False
        
        # Collect all tier data
        all_tier_data = {}
        for class_name in self.tier_manager.class_names:
            class_tiers = self.tier_manager.get_class_tiers(class_name)
            if class_tiers:
                all_tier_data[class_name] = class_tiers
        
        # Save to binary cache
        if self.save_tier_data_binary(all_tier_data):
            self.logger.info("🎯 Tier cache update completed successfully")
            return True
        else:
            return False
    
    def get_class_tiers(self, class_name: str) -> Dict[str, TierData]:
        """
        Get tier data for a specific class.
        
        Args:
            class_name: Hero class name
            
        Returns:
            Dictionary mapping card names to tier data
        """
        if not self.cached_data:
            # Try to load from cache
            if not self.load_tier_data_binary():
                return {}
        
        return self.cached_data.get(class_name, {})
    
    def get_card_tier(self, card_name: str, class_name: str) -> Optional[TierData]:
        """
        Get tier information for a specific card.
        
        Args:
            card_name: Name of the card
            class_name: Hero class name
            
        Returns:
            TierData if found, None otherwise
        """
        class_tiers = self.get_class_tiers(class_name)
        return class_tiers.get(card_name)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.cache_info:
            if not self.load_tier_data_binary():
                return {'status': 'no_cache'}
        
        stats = {
            'status': 'loaded',
            'cache_age_hours': self.get_cache_age_hours(),
            'needs_update': self.is_cache_stale(),
            'total_entries': self.cache_info.total_entries,
            'classes_cached': self.cache_info.classes_cached,
            'cache_size_bytes': self.cache_info.cache_size_bytes,
            'compression_ratio': self.cache_info.compression_ratio,
            'source_hash': self.cache_info.source_hash
        }
        
        # Add performance stats if available
        if self.last_stats:
            stats['performance'] = asdict(self.last_stats)
        
        return stats
    
    def clear_cache(self) -> bool:
        """Clear all cached tier data."""
        try:
            files_to_remove = [
                self.binary_cache_file,
                self.json_cache_file,
                self.metadata_file,
                self.stats_file
            ]
            
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
            
            self.cached_data = None
            self.cache_info = None
            self.last_stats = None
            
            self.logger.info("✅ Tier cache cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear tier cache: {e}")
            return False
    
    def has_data(self) -> bool:
        """Check if tier data is available."""
        if self.cached_data:
            return True
        
        return self.load_tier_data_binary()


# Global instance
_tier_cache_manager = None


def get_tier_cache_manager() -> TierCacheManager:
    """
    Get the global tier cache manager instance.
    
    Returns:
        TierCacheManager instance
    """
    global _tier_cache_manager
    if _tier_cache_manager is None:
        _tier_cache_manager = TierCacheManager()
    return _tier_cache_manager


if __name__ == "__main__":
    # Test the tier cache manager
    logging.basicConfig(level=logging.INFO)
    
    cache_manager = get_tier_cache_manager()
    
    print("Tier Cache Manager Test")
    print("=" * 40)
    
    # Show current status
    stats = cache_manager.get_cache_statistics()
    print(f"Status: {stats['status']}")
    
    if stats['status'] == 'loaded':
        print(f"Cache age: {stats['cache_age_hours']:.1f} hours")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Classes cached: {stats['classes_cached']}")
        print(f"Cache size: {stats['cache_size_bytes']:,} bytes")
        print(f"Compression: {stats['compression_ratio']:.1f}x")
        
        if 'performance' in stats:
            perf = stats['performance']
            print(f"\nPerformance:")
            print(f"  Save time: {perf['save_time_ms']:.1f}ms")
            print(f"  Binary size: {perf['binary_size_bytes']:,} bytes")
            print(f"  JSON size: {perf['json_size_bytes']:,} bytes")
            print(f"  Efficiency: {perf['compression_efficiency']:.1f}%")
    
    # Check if update needed
    if stats.get('needs_update', True):
        print(f"\nUpdating tier cache...")
        success = cache_manager.update_tier_cache(force=True)
        if success:
            print("✅ Cache update completed successfully!")
            
            # Show updated stats
            stats = cache_manager.get_cache_statistics()
            print(f"New cache size: {stats['cache_size_bytes']:,} bytes")
            print(f"Total entries: {stats['total_entries']}")
        else:
            print("❌ Cache update failed!")
    
    # Test tier lookup
    print(f"\nTesting tier lookup...")
    mage_tiers = cache_manager.get_class_tiers('mage')
    if mage_tiers:
        print(f"Mage cards with tiers: {len(mage_tiers)}")
        # Show first few as examples
        for i, (card_name, tier_data) in enumerate(mage_tiers.items()):
            if i >= 3:
                break
            print(f"  {card_name}: {tier_data.tier}")
    else:
        print("No mage tier data found")
    
    print(f"\n🎯 Binary tier caching operational!")