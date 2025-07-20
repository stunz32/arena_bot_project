"""
Intelligent Cache Manager - Advanced Caching with Data Synchronization

Provides intelligent caching for hero and card data with automatic synchronization,
smart invalidation, memory optimization, and performance monitoring.
"""

import logging
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import time
from collections import OrderedDict
import gzip
import sqlite3
from enum import Enum


class CacheStatus(Enum):
    """Cache entry status."""
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    data: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    compression_ratio: float = 1.0
    checksum: str = ""
    source: str = ""
    dependencies: List[str] = None


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    sync_count: int
    memory_usage_mb: float
    compression_ratio: float
    average_access_time_ms: float


class IntelligentCacheManager:
    """
    Advanced caching system with intelligent features.
    
    Provides automatic data synchronization, smart eviction policies,
    compression, dependency tracking, and performance optimization.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_memory_mb: int = 256):
        """Initialize intelligent cache manager."""
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory management
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        
        # Cache storage
        self.memory_cache = OrderedDict()  # LRU cache
        self.persistent_cache_file = self.cache_dir / "persistent_cache.db"
        
        # Performance tracking
        self.statistics = CacheStatistics(
            total_entries=0, total_size_bytes=0, hit_count=0, miss_count=0,
            eviction_count=0, sync_count=0, memory_usage_mb=0.0,
            compression_ratio=1.0, average_access_time_ms=0.0
        )
        
        # Synchronization settings
        self.sync_intervals = {
            'hero_data': 12 * 3600,  # 12 hours
            'card_data': 24 * 3600,  # 24 hours
            'meta_data': 6 * 3600,   # 6 hours
            'user_data': 1 * 3600    # 1 hour
        }
        
        # Background sync
        self.sync_enabled = True
        self.sync_thread = None
        self.sync_lock = threading.Lock()
        
        # Dependency tracking
        self.dependency_graph = {}
        
        # Initialize persistent storage
        self._init_persistent_storage()
        
        # Start background sync
        self._start_background_sync()
        
        self.logger.info(f"IntelligentCacheManager initialized with {max_memory_mb}MB memory limit")
    
    def get(self, key: str, default: Any = None, refresh_if_stale: bool = True) -> Any:
        """Get cached data with intelligent refresh logic."""
        start_time = time.time()
        
        try:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Update access info
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Move to end (most recently used)
                self.memory_cache.move_to_end(key)
                
                # Check if data is fresh
                status = self._get_cache_status(entry)
                
                if status == CacheStatus.FRESH:
                    self.statistics.hit_count += 1
                    return entry.data
                elif status == CacheStatus.STALE and refresh_if_stale:
                    # Trigger background refresh but return stale data
                    self._schedule_refresh(key)
                    self.statistics.hit_count += 1
                    return entry.data
                elif status == CacheStatus.EXPIRED:
                    # Remove expired entry
                    self._evict_entry(key)
                
            # Check persistent cache
            persistent_data = self._get_from_persistent_cache(key)
            if persistent_data:
                # Load into memory cache
                self.set(key, persistent_data['data'], ttl_seconds=persistent_data['ttl_seconds'])
                self.statistics.hit_count += 1
                return persistent_data['data']
            
            # Cache miss
            self.statistics.miss_count += 1
            return default
            
        finally:
            access_time = (time.time() - start_time) * 1000
            self._update_access_time_stats(access_time)
    
    def set(self, key: str, data: Any, ttl_seconds: int = 3600, 
           source: str = "", dependencies: Optional[List[str]] = None,
           compress: bool = True, persist: bool = True) -> bool:
        """Set cached data with advanced options."""
        try:
            # Serialize and compress data
            serialized_data, size_bytes, compression_ratio = self._serialize_data(data, compress)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                compression_ratio=compression_ratio,
                checksum=self._calculate_checksum(serialized_data),
                source=source,
                dependencies=dependencies or []
            )
            
            # Check memory constraints
            if not self._ensure_memory_capacity(size_bytes):
                self.logger.warning(f"Failed to ensure memory capacity for {key}")
                return False
            
            # Store in memory cache
            self.memory_cache[key] = entry
            self.current_memory_bytes += size_bytes
            self.statistics.total_entries += 1
            self.statistics.total_size_bytes += size_bytes
            
            # Update dependency graph
            self._update_dependencies(key, dependencies or [])
            
            # Persist if requested
            if persist:
                self._save_to_persistent_cache(key, entry, serialized_data)
            
            self.logger.debug(f"Cached {key}: {size_bytes} bytes, compression: {compression_ratio:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache entry {key}: {e}")
            return False
    
    def invalidate(self, key: str, cascade: bool = True) -> bool:
        """Invalidate cache entry and optionally its dependents."""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                entry = self.memory_cache.pop(key)
                self.current_memory_bytes -= entry.size_bytes
                self.statistics.total_entries -= 1
                self.statistics.total_size_bytes -= entry.size_bytes
            
            # Remove from persistent cache
            self._remove_from_persistent_cache(key)
            
            # Cascade invalidation to dependents
            if cascade:
                dependents = self._get_dependents(key)
                for dependent in dependents:
                    self.invalidate(dependent, cascade=False)
            
            # Update dependency graph
            self._remove_from_dependencies(key)
            
            self.logger.debug(f"Invalidated cache entry: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error invalidating cache entry {key}: {e}")
            return False
    
    def synchronize_data(self, data_type: str, force: bool = False) -> bool:
        """Synchronize specific data type with external sources."""
        with self.sync_lock:
            try:
                last_sync_key = f"_last_sync_{data_type}"
                last_sync = self.get(last_sync_key)
                
                # Check if sync is needed
                if not force and last_sync:
                    time_since_sync = (datetime.now() - last_sync).total_seconds()
                    if time_since_sync < self.sync_intervals.get(data_type, 3600):
                        return True  # No sync needed
                
                # Perform synchronization based on data type
                success = False
                if data_type == 'hero_data':
                    success = self._sync_hero_data()
                elif data_type == 'card_data':
                    success = self._sync_card_data()
                elif data_type == 'meta_data':
                    success = self._sync_meta_data()
                elif data_type == 'user_data':
                    success = self._sync_user_data()
                
                if success:
                    # Update last sync time
                    self.set(last_sync_key, datetime.now(), ttl_seconds=365*24*3600)  # 1 year
                    self.statistics.sync_count += 1
                    self.logger.info(f"Successfully synchronized {data_type}")
                else:
                    self.logger.warning(f"Failed to synchronize {data_type}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error synchronizing {data_type}: {e}")
                return False
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance and memory usage."""
        try:
            optimization_results = {
                'entries_before': len(self.memory_cache),
                'memory_before_mb': self.current_memory_bytes / (1024 * 1024),
                'actions_taken': []
            }
            
            # 1. Remove expired entries
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if self._get_cache_status(entry) == CacheStatus.EXPIRED:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.invalidate(key, cascade=False)
            
            if expired_keys:
                optimization_results['actions_taken'].append(f"Removed {len(expired_keys)} expired entries")
            
            # 2. Compress uncompressed data
            compressed_count = 0
            for key, entry in list(self.memory_cache.items()):
                if entry.compression_ratio >= 1.0 and entry.size_bytes > 1024:  # 1KB threshold
                    # Recompress the data
                    new_data, new_size, new_ratio = self._serialize_data(entry.data, compress=True)
                    if new_ratio < entry.compression_ratio:
                        entry.size_bytes = new_size
                        entry.compression_ratio = new_ratio
                        compressed_count += 1
            
            if compressed_count > 0:
                optimization_results['actions_taken'].append(f"Compressed {compressed_count} entries")
            
            # 3. Evict least recently used entries if over memory limit
            evicted_count = 0
            while self.current_memory_bytes > self.max_memory_bytes and self.memory_cache:
                # Get least recently used entry
                lru_key = next(iter(self.memory_cache))
                self._evict_entry(lru_key)
                evicted_count += 1
            
            if evicted_count > 0:
                optimization_results['actions_taken'].append(f"Evicted {evicted_count} LRU entries")
            
            # 4. Defragment persistent cache
            if self._defragment_persistent_cache():
                optimization_results['actions_taken'].append("Defragmented persistent cache")
            
            # Update final stats
            optimization_results.update({
                'entries_after': len(self.memory_cache),
                'memory_after_mb': self.current_memory_bytes / (1024 * 1024),
                'memory_freed_mb': optimization_results['memory_before_mb'] - (self.current_memory_bytes / (1024 * 1024))
            })
            
            self.logger.info(f"Cache optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing cache: {e}")
            return {'error': str(e)}
    
    def get_statistics(self) -> CacheStatistics:
        """Get comprehensive cache statistics."""
        # Update current stats
        self.statistics.total_entries = len(self.memory_cache)
        self.statistics.total_size_bytes = self.current_memory_bytes
        self.statistics.memory_usage_mb = self.current_memory_bytes / (1024 * 1024)
        
        # Calculate compression ratio
        if self.memory_cache:
            total_compression = sum(entry.compression_ratio for entry in self.memory_cache.values())
            self.statistics.compression_ratio = total_compression / len(self.memory_cache)
        
        return self.statistics
    
    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health and performance metrics."""
        stats = self.get_statistics()
        
        # Calculate health metrics
        hit_rate = stats.hit_count / (stats.hit_count + stats.miss_count) if (stats.hit_count + stats.miss_count) > 0 else 0
        memory_utilization = stats.memory_usage_mb / (self.max_memory_bytes / (1024 * 1024))
        
        # Determine health status
        health_score = 0
        if hit_rate >= 0.8:
            health_score += 30
        elif hit_rate >= 0.6:
            health_score += 20
        elif hit_rate >= 0.4:
            health_score += 10
        
        if memory_utilization <= 0.8:
            health_score += 25
        elif memory_utilization <= 0.9:
            health_score += 15
        elif memory_utilization <= 0.95:
            health_score += 5
        
        if stats.compression_ratio <= 0.7:
            health_score += 25
        elif stats.compression_ratio <= 0.8:
            health_score += 15
        elif stats.compression_ratio <= 0.9:
            health_score += 10
        
        if stats.average_access_time_ms <= 10:
            health_score += 20
        elif stats.average_access_time_ms <= 50:
            health_score += 10
        elif stats.average_access_time_ms <= 100:
            health_score += 5
        
        if health_score >= 90:
            health_status = "Excellent"
        elif health_score >= 70:
            health_status = "Good"
        elif health_score >= 50:
            health_status = "Fair"
        else:
            health_status = "Poor"
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'hit_rate': hit_rate,
            'memory_utilization': memory_utilization,
            'recommendations': self._get_health_recommendations(health_score, hit_rate, memory_utilization)
        }
    
    def shutdown(self):
        """Shutdown cache manager gracefully."""
        try:
            # Stop background sync
            self.sync_enabled = False
            if self.sync_thread and self.sync_thread.is_alive():
                self.sync_thread.join(timeout=5)
            
            # Save final state to persistent cache
            self._save_all_to_persistent_cache()
            
            self.logger.info("Cache manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during cache shutdown: {e}")
    
    # === INTERNAL METHODS ===
    
    def _init_persistent_storage(self):
        """Initialize SQLite database for persistent caching."""
        try:
            conn = sqlite3.connect(self.persistent_cache_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp TEXT,
                    ttl_seconds INTEGER,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    compression_ratio REAL,
                    checksum TEXT,
                    source TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing persistent storage: {e}")
    
    def _get_cache_status(self, entry: CacheEntry) -> CacheStatus:
        """Determine cache entry status."""
        age_seconds = (datetime.now() - entry.timestamp).total_seconds()
        
        if age_seconds < entry.ttl_seconds * 0.8:
            return CacheStatus.FRESH
        elif age_seconds < entry.ttl_seconds:
            return CacheStatus.STALE
        else:
            return CacheStatus.EXPIRED
    
    def _serialize_data(self, data: Any, compress: bool = True) -> Tuple[bytes, int, float]:
        """Serialize and optionally compress data."""
        # Serialize using pickle
        serialized = pickle.dumps(data)
        original_size = len(serialized)
        
        if compress and original_size > 1024:  # Only compress if > 1KB
            # Compress using gzip
            compressed = gzip.compress(serialized)
            compression_ratio = len(compressed) / original_size
            
            if compression_ratio < 0.9:  # Only use if significant compression
                return compressed, len(compressed), compression_ratio
        
        return serialized, original_size, 1.0
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate MD5 checksum for data integrity."""
        return hashlib.md5(data).hexdigest()
    
    def _ensure_memory_capacity(self, required_bytes: int) -> bool:
        """Ensure sufficient memory capacity by evicting if necessary."""
        if self.current_memory_bytes + required_bytes <= self.max_memory_bytes:
            return True
        
        # Calculate how much memory needs to be freed
        bytes_to_free = (self.current_memory_bytes + required_bytes) - self.max_memory_bytes
        
        # Evict least recently used entries
        evicted_bytes = 0
        while evicted_bytes < bytes_to_free and self.memory_cache:
            lru_key = next(iter(self.memory_cache))
            entry = self.memory_cache[lru_key]
            evicted_bytes += entry.size_bytes
            self._evict_entry(lru_key)
        
        return evicted_bytes >= bytes_to_free
    
    def _evict_entry(self, key: str):
        """Evict entry from memory cache."""
        if key in self.memory_cache:
            entry = self.memory_cache.pop(key)
            self.current_memory_bytes -= entry.size_bytes
            self.statistics.eviction_count += 1
            self.logger.debug(f"Evicted cache entry: {key}")
    
    def _update_dependencies(self, key: str, dependencies: List[str]):
        """Update dependency graph."""
        # Add key to dependency graph
        if key not in self.dependency_graph:
            self.dependency_graph[key] = {'depends_on': set(), 'dependents': set()}
        
        # Update dependencies
        self.dependency_graph[key]['depends_on'] = set(dependencies)
        
        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self.dependency_graph:
                self.dependency_graph[dep] = {'depends_on': set(), 'dependents': set()}
            self.dependency_graph[dep]['dependents'].add(key)
    
    def _get_dependents(self, key: str) -> List[str]:
        """Get all keys that depend on the given key."""
        if key in self.dependency_graph:
            return list(self.dependency_graph[key]['dependents'])
        return []
    
    def _remove_from_dependencies(self, key: str):
        """Remove key from dependency graph."""
        if key in self.dependency_graph:
            # Remove from dependents of dependencies
            for dep in self.dependency_graph[key]['depends_on']:
                if dep in self.dependency_graph:
                    self.dependency_graph[dep]['dependents'].discard(key)
            
            # Remove key
            del self.dependency_graph[key]
    
    def _schedule_refresh(self, key: str):
        """Schedule background refresh for stale data."""
        # This would trigger background data fetching
        # Implementation depends on specific data sources
        self.logger.debug(f"Scheduled refresh for stale key: {key}")
    
    def _get_from_persistent_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get entry from persistent cache."""
        try:
            conn = sqlite3.connect(self.persistent_cache_file)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM cache_entries WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return {
                    'key': row[0],
                    'data': pickle.loads(row[1]),
                    'timestamp': datetime.fromisoformat(row[2]),
                    'ttl_seconds': row[3],
                    'access_count': row[4],
                    'size_bytes': row[5],
                    'compression_ratio': row[6],
                    'checksum': row[7],
                    'source': row[8]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from persistent cache: {e}")
            return None
    
    def _save_to_persistent_cache(self, key: str, entry: CacheEntry, serialized_data: bytes):
        """Save entry to persistent cache."""
        try:
            conn = sqlite3.connect(self.persistent_cache_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, data, timestamp, ttl_seconds, access_count, size_bytes, compression_ratio, checksum, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key, serialized_data, entry.timestamp.isoformat(), entry.ttl_seconds,
                entry.access_count, entry.size_bytes, entry.compression_ratio,
                entry.checksum, entry.source
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving to persistent cache: {e}")
    
    def _remove_from_persistent_cache(self, key: str):
        """Remove entry from persistent cache."""
        try:
            conn = sqlite3.connect(self.persistent_cache_file)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error removing from persistent cache: {e}")
    
    def _save_all_to_persistent_cache(self):
        """Save all memory cache entries to persistent cache."""
        try:
            for key, entry in self.memory_cache.items():
                serialized_data, _, _ = self._serialize_data(entry.data, compress=True)
                self._save_to_persistent_cache(key, entry, serialized_data)
            
        except Exception as e:
            self.logger.error(f"Error saving all to persistent cache: {e}")
    
    def _defragment_persistent_cache(self) -> bool:
        """Defragment and optimize persistent cache."""
        try:
            conn = sqlite3.connect(self.persistent_cache_file)
            cursor = conn.cursor()
            
            # Remove expired entries
            current_time = datetime.now()
            cursor.execute("""
                DELETE FROM cache_entries 
                WHERE datetime(timestamp, '+' || ttl_seconds || ' seconds') < ?
            """, (current_time.isoformat(),))
            
            # Vacuum database
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error defragmenting persistent cache: {e}")
            return False
    
    def _start_background_sync(self):
        """Start background synchronization thread."""
        def sync_worker():
            while self.sync_enabled:
                try:
                    # Sync different data types on different schedules
                    current_time = time.time()
                    
                    for data_type, interval in self.sync_intervals.items():
                        last_sync_key = f"_last_sync_{data_type}"
                        last_sync = self.get(last_sync_key)
                        
                        if not last_sync or (current_time - last_sync.timestamp()) > interval:
                            self.synchronize_data(data_type)
                    
                    # Sleep for 5 minutes before next check
                    time.sleep(300)
                    
                except Exception as e:
                    self.logger.error(f"Error in background sync: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        self.sync_thread = threading.Thread(target=sync_worker, daemon=True)
        self.sync_thread.start()
    
    def _sync_hero_data(self) -> bool:
        """Synchronize hero data from external sources."""
        # Implementation would fetch fresh hero data from HSReplay API
        # For now, return placeholder
        self.logger.debug("Synchronizing hero data")
        return True
    
    def _sync_card_data(self) -> bool:
        """Synchronize card data from external sources."""
        # Implementation would fetch fresh card data from HSReplay API
        # For now, return placeholder
        self.logger.debug("Synchronizing card data")
        return True
    
    def _sync_meta_data(self) -> bool:
        """Synchronize meta analysis data."""
        # Implementation would fetch fresh meta data
        # For now, return placeholder
        self.logger.debug("Synchronizing meta data")
        return True
    
    def _sync_user_data(self) -> bool:
        """Synchronize user preference and performance data."""
        # Implementation would sync user data
        # For now, return placeholder
        self.logger.debug("Synchronizing user data")
        return True
    
    def _update_access_time_stats(self, access_time_ms: float):
        """Update average access time statistics."""
        total_accesses = self.statistics.hit_count + self.statistics.miss_count
        if total_accesses > 0:
            current_total = self.statistics.average_access_time_ms * (total_accesses - 1)
            self.statistics.average_access_time_ms = (current_total + access_time_ms) / total_accesses
    
    def _get_health_recommendations(self, health_score: int, hit_rate: float, memory_utilization: float) -> List[str]:
        """Get recommendations for improving cache health."""
        recommendations = []
        
        if hit_rate < 0.6:
            recommendations.append("Consider increasing cache TTL or improving cache key design")
        
        if memory_utilization > 0.9:
            recommendations.append("Consider increasing memory limit or improving eviction policy")
        
        if health_score < 70:
            recommendations.append("Run cache optimization to improve performance")
        
        if self.statistics.compression_ratio > 0.8:
            recommendations.append("Enable compression for better memory utilization")
        
        if self.statistics.average_access_time_ms > 50:
            recommendations.append("Consider reducing data serialization complexity")
        
        return recommendations


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> IntelligentCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = IntelligentCacheManager()
    return _cache_manager