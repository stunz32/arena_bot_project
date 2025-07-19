#!/usr/bin/env python3
"""
Test script to verify cache performance
"""
import sys
sys.path.insert(0, '.')
from arena_bot.detection.ultimate_detector import get_ultimate_detector
from arena_bot.detection.feature_cache_manager import FeatureCacheManager
import time

print('Testing cache-aware Ultimate Detection loading...')
print('=' * 50)

# Check cache stats first
cache_manager = FeatureCacheManager()
stats = cache_manager.get_cache_stats()
print(f'Cache stats: {stats["total_cached_cards"]} cards, {stats["cache_size_mb"]:.1f}MB')

# Time the loading
start_time = time.time()
detector = get_ultimate_detector()
init_time = time.time() - start_time

# Time the database loading
start_time = time.time()
detector.load_card_database()
load_time = time.time() - start_time

print(f'Detector initialization: {init_time:.3f}s')
print(f'Database loading: {load_time:.3f}s')
print(f'Total time: {init_time + load_time:.3f}s')

# Show status
status = detector.get_status()
if 'feature_database_stats' in status:
    print(f'Feature database stats: {status["feature_database_stats"]}')

print('Test completed successfully!')