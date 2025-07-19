#!/usr/bin/env python3
"""
BUILD FEATURE CACHE
One-time script to compute and cache all feature descriptors for Ultimate Detection Engine.
This eliminates the GUI freeze by pre-computing expensive feature data.
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arena_bot.utils.asset_loader import AssetLoader
from arena_bot.detection.feature_cache_manager import FeatureCacheManager
from arena_bot.detection.feature_ensemble import FreeAlgorithmEnsemble, PatentFreeFeatureDetector

def setup_logging():
    """Setup logging for the cache builder."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / "logs" / "feature_cache_build.log")
        ]
    )
    return logging.getLogger(__name__)

def get_card_list(asset_loader: AssetLoader, max_cards: Optional[int] = None) -> List[str]:
    """Get list of card codes to process."""
    try:
        # Load all available cards
        card_images = asset_loader.load_all_cards(max_cards=max_cards)
        card_codes = list(card_images.keys())
        
        logger.info(f"Found {len(card_codes)} cards to process")
        if max_cards:
            logger.info(f"Limited to {max_cards} cards for testing")
        return card_codes
        
    except Exception as e:
        logger.error(f"Failed to load card list: {e}")
        raise

def build_algorithm_cache(card_codes: List[str], asset_loader: AssetLoader, 
                         cache_manager: FeatureCacheManager, algorithm: str):
    """
    Build cache for a specific algorithm.
    
    Args:
        card_codes: List of card codes to process
        asset_loader: Asset loader instance
        cache_manager: Cache manager instance
        algorithm: Algorithm name (orb, sift, brisk, akaze)
    """
    logger.info(f"🔄 Building {algorithm.upper()} feature cache...")
    
    # Initialize feature detector for this algorithm
    detector = PatentFreeFeatureDetector(algorithm=algorithm, use_cache=True)
    
    cached_count = 0
    skipped_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for i, card_code in enumerate(card_codes):
        try:
            # Check if already cached
            if cache_manager.is_cached(card_code, algorithm):
                skipped_count += 1
                if (i + 1) % 100 == 0:
                    logger.info(f"  {algorithm.upper()}: {i+1}/{len(card_codes)} processed (cached: {cached_count}, skipped: {skipped_count})")
                continue
            
            # Load card image
            card_image = asset_loader.load_card_image(card_code)
            if card_image is None:
                logger.warning(f"Failed to load image for {card_code}")
                failed_count += 1
                continue
            
            # Compute features
            result = detector.compute_features(card_image)
            if result is not None:
                keypoints, descriptors = result
            else:
                keypoints, descriptors = None, None
            
            # Cache the results
            if keypoints is not None and descriptors is not None:
                cache_manager.save_features(card_code, algorithm, keypoints, descriptors)
                cached_count += 1
            else:
                logger.warning(f"No features computed for {card_code} with {algorithm}")
                failed_count += 1
            
            # Progress update
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(card_codes) - i - 1) / rate if rate > 0 else 0
                
                logger.info(f"  {algorithm.upper()}: {i+1}/{len(card_codes)} processed "
                           f"(cached: {cached_count}, skipped: {skipped_count}, failed: {failed_count}) "
                           f"[{rate:.1f} cards/sec, ETA: {eta/60:.1f}min]")
            
        except Exception as e:
            logger.error(f"Failed to process {card_code} for {algorithm}: {e}")
            failed_count += 1
    
    elapsed = time.time() - start_time
    logger.info(f"✅ {algorithm.upper()} cache complete: {cached_count} cached, "
               f"{skipped_count} skipped, {failed_count} failed in {elapsed/60:.1f}min")

def main():
    """Main cache building function."""
    global logger
    logger = setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build feature cache for Ultimate Detection Engine')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit the number of cards to cache for testing (default: all cards)')
    parser.add_argument('--algorithms', nargs='+', default=['orb', 'brisk', 'akaze', 'sift'],
                       help='Algorithms to build cache for (default: all)')
    args = parser.parse_args()
    
    logger.info("🚀 Starting Ultimate Detection Feature Cache Build")
    logger.info("=" * 80)
    
    if args.limit:
        logger.info(f"⚠️ TESTING MODE: Limited to {args.limit} cards")
    else:
        logger.info("🎯 PRODUCTION MODE: Building cache for all cards")
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        
        # Create logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Initialize asset loader
        asset_loader = AssetLoader()
        logger.info("✅ AssetLoader initialized")
        
        # Initialize cache manager
        cache_manager = FeatureCacheManager()
        logger.info("✅ FeatureCacheManager initialized")
        
        # Get card list
        card_codes = get_card_list(asset_loader, max_cards=args.limit)
        logger.info(f"✅ Loaded {len(card_codes)} card codes")
        
        # Display cache stats before building
        stats = cache_manager.get_cache_stats()
        logger.info(f"📊 Pre-build cache stats: {stats['total_cached_cards']} cards, "
                   f"{stats['cache_size_mb']:.1f}MB")
        
        # Build cache for each algorithm
        algorithms = args.algorithms  # Use command line specified algorithms
        
        total_start_time = time.time()
        
        for algorithm in algorithms:
            logger.info(f"\n🎯 Processing {algorithm.upper()} algorithm...")
            build_algorithm_cache(card_codes, asset_loader, cache_manager, algorithm)
        
        # Final statistics
        total_elapsed = time.time() - total_start_time
        final_stats = cache_manager.get_cache_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 FEATURE CACHE BUILD COMPLETE!")
        logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
        logger.info(f"Final cache stats:")
        logger.info(f"  Total cached cards: {final_stats['total_cached_cards']}")
        logger.info(f"  Cache size: {final_stats['cache_size_mb']:.1f}MB")
        logger.info(f"  Cache directory: {final_stats['cache_directory']}")
        
        for algo, algo_stats in final_stats['algorithms'].items():
            logger.info(f"  {algo.upper()}: {algo_stats['cached_cards']} cards, "
                       f"{algo_stats['size_mb']:.1f}MB")
        
        logger.info("\n✅ Ultimate Detection Engine will now load instantly!")
        logger.info("✅ GUI freeze issue has been eliminated!")
        
    except Exception as e:
        logger.error(f"❌ Cache build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()