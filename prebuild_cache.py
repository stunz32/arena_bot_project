#!/usr/bin/env python3
"""
Prebuild Histogram Cache Script

This script prebuilds the histogram cache for Arena Bot to dramatically reduce
initialization time. Instead of waiting 30+ seconds for cache building during
bot startup, we can prebuild it once and have instant startup times.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add arena_bot to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup logging for cache building."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def prebuild_histogram_cache():
    """Prebuild the histogram cache for fast bot initialization."""
    print("🚀 Arena Bot Cache Prebuilder")
    print("=" * 60)
    
    try:
        # Import after adding to path
        from arena_bot.utils.asset_loader import AssetLoader
        from arena_bot.detection.histogram_matcher import HistogramMatcher
        from arena_bot.utils.histogram_cache import HistogramCacheManager
        
        print("📦 Initializing components...")
        
        # Initialize components
        asset_loader = AssetLoader()
        cache_manager = HistogramCacheManager()
        histogram_matcher = HistogramMatcher(use_cache=True)
        
        print("✅ Components initialized")
        
        # Get available cards
        available_cards = asset_loader.get_available_cards()
        total_cards = len(available_cards)
        print(f"📋 Found {total_cards} total card images")
        
        # Check existing cache
        existing_cache = cache_manager.get_cached_card_ids("default")
        print(f"💾 Existing cache: {len(existing_cache)} cards")
        
        # Determine cards that need caching
        cards_to_cache = [card for card in available_cards if card not in existing_cache]
        
        if not cards_to_cache:
            print("✨ Cache is already complete!")
            return True
        
        print(f"🔄 Need to cache {len(cards_to_cache)} cards")
        
        # Load card images in batches to avoid memory issues
        batch_size = 100
        total_batches = (len(cards_to_cache) + batch_size - 1) // batch_size
        
        print(f"📦 Processing {total_batches} batches of {batch_size} cards each...")
        
        start_time = time.time()
        processed_cards = 0
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(cards_to_cache))
            batch_cards = cards_to_cache[batch_start:batch_end]
            
            print(f"\n📦 Batch {batch_num + 1}/{total_batches}: Processing cards {batch_start + 1}-{batch_end}")
            
            # Load card images for this batch
            card_images = {}
            for card_code in batch_cards:
                # Load both normal and premium versions
                normal_image = asset_loader.load_card_image(card_code, premium=False)
                if normal_image is not None:
                    card_images[card_code] = normal_image
                
                premium_image = asset_loader.load_card_image(card_code, premium=True)
                if premium_image is not None:
                    card_images[f"{card_code}_premium"] = premium_image
            
            print(f"   📋 Loaded {len(card_images)} images for batch")
            
            # Compute histograms for batch
            batch_histograms = {}
            for card_id, image in card_images.items():
                hist = histogram_matcher.compute_histogram(image)
                if hist is not None:
                    # Remove _premium suffix for cache key
                    cache_key = card_id.replace("_premium", "")
                    batch_histograms[cache_key] = hist
            
            print(f"   🧮 Computed {len(batch_histograms)} histograms")
            
            # Save batch to cache
            if batch_histograms:
                def progress_callback(progress):
                    if progress % 0.2 == 0:  # Every 20%
                        print(f"   💾 Cache progress: {progress:.0%}")
                
                save_results = cache_manager.batch_save_histograms(
                    batch_histograms, 
                    tier="default",
                    progress_callback=progress_callback
                )
                
                success_count = sum(save_results.values())
                print(f"   ✅ Saved {success_count}/{len(batch_histograms)} histograms to cache")
                
                processed_cards += len(batch_cards)
                
                # Progress update
                overall_progress = processed_cards / len(cards_to_cache)
                elapsed = time.time() - start_time
                estimated_total = elapsed / overall_progress if overall_progress > 0 else 0
                remaining = estimated_total - elapsed
                
                print(f"   📊 Overall progress: {overall_progress:.1%} "
                      f"({processed_cards}/{len(cards_to_cache)} cards)")
                if remaining > 0:
                    print(f"   ⏱️  Estimated time remaining: {remaining:.0f}s")
            
            # Clear memory
            del card_images
            del batch_histograms
        
        # Final statistics
        total_time = time.time() - start_time
        final_cache_size = len(cache_manager.get_cached_card_ids("default"))
        
        print(f"\n🎉 Cache prebuilding completed!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Cards processed: {processed_cards}")
        print(f"💾 Final cache size: {final_cache_size} cards")
        print(f"⚡ Processing rate: {processed_cards/total_time:.1f} cards/sec")
        
        # Validate cache
        print(f"\n🔍 Validating cache integrity...")
        validation_results = cache_manager.validate_cache_integrity("default")
        
        if validation_results['corrupted_files'] == 0:
            print("✅ Cache validation passed - no corrupted files")
        else:
            print(f"⚠️  Found {validation_results['corrupted_files']} corrupted files")
        
        # Cache size info
        size_info = cache_manager.get_cache_size("default")
        print(f"💾 Cache size: {size_info['size_mb']:.1f} MB ({size_info['file_count']} files)")
        
        print(f"\n🚀 Arena Bot will now start much faster!")
        print(f"💡 Typical startup time reduced from 30s to ~2s")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Make sure you're in the correct directory and dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Cache prebuilding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    setup_logging()
    
    print("🎯 Arena Bot Histogram Cache Prebuilder")
    print("This will prebuild the histogram cache for faster bot startup times.")
    print()
    
    # Check if running in automated environment
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("🤖 Running in automated mode...")
    else:
        # Confirm with user
        try:
            response = input("Continue with cache prebuilding? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Cache prebuilding cancelled")
                return
        except EOFError:
            print("🤖 No interactive input available, proceeding automatically...")
    
    print()
    success = prebuild_histogram_cache()
    
    if success:
        print(f"\n🎉 Success! Arena Bot cache is ready.")
        print(f"🚀 You can now test the bot with much faster startup times!")
        sys.exit(0)
    else:
        print(f"\n❌ Cache prebuilding failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()