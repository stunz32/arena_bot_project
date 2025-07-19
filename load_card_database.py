#!/usr/bin/env python3
"""
Load card database for testing.
Populates the histogram matcher with card images for recognition.
"""

import sys
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def load_card_database():
    """Load all card images into the histogram matcher."""
    print("📚 Loading card database...")
    
    try:
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        
        # Get instances
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        
        # Get available cards
        available_cards = asset_loader.get_available_cards()
        print(f"   Found {len(available_cards)} card codes")
        
        # Load a subset for testing (loading all 4000+ would take a while)
        test_cards = available_cards[:500]  # Load first 500 for testing
        print(f"   Loading first {len(test_cards)} cards for testing...")
        
        card_images = {}
        loaded_count = 0
        
        for card_code in test_cards:
            # Load normal version
            normal_image = asset_loader.load_card_image(card_code, premium=False)
            if normal_image is not None:
                card_images[card_code] = normal_image
                loaded_count += 1
            
            # Load premium version if available
            premium_image = asset_loader.load_card_image(card_code, premium=True) 
            if premium_image is not None:
                card_images[f"{card_code}_premium"] = premium_image
                loaded_count += 1
            
            # Progress indicator
            if loaded_count % 50 == 0:
                print(f"   Loaded {loaded_count} images...")
        
        # Load into histogram matcher
        histogram_matcher.load_card_database(card_images)
        
        final_db_size = histogram_matcher.get_database_size()
        print(f"✅ Card database loaded: {final_db_size} histograms")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load card database: {e}")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    success = load_card_database()
    sys.exit(0 if success else 1)