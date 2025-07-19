#!/usr/bin/env python3
"""
Debug the histogram database to see if cards are loaded
"""

import sys
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def debug_histogram_database():
    """Check the histogram matcher database."""
    try:
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        from arena_bot.utils.asset_loader import get_asset_loader
        
        print("🔍 Checking histogram matcher...")
        histogram_matcher = get_histogram_matcher()
        
        print(f"📊 Database size: {histogram_matcher.get_database_size()}")
        print(f"📊 Card histograms: {len(histogram_matcher.card_histograms)}")
        
        if len(histogram_matcher.card_histograms) == 0:
            print("❌ PROBLEM: Histogram database is empty!")
            print("   This explains the 'list index out of range' error")
        
        print("\n🔍 Checking asset loader...")
        asset_loader = get_asset_loader()
        available_cards = asset_loader.get_available_cards()
        print(f"📁 Available cards: {len(available_cards)}")
        print(f"📁 First 10 cards: {available_cards[:10]}")
        
        # Check if assets directory exists
        print(f"📁 Assets directory: {asset_loader.assets_dir}")
        print(f"📁 Assets dir exists: {asset_loader.assets_dir.exists()}")
        
        cards_dir = asset_loader.assets_dir / "cards"
        print(f"📁 Cards directory: {cards_dir}")
        print(f"📁 Cards dir exists: {cards_dir.exists()}")
        
        if cards_dir.exists():
            card_files = list(cards_dir.glob("*.png"))
            print(f"📁 Found {len(card_files)} card image files")
            if card_files:
                print(f"📁 First few files: {[f.name for f in card_files[:5]]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_histogram_database()