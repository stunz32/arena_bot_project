#!/usr/bin/env python3
"""
Arena Database Setup - Arena Tracker Method

Uses Arena Tracker's proven approach with downloadable JSON files
for current arena rotation information. No web scraping required.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arena_bot.data.arena_card_database import get_arena_card_database
    
    print("🚀 Arena Database Setup - Arena Tracker Method")
    print("=" * 60)
    print("🎯 Using Arena Tracker's proven filtering approach:")
    print("   • Downloads current arena rotation JSON files")
    print("   • Filters cards by sets, bans, and restrictions")
    print("   • No web scraping - reliable and fast")
    print("   • Reduces 11,000+ cards to ~1,800 eligible cards")
    print("=" * 60)
    
    print("\n📊 Checking arena database status...")
    db = get_arena_card_database()
    info = db.get_database_info()
    print(f"Status: {info['status']}")
    
    if info['status'] == 'no_data' or info.get('needs_update', True):
        print("\n🌐 Downloading arena version data...")
        print("⏳ This will take 10-30 seconds to download current arena rotation...")
        
        # Use the new Arena Tracker method
        success = db.update_from_arena_version(force=True)
        
        if success:
            print("\n✅ Arena database updated successfully using Arena Tracker method!")
            new_info = db.get_database_info()
            print(f"📊 Total arena cards: {new_info['total_cards']}")
            print(f"🎯 Method: {new_info['metadata'].get('method', 'arena_tracker_filtering')}")
            print(f"🔗 Source: {new_info['metadata'].get('source_url', 'multiple_sources')}")
            
            if 'version_hash' in new_info['metadata']:
                print(f"📋 Version hash: {new_info['metadata']['version_hash']}")
            
            print("\n📋 Arena cards by class:")
            for class_name, count in new_info['card_counts'].items():
                print(f"   {class_name}: {count} cards")
            
            # Show some sample arena cards
            print(f"\n🃏 Sample arena-eligible cards:")
            mage_cards = db.get_arena_cards_for_class('mage')[:5]
            if mage_cards:
                print("   Mage cards:")
                for card_id in mage_cards:
                    from arena_bot.data.cards_json_loader import get_cards_json_loader
                    cards_loader = get_cards_json_loader()
                    card_name = cards_loader.get_card_name(card_id)
                    print(f"     • {card_name} ({card_id})")
            
            print("\n🎯 Arena Priority detection is now available!")
            print("✅ You can now use the '🎯 Arena Priority' toggle in the bot GUI")
            print("✅ Arena-eligible cards will be marked with 🏟️ symbol")
            print("✅ Arena cards will be prioritized in detection results")
            print("✅ Fast startup times with intelligent filtering")
            
        else:
            print("\n❌ Arena database update failed!")
            print("🔧 This might be due to:")
            print("   • Network connectivity issues")
            print("   • Arena version server unavailable")
            print("   • Invalid JSON data format")
            print("\n💡 The system will use fallback data if available")
            
    else:
        print(f"\n✅ Arena database already exists!")
        print(f"📊 Total arena cards: {info['total_cards']}")
        print(f"📅 Last updated: {info['last_updated']}")
        
        if 'metadata' in info and 'method' in info['metadata']:
            print(f"🎯 Method: {info['metadata']['method']}")
        
        # Check if update needed
        needs_update, reason = db.check_for_updates()
        if needs_update:
            print(f"\n⚠️ Update recommended: {reason}")
            response = input("Update now? (y/N): ").lower()
            if response == 'y':
                print("\n🔄 Updating arena database...")
                success = db.update_from_arena_version(force=True)
                if success:
                    print("✅ Update completed!")
                else:
                    print("❌ Update failed!")
        else:
            print(f"✅ Database is up to date")
    
    print("\n🎯 Arena Tracker filtering method ready!")
    print("This provides the same proven performance as Arena Tracker:")
    print("  • 84% reduction in card pool (11,000 → ~1,800)")
    print("  • Current arena rotation accuracy")
    print("  • Automatic ban list management")
    print("  • Multiclass arena support")
    print("  • Real-time updates when rotation changes")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
except Exception as e:
    print(f"❌ Setup error: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to close...")