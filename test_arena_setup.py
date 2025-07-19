#!/usr/bin/env python3
"""
Arena Database Setup Script
Automatically initializes the arena card database by scraping HearthArena.com
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arena_bot.data.arena_card_database import get_arena_card_database
    
    print("🚀 Arena Database Setup")
    print("=" * 50)
    
    print("📊 Checking arena database status...")
    db = get_arena_card_database()
    info = db.get_database_info()
    print(f"Status: {info['status']}")
    
    if info['status'] == 'no_data':
        print("\n🌐 No arena data found - need to scrape HearthArena.com")
        print("⏳ This will take 1-2 minutes to download current arena cards...")
        print("🔧 Setting up Selenium WebDriver for HearthArena scraping...")
        
        # Force update from HearthArena
        success = db.update_from_heartharena(force=True)
        
        if success:
            print("\n✅ Arena database updated successfully!")
            new_info = db.get_database_info()
            print(f"📊 Total arena cards: {new_info['total_cards']}")
            print(f"📈 Mapping success rate: {new_info['mapping_stats']['success_rate']:.1f}%")
            print("\n🎯 Arena Priority detection is now available!")
            
            # Show card counts by class
            print("\n📋 Arena cards by class:")
            for class_name, count in new_info['card_counts'].items():
                print(f"   {class_name}: {count} cards")
                
        else:
            print("\n❌ Arena database update failed!")
            print("🔧 This might be due to:")
            print("   • Missing ChromeDriver for Selenium")
            print("   • Network connectivity issues")
            print("   • HearthArena.com changes")
            
    else:
        print(f"\n✅ Arena database already exists!")
        print(f"📊 Total arena cards: {info['total_cards']}")
        print(f"📅 Last updated: {info['last_updated']}")
        print(f"📈 Mapping success rate: {info['mapping_stats']['success_rate']:.1f}%")
        
        # Check if update needed
        needs_update, reason = db.check_for_updates()
        if needs_update:
            print(f"\n⚠️ Update recommended: {reason}")
            response = input("Update now? (y/N): ").lower()
            if response == 'y':
                print("\n🔄 Updating arena database...")
                success = db.update_from_heartharena(force=True)
                if success:
                    print("✅ Update completed!")
                else:
                    print("❌ Update failed!")
        else:
            print(f"✅ Database is up to date")
    
    print("\n🎯 Arena Priority detection ready!")
    print("You can now use the '🎯 Arena Priority' toggle in the bot GUI")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
except Exception as e:
    print(f"❌ Setup error: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to close...")