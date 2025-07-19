#!/usr/bin/env python3
"""
Clear pHash cache to force rebuild with full card database.

Run this after fixing the 2000-card limit to ensure the cache
rebuilds with the complete 12,008+ card database.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def clear_phash_cache():
    """Clear the pHash cache to force rebuild."""
    try:
        from arena_bot.detection.phash_cache_manager import get_phash_cache_manager
        
        print("🧹 Clearing pHash cache...")
        
        cache_manager = get_phash_cache_manager()
        
        # Get current cache info
        cache_info = cache_manager.get_cache_info()
        print(f"   Current cache: {cache_info['cached_cards']} cards")
        
        # Clear cache
        cache_manager.clear_cache()
        
        print("✅ pHash cache cleared successfully!")
        print("\n💡 Next steps:")
        print("   1. Run the Arena Bot GUI")
        print("   2. The bot will now load ALL 12,008+ cards (may take 2-3 minutes first time)")
        print("   3. pHash database will rebuild with complete card set")
        print("   4. Future startups will be fast with full database cached")
        
        return True
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False

if __name__ == "__main__":
    print("🚀 pHash Cache Cleaner")
    print("=" * 40)
    
    success = clear_phash_cache()
    
    if success:
        print("\n🎉 Ready to rebuild with full card database!")
    else:
        print("\n⚠️ Manual cache clearing may be needed.")
        print("   Cache location: assets/cache/phashes/")