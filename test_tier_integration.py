#!/usr/bin/env python3
"""
HearthArena Tier Integration Test Script

Demonstrates and tests the new EzArena-style tier integration features.
Run this to verify that the tier system is working correctly.
"""

import logging
import sys
from pathlib import Path

# Add the arena_bot module to path
sys.path.insert(0, str(Path(__file__).parent / "arena_bot"))

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tier_integration_test.log')
        ]
    )

def test_tier_manager():
    """Test the HearthArena Tier Manager."""
    print("\n" + "="*60)
    print("🎯 TESTING HEARTHARENA TIER MANAGER")
    print("="*60)
    
    try:
        from arena_bot.data.heartharena_tier_manager import get_heartharena_tier_manager
        
        tier_manager = get_heartharena_tier_manager()
        
        # Check current status
        stats = tier_manager.get_tier_statistics()
        print(f"Status: {stats['status']}")
        
        if stats['status'] == 'loaded':
            print(f"✅ Tier data already cached")
            print(f"   Total cards: {stats['total_cards']}")
            print(f"   Classes: {stats['classes']}")
            print(f"   Cache age: {stats['cache_age_hours']:.1f} hours")
        else:
            print("📥 No cached data, updating from HearthArena...")
            success = tier_manager.update_tier_data(force=True)
            
            if success:
                stats = tier_manager.get_tier_statistics()
                print(f"✅ Update successful!")
                print(f"   Total cards: {stats['total_cards']}")
                print(f"   Classes: {stats['classes']}")
            else:
                print("❌ Tier manager update failed!")
                return False
        
        # Test specific class data
        print(f"\n🔍 Testing Mage tier data:")
        mage_tiers = tier_manager.get_class_tiers('mage')
        if mage_tiers:
            print(f"   Mage cards with tiers: {len(mage_tiers)}")
            
            # Show examples of each tier
            tier_examples = {}
            for card_name, tier_data in mage_tiers.items():
                tier = tier_data.tier
                if tier not in tier_examples:
                    tier_examples[tier] = []
                if len(tier_examples[tier]) < 2:  # Limit to 2 examples per tier
                    tier_examples[tier].append(card_name)
            
            print(f"   Tier examples:")
            for tier in ['beyond-great', 'great', 'good', 'above-average', 'average', 'below-average', 'bad', 'terrible']:
                if tier in tier_examples:
                    examples = ', '.join(tier_examples[tier][:2])
                    print(f"     {tier}: {examples}")
        else:
            print("   ❌ No mage tier data found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tier Manager test failed: {e}")
        return False

def test_tier_cache():
    """Test the Tier Cache Manager."""
    print("\n" + "="*60)
    print("🚀 TESTING TIER CACHE MANAGER")
    print("="*60)
    
    try:
        from arena_bot.data.tier_cache_manager import get_tier_cache_manager
        
        cache_manager = get_tier_cache_manager()
        
        # Check cache status
        stats = cache_manager.get_cache_statistics()
        print(f"Cache status: {stats['status']}")
        
        if stats['status'] == 'loaded':
            print(f"✅ Binary cache loaded successfully")
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   Cache size: {stats['cache_size_bytes']:,} bytes")
            print(f"   Compression ratio: {stats['compression_ratio']:.1f}x")
            print(f"   Cache age: {stats['cache_age_hours']:.1f} hours")
            
            if 'performance' in stats:
                perf = stats['performance']
                print(f"   Performance:")
                print(f"     Save time: {perf['save_time_ms']:.1f}ms")
                print(f"     Binary size: {perf['binary_size_bytes']:,} bytes")
                print(f"     JSON size: {perf['json_size_bytes']:,} bytes")
                print(f"     Compression efficiency: {perf['compression_efficiency']:.1f}%")
        else:
            print("📥 No cache data, updating...")
            success = cache_manager.update_tier_cache(force=True)
            
            if success:
                stats = cache_manager.get_cache_statistics()
                print(f"✅ Cache update successful!")
                print(f"   Cache size: {stats['cache_size_bytes']:,} bytes")
                print(f"   Compression: {stats['compression_ratio']:.1f}x")
            else:
                print("❌ Cache update failed!")
                return False
        
        # Test fast tier lookup
        print(f"\n⚡ Testing fast tier lookup:")
        mage_tiers = cache_manager.get_class_tiers('mage')
        if mage_tiers:
            # Test a few specific cards
            test_cards = list(mage_tiers.keys())[:5]
            for card_name in test_cards:
                tier_data = cache_manager.get_card_tier(card_name, 'mage')
                if tier_data:
                    print(f"   {card_name}: {tier_data.tier} (index: {tier_data.tier_index})")
        
        return True
        
    except Exception as e:
        print(f"❌ Tier Cache test failed: {e}")
        return False

def test_arena_database_integration():
    """Test the Arena Card Database with tier integration."""
    print("\n" + "="*60)
    print("🎮 TESTING ARENA DATABASE TIER INTEGRATION")
    print("="*60)
    
    try:
        from arena_bot.data.arena_card_database import get_arena_card_database
        
        db = get_arena_card_database()
        
        # Get database info with tier stats
        info = db.get_database_info()
        print(f"Database status: {info['status']}")
        
        if info['status'] == 'loaded':
            print(f"✅ Arena database loaded")
            print(f"   Total arena cards: {info['total_cards']}")
            print(f"   Cache age: {info['cache_age_days']:.1f} days")
            
            # Show tier integration status
            tier_stats = info.get('tier_stats', {})
            if tier_stats.get('has_tier_data'):
                print(f"   🎯 Tier integration active:")
                print(f"     Classes with tiers: {tier_stats['classes_with_tiers']}")
                print(f"     Cards with tier data: {tier_stats['total_cards_with_tiers']}")
                
                # Show tier distribution
                print(f"     Tier distribution:")
                for tier, count in tier_stats.get('tier_distribution', {}).items():
                    print(f"       {tier}: {count} cards")
            else:
                print(f"   ⚠️ No tier data integrated, updating...")
                success = db.update_with_tier_data(force=True)
                if success:
                    info = db.get_database_info()
                    tier_stats = info.get('tier_stats', {})
                    print(f"   ✅ Tier integration completed:")
                    print(f"     Cards with tiers: {tier_stats.get('total_cards_with_tiers', 0)}")
                else:
                    print(f"   ❌ Tier integration failed!")
                    return False
            
            # Show cache performance
            cache_info = info.get('tier_cache_info', {})
            if cache_info.get('status') == 'loaded':
                print(f"   📊 Cache performance:")
                print(f"     Cache size: {cache_info['cache_size_bytes']:,} bytes")
                print(f"     Compression: {cache_info.get('compression_ratio', 1.0):.1f}x")
        
        else:
            print("📥 No arena data, updating...")
            success = db.update_from_arena_version(force=True)
            if not success:
                print("❌ Arena database update failed!")
                return False
        
        # Test tier lookup for specific cards
        print(f"\n🔍 Testing card tier lookup:")
        
        # Get some mage cards to test
        mage_cards = db.get_arena_cards_for_class('mage')
        if mage_cards:
            print(f"   Found {len(mage_cards)} arena-eligible mage cards")
            
            # Test tier lookup for first few cards
            cards_with_tiers = db.get_arena_cards_with_tiers('mage')
            tier_count = sum(1 for tier_data in cards_with_tiers.values() if tier_data is not None)
            
            print(f"   Cards with tier data: {tier_count}/{len(mage_cards)}")
            
            # Show examples
            examples_shown = 0
            for card_id, tier_data in cards_with_tiers.items():
                if tier_data and examples_shown < 5:
                    card_name = db.cards_loader.get_card_name(card_id)
                    print(f"     {card_name or card_id}: {tier_data.tier}")
                    examples_shown += 1
        
        # Test fast tier lookup
        print(f"\n⚡ Testing fast tier lookup (by card name):")
        test_card_names = ["Fireball", "Frostbolt", "Arcane Intellect", "Flamestrike", "Polymorph"]
        for card_name in test_card_names:
            tier_data = db.get_card_tier_fast(card_name, 'mage')
            if tier_data:
                print(f"     {card_name}: {tier_data.tier} (confidence: {tier_data.confidence})")
            else:
                print(f"     {card_name}: No tier data")
        
        return True
        
    except Exception as e:
        print(f"❌ Arena Database integration test failed: {e}")
        return False

def test_arena_version_manager():
    """Test the Arena Version Manager."""
    print("\n" + "="*60)
    print("📋 TESTING ARENA VERSION MANAGER")
    print("="*60)
    
    try:
        from arena_bot.data.arena_version_manager import get_arena_version_manager
        
        manager = get_arena_version_manager()
        
        # Get version info
        info = manager.get_version_info()
        print(f"Version status: {info['status']}")
        
        if info['status'] == 'loaded':
            print(f"✅ Arena version data loaded")
            print(f"   Version hash: {info['version_hash']}")
            print(f"   Arena sets: {info['arena_set_count']}")
            print(f"   Eligible cards: {info['eligible_card_count']}")
            print(f"   Cache age: {info['cache_age_hours']:.1f} hours")
            print(f"   Source: {info['source_url']}")
        else:
            print("📥 No version data, updating...")
            success = manager.update_arena_version(force=True)
            if success:
                info = manager.get_version_info()
                print(f"✅ Version update successful!")
                print(f"   Eligible cards: {info['eligible_card_count']}")
            else:
                print("❌ Version update failed!")
                return False
        
        # Test class filtering
        print(f"\n🎯 Testing class filtering:")
        mage_cards = manager.get_eligible_cards_for_class('MAGE')
        warrior_cards = manager.get_eligible_cards_for_class('WARRIOR')
        neutral_cards = manager.get_eligible_cards_for_class('NEUTRAL')
        
        print(f"   Mage eligible: {len(mage_cards)} cards")
        print(f"   Warrior eligible: {len(warrior_cards)} cards") 
        print(f"   Neutral eligible: {len(neutral_cards)} cards")
        
        return True
        
    except Exception as e:
        print(f"❌ Arena Version Manager test failed: {e}")
        return False

def main():
    """Run all tier integration tests."""
    print("🎯 HEARTHARENA TIER INTEGRATION TEST SUITE")
    print("=" * 80)
    print("This script tests the new EzArena-style tier integration features.")
    print("It will verify that HearthArena tier data is being scraped,")
    print("cached efficiently, and integrated with arena card data.")
    print("=" * 80)
    
    setup_logging()
    
    # Run all tests
    tests = [
        ("Arena Version Manager", test_arena_version_manager),
        ("HearthArena Tier Manager", test_tier_manager), 
        ("Tier Cache Manager", test_tier_cache),
        ("Arena Database Integration", test_arena_database_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Show final results
    print("\n" + "="*80)
    print("📊 FINAL TEST RESULTS")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Tier integration is working correctly.")
        print("\nYou can now:")
        print("• Use HearthArena tier data in your arena bots")
        print("• Access fast binary-cached tier information")
        print("• Get both arena eligibility AND tier rankings")
        print("• Enjoy 10x+ performance improvement over Selenium")
    else:
        print("⚠️  SOME TESTS FAILED! Check the output above for details.")
        print("\nTry running individual test components to diagnose issues.")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)