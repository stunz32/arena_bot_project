#!/usr/bin/env python3
"""
Test the Card Eligibility Filter
Verify it reduces database size like Arena Tracker (80-85% reduction).
"""

import sys
import logging
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Test the card eligibility filter."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 CARD ELIGIBILITY FILTER TEST")
    print("=" * 80)
    print("🎯 Testing Arena Tracker's getEligibleCards() implementation")
    print("🎯 Target: Reduce ~11K cards to ~1.8K cards (80-85% reduction)")
    print("=" * 80)
    
    try:
        from arena_bot.data.card_eligibility_filter import get_card_eligibility_filter
        from arena_bot.utils.asset_loader import get_asset_loader
        
        # Initialize components
        filter_system = get_card_eligibility_filter()
        asset_loader = get_asset_loader()
        
        # Get all available cards
        available_cards = asset_loader.get_available_cards()
        
        print(f"\n📊 Starting database size: {len(available_cards)} cards")
        
        # Test with different hero classes
        test_classes = ["MAGE", "WARRIOR", "HUNTER", None]
        
        for hero_class in test_classes:
            print(f"\n🧙 Testing with hero class: {hero_class or 'None (no filter)'}")
            print("-" * 60)
            
            # Get eligible cards
            eligible_cards = filter_system.get_eligible_cards(
                hero_class=hero_class,
                available_cards=available_cards
            )
            
            reduction_pct = (len(available_cards) - len(eligible_cards)) / len(available_cards) * 100
            
            print(f"✅ Eligible cards: {len(eligible_cards)}")
            print(f"📉 Reduction: {reduction_pct:.1f}%")
            
            # Check if we hit Arena Tracker's target
            if reduction_pct >= 80:
                print("🎯 SUCCESS: Achieved Arena Tracker's 80%+ reduction target!")
            elif reduction_pct >= 70:
                print("✅ GOOD: Achieved 70%+ reduction")
            else:
                print("⚠️ WARNING: Reduction below 70%, may need adjustment")
        
        # Test with our known target cards
        print(f"\n🎯 TESTING WITH TARGET CARDS")
        print("-" * 60)
        
        target_cards = ["TOY_380", "ULD_309", "TTN_042"]
        
        # Test with neutral hero (should include all classes)
        eligible_neutral = filter_system.get_eligible_cards(hero_class=None)
        
        target_results = {}
        for card_id in target_cards:
            is_eligible = card_id in eligible_neutral
            target_results[card_id] = is_eligible
            
            card_name = filter_system.cards_loader.get_card_name(card_id)
            card_class = filter_system.cards_loader.get_card_class(card_id)
            card_set = filter_system.cards_loader.get_card_set(card_id)
            collectible = filter_system.cards_loader.is_collectible(card_id)
            
            status = "✅" if is_eligible else "❌"
            print(f"{status} {card_id} ({card_name})")
            print(f"    Class: {card_class}, Set: {card_set}, Collectible: {collectible}")
        
        # Summary
        eligible_targets = sum(target_results.values())
        print(f"\n🏆 TARGET CARD RESULTS: {eligible_targets}/3 target cards eligible")
        
        if eligible_targets == 3:
            print("🎉 SUCCESS: All target cards pass eligibility filter!")
        elif eligible_targets >= 2:
            print("✅ GOOD: Most target cards eligible")
        else:
            print("❌ PROBLEM: Target cards not passing filter - needs adjustment")
        
        # Show filter stats
        print(f"\n📊 FILTER CONFIGURATION:")
        stats = filter_system.get_filter_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return eligible_targets == 3
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)