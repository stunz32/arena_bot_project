#!/usr/bin/env python3
"""
Test the draft recommendation system.
"""

import logging
from arena_bot.ai.draft_advisor import get_draft_advisor, PickRecommendation

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    """Test draft advisor functionality."""
    print("=== Draft Advisor Test ===")
    
    # Initialize draft advisor
    advisor = get_draft_advisor()
    
    # Get statistics
    stats = advisor.get_draft_statistics()
    print(f"✓ Loaded draft advisor with {stats['total_cards']} cards")
    print(f"  Average tier score: {stats['average_tier_score']:.1f}")
    print(f"  Tier distribution: {stats['tier_distribution']}")
    
    # Test with our known cards
    test_cards = ['TOY_380', 'ULD_309', 'TTN_042']
    print(f"\nTesting draft choice: {test_cards}")
    
    # Analyze the draft choice
    choice = advisor.analyze_draft_choice(test_cards, 'warrior')
    
    print(f"\n=== Draft Recommendation ===")
    print(f"Recommended pick: Card {choice.recommended_pick + 1} ({choice.cards[choice.recommended_pick].card_code})")
    print(f"Recommendation level: {choice.recommendation_level.value.upper()}")
    print(f"Reasoning: {choice.reasoning}")
    
    print(f"\n=== Card Details ===")
    for i, card in enumerate(choice.cards):
        marker = "👑" if i == choice.recommended_pick else "  "
        print(f"{marker} Card {i+1}: {card.card_code}")
        print(f"     Tier: {card.tier_letter} (Score: {card.tier_score:.1f})")
        print(f"     Win Rate: {card.win_rate:.1%}")
        print(f"     Pick Rate: {card.pick_rate:.1%}")
        if card.notes:
            print(f"     Notes: {card.notes}")
    
    # Test different scenarios
    print(f"\n=== Testing Different Scenarios ===")
    
    # Test with unknown cards
    unknown_cards = ['UNKNOWN_1', 'UNKNOWN_2', 'UNKNOWN_3']
    unknown_choice = advisor.analyze_draft_choice(unknown_cards, 'mage')
    print(f"Unknown cards choice: Card {unknown_choice.recommended_pick + 1} ({unknown_choice.recommendation_level.value})")
    
    # Test mixed known/unknown
    mixed_cards = ['TOY_380', 'UNKNOWN_CARD', 'TTN_042']
    mixed_choice = advisor.analyze_draft_choice(mixed_cards, 'hunter')
    print(f"Mixed cards choice: Card {mixed_choice.recommended_pick + 1} ({mixed_choice.recommendation_level.value})")
    
    print(f"\n=== Draft Advisor Ready! ===")
    print("✓ Tier database loaded")
    print("✓ Recommendation engine working")
    print("✓ Ready for integration with card detection")

if __name__ == "__main__":
    main()