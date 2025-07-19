#!/usr/bin/env python3
"""Test the new cards.json system"""

import sys
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_cards_json():
    """Test card name resolution from JSON"""
    
    from arena_bot.data.cards_json_loader import get_cards_json_loader
    
    loader = get_cards_json_loader()
    
    # Test some card IDs we've been seeing
    test_cards = [
        "EX1_339",  # One we saw in detection
        "LOE_115",  # Another we saw
        "HERO_05",  # Hero card we saw
        "CFM_606",  # Another candidate
    ]
    
    print("Testing card name resolution:")
    for card_id in test_cards:
        name = loader.get_card_name(card_id)
        collectible = loader.is_collectible(card_id)
        card_set = loader.get_card_set(card_id)
        cost = loader.get_card_cost(card_id)
        
        print(f"  {card_id}: '{name}' | Collectible: {collectible} | Set: {card_set} | Cost: {cost}")
    
    # Try to find cards with "Clay", "Dwarven", "Cyclopean" in their names
    print("\nSearching for your cards in the database...")
    search_terms = ["clay", "dwarven", "cyclopean", "matriarch", "archaeologist", "crusher"]
    
    found_cards = []
    for card_id, card_data in loader.cards_data.items():
        name = card_data.get('name', '').lower()
        for term in search_terms:
            if term in name:
                found_cards.append((card_id, card_data.get('name', ''), card_data.get('collectible', False)))
    
    print(f"Found {len(found_cards)} matching cards:")
    for card_id, name, collectible in found_cards[:10]:  # Show first 10
        print(f"  {card_id}: '{name}' | Collectible: {collectible}")

if __name__ == "__main__":
    test_cards_json()