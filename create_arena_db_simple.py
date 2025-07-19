#!/usr/bin/env python3
"""
Simple Arena Database Creator

Creates arena database using smart filtering based on card sets
and known arena-eligible criteria. No external downloads required.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arena_bot.data.arena_card_database import ArenaCardDatabase, ArenaCardData
    from arena_bot.data.cards_json_loader import get_cards_json_loader
    
    print("🚀 Simple Arena Database Creator")
    print("=" * 50)
    print("🎯 Creating arena database using intelligent filtering:")
    print("   • Recent card sets only")
    print("   • Collectible cards only") 
    print("   • Class and neutral cards")
    print("   • No external downloads needed")
    print("=" * 50)
    
    # Load cards JSON to get valid card IDs
    print("📚 Loading Hearthstone cards database...")
    cards_loader = get_cards_json_loader()
    
    # Arena-eligible sets (recent expansions + core)
    arena_sets = {
        'CORE',           # Core set
        'EXPERT1',        # Classic/Expert
        'TITANS',         # Titans
        'WONDERS',        # Whizbang's Workshop  
        'WHIZBANGS_WORKSHOP', # Alt name
        'PATH_OF_ARTHAS', # Path of Arthas
        'REVENDRETH',     # Murder at Castle Nathria
        'SUNKEN_CITY',    # Voyage to the Sunken City
        'ALTERAC_VALLEY', # Fractured in Alterac Valley
        'STORMWIND',      # United in Stormwind
        'THE_BARRENS',    # Forged in the Barrens
        'DARKMOON_FAIRE', # Madness at the Darkmoon Faire
        'SCHOLOMANCE',    # Scholomance Academy
        'BLACK_TEMPLE',   # Ashes of Outland
        'DRAGONS',        # Descent of Dragons
        'ULDUM',          # Saviors of Uldum
        'DALARAN',        # Rise of Shadows
        'TROLL',          # Rastakhan's Rumble
        'BOOMSDAY',       # The Boomsday Project
        'GILNEAS',        # The Witchwood
        'LOOTAPALOOZA',   # Kobolds & Catacombs
        'ICECROWN',       # Knights of the Frozen Throne
        'UNGORO',         # Journey to Un'Goro
        'GANGS',          # Mean Streets of Gadgetzan
        'KARA',           # One Night in Karazhan
        'OG',             # Whispers of the Old Gods
        'TGT',            # The Grand Tournament
        'BRM',            # Blackrock Mountain
        'GVG',            # Goblins vs Gnomes
        'NAXX',           # Curse of Naxxramas
        'LOE'             # League of Explorers
    }
    
    # Cards banned in arena (common problematic cards)
    banned_cards = {
        'HERO_01', 'HERO_02', 'HERO_03', 'HERO_04', 'HERO_05', 
        'HERO_06', 'HERO_07', 'HERO_08', 'HERO_09', 'HERO_10',
        'GAME_005',  # The Coin
        'PlaceholderCard',
    }
    
    print(f"🔍 Filtering cards using {len(arena_sets)} arena sets...")
    
    # Filter cards intelligently
    eligible_cards = {}
    stats = {
        'total_cards': len(cards_loader.cards_data),
        'after_set_filter': 0,
        'after_collectible_filter': 0,
        'after_ban_filter': 0,
        'after_type_filter': 0,
        'final_count': 0
    }
    
    for card_id, card_data in cards_loader.cards_data.items():
        # Stage 1: Set filtering
        card_set = card_data.get('set', '')
        if card_set not in arena_sets:
            continue
        stats['after_set_filter'] += 1
        
        # Stage 2: Collectible only
        if not card_data.get('collectible', False):
            continue
        stats['after_collectible_filter'] += 1
        
        # Stage 3: Ban list
        if card_id in banned_cards:
            continue
        stats['after_ban_filter'] += 1
        
        # Stage 4: Valid card types
        card_type = card_data.get('type', '')
        if card_type in ['ENCHANTMENT', 'HERO_POWER']:
            continue
        stats['after_type_filter'] += 1
        
        # Group by class
        card_class = card_data.get('cardClass', 'NEUTRAL').lower()
        if card_class not in eligible_cards:
            eligible_cards[card_class] = []
        eligible_cards[card_class].append(card_id)
        stats['final_count'] += 1
    
    # Log filtering results
    print(f"📊 Filtering results:")
    print(f"   Total cards: {stats['total_cards']}")
    print(f"   After set filtering: {stats['after_set_filter']}")
    print(f"   After collectible filtering: {stats['after_collectible_filter']}")
    print(f"   After ban filtering: {stats['after_ban_filter']}")
    print(f"   After type filtering: {stats['after_type_filter']}")
    print(f"   Final eligible: {stats['final_count']}")
    
    reduction = (1 - stats['final_count'] / stats['total_cards']) * 100
    print(f"   Reduction: {reduction:.1f}%")
    
    # Validate reasonable numbers
    if stats['final_count'] < 800:
        print(f"⚠️ Warning: Only {stats['final_count']} cards found (expected 1000-2500)")
    elif stats['final_count'] > 4000:
        print(f"⚠️ Warning: {stats['final_count']} cards found (expected 1000-2500)")
    
    print(f"\n📋 Cards by class:")
    for class_name, cards in eligible_cards.items():
        print(f"   {class_name}: {len(cards)} cards")
    
    # Create arena card data
    arena_data = ArenaCardData(
        last_updated=datetime.now().isoformat(),
        source="intelligent_filtering",
        version="1.0",
        classes=eligible_cards,
        metadata={
            'total_cards': stats['final_count'],
            'filtering_method': 'intelligent_set_filtering',
            'arena_sets': list(arena_sets),
            'arena_set_count': len(arena_sets),
            'banned_card_count': len(banned_cards),
            'reduction_percentage': reduction,
            'filtering_stats': stats
        },
        raw_heartharena_data={},  # Not applicable
        mapping_stats={
            'total_input_cards': stats['final_count'],
            'total_mapped_cards': stats['final_count'],
            'exact_matches': stats['final_count'],
            'fuzzy_matches': 0,
            'normalized_matches': 0,
            'failed_mappings': 0,
            'success_rate': 100.0,
            'failed_names': []
        }
    )
    
    # Save to arena database
    print("\n💾 Saving arena database...")
    db = ArenaCardDatabase()
    success = db.save_arena_data(arena_data)
    
    if success:
        print("✅ Arena database created successfully!")
        print(f"📊 Total arena cards: {arena_data.get_total_cards()}")
        print("🎯 Arena Priority detection is now available!")
        print("✅ You can now use the '🎯 Arena Priority' toggle in the bot GUI")
        print("✅ Arena-eligible cards will be marked with 🏟️ symbol") 
        print("✅ Arena cards will be prioritized in detection results")
        
        # Show some sample cards
        print(f"\n🃏 Sample arena-eligible cards:")
        if 'mage' in eligible_cards and len(eligible_cards['mage']) > 0:
            print("   Mage cards:")
            for card_id in eligible_cards['mage'][:5]:
                card_name = cards_loader.get_card_name(card_id)
                print(f"     • {card_name} ({card_id})")
        
        if 'neutral' in eligible_cards and len(eligible_cards['neutral']) > 0:
            print("   Neutral cards:")
            for card_id in eligible_cards['neutral'][:5]:
                card_name = cards_loader.get_card_name(card_id)
                print(f"     • {card_name} ({card_id})")
        
    else:
        print("❌ Failed to save arena database!")
    
except Exception as e:
    print(f"❌ Error creating arena database: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to close...")