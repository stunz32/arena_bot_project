#!/usr/bin/env python3
"""
Manual Arena Database Creator
Creates a working arena database without web scraping
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
    
    print("🚀 Manual Arena Database Creator")
    print("=" * 50)
    
    # Load cards JSON to get valid card IDs
    print("📚 Loading Hearthstone cards database...")
    cards_loader = get_cards_json_loader()
    
    # Sample arena cards based on typical arena sets
    # This is a representative set of common arena cards
    sample_arena_cards = {
        'mage': [
            'CS2_029',  # Fireball
            'CS2_023',  # Arcane Intellect
            'CS2_027',  # Mirror Image
            'EX1_277',  # Arcane Missiles
            'CS2_024',  # Frostbolt
            'CS2_033',  # Water Elemental
            'CS2_022',  # Polymorph
            'NEW1_012', # Mana Wyrm
            'EX1_559',  # Archmage Antonidas
            'GVG_002',  # Flamecannon
        ],
        'warrior': [
            'CS2_106',  # Fiery War Axe
            'CS2_108',  # Execute
            'EX1_606',  # Shield Slam
            'CS2_105',  # Heroic Strike
            'EX1_607',  # Inner Rage
            'CS2_112',  # Armorsmith
            'EX1_398',  # Arcanite Reaper
            'CS2_114',  # Cleave
            'EX1_414',  # Grommash Hellscream
            'GVG_006',  # Warbot
        ],
        'hunter': [
            'DS1_184',  # Tracking
            'CS2_084',  # Hunter's Mark
            'DS1_183',  # Multi-Shot
            'CS2_084',  # Hunter's Mark
            'DS1_178',  # Tundra Rhino
            'CS2_237',  # Starving Buzzard
            'DS1_175',  # Timber Wolf
            'EX1_539',  # Kill Command
            'EX1_534',  # Savannah Highmane
            'GVG_017',  # Call Pet
        ],
        'priest': [
            'CS2_235',  # Northshire Cleric
            'CS1_130',  # Holy Smite
            'CS2_234',  # Shadow Word: Pain
            'CS2_236',  # Divine Spirit
            'CS1_112',  # Holy Nova
            'EX1_622',  # Shadow Word: Death
            'CS2_003',  # Mind Control
            'EX1_621',  # Circle of Healing
            'EX1_623',  # Temple Enforcer
            'GVG_010',  # Velens Chosen
        ],
        'warlock': [
            'CS2_057',  # Shadow Bolt
            'EX1_302',  # Mortal Coil
            'CS2_059',  # Blood Imp
            'EX1_306',  # Succubus
            'CS2_061',  # Drain Life
            'EX1_323',  # Lord Jaraxxus
            'CS2_062',  # Hellfire
            'EX1_319',  # Flame Imp
            'EX1_301',  # Felguard
            'GVG_015',  # Darkbomb
        ],
        'rogue': [
            'CS2_080',  # Assassinate
            'CS2_076',  # Assassin\'s Blade
            'CS2_074',  # Deadly Poison
            'CS2_077',  # Sprint
            'EX1_129',  # Fan of Knives
            'CS2_233',  # Blade Flurry
            'EX1_613',  # Edwin VanCleef
            'CS2_072',  # Backstab
            'EX1_133',  # Defias Ringleader
            'GVG_025',  # One-eyed Cheat
        ],
        'shaman': [
            'CS2_045',  # Rockbiter Weapon
            'CS2_037',  # Frost Shock
            'CS2_046',  # Bloodlust
            'EX1_241',  # Lava Burst
            'CS2_039',  # Windfury
            'EX1_245',  # Earth Shock
            'CS2_042',  # Fire Elemental
            'EX1_565',  # Flametongue Totem
            'EX1_250',  # Earth Elemental
            'GVG_037',  # Whirling Zap-o-matic
        ],
        'paladin': [
            'CS2_087',  # Blessing of Might
            'CS2_092',  # Blessing of Kings
            'CS2_088',  # Guardian of Kings
            'CS2_089',  # Holy Light
            'CS2_091',  # Light\'s Justice
            'EX1_360',  # Humility
            'EX1_354',  # Lay on Hands
            'CS2_093',  # Consecration
            'EX1_362',  # Argent Protector
            'GVG_061',  # Muster for Battle
        ],
        'druid': [
            'CS2_005',  # Claw
            'CS2_007',  # Healing Touch
            'CS2_008',  # Moonfire
            'CS2_011',  # Savage Roar
            'EX1_571',  # Force of Nature
            'EX1_578',  # Naturalize
            'CS2_012',  # Swipe
            'EX1_573',  # Cenarius
            'CS2_013',  # Wild Growth
            'GVG_080',  # Recycle
        ],
        'demon-hunter': [
            'BT_429',   # Metamorphosis
            'BT_323',   # Consume Magic
            'BT_430',   # Immolation Aura
            'BT_753',   # Spectral Sight
            'BT_486',   # Coordinated Strike
            'BT_351',   # Skull of Gul\'dan
            'BT_355',   # Priestess of Fury
            'BT_480',   # Fel Summoner
            'BT_812',   # Chaos Strike
            'BT_814',   # Twin Slice
        ],
        'neutral': [
            'CS2_189',  # Elven Archer
            'CS1_042',  # Goldshire Footman
            'CS2_162',  # Lord of the Arena
            'CS2_200',  # Boulderfist Ogre
            'CS1_069',  # Fen Creeper
            'CS2_119',  # Oasis Snapjaw
            'CS2_201',  # Core Hound
            'CS2_186',  # War Golem
            'EX1_066',  # Acidic Swamp Ooze
            'CS2_188',  # Abusive Sergeant
            'EX1_015',  # Novice Engineer
            'CS2_147',  # Gnomish Inventor
            'CS2_221',  # Spiteful Smith
            'EX1_593',  # Nightblade
            'CS2_155',  # Archmage
            'EX1_399',  # Gurubashi Berserker
            'CS2_150',  # Stormpike Commando
            'CS2_213',  # Reckless Rocketeer
            'EX1_508',  # Grimscale Oracle
            'CS2_179',  # Sen\'jin Shieldmasta
        ]
    }
    
    print(f"📊 Creating arena database with {sum(len(cards) for cards in sample_arena_cards.values())} cards...")
    
    # Validate card IDs exist in cards.json
    validated_cards = {}
    total_valid = 0
    
    for class_name, card_ids in sample_arena_cards.items():
        valid_cards = []
        for card_id in card_ids:
            if cards_loader.get_card_name(card_id) != f"Unknown ({card_id})":
                valid_cards.append(card_id)
                total_valid += 1
            else:
                print(f"⚠️ Invalid card ID: {card_id}")
        
        validated_cards[class_name] = valid_cards
        print(f"   {class_name}: {len(valid_cards)} valid cards")
    
    print(f"✅ Validated {total_valid} arena cards")
    
    # Create arena card data
    arena_data = ArenaCardData(
        last_updated=datetime.now().isoformat(),
        source="manual_creation",
        version="1.0",
        classes=validated_cards,
        metadata={
            'total_cards': total_valid,
            'creation_method': 'manual',
            'source_url': 'manual_representative_set',
            'note': 'Representative arena card set for testing arena priority detection'
        },
        raw_heartharena_data={},  # Empty since this is manual
        mapping_stats={
            'total_input_cards': total_valid,
            'total_mapped_cards': total_valid,
            'exact_matches': total_valid,
            'fuzzy_matches': 0,
            'normalized_matches': 0,
            'failed_mappings': 0,
            'success_rate': 100.0,
            'failed_names': []
        }
    )
    
    # Save to arena database
    print("💾 Saving arena database...")
    db = ArenaCardDatabase()
    success = db.save_arena_data(arena_data)
    
    if success:
        print("✅ Arena database created successfully!")
        print(f"📊 Total arena cards: {arena_data.get_total_cards()}")
        print(f"📈 Mapping success rate: 100.0%")
        print("\n📋 Arena cards by class:")
        for class_name, count in db.get_arena_card_counts().items():
            print(f"   {class_name}: {count} cards")
        
        print("\n🎯 Arena Priority detection is now available!")
        print("✅ You can now use the '🎯 Arena Priority' toggle in the bot GUI")
        print("✅ Arena-eligible cards will be marked with 🏟️ symbol")
        print("✅ Arena cards will be prioritized in detection results")
        
    else:
        print("❌ Failed to save arena database!")
    
except Exception as e:
    print(f"❌ Error creating manual arena database: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to close...")