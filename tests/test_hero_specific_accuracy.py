"""
Hero-Specific Recommendation Accuracy Testing Against Known Meta Performance

Comprehensive tests validating hero-specific recommendation accuracy against
known meta performance data and established Hearthstone Arena strategies.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import statistics
import math
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import DeckState, DimensionalScores
from arena_bot.data.cards_json_loader import CardsJsonLoader


class TestHeroSpecificAccuracy(unittest.TestCase):
    """Hero-specific recommendation accuracy tests against known meta performance."""
    
    def setUp(self):
        """Set up hero-specific accuracy test fixtures."""
        # Meta performance thresholds for validation
        self.META_ACCURACY_THRESHOLD = 0.75  # 75% accuracy against known meta
        self.HERO_SYNERGY_ACCURACY = 0.80     # 80% accuracy for hero synergy detection
        self.ARCHETYPE_ACCURACY = 0.70        # 70% accuracy for archetype recommendations
        
        # Create comprehensive meta-aware test data
        self.meta_cards_data = self._create_meta_cards_data()
        self.meta_hero_winrates = self._create_meta_hero_winrates()
        self.meta_hsreplay_data = self._create_meta_hsreplay_data()
        
        # Known meta scenarios with expected outcomes
        self.meta_scenarios = self._create_meta_scenarios()
        
        # Hero-specific archetypes and preferences
        self.hero_archetypes = self._create_hero_archetypes()
    
    def _create_meta_cards_data(self):
        """Create cards data based on known meta performance."""
        cards_data = {}
        
        # Hero cards
        hero_data = [
            ("HERO_01", "Garrosh Hellscream", "WARRIOR", 813),
            ("HERO_02", "Jaina Proudmoore", "MAGE", 637),
            ("HERO_03", "Rexxar", "HUNTER", 31),
            ("HERO_04", "Uther Lightbringer", "PALADIN", 671),
            ("HERO_05", "Anduin Wrynn", "PRIEST", 822),
            ("HERO_06", "Valeera Sanguinar", "ROGUE", 930),
            ("HERO_07", "Thrall", "SHAMAN", 1066),
            ("HERO_08", "Gul'dan", "WARLOCK", 893),
            ("HERO_09", "Malfurion Stormrage", "DRUID", 274),
            ("HERO_10", "Illidan Stormrage", "DEMONHUNTER", 56550)
        ]
        
        for card_id, name, player_class, dbf_id in hero_data:
            cards_data[card_id] = {
                "id": card_id,
                "name": name,
                "playerClass": player_class,
                "type": "HERO",
                "dbfId": dbf_id
            }
        
        # Meta cards with known performance characteristics
        meta_cards = [
            # Top-tier universally strong cards
            ("FLAMESTRIKE", "Flamestrike", "MAGE", "SPELL", 7, None, None, "top_tier", ["control"]),
            ("FIREBALL", "Fireball", "MAGE", "SPELL", 4, None, None, "top_tier", ["aggro", "midrange"]),
            ("FROSTBOLT", "Frostbolt", "MAGE", "SPELL", 2, None, None, "strong", ["aggro", "control"]),
            
            # Warrior cards with specific synergies
            ("FIERY_WAR_AXE", "Fiery War Axe", "WARRIOR", "WEAPON", 2, 3, 2, "top_tier", ["aggro", "midrange"]),
            ("EXECUTE", "Execute", "WARRIOR", "SPELL", 1, None, None, "strong", ["control"]),
            ("SHIELD_SLAM", "Shield Slam", "WARRIOR", "SPELL", 1, None, None, "strong", ["control"]),
            
            # Hunter cards favoring aggressive strategies
            ("ANIMAL_COMPANION", "Animal Companion", "HUNTER", "SPELL", 3, None, None, "top_tier", ["aggro", "midrange"]),
            ("HUNTER_MARK", "Hunter's Mark", "HUNTER", "SPELL", 1, None, None, "strong", ["aggro"]),
            ("TRACKING", "Tracking", "HUNTER", "SPELL", 1, None, None, "average", ["aggro", "midrange"]),
            
            # Paladin cards with different archetype affinities
            ("CONSECRATION", "Consecration", "PALADIN", "SPELL", 4, None, None, "strong", ["control", "midrange"]),
            ("BLESSING_OF_KINGS", "Blessing of Kings", "PALADIN", "SPELL", 4, None, None, "strong", ["aggro", "midrange"]),
            ("TRUESILVER_CHAMPION", "Truesilver Champion", "PALADIN", "WEAPON", 4, 4, 2, "strong", ["midrange", "control"]),
            
            # Priest cards with control focus
            ("SHADOW_WORD_DEATH", "Shadow Word: Death", "PRIEST", "SPELL", 3, None, None, "strong", ["control"]),
            ("HOLY_NOVA", "Holy Nova", "PRIEST", "SPELL", 5, None, None, "strong", ["control"]),
            ("POWER_WORD_SHIELD", "Power Word: Shield", "PRIEST", "SPELL", 1, None, None, "average", ["control", "midrange"]),
            
            # Rogue cards with tempo focus
            ("BACKSTAB", "Backstab", "ROGUE", "SPELL", 0, None, None, "strong", ["aggro", "midrange"]),
            ("SI7_AGENT", "SI:7 Agent", "ROGUE", "MINION", 3, 3, 3, "strong", ["midrange"]),
            ("EVISCERATE", "Eviscerate", "ROGUE", "SPELL", 2, None, None, "strong", ["aggro", "midrange"]),
            
            # Shaman cards with overload synergy
            ("LIGHTNING_BOLT", "Lightning Bolt", "SHAMAN", "SPELL", 1, None, None, "strong", ["aggro"]),
            ("FIRE_ELEMENTAL", "Fire Elemental", "SHAMAN", "MINION", 6, 6, 5, "strong", ["midrange", "control"]),
            ("HEX", "Hex", "SHAMAN", "SPELL", 3, None, None, "strong", ["control", "midrange"]),
            
            # Warlock cards with health cost synergy
            ("FLAME_IMP", "Flame Imp", "WARLOCK", "MINION", 1, 3, 2, "strong", ["aggro"]),
            ("DOOMGUARD", "Doomguard", "WARLOCK", "MINION", 5, 5, 7, "strong", ["aggro", "midrange"]),
            ("HELLFIRE", "Hellfire", "WARLOCK", "SPELL", 4, None, None, "strong", ["control"]),
            
            # Druid cards with ramp and choice synergy
            ("INNERVATE", "Innervate", "DRUID", "SPELL", 0, None, None, "strong", ["aggro", "midrange"]),
            ("SWIPE", "Swipe", "DRUID", "SPELL", 4, None, None, "strong", ["control", "midrange"]),
            ("WILD_GROWTH", "Wild Growth", "DRUID", "SPELL", 2, None, None, "strong", ["control"]),
            
            # Demon Hunter cards with aggressive focus
            ("TWIN_SLICE", "Twin Slice", "DEMONHUNTER", "SPELL", 1, None, None, "strong", ["aggro"]),
            ("CHAOS_STRIKE", "Chaos Strike", "DEMONHUNTER", "SPELL", 2, None, None, "strong", ["aggro", "midrange"]),
            ("METAMORPHOSIS", "Metamorphosis", "DEMONHUNTER", "SPELL", 5, None, None, "strong", ["midrange", "control"]),
            
            # Neutral cards with varying power levels
            ("CHILLWIND_YETI", "Chillwind Yeti", "NEUTRAL", "MINION", 4, 4, 5, "strong", ["midrange"]),
            ("BOULDERFIST_OGRE", "Boulderfist Ogre", "NEUTRAL", "MINION", 6, 6, 7, "strong", ["midrange", "control"]),
            ("WISP", "Wisp", "NEUTRAL", "MINION", 0, 1, 1, "weak", ["aggro"]),
            ("MAGMA_RAGER", "Magma Rager", "NEUTRAL", "MINION", 3, 5, 1, "weak", []),
            
            # Tech cards with situational value
            ("THE_BLACK_KNIGHT", "The Black Knight", "NEUTRAL", "MINION", 6, 4, 5, "situational", ["control"]),
            ("BIG_GAME_HUNTER", "Big Game Hunter", "NEUTRAL", "MINION", 3, 4, 2, "situational", ["control", "midrange"]),
        ]
        
        for card_id, name, player_class, card_type, cost, attack, health, tier, archetypes in meta_cards:
            card_data = {
                "id": card_id,
                "name": name,
                "playerClass": player_class,
                "type": card_type,
                "cost": cost,
                "dbfId": hash(card_id) % 10000 + 5000,
                "meta_tier": tier,
                "archetype_affinity": archetypes
            }
            
            if attack is not None:
                card_data["attack"] = attack
            if health is not None:
                card_data["health"] = health
                
            cards_data[card_id] = card_data
        
        return cards_data
    
    def _create_meta_hero_winrates(self):
        """Create hero winrates based on known meta performance."""
        # Based on actual Arena meta data (approximate)
        return {
            "MAGE": 0.5580,        # Top tier - excellent spells and AOE
            "PALADIN": 0.5520,     # Strong tier - good weapons and buffs
            "WARRIOR": 0.5480,     # Strong tier - weapons and removal
            "HUNTER": 0.5420,      # Good tier - aggressive tools
            "ROGUE": 0.5380,       # Good tier - tempo and removal
            "WARLOCK": 0.5340,     # Average tier - inconsistent but powerful
            "DRUID": 0.5280,       # Average tier - ramp and choice cards
            "DEMONHUNTER": 0.5240, # Average tier - newer class, aggressive
            "SHAMAN": 0.5180,      # Below average - overload drawbacks
            "PRIEST": 0.5120       # Lowest tier - reactive, inconsistent
        }
    
    def _create_meta_hsreplay_data(self):
        """Create HSReplay data reflecting known meta performance."""
        hsreplay_data = {}
        
        # Map tiers to winrates
        tier_winrates = {
            "top_tier": 0.62,
            "strong": 0.57,
            "average": 0.50,
            "weak": 0.43,
            "situational": 0.52
        }
        
        for card_id, card_data in self.meta_cards_data.items():
            if card_data["type"] != "HERO":
                tier = card_data.get("meta_tier", "average")
                base_winrate = tier_winrates[tier]
                
                # Add class-specific modifiers
                class_modifiers = {
                    "MAGE": 0.02,      # Mage cards perform better
                    "PALADIN": 0.015,
                    "WARRIOR": 0.01,
                    "HUNTER": 0.005,
                    "ROGUE": 0.005,
                    "NEUTRAL": 0.0,
                    "WARLOCK": -0.005,
                    "DRUID": -0.01,
                    "DEMONHUNTER": -0.01,
                    "SHAMAN": -0.015,
                    "PRIEST": -0.02    # Priest cards underperform
                }
                
                class_modifier = class_modifiers.get(card_data["playerClass"], 0.0)
                final_winrate = base_winrate + class_modifier
                
                hsreplay_data[card_id] = {
                    "overall_winrate": final_winrate,
                    "play_rate": 0.20 if tier == "top_tier" else 0.12,
                    "pick_rate": 0.35 if tier == "top_tier" else 0.25,
                    "meta_tier": tier,
                    "archetype_performance": {
                        archetype: final_winrate + 0.01 for archetype in card_data.get("archetype_affinity", [])
                    }
                }
        
        return hsreplay_data
    
    def _create_meta_scenarios(self):
        """Create test scenarios based on known meta situations."""
        return [
            {
                "name": "Mage vs Priest vs Shaman - Clear Meta Ranking",
                "hero_choices": ["MAGE", "PRIEST", "SHAMAN"],
                "expected_order": ["MAGE", "PRIEST", "SHAMAN"],  # Based on meta winrates
                "scenario_type": "hero_meta_ranking"
            },
            {
                "name": "Mage Spell Synergy Test",
                "hero_class": "MAGE",
                "card_choices": ["FLAMESTRIKE", "CHILLWIND_YETI", "WISP"],
                "expected_pick": "FLAMESTRIKE",  # Excellent Mage spell
                "scenario_type": "class_synergy"
            },
            {
                "name": "Warrior Weapon vs Spell Choice",
                "hero_class": "WARRIOR",
                "card_choices": ["FIERY_WAR_AXE", "POWER_WORD_SHIELD", "TRACKING"],
                "expected_pick": "FIERY_WAR_AXE",  # Perfect Warrior synergy
                "scenario_type": "class_synergy"
            },
            {
                "name": "Hunter Aggressive vs Control Cards",
                "hero_class": "HUNTER",
                "card_choices": ["ANIMAL_COMPANION", "HOLY_NOVA", "CONSECRATION"],
                "expected_pick": "ANIMAL_COMPANION",  # Fits Hunter aggro style
                "scenario_type": "archetype_match"
            },
            {
                "name": "Priest Control Card Selection",
                "hero_class": "PRIEST",
                "card_choices": ["SHADOW_WORD_DEATH", "FLAME_IMP", "LIGHTNING_BOLT"],
                "expected_pick": "SHADOW_WORD_DEATH",  # Priest control tool
                "scenario_type": "archetype_match"
            },
            {
                "name": "Power Level vs Class Synergy Tradeoff",
                "hero_class": "SHAMAN",
                "card_choices": ["FLAMESTRIKE", "LIGHTNING_BOLT", "WISP"],
                "expected_pick": "FLAMESTRIKE",  # Power level should win over weak synergy
                "scenario_type": "power_vs_synergy"
            },
            {
                "name": "Paladin Midrange Archetype",
                "hero_class": "PALADIN",
                "card_choices": ["TRUESILVER_CHAMPION", "BACKSTAB", "INNERVATE"],
                "expected_pick": "TRUESILVER_CHAMPION",  # Perfect Paladin weapon
                "scenario_type": "archetype_match"
            },
            {
                "name": "Neutral High-Value Cards",
                "hero_class": "WARRIOR",
                "card_choices": ["THE_BLACK_KNIGHT", "CHILLWIND_YETI", "MAGMA_RAGER"],
                "expected_pick": "CHILLWIND_YETI",  # Consistent strong neutral
                "scenario_type": "neutral_evaluation"
            }
        ]
    
    def _create_hero_archetypes(self):
        """Create hero archetype preferences based on meta knowledge."""
        return {
            "MAGE": {
                "primary_archetype": "control",
                "secondary_archetype": "midrange",
                "strengths": ["spells", "aoe", "removal"],
                "preferred_curve": "high"
            },
            "PALADIN": {
                "primary_archetype": "midrange",
                "secondary_archetype": "control",
                "strengths": ["weapons", "buffs", "divine_shield"],
                "preferred_curve": "balanced"
            },
            "WARRIOR": {
                "primary_archetype": "midrange",
                "secondary_archetype": "control",
                "strengths": ["weapons", "armor", "removal"],
                "preferred_curve": "balanced"
            },
            "HUNTER": {
                "primary_archetype": "aggro",
                "secondary_archetype": "midrange",
                "strengths": ["beasts", "direct_damage", "tempo"],
                "preferred_curve": "low"
            },
            "ROGUE": {
                "primary_archetype": "midrange",
                "secondary_archetype": "aggro",
                "strengths": ["combo", "tempo", "removal"],
                "preferred_curve": "low"
            },
            "WARLOCK": {
                "primary_archetype": "aggro",
                "secondary_archetype": "control",
                "strengths": ["demons", "card_draw", "sacrifice"],
                "preferred_curve": "extreme"
            },
            "DRUID": {
                "primary_archetype": "midrange",
                "secondary_archetype": "control",
                "strengths": ["ramp", "choice", "big_minions"],
                "preferred_curve": "high"
            },
            "DEMONHUNTER": {
                "primary_archetype": "aggro",
                "secondary_archetype": "midrange",
                "strengths": ["attack", "outcast", "tempo"],
                "preferred_curve": "low"
            },
            "SHAMAN": {
                "primary_archetype": "midrange",
                "secondary_archetype": "control",
                "strengths": ["overload", "totems", "elementals"],
                "preferred_curve": "balanced"
            },
            "PRIEST": {
                "primary_archetype": "control",
                "secondary_archetype": "midrange",
                "strengths": ["healing", "removal", "card_generation"],
                "preferred_curve": "high"
            }
        }
    
    def test_hero_meta_ranking_accuracy(self):
        """Test hero ranking accuracy against known meta performance."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Test meta ranking scenarios
            meta_correct_predictions = 0
            meta_total_predictions = 0
            
            meta_scenarios = [s for s in self.meta_scenarios if s["scenario_type"] == "hero_meta_ranking"]
            
            for scenario in meta_scenarios:
                hero_choices = scenario["hero_choices"]
                expected_order = scenario["expected_order"]
                
                recommendations = hero_advisor.recommend_hero(hero_choices)
                actual_order = [rec["hero_class"] for rec in sorted(recommendations, key=lambda x: x["rank"])]
                
                # Check if ranking matches meta expectations
                for i, expected_hero in enumerate(expected_order):
                    if i < len(actual_order) and actual_order[i] == expected_hero:
                        meta_correct_predictions += 1
                    meta_total_predictions += 1
            
            # Test with multiple hero combinations to get better sample size
            additional_tests = [
                (["MAGE", "PALADIN", "WARRIOR"], ["MAGE", "PALADIN", "WARRIOR"]),
                (["HUNTER", "ROGUE", "WARLOCK"], ["HUNTER", "ROGUE", "WARLOCK"]),
                (["DRUID", "DEMONHUNTER", "SHAMAN"], ["DRUID", "DEMONHUNTER", "SHAMAN"]),
                (["MAGE", "PRIEST"], ["MAGE", "PRIEST"]),
                (["WARRIOR", "SHAMAN"], ["WARRIOR", "SHAMAN"])
            ]
            
            for heroes, expected in additional_tests:
                recommendations = hero_advisor.recommend_hero(heroes)
                actual_order = [rec["hero_class"] for rec in sorted(recommendations, key=lambda x: x["rank"])]
                
                for i, expected_hero in enumerate(expected):
                    if i < len(actual_order) and actual_order[i] == expected_hero:
                        meta_correct_predictions += 1
                    meta_total_predictions += 1
            
            # Calculate meta accuracy
            meta_accuracy = meta_correct_predictions / meta_total_predictions if meta_total_predictions > 0 else 0
            
            self.assertGreater(meta_accuracy, self.META_ACCURACY_THRESHOLD,
                             f"Meta ranking accuracy {meta_accuracy:.3f} below threshold {self.META_ACCURACY_THRESHOLD}")
    
    def test_hero_specific_card_synergy_accuracy(self):
        """Test accuracy of hero-specific card synergy detection."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hsreplay.return_value = self.meta_hsreplay_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            synergy_correct_predictions = 0
            synergy_total_predictions = 0
            
            # Test class synergy scenarios
            synergy_scenarios = [s for s in self.meta_scenarios if s["scenario_type"] == "class_synergy"]
            
            for scenario in synergy_scenarios:
                hero_class = scenario["hero_class"]
                card_choices = [self.meta_cards_data[card_id] for card_id in scenario["card_choices"]]
                expected_pick = scenario["expected_pick"]
                
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class=hero_class
                )
                
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_class)
                recommended_index = recommendation.decision_data["recommended_pick_index"]
                recommended_card = card_choices[recommended_index]
                
                if recommended_card["id"] == expected_pick:
                    synergy_correct_predictions += 1
                synergy_total_predictions += 1
            
            # Calculate synergy accuracy
            synergy_accuracy = synergy_correct_predictions / synergy_total_predictions if synergy_total_predictions > 0 else 0
            
            self.assertGreater(synergy_accuracy, self.HERO_SYNERGY_ACCURACY,
                             f"Hero synergy accuracy {synergy_accuracy:.3f} below threshold {self.HERO_SYNERGY_ACCURACY}")
    
    def test_archetype_matching_accuracy(self):
        """Test accuracy of archetype-appropriate card recommendations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hsreplay.return_value = self.meta_hsreplay_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            archetype_correct_predictions = 0
            archetype_total_predictions = 0
            
            # Test archetype matching scenarios
            archetype_scenarios = [s for s in self.meta_scenarios if s["scenario_type"] == "archetype_match"]
            
            for scenario in archetype_scenarios:
                hero_class = scenario["hero_class"]
                card_choices = [self.meta_cards_data[card_id] for card_id in scenario["card_choices"]]
                expected_pick = scenario["expected_pick"]
                
                # Create deck state that matches hero's primary archetype
                hero_archetype = self.hero_archetypes[hero_class]
                primary_archetype = hero_archetype["primary_archetype"]
                
                archetype_weights = {"aggro": 0.2, "midrange": 0.3, "control": 0.5}
                if primary_archetype == "aggro":
                    archetype_weights = {"aggro": 0.7, "midrange": 0.3, "control": 0.0}
                elif primary_archetype == "midrange":
                    archetype_weights = {"aggro": 0.3, "midrange": 0.6, "control": 0.1}
                elif primary_archetype == "control":
                    archetype_weights = {"aggro": 0.1, "midrange": 0.3, "control": 0.6}
                
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings=archetype_weights,
                    synergy_groups={},
                    hero_class=hero_class
                )
                
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_class)
                recommended_index = recommendation.decision_data["recommended_pick_index"]
                recommended_card = card_choices[recommended_index]
                
                if recommended_card["id"] == expected_pick:
                    archetype_correct_predictions += 1
                archetype_total_predictions += 1
            
            # Calculate archetype accuracy
            archetype_accuracy = archetype_correct_predictions / archetype_total_predictions if archetype_total_predictions > 0 else 0
            
            self.assertGreater(archetype_accuracy, self.ARCHETYPE_ACCURACY,
                             f"Archetype matching accuracy {archetype_accuracy:.3f} below threshold {self.ARCHETYPE_ACCURACY}")
    
    def test_power_level_vs_synergy_tradeoffs(self):
        """Test handling of power level vs synergy tradeoffs."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hsreplay.return_value = self.meta_hsreplay_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Test power vs synergy scenarios
            power_vs_synergy_scenarios = [s for s in self.meta_scenarios if s["scenario_type"] == "power_vs_synergy"]
            
            correct_tradeoffs = 0
            total_tradeoffs = 0
            
            for scenario in power_vs_synergy_scenarios:
                hero_class = scenario["hero_class"]
                card_choices = [self.meta_cards_data[card_id] for card_id in scenario["card_choices"]]
                expected_pick = scenario["expected_pick"]
                
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class=hero_class
                )
                
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_class)
                recommended_index = recommendation.decision_data["recommended_pick_index"]
                recommended_card = card_choices[recommended_index]
                
                if recommended_card["id"] == expected_pick:
                    correct_tradeoffs += 1
                total_tradeoffs += 1
            
            # Should correctly handle power vs synergy tradeoffs at least 70% of the time
            tradeoff_accuracy = correct_tradeoffs / total_tradeoffs if total_tradeoffs > 0 else 1.0
            self.assertGreater(tradeoff_accuracy, 0.70, 
                             f"Power vs synergy tradeoff accuracy {tradeoff_accuracy:.3f} below 70%")
    
    def test_neutral_card_evaluation_consistency(self):
        """Test consistent evaluation of neutral cards across different heroes."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hsreplay.return_value = self.meta_hsreplay_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            card_evaluator = CardEvaluationEngine()
            
            # Test neutral card evaluation across different heroes
            neutral_cards = [
                "CHILLWIND_YETI",
                "BOULDERFIST_OGRE", 
                "THE_BLACK_KNIGHT",
                "BIG_GAME_HUNTER"
            ]
            
            hero_classes = ["MAGE", "WARRIOR", "HUNTER", "PRIEST"]
            
            # Evaluate each neutral card with each hero
            for card_id in neutral_cards:
                card_data = self.meta_cards_data[card_id]
                scores_by_hero = {}
                
                for hero_class in hero_classes:
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class=hero_class
                    )
                    
                    evaluation = card_evaluator.evaluate_card(card_data, deck_state, hero_class)
                    scores_by_hero[hero_class] = evaluation.total_score
                
                # Neutral cards should have relatively consistent scores across heroes
                scores = list(scores_by_hero.values())
                if len(scores) > 1:
                    score_std = statistics.stdev(scores)
                    score_mean = statistics.mean(scores)
                    coefficient_of_variation = score_std / score_mean if score_mean > 0 else 0
                    
                    # Coefficient of variation should be relatively low for neutral cards
                    self.assertLess(coefficient_of_variation, 0.25, 
                                   f"Neutral card {card_id} has inconsistent evaluation across heroes")
    
    def test_meta_tier_correlation_with_recommendations(self):
        """Test that meta tier correlates with recommendation frequency."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hsreplay.return_value = self.meta_hsreplay_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Track recommendation frequency by meta tier
            tier_recommendations = {"top_tier": 0, "strong": 0, "average": 0, "weak": 0, "situational": 0}
            tier_total_choices = {"top_tier": 0, "strong": 0, "average": 0, "weak": 0, "situational": 0}
            
            # Create many test scenarios
            hero_classes = ["MAGE", "WARRIOR", "HUNTER", "PALADIN", "PRIEST"]
            
            for hero_class in hero_classes:
                # Get cards of different tiers for this hero
                hero_cards = [card for card in self.meta_cards_data.values() 
                            if card["type"] != "HERO" and 
                            (card["playerClass"] == hero_class or card["playerClass"] == "NEUTRAL")]
                
                # Group by tier
                cards_by_tier = {}
                for card in hero_cards:
                    tier = card.get("meta_tier", "average")
                    if tier not in cards_by_tier:
                        cards_by_tier[tier] = []
                    cards_by_tier[tier].append(card)
                
                # Create choice scenarios mixing different tiers
                for _ in range(5):  # 5 scenarios per hero
                    choice_cards = []
                    choice_tiers = []
                    
                    # Pick one card from each available tier (up to 3 cards)
                    available_tiers = list(cards_by_tier.keys())[:3]
                    for tier in available_tiers:
                        if cards_by_tier[tier]:
                            choice_cards.append(cards_by_tier[tier][0])
                            choice_tiers.append(tier)
                    
                    if len(choice_cards) >= 2:
                        deck_state = DeckState(
                            cards_drafted=[],
                            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                            archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                            synergy_groups={},
                            hero_class=hero_class
                        )
                        
                        recommendation = grandmaster_advisor.get_recommendation(choice_cards, deck_state, hero_class)
                        recommended_index = recommendation.decision_data["recommended_pick_index"]
                        recommended_tier = choice_tiers[recommended_index]
                        
                        # Track recommendations and choices
                        tier_recommendations[recommended_tier] += 1
                        for tier in choice_tiers:
                            tier_total_choices[tier] += 1
            
            # Calculate recommendation rates by tier
            recommendation_rates = {}
            for tier in tier_recommendations:
                if tier_total_choices[tier] > 0:
                    recommendation_rates[tier] = tier_recommendations[tier] / tier_total_choices[tier]
                else:
                    recommendation_rates[tier] = 0.0
            
            # Top tier cards should be recommended more often than weak cards
            if "top_tier" in recommendation_rates and "weak" in recommendation_rates:
                self.assertGreater(recommendation_rates["top_tier"], recommendation_rates["weak"],
                                 "Top tier cards should be recommended more often than weak cards")
            
            # Strong cards should be recommended more often than average cards
            if "strong" in recommendation_rates and "average" in recommendation_rates:
                self.assertGreater(recommendation_rates["strong"], recommendation_rates["average"],
                                 "Strong cards should be recommended more often than average cards")
    
    def test_hero_winrate_correlation_with_recommendation_confidence(self):
        """Test that hero winrates correlate with recommendation confidence."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.meta_cards_data
            mock_hero_winrates.return_value = self.meta_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Test confidence correlation with meta performance
            hero_winrate_confidence_pairs = []
            
            # Test different hero combinations
            test_combinations = [
                ["MAGE", "PRIEST"],      # High vs Low winrate
                ["PALADIN", "SHAMAN"],   # Strong vs Weak
                ["WARRIOR", "DRUID"],    # Good vs Average
                ["HUNTER", "WARLOCK"],   # Similar but different
                ["ROGUE", "DEMONHUNTER"] # Close winrates
            ]
            
            for heroes in test_combinations:
                recommendations = hero_advisor.recommend_hero(heroes)
                
                for rec in recommendations:
                    hero_class = rec["hero_class"]
                    confidence = rec.get("confidence", 0.5)
                    winrate = self.meta_hero_winrates[hero_class]
                    
                    hero_winrate_confidence_pairs.append((winrate, confidence))
            
            # Calculate correlation between winrate and confidence
            if len(hero_winrate_confidence_pairs) >= 3:
                winrates = [pair[0] for pair in hero_winrate_confidence_pairs]
                confidences = [pair[1] for pair in hero_winrate_confidence_pairs]
                
                correlation = self._calculate_correlation(winrates, confidences)
                
                # Should be positive correlation (higher winrate = higher confidence)
                self.assertGreater(correlation, 0.3, 
                                 f"Winrate-confidence correlation {correlation:.3f} too low")
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class TestHeroSpecificAccuracyEdgeCases(unittest.TestCase):
    """Test edge cases in hero-specific accuracy validation."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.minimal_data = {
            "HERO_01": {
                "id": "HERO_01",
                "name": "Test Hero",
                "playerClass": "WARRIOR",
                "type": "HERO",
                "dbfId": 999
            },
            "TEST_001": {
                "id": "TEST_001",
                "name": "Test Card",
                "playerClass": "NEUTRAL",
                "type": "MINION",
                "cost": 3,
                "meta_tier": "average",
                "dbfId": 1000
            }
        }
    
    def test_accuracy_with_unknown_meta_cards(self):
        """Test accuracy validation when cards have no meta information."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.minimal_data
            mock_hsreplay.return_value = {}
            mock_hero_winrates.return_value = {"WARRIOR": 0.50}
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Test with unknown meta cards
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            card_choices = [self.minimal_data["TEST_001"]] * 3
            recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
            
            # Should still provide a recommendation
            self.assertIsNotNone(recommendation)
            self.assertIn("recommended_pick_index", recommendation.decision_data)
    
    def test_accuracy_with_identical_meta_performance(self):
        """Test accuracy when all options have identical meta performance."""
        identical_winrates = {
            "WARRIOR": 0.50,
            "MAGE": 0.50,
            "HUNTER": 0.50
        }
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.minimal_data
            mock_hero_winrates.return_value = identical_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Test with identical winrates
            recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            # Should still provide consistent ranking
            self.assertEqual(len(recommendations), 3)
            ranks = [rec["rank"] for rec in recommendations]
            self.assertEqual(sorted(ranks), [1, 2, 3])


def run_hero_specific_accuracy_tests():
    """Run all hero-specific accuracy tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestHeroSpecificAccuracy))
    test_suite.addTest(unittest.makeSuite(TestHeroSpecificAccuracyEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_hero_specific_accuracy_tests()
    
    if success:
        print("\n✅ All hero-specific accuracy tests passed!")
    else:
        print("\n❌ Some hero-specific accuracy tests failed!")
        sys.exit(1)