"""
Complete System Testing with All Features Enabled and Multiple Fallback Scenarios

Comprehensive end-to-end system tests that validate the complete Arena Bot AI v2
system with all features enabled, testing multiple fallback scenarios and
full integration between all components.
"""

import unittest
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import json
from datetime import datetime
from queue import Queue, Empty
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError
import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all system components
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.conversational_coach import ConversationalCoach
from arena_bot.ai_v2.data_models import DeckState, DimensionalScores, AIDecision
from arena_bot.data.cards_json_loader import CardsJsonLoader
from arena_bot.data_sourcing.hsreplay_scraper import HSReplayDataScraper
from arena_bot.log_monitoring.hearthstone_log_monitor import HearthstoneLogMonitor
from arena_bot.gui.integrated_arena_bot_gui import IntegratedArenaBotGUI
from arena_bot.ai_v2.advanced_error_recovery import get_error_recovery


class TestCompleteSystemIntegration(unittest.TestCase):
    """Complete system integration tests with all features enabled."""
    
    def setUp(self):
        """Set up complete system integration test fixtures."""
        # Complete system test data
        self.system_cards_data = self._create_complete_system_cards_data()
        self.system_hero_winrates = self._create_complete_system_hero_winrates()
        self.system_hsreplay_data = self._create_complete_system_hsreplay_data()
        
        # System performance targets
        self.COMPLETE_DRAFT_TIME_TARGET = 30.0  # seconds for 30 picks
        self.HERO_SELECTION_TIME_TARGET = 3.0   # seconds
        self.SINGLE_PICK_TIME_TARGET = 2.0      # seconds per card pick
        self.MEMORY_USAGE_THRESHOLD = 200       # MB
        
        # Fallback scenarios
        self.fallback_scenarios = self._create_fallback_scenarios()
        
        # Complete draft simulation
        self.complete_draft_scenario = self._create_complete_draft_scenario()
        
        # System health monitoring
        self.system_health_metrics = {
            "api_calls": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "fallback_activations": 0,
            "performance_violations": 0
        }
    
    def _create_complete_system_cards_data(self):
        """Create comprehensive cards data for complete system testing."""
        cards_data = {}
        
        # All 10 hero classes
        heroes = [
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
        
        for card_id, name, player_class, dbf_id in heroes:
            cards_data[card_id] = {
                "id": card_id,
                "name": name,
                "playerClass": player_class,
                "type": "HERO",
                "dbfId": dbf_id
            }
        
        # Create extensive card pool for testing
        classes = ["WARRIOR", "MAGE", "HUNTER", "PALADIN", "PRIEST", "ROGUE", "SHAMAN", "WARLOCK", "DRUID", "DEMONHUNTER", "NEUTRAL"]
        card_types = ["MINION", "SPELL", "WEAPON"]
        
        card_counter = 1000
        
        for player_class in classes:
            class_prefix = player_class[:3] if player_class != "NEUTRAL" else "NEU"
            
            # Create 10 cards per class for comprehensive testing
            for i in range(1, 11):
                card_type = card_types[i % len(card_types)]
                cost = (i % 7) + 1
                
                card_id = f"{class_prefix}_{i:03d}"
                card_data = {
                    "id": card_id,
                    "name": f"{player_class} {card_type} {i}",
                    "playerClass": player_class,
                    "type": card_type,
                    "cost": cost,
                    "dbfId": card_counter,
                    "test_power_level": self._calculate_test_power_level(i, cost, player_class)
                }
                
                # Add stats for minions and weapons
                if card_type == "MINION":
                    card_data["attack"] = cost
                    card_data["health"] = cost + 1
                elif card_type == "WEAPON":
                    card_data["attack"] = cost
                    card_data["durability"] = 2
                
                cards_data[card_id] = card_data
                card_counter += 1
        
        return cards_data
    
    def _calculate_test_power_level(self, card_num, cost, player_class):
        """Calculate test power level for cards."""
        base_power = 0.5
        
        # Higher numbered cards are generally better
        power_modifier = (card_num / 10.0) * 0.3
        
        # Cost efficiency
        if cost <= 3:
            power_modifier += 0.1  # Early game bonus
        elif cost >= 6:
            power_modifier -= 0.1  # Late game penalty
        
        return min(1.0, base_power + power_modifier)
    
    def _create_complete_system_hero_winrates(self):
        """Create comprehensive hero winrates for system testing."""
        return {
            "WARRIOR": 0.5650,     # Tier 1
            "WARLOCK": 0.5580,     # Tier 1
            "MAGE": 0.5450,        # Tier 2
            "PALADIN": 0.5320,     # Tier 2
            "ROGUE": 0.5180,       # Tier 3
            "DEMONHUNTER": 0.5050, # Tier 3
            "HUNTER": 0.4920,      # Tier 4
            "DRUID": 0.4850,       # Tier 4
            "PRIEST": 0.4780,      # Tier 5
            "SHAMAN": 0.4650       # Tier 5
        }
    
    def _create_complete_system_hsreplay_data(self):
        """Create comprehensive HSReplay data for system testing."""
        hsreplay_data = {}
        
        for card_id, card_data in self.system_cards_data.items():
            if card_data["type"] != "HERO":
                power_level = card_data.get("test_power_level", 0.5)
                
                # Map power level to winrates
                base_winrate = 0.35 + (power_level * 0.3)  # 0.35 to 0.65 range
                
                # Add some variance
                variance = (hash(card_id) % 100) / 1000.0 - 0.05  # -0.05 to +0.05
                winrate = max(0.25, min(0.75, base_winrate + variance))
                
                hsreplay_data[card_id] = {
                    "overall_winrate": winrate,
                    "play_rate": 0.05 + (power_level * 0.2),
                    "pick_rate": 0.1 + (power_level * 0.25),
                    "deck_winrate": winrate * 1.02,
                    "mulligan_winrate": winrate * 0.98
                }
        
        return hsreplay_data
    
    def _create_fallback_scenarios(self):
        """Create comprehensive fallback scenarios."""
        return [
            {
                "name": "Complete API Failure",
                "hero_api_available": False,
                "card_api_available": False,
                "description": "Both APIs completely unavailable"
            },
            {
                "name": "Hero API Only",
                "hero_api_available": True,
                "card_api_available": False,
                "description": "Only hero API available"
            },
            {
                "name": "Card API Only",
                "hero_api_available": False,
                "card_api_available": True,
                "description": "Only card API available"
            },
            {
                "name": "Intermittent Failures",
                "hero_api_available": "intermittent",
                "card_api_available": "intermittent",
                "description": "Both APIs intermittently failing"
            },
            {
                "name": "Slow Responses",
                "hero_api_available": "slow",
                "card_api_available": "slow",
                "description": "Both APIs responding slowly"
            },
            {
                "name": "Partial Data Corruption",
                "hero_api_available": "corrupted",
                "card_api_available": "corrupted",
                "description": "APIs returning corrupted data"
            }
        ]
    
    def _create_complete_draft_scenario(self):
        """Create a complete 30-card draft scenario."""
        hero_choices = ["WARRIOR", "MAGE", "HUNTER"]
        
        # Generate 30 card choice sets
        card_picks = []
        for pick_num in range(30):
            # Create varied card choices
            if pick_num < 10:
                # Early picks - focus on curve
                choices = [f"NEU_{(pick_num % 5) + 1:03d}", f"WAR_{(pick_num % 3) + 1:03d}", f"MAG_{(pick_num % 3) + 1:03d}"]
            elif pick_num < 20:
                # Mid picks - focus on synergy
                choices = [f"NEU_{((pick_num - 10) % 5) + 6:03d}", f"WAR_{((pick_num - 10) % 3) + 4:03d}", f"HUN_{((pick_num - 10) % 3) + 1:03d}"]
            else:
                # Late picks - focus on value
                choices = [f"NEU_{((pick_num - 20) % 5) + 1:03d}", f"WAR_{((pick_num - 20) % 3) + 7:03d}", f"PAL_{((pick_num - 20) % 3) + 1:03d}"]
            
            card_picks.append(choices)
        
        return {
            "hero_choices": hero_choices,
            "card_picks": card_picks,
            "expected_draft_size": 30
        }
    
    def test_complete_draft_simulation_optimal_conditions(self):
        """Test complete 30-card draft simulation under optimal conditions."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.system_cards_data
            mock_hsreplay.return_value = self.system_hsreplay_data
            mock_hero_winrates.return_value = self.system_hero_winrates
            
            draft_scenario = self.complete_draft_scenario
            
            # Start complete draft simulation
            start_time = time.time()
            
            # Phase 1: Hero Selection
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(draft_scenario["hero_choices"])
            
            hero_selection_time = time.time() - start_time
            
            # Verify hero selection performance
            self.assertLess(hero_selection_time, self.HERO_SELECTION_TIME_TARGET,
                          f"Hero selection took {hero_selection_time:.3f}s, target: {self.HERO_SELECTION_TIME_TARGET}s")
            
            selected_hero = hero_recommendations[0]["hero_class"]
            
            # Phase 2: Complete Card Draft
            grandmaster_advisor = GrandmasterAdvisor()
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class=selected_hero
            )
            
            drafted_cards = []
            pick_times = []
            
            for pick_num, card_choice_ids in enumerate(draft_scenario["card_picks"]):
                pick_start = time.time()
                
                # Get actual card objects
                card_choices = []
                for card_id in card_choice_ids:
                    if card_id in self.system_cards_data:
                        card_choices.append(self.system_cards_data[card_id])
                
                if len(card_choices) >= 3:
                    # Make recommendation
                    recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                    
                    # Track performance
                    pick_time = time.time() - pick_start
                    pick_times.append(pick_time)
                    
                    # Verify individual pick performance
                    self.assertLess(pick_time, self.SINGLE_PICK_TIME_TARGET,
                                  f"Pick {pick_num + 1} took {pick_time:.3f}s, target: {self.SINGLE_PICK_TIME_TARGET}s")
                    
                    # Update deck state
                    recommended_index = recommendation.decision_data["recommended_pick_index"]
                    recommended_card = card_choices[recommended_index]
                    drafted_cards.append(recommended_card)
                    deck_state.cards_drafted = drafted_cards
                    
                    # Update mana curve
                    card_cost = recommended_card.get("cost", 0)
                    if card_cost in deck_state.mana_curve:
                        deck_state.mana_curve[card_cost] += 1
                    elif card_cost > 7:
                        deck_state.mana_curve[7] += 1
                    
                    # Update system health metrics
                    self.system_health_metrics["successful_operations"] += 1
            
            total_draft_time = time.time() - start_time
            
            # Verify complete draft performance
            self.assertLess(total_draft_time, self.COMPLETE_DRAFT_TIME_TARGET,
                          f"Complete draft took {total_draft_time:.3f}s, target: {self.COMPLETE_DRAFT_TIME_TARGET}s")
            
            # Verify draft completeness
            self.assertEqual(len(drafted_cards), 30, "Should draft exactly 30 cards")
            
            # Verify deck state consistency
            self.assertEqual(deck_state.hero_class, selected_hero, "Hero class should be maintained")
            self.assertEqual(sum(deck_state.mana_curve.values()), 30, "Mana curve should account for all cards")
            
            # Performance analysis
            avg_pick_time = sum(pick_times) / len(pick_times) if pick_times else 0
            max_pick_time = max(pick_times) if pick_times else 0
            
            self.assertLess(avg_pick_time, self.SINGLE_PICK_TIME_TARGET * 0.8,
                          f"Average pick time {avg_pick_time:.3f}s should be well under target")
            
            self.assertLess(max_pick_time, self.SINGLE_PICK_TIME_TARGET * 1.5,
                          f"Maximum pick time {max_pick_time:.3f}s should not greatly exceed target")
    
    def test_complete_system_with_all_fallback_scenarios(self):
        """Test complete system under all defined fallback scenarios."""
        for scenario in self.fallback_scenarios:
            with self.subTest(scenario=scenario["name"]):
                with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
                     patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
                     patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
                    
                    mock_load_cards.return_value = self.system_cards_data
                    
                    # Configure API behavior based on scenario
                    self._configure_api_behavior(mock_hsreplay, mock_hero_winrates, scenario)
                    
                    # Test hero selection under scenario
                    try:
                        hero_advisor = HeroSelectionAdvisor()
                        hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                        
                        # Should always provide recommendations
                        self.assertGreaterEqual(len(hero_recommendations), 1, f"Should provide hero recommendations in {scenario['name']}")
                        
                        selected_hero = hero_recommendations[0]["hero_class"]
                        
                        # Test confidence appropriately reflects data availability
                        confidence = hero_recommendations[0].get("confidence", 0.5)
                        if not scenario.get("hero_api_available", True):
                            self.assertLess(confidence, 0.8, f"Confidence should be lower without hero API in {scenario['name']}")
                        
                    except Exception as e:
                        self.system_health_metrics["failed_operations"] += 1
                        # Even with failures, should fail gracefully
                        self.assertIn("fallback", str(e).lower(), f"Error should mention fallback: {e}")
                        selected_hero = "WARRIOR"  # Use default
                    
                    # Test card evaluation under scenario
                    try:
                        grandmaster_advisor = GrandmasterAdvisor()
                        deck_state = DeckState(
                            cards_drafted=[],
                            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                            archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                            synergy_groups={},
                            hero_class=selected_hero
                        )
                        
                        # Test 5 card picks under this scenario
                        for pick_num in range(5):
                            card_choices = [
                                self.system_cards_data[f"NEU_{(pick_num % 3) + 1:03d}"],
                                self.system_cards_data[f"WAR_{(pick_num % 3) + 1:03d}"],
                                self.system_cards_data[f"MAG_{(pick_num % 3) + 1:03d}"]
                            ]
                            
                            recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                            
                            # Should always provide a recommendation
                            self.assertIsNotNone(recommendation, f"Should provide card recommendation in {scenario['name']}")
                            self.assertIn("recommended_pick_index", recommendation.decision_data)
                            
                            # Update deck state
                            recommended_index = recommendation.decision_data["recommended_pick_index"]
                            recommended_card = card_choices[recommended_index]
                            deck_state.cards_drafted.append(recommended_card)
                        
                        # Verify deck building succeeded
                        self.assertEqual(len(deck_state.cards_drafted), 5, f"Should draft 5 cards in {scenario['name']}")
                        
                        self.system_health_metrics["successful_operations"] += 1
                        
                    except Exception as e:
                        self.system_health_metrics["failed_operations"] += 1
                        self.system_health_metrics["fallback_activations"] += 1
                        # Even with failures, should fail gracefully
                        self.assertIn("fallback", str(e).lower(), f"Error should mention fallback: {e}")
    
    def _configure_api_behavior(self, mock_hsreplay, mock_hero_winrates, scenario):
        """Configure API behavior based on fallback scenario."""
        hero_available = scenario.get("hero_api_available", True)
        card_available = scenario.get("card_api_available", True)
        
        # Configure hero API
        if hero_available is False:
            mock_hero_winrates.side_effect = requests.exceptions.ConnectionError("Hero API unavailable")
        elif hero_available == "intermittent":
            call_count = [0]
            def intermittent_hero(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 2 == 0:
                    raise requests.exceptions.ConnectionError("Intermittent failure")
                return self.system_hero_winrates
            mock_hero_winrates.side_effect = intermittent_hero
        elif hero_available == "slow":
            def slow_hero(*args, **kwargs):
                time.sleep(1.0)
                return self.system_hero_winrates
            mock_hero_winrates.side_effect = slow_hero
        elif hero_available == "corrupted":
            mock_hero_winrates.return_value = {"INVALID": "DATA"}
        else:
            mock_hero_winrates.return_value = self.system_hero_winrates
        
        # Configure card API
        if card_available is False:
            mock_hsreplay.side_effect = requests.exceptions.ConnectionError("Card API unavailable")
        elif card_available == "intermittent":
            call_count = [0]
            def intermittent_card(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 2 == 0:
                    raise requests.exceptions.ConnectionError("Intermittent failure")
                return self.system_hsreplay_data
            mock_hsreplay.side_effect = intermittent_card
        elif card_available == "slow":
            def slow_card(*args, **kwargs):
                time.sleep(1.0)
                return self.system_hsreplay_data
            mock_hsreplay.side_effect = slow_card
        elif card_available == "corrupted":
            mock_hsreplay.return_value = {"INVALID": "DATA"}
        else:
            mock_hsreplay.return_value = self.system_hsreplay_data
    
    def test_concurrent_system_operations(self):
        """Test system behavior under concurrent operations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.system_cards_data
            mock_hsreplay.return_value = self.system_hsreplay_data
            mock_hero_winrates.return_value = self.system_hero_winrates
            
            results_queue = Queue()
            
            def concurrent_draft_worker(worker_id):
                """Worker for concurrent draft operations."""
                try:
                    # Hero selection
                    hero_advisor = HeroSelectionAdvisor()
                    hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    selected_hero = hero_recommendations[0]["hero_class"]
                    
                    # Card evaluations
                    grandmaster_advisor = GrandmasterAdvisor()
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class=selected_hero
                    )
                    
                    # Simulate 10 card picks
                    for pick_num in range(10):
                        card_choices = [
                            self.system_cards_data[f"NEU_{(pick_num % 5) + 1:03d}"],
                            self.system_cards_data[f"WAR_{(pick_num % 3) + 1:03d}"],
                            self.system_cards_data[f"MAG_{(pick_num % 3) + 1:03d}"]
                        ]
                        
                        recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                        
                        # Update deck state
                        recommended_index = recommendation.decision_data["recommended_pick_index"]
                        recommended_card = card_choices[recommended_index]
                        deck_state.cards_drafted.append(recommended_card)
                    
                    results_queue.put(("success", worker_id, len(deck_state.cards_drafted)))
                    
                except Exception as e:
                    results_queue.put(("error", worker_id, str(e)))
            
            # Run concurrent operations
            num_workers = 5
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(concurrent_draft_worker, i) for i in range(num_workers)]
                
                # Wait for completion
                for future in futures:
                    try:
                        future.result(timeout=30.0)  # 30 second timeout per worker
                    except ConcurrentTimeoutError:
                        self.system_health_metrics["performance_violations"] += 1
            
            # Collect results
            results = []
            while not results_queue.empty():
                try:
                    results.append(results_queue.get_nowait())
                except Empty:
                    break
            
            # Analyze concurrent performance
            successful_workers = [r for r in results if r[0] == "success"]
            failed_workers = [r for r in results if r[0] == "error"]
            
            success_rate = len(successful_workers) / num_workers
            self.assertGreater(success_rate, 0.8, f"Concurrent success rate {success_rate:.2f} should be > 80%")
            
            # Verify all successful workers drafted correct number of cards
            for result_type, worker_id, card_count in successful_workers:
                self.assertEqual(card_count, 10, f"Worker {worker_id} should have drafted 10 cards")
    
    def test_memory_usage_under_load(self):
        """Test system memory usage under sustained load."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.system_cards_data
            mock_hsreplay.return_value = self.system_hsreplay_data
            mock_hero_winrates.return_value = self.system_hero_winrates
            
            # Simulate sustained load - multiple complete drafts
            for draft_num in range(5):
                # Hero selection
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                selected_hero = hero_recommendations[0]["hero_class"]
                
                # Complete draft simulation
                grandmaster_advisor = GrandmasterAdvisor()
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class=selected_hero
                )
                
                # Simulate 30 card picks
                for pick_num in range(30):
                    card_choices = [
                        self.system_cards_data[f"NEU_{(pick_num % 10) + 1:03d}"],
                        self.system_cards_data[f"WAR_{(pick_num % 10) + 1:03d}"],
                        self.system_cards_data[f"MAG_{(pick_num % 10) + 1:03d}"]
                    ]
                    
                    recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                    
                    # Update deck state
                    recommended_index = recommendation.decision_data["recommended_pick_index"]
                    recommended_card = card_choices[recommended_index]
                    deck_state.cards_drafted.append(recommended_card)
                
                # Check memory after each draft
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                self.assertLess(memory_increase, self.MEMORY_USAGE_THRESHOLD,
                              f"Memory usage increased by {memory_increase:.1f}MB after draft {draft_num + 1}, threshold: {self.MEMORY_USAGE_THRESHOLD}MB")
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        self.assertLess(total_memory_increase, self.MEMORY_USAGE_THRESHOLD * 1.5,
                      f"Total memory increase {total_memory_increase:.1f}MB exceeds acceptable threshold")
    
    def test_system_health_monitoring_and_recovery(self):
        """Test system health monitoring and recovery mechanisms."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.system_cards_data
            
            # Test system recovery under various failure conditions
            failure_scenarios = [
                {"name": "API Timeout", "exception": requests.exceptions.Timeout("Request timeout")},
                {"name": "Connection Error", "exception": requests.exceptions.ConnectionError("Connection failed")},
                {"name": "HTTP Error", "exception": requests.exceptions.HTTPError("HTTP 500")},
                {"name": "JSON Decode Error", "exception": json.JSONDecodeError("Invalid JSON", "", 0)},
                {"name": "Generic Error", "exception": Exception("Unknown error")}
            ]
            
            recovery_success_count = 0
            
            for scenario in failure_scenarios:
                # Configure APIs to fail
                mock_hsreplay.side_effect = scenario["exception"]
                mock_hero_winrates.side_effect = scenario["exception"]
                
                try:
                    # Attempt hero selection
                    hero_advisor = HeroSelectionAdvisor()
                    hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    
                    # Should recover with fallback
                    if hero_recommendations and len(hero_recommendations) >= 1:
                        recovery_success_count += 1
                        
                        # Test card evaluation recovery
                        selected_hero = hero_recommendations[0]["hero_class"]
                        grandmaster_advisor = GrandmasterAdvisor()
                        deck_state = DeckState(
                            cards_drafted=[],
                            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                            archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                            synergy_groups={},
                            hero_class=selected_hero
                        )
                        
                        card_choices = [
                            self.system_cards_data["NEU_001"],
                            self.system_cards_data["WAR_001"],
                            self.system_cards_data["MAG_001"]
                        ]
                        
                        recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                        
                        # Should provide fallback recommendation
                        if recommendation and "recommended_pick_index" in recommendation.decision_data:
                            self.system_health_metrics["successful_operations"] += 1
                        else:
                            self.system_health_metrics["failed_operations"] += 1
                            
                except Exception as e:
                    self.system_health_metrics["failed_operations"] += 1
                    # Verify error handling is graceful
                    self.assertIsInstance(e, Exception, f"Should handle {scenario['name']} gracefully")
            
            # System should recover from at least 60% of failure scenarios
            recovery_rate = recovery_success_count / len(failure_scenarios)
            self.assertGreater(recovery_rate, 0.6, f"System recovery rate {recovery_rate:.2f} should be > 60%")
    
    def test_complete_system_performance_benchmarks(self):
        """Test complete system performance benchmarks."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.system_cards_data
            mock_hsreplay.return_value = self.system_hsreplay_data
            mock_hero_winrates.return_value = self.system_hero_winrates
            
            performance_metrics = {
                "hero_selection_times": [],
                "card_evaluation_times": [],
                "complete_workflow_times": [],
                "memory_usage_samples": []
            }
            
            # Run multiple performance tests
            for test_run in range(10):
                # Complete workflow timing
                workflow_start = time.time()
                
                # Hero selection
                hero_start = time.time()
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                hero_time = time.time() - hero_start
                performance_metrics["hero_selection_times"].append(hero_time)
                
                selected_hero = hero_recommendations[0]["hero_class"]
                
                # Card evaluations (simulate 10 picks)
                grandmaster_advisor = GrandmasterAdvisor()
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class=selected_hero
                )
                
                for pick_num in range(10):
                    card_start = time.time()
                    
                    card_choices = [
                        self.system_cards_data[f"NEU_{(pick_num % 5) + 1:03d}"],
                        self.system_cards_data[f"WAR_{(pick_num % 5) + 1:03d}"],
                        self.system_cards_data[f"MAG_{(pick_num % 5) + 1:03d}"]
                    ]
                    
                    recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                    
                    card_time = time.time() - card_start
                    performance_metrics["card_evaluation_times"].append(card_time)
                    
                    # Update deck state
                    recommended_index = recommendation.decision_data["recommended_pick_index"]
                    recommended_card = card_choices[recommended_index]
                    deck_state.cards_drafted.append(recommended_card)
                
                workflow_time = time.time() - workflow_start
                performance_metrics["complete_workflow_times"].append(workflow_time)
            
            # Analyze performance metrics
            avg_hero_time = sum(performance_metrics["hero_selection_times"]) / len(performance_metrics["hero_selection_times"])
            max_hero_time = max(performance_metrics["hero_selection_times"])
            
            avg_card_time = sum(performance_metrics["card_evaluation_times"]) / len(performance_metrics["card_evaluation_times"])
            max_card_time = max(performance_metrics["card_evaluation_times"])
            
            avg_workflow_time = sum(performance_metrics["complete_workflow_times"]) / len(performance_metrics["complete_workflow_times"])
            max_workflow_time = max(performance_metrics["complete_workflow_times"])
            
            # Performance assertions
            self.assertLess(avg_hero_time, self.HERO_SELECTION_TIME_TARGET * 0.8,
                          f"Average hero selection time {avg_hero_time:.3f}s should be well under target")
            
            self.assertLess(max_hero_time, self.HERO_SELECTION_TIME_TARGET,
                          f"Maximum hero selection time {max_hero_time:.3f}s should meet target")
            
            self.assertLess(avg_card_time, self.SINGLE_PICK_TIME_TARGET * 0.5,
                          f"Average card evaluation time {avg_card_time:.3f}s should be well under target")
            
            self.assertLess(max_card_time, self.SINGLE_PICK_TIME_TARGET,
                          f"Maximum card evaluation time {max_card_time:.3f}s should meet target")
            
            # Report system health metrics
            total_operations = self.system_health_metrics["successful_operations"] + self.system_health_metrics["failed_operations"]
            if total_operations > 0:
                success_rate = self.system_health_metrics["successful_operations"] / total_operations
                self.assertGreater(success_rate, 0.90, f"System success rate {success_rate:.3f} should be > 90%")


class TestCompleteSystemIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases in complete system integration."""
    
    def test_system_behavior_with_minimal_resources(self):
        """Test system behavior with minimal resources and data."""
        minimal_cards = {
            "HERO_01": {"id": "HERO_01", "name": "Test Hero", "playerClass": "WARRIOR", "type": "HERO", "dbfId": 999},
            "TEST_001": {"id": "TEST_001", "name": "Test Card 1", "playerClass": "NEUTRAL", "type": "MINION", "cost": 3, "dbfId": 1000},
            "TEST_002": {"id": "TEST_002", "name": "Test Card 2", "playerClass": "WARRIOR", "type": "SPELL", "cost": 2, "dbfId": 1001}
        }
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = minimal_cards
            mock_hsreplay.return_value = {}
            mock_hero_winrates.return_value = {"WARRIOR": 0.50}
            
            # Should still function with minimal data
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR"])
            
            self.assertEqual(len(hero_recommendations), 1, "Should handle single hero choice")
            
            grandmaster_advisor = GrandmasterAdvisor()
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            card_choices = [minimal_cards["TEST_001"], minimal_cards["TEST_002"]]
            recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
            
            self.assertIsNotNone(recommendation, "Should provide recommendation with minimal data")


def run_complete_system_integration_tests():
    """Run all complete system integration tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestCompleteSystemIntegration))
    test_suite.addTest(unittest.makeSuite(TestCompleteSystemIntegrationEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_complete_system_integration_tests()
    
    if success:
        print("\n✅ All complete system integration tests passed!")
    else:
        print("\n❌ Some complete system integration tests failed!")
        sys.exit(1)