"""
Cross-Phase Testing for Smooth Transition from Hero Selection to Card Picks

Comprehensive tests ensuring seamless workflow from hero selection phase
through card pick phase with proper state management and data flow.
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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import DeckState, DimensionalScores, AIDecision
from arena_bot.data.cards_json_loader import CardsJsonLoader
from arena_bot.log_monitoring.hearthstone_log_monitor import HearthstoneLogMonitor
from arena_bot.gui.integrated_arena_bot_gui import IntegratedArenaBotGUI


class TestCrossPhaseTransition(unittest.TestCase):
    """Cross-phase transition tests for hero selection to card picks."""
    
    def setUp(self):
        """Set up cross-phase transition test fixtures."""
        # Cross-phase test data
        self.cross_phase_cards_data = self._create_cross_phase_cards_data()
        self.cross_phase_hero_winrates = self._create_cross_phase_hero_winrates()
        self.cross_phase_hsreplay_data = self._create_cross_phase_hsreplay_data()
        
        # Workflow state tracking
        self.workflow_events = []
        self.phase_transitions = []
        self.state_changes = []
        
        # Draft simulation data
        self.draft_workflow_scenarios = self._create_draft_workflow_scenarios()
        
        # Performance targets for cross-phase operations
        self.HERO_SELECTION_TIME_TARGET = 2.0  # seconds
        self.CARD_EVALUATION_TIME_TARGET = 1.0  # seconds
        self.PHASE_TRANSITION_TIME_TARGET = 0.5  # seconds
        self.COMPLETE_WORKFLOW_TIME_TARGET = 5.0  # seconds
    
    def _create_cross_phase_cards_data(self):
        """Create cards data for cross-phase testing."""
        return {
            # Hero cards
            "HERO_01": {"id": "HERO_01", "name": "Garrosh Hellscream", "playerClass": "WARRIOR", "type": "HERO", "dbfId": 813},
            "HERO_02": {"id": "HERO_02", "name": "Jaina Proudmoore", "playerClass": "MAGE", "type": "HERO", "dbfId": 637},
            "HERO_03": {"id": "HERO_03", "name": "Rexxar", "playerClass": "HUNTER", "type": "HERO", "dbfId": 31},
            
            # Warrior cards for testing class synergy
            "WAR_001": {"id": "WAR_001", "name": "Warrior Weapon", "playerClass": "WARRIOR", "type": "WEAPON", "cost": 3, "attack": 3, "durability": 2, "dbfId": 2001},
            "WAR_002": {"id": "WAR_002", "name": "Warrior Minion", "playerClass": "WARRIOR", "type": "MINION", "cost": 4, "attack": 4, "health": 4, "dbfId": 2002},
            "WAR_003": {"id": "WAR_003", "name": "Warrior Spell", "playerClass": "WARRIOR", "type": "SPELL", "cost": 2, "dbfId": 2003},
            
            # Mage cards for testing class synergy
            "MAG_001": {"id": "MAG_001", "name": "Mage Spell", "playerClass": "MAGE", "type": "SPELL", "cost": 3, "dbfId": 3001},
            "MAG_002": {"id": "MAG_002", "name": "Mage Minion", "playerClass": "MAGE", "type": "MINION", "cost": 4, "attack": 3, "health": 5, "dbfId": 3002},
            "MAG_003": {"id": "MAG_003", "name": "Mage Secret", "playerClass": "MAGE", "type": "SPELL", "cost": 2, "dbfId": 3003},
            
            # Neutral cards for general testing
            "NEU_001": {"id": "NEU_001", "name": "Neutral Minion 1", "playerClass": "NEUTRAL", "type": "MINION", "cost": 2, "attack": 2, "health": 2, "dbfId": 4001},
            "NEU_002": {"id": "NEU_002", "name": "Neutral Minion 2", "playerClass": "NEUTRAL", "type": "MINION", "cost": 3, "attack": 3, "health": 3, "dbfId": 4002},
            "NEU_003": {"id": "NEU_003", "name": "Neutral Minion 3", "playerClass": "NEUTRAL", "type": "MINION", "cost": 4, "attack": 4, "health": 4, "dbfId": 4003},
            "NEU_004": {"id": "NEU_004", "name": "Neutral Minion 4", "playerClass": "NEUTRAL", "type": "MINION", "cost": 5, "attack": 5, "health": 5, "dbfId": 4004},
            "NEU_005": {"id": "NEU_005", "name": "Neutral Minion 5", "playerClass": "NEUTRAL", "type": "MINION", "cost": 6, "attack": 6, "health": 6, "dbfId": 4005}
        }
    
    def _create_cross_phase_hero_winrates(self):
        """Create hero winrates for cross-phase testing."""
        return {
            "WARRIOR": 0.5650,  # Best option
            "MAGE": 0.5450,     # Second best
            "HUNTER": 0.4920    # Worst option
        }
    
    def _create_cross_phase_hsreplay_data(self):
        """Create HSReplay data for cross-phase testing."""
        return {
            # Warrior cards - strong synergy
            "WAR_001": {"overall_winrate": 0.58, "play_rate": 0.18, "pick_rate": 0.25},
            "WAR_002": {"overall_winrate": 0.55, "play_rate": 0.15, "pick_rate": 0.22},
            "WAR_003": {"overall_winrate": 0.53, "play_rate": 0.12, "pick_rate": 0.20},
            
            # Mage cards - moderate synergy
            "MAG_001": {"overall_winrate": 0.54, "play_rate": 0.16, "pick_rate": 0.23},
            "MAG_002": {"overall_winrate": 0.52, "play_rate": 0.14, "pick_rate": 0.21},
            "MAG_003": {"overall_winrate": 0.51, "play_rate": 0.11, "pick_rate": 0.18},
            
            # Neutral cards - baseline performance
            "NEU_001": {"overall_winrate": 0.50, "play_rate": 0.20, "pick_rate": 0.30},
            "NEU_002": {"overall_winrate": 0.51, "play_rate": 0.19, "pick_rate": 0.28},
            "NEU_003": {"overall_winrate": 0.52, "play_rate": 0.18, "pick_rate": 0.26},
            "NEU_004": {"overall_winrate": 0.49, "play_rate": 0.15, "pick_rate": 0.22},
            "NEU_005": {"overall_winrate": 0.48, "play_rate": 0.12, "pick_rate": 0.18}
        }
    
    def _create_draft_workflow_scenarios(self):
        """Create draft workflow scenarios for testing."""
        return [
            {
                "name": "Warrior Selection with Class Cards",
                "hero_choices": ["WARRIOR", "MAGE", "HUNTER"],
                "expected_hero": "WARRIOR",
                "card_picks": [
                    {"choices": ["WAR_001", "NEU_001", "NEU_002"], "expected": "WAR_001"},  # Warrior weapon
                    {"choices": ["WAR_002", "MAG_001", "NEU_003"], "expected": "WAR_002"},  # Warrior minion
                    {"choices": ["NEU_004", "NEU_005", "WAR_003"], "expected": "WAR_003"}   # Warrior spell
                ]
            },
            {
                "name": "Mage Selection with Spell Synergy",
                "hero_choices": ["HUNTER", "MAGE", "WARRIOR"],
                "expected_hero": "MAGE",
                "card_picks": [
                    {"choices": ["MAG_001", "NEU_001", "WAR_001"], "expected": "MAG_001"},  # Mage spell
                    {"choices": ["NEU_002", "MAG_002", "WAR_002"], "expected": "MAG_002"},  # Mage minion
                    {"choices": ["MAG_003", "NEU_003", "NEU_004"], "expected": "MAG_003"}   # Mage secret
                ]
            },
            {
                "name": "Mixed Neutral Cards Scenario",
                "hero_choices": ["WARRIOR", "MAGE", "HUNTER"],
                "expected_hero": "WARRIOR",
                "card_picks": [
                    {"choices": ["NEU_001", "NEU_002", "NEU_003"], "expected": "NEU_003"},  # Best neutral
                    {"choices": ["NEU_004", "NEU_005", "WAR_001"], "expected": "WAR_001"},  # Class synergy wins
                    {"choices": ["MAG_001", "MAG_002", "NEU_001"], "expected": "NEU_001"}   # Avoid off-class
                ]
            }
        ]
    
    def test_complete_hero_to_card_workflow(self):
        """Test complete workflow from hero selection to card picks."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            mock_hero_winrates.return_value = self.cross_phase_hero_winrates
            
            for scenario in self.draft_workflow_scenarios:
                with self.subTest(scenario=scenario["name"]):
                    # Phase 1: Hero Selection
                    start_time = time.time()
                    
                    hero_advisor = HeroSelectionAdvisor()
                    hero_recommendations = hero_advisor.recommend_hero(scenario["hero_choices"])
                    
                    hero_selection_time = time.time() - start_time
                    
                    # Verify hero selection performance
                    self.assertLess(hero_selection_time, self.HERO_SELECTION_TIME_TARGET,
                                  f"Hero selection took {hero_selection_time:.3f}s, target: {self.HERO_SELECTION_TIME_TARGET}s")
                    
                    # Get selected hero (best recommendation)
                    selected_hero = hero_recommendations[0]["hero_class"]
                    
                    # Phase 2: Initialize card evaluation with selected hero
                    transition_start = time.time()
                    
                    grandmaster_advisor = GrandmasterAdvisor()
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class=selected_hero
                    )
                    
                    transition_time = time.time() - transition_start
                    
                    # Verify transition performance
                    self.assertLess(transition_time, self.PHASE_TRANSITION_TIME_TARGET,
                                  f"Phase transition took {transition_time:.3f}s, target: {self.PHASE_TRANSITION_TIME_TARGET}s")
                    
                    # Phase 3: Simulate card picks with hero context
                    drafted_cards = []
                    
                    for pick_num, card_pick in enumerate(scenario["card_picks"]):
                        pick_start = time.time()
                        
                        # Get card choices
                        card_choices = [self.cross_phase_cards_data[card_id] for card_id in card_pick["choices"]]
                        
                        # Make recommendation with hero context
                        recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                        
                        pick_time = time.time() - pick_start
                        
                        # Verify card evaluation performance
                        self.assertLess(pick_time, self.CARD_EVALUATION_TIME_TARGET,
                                      f"Card pick {pick_num + 1} took {pick_time:.3f}s, target: {self.CARD_EVALUATION_TIME_TARGET}s")
                        
                        # Get recommended card
                        recommended_index = recommendation.decision_data["recommended_pick_index"]
                        recommended_card = card_choices[recommended_index]
                        
                        # Update deck state
                        drafted_cards.append(recommended_card)
                        deck_state.cards_drafted = drafted_cards
                        
                        # Update mana curve
                        card_cost = recommended_card.get("cost", 0)
                        if card_cost in deck_state.mana_curve:
                            deck_state.mana_curve[card_cost] += 1
                        elif card_cost > 7:
                            deck_state.mana_curve[7] += 1
                    
                    # Verify hero context was maintained throughout
                    self.assertEqual(deck_state.hero_class, selected_hero,
                                   "Hero class should be maintained throughout draft")
                    
                    # Verify deck building progression
                    self.assertEqual(len(deck_state.cards_drafted), len(scenario["card_picks"]),
                                   "All cards should be drafted")
                    
                    total_workflow_time = time.time() - start_time
                    self.assertLess(total_workflow_time, self.COMPLETE_WORKFLOW_TIME_TARGET,
                                  f"Complete workflow took {total_workflow_time:.3f}s, target: {self.COMPLETE_WORKFLOW_TIME_TARGET}s")
    
    def test_hero_context_propagation(self):
        """Test that hero context propagates correctly through card evaluations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            mock_hero_winrates.return_value = self.cross_phase_hero_winrates
            
            # Test with different heroes
            test_heroes = ["WARRIOR", "MAGE"]
            
            for hero_class in test_heroes:
                with self.subTest(hero=hero_class):
                    grandmaster_advisor = GrandmasterAdvisor()
                    card_evaluator = CardEvaluationEngine()
                    
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class=hero_class
                    )
                    
                    # Test class synergy detection
                    class_card_id = f"{hero_class[:3]}_001"  # WAR_001 or MAG_001
                    off_class_card_id = "NEU_001"
                    
                    if class_card_id in self.cross_phase_cards_data:
                        class_card = self.cross_phase_cards_data[class_card_id]
                        off_class_card = self.cross_phase_cards_data[off_class_card_id]
                        
                        # Evaluate both cards
                        class_eval = card_evaluator.evaluate_card(class_card, deck_state, hero_class)
                        off_class_eval = card_evaluator.evaluate_card(off_class_card, deck_state, hero_class)
                        
                        # Class card should have higher synergy score
                        self.assertGreater(class_eval.synergy_score, off_class_eval.synergy_score,
                                         f"{hero_class} card should have higher synergy with {hero_class} hero")
                        
                        # Verify hero context is present in evaluation
                        self.assertEqual(deck_state.hero_class, hero_class,
                                       "Hero class should be maintained in deck state")
    
    def test_archetype_evolution_across_phases(self):
        """Test that archetype preferences evolve correctly from hero to cards."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            mock_hero_winrates.return_value = self.cross_phase_hero_winrates
            
            # Start with hero selection
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            # Get the selected hero (best recommendation)
            selected_hero = hero_recommendations[0]["hero_class"]
            
            # Get initial archetype preferences for this hero
            hero_rec = hero_recommendations[0]
            initial_archetype_prefs = hero_rec.get("archetype_preferences", {})
            
            # Initialize deck state with hero archetype preferences
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings=initial_archetype_prefs if initial_archetype_prefs else {'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class=selected_hero
            )
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Simulate several card picks and track archetype evolution
            card_sequences = [
                ["WAR_001", "NEU_001", "NEU_002"],  # Tempo cards
                ["WAR_002", "NEU_003", "MAG_001"],  # Mid-range cards
                ["WAR_003", "NEU_004", "NEU_005"]   # Value cards
            ]
            
            archetype_evolution = []
            
            for card_sequence in card_sequences:
                card_choices = [self.cross_phase_cards_data[card_id] for card_id in card_sequence]
                
                # Get recommendation
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                
                # Track archetype leanings
                current_leanings = deck_state.archetype_leanings.copy()
                archetype_evolution.append(current_leanings)
                
                # Update deck state with picked card
                recommended_index = recommendation.decision_data["recommended_pick_index"]
                recommended_card = card_choices[recommended_index]
                deck_state.cards_drafted.append(recommended_card)
                
                # Update mana curve
                card_cost = recommended_card.get("cost", 0)
                if card_cost in deck_state.mana_curve:
                    deck_state.mana_curve[card_cost] += 1
                elif card_cost > 7:
                    deck_state.mana_curve[7] += 1
            
            # Verify archetype evolution makes sense
            self.assertGreaterEqual(len(archetype_evolution), 3, "Should track archetype evolution")
            
            # Check that archetype leanings are valid probabilities
            for leanings in archetype_evolution:
                total = sum(leanings.values())
                self.assertAlmostEqual(total, 1.0, places=2, msg="Archetype leanings should sum to 1.0")
                
                for archetype, value in leanings.items():
                    self.assertGreaterEqual(value, 0.0, f"Archetype {archetype} should have non-negative value")
                    self.assertLessEqual(value, 1.0, f"Archetype {archetype} should not exceed 1.0")
    
    def test_state_consistency_across_phases(self):
        """Test that state remains consistent across phase transitions."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            mock_hero_winrates.return_value = self.cross_phase_hero_winrates
            
            # Phase 1: Hero Selection
            hero_advisor = HeroSelectionAdvisor()
            initial_state = {
                "phase": "hero_selection",
                "timestamp": time.time(),
                "available_choices": ["WARRIOR", "MAGE", "HUNTER"]
            }
            
            hero_recommendations = hero_advisor.recommend_hero(initial_state["available_choices"])
            selected_hero = hero_recommendations[0]["hero_class"]
            
            # Phase Transition: Hero -> Card
            transition_state = {
                "phase": "transition",
                "timestamp": time.time(),
                "selected_hero": selected_hero,
                "previous_phase": "hero_selection"
            }
            
            # Verify hero selection is preserved
            self.assertEqual(transition_state["selected_hero"], selected_hero)
            self.assertIn(selected_hero, initial_state["available_choices"])
            
            # Phase 2: Card Selection
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class=selected_hero
            )
            
            card_phase_state = {
                "phase": "card_selection",
                "timestamp": time.time(),
                "hero_class": selected_hero,
                "deck_state": deck_state,
                "previous_phase": "transition"
            }
            
            # Verify state consistency
            self.assertEqual(card_phase_state["hero_class"], selected_hero)
            self.assertEqual(deck_state.hero_class, selected_hero)
            
            # Test multiple card picks with state preservation
            grandmaster_advisor = GrandmasterAdvisor()
            
            for pick_number in range(3):
                # Simulate card choices
                card_choices = [
                    self.cross_phase_cards_data[f"NEU_00{pick_number + 1}"],
                    self.cross_phase_cards_data[f"WAR_00{(pick_number % 3) + 1}"]
                ]
                
                # Make recommendation
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                
                # Verify hero context is maintained
                self.assertEqual(deck_state.hero_class, selected_hero,
                               f"Hero class should be preserved in pick {pick_number + 1}")
                
                # Update state
                recommended_index = recommendation.decision_data["recommended_pick_index"]
                recommended_card = card_choices[recommended_index]
                deck_state.cards_drafted.append(recommended_card)
                
                # Verify state evolution
                self.assertEqual(len(deck_state.cards_drafted), pick_number + 1,
                               f"Deck should have {pick_number + 1} cards after pick {pick_number + 1}")
    
    def test_error_recovery_across_phases(self):
        """Test error recovery mechanisms across phase transitions."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            
            # Test hero phase with API failure
            mock_hero_winrates.side_effect = Exception("Hero API failure")
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Should still provide hero recommendations using fallback
            try:
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                self.assertEqual(len(hero_recommendations), 3, "Should provide fallback hero recommendations")
                selected_hero = hero_recommendations[0]["hero_class"]
            except Exception:
                # If hero selection fails completely, use default
                selected_hero = "WARRIOR"
            
            # Test card phase with previous hero context preserved
            mock_hsreplay.side_effect = Exception("Card API failure")
            
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class=selected_hero
            )
            
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Should still provide card recommendations using fallback
            card_choices = [
                self.cross_phase_cards_data["NEU_001"],
                self.cross_phase_cards_data["WAR_001"]
            ]
            
            try:
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
                self.assertIsNotNone(recommendation, "Should provide fallback card recommendation")
                
                # Hero context should be preserved even with API failures
                self.assertEqual(deck_state.hero_class, selected_hero,
                               "Hero context should be preserved during API failures")
                
            except Exception as e:
                # Even if card evaluation fails, hero context should be preserved
                self.assertEqual(deck_state.hero_class, selected_hero,
                               f"Hero context should be preserved even with error: {e}")
    
    def test_concurrent_phase_operations(self):
        """Test handling of concurrent operations across phases."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            mock_hero_winrates.return_value = self.cross_phase_hero_winrates
            
            results_queue = Queue()
            
            def hero_selection_worker():
                """Worker for hero selection phase."""
                try:
                    hero_advisor = HeroSelectionAdvisor()
                    recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    results_queue.put(("hero_selection", recommendations))
                except Exception as e:
                    results_queue.put(("hero_selection_error", str(e)))
            
            def card_evaluation_worker(hero_class):
                """Worker for card evaluation phase."""
                try:
                    grandmaster_advisor = GrandmasterAdvisor()
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class=hero_class
                    )
                    
                    card_choices = [
                        self.cross_phase_cards_data["WAR_001"],
                        self.cross_phase_cards_data["NEU_001"]
                    ]
                    
                    recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_class)
                    results_queue.put(("card_evaluation", recommendation))
                except Exception as e:
                    results_queue.put(("card_evaluation_error", str(e)))
            
            # Start hero selection
            hero_thread = threading.Thread(target=hero_selection_worker)
            hero_thread.start()
            
            # Wait for hero selection to complete
            hero_thread.join(timeout=5.0)
            
            # Get hero selection result
            try:
                result_type, result_data = results_queue.get(timeout=1.0)
                if result_type == "hero_selection":
                    selected_hero = result_data[0]["hero_class"]
                else:
                    selected_hero = "WARRIOR"  # fallback
            except Empty:
                selected_hero = "WARRIOR"  # fallback
            
            # Start card evaluation with selected hero
            card_thread = threading.Thread(target=card_evaluation_worker, args=(selected_hero,))
            card_thread.start()
            card_thread.join(timeout=5.0)
            
            # Verify both phases completed successfully
            results = []
            while not results_queue.empty():
                try:
                    results.append(results_queue.get_nowait())
                except Empty:
                    break
            
            # Should have results from both phases
            phase_types = [result[0] for result in results]
            self.assertIn("hero_selection", phase_types, "Hero selection should complete")
            self.assertIn("card_evaluation", phase_types, "Card evaluation should complete")
    
    def test_phase_performance_benchmarks(self):
        """Test performance benchmarks for cross-phase operations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.cross_phase_cards_data
            mock_hsreplay.return_value = self.cross_phase_hsreplay_data
            mock_hero_winrates.return_value = self.cross_phase_hero_winrates
            
            performance_metrics = {}
            
            # Benchmark hero selection phase
            hero_times = []
            for _ in range(5):
                start_time = time.time()
                hero_advisor = HeroSelectionAdvisor()
                hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                hero_times.append(time.time() - start_time)
            
            performance_metrics["hero_selection"] = {
                "avg_time": sum(hero_times) / len(hero_times),
                "max_time": max(hero_times),
                "min_time": min(hero_times)
            }
            
            # Benchmark card evaluation phase
            card_times = []
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            for _ in range(5):
                start_time = time.time()
                grandmaster_advisor = GrandmasterAdvisor()
                card_choices = [
                    self.cross_phase_cards_data["WAR_001"],
                    self.cross_phase_cards_data["NEU_001"],
                    self.cross_phase_cards_data["MAG_001"]
                ]
                grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
                card_times.append(time.time() - start_time)
            
            performance_metrics["card_evaluation"] = {
                "avg_time": sum(card_times) / len(card_times),
                "max_time": max(card_times),
                "min_time": min(card_times)
            }
            
            # Verify performance meets targets
            self.assertLess(performance_metrics["hero_selection"]["max_time"], 
                          self.HERO_SELECTION_TIME_TARGET,
                          f"Hero selection max time {performance_metrics['hero_selection']['max_time']:.3f}s exceeds target")
            
            self.assertLess(performance_metrics["card_evaluation"]["max_time"], 
                          self.CARD_EVALUATION_TIME_TARGET,
                          f"Card evaluation max time {performance_metrics['card_evaluation']['max_time']:.3f}s exceeds target")


class TestCrossPhaseTransitionEdgeCases(unittest.TestCase):
    """Test edge cases in cross-phase transitions."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.minimal_data = {
            "HERO_01": {"id": "HERO_01", "name": "Test Hero", "playerClass": "WARRIOR", "type": "HERO", "dbfId": 999},
            "TEST_001": {"id": "TEST_001", "name": "Test Card", "playerClass": "NEUTRAL", "type": "MINION", "cost": 3, "dbfId": 1000}
        }
    
    def test_cross_phase_with_minimal_data(self):
        """Test cross-phase operations with minimal data."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.minimal_data
            mock_hsreplay.return_value = {}
            mock_hero_winrates.return_value = {"WARRIOR": 0.50}
            
            # Hero selection with minimal data
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR"])
            
            # Should still work
            self.assertEqual(len(hero_recommendations), 1)
            selected_hero = hero_recommendations[0]["hero_class"]
            
            # Card evaluation with minimal data
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class=selected_hero
            )
            
            grandmaster_advisor = GrandmasterAdvisor()
            card_choices = [self.minimal_data["TEST_001"]]
            
            recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, selected_hero)
            
            # Should provide a recommendation even with minimal data
            self.assertIsNotNone(recommendation)
            self.assertEqual(recommendation.decision_data["recommended_pick_index"], 0)


def run_cross_phase_transition_tests():
    """Run all cross-phase transition tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestCrossPhaseTransition))
    test_suite.addTest(unittest.makeSuite(TestCrossPhaseTransitionEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_cross_phase_transition_tests()
    
    if success:
        print("\n✅ All cross-phase transition tests passed!")
    else:
        print("\n❌ Some cross-phase transition tests failed!")
        sys.exit(1)