"""
Statistical Accuracy Validation for Hero and Card Recommendations

Comprehensive statistical tests to validate the accuracy of both hero and card
recommendations against known meta performance and expected outcomes.
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


class TestStatisticalAccuracyValidation(unittest.TestCase):
    """Statistical accuracy validation for recommendations."""
    
    def setUp(self):
        """Set up statistical validation fixtures."""
        # Statistical validation thresholds
        self.HERO_RANKING_ACCURACY_THRESHOLD = 0.80  # 80% accuracy for hero ranking
        self.CARD_RANKING_ACCURACY_THRESHOLD = 0.75   # 75% accuracy for card ranking
        self.CONFIDENCE_CORRELATION_THRESHOLD = 0.60   # Confidence should correlate with accuracy
        self.WINRATE_PREDICTION_ERROR_THRESHOLD = 0.05 # 5% error threshold for winrate prediction
        
        # Create comprehensive test data with known statistical properties
        self.statistical_cards_data = self._create_statistical_cards_data()
        self.statistical_hero_winrates = self._create_statistical_hero_winrates()
        self.statistical_hsreplay_data = self._create_statistical_hsreplay_data()
        
        # Create test scenarios with known correct answers
        self.validation_scenarios = self._create_validation_scenarios()
    
    def _create_statistical_cards_data(self):
        """Create cards data with known statistical properties."""
        cards_data = {}
        
        # Create hero cards with distinct winrates
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
        
        # Create test cards with varying power levels
        card_templates = [
            # High-power cards
            ("STRONG_001", "Overpowered Card", "NEUTRAL", "MINION", 3, 5, 5, "very_strong"),
            ("STRONG_002", "Excellent Spell", "MAGE", "SPELL", 2, None, None, "very_strong"),
            ("STRONG_003", "Amazing Weapon", "WARRIOR", "WEAPON", 3, 3, 3, "very_strong"),
            
            # Medium-power cards
            ("MEDIUM_001", "Decent Minion", "NEUTRAL", "MINION", 4, 4, 4, "medium"),
            ("MEDIUM_002", "Fair Spell", "HUNTER", "SPELL", 3, None, None, "medium"),
            ("MEDIUM_003", "Average Card", "PALADIN", "MINION", 5, 5, 5, "medium"),
            
            # Low-power cards
            ("WEAK_001", "Terrible Card", "NEUTRAL", "MINION", 7, 1, 1, "very_weak"),
            ("WEAK_002", "Bad Spell", "PRIEST", "SPELL", 5, None, None, "very_weak"),
            ("WEAK_003", "Awful Minion", "WARLOCK", "MINION", 6, 2, 2, "very_weak"),
            
            # Situational cards
            ("SITUATIONAL_001", "Niche Card", "ROGUE", "SPELL", 2, None, None, "situational"),
            ("SITUATIONAL_002", "Tech Card", "SHAMAN", "MINION", 3, 2, 3, "situational"),
            ("SITUATIONAL_003", "Counter Card", "DRUID", "SPELL", 4, None, None, "situational")
        ]
        
        for card_id, name, player_class, card_type, cost, attack, health, power_level in card_templates:
            card_data = {
                "id": card_id,
                "name": name,
                "playerClass": player_class,
                "type": card_type,
                "cost": cost,
                "dbfId": hash(card_id) % 10000 + 3000,
                "power_level": power_level  # For testing purposes
            }
            
            if attack is not None:
                card_data["attack"] = attack
            if health is not None:
                card_data["health"] = health
                
            cards_data[card_id] = card_data
        
        return cards_data
    
    def _create_statistical_hero_winrates(self):
        """Create hero winrates with clear statistical differences."""
        # Winrates designed to create clear ranking
        return {
            "WARRIOR": 0.5650,    # Rank 1 (highest)
            "WARLOCK": 0.5580,    # Rank 2
            "MAGE": 0.5450,       # Rank 3
            "PALADIN": 0.5320,    # Rank 4
            "ROGUE": 0.5180,      # Rank 5
            "DEMONHUNTER": 0.5050, # Rank 6
            "HUNTER": 0.4920,     # Rank 7
            "DRUID": 0.4850,      # Rank 8
            "PRIEST": 0.4780,     # Rank 9
            "SHAMAN": 0.4650      # Rank 10 (lowest)
        }
    
    def _create_statistical_hsreplay_data(self):
        """Create HSReplay data that matches card power levels."""
        hsreplay_data = {}
        
        # Map power levels to winrates
        power_level_winrates = {
            "very_strong": 0.62,    # Clearly above average
            "medium": 0.50,         # Average
            "very_weak": 0.38,      # Clearly below average
            "situational": 0.48     # Slightly below average
        }
        
        for card_id, card_data in self.statistical_cards_data.items():
            if card_data["type"] != "HERO":
                power_level = card_data.get("power_level", "medium")
                base_winrate = power_level_winrates[power_level]
                
                hsreplay_data[card_id] = {
                    "overall_winrate": base_winrate,
                    "play_rate": 0.15 if power_level == "very_strong" else 0.08,
                    "pick_rate": 0.25 if power_level == "very_strong" else 0.12,
                    "power_level": power_level  # For validation
                }
        
        return hsreplay_data
    
    def _create_validation_scenarios(self):
        """Create test scenarios with known correct answers."""
        return [
            {
                "name": "Clear Hero Ranking",
                "hero_choices": ["WARRIOR", "SHAMAN", "PRIEST"],
                "expected_order": ["WARRIOR", "PRIEST", "SHAMAN"],  # Based on winrates
                "scenario_type": "hero_selection"
            },
            {
                "name": "Power Level Card Ranking",
                "card_choices": ["STRONG_001", "MEDIUM_001", "WEAK_001"],
                "expected_order": ["STRONG_001", "MEDIUM_001", "WEAK_001"],
                "hero_class": "WARRIOR",
                "scenario_type": "card_selection"
            },
            {
                "name": "Class Synergy Test",
                "card_choices": ["STRONG_002", "MEDIUM_002", "WEAK_002"],  # Mage, Hunter, Priest spells
                "expected_order_mage": ["STRONG_002", "MEDIUM_002", "WEAK_002"],  # Mage spell first
                "expected_order_hunter": ["MEDIUM_002", "STRONG_002", "WEAK_002"], # Hunter spell first
                "scenario_type": "class_synergy"
            },
            {
                "name": "Extreme Winrate Differences",
                "hero_choices": ["WARRIOR", "MAGE", "SHAMAN"],
                "expected_order": ["WARRIOR", "MAGE", "SHAMAN"],
                "scenario_type": "hero_selection"
            }
        ]
    
    def test_hero_ranking_statistical_accuracy(self):
        """Test statistical accuracy of hero ranking recommendations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hero_winrates.return_value = self.statistical_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            correct_predictions = 0
            total_predictions = 0
            
            # Test all hero selection scenarios
            hero_scenarios = [s for s in self.validation_scenarios if s["scenario_type"] == "hero_selection"]
            
            for scenario in hero_scenarios:
                hero_choices = scenario["hero_choices"]
                expected_order = scenario["expected_order"]
                
                recommendations = hero_advisor.recommend_hero(hero_choices)
                
                # Sort recommendations by rank to get actual order
                actual_order = [rec["hero_class"] for rec in sorted(recommendations, key=lambda x: x["rank"])]
                
                # Check if ranking matches expected order
                for i in range(len(expected_order)):
                    if i < len(actual_order) and actual_order[i] == expected_order[i]:
                        correct_predictions += 1
                    total_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            self.assertGreater(accuracy, self.HERO_RANKING_ACCURACY_THRESHOLD,
                             f"Hero ranking accuracy {accuracy:.3f} below threshold {self.HERO_RANKING_ACCURACY_THRESHOLD}")
            
            # Additional validation: Check that winrates correlate with rankings
            correlation_tests = []
            for scenario in hero_scenarios:
                hero_choices = scenario["hero_choices"]
                recommendations = hero_advisor.recommend_hero(hero_choices)
                
                # Get winrates and ranks
                winrates = [self.statistical_hero_winrates[hero] for hero in hero_choices]
                ranks = []
                for hero in hero_choices:
                    hero_rec = next(rec for rec in recommendations if rec["hero_class"] == hero)
                    ranks.append(hero_rec["rank"])
                
                # Higher winrate should mean lower rank (better ranking)
                correlation = self._calculate_spearman_correlation(winrates, [-r for r in ranks])
                correlation_tests.append(correlation)
            
            avg_correlation = statistics.mean(correlation_tests)
            self.assertGreater(avg_correlation, 0.7, 
                             f"Winrate-ranking correlation {avg_correlation:.3f} too low")
    
    def test_card_ranking_statistical_accuracy(self):
        """Test statistical accuracy of card ranking recommendations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hsreplay.return_value = self.statistical_hsreplay_data
            mock_hero_winrates.return_value = self.statistical_hero_winrates
            
            grandmaster_advisor = GrandmasterAdvisor()
            correct_predictions = 0
            total_predictions = 0
            
            # Test card selection scenarios
            card_scenarios = [s for s in self.validation_scenarios if s["scenario_type"] == "card_selection"]
            
            for scenario in card_scenarios:
                card_choices = [self.statistical_cards_data[card_id] for card_id in scenario["card_choices"]]
                expected_order = scenario["expected_order"]
                hero_class = scenario["hero_class"]
                
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class=hero_class
                )
                
                recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_class)
                
                # Evaluate all cards to get rankings
                card_evaluator = CardEvaluationEngine()
                evaluations = []
                for card in card_choices:
                    evaluation = card_evaluator.evaluate_card(card, deck_state, hero_class)
                    evaluations.append((card["id"], evaluation.total_score))
                
                # Sort by score to get actual ranking
                actual_order = [card_id for card_id, score in sorted(evaluations, key=lambda x: x[1], reverse=True)]
                
                # Check ranking accuracy
                for i in range(len(expected_order)):
                    if i < len(actual_order) and actual_order[i] == expected_order[i]:
                        correct_predictions += 1
                    total_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            self.assertGreater(accuracy, self.CARD_RANKING_ACCURACY_THRESHOLD,
                             f"Card ranking accuracy {accuracy:.3f} below threshold {self.CARD_RANKING_ACCURACY_THRESHOLD}")
    
    def test_class_synergy_statistical_validation(self):
        """Test statistical validation of class synergy effects."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hsreplay.return_value = self.statistical_hsreplay_data
            mock_hero_winrates.return_value = self.statistical_hero_winrates
            
            card_evaluator = CardEvaluationEngine()
            synergy_validations = []
            
            # Test class-specific synergy scenarios
            synergy_scenarios = [s for s in self.validation_scenarios if s["scenario_type"] == "class_synergy"]
            
            for scenario in synergy_scenarios:
                card_choices = [self.statistical_cards_data[card_id] for card_id in scenario["card_choices"]]
                
                # Test with Mage
                mage_deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.2, 'midrange': 0.3, 'control': 0.5},
                    synergy_groups={},
                    hero_class="MAGE"
                )
                
                # Test with Hunter
                hunter_deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.7, 'midrange': 0.3, 'control': 0.0},
                    synergy_groups={},
                    hero_class="HUNTER"
                )
                
                # Evaluate cards for both classes
                mage_evaluations = []
                hunter_evaluations = []
                
                for card in card_choices:
                    mage_eval = card_evaluator.evaluate_card(card, mage_deck_state, "MAGE")
                    hunter_eval = card_evaluator.evaluate_card(card, hunter_deck_state, "HUNTER")
                    
                    mage_evaluations.append((card["id"], mage_eval.total_score))
                    hunter_evaluations.append((card["id"], hunter_eval.total_score))
                
                # Mage spell should score higher for Mage than for Hunter
                mage_spell_id = "STRONG_002"  # Mage spell
                hunter_spell_id = "MEDIUM_002"  # Hunter spell
                
                mage_spell_mage_score = next(score for card_id, score in mage_evaluations if card_id == mage_spell_id)
                mage_spell_hunter_score = next(score for card_id, score in hunter_evaluations if card_id == mage_spell_id)
                
                hunter_spell_mage_score = next(score for card_id, score in mage_evaluations if card_id == hunter_spell_id)
                hunter_spell_hunter_score = next(score for card_id, score in hunter_evaluations if card_id == hunter_spell_id)
                
                # Class synergy validation
                mage_synergy_advantage = mage_spell_mage_score - mage_spell_hunter_score
                hunter_synergy_advantage = hunter_spell_hunter_score - hunter_spell_mage_score
                
                synergy_validations.append(mage_synergy_advantage > 0)  # Mage spell better for Mage
                synergy_validations.append(hunter_synergy_advantage > 0)  # Hunter spell better for Hunter
            
            # At least 70% of synergy tests should pass
            synergy_accuracy = sum(synergy_validations) / len(synergy_validations) if synergy_validations else 0
            self.assertGreater(synergy_accuracy, 0.70,
                             f"Class synergy accuracy {synergy_accuracy:.3f} below 70% threshold")
    
    def test_confidence_calibration_validation(self):
        """Test that confidence levels correlate with prediction accuracy."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hsreplay.return_value = self.statistical_hsreplay_data
            mock_hero_winrates.return_value = self.statistical_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            
            confidence_accuracy_pairs = []
            
            # Test hero recommendations confidence calibration
            for scenario in self.validation_scenarios:
                if scenario["scenario_type"] == "hero_selection":
                    hero_choices = scenario["hero_choices"]
                    expected_order = scenario["expected_order"]
                    
                    recommendations = hero_advisor.recommend_hero(hero_choices)
                    
                    for rec in recommendations:
                        confidence = rec.get("confidence", 0.5)
                        
                        # Check if this recommendation is correct
                        expected_rank = expected_order.index(rec["hero_class"]) + 1 if rec["hero_class"] in expected_order else 999
                        actual_rank = rec["rank"]
                        
                        is_correct = expected_rank == actual_rank
                        confidence_accuracy_pairs.append((confidence, 1.0 if is_correct else 0.0))
            
            # Test card recommendations confidence calibration
            for scenario in self.validation_scenarios:
                if scenario["scenario_type"] == "card_selection":
                    card_choices = [self.statistical_cards_data[card_id] for card_id in scenario["card_choices"]]
                    expected_order = scenario["expected_order"]
                    hero_class = scenario["hero_class"]
                    
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class=hero_class
                    )
                    
                    recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_class)
                    confidence = recommendation.decision_data.get("confidence_level", 0.5)
                    
                    # Check if recommendation is correct
                    recommended_index = recommendation.decision_data["recommended_pick_index"]
                    recommended_card_id = card_choices[recommended_index]["id"]
                    
                    is_correct = recommended_card_id == expected_order[0]  # Should recommend the best card
                    confidence_accuracy_pairs.append((confidence, 1.0 if is_correct else 0.0))
            
            # Calculate correlation between confidence and accuracy
            if len(confidence_accuracy_pairs) >= 3:
                confidences = [pair[0] for pair in confidence_accuracy_pairs]
                accuracies = [pair[1] for pair in confidence_accuracy_pairs]
                
                correlation = self._calculate_pearson_correlation(confidences, accuracies)
                
                self.assertGreater(correlation, self.CONFIDENCE_CORRELATION_THRESHOLD,
                                 f"Confidence-accuracy correlation {correlation:.3f} below threshold {self.CONFIDENCE_CORRELATION_THRESHOLD}")
    
    def test_winrate_prediction_accuracy(self):
        """Test accuracy of winrate predictions against known values."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hero_winrates.return_value = self.statistical_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            
            prediction_errors = []
            
            # Test winrate prediction accuracy
            for hero_class, true_winrate in self.statistical_hero_winrates.items():
                recommendations = hero_advisor.recommend_hero([hero_class])
                
                if recommendations:
                    predicted_winrate = recommendations[0].get("winrate", 0.5)
                    error = abs(predicted_winrate - true_winrate)
                    relative_error = error / true_winrate if true_winrate > 0 else error
                    
                    prediction_errors.append(relative_error)
            
            # Calculate average prediction error
            avg_error = statistics.mean(prediction_errors) if prediction_errors else 1.0
            
            self.assertLess(avg_error, self.WINRATE_PREDICTION_ERROR_THRESHOLD,
                          f"Average winrate prediction error {avg_error:.4f} above threshold {self.WINRATE_PREDICTION_ERROR_THRESHOLD}")
    
    def test_ranking_consistency_validation(self):
        """Test ranking consistency across similar scenarios."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hsreplay.return_value = self.statistical_hsreplay_data
            mock_hero_winrates.return_value = self.statistical_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Test ranking consistency with multiple runs
            consistency_tests = []
            
            test_combinations = [
                ["WARRIOR", "MAGE", "HUNTER"],
                ["PALADIN", "PRIEST", "ROGUE"],
                ["SHAMAN", "WARLOCK", "DRUID"]
            ]
            
            for hero_combination in test_combinations:
                rankings = []
                
                # Run the same test multiple times
                for _ in range(5):
                    recommendations = hero_advisor.recommend_hero(hero_combination)
                    ranking = [rec["hero_class"] for rec in sorted(recommendations, key=lambda x: x["rank"])]
                    rankings.append(ranking)
                
                # Check if all rankings are identical (deterministic)
                first_ranking = rankings[0]
                consistency = all(ranking == first_ranking for ranking in rankings)
                consistency_tests.append(consistency)
            
            # All rankings should be consistent
            consistency_rate = sum(consistency_tests) / len(consistency_tests) if consistency_tests else 0
            self.assertEqual(consistency_rate, 1.0, "Rankings should be deterministic and consistent")
    
    def test_edge_case_statistical_behavior(self):
        """Test statistical behavior in edge cases."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.statistical_cards_data
            mock_hsreplay.return_value = {}  # No HSReplay data
            mock_hero_winrates.return_value = {}  # No hero data
            
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Test with missing data - should still provide reasonable recommendations
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            # Should still return 3 recommendations
            self.assertEqual(len(hero_recommendations), 3)
            
            # Confidence should be lower without data
            for rec in hero_recommendations:
                confidence = rec.get("confidence", 1.0)
                self.assertLess(confidence, 0.8, "Confidence should be lower without statistical data")
            
            # Test card recommendations without data
            card_choices = [
                self.statistical_cards_data["STRONG_001"],
                self.statistical_cards_data["MEDIUM_001"],
                self.statistical_cards_data["WEAK_001"]
            ]
            
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            card_recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
            
            # Should still provide a recommendation
            self.assertIsNotNone(card_recommendation)
            self.assertIn("recommended_pick_index", card_recommendation.decision_data)
            
            # Confidence should be lower
            confidence = card_recommendation.decision_data.get("confidence_level", 1.0)
            self.assertLess(confidence, 0.8, "Card recommendation confidence should be lower without data")
    
    # === UTILITY METHODS ===
    
    def _calculate_pearson_correlation(self, x: List[float], y: List[float]) -> float:
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
    
    def _calculate_spearman_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Spearman rank correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # Convert to ranks
        def rank_data(data):
            sorted_data = sorted(enumerate(data), key=lambda x: x[1])
            ranks = [0] * len(data)
            for rank, (original_index, value) in enumerate(sorted_data):
                ranks[original_index] = rank + 1
            return ranks
        
        x_ranks = rank_data(x)
        y_ranks = rank_data(y)
        
        return self._calculate_pearson_correlation(x_ranks, y_ranks)


class TestStatisticalAccuracyValidationEdgeCases(unittest.TestCase):
    """Test edge cases in statistical accuracy validation."""
    
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
                "dbfId": 1000
            }
        }
    
    def test_statistical_validation_with_insufficient_data(self):
        """Test statistical validation when there's insufficient data."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.minimal_data
            mock_hero_winrates.return_value = {"WARRIOR": 0.50}
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Test with only one hero choice
            recommendations = hero_advisor.recommend_hero(["WARRIOR"])
            
            # Should still work with single choice
            self.assertEqual(len(recommendations), 1)
            self.assertEqual(recommendations[0]["hero_class"], "WARRIOR")
    
    def test_statistical_validation_with_identical_values(self):
        """Test statistical validation when all values are identical."""
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
            
            # Should still provide recommendations
            self.assertEqual(len(recommendations), 3)
            
            # Since winrates are identical, ranking should be consistent but not necessarily meaningful
            ranks = [rec["rank"] for rec in recommendations]
            self.assertEqual(sorted(ranks), [1, 2, 3])


def run_statistical_accuracy_validation_tests():
    """Run all statistical accuracy validation tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestStatisticalAccuracyValidation))
    test_suite.addTest(unittest.makeSuite(TestStatisticalAccuracyValidationEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_statistical_accuracy_validation_tests()
    
    if success:
        print("\n✅ All statistical accuracy validation tests passed!")
    else:
        print("\n❌ Some statistical accuracy validation tests failed!")
        sys.exit(1)