"""
Integration Tests for Hero-Aware Card Recommendation Pipeline

Comprehensive integration tests for the hero-aware card recommendation system,
testing the complete pipeline from card evaluation to final recommendations.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import DimensionalScores, DeckState, AIDecision
from arena_bot.data.cards_json_loader import CardsJsonLoader


class TestHeroAwareCardRecommendations(unittest.TestCase):
    """Integration tests for hero-aware card recommendation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock cards data
        self.mock_cards_data = {
            "CS2_023": {
                "id": "CS2_023",
                "name": "Arcane Intellect",
                "playerClass": "MAGE",
                "type": "SPELL",
                "cost": 3,
                "dbfId": 489
            },
            "CS2_102": {
                "id": "CS2_102", 
                "name": "Heroic Strike",
                "playerClass": "WARRIOR",
                "type": "SPELL",
                "cost": 2,
                "dbfId": 1003
            },
            "CS2_061": {
                "id": "CS2_061",
                "name": "Drain Life",
                "playerClass": "WARLOCK", 
                "type": "SPELL",
                "cost": 1,
                "dbfId": 1009
            },
            "CS2_025": {
                "id": "CS2_025",
                "name": "Arcane Shot",
                "playerClass": "HUNTER",
                "type": "SPELL",
                "cost": 1,
                "dbfId": 25
            },
            "CS2_089": {
                "id": "CS2_089",
                "name": "Holy Light",
                "playerClass": "PALADIN",
                "type": "SPELL", 
                "cost": 2,
                "dbfId": 1060
            },
            "EX1_001": {
                "id": "EX1_001",
                "name": "Lightwarden",
                "playerClass": "PRIEST",
                "type": "MINION",
                "cost": 1,
                "attack": 1,
                "health": 2,
                "dbfId": 1659
            },
            "EX1_002": {
                "id": "EX1_002",
                "name": "The Black Knight",
                "playerClass": "NEUTRAL",
                "type": "MINION",
                "cost": 6,
                "attack": 4,
                "health": 5,
                "dbfId": 396
            }
        }
        
        # Mock HSReplay card data with hero-specific performance
        self.mock_hsreplay_data = {
            "CS2_023": {  # Arcane Intellect
                "overall_winrate": 0.54,
                "hero_specific": {
                    "MAGE": 0.57,
                    "WARRIOR": 0.45,  # Poor in Warrior
                    "HUNTER": 0.43
                }
            },
            "CS2_102": {  # Heroic Strike
                "overall_winrate": 0.52,
                "hero_specific": {
                    "WARRIOR": 0.58,  # Excellent in Warrior
                    "MAGE": 0.46,
                    "HUNTER": 0.48
                }
            },
            "CS2_061": {  # Drain Life
                "overall_winrate": 0.49,
                "hero_specific": {
                    "WARLOCK": 0.55,
                    "WARRIOR": 0.44,
                    "MAGE": 0.45
                }
            },
            "EX1_002": {  # The Black Knight
                "overall_winrate": 0.58,
                "hero_specific": {
                    "WARRIOR": 0.62,  # Strong in control heroes
                    "MAGE": 0.59,
                    "HUNTER": 0.53   # Weaker in aggro heroes
                }
            }
        }
        
        # Mock current deck state
        self.mock_deck_state = DeckState(
            cards_drafted=[],
            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
            archetype_leanings={'aggro': 0.3, 'midrange': 0.4, 'control': 0.3},
            synergy_groups={},
            hero_class="WARRIOR"
        )
    
    def test_hero_aware_card_evaluation_pipeline(self):
        """Test complete hero-aware card evaluation pipeline."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            
            # Initialize components
            card_evaluator = CardEvaluationEngine()
            
            # Test cards for Warrior hero
            test_cards = ["CS2_023", "CS2_102", "CS2_061"]  # Arcane Intellect, Heroic Strike, Drain Life
            hero_class = "WARRIOR"
            
            evaluations = []
            for card_id in test_cards:
                card_data = self.mock_cards_data[card_id]
                evaluation = card_evaluator.evaluate_card(card_data, self.mock_deck_state, hero_class)
                evaluations.append((card_id, evaluation))
            
            # Verify evaluations
            self.assertEqual(len(evaluations), 3)
            
            # Extract scores for comparison
            scores = {}
            for card_id, evaluation in evaluations:
                scores[card_id] = evaluation.total_score
            
            # Heroic Strike should score highest for Warrior
            self.assertGreater(scores["CS2_102"], scores["CS2_023"], 
                             "Heroic Strike should score higher than Arcane Intellect for Warrior")
            self.assertGreater(scores["CS2_102"], scores["CS2_061"],
                             "Heroic Strike should score higher than Drain Life for Warrior")
    
    def test_hero_specific_synergy_evaluation(self):
        """Test hero-specific synergy evaluation in card recommendations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            
            card_evaluator = CardEvaluationEngine()
            
            # Test with Mage-specific deck state
            mage_deck_state = DeckState(
                cards_drafted=[{"id": "CS2_023", "name": "Arcane Intellect"}],  # Already has spell synergy
                mana_curve={1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.2, 'midrange': 0.3, 'control': 0.5},
                synergy_groups={'spells': 1},
                hero_class="MAGE"
            )
            
            # Evaluate another spell card for Mage
            arcane_shot = self.mock_cards_data["CS2_025"]
            mage_evaluation = card_evaluator.evaluate_card(arcane_shot, mage_deck_state, "MAGE")
            
            # Test with Warrior deck state
            warrior_deck_state = DeckState(
                cards_drafted=[{"id": "CS2_102", "name": "Heroic Strike"}],
                mana_curve={1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.7, 'midrange': 0.3, 'control': 0.0},
                synergy_groups={'weapons': 1},
                hero_class="WARRIOR"
            )
            
            # Evaluate the same spell card for Warrior
            warrior_evaluation = card_evaluator.evaluate_card(arcane_shot, warrior_deck_state, "WARRIOR")
            
            # Verify hero-specific differences
            self.assertIsInstance(mage_evaluation, DimensionalScores)
            self.assertIsInstance(warrior_evaluation, DimensionalScores)
            
            # Mage should have different synergy score than Warrior for spells
            self.assertNotEqual(mage_evaluation.synergy_score, warrior_evaluation.synergy_score)
    
    def test_complete_recommendation_pipeline_with_heroes(self):
        """Test complete recommendation pipeline with different heroes."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            mock_hero_winrates.return_value = {"WARRIOR": 0.55, "MAGE": 0.52}
            
            # Initialize Grandmaster Advisor
            advisor = GrandmasterAdvisor()
            
            # Test recommendation for Warrior
            warrior_choices = [
                self.mock_cards_data["CS2_023"],  # Arcane Intellect
                self.mock_cards_data["CS2_102"],  # Heroic Strike
                self.mock_cards_data["CS2_061"]   # Drain Life
            ]
            
            warrior_recommendation = advisor.get_recommendation(
                warrior_choices, self.mock_deck_state, "WARRIOR"
            )
            
            # Verify recommendation structure
            self.assertIsInstance(warrior_recommendation, AIDecision)
            self.assertIn('recommended_pick_index', warrior_recommendation.decision_data)
            self.assertIn('confidence_level', warrior_recommendation.decision_data)
            self.assertIn('explanations', warrior_recommendation.decision_data)
            
            # Heroic Strike should be recommended for Warrior
            recommended_index = warrior_recommendation.decision_data['recommended_pick_index']
            recommended_card = warrior_choices[recommended_index]
            self.assertEqual(recommended_card['id'], "CS2_102")  # Heroic Strike
            
            # Test recommendation for Mage with same choices
            mage_deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.2, 'midrange': 0.4, 'control': 0.4},
                synergy_groups={},
                hero_class="MAGE"
            )
            
            mage_recommendation = advisor.get_recommendation(
                warrior_choices, mage_deck_state, "MAGE"
            )
            
            # Arcane Intellect should be recommended for Mage
            mage_recommended_index = mage_recommendation.decision_data['recommended_pick_index']
            mage_recommended_card = warrior_choices[mage_recommended_index]
            self.assertEqual(mage_recommended_card['id'], "CS2_023")  # Arcane Intellect
            
            # Verify different recommendations for different heroes
            self.assertNotEqual(recommended_index, mage_recommended_index,
                              "Different heroes should get different recommendations for same card choices")
    
    def test_archetype_aware_recommendations_by_hero(self):
        """Test that recommendations adapt to hero-specific archetype preferences."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            mock_hero_winrates.return_value = {"HUNTER": 0.49, "PRIEST": 0.50}
            
            advisor = GrandmasterAdvisor()
            
            # Test with aggressive Hunter deck
            hunter_aggro_deck = DeckState(
                cards_drafted=[],
                mana_curve={1: 2, 2: 3, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.8, 'midrange': 0.2, 'control': 0.0},
                synergy_groups={},
                hero_class="HUNTER"
            )
            
            # Test with control Priest deck
            priest_control_deck = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1},
                archetype_leanings={'aggro': 0.1, 'midrange': 0.3, 'control': 0.6},
                synergy_groups={},
                hero_class="PRIEST"
            )
            
            # Choice between aggressive and control cards
            choices = [
                self.mock_cards_data["CS2_025"],  # Arcane Shot (aggressive)
                self.mock_cards_data["EX1_001"],  # Lightwarden (midrange)
                self.mock_cards_data["EX1_002"]   # The Black Knight (control)
            ]
            
            # Get recommendations for both heroes
            hunter_rec = advisor.get_recommendation(choices, hunter_aggro_deck, "HUNTER")
            priest_rec = advisor.get_recommendation(choices, priest_control_deck, "PRIEST")
            
            # Hunter should prefer aggressive options
            hunter_choice = choices[hunter_rec.decision_data['recommended_pick_index']]
            # Priest should prefer control options
            priest_choice = choices[priest_rec.decision_data['recommended_pick_index']]
            
            # Verify archetype-appropriate recommendations
            self.assertNotEqual(hunter_choice['id'], priest_choice['id'],
                              "Aggressive and control decks should get different recommendations")
    
    def test_mana_curve_optimization_by_hero(self):
        """Test mana curve optimization considering hero-specific preferences."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            mock_hero_winrates.return_value = {"WARRIOR": 0.55}
            
            card_evaluator = CardEvaluationEngine()
            
            # Test deck with too many 2-drops
            heavy_2_drop_deck = DeckState(
                cards_drafted=[],
                mana_curve={1: 1, 2: 5, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},  # Heavy on 2-drops
                archetype_leanings={'aggro': 0.6, 'midrange': 0.4, 'control': 0.0},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            # Choice between different mana costs
            low_cost_card = self.mock_cards_data["CS2_061"]  # 1 mana
            medium_cost_card = self.mock_cards_data["CS2_102"]  # 2 mana
            high_cost_card = self.mock_cards_data["CS2_023"]  # 3 mana
            
            # Evaluate each card
            low_eval = card_evaluator.evaluate_card(low_cost_card, heavy_2_drop_deck, "WARRIOR")
            medium_eval = card_evaluator.evaluate_card(medium_cost_card, heavy_2_drop_deck, "WARRIOR")
            high_eval = card_evaluator.evaluate_card(high_cost_card, heavy_2_drop_deck, "WARRIOR")
            
            # Curve balancing should favor non-2-drop cards
            # Should prefer 1 or 3 mana over adding another 2 mana
            self.assertTrue(
                low_eval.curve_score > medium_eval.curve_score or 
                high_eval.curve_score > medium_eval.curve_score,
                "Curve optimization should favor balancing over adding more 2-drops"
            )
    
    def test_hero_specific_fallback_recommendations(self):
        """Test fallback recommendations when HSReplay data is unavailable."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = {}  # No HSReplay data
            mock_hero_winrates.return_value = {}  # No hero data
            
            advisor = GrandmasterAdvisor()
            
            # Test fallback recommendations
            choices = [
                self.mock_cards_data["CS2_023"],  # Arcane Intellect
                self.mock_cards_data["CS2_102"],  # Heroic Strike
                self.mock_cards_data["CS2_061"]   # Drain Life
            ]
            
            fallback_recommendation = advisor.get_recommendation(
                choices, self.mock_deck_state, "WARRIOR"
            )
            
            # Should still provide valid recommendation
            self.assertIsInstance(fallback_recommendation, AIDecision)
            self.assertIn('recommended_pick_index', fallback_recommendation.decision_data)
            self.assertIn('confidence_level', fallback_recommendation.decision_data)
            
            # Confidence should be lower without data
            confidence = fallback_recommendation.decision_data['confidence_level']
            self.assertLess(confidence, 0.8, "Confidence should be lower without HSReplay data")
    
    def test_recommendation_explanation_quality(self):
        """Test quality and completeness of recommendation explanations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            mock_hero_winrates.return_value = {"WARRIOR": 0.55}
            
            advisor = GrandmasterAdvisor()
            
            choices = [
                self.mock_cards_data["CS2_023"],  # Arcane Intellect
                self.mock_cards_data["CS2_102"],  # Heroic Strike
                self.mock_cards_data["CS2_061"]   # Drain Life
            ]
            
            recommendation = advisor.get_recommendation(
                choices, self.mock_deck_state, "WARRIOR"
            )
            
            explanations = recommendation.decision_data['explanations']
            
            # Should have explanations for all choices
            self.assertEqual(len(explanations), 3)
            
            # Each explanation should mention hero-specific considerations
            for explanation in explanations:
                self.assertIsInstance(explanation, str)
                self.assertGreater(len(explanation), 20, "Explanation should be detailed")
                
                # Should mention hero or class-specific information
                self.assertTrue(
                    'WARRIOR' in explanation.upper() or 
                    'warrior' in explanation.lower() or
                    'hero' in explanation.lower(),
                    "Explanation should mention hero-specific considerations"
                )
    
    def test_recommendation_consistency_across_sessions(self):
        """Test recommendation consistency across multiple sessions."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_data
            mock_hero_winrates.return_value = {"WARRIOR": 0.55}
            
            # Test multiple advisor instances
            recommendations = []
            for i in range(5):
                advisor = GrandmasterAdvisor()
                
                choices = [
                    self.mock_cards_data["CS2_023"],
                    self.mock_cards_data["CS2_102"],
                    self.mock_cards_data["CS2_061"]
                ]
                
                rec = advisor.get_recommendation(choices, self.mock_deck_state, "WARRIOR")
                recommendations.append(rec.decision_data['recommended_pick_index'])
            
            # All recommendations should be identical (deterministic)
            first_rec = recommendations[0]
            for rec in recommendations[1:]:
                self.assertEqual(rec, first_rec, "Recommendations should be consistent across sessions")


class TestHeroAwareCardRecommendationEdgeCases(unittest.TestCase):
    """Test edge cases in hero-aware card recommendations."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.minimal_cards_data = {
            "TEST_001": {
                "id": "TEST_001",
                "name": "Test Card",
                "playerClass": "NEUTRAL",
                "type": "MINION",
                "cost": 3,
                "dbfId": 9999
            }
        }
    
    def test_recommendation_with_unknown_hero(self):
        """Test recommendations with unknown hero class."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay:
            
            mock_load_cards.return_value = self.minimal_cards_data
            mock_hsreplay.return_value = {}
            
            card_evaluator = CardEvaluationEngine()
            
            # Test with unknown hero
            unknown_deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="UNKNOWN_HERO"
            )
            
            test_card = self.minimal_cards_data["TEST_001"]
            
            # Should handle unknown hero gracefully
            try:
                evaluation = card_evaluator.evaluate_card(test_card, unknown_deck_state, "UNKNOWN_HERO")
                self.assertIsInstance(evaluation, DimensionalScores)
            except Exception as e:
                self.fail(f"Should handle unknown hero gracefully, but got: {e}")
    
    def test_recommendation_with_empty_choices(self):
        """Test recommendations with empty card choices."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.minimal_cards_data
            mock_hsreplay.return_value = {}
            mock_hero_winrates.return_value = {"WARRIOR": 0.55}
            
            advisor = GrandmasterAdvisor()
            
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            # Test with empty choices
            try:
                recommendation = advisor.get_recommendation([], deck_state, "WARRIOR")
                # Should handle gracefully or return appropriate error
                if recommendation:
                    self.assertIsInstance(recommendation, AIDecision)
            except Exception as e:
                # Should raise specific exception, not generic error
                self.assertIn("choices", str(e).lower())


def run_hero_aware_card_recommendation_tests():
    """Run all hero-aware card recommendation tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestHeroAwareCardRecommendations))
    test_suite.addTest(unittest.makeSuite(TestHeroAwareCardRecommendationEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_hero_aware_card_recommendation_tests()
    
    if success:
        print("\n✅ All hero-aware card recommendation tests passed!")
    else:
        print("\n❌ Some hero-aware card recommendation tests failed!")
        sys.exit(1)