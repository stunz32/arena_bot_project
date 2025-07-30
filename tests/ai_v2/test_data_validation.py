"""
Data validation test suite for Phase 1 AI components.
Tests data integrity, validation, sanitization, and consistency.
"""

import unittest
import json
import copy
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.deck_analyzer import StrategicDeckAnalyzer
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState, AIDecision
from arena_bot.ai_v2.exceptions import AIEngineError, ValidationError


class DataValidationTests(unittest.TestCase):
    """
    Comprehensive data validation test suite.
    """
    
    def setUp(self):
        """Set up data validation testing environment."""
        self.card_evaluator = CardEvaluationEngine()
        self.deck_analyzer = StrategicDeckAnalyzer()
        self.grandmaster_advisor = GrandmasterAdvisor()
        
        # Valid test data
        self.valid_card = CardInstance(
            name="Test Card",
            cost=3,
            attack=3,
            health=4,
            card_type="minion",
            rarity="common",
            card_set="classic",
            keywords=["taunt"],
            description="A test card with taunt"
        )
        
        self.valid_deck_state = DeckState(
            drafted_cards=[self.valid_card],
            available_choices=[
                CardOption(self.valid_card, 0.8),
                CardOption(self._create_spell_card(), 0.7),
                CardOption(self._create_weapon_card(), 0.6)
            ],
            draft_pick_number=5,
            wins=1,
            losses=0
        )
    
    def _create_spell_card(self):
        """Create a valid spell card for testing."""
        return CardInstance(
            name="Test Spell",
            cost=2,
            attack=None,
            health=None,
            card_type="spell",
            rarity="common",
            card_set="classic",
            keywords=["spell"],
            description="A test spell"
        )
    
    def _create_weapon_card(self):
        """Create a valid weapon card for testing."""
        return CardInstance(
            name="Test Weapon",
            cost=3,
            attack=2,
            health=3,  # Durability for weapons
            card_type="weapon",
            rarity="rare",
            card_set="classic",
            keywords=["weapon"],
            description="A test weapon"
        )

    # CardInstance Validation Tests
    
    def test_card_instance_valid_data(self):
        """Test CardInstance with valid data."""
        card = self.valid_card
        
        # Should create successfully
        self.assertEqual(card.name, "Test Card")
        self.assertEqual(card.cost, 3)
        self.assertEqual(card.attack, 3)
        self.assertEqual(card.health, 4)
        self.assertEqual(card.card_type, "minion")
        self.assertEqual(card.rarity, "common")
        self.assertEqual(card.card_set, "classic")
        self.assertEqual(card.keywords, ["taunt"])
        self.assertEqual(card.description, "A test card with taunt")
    
    def test_card_instance_invalid_name(self):
        """Test CardInstance with invalid names."""
        invalid_names = [None, "", "   ", "A" * 1000]  # None, empty, whitespace, too long
        
        for invalid_name in invalid_names:
            with self.subTest(name=invalid_name):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    CardInstance(
                        name=invalid_name,
                        cost=3,
                        attack=3,
                        health=4,
                        card_type="minion",
                        rarity="common",
                        card_set="classic",
                        keywords=[],
                        description="Test"
                    )
    
    def test_card_instance_invalid_cost(self):
        """Test CardInstance with invalid costs."""
        invalid_costs = [-1, 100, None, "three", float('inf')]
        
        for invalid_cost in invalid_costs:
            with self.subTest(cost=invalid_cost):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    CardInstance(
                        name="Test Card",
                        cost=invalid_cost,
                        attack=3,
                        health=4,
                        card_type="minion",
                        rarity="common",
                        card_set="classic",
                        keywords=[],
                        description="Test"
                    )
    
    def test_card_instance_invalid_stats(self):
        """Test CardInstance with invalid attack/health."""
        invalid_stats = [-5, 100, "high", float('nan')]
        
        for invalid_stat in invalid_stats:
            with self.subTest(stat=invalid_stat):
                # Invalid attack
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    CardInstance(
                        name="Test Card",
                        cost=3,
                        attack=invalid_stat,
                        health=4,
                        card_type="minion",
                        rarity="common",
                        card_set="classic",
                        keywords=[],
                        description="Test"
                    )
                
                # Invalid health
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    CardInstance(
                        name="Test Card",
                        cost=3,
                        attack=3,
                        health=invalid_stat,
                        card_type="minion",
                        rarity="common",
                        card_set="classic",
                        keywords=[],
                        description="Test"
                    )
    
    def test_card_instance_invalid_card_type(self):
        """Test CardInstance with invalid card types."""
        invalid_types = [None, "", "invalid_type", 123, ["minion"]]
        
        for invalid_type in invalid_types:
            with self.subTest(card_type=invalid_type):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    CardInstance(
                        name="Test Card",
                        cost=3,
                        attack=3,
                        health=4,
                        card_type=invalid_type,
                        rarity="common",
                        card_set="classic",
                        keywords=[],
                        description="Test"
                    )
    
    def test_card_instance_spell_validation(self):
        """Test spell cards should not have attack/health (unless specified)."""
        # Valid spell with None stats
        valid_spell = CardInstance(
            name="Fireball",
            cost=4,
            attack=None,
            health=None,
            card_type="spell",
            rarity="common",
            card_set="classic",
            keywords=["spell", "damage"],
            description="Deal 6 damage"
        )
        
        self.assertEqual(valid_spell.card_type, "spell")
        self.assertIsNone(valid_spell.attack)
        self.assertIsNone(valid_spell.health)
    
    def test_card_instance_weapon_validation(self):
        """Test weapon cards should have attack and durability."""
        valid_weapon = self._create_weapon_card()
        
        self.assertEqual(valid_weapon.card_type, "weapon")
        self.assertIsNotNone(valid_weapon.attack)
        self.assertIsNotNone(valid_weapon.health)  # Durability
    
    def test_card_instance_keywords_validation(self):
        """Test keywords validation."""
        # Valid keywords
        valid_keywords = ["taunt", "charge", "divine_shield", "windfury"]
        card = CardInstance(
            name="Test Card",
            cost=3,
            attack=3,
            health=4,
            card_type="minion",
            rarity="common",
            card_set="classic",
            keywords=valid_keywords,
            description="Test"
        )
        self.assertEqual(card.keywords, valid_keywords)
        
        # Invalid keywords type
        invalid_keywords_types = [None, "taunt", {"taunt": True}, 123]
        
        for invalid_keywords in invalid_keywords_types:
            with self.subTest(keywords=invalid_keywords):
                with self.assertRaises((ValidationError, TypeError)):
                    CardInstance(
                        name="Test Card",
                        cost=3,
                        attack=3,
                        health=4,
                        card_type="minion",
                        rarity="common",
                        card_set="classic",
                        keywords=invalid_keywords,
                        description="Test"
                    )

    # CardOption Validation Tests
    
    def test_card_option_valid_data(self):
        """Test CardOption with valid data."""
        card_option = CardOption(self.valid_card, 0.8)
        
        self.assertEqual(card_option.card, self.valid_card)
        self.assertEqual(card_option.detection_confidence, 0.8)
    
    def test_card_option_invalid_confidence(self):
        """Test CardOption with invalid confidence values."""
        invalid_confidences = [-0.1, 1.1, None, "high", float('inf'), float('nan')]
        
        for invalid_confidence in invalid_confidences:
            with self.subTest(confidence=invalid_confidence):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    CardOption(self.valid_card, invalid_confidence)
    
    def test_card_option_invalid_card(self):
        """Test CardOption with invalid card."""
        invalid_cards = [None, "not a card", 123, {}]
        
        for invalid_card in invalid_cards:
            with self.subTest(card=invalid_card):
                with self.assertRaises((ValidationError, TypeError)):
                    CardOption(invalid_card, 0.8)

    # DeckState Validation Tests
    
    def test_deck_state_valid_data(self):
        """Test DeckState with valid data."""
        deck_state = self.valid_deck_state
        
        self.assertEqual(len(deck_state.drafted_cards), 1)
        self.assertEqual(len(deck_state.available_choices), 3)
        self.assertEqual(deck_state.draft_pick_number, 5)
        self.assertEqual(deck_state.wins, 1)
        self.assertEqual(deck_state.losses, 0)
    
    def test_deck_state_invalid_drafted_cards(self):
        """Test DeckState with invalid drafted cards."""
        invalid_drafted_cards = [None, "not a list", 123, [None], ["not a card"]]
        
        for invalid_cards in invalid_drafted_cards:
            with self.subTest(drafted_cards=invalid_cards):
                with self.assertRaises((ValidationError, TypeError)):
                    DeckState(
                        drafted_cards=invalid_cards,
                        available_choices=[],
                        draft_pick_number=1,
                        wins=0,
                        losses=0
                    )
    
    def test_deck_state_invalid_available_choices(self):
        """Test DeckState with invalid available choices."""
        invalid_choices = [None, "not a list", 123, [None], ["not a card option"]]
        
        for invalid_choice_list in invalid_choices:
            with self.subTest(available_choices=invalid_choice_list):
                with self.assertRaises((ValidationError, TypeError)):
                    DeckState(
                        drafted_cards=[],
                        available_choices=invalid_choice_list,
                        draft_pick_number=1,
                        wins=0,
                        losses=0
                    )
    
    def test_deck_state_invalid_pick_number(self):
        """Test DeckState with invalid draft pick numbers."""
        invalid_pick_numbers = [0, -1, 31, None, "five", float('inf')]
        
        for invalid_pick in invalid_pick_numbers:
            with self.subTest(draft_pick_number=invalid_pick):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    DeckState(
                        drafted_cards=[],
                        available_choices=[],
                        draft_pick_number=invalid_pick,
                        wins=0,
                        losses=0
                    )
    
    def test_deck_state_invalid_wins_losses(self):
        """Test DeckState with invalid wins/losses."""
        invalid_records = [-1, 13, None, "many", float('inf')]
        
        for invalid_record in invalid_records:
            with self.subTest(record=invalid_record):
                # Invalid wins
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    DeckState(
                        drafted_cards=[],
                        available_choices=[],
                        draft_pick_number=1,
                        wins=invalid_record,
                        losses=0
                    )
                
                # Invalid losses
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    DeckState(
                        drafted_cards=[],
                        available_choices=[],
                        draft_pick_number=1,
                        wins=0,
                        losses=invalid_record
                    )
    
    def test_deck_state_consistency_validation(self):
        """Test DeckState internal consistency validation."""
        # Deck should not have more than 30 cards
        too_many_cards = [self.valid_card] * 31
        with self.assertRaises(ValidationError):
            DeckState(
                drafted_cards=too_many_cards,
                available_choices=[],
                draft_pick_number=31,
                wins=0,
                losses=0
            )
        
        # Pick number should match cards + 1
        inconsistent_deck = DeckState(
            drafted_cards=[self.valid_card] * 5,  # 5 cards
            available_choices=[CardOption(self.valid_card, 0.8)],
            draft_pick_number=10,  # But pick 10? Should be 6
            wins=0,
            losses=0
        )
        # This might be allowed depending on implementation
        self.assertIsInstance(inconsistent_deck, DeckState)
        
        # Total games should not exceed maximum
        with self.assertRaises(ValidationError):
            DeckState(
                drafted_cards=[],
                available_choices=[],
                draft_pick_number=1,
                wins=15,  # Too many wins + losses
                losses=15
            )

    # AIDecision Validation Tests
    
    def test_ai_decision_valid_data(self):
        """Test AIDecision with valid data."""
        decision = AIDecision(
            recommended_card="Test Card",
            confidence_score=0.85,
            reasoning="This card provides good value",
            strategic_context={"archetype": "tempo"},
            card_evaluations={"Test Card": {"score": 8.5}},
            timestamp=datetime.now()
        )
        
        self.assertEqual(decision.recommended_card, "Test Card")
        self.assertEqual(decision.confidence_score, 0.85)
        self.assertIn("value", decision.reasoning)
        self.assertIsInstance(decision.strategic_context, dict)
        self.assertIsInstance(decision.card_evaluations, dict)
        self.assertIsInstance(decision.timestamp, datetime)
    
    def test_ai_decision_invalid_confidence(self):
        """Test AIDecision with invalid confidence scores."""
        invalid_confidences = [-0.1, 1.1, None, "high", float('inf'), float('nan')]
        
        for invalid_confidence in invalid_confidences:
            with self.subTest(confidence=invalid_confidence):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    AIDecision(
                        recommended_card="Test Card",
                        confidence_score=invalid_confidence,
                        reasoning="Test reasoning",
                        strategic_context={},
                        card_evaluations={},
                        timestamp=datetime.now()
                    )
    
    def test_ai_decision_invalid_recommended_card(self):
        """Test AIDecision with invalid recommended card."""
        invalid_cards = [None, "", "   ", 123, []]
        
        for invalid_card in invalid_cards:
            with self.subTest(recommended_card=invalid_card):
                with self.assertRaises((ValidationError, ValueError, TypeError)):
                    AIDecision(
                        recommended_card=invalid_card,
                        confidence_score=0.8,
                        reasoning="Test reasoning",
                        strategic_context={},
                        card_evaluations={},
                        timestamp=datetime.now()
                    )

    # Data Sanitization Tests
    
    def test_input_sanitization_card_names(self):
        """Test input sanitization for card names."""
        # Test with potentially problematic characters
        problematic_name = "Test'Card\"<script>alert('xss')</script>"
        
        try:
            card = CardInstance(
                name=problematic_name,
                cost=3,
                attack=3,
                health=4,
                card_type="minion",
                rarity="common",
                card_set="classic",
                keywords=[],
                description="Test card"
            )
            
            # Name should be sanitized or validation should reject it
            if hasattr(card, 'name'):
                # If accepted, should be sanitized
                self.assertNotIn('<script>', card.name)
                self.assertNotIn('alert', card.name)
        except ValidationError:
            # Acceptable to reject problematic input
            pass
    
    def test_input_sanitization_descriptions(self):
        """Test input sanitization for descriptions."""
        malicious_description = "<img src=x onerror=alert('xss')>Evil description"
        
        try:
            card = CardInstance(
                name="Test Card",
                cost=3,
                attack=3,
                health=4,
                card_type="minion",
                rarity="common",
                card_set="classic",
                keywords=[],
                description=malicious_description
            )
            
            # Description should be sanitized
            if hasattr(card, 'description'):
                self.assertNotIn('<img', card.description)
                self.assertNotIn('onerror', card.description)
                self.assertNotIn('alert', card.description)
        except ValidationError:
            # Acceptable to reject malicious input
            pass
    
    def test_numerical_sanitization(self):
        """Test sanitization of numerical inputs."""
        # Test with extreme values
        extreme_values = [
            float('inf'), float('-inf'), float('nan'),
            999999999, -999999999, 1e100
        ]
        
        for extreme_value in extreme_values:
            with self.subTest(value=extreme_value):
                try:
                    card = CardInstance(
                        name="Test Card",
                        cost=extreme_value if isinstance(extreme_value, int) else 3,
                        attack=extreme_value if not isinstance(extreme_value, int) else 3,
                        health=4,
                        card_type="minion",
                        rarity="common",
                        card_set="classic",
                        keywords=[],
                        description="Test"
                    )
                    
                    # If accepted, values should be sanitized to reasonable ranges
                    if hasattr(card, 'cost'):
                        self.assertGreaterEqual(card.cost, 0)
                        self.assertLessEqual(card.cost, 20)
                    if hasattr(card, 'attack') and card.attack is not None:
                        self.assertGreaterEqual(card.attack, 0)
                        self.assertLessEqual(card.attack, 50)
                        
                except (ValidationError, ValueError, TypeError):
                    # Acceptable to reject extreme values
                    pass

    # Data Consistency Tests
    
    def test_card_evaluation_consistency(self):
        """Test consistency of card evaluation results."""
        card = self.valid_card
        deck_state = self.valid_deck_state
        
        # Multiple evaluations should return identical results
        results = []
        for _ in range(5):
            result = self.card_evaluator.evaluate_card(card, deck_state)
            results.append(result)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result['overall_score'], first_result['overall_score'])
            self.assertEqual(result['base_value'], first_result['base_value'])
    
    def test_deck_analysis_consistency(self):
        """Test consistency of deck analysis results."""
        deck_state = self.valid_deck_state
        
        # Multiple analyses should return consistent results
        results = []
        for _ in range(3):
            result = self.deck_analyzer.analyze_deck(deck_state)
            results.append(result)
        
        # Primary archetype should be consistent
        first_archetype = results[0]['primary_archetype']
        for result in results[1:]:
            self.assertEqual(result['primary_archetype'], first_archetype)
    
    def test_ai_decision_consistency(self):
        """Test consistency of AI decision making."""
        deck_state = self.valid_deck_state
        
        # Multiple analyses should return consistent recommendations
        decisions = []
        for _ in range(3):
            decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
            decisions.append(decision)
        
        # Recommended card should be consistent
        first_recommendation = decisions[0].recommended_card
        for decision in decisions[1:]:
            self.assertEqual(decision.recommended_card, first_recommendation)

    # Data Integrity Tests
    
    def test_data_structure_immutability(self):
        """Test that data structures maintain immutability where required."""
        original_deck = copy.deepcopy(self.valid_deck_state)
        
        # Analyze deck (should not modify original)
        result = self.deck_analyzer.analyze_deck(original_deck)
        
        # Original deck should be unchanged
        self.assertEqual(len(original_deck.drafted_cards), 1)
        self.assertEqual(original_deck.draft_pick_number, 5)
        self.assertEqual(original_deck.wins, 1)
        self.assertEqual(original_deck.losses, 0)
    
    def test_concurrent_data_integrity(self):
        """Test data integrity under concurrent access."""
        import threading
        
        results = []
        errors = []
        
        def analyze_concurrently():
            try:
                # Create independent copy for each thread
                deck_copy = copy.deepcopy(self.valid_deck_state)
                result = self.grandmaster_advisor.analyze_draft_choice(deck_copy)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent analyses
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=analyze_concurrently)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # No errors and consistent results
        self.assertEqual(len(errors), 0, f"Concurrent data integrity errors: {errors}")
        self.assertEqual(len(results), 5)
        
        # All results should be valid AIDecision objects
        for result in results:
            self.assertIsInstance(result, AIDecision)
            self.assertIsInstance(result.recommended_card, str)
            self.assertIsInstance(result.confidence_score, (int, float))

    # Serialization and Deserialization Tests
    
    def test_card_instance_serialization(self):
        """Test CardInstance serialization/deserialization."""
        card = self.valid_card
        
        # Convert to dict
        card_dict = {
            'name': card.name,
            'cost': card.cost,
            'attack': card.attack,
            'health': card.health,
            'card_type': card.card_type,
            'rarity': card.rarity,
            'card_set': card.card_set,
            'keywords': card.keywords,
            'description': card.description
        }
        
        # Should be serializable to JSON
        json_str = json.dumps(card_dict)
        self.assertIsInstance(json_str, str)
        
        # Should be deserializable
        deserialized_dict = json.loads(json_str)
        self.assertEqual(deserialized_dict['name'], card.name)
        self.assertEqual(deserialized_dict['cost'], card.cost)
    
    def test_ai_decision_serialization(self):
        """Test AIDecision serialization for audit trails."""
        decision = AIDecision(
            recommended_card="Test Card",
            confidence_score=0.85,
            reasoning="Test reasoning",
            strategic_context={"archetype": "tempo"},
            card_evaluations={"Test Card": {"score": 8.5}},
            timestamp=datetime.now()
        )
        
        # Should be serializable (with timestamp handling)
        decision_dict = {
            'recommended_card': decision.recommended_card,
            'confidence_score': decision.confidence_score,
            'reasoning': decision.reasoning,
            'strategic_context': decision.strategic_context,
            'card_evaluations': decision.card_evaluations,
            'timestamp': decision.timestamp.isoformat()
        }
        
        json_str = json.dumps(decision_dict)
        self.assertIsInstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized['recommended_card'], decision.recommended_card)
        self.assertEqual(deserialized['confidence_score'], decision.confidence_score)

    # Edge Case Data Tests
    
    def test_edge_case_empty_data(self):
        """Test handling of edge case empty data."""
        # Empty deck state
        empty_deck = DeckState(
            drafted_cards=[],
            available_choices=[CardOption(self.valid_card, 0.8)],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        # Should handle gracefully
        result = self.grandmaster_advisor.analyze_draft_choice(empty_deck)
        self.assertIsInstance(result, AIDecision)
    
    def test_edge_case_maximum_data(self):
        """Test handling of maximum allowable data."""
        # Maximum deck size
        max_cards = [self.valid_card] * 29
        max_deck = DeckState(
            drafted_cards=max_cards,
            available_choices=[CardOption(self.valid_card, 0.8)],
            draft_pick_number=30,
            wins=11,
            losses=2
        )
        
        # Should handle large deck gracefully
        result = self.grandmaster_advisor.analyze_draft_choice(max_deck)
        self.assertIsInstance(result, AIDecision)
    
    def test_edge_case_unusual_card_combinations(self):
        """Test handling of unusual card combinations."""
        # Zero cost card
        zero_cost_card = CardInstance(
            name="Free Card",
            cost=0,
            attack=1,
            health=1,
            card_type="minion",
            rarity="common",
            card_set="test",
            keywords=[],
            description="Costs nothing"
        )
        
        # High cost card
        expensive_card = CardInstance(
            name="Expensive Card",
            cost=12,
            attack=12,
            health=12,
            card_type="minion",
            rarity="legendary",
            card_set="test",
            keywords=[],
            description="Very expensive"
        )
        
        unusual_deck = DeckState(
            drafted_cards=[zero_cost_card, expensive_card],
            available_choices=[CardOption(self.valid_card, 0.8)],
            draft_pick_number=3,
            wins=0,
            losses=0
        )
        
        # Should handle unusual combinations
        result = self.grandmaster_advisor.analyze_draft_choice(unusual_deck)
        self.assertIsInstance(result, AIDecision)


class DataIntegrityTests(unittest.TestCase):
    """
    Tests for data integrity across component boundaries.
    """
    
    def test_data_flow_integrity(self):
        """Test data integrity as it flows between components."""
        # Create test data
        card = CardInstance(
            name="Flow Test Card",
            cost=4,
            attack=4,
            health=5,
            card_type="minion",
            rarity="rare",
            card_set="test",
            keywords=["taunt"],
            description="Testing data flow"
        )
        
        deck_state = DeckState(
            drafted_cards=[card],
            available_choices=[CardOption(card, 0.9)],
            draft_pick_number=2,
            wins=0,
            losses=0
        )
        
        # Flow through all components
        advisor = GrandmasterAdvisor()
        decision = advisor.analyze_draft_choice(deck_state)
        
        # Verify data integrity maintained
        self.assertEqual(card.name, "Flow Test Card")
        self.assertEqual(card.cost, 4)
        self.assertEqual(deck_state.draft_pick_number, 2)
        self.assertIsInstance(decision, AIDecision)
        self.assertIn(card.name, decision.card_evaluations)


if __name__ == '__main__':
    print("Starting Data Validation Test Suite...")
    print("=" * 60)
    
    # Run with high verbosity to see all validation results
    unittest.main(verbosity=2)