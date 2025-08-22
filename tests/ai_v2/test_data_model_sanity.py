"""
Sanity tests for AI v2 data models and exceptions

Tests basic functionality of enums, exceptions, and data model structure
to ensure imports work correctly and expected behavior is maintained.
"""

import unittest
import sys
import os

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.errors import (
    AIEngineError, ModelLoadingError, InferenceTimeout, InvalidInputError,
    ValidationError, AnalysisError, ConfidenceError
)

from arena_bot.ai_v2.data_models import (
    ConversationTone, UserSkillLevel, ConfidenceLevel,
    CardInstance, DraftChoice
)


class TestAIv2Exceptions(unittest.TestCase):
    """Test AI v2 exception classes"""
    
    def test_exception_imports(self):
        """Test that all exception classes are importable"""
        # Should not raise any import errors
        exceptions = [
            AIEngineError, ModelLoadingError, InferenceTimeout, 
            InvalidInputError, ValidationError, AnalysisError, ConfidenceError
        ]
        
        for exc_class in exceptions:
            self.assertTrue(issubclass(exc_class, Exception))
            self.assertTrue(issubclass(exc_class, AIEngineError))
    
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy"""
        # All should inherit from AIEngineError
        derived_exceptions = [
            ModelLoadingError, InferenceTimeout, InvalidInputError,
            ValidationError, AnalysisError, ConfidenceError
        ]
        
        for exc_class in derived_exceptions:
            self.assertTrue(issubclass(exc_class, AIEngineError))
            self.assertTrue(issubclass(exc_class, Exception))
    
    def test_exception_instantiation(self):
        """Test that exceptions can be instantiated and raised"""
        # Test with message
        with self.assertRaises(AIEngineError):
            raise AIEngineError("Test error message")
        
        with self.assertRaises(ModelLoadingError):
            raise ModelLoadingError("Model failed to load")
        
        # Test inheritance works for catching
        with self.assertRaises(AIEngineError):
            raise ModelLoadingError("Should be caught as AIEngineError")
    
    def test_exception_messages(self):
        """Test exception messages are preserved"""
        test_message = "This is a test error message"
        
        try:
            raise ValidationError(test_message)
        except ValidationError as e:
            self.assertEqual(str(e), test_message)


class TestAIv2Enums(unittest.TestCase):
    """Test AI v2 enum classes"""
    
    def test_conversation_tone_enum(self):
        """Test ConversationTone enum values"""
        # Test expected values
        self.assertEqual(ConversationTone.NEUTRAL.value, "neutral")
        self.assertEqual(ConversationTone.FRIENDLY.value, "friendly")
        self.assertEqual(ConversationTone.EXPERT.value, "expert")
        
        # Test all values are lowercase strings
        for tone in ConversationTone:
            self.assertIsInstance(tone.value, str)
            self.assertEqual(tone.value, tone.value.lower())
    
    def test_user_skill_level_enum(self):
        """Test UserSkillLevel enum values"""
        # Test expected values
        self.assertEqual(UserSkillLevel.NOVICE.value, "novice")
        self.assertEqual(UserSkillLevel.INTERMEDIATE.value, "intermediate")
        self.assertEqual(UserSkillLevel.EXPERT.value, "expert")
        
        # Test all values are lowercase strings
        for level in UserSkillLevel:
            self.assertIsInstance(level.value, str)
            self.assertEqual(level.value, level.value.lower())
    
    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values"""
        # Test expected values exist
        expected_levels = ["very_low", "low", "medium", "high", "very_high"]
        
        actual_values = [level.value for level in ConfidenceLevel]
        
        for expected in expected_levels:
            self.assertIn(expected, actual_values)
    
    def test_enum_importability(self):
        """Test that enums can be imported directly"""
        # Should be able to import and use without errors
        tone = ConversationTone.FRIENDLY
        level = UserSkillLevel.INTERMEDIATE
        confidence = ConfidenceLevel.HIGH
        
        self.assertEqual(tone.value, "friendly")
        self.assertEqual(level.value, "intermediate")
        self.assertEqual(confidence.value, "high")


class TestDataModelBasics(unittest.TestCase):
    """Test basic data model functionality"""
    
    def test_card_instance_creation(self):
        """Test CardInstance can be created with required fields"""
        card = CardInstance(
            name="Fireball",
            cost=4,
            attack=0,
            health=0,
            card_type="spell",
            rarity="common"
        )
        
        self.assertEqual(card.name, "Fireball")
        self.assertEqual(card.cost, 4)
        self.assertEqual(card.card_type, "spell")
        self.assertEqual(card.rarity, "common")
        
        # Should validate successfully
        card.validate()
    
    def test_card_instance_validation(self):
        """Test CardInstance validation rules"""
        # Valid minion
        minion = CardInstance(
            name="Ogre Magi",
            cost=4,
            attack=4,
            health=4,
            card_type="minion",
            rarity="common"
        )
        minion.validate()  # Should not raise
        
        # Invalid minion (no health)
        with self.assertRaises(ValueError):
            invalid_minion = CardInstance(
                name="Bad Minion",
                cost=3,
                attack=3,
                health=0,  # Minions need health > 0
                card_type="minion",
                rarity="common"
            )
            invalid_minion.validate()
        
        # Empty name should fail
        with self.assertRaises(ValueError):
            invalid_name = CardInstance(
                name="",
                cost=1,
                card_type="spell",
                rarity="common"
            )
            invalid_name.validate()
    
    def test_draft_choice_creation(self):
        """Test DraftChoice can be created and validated"""
        card = CardInstance(
            name="Flamestrike",
            cost=7,
            card_type="spell",
            rarity="rare"
        )
        
        choice = DraftChoice(
            chosen_card=card,
            pick_number=15,
            reasoning="Good AOE removal"
        )
        
        self.assertEqual(choice.chosen_card, card)
        self.assertEqual(choice.pick_number, 15)
        self.assertEqual(choice.reasoning, "Good AOE removal")
        
        # Should validate successfully
        choice.validate()
    
    def test_draft_choice_validation(self):
        """Test DraftChoice validation rules"""
        card = CardInstance(name="Test Card", cost=3, card_type="spell")  # Spell doesn't need health
        
        # Valid choice
        choice = DraftChoice(chosen_card=card, pick_number=1)
        choice.validate()  # Should not raise
        
        # Invalid pick number
        with self.assertRaises(ValueError):
            invalid_choice = DraftChoice(chosen_card=card, pick_number=0)
            invalid_choice.validate()
        
        with self.assertRaises(ValueError):
            invalid_choice = DraftChoice(chosen_card=card, pick_number=31)
            invalid_choice.validate()
        
        # Invalid chosen_card type
        with self.assertRaises(ValueError):
            invalid_choice = DraftChoice(chosen_card="not a card", pick_number=5)
            invalid_choice.validate()
    
    def test_data_model_serialization(self):
        """Test that data models can be serialized/deserialized"""
        card = CardInstance(
            name="Test Card",
            cost=3,
            attack=2,
            health=1,
            card_type="minion"
        )
        
        # Should be able to convert to dict
        card_dict = card.to_dict()
        self.assertIsInstance(card_dict, dict)
        self.assertEqual(card_dict["name"], "Test Card")
        self.assertEqual(card_dict["cost"], 3)
        
        # Should be able to convert to JSON
        card_json = card.to_json()
        self.assertIsInstance(card_json, str)
        self.assertIn("Test Card", card_json)


if __name__ == '__main__':
    unittest.main()