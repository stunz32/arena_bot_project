#!/usr/bin/env python3
"""
Test suite for AI Helper v2 data models
Tests the core data structures with validation, serialization, and versioning
"""

import pytest
import json
from datetime import datetime, timezone
from arena_bot.ai_v2.data_models import (
    # Enums
    CardClass, CardRarity, CardType, ArchetypePreference, 
    DraftPhase, ConfidenceLevel,
    # Data Models
    CardInfo, CardOption, EvaluationScores, DeckState, 
    StrategicContext, AIDecision,
    # Utilities
    DataModelMigrator, validate_required_field, validate_range,
    # Factory Functions
    create_card_info, create_deck_state, create_ai_decision,
    # Constants
    DATA_MODEL_VERSION, SUPPORTED_VERSIONS
)


class TestCardInfo:
    """Test CardInfo data model"""
    
    def test_card_info_creation(self):
        """Test basic card info creation"""
        card = CardInfo(
            name="Fireball",
            cost=4,
            attack=0,
            health=0,
            card_class=CardClass.MAGE,
            card_type=CardType.SPELL,
            rarity=CardRarity.COMMON,
            text="Deal 6 damage."
        )
        
        assert card.name == "Fireball"
        assert card.cost == 4
        assert card.card_class == CardClass.MAGE
        assert card.card_type == CardType.SPELL
        assert card.rarity == CardRarity.COMMON
        assert card.text == "Deal 6 damage."
        
    def test_card_info_validation(self):
        """Test card info validation"""
        # Valid minion
        card = CardInfo(name="Yeti", cost=4, attack=4, health=5, card_type=CardType.MINION)
        card.validate()  # Should not raise
        
        # Invalid minion (no health)
        with pytest.raises(ValueError, match="Minions must have health > 0"):
            bad_card = CardInfo(name="Bad Minion", cost=2, attack=2, health=0, card_type=CardType.MINION)
            bad_card.validate()
            
        # Empty name
        with pytest.raises(ValueError, match="Card name cannot be empty"):
            bad_card = CardInfo(name="", cost=1)
            bad_card.validate()
    
    def test_card_info_serialization(self):
        """Test card info serialization/deserialization"""
        original = CardInfo(
            name="Lightning Bolt",
            cost=1,
            card_class=CardClass.SHAMAN,
            card_type=CardType.SPELL,
            rarity=CardRarity.COMMON
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data["name"] == "Lightning Bolt"
        assert data["cost"] == 1
        assert data["card_class"] == "Shaman"
        
        # Test from_dict
        restored = CardInfo.from_dict(data)
        assert restored.name == original.name
        assert restored.cost == original.cost
        assert restored.card_class == original.card_class
        
        # Test JSON round-trip
        json_str = original.to_json()
        restored_json = CardInfo.from_json(json_str)
        assert restored_json.name == original.name
        
    def test_card_info_enum_conversion(self):
        """Test automatic enum conversion from strings"""
        card = CardInfo(
            name="Test Card",
            cost=2,
            card_class="Warrior",  # String instead of enum
            card_type="Minion",    # String instead of enum
            rarity="Epic"          # String instead of enum
        )
        
        assert card.card_class == CardClass.WARRIOR
        assert card.card_type == CardType.MINION
        assert card.rarity == CardRarity.EPIC
        
    def test_card_info_range_validation(self):
        """Test cost and stat range validation"""
        # Valid ranges
        card = CardInfo(name="Valid", cost=10, attack=30, health=30)
        card.validate()
        
        # Invalid cost (negative)
        with pytest.raises(ValueError, match="cost.*>= 0"):
            CardInfo(name="Invalid", cost=-1)
            
        # Invalid cost (too high)
        with pytest.raises(ValueError, match="cost.*<= 30"):
            CardInfo(name="Invalid", cost=31)


class TestEvaluationScores:
    """Test EvaluationScores data model"""
    
    def test_evaluation_scores_creation(self):
        """Test evaluation scores creation and composite calculation"""
        scores = EvaluationScores(
            base_value=75.0,
            tempo_score=80.0,
            value_score=70.0,
            synergy_score=85.0,
            curve_score=75.0,
            redraftability_score=60.0,
            confidence=ConfidenceLevel.HIGH
        )
        
        assert scores.base_value == 75.0
        assert scores.confidence == ConfidenceLevel.HIGH
        assert 70.0 < scores.composite_score < 80.0  # Weighted average
        
    def test_evaluation_scores_validation(self):
        """Test score range validation"""
        # Valid scores
        scores = EvaluationScores(
            base_value=50.0, tempo_score=50.0, value_score=50.0,
            synergy_score=50.0, curve_score=50.0, redraftability_score=50.0
        )
        scores.validate()
        
        # Invalid score (too high)
        with pytest.raises(ValueError, match="base_value.*<= 100"):
            EvaluationScores(
                base_value=101.0, tempo_score=50.0, value_score=50.0,
                synergy_score=50.0, curve_score=50.0, redraftability_score=50.0
            )
            
        # Invalid score (negative)
        with pytest.raises(ValueError, match="tempo_score.*>= 0"):
            EvaluationScores(
                base_value=50.0, tempo_score=-1.0, value_score=50.0,
                synergy_score=50.0, curve_score=50.0, redraftability_score=50.0
            )


class TestDeckState:
    """Test DeckState data model"""
    
    def test_deck_state_creation(self):
        """Test deck state creation and automatic calculations"""
        card1 = CardInfo(name="Card1", cost=1, card_class=CardClass.MAGE)
        card2 = CardInfo(name="Card2", cost=3, card_class=CardClass.MAGE)
        card3 = CardInfo(name="Card3", cost=5, card_class=CardClass.NEUTRAL, rarity=CardRarity.LEGENDARY)
        
        deck_state = DeckState(
            cards=[card1, card2, card3],
            hero_class=CardClass.MAGE,
            current_pick=4,
            archetype_preference=ArchetypePreference.TEMPO
        )
        
        assert len(deck_state.cards) == 3
        assert deck_state.hero_class == CardClass.MAGE
        assert deck_state.current_pick == 4
        assert deck_state.archetype_preference == ArchetypePreference.TEMPO
        assert deck_state.draft_phase == DraftPhase.EARLY
        assert deck_state.total_cards == 3
        assert deck_state.average_cost == 3.0  # (1+3+5)/3
        
        # Check mana curve
        assert deck_state.mana_curve[1] == 1  # One 1-cost card
        assert deck_state.mana_curve[3] == 1  # One 3-cost card
        assert deck_state.mana_curve[5] == 1  # One 5-cost card
        
        # Check distributions
        assert deck_state.class_distribution["Mage"] == 2
        assert deck_state.class_distribution["Neutral"] == 1
        assert deck_state.rarity_distribution["Common"] == 2  # Default rarity
        assert deck_state.rarity_distribution["Legendary"] == 1
        
    def test_deck_state_draft_phase_calculation(self):
        """Test automatic draft phase calculation"""
        # Early phase
        early_deck = DeckState(current_pick=5)
        assert early_deck.draft_phase == DraftPhase.EARLY
        
        # Mid phase
        mid_deck = DeckState(current_pick=15)
        assert mid_deck.draft_phase == DraftPhase.MID
        
        # Late phase
        late_deck = DeckState(current_pick=25)
        assert late_deck.draft_phase == DraftPhase.LATE
        
    def test_deck_state_add_card(self):
        """Test dynamic card addition"""
        deck_state = DeckState(hero_class=CardClass.HUNTER)
        assert deck_state.total_cards == 0
        
        new_card = CardInfo(name="Hunter Card", cost=2, card_class=CardClass.HUNTER)
        deck_state.add_card(new_card)
        
        assert deck_state.total_cards == 1
        assert deck_state.mana_curve[2] == 1
        assert deck_state.class_distribution["Hunter"] == 1
        
    def test_deck_state_validation(self):
        """Test deck state validation"""
        # Valid deck
        deck_state = DeckState(hero_class=CardClass.PALADIN, current_pick=10)
        deck_state.validate()
        
        # Invalid: too many cards
        too_many_cards = [CardInfo(name=f"Card{i}", cost=1) for i in range(31)]
        with pytest.raises(ValueError, match="Deck cannot have more than 30 cards"):
            bad_deck = DeckState(cards=too_many_cards)
            bad_deck.validate()
            
        # Invalid: pick number too high
        with pytest.raises(ValueError, match="Cannot have more than 30 picks"):
            bad_deck = DeckState(current_pick=31)
            bad_deck.validate()


class TestAIDecision:
    """Test AIDecision data model (universal data contract)"""
    
    def test_ai_decision_creation(self):
        """Test AI decision creation with all components"""
        # Create test cards and evaluations
        card1 = CardInfo(name="Option1", cost=2)
        card2 = CardInfo(name="Option2", cost=3)
        card3 = CardInfo(name="Option3", cost=4)
        
        option1 = CardOption(card_info=card1, position=1, detection_confidence=0.95)
        option2 = CardOption(card_info=card2, position=2, detection_confidence=0.90)
        option3 = CardOption(card_info=card3, position=3, detection_confidence=0.85)
        
        scores1 = EvaluationScores(75, 80, 70, 85, 75, 60)
        scores2 = EvaluationScores(80, 75, 75, 80, 80, 65)
        scores3 = EvaluationScores(70, 70, 80, 75, 70, 70)
        
        decision = AIDecision(
            recommended_pick=2,
            card_evaluations=[(option1, scores1), (option2, scores2), (option3, scores3)],
            confidence=ConfidenceLevel.HIGH,
            reasoning="Option 2 provides better tempo for current deck state",
            analysis_duration_ms=150.5
        )
        
        assert decision.recommended_pick == 2
        assert len(decision.card_evaluations) == 3
        assert decision.confidence == ConfidenceLevel.HIGH
        assert decision.reasoning.startswith("Option 2")
        assert decision.analysis_duration_ms == 150.5
        assert decision.correlation_id is not None
        assert decision.decision_timestamp is not None
        
    def test_ai_decision_validation(self):
        """Test AI decision validation"""
        card = CardInfo(name="Test", cost=1, card_type="Spell")
        option = CardOption(card_info=card, position=1)
        scores = EvaluationScores(50, 50, 50, 50, 50, 50)
        
        # Valid decision
        decision = AIDecision(
            recommended_pick=1,
            card_evaluations=[(option, scores)]
        )
        
        # Should fail: not exactly 3 evaluations
        with pytest.raises(ValueError, match="Must have exactly 3 card evaluations"):
            decision.validate()
            
        # Create proper 3-card decision
        card2 = CardInfo(name="Test2", cost=2, card_type="Spell")
        card3 = CardInfo(name="Test3", cost=3, card_type="Spell")
        option2 = CardOption(card_info=card2, position=2)
        option3 = CardOption(card_info=card3, position=3)
        
        valid_decision = AIDecision(
            recommended_pick=2,
            card_evaluations=[(option, scores), (option2, scores), (option3, scores)]
        )
        valid_decision.validate()  # Should not raise
        
    def test_ai_decision_helper_methods(self):
        """Test AI decision helper methods"""
        card1 = CardInfo(name="Card1", cost=1)
        card2 = CardInfo(name="Card2", cost=2)
        card3 = CardInfo(name="Card3", cost=3)
        
        option1 = CardOption(card_info=card1, position=1)
        option2 = CardOption(card_info=card2, position=2)
        option3 = CardOption(card_info=card3, position=3)
        
        scores = EvaluationScores(50, 50, 50, 50, 50, 50)
        
        decision = AIDecision(
            recommended_pick=2,
            card_evaluations=[(option1, scores), (option2, scores), (option3, scores)]
        )
        
        # Test get_recommended_card
        recommended_card = decision.get_recommended_card()
        assert recommended_card.card_info.name == "Card2"
        assert recommended_card.position == 2
        
        # Test get_evaluation_for_card
        eval_for_pos_3 = decision.get_evaluation_for_card(3)
        assert eval_for_pos_3.base_value == 50


class TestValidationHelpers:
    """Test validation helper functions"""
    
    def test_validate_required_field(self):
        """Test required field validation"""
        # Valid cases
        assert validate_required_field("test", "name") == "test"
        assert validate_required_field(5, "count", int) == 5
        
        # None value
        with pytest.raises(ValueError, match="Required field 'name' cannot be None"):
            validate_required_field(None, "name")
            
        # Type conversion
        assert validate_required_field("10", "count", int) == 10
        
        # Invalid type conversion
        with pytest.raises(ValueError, match="Field 'count' must be of type"):
            validate_required_field("invalid", "count", int)
            
    def test_validate_range(self):
        """Test range validation"""
        # Valid range
        assert validate_range(5, "value", 0, 10) == 5
        
        # Below minimum
        with pytest.raises(ValueError, match="value.*>= 0"):
            validate_range(-1, "value", 0, 10)
            
        # Above maximum
        with pytest.raises(ValueError, match="value.*<= 10"):
            validate_range(11, "value", 0, 10)
            
        # No limits
        assert validate_range(100, "value") == 100


class TestDataModelMigrator:
    """Test data model version migration"""
    
    def test_version_compatibility_check(self):
        """Test version compatibility checking"""
        assert DataModelMigrator.is_version_compatible("2.0.0")
        assert DataModelMigrator.is_version_compatible("1.0.0")
        assert not DataModelMigrator.is_version_compatible("3.0.0")
        
    def test_migration_same_version(self):
        """Test migration with same version (no-op)"""
        data = {"test": "data", "_version": "2.0.0"}
        result = DataModelMigrator.migrate_data(data, "2.0.0", "2.0.0")
        assert result == data
        
    def test_migration_version_upgrade(self):
        """Test basic version migration"""
        data = {"test": "data"}
        result = DataModelMigrator.migrate_data(data, "1.0.0", "2.0.0")
        assert result["_version"] == "2.0.0"
        assert result["test"] == "data"


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_card_info(self):
        """Test card info factory function"""
        card = create_card_info("Test Card", 3, card_class=CardClass.PRIEST)
        assert isinstance(card, CardInfo)
        assert card.name == "Test Card"
        assert card.cost == 3
        assert card.card_class == CardClass.PRIEST
        
    def test_create_deck_state(self):
        """Test deck state factory function"""
        deck = create_deck_state(CardClass.WARLOCK, current_pick=15)
        assert isinstance(deck, DeckState)
        assert deck.hero_class == CardClass.WARLOCK
        assert deck.current_pick == 15
        assert deck.draft_phase == DraftPhase.MID
        
    def test_create_ai_decision(self):
        """Test AI decision factory function"""
        card = CardInfo(name="Test", cost=1, card_type="Spell")
        option = CardOption(card_info=card, position=1)
        scores = EvaluationScores(50, 50, 50, 50, 50, 50)
        
        # Create minimal valid decision
        card2 = CardInfo(name="Test2", cost=2, card_type="Spell")
        card3 = CardInfo(name="Test3", cost=3, card_type="Spell")
        option2 = CardOption(card_info=card2, position=2)
        option3 = CardOption(card_info=card3, position=3)
        
        decision = create_ai_decision(
            1, 
            [(option, scores), (option2, scores), (option3, scores)],
            confidence=ConfidenceLevel.MEDIUM
        )
        
        assert isinstance(decision, AIDecision)
        assert decision.recommended_pick == 1
        assert decision.confidence == ConfidenceLevel.MEDIUM


class TestChecksumValidation:
    """Test data integrity validation"""
    
    def test_checksum_generation(self):
        """Test checksum generation for data integrity"""
        card = CardInfo(name="Test", cost=1, card_type="Spell")
        checksum = card.get_checksum()
        
        assert isinstance(checksum, str)
        assert len(checksum) == 16  # SHA-256 truncated to 16 chars
        
        # Same data should produce same checksum
        card2 = CardInfo(name="Test", cost=1, card_type="Spell")
        assert card.get_checksum() == card2.get_checksum()
        
        # Different data should produce different checksum
        card3 = CardInfo(name="Different", cost=1)
        assert card.get_checksum() != card3.get_checksum()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])