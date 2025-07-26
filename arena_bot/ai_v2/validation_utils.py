"""
AI v2 Validation Utilities - Comprehensive Type Safety

Provides validation functions for all data types used in the AI v2 system
to ensure type safety and catch invalid inputs at system boundaries.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    
    def __init__(self, field: str, value: Any, expected_type: str, message: str = None):
        self.field = field
        self.value = value
        self.expected_type = expected_type
        self.message = message or f"Invalid {field}: expected {expected_type}, got {type(value).__name__}"
        super().__init__(self.message)


class CardDataValidator:
    """Validator for card-related data."""
    
    @staticmethod
    def validate_card_id(card_id: Any, field_name: str = "card_id") -> str:
        """Validate and sanitize card ID."""
        if card_id is None:
            raise ValidationError(field_name, card_id, "str", "Card ID cannot be None")
        
        if not isinstance(card_id, str):
            try:
                card_id = str(card_id)
            except (ValueError, TypeError):
                raise ValidationError(field_name, card_id, "str", "Card ID must be convertible to string")
        
        card_id = card_id.strip()
        if not card_id:
            raise ValidationError(field_name, card_id, "str", "Card ID cannot be empty")
        
        # Basic format validation (alphanumeric + underscore)
        if not card_id.replace('_', '').replace('-', '').isalnum():
            logger.warning(f"Card ID {card_id} contains unusual characters")
        
        return card_id
    
    @staticmethod
    def validate_card_list(card_list: Any, field_name: str = "card_list", allow_empty: bool = True) -> List[str]:
        """Validate list of card IDs."""
        if card_list is None:
            if allow_empty:
                return []
            raise ValidationError(field_name, card_list, "List[str]", "Card list cannot be None")
        
        if not isinstance(card_list, list):
            raise ValidationError(field_name, card_list, "List[str]", "Card list must be a list")
        
        if not allow_empty and len(card_list) == 0:
            raise ValidationError(field_name, card_list, "List[str]", "Card list cannot be empty")
        
        validated_cards = []
        for i, card_id in enumerate(card_list):
            try:
                validated_card = CardDataValidator.validate_card_id(card_id, f"{field_name}[{i}]")
                validated_cards.append(validated_card)
            except ValidationError as e:
                logger.error(f"Invalid card in {field_name} at index {i}: {e}")
                # Skip invalid cards but continue processing
                continue
        
        return validated_cards
    
    @staticmethod
    def validate_hero_class(hero_class: Any, field_name: str = "hero_class") -> str:
        """Validate hero class."""
        if hero_class is None:
            raise ValidationError(field_name, hero_class, "str", "Hero class cannot be None")
        
        if not isinstance(hero_class, str):
            hero_class = str(hero_class)
        
        hero_class = hero_class.upper().strip()
        
        valid_heroes = {
            'MAGE', 'PALADIN', 'ROGUE', 'HUNTER', 'WARLOCK', 
            'WARRIOR', 'SHAMAN', 'DRUID', 'PRIEST', 'DEMONHUNTER', 'NEUTRAL'
        }
        
        if hero_class not in valid_heroes:
            logger.warning(f"Unknown hero class: {hero_class}")
            # Don't raise error for forward compatibility, but log warning
        
        return hero_class


class NumericValidator:
    """Validator for numeric data."""
    
    @staticmethod
    def validate_score(score: Any, field_name: str = "score", min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate numeric score within bounds."""
        if score is None:
            return 0.0
        
        try:
            score = float(score)
        except (ValueError, TypeError):
            raise ValidationError(field_name, score, "float", "Score must be numeric")
        
        if not (min_val <= score <= max_val):
            logger.warning(f"{field_name} {score} outside expected range [{min_val}, {max_val}]")
            # Clamp to valid range instead of raising error
            score = max(min_val, min(max_val, score))
        
        return score
    
    @staticmethod
    def validate_percentage(percentage: Any, field_name: str = "percentage") -> float:
        """Validate percentage value (0-100)."""
        if percentage is None:
            return 0.0
        
        try:
            percentage = float(percentage)
        except (ValueError, TypeError):
            raise ValidationError(field_name, percentage, "float", "Percentage must be numeric")
        
        if not (0.0 <= percentage <= 100.0):
            logger.warning(f"{field_name} {percentage}% outside valid range [0, 100]")
            percentage = max(0.0, min(100.0, percentage))
        
        return percentage
    
    @staticmethod
    def validate_mana_cost(cost: Any, field_name: str = "mana_cost") -> int:
        """Validate mana cost."""
        if cost is None:
            return 0
        
        try:
            cost = int(cost)
        except (ValueError, TypeError):
            raise ValidationError(field_name, cost, "int", "Mana cost must be integer")
        
        if cost < 0:
            logger.warning(f"Negative mana cost: {cost}")
            cost = 0
        elif cost > 20:
            logger.warning(f"Unusually high mana cost: {cost}")
        
        return cost


class DeckStateValidator:
    """Validator for DeckState objects."""
    
    @staticmethod
    def validate_deck_state(deck_state: Any, field_name: str = "deck_state") -> 'DeckState':
        """Validate DeckState object with comprehensive checks."""
        if deck_state is None:
            from .data_models import DeckState
            logger.warning(f"{field_name} is None, creating default DeckState")
            return DeckState(hero_class="NEUTRAL", archetype="Unknown")
        
        # Check if it's already a DeckState object
        if hasattr(deck_state, 'hero_class') and hasattr(deck_state, 'archetype'):
            # Validate existing DeckState fields
            try:
                hero_class = CardDataValidator.validate_hero_class(
                    getattr(deck_state, 'hero_class', 'NEUTRAL'), 
                    f"{field_name}.hero_class"
                )
                deck_state.hero_class = hero_class
                
                # Validate drafted cards
                drafted_cards = getattr(deck_state, 'drafted_cards', [])
                if drafted_cards:
                    validated_cards = CardDataValidator.validate_card_list(
                        drafted_cards, 
                        f"{field_name}.drafted_cards"
                    )
                    deck_state.drafted_cards = validated_cards
                
                # Validate pick number
                pick_number = getattr(deck_state, 'pick_number', 0)
                deck_state.pick_number = max(0, min(30, int(pick_number)))
                
                # Validate mana curve
                mana_curve = getattr(deck_state, 'mana_curve', {})
                if mana_curve and isinstance(mana_curve, dict):
                    validated_curve = {}
                    for cost, count in mana_curve.items():
                        try:
                            cost_int = NumericValidator.validate_mana_cost(cost, f"{field_name}.mana_curve.cost")
                            count_int = max(0, int(count))
                            validated_curve[cost_int] = count_int
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid mana curve entry: {cost}={count}")
                            continue
                    deck_state.mana_curve = validated_curve
                
                return deck_state
                
            except Exception as e:
                logger.error(f"Failed to validate DeckState: {e}")
                from .data_models import DeckState
                return DeckState(hero_class="NEUTRAL", archetype="Unknown")
        
        # If it's not a DeckState object, try to create one
        try:
            from .data_models import DeckState
            if isinstance(deck_state, dict):
                return DeckState(
                    hero_class=CardDataValidator.validate_hero_class(
                        deck_state.get('hero_class', 'NEUTRAL')
                    ),
                    archetype=str(deck_state.get('archetype', 'Unknown')),
                    drafted_cards=CardDataValidator.validate_card_list(
                        deck_state.get('drafted_cards', [])
                    ),
                    pick_number=max(0, min(30, int(deck_state.get('pick_number', 0))))
                )
            else:
                logger.error(f"Cannot convert {type(deck_state)} to DeckState")
                return DeckState(hero_class="NEUTRAL", archetype="Unknown")
                
        except Exception as e:
            logger.error(f"Failed to create DeckState from {deck_state}: {e}")
            from .data_models import DeckState
            return DeckState(hero_class="NEUTRAL", archetype="Unknown")


class AIDecisionValidator:
    """Validator for AI decision outputs."""
    
    @staticmethod
    def validate_pick_index(index: Any, offered_cards_count: int = 3, field_name: str = "pick_index") -> int:
        """Validate card pick index."""
        if index is None:
            logger.warning(f"{field_name} is None, defaulting to 0")
            return 0
        
        try:
            index = int(index)
        except (ValueError, TypeError):
            raise ValidationError(field_name, index, "int", "Pick index must be integer")
        
        if not (0 <= index < offered_cards_count):
            logger.warning(f"Pick index {index} out of bounds [0, {offered_cards_count-1}], clamping")
            index = max(0, min(offered_cards_count - 1, index))
        
        return index
    
    @staticmethod
    def validate_analysis_list(analysis: Any, expected_length: int = 3, field_name: str = "analysis") -> List[Dict]:
        """Validate card analysis list."""
        if analysis is None:
            logger.warning(f"{field_name} is None, creating empty list")
            return []
        
        if not isinstance(analysis, list):
            logger.error(f"{field_name} must be a list, got {type(analysis)}")
            return []
        
        if len(analysis) != expected_length:
            logger.warning(f"{field_name} length {len(analysis)} != expected {expected_length}")
        
        # Validate each analysis entry
        validated_analysis = []
        for i, entry in enumerate(analysis):
            if not isinstance(entry, dict):
                logger.warning(f"{field_name}[{i}] is not a dict, skipping")
                continue
            
            # Ensure required fields exist
            validated_entry = {
                'card_id': str(entry.get('card_id', f'unknown_card_{i}')),
                'scores': entry.get('scores', {}),
                'explanation': str(entry.get('explanation', 'No explanation available'))
            }
            validated_analysis.append(validated_entry)
        
        return validated_analysis


def validate_system_boundary_input(data: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """
    Main validation function for system boundary inputs.
    
    Args:
        data: Input data dictionary
        operation: Operation type for context-specific validation
        
    Returns:
        Validated and sanitized data dictionary
    """
    validated_data = {}
    
    try:
        # Common validations based on operation type
        if operation in ['hero_recommendation', 'get_hero_recommendation']:
            if 'hero_classes' in data:
                validated_data['hero_classes'] = CardDataValidator.validate_card_list(
                    data['hero_classes'], 'hero_classes', allow_empty=False
                )
        
        elif operation in ['card_evaluation', 'evaluate_card']:
            if 'card_id' in data:
                validated_data['card_id'] = CardDataValidator.validate_card_id(data['card_id'])
            
            if 'deck_state' in data:
                validated_data['deck_state'] = DeckStateValidator.validate_deck_state(data['deck_state'])
        
        elif operation in ['ai_decision', 'get_recommendation']:
            if 'deck_state' in data:
                validated_data['deck_state'] = DeckStateValidator.validate_deck_state(data['deck_state'])
            
            if 'offered_cards' in data:
                validated_data['offered_cards'] = CardDataValidator.validate_card_list(
                    data['offered_cards'], 'offered_cards', allow_empty=False
                )
        
        # Copy through any unvalidated fields with warning
        for key, value in data.items():
            if key not in validated_data:
                logger.debug(f"Unvalidated field passed through: {key}")
                validated_data[key] = value
        
        return validated_data
        
    except ValidationError as e:
        logger.error(f"Validation failed for {operation}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected validation error for {operation}: {e}")
        raise ValidationError("validation", data, "dict", f"Validation failed: {e}")


def log_validation_stats():
    """Log validation statistics for monitoring."""
    # This could be enhanced to track validation metrics
    logger.info("Validation system active and monitoring type safety")