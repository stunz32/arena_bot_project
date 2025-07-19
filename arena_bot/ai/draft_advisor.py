#!/usr/bin/env python3
"""
Draft recommendation system for Hearthstone Arena.
Provides tier-based card recommendations and pick suggestions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PickRecommendation(Enum):
    """Recommendation levels for card picks."""
    EXCELLENT = "excellent"     # Clear best pick
    GOOD = "good"              # Solid choice
    AVERAGE = "average"        # Reasonable pick
    POOR = "poor"             # Avoid if possible
    TERRIBLE = "terrible"     # Never pick

@dataclass
class CardTier:
    """Card tier information."""
    card_code: str
    tier_score: float
    tier_letter: str  # S, A, B, C, D, F
    pick_rate: float
    win_rate: float
    notes: str = ""

@dataclass
class DraftChoice:
    """Represents a single draft choice between 3 cards."""
    cards: List[CardTier]
    recommended_pick: int  # Index of recommended card (0, 1, or 2)
    recommendation_level: PickRecommendation
    reasoning: str

class DraftAdvisor:
    """
    Arena draft advisor providing tier-based recommendations.
    Uses community tier lists and statistical data for optimal picks.
    """
    
    def __init__(self, tier_data_path: Optional[Path] = None):
        """
        Initialize draft advisor.
        
        Args:
            tier_data_path: Path to tier list data file
        """
        self.logger = logging.getLogger(__name__)
        
        # Default tier data path
        if tier_data_path is None:
            tier_data_path = Path(__file__).parent.parent.parent / "assets" / "tier_data.json"
        
        self.tier_data_path = tier_data_path
        self.tier_database: Dict[str, CardTier] = {}
        self.class_multipliers: Dict[str, float] = {}
        
        # Load tier data
        self._load_tier_database()
        
        self.logger.info(f"DraftAdvisor initialized with {len(self.tier_database)} cards")
    
    def _load_tier_database(self):
        """Load card tier data from file."""
        try:
            if self.tier_data_path.exists():
                with open(self.tier_data_path, 'r') as f:
                    data = json.load(f)
                
                # Load card tiers
                for card_code, tier_data in data.get('cards', {}).items():
                    self.tier_database[card_code] = CardTier(
                        card_code=card_code,
                        tier_score=tier_data.get('tier_score', 50.0),
                        tier_letter=tier_data.get('tier_letter', 'C'),
                        pick_rate=tier_data.get('pick_rate', 0.33),
                        win_rate=tier_data.get('win_rate', 0.50),
                        notes=tier_data.get('notes', '')
                    )
                
                # Load class multipliers
                self.class_multipliers = data.get('class_multipliers', {})
                
                self.logger.info(f"Loaded {len(self.tier_database)} card tiers from {self.tier_data_path}")
            else:
                self.logger.warning(f"Tier data file not found: {self.tier_data_path}")
                self._create_default_tier_data()
                
        except Exception as e:
            self.logger.error(f"Error loading tier database: {e}")
            self._create_default_tier_data()
    
    def _create_default_tier_data(self):
        """Create default tier data for basic functionality."""
        self.logger.info("Creating default tier data")
        
        # Default tier assignments for common cards
        default_tiers = {
            # High-tier cards
            'TOY_380': CardTier('TOY_380', 85.0, 'A', 0.85, 0.65, 'Strong tempo minion'),
            'ULD_309': CardTier('ULD_309', 80.0, 'A', 0.80, 0.62, 'Excellent value'),
            'TTN_042': CardTier('TTN_042', 75.0, 'B', 0.70, 0.58, 'Solid midrange option'),
            
            # Example tier ranges
            'LEGENDARY_EXAMPLE': CardTier('LEGENDARY_EXAMPLE', 90.0, 'S', 0.95, 0.70, 'Game-winning'),
            'COMMON_EXAMPLE': CardTier('COMMON_EXAMPLE', 45.0, 'C', 0.40, 0.48, 'Filler card'),
        }
        
        self.tier_database = default_tiers
        
        # Default class multipliers (neutral = 1.0)
        self.class_multipliers = {
            'warrior': 1.0,
            'hunter': 1.0,
            'mage': 1.0,
            'paladin': 1.0,
            'priest': 1.0,
            'rogue': 1.0,
            'shaman': 1.0,
            'warlock': 1.0,
            'druid': 1.0,
            'demon_hunter': 1.0,
            'neutral': 1.0
        }
    
    def get_card_tier(self, card_code: str) -> CardTier:
        """
        Get tier information for a card.
        
        Args:
            card_code: Hearthstone card code
            
        Returns:
            CardTier object with tier information
        """
        if card_code in self.tier_database:
            return self.tier_database[card_code]
        else:
            # Return default tier for unknown cards
            return CardTier(
                card_code=card_code,
                tier_score=50.0,  # Average tier
                tier_letter='C',
                pick_rate=0.33,   # Equal probability
                win_rate=0.50,    # Neutral win rate
                notes='Unknown card - using default tier'
            )
    
    def analyze_draft_choice(self, card_codes: List[str], player_class: str = 'neutral') -> DraftChoice:
        """
        Analyze a draft choice and provide recommendation.
        
        Args:
            card_codes: List of 3 card codes to choose from
            player_class: Player's class for class-specific bonuses
            
        Returns:
            DraftChoice with recommendation and reasoning
        """
        if len(card_codes) != 3:
            raise ValueError("Draft choice must have exactly 3 cards")
        
        # Get tier information for all cards
        card_tiers = [self.get_card_tier(code) for code in card_codes]
        
        # Apply class multipliers
        class_multiplier = self.class_multipliers.get(player_class.lower(), 1.0)
        
        # Calculate adjusted scores
        adjusted_scores = []
        for tier in card_tiers:
            # Class cards get bonus, neutrals stay the same
            bonus = 1.1 if player_class.lower() != 'neutral' and tier.card_code != 'neutral' else 1.0
            adjusted_score = tier.tier_score * class_multiplier * bonus
            adjusted_scores.append(adjusted_score)
        
        # Find best pick
        best_index = adjusted_scores.index(max(adjusted_scores))
        best_score = adjusted_scores[best_index]
        second_best = sorted(adjusted_scores, reverse=True)[1]
        score_difference = best_score - second_best
        
        # Determine recommendation level
        if score_difference >= 20:
            recommendation_level = PickRecommendation.EXCELLENT
        elif score_difference >= 10:
            recommendation_level = PickRecommendation.GOOD
        elif score_difference >= 5:
            recommendation_level = PickRecommendation.AVERAGE
        elif score_difference >= 0:
            recommendation_level = PickRecommendation.POOR
        else:
            recommendation_level = PickRecommendation.TERRIBLE
        
        # Generate reasoning
        best_card = card_tiers[best_index]
        reasoning = self._generate_reasoning(card_tiers, best_index, score_difference, player_class)
        
        return DraftChoice(
            cards=card_tiers,
            recommended_pick=best_index,
            recommendation_level=recommendation_level,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, cards: List[CardTier], best_index: int, score_diff: float, player_class: str) -> str:
        """Generate human-readable reasoning for the pick."""
        best_card = cards[best_index]
        
        reasons = []
        
        # Tier-based reasoning
        if best_card.tier_score >= 80:
            reasons.append(f"{best_card.card_code} is a high-tier card ({best_card.tier_letter}-tier)")
        elif best_card.tier_score >= 60:
            reasons.append(f"{best_card.card_code} is a solid mid-tier pick ({best_card.tier_letter}-tier)")
        else:
            reasons.append(f"{best_card.card_code} is the best of the available options")
        
        # Win rate reasoning
        if best_card.win_rate >= 0.60:
            reasons.append(f"has excellent win rate ({best_card.win_rate:.1%})")
        elif best_card.win_rate >= 0.55:
            reasons.append(f"has good win rate ({best_card.win_rate:.1%})")
        
        # Score difference reasoning
        if score_diff >= 20:
            reasons.append("clearly outclasses the other options")
        elif score_diff >= 10:
            reasons.append("is notably better than alternatives")
        elif score_diff >= 5:
            reasons.append("has a slight edge over other picks")
        else:
            reasons.append("all options are relatively close in value")
        
        # Add card notes if available
        if best_card.notes:
            reasons.append(f"Note: {best_card.notes}")
        
        return ". ".join(reasons) + "."
    
    def get_draft_statistics(self) -> Dict[str, any]:
        """Get statistics about the loaded tier database."""
        if not self.tier_database:
            return {"total_cards": 0}
        
        scores = [card.tier_score for card in self.tier_database.values()]
        win_rates = [card.win_rate for card in self.tier_database.values()]
        
        tier_counts = {}
        for card in self.tier_database.values():
            tier_counts[card.tier_letter] = tier_counts.get(card.tier_letter, 0) + 1
        
        return {
            "total_cards": len(self.tier_database),
            "average_tier_score": sum(scores) / len(scores),
            "average_win_rate": sum(win_rates) / len(win_rates),
            "tier_distribution": tier_counts,
            "score_range": (min(scores), max(scores))
        }


def get_draft_advisor() -> DraftAdvisor:
    """Get the global draft advisor instance."""
    return DraftAdvisor()