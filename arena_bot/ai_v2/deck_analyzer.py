"""
Strategic Deck Analyzer

Analyzes the overall draft state and provides strategic insights
about archetype conformance, missing components, and optimization opportunities.
"""

import logging
from typing import Dict, List, Any, Optional
from .data_models import DeckState, DimensionalScores
from .archetype_config import ARCHETYPE_WEIGHTS


class StrategicDeckAnalyzer:
    """
    Analyzes deck state for strategic insights.
    
    Provides archetype conformance scoring, gap analysis,
    and cut candidate recommendations for Underground Arena.
    """
    
    def __init__(self, card_evaluator=None):
        """Initialize the strategic deck analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Card evaluator for getting dimensional scores
        self.card_evaluator = card_evaluator
        
        self.logger.info("StrategicDeckAnalyzer initialized with real analysis logic")
    
    def analyze_deck(self, deck_state: DeckState) -> Dict[str, Any]:
        """
        Comprehensive deck analysis.
        
        Args:
            deck_state: Current draft state
            
        Returns:
            Dictionary with archetype conformance, gaps, and recommendations
        """
        # Calculate real archetype conformance
        archetype_conformance = self._calculate_archetype_conformance(deck_state)
        
        # Identify strategic gaps
        strategic_gaps = self._identify_strategic_gaps(deck_state)
        
        # Find cut candidates for Underground Arena
        cut_candidates = self._identify_cut_candidates(deck_state)
        
        analysis = {
            "archetype_conformance": archetype_conformance,
            "strategic_gaps": strategic_gaps,
            "cut_candidates": cut_candidates,
            "curve_analysis": self._analyze_curve(deck_state),
            "draft_phase_advice": self._get_phase_advice(deck_state),
            "pivot_opportunities": []  # Alternative archetype suggestions
        }
        
        return analysis
    
    def _calculate_archetype_conformance(self, deck_state: DeckState) -> float:
        """
        Calculate how well the deck matches its chosen archetype.
        
        Uses ideal metrics from archetype_config.py to score conformance.
        Returns a score from 0.0 (no conformance) to 1.0 (perfect conformance).
        """
        try:
            # Return neutral score if no cards drafted yet
            if not deck_state.drafted_cards or len(deck_state.drafted_cards) < 3:
                return 0.5
            
            # Get ideal weights for the chosen archetype
            archetype = deck_state.archetype
            if archetype not in ARCHETYPE_WEIGHTS:
                self.logger.warning(f"Unknown archetype '{archetype}', using Balanced")
                archetype = "Balanced"
            
            ideal_weights = ARCHETYPE_WEIGHTS[archetype]
            
            # Calculate actual dimensional score averages for drafted cards
            if not self.card_evaluator:
                self.logger.warning("No card evaluator available, returning default conformance")
                return 0.6
            
            # Get dimensional scores for all drafted cards
            card_scores = []
            for card_id in deck_state.drafted_cards:
                try:
                    scores = self.card_evaluator.evaluate_card(card_id, deck_state)
                    card_scores.append(scores)
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate card {card_id}: {e}")
                    continue
            
            if not card_scores:
                return 0.5
            
            # Calculate average scores across all cards
            avg_scores = {
                "base_value": sum(s.base_value for s in card_scores) / len(card_scores),
                "tempo_score": sum(s.tempo_score for s in card_scores) / len(card_scores),
                "value_score": sum(s.value_score for s in card_scores) / len(card_scores),
                "synergy_score": sum(s.synergy_score for s in card_scores) / len(card_scores),
                "curve_score": sum(s.curve_score for s in card_scores) / len(card_scores),
                "re_draftability_score": sum(s.re_draftability_score for s in card_scores) / len(card_scores),
                "greed_score": sum(s.greed_score for s in card_scores) / len(card_scores),
            }
            
            # Calculate conformance by comparing actual vs ideal patterns
            # We use normalized differences to measure how well the pattern matches
            conformance_scores = []
            
            for dimension in ["base_value", "tempo_score", "value_score", "synergy_score", "curve_score", "greed_score"]:
                if dimension in ideal_weights:
                    actual = avg_scores[dimension]
                    ideal = ideal_weights[dimension]
                    
                    # Normalize both values to 0-1 range for comparison
                    # Ideal weights are typically 0.3-1.6, actual scores are 0-1
                    normalized_ideal = min(ideal / 1.6, 1.0)  # Cap at 1.0
                    normalized_actual = min(actual, 1.0)
                    
                    # Calculate similarity (1.0 - absolute difference)
                    similarity = 1.0 - abs(normalized_ideal - normalized_actual)
                    conformance_scores.append(similarity)
            
            # Return average conformance across all dimensions
            if conformance_scores:
                final_conformance = sum(conformance_scores) / len(conformance_scores)
                return max(0.0, min(1.0, final_conformance))  # Clamp to 0-1 range
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating archetype conformance: {e}")
            return 0.5
    
    def _identify_strategic_gaps(self, deck_state: DeckState) -> List[str]:
        """
        Identify missing strategic components.
        
        Examples:
        - "Control deck needs AOE removal"
        - "Aggro deck lacks early minions" 
        - "No card draw for value strategy"
        """
        gaps = []
        
        try:
            # Only analyze gaps if we have enough cards and are in mid-to-late draft
            if not deck_state.drafted_cards or deck_state.pick_number < 8:
                return gaps
            
            archetype = deck_state.archetype
            card_count = len(deck_state.drafted_cards)
            
            # Analyze mana curve
            curve = deck_state.mana_curve or {}
            early_cards = curve.get(1, 0) + curve.get(2, 0) + curve.get(3, 0)
            mid_cards = curve.get(4, 0) + curve.get(5, 0) + curve.get(6, 0)
            late_cards = sum(curve.get(cost, 0) for cost in range(7, 11))
            
            # Early curve (picks 8-15): Check basic curve needs
            if 8 <= deck_state.pick_number <= 15:
                if archetype == "Aggro":
                    if early_cards < card_count * 0.6:
                        gaps.append("Aggro deck needs more early minions (1-3 mana)")
                    if late_cards > card_count * 0.2:
                        gaps.append("Aggro deck has too many expensive cards")
                
                elif archetype == "Control":
                    if early_cards > card_count * 0.4:
                        gaps.append("Control deck may have too many early cards")
                    if late_cards < card_count * 0.3:
                        gaps.append("Control deck needs more late game threats")
                
                elif archetype == "Tempo":
                    if early_cards < card_count * 0.4:
                        gaps.append("Tempo deck needs more early pressure")
                    if mid_cards < card_count * 0.4:
                        gaps.append("Tempo deck needs solid mid-game presence")
            
            # Mid-to-late draft (picks 16+): Check specialized needs
            elif deck_state.pick_number >= 16:
                if archetype == "Aggro":
                    if early_cards < card_count * 0.5:
                        gaps.append("Critical: Aggro deck severely lacks early game")
                    
                    # Check for direct damage / reach
                    # This is a simplified check - in a full implementation, 
                    # we'd analyze card tags/effects
                    avg_tempo = 0.0
                    if self.card_evaluator and deck_state.drafted_cards:
                        tempo_scores = []
                        for card_id in deck_state.drafted_cards:
                            try:
                                scores = self.card_evaluator.evaluate_card(card_id, deck_state)
                                tempo_scores.append(scores.tempo_score)
                            except:
                                continue
                        if tempo_scores:
                            avg_tempo = sum(tempo_scores) / len(tempo_scores)
                    
                    if avg_tempo < 0.6:
                        gaps.append("Aggro deck needs more immediate board impact")
                
                elif archetype == "Control":
                    if late_cards < card_count * 0.25:
                        gaps.append("Control deck needs more win conditions")
                    
                    # Check for removal/AOE (simplified)
                    avg_value = 0.0
                    if self.card_evaluator and deck_state.drafted_cards:
                        value_scores = []
                        for card_id in deck_state.drafted_cards:
                            try:
                                scores = self.card_evaluator.evaluate_card(card_id, deck_state)
                                value_scores.append(scores.value_score)
                            except:
                                continue
                        if value_scores:
                            avg_value = sum(value_scores) / len(value_scores)
                    
                    if avg_value < 0.6:
                        gaps.append("Control deck needs more card advantage engines")
                
                elif archetype == "Synergy":
                    # Check synergy score
                    avg_synergy = 0.0
                    if self.card_evaluator and deck_state.drafted_cards:
                        synergy_scores = []
                        for card_id in deck_state.drafted_cards:
                            try:
                                scores = self.card_evaluator.evaluate_card(card_id, deck_state)
                                synergy_scores.append(scores.synergy_score)
                            except:
                                continue
                        if synergy_scores:
                            avg_synergy = sum(synergy_scores) / len(synergy_scores)
                    
                    if avg_synergy < 0.7:
                        gaps.append("Synergy deck needs more tribal/archetype enablers")
                
                elif archetype == "Attrition":
                    # Check for value engines
                    avg_value = 0.0
                    if self.card_evaluator and deck_state.drafted_cards:
                        value_scores = []
                        for card_id in deck_state.drafted_cards:
                            try:
                                scores = self.card_evaluator.evaluate_card(card_id, deck_state)
                                value_scores.append(scores.value_score)
                            except:
                                continue
                        if value_scores:
                            avg_value = sum(value_scores) / len(value_scores)
                    
                    if avg_value < 0.7:
                        gaps.append("Attrition deck needs more card advantage")
                    
                    if early_cards > card_count * 0.5:
                        gaps.append("Attrition deck may be too aggressive for late game plan")
            
            # Universal curve checks for late draft
            if deck_state.pick_number >= 20:
                total_cards = sum(curve.values())
                if total_cards > 0:
                    if early_cards / total_cards < 0.2:
                        gaps.append("Deck needs more early game presence")
                    elif early_cards / total_cards > 0.8:
                        gaps.append("Deck may struggle in late game")
        
        except Exception as e:
            self.logger.error(f"Error identifying strategic gaps: {e}")
        
        return gaps
    
    def _identify_cut_candidates(self, deck_state: DeckState) -> List[str]:
        """
        Identify cards that could be replaced in Underground Arena redraft.
        
        Uses formula: cut_score = (1 - archetype_conformance) * re_draftability_score
        Cards with higher cut scores are better candidates for replacement.
        """
        cut_candidates = []
        
        try:
            # Only analyze if we have enough cards and card evaluator
            if not deck_state.drafted_cards or len(deck_state.drafted_cards) < 5:
                return cut_candidates
            
            if not self.card_evaluator:
                return cut_candidates
            
            archetype = deck_state.archetype
            if archetype not in ARCHETYPE_WEIGHTS:
                archetype = "Balanced"
            
            ideal_weights = ARCHETYPE_WEIGHTS[archetype]
            
            # Calculate cut scores for each card
            card_cut_scores = []
            
            for card_id in deck_state.drafted_cards:
                try:
                    # Get card's dimensional scores
                    card_scores = self.card_evaluator.evaluate_card(card_id, deck_state)
                    
                    # Calculate individual card's archetype conformance
                    card_conformance_scores = []
                    
                    for dimension in ["base_value", "tempo_score", "value_score", "synergy_score", "curve_score", "greed_score"]:
                        if dimension in ideal_weights:
                            actual = getattr(card_scores, dimension, 0.0)
                            ideal = ideal_weights[dimension]
                            
                            # Normalize values for comparison
                            normalized_ideal = min(ideal / 1.6, 1.0)
                            normalized_actual = min(actual, 1.0)
                            
                            # Calculate similarity
                            similarity = 1.0 - abs(normalized_ideal - normalized_actual)
                            card_conformance_scores.append(similarity)
                    
                    # Calculate card's archetype conformance
                    if card_conformance_scores:
                        card_conformance = sum(card_conformance_scores) / len(card_conformance_scores)
                    else:
                        card_conformance = 0.5  # Default if no scores
                    
                    # Calculate cut score using the formula
                    # Higher re_draftability_score means more flexible/replaceable
                    # Lower conformance means doesn't fit archetype well
                    cut_score = (1.0 - card_conformance) * card_scores.re_draftability_score
                    
                    card_cut_scores.append({
                        'card_id': card_id,
                        'cut_score': cut_score,
                        'conformance': card_conformance,
                        're_draftability': card_scores.re_draftability_score
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate cut candidate {card_id}: {e}")
                    continue
            
            # Sort by cut score (highest first) and take top candidates
            card_cut_scores.sort(key=lambda x: x['cut_score'], reverse=True)
            
            # Identify top cut candidates
            # We'll consider cards with cut_score > 0.4 as good candidates
            # And always include at least 1-2 cards if any exist
            cut_threshold = 0.4
            min_candidates = min(2, len(card_cut_scores))
            
            for i, card_data in enumerate(card_cut_scores):
                if (card_data['cut_score'] > cut_threshold or i < min_candidates):
                    # Only include if it's actually a reasonable candidate
                    # (low conformance OR high re-draftability)
                    if (card_data['conformance'] < 0.6 or card_data['re_draftability'] > 0.7):
                        cut_candidates.append(card_data['card_id'])
                        
                        # Limit to top 5 candidates to avoid overwhelming the user
                        if len(cut_candidates) >= 5:
                            break
            
        except Exception as e:
            self.logger.error(f"Error identifying cut candidates: {e}")
        
        return cut_candidates
    
    def _analyze_curve(self, deck_state: DeckState) -> Dict[str, Any]:
        """Analyze mana curve efficiency."""
        curve_stats = {
            "average_cost": 0.0,
            "curve_smoothness": 0.0,
            "early_game_density": 0.0,
            "late_game_presence": 0.0
        }
        
        if deck_state.drafted_cards:
            # Calculate basic curve statistics
            total_cost = sum(deck_state.mana_curve.get(cost, 0) * cost 
                           for cost in range(11))
            total_cards = len(deck_state.drafted_cards)
            curve_stats["average_cost"] = total_cost / total_cards if total_cards > 0 else 0
        
        return curve_stats
    
    def _get_phase_advice(self, deck_state: DeckState) -> str:
        """Get advice based on current draft phase."""
        if deck_state.pick_number <= 10:
            return "Focus on curve and strong individual cards"
        elif deck_state.pick_number <= 20:
            return "Solidify archetype and fill strategic gaps"
        else:
            return "Fine-tune curve and consider tech choices"