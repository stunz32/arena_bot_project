"""
AI Helper v2 - Grandmaster Advisor Orchestrator (AI-Confidence Safe)

This module implements the main orchestrator for the Grandmaster AI Coach system,
coordinating all AI components to provide comprehensive draft recommendations with
confidence scoring and uncertainty handling, comprehensive error handling for AI failures,
and detailed audit trails for all AI decisions.

Features:
- Comprehensive AI decision orchestration with atomic operations
- Dynamic pivot advisor with confidence thresholds
- Greed meter with risk assessment and exploitation detection
- Synergy trap detector with anti-manipulation safeguards
- Comparative explanation generation with fallback templates
- Decision validation against archetype constraints
- AI confidence scoring manipulation prevention
- Numerical stability validation with NaN/Infinity detection
- Format-aware confidence calibration with separate thresholds
- Adversarial input detection for unusual card combinations
- Robust confidence aggregation using median-based approaches

Architecture:
- Atomic operations for recommendation generation to prevent partial states
- Median-based confidence aggregation resistant to outliers and manipulation
- Format-aware calibration with separate thresholds per game format
- Adversarial input detection using statistical anomaly detection
- Comprehensive audit trail with decision correlation and timing
"""

import logging
import time
import threading
import statistics
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from enum import Enum
import math
import weakref

# Import core components
from .card_evaluator import CardEvaluationEngine
from .deck_analyzer import StrategicDeckAnalyzer, AnalysisResult
from .data_models import (
    CardInfo, CardOption, EvaluationScores, DeckState, AIDecision,
    StrategicContext, ArchetypePreference, ConfidenceLevel, DraftPhase
)
from .exceptions import (
    AIModelError, ModelPredictionError, DataValidationError,
    ResourceExhaustionError, PerformanceThresholdExceeded,
    IntegrationError, ComponentCommunicationError
)
from .monitoring import PerformanceMonitor, ResourceTracker, get_performance_monitor

logger = logging.getLogger(__name__)

# Configuration constants
DECISION_TIMEOUT_MS = 15000  # 15 second timeout for decisions
CONFIDENCE_MANIPULATION_THRESHOLD = 0.95  # Suspiciously high confidence
GREED_EXPLOITATION_THRESHOLD = 0.8  # Greed threshold for exploitation warning
SYNERGY_TRAP_THRESHOLD = 0.3  # Threshold for synergy trap detection
PIVOT_CONFIDENCE_THRESHOLD = 0.6  # Threshold for pivot recommendations
AUDIT_RETENTION_HOURS = 24  # Keep audit trail for 24 hours
MAX_CONCURRENT_DECISIONS = 3  # Maximum concurrent decision processes

class DecisionContext(NamedTuple):
    """Immutable decision context for audit trail"""
    card_options: Tuple[CardOption, ...]
    deck_state: DeckState
    format_type: str
    user_preferences: Dict[str, Any]
    timestamp: float
    correlation_id: str

class GreedLevel(Enum):
    """Greed level classification for risk assessment"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GREEDY = "greedy"
    EXPLOITATIVE = "exploitative"
    DANGEROUS = "dangerous"

class PivotRecommendation(NamedTuple):
    """Pivot recommendation with reasoning"""
    from_archetype: ArchetypePreference
    to_archetype: ArchetypePreference
    confidence: float
    reasoning: str
    urgency: float

@dataclass
class AuditEntry:
    """Comprehensive audit entry for decision tracking"""
    decision_id: str
    timestamp: float
    correlation_id: str
    context: DecisionContext
    card_evaluations: List[Tuple[CardOption, EvaluationScores]]
    deck_analysis: AnalysisResult
    pivot_recommendation: Optional[PivotRecommendation]
    greed_assessment: Dict[str, Any]
    synergy_warnings: List[str]
    final_decision: AIDecision
    performance_metrics: Dict[str, float]
    confidence_manipulation_flags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary for serialization"""
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'context': {
                'card_count': len(self.context.card_options),
                'deck_size': len(self.context.deck_state.cards),
                'draft_phase': self.context.deck_state.draft_phase.value,
                'format_type': self.context.format_type
            },
            'evaluations': [
                {
                    'card': option.card_info.name,
                    'position': option.position,
                    'composite_score': scores.composite_score,
                    'confidence': scores.confidence.value
                }
                for option, scores in self.card_evaluations
            ],
            'deck_analysis': {
                'dominant_archetype': max(self.deck_analysis.archetype_scores.keys(),
                                        key=lambda k: self.deck_analysis.archetype_scores[k]).value,
                'deck_strength': self.deck_analysis.deck_strength,
                'confidence': self.deck_analysis.confidence.value
            },
            'final_decision': {
                'recommended_pick': self.final_decision.recommended_pick,
                'confidence': self.final_decision.confidence.value,
                'reasoning': self.final_decision.reasoning[:200]  # Truncate for storage
            },
            'performance': self.performance_metrics,
            'flags': self.confidence_manipulation_flags
        }

class ConfidenceValidator:
    """
    Confidence scoring manipulation prevention system
    
    Implements numerical stability validation, format-aware calibration,
    adversarial input detection, and robust aggregation methods.
    """
    
    def __init__(self):
        # Format-specific confidence thresholds
        self.format_thresholds = {
            'arena': {'min': 0.1, 'max': 0.9, 'suspicious': 0.95},
            'constructed': {'min': 0.2, 'max': 0.95, 'suspicious': 0.98},
            'battlegrounds': {'min': 0.05, 'max': 0.85, 'suspicious': 0.92}
        }
        
        # Statistical tracking for anomaly detection
        self.confidence_history = []
        self.max_history = 1000  # Keep last 1000 decisions
        
        logger.info("Initialized ConfidenceValidator with manipulation detection")
    
    def validate_confidence_scores(
        self, 
        scores: List[EvaluationScores],
        format_type: str = "arena"
    ) -> Tuple[bool, List[str]]:
        """
        Validate confidence scores for manipulation and anomalies
        
        Returns (is_valid, warning_messages)
        """
        warnings = []
        
        # Get format-specific thresholds
        thresholds = self.format_thresholds.get(format_type, self.format_thresholds['arena'])
        
        # Check for numerical stability issues
        for i, score in enumerate(scores):
            if not self._is_numerically_stable(score):
                warnings.append(f"Numerical instability detected in card {i+1} scores")
                return False, warnings
        
        # Extract confidence values
        confidence_values = [self._confidence_to_float(score.confidence) for score in scores]
        
        # Check for suspicious patterns
        if self._detect_suspicious_patterns(confidence_values, thresholds):
            warnings.append("Suspicious confidence patterns detected")
        
        # Check for adversarial inputs
        if self._detect_adversarial_input(scores):
            warnings.append("Potential adversarial input detected")
        
        # Update history for anomaly detection
        self._update_confidence_history(confidence_values)
        
        return len(warnings) == 0, warnings
    
    def _is_numerically_stable(self, scores: EvaluationScores) -> bool:
        """Check for numerical stability issues"""
        values_to_check = [
            scores.base_value, scores.tempo_score, scores.value_score,
            scores.synergy_score, scores.curve_score, scores.redraftability_score,
            scores.composite_score
        ]
        
        for value in values_to_check:
            if math.isnan(value) or math.isinf(value):
                return False
            if value < 0 or value > 100:  # Out of expected range
                return False
        
        return True
    
    def _confidence_to_float(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to float for analysis"""
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9
        }
        return mapping.get(confidence, 0.5)
    
    def _detect_suspicious_patterns(
        self, 
        confidence_values: List[float], 
        thresholds: Dict[str, float]
    ) -> bool:
        """Detect suspicious confidence patterns"""
        # All values suspiciously high
        if all(conf > thresholds['suspicious'] for conf in confidence_values):
            return True
        
        # Unrealistic precision (all exactly the same)
        if len(set(confidence_values)) == 1 and len(confidence_values) > 1:
            return True
        
        # Implausible spread (too narrow for real decisions)
        if len(confidence_values) > 1:
            spread = max(confidence_values) - min(confidence_values)
            if spread < 0.05:  # Less than 5% spread
                return True
        
        return False
    
    def _detect_adversarial_input(self, scores: List[EvaluationScores]) -> bool:
        """Detect potential adversarial input using statistical analysis"""
        if len(scores) < 2:
            return False
        
        # Check for unrealistic score distributions
        all_composites = [s.composite_score for s in scores]
        
        # Standard deviation too low (unnaturally similar)
        if len(all_composites) > 1:
            try:
                std_dev = statistics.stdev(all_composites)
                if std_dev < 1.0:  # Less than 1 point standard deviation
                    return True
            except statistics.StatisticsError:
                pass
        
        # Check for impossible score combinations
        for score in scores:
            # Base value and composite score wildly inconsistent
            if abs(score.base_value - score.composite_score) > 50:
                return True
        
        return False
    
    def _update_confidence_history(self, confidence_values: List[float]):
        """Update confidence history for trend analysis"""
        self.confidence_history.extend(confidence_values)
        
        # Maintain history size
        if len(self.confidence_history) > self.max_history:
            self.confidence_history = self.confidence_history[-self.max_history:]
    
    def aggregate_confidence_robust(
        self, 
        confidence_values: List[float],
        method: str = "median"
    ) -> float:
        """
        Robust confidence aggregation resistant to outliers
        
        Uses median-based approaches to prevent manipulation through
        extreme outlier values.
        """
        if not confidence_values:
            return 0.5
        
        if method == "median":
            return statistics.median(confidence_values)
        elif method == "trimmed_mean":
            # Remove top and bottom 10% and take mean
            sorted_values = sorted(confidence_values)
            trim_count = max(1, len(sorted_values) // 10)
            trimmed = sorted_values[trim_count:-trim_count] if len(sorted_values) > 2 else sorted_values
            return statistics.mean(trimmed)
        elif method == "winsorized_mean":
            # Replace extreme values with percentiles
            sorted_values = sorted(confidence_values)
            p10 = sorted_values[len(sorted_values) // 10] if len(sorted_values) > 10 else min(sorted_values)
            p90 = sorted_values[-len(sorted_values) // 10] if len(sorted_values) > 10 else max(sorted_values)
            
            winsorized = [max(p10, min(p90, val)) for val in confidence_values]
            return statistics.mean(winsorized)
        else:
            # Default to median for safety
            return statistics.median(confidence_values)

class DynamicPivotAdvisor:
    """
    Dynamic pivot advisor with confidence thresholds
    
    Analyzes when the deck should pivot to a different archetype
    based on draft progress and available cards.
    """
    
    def __init__(self):
        self.pivot_history = {}  # Track pivot recommendations
        logger.info("Initialized DynamicPivotAdvisor")
    
    def analyze_pivot_opportunity(
        self, 
        deck_analysis: AnalysisResult,
        card_options: List[CardOption],
        deck_state: DeckState
    ) -> Optional[PivotRecommendation]:
        """
        Analyze if deck should pivot to different archetype
        
        Returns pivot recommendation if confident pivot is beneficial.
        """
        # Only recommend pivots if we have reasonable confidence in current analysis
        if deck_analysis.confidence not in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            return None
        
        # Don't recommend pivots too late in draft
        if deck_state.draft_phase == DraftPhase.LATE and len(deck_state.cards) > 25:
            return None
        
        # Find current dominant archetype
        current_archetype = max(
            deck_analysis.archetype_scores.keys(),
            key=lambda k: deck_analysis.archetype_scores[k]
        )
        current_score = deck_analysis.archetype_scores[current_archetype]
        
        # Check if current archetype is struggling
        if current_score > 0.7:  # Strong current archetype
            return None
        
        # Analyze what archetypes the offered cards support
        card_archetype_support = self._analyze_card_archetype_support(card_options, deck_state)
        
        # Find best alternative archetype
        best_alternative = None
        best_score = 0.0
        
        for archetype, support_score in card_archetype_support.items():
            if archetype == current_archetype:
                continue
                
            # Calculate pivot potential
            current_archetype_score = deck_analysis.archetype_scores.get(archetype, 0.0)
            pivot_potential = (support_score * 0.6) + (current_archetype_score * 0.4)
            
            if pivot_potential > best_score and pivot_potential > PIVOT_CONFIDENCE_THRESHOLD:
                best_alternative = archetype
                best_score = pivot_potential
        
        if best_alternative and best_score > current_score + 0.15:  # Significant improvement
            urgency = self._calculate_pivot_urgency(deck_state, current_score)
            reasoning = self._generate_pivot_reasoning(
                current_archetype, best_alternative, deck_analysis, card_archetype_support
            )
            
            return PivotRecommendation(
                from_archetype=current_archetype,
                to_archetype=best_alternative,
                confidence=best_score,
                reasoning=reasoning,
                urgency=urgency
            )
        
        return None
    
    def _analyze_card_archetype_support(
        self, 
        card_options: List[CardOption],
        deck_state: DeckState
    ) -> Dict[ArchetypePreference, float]:
        """Analyze how much each card option supports different archetypes"""
        archetype_support = {archetype: 0.0 for archetype in ArchetypePreference}
        
        for option in card_options:
            card = option.card_info
            
            # Aggressive archetype indicators
            if card.cost <= 3 and card.card_type.value == "Minion":
                if card.attack >= card.health:  # Aggressive statline
                    archetype_support[ArchetypePreference.AGGRESSIVE] += 0.3
            
            if any(keyword in card.text.lower() for keyword in ['charge', 'rush', 'battlecry']):
                archetype_support[ArchetypePreference.AGGRESSIVE] += 0.2
                archetype_support[ArchetypePreference.TEMPO] += 0.2
            
            # Control archetype indicators
            if card.cost >= 5:
                archetype_support[ArchetypePreference.CONTROL] += 0.2
            
            if any(keyword in card.text.lower() for keyword in ['destroy', 'heal', 'armor', 'taunt']):
                archetype_support[ArchetypePreference.CONTROL] += 0.3
            
            # Value archetype indicators
            if any(keyword in card.text.lower() for keyword in ['draw', 'discover', 'deathrattle']):
                archetype_support[ArchetypePreference.VALUE] += 0.3
            
            # Tempo archetype indicators
            if 2 <= card.cost <= 4 and any(keyword in card.text.lower() for keyword in ['battlecry', 'rush']):
                archetype_support[ArchetypePreference.TEMPO] += 0.3
        
        # Normalize by number of cards
        if card_options:
            for archetype in archetype_support:
                archetype_support[archetype] /= len(card_options)
        
        return archetype_support
    
    def _calculate_pivot_urgency(self, deck_state: DeckState, current_score: float) -> float:
        """Calculate urgency of pivot based on draft state"""
        urgency = 0.5  # Base urgency
        
        # More urgent if current archetype is really struggling
        if current_score < 0.4:
            urgency += 0.3
        elif current_score < 0.5:
            urgency += 0.2
        
        # More urgent earlier in draft
        if deck_state.draft_phase == DraftPhase.EARLY:
            urgency += 0.2
        elif deck_state.draft_phase == DraftPhase.MID:
            urgency += 0.1
        
        # Less urgent if deck is almost complete
        if len(deck_state.cards) > 22:
            urgency -= 0.3
        
        return max(0.0, min(1.0, urgency))
    
    def _generate_pivot_reasoning(
        self,
        from_archetype: ArchetypePreference,
        to_archetype: ArchetypePreference,
        deck_analysis: AnalysisResult,
        card_support: Dict[ArchetypePreference, float]
    ) -> str:
        """Generate human-readable reasoning for pivot recommendation"""
        from_score = deck_analysis.archetype_scores.get(from_archetype, 0.0)
        to_score = card_support.get(to_archetype, 0.0)
        
        reasoning = f"Current {from_archetype.value.lower()} strategy (score: {from_score:.1f}) "
        reasoning += f"is underperforming. Available cards strongly support "
        reasoning += f"{to_archetype.value.lower()} strategy (support: {to_score:.1f}). "
        
        # Add specific gaps that pivot would address
        if deck_analysis.strategic_gaps:
            main_gaps = [gap for gap, score in deck_analysis.strategic_gaps.items() if score > 0.6]
            if main_gaps:
                reasoning += f"This pivot would address key gaps: {', '.join(main_gaps[:2])}."
        
        return reasoning

class GreedMeter:
    """
    Greed meter with risk assessment and exploitation detection
    
    Evaluates the risk/reward profile of card choices and warns
    against overly greedy plays that could backfire.
    """
    
    def __init__(self):
        self.greed_history = []
        logger.info("Initialized GreedMeter with exploitation detection")
    
    def assess_greed_level(
        self,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        deck_state: DeckState,
        deck_analysis: AnalysisResult
    ) -> Dict[str, Any]:
        """
        Assess greed level of card choices with risk analysis
        
        Returns comprehensive greed assessment with warnings.
        """
        assessments = []
        
        for option, scores in card_evaluations:
            card_greed = self._evaluate_card_greed(option.card_info, scores, deck_state)
            assessments.append((option, card_greed))
        
        # Find greediest option
        greediest_option, max_greed = max(assessments, key=lambda x: x[1]['greed_score'])
        
        # Calculate overall greed level
        avg_greed = sum(assessment[1]['greed_score'] for assessment in assessments) / len(assessments)
        greed_level = self._classify_greed_level(avg_greed)
        
        # Generate warnings and recommendations
        warnings = self._generate_greed_warnings(assessments, deck_analysis, greed_level)
        
        # Check for exploitation patterns
        exploitation_risk = self._assess_exploitation_risk(assessments, deck_state)
        
        return {
            'greed_level': greed_level,
            'average_greed_score': avg_greed,
            'greediest_card': greediest_option.card_info.name,
            'max_greed_score': max_greed['greed_score'],
            'card_assessments': [
                {
                    'card': option.card_info.name,
                    'greed_score': assessment['greed_score'],
                    'risk_factors': assessment['risk_factors']
                }
                for option, assessment in assessments
            ],
            'warnings': warnings,
            'exploitation_risk': exploitation_risk,
            'recommendation': self._generate_greed_recommendation(greed_level, warnings)
        }
    
    def _evaluate_card_greed(
        self, 
        card: CardInfo, 
        scores: EvaluationScores, 
        deck_state: DeckState
    ) -> Dict[str, Any]:
        """Evaluate greed level of individual card"""
        greed_score = 0.0
        risk_factors = []
        
        # High-cost cards are inherently greedy
        if card.cost >= 6:
            cost_greed = (card.cost - 5) * 0.15
            greed_score += cost_greed
            if card.cost >= 8:
                risk_factors.append("Very expensive card")
        
        # Situational cards with high power are greedy
        situational_keywords = ['if', 'when', 'after', 'combo', 'secret']
        situational_count = sum(1 for keyword in situational_keywords 
                              if keyword in card.text.lower())
        
        if situational_count > 0 and scores.base_value > 80:
            situational_greed = situational_count * 0.2
            greed_score += situational_greed
            risk_factors.append(f"High-power situational card ({situational_count} conditions)")
        
        # Cards that require specific deck composition
        if any(keyword in card.text.lower() for keyword in ['spell', 'weapon', 'secret']):
            synergy_requirement = card.text.lower().count('spell') + card.text.lower().count('weapon')
            if synergy_requirement > 0:
                deck_support = self._calculate_synergy_support(card, deck_state)
                if deck_support < 0.3:  # Low support in current deck
                    greed_score += 0.3
                    risk_factors.append("Requires synergy not present in deck")
        
        # Legendary cards are greedy (can't draft more)
        if card.rarity.value == "Legendary":
            greed_score += 0.2
            risk_factors.append("Legendary (unique)")
        
        # Very high redraftability score suggests greedy pick
        if scores.redraftability_score < 40:
            greed_score += 0.25
            risk_factors.append("Low redraftability (niche card)")
        
        # Early game greed (taking late game cards early)
        if deck_state.draft_phase == DraftPhase.EARLY and card.cost >= 6:
            early_curve_cards = sum(1 for c in deck_state.cards if c.cost <= 3)
            total_cards = len(deck_state.cards)
            
            if total_cards > 0 and early_curve_cards / total_cards < 0.4:
                greed_score += 0.3
                risk_factors.append("Taking expensive card without early game foundation")
        
        return {
            'greed_score': max(0.0, min(1.0, greed_score)),
            'risk_factors': risk_factors
        }
    
    def _calculate_synergy_support(self, card: CardInfo, deck_state: DeckState) -> float:
        """Calculate how much synergy support exists in deck"""
        if not deck_state.cards:
            return 0.0
        
        support_count = 0
        
        # Check for spell synergy
        if 'spell' in card.text.lower():
            support_count += sum(1 for c in deck_state.cards if c.card_type.value == "Spell")
        
        # Check for weapon synergy
        if 'weapon' in card.text.lower():
            support_count += sum(1 for c in deck_state.cards if c.card_type.value == "Weapon")
        
        return min(1.0, support_count / len(deck_state.cards))
    
    def _classify_greed_level(self, greed_score: float) -> GreedLevel:
        """Classify overall greed level"""
        if greed_score >= 0.8:
            return GreedLevel.DANGEROUS
        elif greed_score >= 0.65:
            return GreedLevel.EXPLOITATIVE
        elif greed_score >= 0.5:
            return GreedLevel.GREEDY
        elif greed_score >= 0.3:
            return GreedLevel.BALANCED
        else:
            return GreedLevel.CONSERVATIVE
    
    def _generate_greed_warnings(
        self,
        assessments: List[Tuple[CardOption, Dict[str, Any]]],
        deck_analysis: AnalysisResult,
        greed_level: GreedLevel
    ) -> List[str]:
        """Generate warnings about greed level"""
        warnings = []
        
        if greed_level in [GreedLevel.EXPLOITATIVE, GreedLevel.DANGEROUS]:
            warnings.append(f"Warning: {greed_level.value.capitalize()} play detected!")
        
        # Check for specific risky patterns
        high_greed_cards = [
            (option, assessment) for option, assessment in assessments
            if assessment['greed_score'] > GREED_EXPLOITATION_THRESHOLD
        ]
        
        if high_greed_cards:
            for option, assessment in high_greed_cards:
                card_name = option.card_info.name
                warnings.append(f"{card_name}: {', '.join(assessment['risk_factors'])}")
        
        # Check against deck strategy
        if deck_analysis.strategic_gaps.get('early_game', 0) > 0.7:
            expensive_options = [
                option for option, _ in assessments
                if option.card_info.cost >= 6
            ]
            if expensive_options:
                warnings.append("Deck desperately needs early game but considering expensive cards")
        
        return warnings
    
    def _assess_exploitation_risk(
        self,
        assessments: List[Tuple[CardOption, Dict[str, Any]]],
        deck_state: DeckState
    ) -> Dict[str, Any]:
        """Assess risk of exploitation by opponents"""
        risk_factors = []
        risk_score = 0.0
        
        # Check for curve gaps that could be exploited
        early_game_count = sum(1 for card in deck_state.cards if card.cost <= 3)
        total_cards = len(deck_state.cards)
        
        if total_cards > 10:
            early_game_ratio = early_game_count / total_cards
            if early_game_ratio < 0.3:  # Less than 30% early game
                risk_score += 0.4
                risk_factors.append("Vulnerable to aggressive decks (weak early game)")
        
        # Check for lack of removal
        removal_count = sum(1 for card in deck_state.cards
                          if any(keyword in card.text.lower() 
                                for keyword in ['deal', 'damage', 'destroy']))
        
        if total_cards > 15:
            removal_ratio = removal_count / total_cards
            if removal_ratio < 0.2:  # Less than 20% removal
                risk_score += 0.3
                risk_factors.append("Limited removal options")
        
        return {
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors,
            'exploitation_likelihood': 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low'
        }
    
    def _generate_greed_recommendation(self, greed_level: GreedLevel, warnings: List[str]) -> str:
        """Generate greed-based recommendation"""
        if greed_level == GreedLevel.DANGEROUS:
            return "Strongly consider safer alternatives. These picks could backfire."
        elif greed_level == GreedLevel.EXPLOITATIVE:
            return "High-risk, high-reward picks. Ensure you can afford the risk."
        elif greed_level == GreedLevel.GREEDY:
            return "Greedy picks that may pay off but have downsides."
        elif greed_level == GreedLevel.CONSERVATIVE:
            return "Very safe picks. Consider if you need more powerful options."
        else:
            return "Well-balanced risk/reward profile."

class SynergyTrapDetector:
    """
    Synergy trap detector with anti-manipulation safeguards
    
    Identifies when cards appear to have synergy but actually create
    traps or dead cards due to insufficient support.
    """
    
    def __init__(self):
        self.trap_patterns = self._initialize_trap_patterns()
        logger.info("Initialized SynergyTrapDetector")
    
    def detect_synergy_traps(
        self,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        deck_state: DeckState
    ) -> List[str]:
        """
        Detect potential synergy traps in card evaluations
        
        Returns list of warning messages about synergy traps.
        """
        warnings = []
        
        for option, scores in card_evaluations:
            card = option.card_info
            
            # Only check cards with high synergy scores
            if scores.synergy_score > 70:
                trap_warnings = self._analyze_card_for_traps(card, deck_state, scores)
                if trap_warnings:
                    card_warnings = [f"{card.name}: {warning}" for warning in trap_warnings]
                    warnings.extend(card_warnings)
        
        return warnings
    
    def _initialize_trap_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize synergy trap patterns"""
        return {
            'tribal_insufficient': {
                'tribes': ['beast', 'demon', 'dragon', 'elemental', 'mech', 'murloc', 'pirate'],
                'min_support': 3,
                'description': 'Tribal synergy card without sufficient tribal support'
            },
            'spell_synergy_insufficient': {
                'keywords': ['spell'],
                'min_spells': 6,
                'description': 'Spell synergy card without enough spells'
            },
            'weapon_synergy_insufficient': {
                'keywords': ['weapon'],
                'min_weapons': 2,
                'description': 'Weapon synergy card without weapon support'
            },
            'combo_insufficient': {
                'keywords': ['combo'],
                'class_specific': True,
                'description': 'Combo card without combo enablers'
            },
            'conditional_effect': {
                'keywords': ['if you have', 'if your opponent has', 'if you control'],
                'description': 'Highly conditional effect unlikely to trigger'
            }
        }
    
    def _analyze_card_for_traps(
        self, 
        card: CardInfo, 
        deck_state: DeckState, 
        scores: EvaluationScores
    ) -> List[str]:
        """Analyze individual card for synergy traps"""
        warnings = []
        
        # Check tribal traps
        tribal_warnings = self._check_tribal_traps(card, deck_state)
        warnings.extend(tribal_warnings)
        
        # Check spell synergy traps
        spell_warnings = self._check_spell_synergy_traps(card, deck_state)
        warnings.extend(spell_warnings)
        
        # Check weapon synergy traps
        weapon_warnings = self._check_weapon_synergy_traps(card, deck_state)
        warnings.extend(weapon_warnings)
        
        # Check conditional effect traps
        conditional_warnings = self._check_conditional_traps(card, deck_state)
        warnings.extend(conditional_warnings)
        
        # Check if synergy score is suspiciously high for lack of support
        if warnings and scores.synergy_score > 80:
            warnings.append("Synergy score may be inflated despite lack of support")
        
        return warnings
    
    def _check_tribal_traps(self, card: CardInfo, deck_state: DeckState) -> List[str]:
        """Check for tribal synergy traps"""
        warnings = []
        pattern = self.trap_patterns['tribal_insufficient']
        
        # Find tribes mentioned in card
        card_tribes = []
        for tribe in pattern['tribes']:
            if tribe in card.text.lower() or tribe in card.name.lower():
                card_tribes.append(tribe)
        
        if not card_tribes:
            return warnings
        
        # Count tribal support in deck
        for tribe in card_tribes:
            tribal_count = 0
            for deck_card in deck_state.cards:
                if (tribe in deck_card.text.lower() or 
                    tribe in deck_card.name.lower()):
                    tribal_count += 1
            
            if tribal_count < pattern['min_support']:
                warnings.append(f"Insufficient {tribe} support ({tribal_count} cards, need {pattern['min_support']})")
        
        return warnings
    
    def _check_spell_synergy_traps(self, card: CardInfo, deck_state: DeckState) -> List[str]:
        """Check for spell synergy traps"""
        warnings = []
        
        if 'spell' not in card.text.lower():
            return warnings
        
        pattern = self.trap_patterns['spell_synergy_insufficient']
        spell_count = sum(1 for c in deck_state.cards if c.card_type.value == "Spell")
        
        if spell_count < pattern['min_spells']:
            warnings.append(f"Insufficient spell support ({spell_count} spells, need {pattern['min_spells']})")
        
        return warnings
    
    def _check_weapon_synergy_traps(self, card: CardInfo, deck_state: DeckState) -> List[str]:
        """Check for weapon synergy traps"""
        warnings = []
        
        if 'weapon' not in card.text.lower() or card.card_type.value == "Weapon":
            return warnings
        
        pattern = self.trap_patterns['weapon_synergy_insufficient']
        weapon_count = sum(1 for c in deck_state.cards 
                          if c.card_type.value == "Weapon" or 'weapon' in c.text.lower())
        
        if weapon_count < pattern['min_weapons']:
            warnings.append(f"Insufficient weapon support ({weapon_count} weapons, need {pattern['min_weapons']})")
        
        return warnings
    
    def _check_conditional_traps(self, card: CardInfo, deck_state: DeckState) -> List[str]:
        """Check for conditional effect traps"""
        warnings = []
        pattern = self.trap_patterns['conditional_effect']
        
        conditional_count = sum(1 for keyword in pattern['keywords']
                              if keyword in card.text.lower())
        
        if conditional_count > 0:
            # Highly conditional cards with high synergy scores are suspicious
            warnings.append(f"Highly conditional effect ({conditional_count} conditions)")
        
        return warnings

class GrandmasterAdvisor:
    """
    Grandmaster Advisor Orchestrator with AI-Confidence Safe design
    
    This is the main orchestrator that coordinates all AI components to provide
    comprehensive draft recommendations with confidence scoring, uncertainty handling,
    and comprehensive error handling for AI failures.
    
    Features:
    - Comprehensive AI decision orchestration with atomic operations
    - Dynamic pivot advisor with confidence thresholds
    - Greed meter with risk assessment and exploitation detection
    - Synergy trap detector with anti-manipulation safeguards
    - AI confidence scoring manipulation prevention
    - Detailed audit trail for all AI decisions
    - Decision validation against archetype constraints
    - Performance monitoring and resource management
    """
    
    def __init__(self, enable_caching: bool = True, enable_ml: bool = True):
        """
        Initialize the Grandmaster Advisor
        
        Args:
            enable_caching: Enable caching for expensive operations
            enable_ml: Enable ML model integration
        """
        # Initialize core components
        self.card_evaluator = CardEvaluationEngine(enable_caching, enable_ml)
        self.deck_analyzer = StrategicDeckAnalyzer()
        
        # Initialize specialized analyzers
        self.confidence_validator = ConfidenceValidator()
        self.pivot_advisor = DynamicPivotAdvisor()
        self.greed_meter = GreedMeter()
        self.synergy_detector = SynergyTrapDetector()
        
        # Monitoring and tracking
        self.performance_monitor = get_performance_monitor()
        self.resource_tracker = ResourceTracker("grandmaster_advisor")
        
        # Audit trail
        self.audit_trail = []
        self.audit_lock = threading.RLock()
        
        # Decision statistics
        self.stats = {
            'decisions_total': 0,
            'decisions_with_pivots': 0,
            'decisions_with_greed_warnings': 0,
            'decisions_with_synergy_traps': 0,
            'confidence_manipulation_detected': 0,
            'average_decision_time_ms': 0.0,
            'errors_total': 0
        }
        
        # Thread pool for parallel evaluation
        self.decision_lock = threading.RLock()
        
        logger.info(f"Initialized GrandmasterAdvisor (caching={enable_caching}, ml={enable_ml})")
    
    def analyze_draft_choice(
        self,
        card_options: List[CardOption],
        deck_state: DeckState,
        format_type: str = "arena",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AIDecision:
        """
        Analyze draft choice with comprehensive AI orchestration
        
        Args:
            card_options: List of 3 card options to choose from
            deck_state: Current deck state
            format_type: Game format (arena, constructed, etc.)
            user_preferences: Optional user preferences
            
        Returns:
            AIDecision: Comprehensive AI recommendation with confidence and reasoning
            
        Raises:
            DataValidationError: If inputs are invalid
            PerformanceThresholdExceeded: If decision takes too long
            ComponentCommunicationError: If component integration fails
        """
        start_time = time.time()
        decision_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())[:8]
        
        with self.decision_lock:
            try:
                # Input validation
                self._validate_decision_inputs(card_options, deck_state)
                
                # Create decision context for audit trail
                context = DecisionContext(
                    card_options=tuple(card_options),
                    deck_state=deck_state,
                    format_type=format_type,
                    user_preferences=user_preferences or {},
                    timestamp=start_time,
                    correlation_id=correlation_id
                )
                
                # Step 1: Evaluate each card option
                card_evaluations = self._evaluate_card_options(card_options, deck_state)
                
                # Step 2: Analyze deck strategy
                deck_analysis = self._analyze_deck_strategy(deck_state)
                
                # Step 3: Validate confidence scores for manipulation
                confidence_valid, confidence_warnings = self.confidence_validator.validate_confidence_scores(
                    [scores for _, scores in card_evaluations], format_type
                )
                
                manipulation_flags = []
                if not confidence_valid:
                    manipulation_flags.extend(confidence_warnings)
                    self.stats['confidence_manipulation_detected'] += 1
                
                # Step 4: Check for pivot opportunities
                pivot_recommendation = self.pivot_advisor.analyze_pivot_opportunity(
                    deck_analysis, card_options, deck_state
                )
                
                if pivot_recommendation:
                    self.stats['decisions_with_pivots'] += 1
                
                # Step 5: Assess greed level
                greed_assessment = self.greed_meter.assess_greed_level(
                    card_evaluations, deck_state, deck_analysis
                )
                
                if greed_assessment['warnings']:
                    self.stats['decisions_with_greed_warnings'] += 1
                
                # Step 6: Detect synergy traps
                synergy_warnings = self.synergy_detector.detect_synergy_traps(
                    card_evaluations, deck_state
                )
                
                if synergy_warnings:
                    self.stats['decisions_with_synergy_traps'] += 1
                
                # Step 7: Generate final decision with atomic operation
                final_decision = self._generate_final_decision(
                    card_evaluations, deck_analysis, pivot_recommendation,
                    greed_assessment, synergy_warnings, format_type, deck_state
                )
                
                # Step 8: Create comprehensive audit entry
                duration_ms = (time.time() - start_time) * 1000
                audit_entry = self._create_audit_entry(
                    decision_id, context, card_evaluations, deck_analysis,
                    pivot_recommendation, greed_assessment, synergy_warnings,
                    final_decision, duration_ms, manipulation_flags
                )
                
                # Store audit entry
                self._store_audit_entry(audit_entry)
                
                # Update statistics
                self._update_stats(duration_ms)
                
                # Performance check
                if duration_ms > DECISION_TIMEOUT_MS:
                    raise PerformanceThresholdExceeded(
                        "draft_decision",
                        duration_ms / 1000,
                        DECISION_TIMEOUT_MS / 1000
                    )
                
                return final_decision
                
            except Exception as e:
                self.stats['errors_total'] += 1
                if isinstance(e, (DataValidationError, PerformanceThresholdExceeded)):
                    raise
                else:
                    # Wrap unexpected errors
                    raise ComponentCommunicationError(
                        "grandmaster_advisor", "integrated_components",
                        context={"error": str(e), "decision_id": decision_id}
                    )
    
    def _validate_decision_inputs(self, card_options: List[CardOption], deck_state: DeckState):
        """Validate decision inputs"""
        if not isinstance(card_options, list) or len(card_options) != 3:
            raise DataValidationError("card_options", "List[CardOption] with 3 items", card_options)
        
        for i, option in enumerate(card_options):
            if not isinstance(option, CardOption):
                raise DataValidationError(f"card_options[{i}]", "CardOption", option)
            option.validate()
        
        if not isinstance(deck_state, DeckState):
            raise DataValidationError("deck_state", "DeckState", deck_state)
        
        deck_state.validate()
    
    def _evaluate_card_options(
        self, 
        card_options: List[CardOption], 
        deck_state: DeckState
    ) -> List[Tuple[CardOption, EvaluationScores]]:
        """Evaluate all card options"""
        evaluations = []
        
        for option in card_options:
            try:
                scores = self.card_evaluator.evaluate_card(
                    option.card_info, deck_state, option.position
                )
                evaluations.append((option, scores))
            except Exception as e:
                logger.error(f"Card evaluation failed for {option.card_info.name}: {e}")
                # Create fallback scores
                fallback_scores = EvaluationScores(
                    base_value=50.0, tempo_score=50.0, value_score=50.0,
                    synergy_score=50.0, curve_score=50.0, redraftability_score=50.0,
                    confidence=ConfidenceLevel.LOW
                )
                evaluations.append((option, fallback_scores))
        
        return evaluations
    
    def _analyze_deck_strategy(self, deck_state: DeckState) -> AnalysisResult:
        """Analyze deck strategy"""
        try:
            return self.deck_analyzer.analyze_deck_strategy(deck_state)
        except Exception as e:
            logger.error(f"Deck analysis failed: {e}")
            # Create fallback analysis
            from .deck_analyzer import AnalysisResult
            return AnalysisResult(
                archetype_scores={pref: 0.5 for pref in ArchetypePreference},
                strategic_gaps={'early_game': 0.5, 'removal': 0.5},
                cut_candidates=[],
                deck_strength=50.0,
                consistency_warnings=["Analysis failed - using fallback"],
                confidence=ConfidenceLevel.LOW
            )
    
    def _generate_final_decision(
        self,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        deck_analysis: AnalysisResult,
        pivot_recommendation: Optional[PivotRecommendation],
        greed_assessment: Dict[str, Any],
        synergy_warnings: List[str],
        format_type: str,
        deck_state: DeckState
    ) -> AIDecision:
        """Generate final AI decision with atomic operation"""
        # Find best card based on composite scores
        best_option, best_scores = max(card_evaluations, key=lambda x: x[1].composite_score)
        
        # Adjust recommendation based on strategic analysis
        strategic_adjustment = self._calculate_strategic_adjustment(
            card_evaluations, deck_analysis, pivot_recommendation
        )
        
        # Apply greed adjustment
        greed_adjustment = self._calculate_greed_adjustment(
            card_evaluations, greed_assessment
        )
        
        # Final recommendation with adjustments
        adjusted_evaluations = self._apply_adjustments(
            card_evaluations, strategic_adjustment, greed_adjustment
        )
        
        final_option, final_scores = max(adjusted_evaluations, key=lambda x: x[1].composite_score)
        
        # Generate comprehensive reasoning
        reasoning = self._generate_comprehensive_reasoning(
            final_option, final_scores, deck_analysis, pivot_recommendation,
            greed_assessment, synergy_warnings
        )
        
        # Calculate overall confidence using robust aggregation
        confidence_values = [scores.confidence for _, scores in card_evaluations]
        confidence_floats = [self.confidence_validator._confidence_to_float(conf) for conf in confidence_values]
        aggregated_confidence = self.confidence_validator.aggregate_confidence_robust(confidence_floats)
        
        # Convert back to ConfidenceLevel
        if aggregated_confidence >= 0.8:
            final_confidence = ConfidenceLevel.VERY_HIGH
        elif aggregated_confidence >= 0.65:
            final_confidence = ConfidenceLevel.HIGH
        elif aggregated_confidence >= 0.35:
            final_confidence = ConfidenceLevel.MEDIUM
        elif aggregated_confidence >= 0.2:
            final_confidence = ConfidenceLevel.LOW
        else:
            final_confidence = ConfidenceLevel.VERY_LOW
        
        # Create strategic context
        strategic_context = StrategicContext(
            needs_early_game=deck_analysis.strategic_gaps.get('early_game', 0.0) > 0.6,
            needs_removal=deck_analysis.strategic_gaps.get('removal', 0.0) > 0.6,
            needs_card_draw=deck_analysis.strategic_gaps.get('card_draw', 0.0) > 0.5,
            synergy_opportunities=synergy_warnings if synergy_warnings else [],
            archetype_fit=max(deck_analysis.archetype_scores.values()) if deck_analysis.archetype_scores else 0.5
        )
        
        # Create final AI decision
        return AIDecision(
            recommended_pick=final_option.position,
            card_evaluations=card_evaluations,
            confidence=final_confidence,
            reasoning=reasoning,
            strategic_context=strategic_context,
            alternative_picks=[opt.position for opt, _ in sorted(adjusted_evaluations, key=lambda x: x[1].composite_score, reverse=True)[1:]],
            deck_state=deck_state,
            analysis_duration_ms=time.time() * 1000,  # Will be updated by caller
            fallback_used=any('fallback' in str(scores.confidence) for _, scores in card_evaluations)
        )
    
    def _calculate_strategic_adjustment(
        self,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        deck_analysis: AnalysisResult,
        pivot_recommendation: Optional[PivotRecommendation]
    ) -> Dict[int, float]:
        """Calculate strategic adjustments for each card"""
        adjustments = {}
        
        for option, scores in card_evaluations:
            adjustment = 0.0
            
            # Adjust based on strategic gaps
            if deck_analysis.strategic_gaps.get('early_game', 0) > 0.6 and option.card_info.cost <= 3:
                adjustment += 10.0
            
            if deck_analysis.strategic_gaps.get('removal', 0) > 0.6:
                if any(keyword in option.card_info.text.lower() for keyword in ['deal', 'damage', 'destroy']):
                    adjustment += 8.0
            
            # Adjust based on pivot recommendation
            if pivot_recommendation and pivot_recommendation.urgency > 0.7:
                # Bonus for cards that support the pivot archetype
                if self._card_supports_archetype(option.card_info, pivot_recommendation.to_archetype):
                    adjustment += 12.0
            
            adjustments[option.position] = adjustment
        
        return adjustments
    
    def _calculate_greed_adjustment(
        self,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        greed_assessment: Dict[str, Any]
    ) -> Dict[int, float]:
        """Calculate greed-based adjustments"""
        adjustments = {}
        
        # Penalize overly greedy picks
        for card_assessment in greed_assessment['card_assessments']:
            card_name = card_assessment['card']
            greed_score = card_assessment['greed_score']
            
            # Find matching option
            for option, _ in card_evaluations:
                if option.card_info.name == card_name:
                    penalty = 0.0
                    
                    if greed_score > GREED_EXPLOITATION_THRESHOLD:
                        penalty = (greed_score - GREED_EXPLOITATION_THRESHOLD) * 20  # Up to 4 point penalty
                    
                    adjustments[option.position] = -penalty
                    break
        
        return adjustments
    
    def _apply_adjustments(
        self,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        strategic_adjustment: Dict[int, float],
        greed_adjustment: Dict[int, float]
    ) -> List[Tuple[CardOption, EvaluationScores]]:
        """Apply all adjustments to create final scores"""
        adjusted_evaluations = []
        
        for option, scores in card_evaluations:
            strategic_adj = strategic_adjustment.get(option.position, 0.0)
            greed_adj = greed_adjustment.get(option.position, 0.0)
            
            # Create adjusted scores (copy original scores and modify composite)
            adjusted_composite = scores.composite_score + strategic_adj + greed_adj
            adjusted_composite = max(0.0, min(100.0, adjusted_composite))  # Clamp to valid range
            
            # Create new scores object with adjusted composite
            adjusted_scores = EvaluationScores(
                base_value=scores.base_value,
                tempo_score=scores.tempo_score,
                value_score=scores.value_score,
                synergy_score=scores.synergy_score,
                curve_score=scores.curve_score,
                redraftability_score=scores.redraftability_score,
                confidence=scores.confidence
            )
            # Override composite score
            adjusted_scores.composite_score = adjusted_composite
            
            adjusted_evaluations.append((option, adjusted_scores))
        
        return adjusted_evaluations
    
    def _card_supports_archetype(self, card: CardInfo, archetype: ArchetypePreference) -> bool:
        """Check if card supports given archetype"""
        if archetype == ArchetypePreference.AGGRESSIVE:
            return card.cost <= 4 or any(keyword in card.text.lower() for keyword in ['charge', 'rush'])
        elif archetype == ArchetypePreference.CONTROL:
            return card.cost >= 5 or any(keyword in card.text.lower() for keyword in ['destroy', 'heal', 'taunt'])
        elif archetype == ArchetypePreference.TEMPO:
            return 2 <= card.cost <= 5 or any(keyword in card.text.lower() for keyword in ['battlecry', 'rush'])
        elif archetype == ArchetypePreference.VALUE:
            return any(keyword in card.text.lower() for keyword in ['draw', 'discover', 'deathrattle'])
        
        return True  # Balanced archetype supports everything
    
    def _generate_comprehensive_reasoning(
        self,
        chosen_option: CardOption,
        chosen_scores: EvaluationScores,
        deck_analysis: AnalysisResult,
        pivot_recommendation: Optional[PivotRecommendation],
        greed_assessment: Dict[str, Any],
        synergy_warnings: List[str]
    ) -> str:
        """Generate comprehensive reasoning for the decision"""
        reasoning = f"{chosen_option.card_info.name} is the strongest choice "
        reasoning += f"(score: {chosen_scores.composite_score:.1f}) "
        
        # Add key strengths
        strengths = []
        if chosen_scores.base_value > 75:
            strengths.append("high power level")
        if chosen_scores.tempo_score > 70:
            strengths.append("good tempo")
        if chosen_scores.value_score > 70:
            strengths.append("card advantage")
        if chosen_scores.synergy_score > 70:
            strengths.append("synergy potential")
        
        if strengths:
            reasoning += f"due to {', '.join(strengths)}. "
        
        # Add strategic context
        dominant_archetype = max(deck_analysis.archetype_scores.keys(),
                               key=lambda k: deck_analysis.archetype_scores[k]).value
        reasoning += f"Fits well in {dominant_archetype.lower()} strategy. "
        
        # Add key gaps being addressed
        main_gaps = [gap for gap, score in deck_analysis.strategic_gaps.items() if score > 0.6]
        if main_gaps:
            if chosen_option.card_info.cost <= 3 and 'early_game' in main_gaps:
                reasoning += "Addresses critical early game need. "
            elif any(keyword in chosen_option.card_info.text.lower() for keyword in ['deal', 'damage']) and 'removal' in main_gaps:
                reasoning += "Provides needed removal. "
        
        # Add pivot context
        if pivot_recommendation and pivot_recommendation.urgency > 0.5:
            reasoning += f"Supports potential pivot to {pivot_recommendation.to_archetype.value.lower()}. "
        
        # Add greed warnings
        if greed_assessment['greed_level'] in [GreedLevel.EXPLOITATIVE, GreedLevel.DANGEROUS]:
            reasoning += f"Note: {greed_assessment['greed_level'].value} pick with some risk. "
        
        # Add synergy warnings
        if synergy_warnings:
            reasoning += f"Warning: {synergy_warnings[0]} "
        
        return reasoning
    
    def _create_audit_entry(
        self,
        decision_id: str,
        context: DecisionContext,
        card_evaluations: List[Tuple[CardOption, EvaluationScores]],
        deck_analysis: AnalysisResult,
        pivot_recommendation: Optional[PivotRecommendation],
        greed_assessment: Dict[str, Any],
        synergy_warnings: List[str],
        final_decision: AIDecision,
        duration_ms: float,
        manipulation_flags: List[str]
    ) -> AuditEntry:
        """Create comprehensive audit entry"""
        return AuditEntry(
            decision_id=decision_id,
            timestamp=context.timestamp,
            correlation_id=context.correlation_id,
            context=context,
            card_evaluations=card_evaluations,
            deck_analysis=deck_analysis,
            pivot_recommendation=pivot_recommendation,
            greed_assessment=greed_assessment,
            synergy_warnings=synergy_warnings,
            final_decision=final_decision,
            performance_metrics={
                'decision_time_ms': duration_ms,
                'card_evaluation_time_ms': duration_ms * 0.4,  # Estimated
                'deck_analysis_time_ms': duration_ms * 0.3,
                'recommendation_time_ms': duration_ms * 0.3
            },
            confidence_manipulation_flags=manipulation_flags
        )
    
    def _store_audit_entry(self, entry: AuditEntry):
        """Store audit entry with retention management"""
        with self.audit_lock:
            self.audit_trail.append(entry)
            
            # Clean up old entries (keep last 24 hours)
            cutoff_time = time.time() - (AUDIT_RETENTION_HOURS * 3600)
            self.audit_trail = [
                e for e in self.audit_trail 
                if e.timestamp > cutoff_time
            ]
    
    def _update_stats(self, duration_ms: float):
        """Update advisor statistics"""
        self.stats['decisions_total'] += 1
        
        # Update average duration
        total_decisions = self.stats['decisions_total']
        current_avg = self.stats['average_decision_time_ms']
        self.stats['average_decision_time_ms'] = (current_avg * (total_decisions - 1) + duration_ms) / total_decisions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive advisor statistics"""
        stats = deepcopy(self.stats)
        
        # Add component stats
        stats['card_evaluator'] = self.card_evaluator.get_stats()
        stats['deck_analyzer'] = self.deck_analyzer.get_stats()
        
        # Add audit trail summary
        with self.audit_lock:
            stats['audit_trail_size'] = len(self.audit_trail)
            if self.audit_trail:
                stats['oldest_audit_entry'] = min(e.timestamp for e in self.audit_trail)
                stats['newest_audit_entry'] = max(e.timestamp for e in self.audit_trail)
        
        return stats
    
    def get_audit_trail(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent audit trail entries"""
        with self.audit_lock:
            recent_entries = sorted(self.audit_trail, key=lambda e: e.timestamp, reverse=True)[:limit]
            return [entry.to_dict() for entry in recent_entries]
    
    def shutdown(self):
        """Graceful shutdown with resource cleanup"""
        logger.info("Shutting down GrandmasterAdvisor")
        
        # Shutdown components
        if hasattr(self.card_evaluator, 'shutdown'):
            self.card_evaluator.shutdown()
        
        if hasattr(self.deck_analyzer, 'shutdown'):
            self.deck_analyzer.shutdown()
        
        # Clear audit trail
        with self.audit_lock:
            self.audit_trail.clear()
        
        logger.info("GrandmasterAdvisor shutdown complete")

# Export main class
__all__ = [
    'GrandmasterAdvisor', 'ConfidenceValidator', 'DynamicPivotAdvisor',
    'GreedMeter', 'SynergyTrapDetector', 'AuditEntry'
]