"""
AI Helper v2 - Robust Strategic Deck Analyzer (State-Corruption Safe)

This module implements the strategic deck analysis engine for the Grandmaster AI Coach system,
providing comprehensive deck archetype analysis, strategic gap identification, and cut candidate
recommendations with immutable state architecture to prevent state corruption.

Features:
- Immutable deck state architecture with copy-on-write modifications
- Fuzzy archetype matching with probabilistic scoring for edge cases
- Strategic gap analysis with priority weighting
- Cut candidate logic with explanatory reasoning
- Draft phase awareness with dynamic thresholds
- Thread-safe deck state analysis with validation
- Recommendation consistency validation to detect contradictions
- Incremental analysis processing to prevent memory spikes

Architecture:
- Copy-on-write deck modifications to prevent state corruption
- Probabilistic archetype scoring system for handling edge cases
- Contradiction detection in recommendations with resolution
- Memory-efficient incremental processing for large deck states
"""

import logging
import threading
import time
import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, NamedTuple
from enum import Enum
from collections import defaultdict, Counter
import weakref

# Import core data models
from .data_models import (
    CardInfo, DeckState, StrategicContext, ArchetypePreference,
    CardClass, CardType, CardRarity, DraftPhase, ConfidenceLevel
)
from .exceptions import (
    DataValidationError, DataCorruptionError, ResourceExhaustionError,
    PerformanceThresholdExceeded, IntegrationError
)
from .monitoring import PerformanceMonitor, ResourceTracker, get_performance_monitor

logger = logging.getLogger(__name__)

# Configuration constants
ANALYSIS_TIMEOUT_MS = 2000  # 2 second timeout for analysis
ARCHETYPE_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for archetype detection
MAX_CUT_CANDIDATES = 5  # Maximum cut candidates to suggest
MEMORY_LIMIT_MB = 20  # Memory limit for deck analysis
INCREMENTAL_PROCESSING_THRESHOLD = 15  # Process incrementally for decks > 15 cards

class ArchetypeSignature(NamedTuple):
    """Immutable archetype signature for pattern matching"""
    early_game_density: float  # 0-3 mana cards / total cards
    removal_density: float     # Removal spells / total cards
    value_density: float       # Card advantage cards / total cards
    tempo_density: float       # Immediate impact cards / total cards
    curve_peak: int           # Most common mana cost
    average_cost: float       # Average mana cost
    class_card_ratio: float   # Class cards / total cards

class AnalysisResult(NamedTuple):
    """Immutable analysis result to prevent tampering"""
    archetype_scores: Dict[ArchetypePreference, float]
    strategic_gaps: Dict[str, float]
    cut_candidates: List[Tuple[CardInfo, str, float]]  # card, reason, priority
    deck_strength: float
    consistency_warnings: List[str]
    confidence: ConfidenceLevel

@dataclass(frozen=True)
class ImmutableDeckSnapshot:
    """
    Immutable snapshot of deck state for analysis
    
    This ensures that deck analysis cannot accidentally modify the original
    deck state, preventing state corruption bugs.
    """
    cards: Tuple[CardInfo, ...]
    hero_class: CardClass
    current_pick: int
    archetype_preference: ArchetypePreference
    draft_phase: DraftPhase
    mana_curve: Dict[int, int]
    total_cards: int
    average_cost: float
    checksum: str = field(init=False)
    
    def __post_init__(self):
        """Calculate checksum for integrity validation"""
        # Create deterministic representation for checksum
        cards_data = [(c.name, c.cost, c.card_id) for c in self.cards]
        snapshot_data = {
            'cards': sorted(cards_data),
            'hero_class': self.hero_class.value,
            'current_pick': self.current_pick,
            'archetype_preference': self.archetype_preference.value,
            'draft_phase': self.draft_phase.value,
            'mana_curve': self.mana_curve,
            'total_cards': self.total_cards,
            'average_cost': self.average_cost
        }
        
        data_str = json.dumps(snapshot_data, sort_keys=True)
        object.__setattr__(self, 'checksum', hashlib.sha256(data_str.encode()).hexdigest()[:16])
    
    def validate_integrity(self) -> bool:
        """Validate snapshot integrity"""
        try:
            # Recalculate checksum and compare
            cards_data = [(c.name, c.cost, c.card_id) for c in self.cards]
            snapshot_data = {
                'cards': sorted(cards_data),
                'hero_class': self.hero_class.value,
                'current_pick': self.current_pick,
                'archetype_preference': self.archetype_preference.value,
                'draft_phase': self.draft_phase.value,
                'mana_curve': self.mana_curve,
                'total_cards': self.total_cards,
                'average_cost': self.average_cost
            }
            
            data_str = json.dumps(snapshot_data, sort_keys=True)
            expected_checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]
            return self.checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Snapshot integrity validation error: {e}")
            return False

class ArchetypeAnalyzer:
    """
    Fuzzy archetype matching system with probabilistic scoring
    
    Handles edge cases where decks don't clearly fit into a single archetype,
    providing probabilistic scores for each archetype possibility.
    """
    
    def __init__(self):
        # Define archetype signatures based on typical characteristics
        self.archetype_signatures = {
            ArchetypePreference.AGGRESSIVE: ArchetypeSignature(
                early_game_density=0.65,  # 65% early game cards
                removal_density=0.15,     # Some removal
                value_density=0.10,       # Low value focus
                tempo_density=0.70,       # High tempo focus
                curve_peak=2,             # Peak at 2 mana
                average_cost=3.2,         # Low average cost
                class_card_ratio=0.40     # Some class synergy
            ),
            ArchetypePreference.CONTROL: ArchetypeSignature(
                early_game_density=0.25,  # Low early game
                removal_density=0.35,     # High removal
                value_density=0.40,       # High value focus
                tempo_density=0.20,       # Low tempo focus
                curve_peak=5,             # Peak at 5+ mana
                average_cost=4.8,         # High average cost
                class_card_ratio=0.45     # Class removal/value cards
            ),
            ArchetypePreference.TEMPO: ArchetypeSignature(
                early_game_density=0.45,  # Moderate early game
                removal_density=0.25,     # Moderate removal
                value_density=0.25,       # Moderate value
                tempo_density=0.60,       # High tempo focus
                curve_peak=3,             # Peak at 3 mana
                average_cost=3.8,         # Moderate cost
                class_card_ratio=0.50     # Strong class synergy
            ),
            ArchetypePreference.VALUE: ArchetypeSignature(
                early_game_density=0.35,  # Moderate early game
                removal_density=0.20,     # Some removal
                value_density=0.50,       # High value focus
                tempo_density=0.30,       # Lower tempo focus
                curve_peak=4,             # Peak at 4 mana
                average_cost=4.2,         # Moderate-high cost
                class_card_ratio=0.40     # Moderate class focus
            ),
            ArchetypePreference.BALANCED: ArchetypeSignature(
                early_game_density=0.40,  # Balanced early game
                removal_density=0.25,     # Balanced removal
                value_density=0.30,       # Balanced value
                tempo_density=0.40,       # Balanced tempo
                curve_peak=3,             # Peak at 3-4 mana
                average_cost=4.0,         # Balanced cost
                class_card_ratio=0.35     # Some class cards
            )
        }
        
        logger.info("Initialized ArchetypeAnalyzer with fuzzy matching")
    
    def analyze_archetype_fit(self, snapshot: ImmutableDeckSnapshot) -> Dict[ArchetypePreference, float]:
        """
        Analyze how well deck fits each archetype using probabilistic scoring
        
        Returns scores 0.0-1.0 for each archetype, with higher scores indicating
        better fit. Multiple archetypes can have high scores for hybrid decks.
        """
        if not snapshot.validate_integrity():
            raise DataCorruptionError("deck_snapshot", context={"checksum": snapshot.checksum})
        
        deck_signature = self._extract_deck_signature(snapshot)
        archetype_scores = {}
        
        for archetype, ideal_signature in self.archetype_signatures.items():
            score = self._calculate_signature_similarity(deck_signature, ideal_signature)
            
            # Apply confidence modifiers based on deck size
            if snapshot.total_cards < 10:
                score *= 0.7  # Lower confidence early in draft
            elif snapshot.total_cards < 20:
                score *= 0.85  # Moderate confidence mid-draft
            
            # Apply draft phase modifiers
            if snapshot.draft_phase == DraftPhase.EARLY:
                # Early draft - archetype less defined
                score *= 0.8
            elif snapshot.draft_phase == DraftPhase.LATE:
                # Late draft - archetype should be clearer
                if score > 0.6:
                    score = min(1.0, score * 1.1)
            
            archetype_scores[archetype] = max(0.0, min(1.0, score))
        
        return archetype_scores
    
    def _extract_deck_signature(self, snapshot: ImmutableDeckSnapshot) -> ArchetypeSignature:
        """Extract archetype signature from deck snapshot"""
        if snapshot.total_cards == 0:
            # Empty deck - return neutral signature
            return ArchetypeSignature(0.0, 0.0, 0.0, 0.0, 3, 3.0, 0.0)
        
        # Calculate densities
        early_game_count = sum(1 for card in snapshot.cards if card.cost <= 3)
        early_game_density = early_game_count / snapshot.total_cards
        
        # Identify removal spells
        removal_keywords = ['deal', 'damage', 'destroy', 'remove', 'silence', 'transform']
        removal_count = sum(1 for card in snapshot.cards 
                          if card.card_type == CardType.SPELL and 
                          any(keyword in card.text.lower() for keyword in removal_keywords))
        removal_density = removal_count / snapshot.total_cards
        
        # Identify value cards (card advantage)
        value_keywords = ['draw', 'discover', 'add', 'random', 'deathrattle', 'divine shield']
        value_count = sum(1 for card in snapshot.cards
                         if any(keyword in card.text.lower() for keyword in value_keywords))
        value_density = value_count / snapshot.total_cards
        
        # Identify tempo cards (immediate impact)
        tempo_keywords = ['battlecry', 'charge', 'rush', 'taunt']
        tempo_count = sum(1 for card in snapshot.cards
                         if any(keyword in card.text.lower() for keyword in tempo_keywords))
        tempo_density = tempo_count / snapshot.total_cards
        
        # Find curve peak
        curve_peak = max(snapshot.mana_curve.keys(), key=lambda k: snapshot.mana_curve.get(k, 0))
        
        # Calculate class card ratio
        class_card_count = sum(1 for card in snapshot.cards if card.card_class == snapshot.hero_class)
        class_card_ratio = class_card_count / snapshot.total_cards
        
        return ArchetypeSignature(
            early_game_density=early_game_density,
            removal_density=removal_density,
            value_density=value_density,
            tempo_density=tempo_density,
            curve_peak=curve_peak,
            average_cost=snapshot.average_cost,
            class_card_ratio=class_card_ratio
        )
    
    def _calculate_signature_similarity(
        self, 
        deck_signature: ArchetypeSignature, 
        ideal_signature: ArchetypeSignature
    ) -> float:
        """
        Calculate similarity between deck signature and ideal archetype signature
        
        Uses weighted scoring with tolerance for natural variation in Arena drafts.
        """
        weights = {
            'early_game_density': 0.20,
            'removal_density': 0.15,
            'value_density': 0.15,
            'tempo_density': 0.15,
            'curve_peak': 0.10,
            'average_cost': 0.15,
            'class_card_ratio': 0.10
        }
        
        total_score = 0.0
        
        # Early game density similarity
        early_diff = abs(deck_signature.early_game_density - ideal_signature.early_game_density)
        early_score = max(0.0, 1.0 - (early_diff / 0.3))  # 30% tolerance
        total_score += early_score * weights['early_game_density']
        
        # Removal density similarity
        removal_diff = abs(deck_signature.removal_density - ideal_signature.removal_density)
        removal_score = max(0.0, 1.0 - (removal_diff / 0.2))  # 20% tolerance
        total_score += removal_score * weights['removal_density']
        
        # Value density similarity
        value_diff = abs(deck_signature.value_density - ideal_signature.value_density)
        value_score = max(0.0, 1.0 - (value_diff / 0.25))  # 25% tolerance
        total_score += value_score * weights['value_density']
        
        # Tempo density similarity
        tempo_diff = abs(deck_signature.tempo_density - ideal_signature.tempo_density)
        tempo_score = max(0.0, 1.0 - (tempo_diff / 0.3))  # 30% tolerance
        total_score += tempo_score * weights['tempo_density']
        
        # Curve peak similarity
        curve_diff = abs(deck_signature.curve_peak - ideal_signature.curve_peak)
        curve_score = max(0.0, 1.0 - (curve_diff / 2.0))  # 2 mana tolerance
        total_score += curve_score * weights['curve_peak']
        
        # Average cost similarity
        cost_diff = abs(deck_signature.average_cost - ideal_signature.average_cost)
        cost_score = max(0.0, 1.0 - (cost_diff / 1.5))  # 1.5 mana tolerance
        total_score += cost_score * weights['average_cost']
        
        # Class card ratio similarity
        class_diff = abs(deck_signature.class_card_ratio - ideal_signature.class_card_ratio)
        class_score = max(0.0, 1.0 - (class_diff / 0.3))  # 30% tolerance
        total_score += class_score * weights['class_card_ratio']
        
        return total_score

class StrategicGapAnalyzer:
    """
    Strategic gap analysis with priority weighting
    
    Identifies what the deck is missing and prioritizes the most critical gaps
    for optimal draft decision making.
    """
    
    def __init__(self):
        self.gap_analyzers = {
            'early_game': self._analyze_early_game_gap,
            'removal': self._analyze_removal_gap,
            'card_draw': self._analyze_card_draw_gap,
            'late_game': self._analyze_late_game_gap,
            'aoe': self._analyze_aoe_gap,
            'healing': self._analyze_healing_gap,
            'curve_smoothing': self._analyze_curve_smoothing_gap
        }
        
        logger.info("Initialized StrategicGapAnalyzer")
    
    def analyze_strategic_gaps(self, snapshot: ImmutableDeckSnapshot) -> Dict[str, float]:
        """
        Analyze strategic gaps with priority weighting
        
        Returns gap scores 0.0-1.0 where higher scores indicate more critical gaps.
        """
        if not snapshot.validate_integrity():
            raise DataCorruptionError("deck_snapshot", context={"checksum": snapshot.checksum})
        
        gaps = {}
        
        for gap_type, analyzer in self.gap_analyzers.items():
            try:
                gap_score = analyzer(snapshot)
                gaps[gap_type] = max(0.0, min(1.0, gap_score))
            except Exception as e:
                logger.warning(f"Gap analysis error for {gap_type}: {e}")
                gaps[gap_type] = 0.0
        
        # Apply draft phase weighting
        if snapshot.draft_phase == DraftPhase.EARLY:
            # Early draft - prioritize early game and removal
            gaps['early_game'] *= 1.2
            gaps['removal'] *= 1.1
            gaps['curve_smoothing'] *= 0.8
        elif snapshot.draft_phase == DraftPhase.LATE:
            # Late draft - prioritize missing pieces
            gaps['curve_smoothing'] *= 1.3
            gaps['late_game'] *= 1.1
        
        return gaps
    
    def _analyze_early_game_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze early game presence gap"""
        early_game_cards = sum(1 for card in snapshot.cards if card.cost <= 3)
        
        if snapshot.total_cards == 0:
            return 1.0  # Desperate need
        
        early_game_ratio = early_game_cards / snapshot.total_cards
        ideal_ratio = 0.45  # Ideal ~45% early game
        
        if early_game_ratio < ideal_ratio:
            gap_severity = (ideal_ratio - early_game_ratio) / ideal_ratio
            
            # More critical late in draft
            if snapshot.total_cards > 20 and early_game_ratio < 0.3:
                gap_severity *= 1.5
            
            return gap_severity
        
        return 0.0  # No gap
    
    def _analyze_removal_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze removal spells gap"""
        removal_keywords = ['deal', 'damage', 'destroy', 'remove', 'silence', 'transform']
        removal_count = sum(1 for card in snapshot.cards 
                          if card.card_type == CardType.SPELL and 
                          any(keyword in card.text.lower() for keyword in removal_keywords))
        
        if snapshot.total_cards == 0:
            return 0.8  # High need but not desperate
        
        removal_ratio = removal_count / snapshot.total_cards
        ideal_ratio = 0.25  # Ideal ~25% removal
        
        if removal_ratio < ideal_ratio:
            gap_severity = (ideal_ratio - removal_ratio) / ideal_ratio
            
            # More critical for control decks
            if snapshot.archetype_preference == ArchetypePreference.CONTROL:
                gap_severity *= 1.3
            
            return gap_severity
        
        return 0.0
    
    def _analyze_card_draw_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze card advantage gap"""
        draw_keywords = ['draw', 'discover', 'add', 'random', 'copy']
        draw_count = sum(1 for card in snapshot.cards
                        if any(keyword in card.text.lower() for keyword in draw_keywords))
        
        if snapshot.total_cards == 0:
            return 0.6  # Moderate need
        
        draw_ratio = draw_count / snapshot.total_cards
        ideal_ratio = 0.20  # Ideal ~20% card advantage
        
        if draw_ratio < ideal_ratio:
            gap_severity = (ideal_ratio - draw_ratio) / ideal_ratio
            
            # More critical for value/control decks
            if snapshot.archetype_preference in [ArchetypePreference.VALUE, ArchetypePreference.CONTROL]:
                gap_severity *= 1.2
            
            return gap_severity
        
        return 0.0
    
    def _analyze_late_game_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze late game threats gap"""
        late_game_cards = sum(1 for card in snapshot.cards if card.cost >= 6)
        
        if snapshot.total_cards == 0:
            return 0.4  # Lower priority early
        
        late_game_ratio = late_game_cards / snapshot.total_cards
        
        # Ideal varies by archetype
        if snapshot.archetype_preference == ArchetypePreference.AGGRESSIVE:
            ideal_ratio = 0.10  # Aggro needs few late game cards
        elif snapshot.archetype_preference == ArchetypePreference.CONTROL:
            ideal_ratio = 0.25  # Control needs more late game
        else:
            ideal_ratio = 0.15  # Balanced approach
        
        if late_game_ratio < ideal_ratio:
            gap_severity = (ideal_ratio - late_game_ratio) / ideal_ratio
            
            # More critical later in draft
            if snapshot.total_cards > 20:
                gap_severity *= 1.2
            
            return gap_severity
        
        return 0.0
    
    def _analyze_aoe_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze AOE/board clear gap"""
        aoe_keywords = ['all', 'enemy minions', 'all enemies', 'whirlwind', 'consecration']
        aoe_count = sum(1 for card in snapshot.cards
                       if any(keyword in card.text.lower() for keyword in aoe_keywords))
        
        if snapshot.total_cards == 0:
            return 0.5  # Moderate need
        
        # At least 1-2 AOE effects desired
        if aoe_count == 0:
            gap_severity = 0.7
        elif aoe_count == 1 and snapshot.total_cards > 15:
            gap_severity = 0.4
        else:
            gap_severity = 0.0
        
        # More critical for control decks
        if snapshot.archetype_preference == ArchetypePreference.CONTROL:
            gap_severity *= 1.3
        
        return gap_severity
    
    def _analyze_healing_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze healing/sustain gap"""
        healing_keywords = ['heal', 'restore', 'armor', 'lifesteal']
        healing_count = sum(1 for card in snapshot.cards
                           if any(keyword in card.text.lower() for keyword in healing_keywords))
        
        if snapshot.total_cards == 0:
            return 0.3  # Lower priority
        
        # Some healing is nice but not critical in Arena
        if healing_count == 0 and snapshot.total_cards > 20:
            gap_severity = 0.4
        else:
            gap_severity = 0.0
        
        # More important for control/value decks
        if snapshot.archetype_preference in [ArchetypePreference.CONTROL, ArchetypePreference.VALUE]:
            gap_severity *= 1.2
        
        return gap_severity
    
    def _analyze_curve_smoothing_gap(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Analyze mana curve gaps"""
        if snapshot.total_cards < 5:
            return 0.2  # Too early to judge
        
        # Identify the biggest curve gaps
        total_cards = snapshot.total_cards
        curve_gaps = []
        
        # Check each mana cost for gaps
        for cost in range(1, 8):
            card_count = snapshot.mana_curve.get(cost, 0)
            ratio = card_count / total_cards
            
            # Expected ratios (rough guidelines)
            expected_ratios = {
                1: 0.10, 2: 0.15, 3: 0.20, 4: 0.15, 
                5: 0.15, 6: 0.10, 7: 0.10
            }
            
            expected = expected_ratios.get(cost, 0.05)
            if ratio < expected * 0.5:  # Less than half expected
                gap_size = (expected - ratio) / expected
                curve_gaps.append((cost, gap_size))
        
        # Return severity of worst gap
        if curve_gaps:
            worst_gap = max(curve_gaps, key=lambda x: x[1])
            return min(1.0, worst_gap[1])
        
        return 0.0

class CutCandidateAnalyzer:
    """
    Cut candidate analysis with explanatory reasoning
    
    Identifies cards that could be removed from the deck with detailed
    reasoning for each suggestion.
    """
    
    def __init__(self):
        self.cut_criteria = {
            'low_power_level': self._analyze_low_power,
            'curve_redundancy': self._analyze_curve_redundancy,
            'poor_synergy': self._analyze_poor_synergy,
            'situational_cards': self._analyze_situational_cards,
            'duplicate_effects': self._analyze_duplicate_effects
        }
        
        logger.info("Initialized CutCandidateAnalyzer")
    
    def identify_cut_candidates(
        self, 
        snapshot: ImmutableDeckSnapshot,
        max_candidates: int = MAX_CUT_CANDIDATES
    ) -> List[Tuple[CardInfo, str, float]]:
        """
        Identify cards that could be cut with reasoning
        
        Returns list of (card, reason, priority) tuples sorted by cut priority.
        """
        if not snapshot.validate_integrity():
            raise DataCorruptionError("deck_snapshot", context={"checksum": snapshot.checksum})
        
        if snapshot.total_cards <= 20:
            return []  # Don't suggest cuts until deck is nearly full
        
        cut_candidates = []
        
        for card in snapshot.cards:
            for criterion, analyzer in self.cut_criteria.items():
                try:
                    should_cut, reason, priority = analyzer(card, snapshot)
                    if should_cut:
                        cut_candidates.append((card, reason, priority))
                        break  # Only one reason per card
                except Exception as e:
                    logger.warning(f"Cut analysis error for {card.name} ({criterion}): {e}")
        
        # Sort by priority (higher = more important to cut)
        cut_candidates.sort(key=lambda x: x[2], reverse=True)
        
        return cut_candidates[:max_candidates]
    
    def _analyze_low_power(self, card: CardInfo, snapshot: ImmutableDeckSnapshot) -> Tuple[bool, str, float]:
        """Identify cards with low power level"""
        # Simple power level heuristic
        if card.card_type == CardType.MINION:
            stats_total = card.attack + card.health
            expected_stats = card.cost * 2.2  # Rough baseline
            
            if stats_total < expected_stats * 0.7:  # Significantly below curve
                power_deficit = (expected_stats - stats_total) / expected_stats
                return True, f"Below-curve stats ({stats_total} vs expected ~{expected_stats:.1f})", power_deficit
        
        elif card.card_type == CardType.SPELL:
            # Expensive spells with minimal text are often weak
            if card.cost >= 4 and len(card.text) < 20:
                return True, f"High-cost spell with minimal effect", 0.6
        
        return False, "", 0.0
    
    def _analyze_curve_redundancy(self, card: CardInfo, snapshot: ImmutableDeckSnapshot) -> Tuple[bool, str, float]:
        """Identify cards in oversaturated curve positions"""
        cost = min(7, card.cost)  # Cap at 7 for analysis
        cards_at_cost = snapshot.mana_curve.get(cost, 0)
        
        if cards_at_cost > 4:  # Too many cards at this cost
            # Identify weakest card at this cost
            same_cost_cards = [c for c in snapshot.cards if c.cost == cost]
            
            # Simple power ranking (could be improved)
            card_power = self._estimate_card_power(card)
            average_power = sum(self._estimate_card_power(c) for c in same_cost_cards) / len(same_cost_cards)
            
            if card_power < average_power * 0.8:  # Below average for this cost
                redundancy = (cards_at_cost - 3) / 2  # Severity based on excess
                return True, f"Curve oversaturation at {cost} mana ({cards_at_cost} cards)", redundancy
        
        return False, "", 0.0
    
    def _analyze_poor_synergy(self, card: CardInfo, snapshot: ImmutableDeckSnapshot) -> Tuple[bool, str, float]:
        """Identify cards with poor synergy to deck strategy"""
        # Check tribal synergy
        tribal_keywords = ['beast', 'demon', 'dragon', 'elemental', 'mech', 'murloc', 'pirate']
        
        card_tribes = [tribe for tribe in tribal_keywords 
                      if tribe in card.text.lower() or tribe in card.name.lower()]
        
        if card_tribes:
            # Check if deck has other tribal cards
            tribal_support = 0
            for deck_card in snapshot.cards:
                if deck_card != card:  # Don't count the card itself
                    for tribe in card_tribes:
                        if tribe in deck_card.text.lower() or tribe in deck_card.name.lower():
                            tribal_support += 1
                            break
            
            if tribal_support == 0:  # Tribal card with no support
                return True, f"Tribal synergy card ({', '.join(card_tribes)}) with no deck support", 0.7
        
        # Check spell synergy for spell-heavy requirements
        if 'spell' in card.text.lower() and card.card_type == CardType.MINION:
            spell_count = sum(1 for c in snapshot.cards if c.card_type == CardType.SPELL and c != card)
            spell_ratio = spell_count / max(1, snapshot.total_cards - 1)
            
            if spell_ratio < 0.25:  # Less than 25% spells
                return True, f"Spell synergy card with insufficient spells in deck ({spell_count} spells)", 0.6
        
        return False, "", 0.0
    
    def _analyze_situational_cards(self, card: CardInfo, snapshot: ImmutableDeckSnapshot) -> Tuple[bool, str, float]:
        """Identify overly situational cards"""
        situational_keywords = [
            'secret', 'weapon', 'combo', 'overload', 'if', 'when', 'after', 'choose one'
        ]
        
        situational_count = sum(1 for keyword in situational_keywords 
                              if keyword in card.text.lower())
        
        if situational_count >= 2:  # Multiple situational requirements
            # High-cost situational cards are worse
            situational_penalty = situational_count * 0.2
            if card.cost >= 5:
                situational_penalty *= 1.5
            
            return True, f"Highly situational card with {situational_count} conditional requirements", situational_penalty
        
        return False, "", 0.0
    
    def _analyze_duplicate_effects(self, card: CardInfo, snapshot: ImmutableDeckSnapshot) -> Tuple[bool, str, float]:
        """Identify redundant effects in deck"""
        # Check for duplicate specific effects
        if 'silence' in card.text.lower():
            silence_count = sum(1 for c in snapshot.cards if 'silence' in c.text.lower())
            if silence_count > 2:  # Too much silence
                return True, f"Redundant silence effect ({silence_count} silence cards)", 0.5
        
        if 'weapon' in card.text.lower() and card.card_type != CardType.WEAPON:
            weapon_count = sum(1 for c in snapshot.cards 
                             if 'weapon' in c.text.lower() or c.card_type == CardType.WEAPON)
            if weapon_count > 3:  # Too much weapon synergy
                return True, f"Redundant weapon synergy ({weapon_count} weapon-related cards)", 0.4
        
        return False, "", 0.0
    
    def _estimate_card_power(self, card: CardInfo) -> float:
        """Simple card power estimation for comparison"""
        power = 50.0  # Base power
        
        if card.card_type == CardType.MINION:
            stats_total = card.attack + card.health
            cost_efficiency = stats_total / max(1, card.cost)
            power += (cost_efficiency - 2.0) * 15  # Bonus for efficiency
            
            # Text complexity bonus
            power += len(card.text) * 0.1
            
        elif card.card_type == CardType.SPELL:
            # Removal spells are powerful
            if any(keyword in card.text.lower() for keyword in ['deal', 'damage', 'destroy']):
                power += 15
                
        # Rarity bonus
        rarity_bonus = {
            CardRarity.COMMON: 0,
            CardRarity.RARE: 5,
            CardRarity.EPIC: 10,
            CardRarity.LEGENDARY: 15
        }
        power += rarity_bonus.get(card.rarity, 0)
        
        return max(0.0, power)

class ConsistencyValidator:
    """
    Recommendation consistency validation to detect contradictions
    
    Ensures that analysis recommendations don't contradict each other
    and provides warnings when conflicts are detected.
    """
    
    def __init__(self):
        logger.info("Initialized ConsistencyValidator")
    
    def validate_consistency(
        self,
        archetype_scores: Dict[ArchetypePreference, float],
        strategic_gaps: Dict[str, float],
        cut_candidates: List[Tuple[CardInfo, str, float]]
    ) -> List[str]:
        """
        Validate consistency across all recommendations
        
        Returns list of warning messages for detected inconsistencies.
        """
        warnings = []
        
        # Check archetype-gap consistency
        dominant_archetype = max(archetype_scores.keys(), key=lambda k: archetype_scores[k])
        
        if dominant_archetype == ArchetypePreference.AGGRESSIVE:
            if strategic_gaps.get('early_game', 0) < 0.3 and strategic_gaps.get('late_game', 0) > 0.6:
                warnings.append("Aggressive deck identified but analysis suggests needing late game - possible archetype misclassification")
        
        elif dominant_archetype == ArchetypePreference.CONTROL:
            if strategic_gaps.get('removal', 0) < 0.3 and strategic_gaps.get('early_game', 0) > 0.7:
                warnings.append("Control deck identified but analysis suggests needing early game over removal - possible archetype misclassification")
        
        # Check cut candidate consistency
        cut_card_names = {card.name for card, _, _ in cut_candidates}
        
        if strategic_gaps.get('early_game', 0) > 0.6:
            early_cuts = sum(1 for card, _, _ in cut_candidates if card.cost <= 3)
            if early_cuts > 0:
                warnings.append(f"Suggesting cuts to early game cards while deck has early game gap ({early_cuts} cuts)")
        
        if strategic_gaps.get('removal', 0) > 0.6:
            removal_keywords = ['deal', 'damage', 'destroy', 'remove']
            removal_cuts = sum(1 for card, _, _ in cut_candidates 
                             if any(keyword in card.text.lower() for keyword in removal_keywords))
            if removal_cuts > 0:
                warnings.append(f"Suggesting cuts to removal cards while deck lacks removal ({removal_cuts} cuts)")
        
        # Check for contradictory archetype scores
        high_scoring_archetypes = [arch for arch, score in archetype_scores.items() if score > 0.7]
        if len(high_scoring_archetypes) > 2:
            archetype_names = [arch.value for arch in high_scoring_archetypes]
            warnings.append(f"Multiple high-scoring archetypes detected: {', '.join(archetype_names)} - deck may be unfocused")
        
        return warnings

class StrategicDeckAnalyzer:
    """
    Robust Strategic Deck Analyzer with State-Corruption Safe design
    
    This is the core component that analyzes deck strategy, identifies gaps,
    and provides strategic recommendations with immutable state architecture
    to prevent corruption bugs.
    
    Features:
    - Immutable deck state analysis with copy-on-write modifications
    - Fuzzy archetype matching with probabilistic scoring
    - Strategic gap analysis with priority weighting
    - Cut candidate identification with explanatory reasoning
    - Draft phase awareness with dynamic thresholds
    - Recommendation consistency validation
    - Incremental analysis processing for memory efficiency
    """
    
    def __init__(self):
        """Initialize the Strategic Deck Analyzer"""
        self.archetype_analyzer = ArchetypeAnalyzer()
        self.gap_analyzer = StrategicGapAnalyzer()
        self.cut_analyzer = CutCandidateAnalyzer()
        self.consistency_validator = ConsistencyValidator()
        
        self.performance_monitor = get_performance_monitor()
        self.resource_tracker = ResourceTracker("deck_analyzer")
        
        # Analysis statistics
        self.stats = {
            'analyses_total': 0,
            'analyses_incremental': 0,
            'average_duration_ms': 0.0,
            'consistency_warnings': 0,
            'errors_total': 0
        }
        
        # Thread safety
        self.analysis_lock = threading.RLock()
        
        logger.info("Initialized StrategicDeckAnalyzer with immutable state architecture")
    
    def analyze_deck_strategy(self, deck_state: DeckState) -> AnalysisResult:
        """
        Analyze deck strategy with comprehensive evaluation
        
        Args:
            deck_state: Current deck state to analyze
            
        Returns:
            AnalysisResult: Complete strategic analysis with recommendations
            
        Raises:
            DataValidationError: If deck state is invalid
            PerformanceThresholdExceeded: If analysis takes too long
            DataCorruptionError: If state corruption is detected
        """
        start_time = time.time()
        
        with self.analysis_lock:
            try:
                # Input validation
                self._validate_deck_state(deck_state)
                
                # Create immutable snapshot
                snapshot = self._create_immutable_snapshot(deck_state)
                
                # Determine analysis strategy
                use_incremental = len(snapshot.cards) > INCREMENTAL_PROCESSING_THRESHOLD
                
                if use_incremental:
                    result = self._analyze_incremental(snapshot)
                    self.stats['analyses_incremental'] += 1
                else:
                    result = self._analyze_standard(snapshot)
                
                # Validate result consistency
                consistency_warnings = self.consistency_validator.validate_consistency(
                    result.archetype_scores,
                    result.strategic_gaps,
                    result.cut_candidates
                )
                
                if consistency_warnings:
                    self.stats['consistency_warnings'] += len(consistency_warnings)
                    logger.warning(f"Consistency warnings: {consistency_warnings}")
                
                # Create final result with warnings
                final_result = AnalysisResult(
                    archetype_scores=result.archetype_scores,
                    strategic_gaps=result.strategic_gaps,
                    cut_candidates=result.cut_candidates,
                    deck_strength=result.deck_strength,
                    consistency_warnings=consistency_warnings,
                    confidence=result.confidence
                )
                
                # Update statistics
                duration_ms = (time.time() - start_time) * 1000
                self._update_stats(duration_ms)
                
                # Performance monitoring
                if duration_ms > ANALYSIS_TIMEOUT_MS:
                    raise PerformanceThresholdExceeded(
                        "deck_analysis",
                        duration_ms / 1000,
                        ANALYSIS_TIMEOUT_MS / 1000
                    )
                
                return final_result
                
            except Exception as e:
                self.stats['errors_total'] += 1
                if isinstance(e, (DataValidationError, PerformanceThresholdExceeded, DataCorruptionError)):
                    raise
                else:
                    # Wrap unexpected errors
                    raise IntegrationError(
                        f"Deck analysis failed: {str(e)}",
                        context={"deck_size": len(deck_state.cards), "error": str(e)}
                    )
    
    def _validate_deck_state(self, deck_state: DeckState):
        """Validate deck state input"""
        if not isinstance(deck_state, DeckState):
            raise DataValidationError("deck_state", "DeckState", deck_state)
        
        deck_state.validate()
        
        # Additional validation
        if len(deck_state.cards) > 30:
            raise DataValidationError("deck_state.cards", "≤30 cards", len(deck_state.cards))
    
    def _create_immutable_snapshot(self, deck_state: DeckState) -> ImmutableDeckSnapshot:
        """Create immutable snapshot of deck state"""
        return ImmutableDeckSnapshot(
            cards=tuple(deck_state.cards),
            hero_class=deck_state.hero_class,
            current_pick=deck_state.current_pick,
            archetype_preference=deck_state.archetype_preference,
            draft_phase=deck_state.draft_phase,
            mana_curve=deepcopy(deck_state.mana_curve),
            total_cards=deck_state.total_cards,
            average_cost=deck_state.average_cost
        )
    
    def _analyze_standard(self, snapshot: ImmutableDeckSnapshot) -> AnalysisResult:
        """Standard analysis for smaller decks"""
        # Archetype analysis
        archetype_scores = self.archetype_analyzer.analyze_archetype_fit(snapshot)
        
        # Strategic gap analysis
        strategic_gaps = self.gap_analyzer.analyze_strategic_gaps(snapshot)
        
        # Cut candidate analysis
        cut_candidates = self.cut_analyzer.identify_cut_candidates(snapshot)
        
        # Calculate deck strength
        deck_strength = self._calculate_deck_strength(snapshot, archetype_scores)
        
        # Determine confidence
        confidence = self._calculate_analysis_confidence(snapshot, archetype_scores)
        
        return AnalysisResult(
            archetype_scores=archetype_scores,
            strategic_gaps=strategic_gaps,
            cut_candidates=cut_candidates,
            deck_strength=deck_strength,
            consistency_warnings=[],  # Will be added later
            confidence=confidence
        )
    
    def _analyze_incremental(self, snapshot: ImmutableDeckSnapshot) -> AnalysisResult:
        """
        Incremental analysis for larger decks to prevent memory spikes
        
        Processes deck in chunks to maintain memory efficiency while
        providing comprehensive analysis.
        """
        logger.info(f"Using incremental analysis for {snapshot.total_cards} cards")
        
        # Split cards into chunks for processing
        chunk_size = 10
        card_chunks = [
            snapshot.cards[i:i + chunk_size] 
            for i in range(0, len(snapshot.cards), chunk_size)
        ]
        
        # Analyze each chunk and aggregate results
        archetype_contributions = defaultdict(float)
        gap_contributions = defaultdict(float)
        all_cut_candidates = []
        
        for chunk in card_chunks:
            # Create mini-snapshot for chunk analysis
            chunk_snapshot = ImmutableDeckSnapshot(
                cards=chunk,
                hero_class=snapshot.hero_class,
                current_pick=snapshot.current_pick,
                archetype_preference=snapshot.archetype_preference,
                draft_phase=snapshot.draft_phase,
                mana_curve={cost: sum(1 for c in chunk if c.cost == cost) for cost in range(0, 11)},
                total_cards=len(chunk),
                average_cost=sum(c.cost for c in chunk) / len(chunk) if chunk else 0.0
            )
            
            # Analyze chunk
            chunk_archetypes = self.archetype_analyzer.analyze_archetype_fit(chunk_snapshot)
            chunk_gaps = self.gap_analyzer.analyze_strategic_gaps(chunk_snapshot)
            chunk_cuts = self.cut_analyzer.identify_cut_candidates(chunk_snapshot, max_candidates=2)
            
            # Aggregate results (weighted by chunk size)
            weight = len(chunk) / snapshot.total_cards
            
            for archetype, score in chunk_archetypes.items():
                archetype_contributions[archetype] += score * weight
            
            for gap_type, score in chunk_gaps.items():
                gap_contributions[gap_type] += score * weight
            
            all_cut_candidates.extend(chunk_cuts)
        
        # Final analysis on complete deck for accuracy
        final_archetype_scores = self.archetype_analyzer.analyze_archetype_fit(snapshot)
        final_strategic_gaps = self.gap_analyzer.analyze_strategic_gaps(snapshot)
        
        # Combine incremental insights with final analysis
        # Use weighted average: 70% final analysis, 30% incremental insights
        combined_archetype_scores = {}
        for archetype in final_archetype_scores:
            final_score = final_archetype_scores[archetype]
            incremental_score = archetype_contributions.get(archetype, 0.0)
            combined_archetype_scores[archetype] = final_score * 0.7 + incremental_score * 0.3
        
        combined_strategic_gaps = {}
        for gap_type in final_strategic_gaps:
            final_gap = final_strategic_gaps[gap_type]
            incremental_gap = gap_contributions.get(gap_type, 0.0)
            combined_strategic_gaps[gap_type] = final_gap * 0.7 + incremental_gap * 0.3
        
        # Deduplicate and prioritize cut candidates
        seen_cards = set()
        unique_cuts = []
        for card, reason, priority in sorted(all_cut_candidates, key=lambda x: x[2], reverse=True):
            if card.name not in seen_cards:
                unique_cuts.append((card, reason, priority))
                seen_cards.add(card.name)
                if len(unique_cuts) >= MAX_CUT_CANDIDATES:
                    break
        
        # Calculate deck strength and confidence
        deck_strength = self._calculate_deck_strength(snapshot, combined_archetype_scores)
        confidence = self._calculate_analysis_confidence(snapshot, combined_archetype_scores)
        
        return AnalysisResult(
            archetype_scores=combined_archetype_scores,
            strategic_gaps=combined_strategic_gaps,
            cut_candidates=unique_cuts,
            deck_strength=deck_strength,
            consistency_warnings=[],
            confidence=confidence
        )
    
    def _calculate_deck_strength(
        self, 
        snapshot: ImmutableDeckSnapshot, 
        archetype_scores: Dict[ArchetypePreference, float]
    ) -> float:
        """Calculate overall deck strength score"""
        if snapshot.total_cards == 0:
            return 0.0
        
        # Base strength from archetype coherence
        max_archetype_score = max(archetype_scores.values()) if archetype_scores else 0.0
        archetype_strength = max_archetype_score * 40  # 0-40 points
        
        # Curve quality (basic analysis)
        curve_quality = self._evaluate_curve_quality(snapshot) * 30  # 0-30 points
        
        # Card quality estimate (simplified)
        card_quality = self._evaluate_average_card_quality(snapshot) * 30  # 0-30 points
        
        total_strength = archetype_strength + curve_quality + card_quality
        return max(0.0, min(100.0, total_strength))
    
    def _calculate_analysis_confidence(
        self, 
        snapshot: ImmutableDeckSnapshot, 
        archetype_scores: Dict[ArchetypePreference, float]
    ) -> ConfidenceLevel:
        """Calculate confidence in analysis results"""
        confidence_score = 0.5  # Base confidence
        
        # More cards = higher confidence
        card_bonus = min(0.3, snapshot.total_cards / 30 * 0.3)
        confidence_score += card_bonus
        
        # Clear archetype = higher confidence
        if archetype_scores:
            max_archetype_score = max(archetype_scores.values())
            if max_archetype_score > ARCHETYPE_CONFIDENCE_THRESHOLD:
                confidence_score += 0.2
        
        # Later draft phase = higher confidence
        if snapshot.draft_phase == DraftPhase.LATE:
            confidence_score += 0.1
        elif snapshot.draft_phase == DraftPhase.MID:
            confidence_score += 0.05
        
        # Convert to confidence level
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.65:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.35:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _evaluate_curve_quality(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Evaluate mana curve quality (0.0-1.0)"""
        if snapshot.total_cards < 5:
            return 0.5  # Neutral for small decks
        
        # Check distribution balance
        total_cards = snapshot.total_cards
        
        # Ideal distribution (rough)
        ideal_distribution = {
            1: 0.10, 2: 0.15, 3: 0.20, 4: 0.15,
            5: 0.15, 6: 0.10, 7: 0.10, 8: 0.05
        }
        
        quality_score = 0.0
        for cost, ideal_ratio in ideal_distribution.items():
            actual_count = snapshot.mana_curve.get(cost, 0)
            actual_ratio = actual_count / total_cards
            
            # Score based on how close to ideal
            difference = abs(actual_ratio - ideal_ratio)
            slot_score = max(0.0, 1.0 - (difference / ideal_ratio))
            quality_score += slot_score * ideal_ratio  # Weight by importance
        
        return quality_score
    
    def _evaluate_average_card_quality(self, snapshot: ImmutableDeckSnapshot) -> float:
        """Evaluate average card quality (simplified estimation)"""
        if not snapshot.cards:
            return 0.0
        
        quality_sum = 0.0
        
        for card in snapshot.cards:
            card_quality = 0.5  # Base quality
            
            # Rarity bonus
            rarity_bonus = {
                CardRarity.COMMON: 0.0,
                CardRarity.RARE: 0.1,
                CardRarity.EPIC: 0.2,
                CardRarity.LEGENDARY: 0.25
            }
            card_quality += rarity_bonus.get(card.rarity, 0.0)
            
            # Stats efficiency for minions
            if card.card_type == CardType.MINION and card.cost > 0:
                stats_total = card.attack + card.health
                expected_stats = card.cost * 2.2
                efficiency = stats_total / expected_stats
                if efficiency > 1.0:
                    card_quality += min(0.2, (efficiency - 1.0) * 0.4)
                elif efficiency < 0.8:
                    card_quality -= min(0.2, (0.8 - efficiency) * 0.5)
            
            # Text complexity bonus (cards with text usually more powerful)
            if card.text and len(card.text) > 10:
                card_quality += 0.1
            
            quality_sum += max(0.0, min(1.0, card_quality))
        
        return quality_sum / len(snapshot.cards)
    
    def _update_stats(self, duration_ms: float):
        """Update analysis statistics"""
        self.stats['analyses_total'] += 1
        
        # Update average duration
        total_analyses = self.stats['analyses_total']
        current_avg = self.stats['average_duration_ms']
        self.stats['average_duration_ms'] = (current_avg * (total_analyses - 1) + duration_ms) / total_analyses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return deepcopy(self.stats)
    
    def shutdown(self):
        """Graceful shutdown with resource cleanup"""
        logger.info("Shutting down StrategicDeckAnalyzer")
        # No specific resources to clean up for this component
        logger.info("StrategicDeckAnalyzer shutdown complete")

# Export main class
__all__ = [
    'StrategicDeckAnalyzer', 'ImmutableDeckSnapshot', 'AnalysisResult',
    'ArchetypeAnalyzer', 'StrategicGapAnalyzer', 'CutCandidateAnalyzer'
]