"""
Card Evaluation Engine - Multi-Dimensional Card Analysis

The core of the AI v2 system. Evaluates cards across multiple strategic
dimensions using real HSReplay data combined with archetype-specific logic.
"""

import logging
import statistics
from typing import Optional, Dict, Any, List
from datetime import datetime
from .data_models import DimensionalScores, DeckState
from .archetype_config import get_archetype_weights
from .validation_utils import CardDataValidator, DeckStateValidator, ValidationError

# Import data sources
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.cards_json_loader import get_cards_json_loader
from data_sourcing.hsreplay_scraper import get_hsreplay_scraper


class CardEvaluationEngine:
    """
    Multi-dimensional card evaluation system.
    
    Combines HSReplay statistical data with strategic analysis across
    7 dimensions: base_value, tempo, value, synergy, curve, re_draftability, greed.
    """
    
    def __init__(self):
        """Initialize the card evaluation engine with complete HSReplay integration."""
        self.logger = logging.getLogger(__name__)
        
        # Core data sources
        self.cards_loader = get_cards_json_loader()
        self.hsreplay_scraper = get_hsreplay_scraper()
        
        # Data caches
        self.hsreplay_stats = {}  # Card statistics cache
        self.hero_winrates = {}   # Hero performance cache
        self.last_stats_fetch = None
        
        # Performance optimization caches
        self.card_data_cache = {}  # Cached card data for fast lookup
        self.evaluation_cache = {}  # Cached evaluations by card+hero+archetype
        self.text_analysis_cache = {}  # Pre-analyzed card text features
        self.hero_modifier_cache = {}  # Cached hero-specific modifiers
        
        # Performance tracking
        self.evaluation_count = 0
        self.cache_hits = 0
        self.cache_miss_count = 0
        
        # Cache management settings
        self.max_cache_size = 10000
        self.cache_ttl_minutes = 30
        
        # Tribal/keyword synergy definitions
        self.tribal_synergies = self._build_tribal_synergies()
        self.keyword_synergies = self._build_keyword_synergies()
        
        # Pre-warm caches for better performance
        self._prewarm_caches()
        
        self.logger.info("CardEvaluationEngine initialized with HSReplay integration and performance caches")
    
    def evaluate_card(self, card_id: str, deck_state: DeckState) -> DimensionalScores:
        """
        Evaluate a card across all strategic dimensions with hero context.
        
        Args:
            card_id: The card to evaluate
            deck_state: Current draft state for context (hero, archetype, drafted cards)
            
        Returns:
            DimensionalScores with all dimensional analysis
        """
        start_time = datetime.now()
        self.evaluation_count += 1
        
        # Enhanced input validation with comprehensive type checking
        try:
            card_id = CardDataValidator.validate_card_id(card_id, "card_id")
            deck_state = DeckStateValidator.validate_deck_state(deck_state, "deck_state")
            hero_class = deck_state.hero_class  # Now guaranteed to be valid
            self.logger.debug(f"Input validation passed for card evaluation: {card_id}, hero: {hero_class}")
        except ValidationError as e:
            self.logger.error(f"Input validation failed: {e}")
            return self._create_fallback_scores(f"Validation error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {e}")
            return self._create_fallback_scores(f"Validation failed: {e}")
        
        # Generate cache key for this evaluation
        cache_key = self._generate_cache_key(card_id, deck_state)
        
        # Check cache first
        cached_result = self._get_cached_evaluation(cache_key)
        if cached_result:
            self.cache_hits += 1
            return cached_result
        
        self.cache_miss_count += 1
        
        # Ensure fresh HSReplay data with graceful degradation
        try:
            self._ensure_fresh_data()
        except Exception as e:
            self.logger.warning(f"⚠️ HSReplay data unavailable: {e}")
            # Continue with cached data - don't crash
            if not hasattr(self, 'hsreplay_stats') or not self.hsreplay_stats:
                self.hsreplay_stats = {}
                self.logger.info("Using default values due to HSReplay unavailability")
        
        # Calculate each dimension with hero context and defensive programming
        try:
            base_value = self._calculate_base_value(card_id, hero_class)
            if not isinstance(base_value, (int, float)) or base_value < 0:
                self.logger.warning(f"⚠️ Invalid base_value {base_value} for {card_id}")
                base_value = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating base_value for {card_id}: {e}")
            base_value = 0.5
            
        try:
            tempo_score = self._calculate_tempo_score(card_id, deck_state)
            if not isinstance(tempo_score, (int, float)) or tempo_score < 0:
                self.logger.warning(f"⚠️ Invalid tempo_score {tempo_score} for {card_id}")
                tempo_score = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating tempo_score for {card_id}: {e}")
            tempo_score = 0.5
            
        try:
            value_score = self._calculate_value_score(card_id, deck_state)
            if not isinstance(value_score, (int, float)) or value_score < 0:
                self.logger.warning(f"⚠️ Invalid value_score {value_score} for {card_id}")
                value_score = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating value_score for {card_id}: {e}")
            value_score = 0.5
            
        try:
            synergy_score = self._calculate_synergy_score(card_id, deck_state)
            if not isinstance(synergy_score, (int, float)) or synergy_score < 0:
                self.logger.warning(f"⚠️ Invalid synergy_score {synergy_score} for {card_id}")
                synergy_score = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating synergy_score for {card_id}: {e}")
            synergy_score = 0.5
            
        try:
            curve_score = self._calculate_curve_score(card_id, deck_state)
            if not isinstance(curve_score, (int, float)) or curve_score < 0:
                self.logger.warning(f"⚠️ Invalid curve_score {curve_score} for {card_id}")
                curve_score = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating curve_score for {card_id}: {e}")
            curve_score = 0.5
            
        try:
            re_draftability_score = self._calculate_re_draftability_score(card_id)
            if not isinstance(re_draftability_score, (int, float)) or re_draftability_score < 0:
                self.logger.warning(f"⚠️ Invalid re_draftability_score {re_draftability_score} for {card_id}")
                re_draftability_score = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating re_draftability_score for {card_id}: {e}")
            re_draftability_score = 0.5
        
        # Calculate greed score from dimensional variance with validation
        try:
            dimensional_values = [tempo_score, value_score, synergy_score, curve_score]
            # Filter out invalid values
            valid_values = [v for v in dimensional_values if isinstance(v, (int, float)) and 0 <= v <= 1]
            if len(valid_values) < 2:
                greed_score = 0.5  # Default if not enough valid values
            else:
                greed_score = self._calculate_greed_score(valid_values)
                if not isinstance(greed_score, (int, float)) or greed_score < 0:
                    greed_score = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating greed_score for {card_id}: {e}")
            greed_score = 0.5
        
        # Calculate confidence based on data quality with validation
        try:
            confidence = self._calculate_evaluation_confidence(card_id)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                self.logger.warning(f"⚠️ Invalid confidence {confidence} for {card_id}")
                confidence = 0.5
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence for {card_id}: {e}")
            confidence = 0.5
        
        result = DimensionalScores(
            card_id=card_id,
            base_value=base_value,
            tempo_score=tempo_score,
            value_score=value_score,
            synergy_score=synergy_score,
            curve_score=curve_score,
            re_draftability_score=re_draftability_score,
            greed_score=greed_score,
            confidence=confidence
        )
        
        eval_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cache the result for future use
        self._cache_evaluation(cache_key, result)
        
        # Log performance warning if evaluation is slow
        if eval_time > 100:  # 100ms threshold
            self.logger.warning(
                f"Slow card evaluation: {card_id} took {eval_time:.1f}ms (hero: {hero_class})"
            )
        else:
            self.logger.debug(f"Card evaluation completed in {eval_time:.1f}ms: {card_id}")
        
        return result
    
    def _calculate_base_value(self, card_id: str, hero_class: Optional[str] = None) -> float:
        """
        BREAKTHROUGH: Calculate base card value using real HSReplay deck winrates.
        
        Formula: base_value = (deck_winrate - 50.0) / 50.0
        This directly translates HSReplay statistical performance to our scoring system.
        
        Args:
            card_id: Card to evaluate
            hero_class: Hero context for hero-specific adjustments
            
        Returns:
            Base value score from -1.0 (terrible) to +1.0 (excellent)
        """
        try:
            # Get HSReplay card statistics
            card_stats = self.hsreplay_stats.get(card_id)
            
            if card_stats and 'deck_win_rate' in card_stats:
                deck_winrate = card_stats['deck_win_rate']
                # Convert winrate to our -1 to +1 scale
                base_value = (deck_winrate - 50.0) / 50.0
                
                # Apply hero-specific adjustments
                if hero_class:
                    base_value = self._apply_hero_base_value_modifier(card_id, base_value, hero_class)
                
                # Clamp to valid range
                return max(-1.0, min(1.0, base_value))
            
            else:
                # Fallback to card quality estimation
                return self._estimate_base_value_fallback(card_id, hero_class)
                
        except Exception as e:
            self.logger.warning(f"Error calculating base value for {card_id}: {e}")
            return 0.0  # Neutral fallback
    
    def _calculate_tempo_score(self, card_id: str, deck_state: DeckState) -> float:
        """Calculate tempo impact score enhanced with hero-specific tempo considerations."""
        try:
            # Use cached card data for performance
            card_data = self._get_cached_card_data(card_id)
            cost = card_data.get('cost', 0)
            attack = card_data.get('attack', 0)
            health = card_data.get('health', 0)
            card_type = card_data.get('type', '')
            
            # Use cached text analysis for performance
            text_features = self._get_cached_text_analysis(card_id)
            
            # Base tempo from stats-to-cost ratio
            if cost > 0 and card_type == 'MINION':
                stat_total = attack + health
                stat_efficiency = stat_total / cost
                # Normalize around 2.5 stats per mana (vanilla test)
                tempo_base = (stat_efficiency - 2.5) / 2.5
            else:
                tempo_base = 0.0
            
            # Immediate impact bonuses using cached analysis
            immediate_impact = 0.0
            if text_features.get('charge') or text_features.get('rush'):
                immediate_impact += 0.3
            if text_features.get('battlecry'):
                immediate_impact += 0.2
            if text_features.get('taunt'):
                immediate_impact += 0.15
            
            # Hero-specific tempo modifiers
            hero_modifier = self._get_hero_tempo_modifier(deck_state.hero_class, card_id)
            
            final_score = (tempo_base + immediate_impact) * hero_modifier
            return max(-1.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating tempo score for {card_id}: {e}")
            return 0.0
    
    def _calculate_value_score(self, card_id: str, deck_state: DeckState) -> float:
        """Calculate long-term value score with hero-specific value preferences."""
        try:
            # Use cached card data for performance
            card_data = self._get_cached_card_data(card_id)
            card_type = card_data.get('type', '')
            
            # Use cached text analysis for performance
            text_features = self._get_cached_text_analysis(card_id)
            
            # Card advantage mechanisms
            value_score = 0.0
            
            # Draw effects
            if text_features.get('draw'):
                draw_count = text_features.get('draw_count', 1)
                value_score += draw_count * 0.3
            
            # Discover/generation effects
            if text_features.get('has_generation'):
                value_score += 0.25
            
            # Lifesteal for sustained value
            if text_features.get('lifesteal'):
                value_score += 0.2
            
            # Deathrattle for sticky value
            if text_features.get('deathrattle'):
                value_score += 0.15
            
            # High-value spell generation
            if card_type == 'SPELL' and text_features.get('is_removal'):
                value_score += 0.1
            
            # Hero-specific value bonuses (e.g., Warlock values card draw more)
            card_text = card_data.get('text', '')
            hero_modifier = self._get_hero_value_modifier(deck_state.hero_class, card_id, card_text)
            
            final_score = value_score * hero_modifier
            return max(-1.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating value score for {card_id}: {e}")
            return 0.0
    
    def _calculate_synergy_score(self, card_id: str, deck_state: DeckState) -> float:
        """Calculate synergy score with hero-specific synergy bonuses and cross-deck detection."""
        try:
            if not deck_state.drafted_cards:
                return 0.0  # No synergies possible with empty deck
            
            card_data = self.cards_loader.cards_data.get(card_id, {})
            text = card_data.get('text', '').lower()
            race = card_data.get('race', '')
            
            synergy_score = 0.0
            
            # Tribal synergies
            if race:
                tribal_count = sum(1 for drafted_id in deck_state.drafted_cards 
                                 if self.cards_loader.get_card_attribute(drafted_id, 'race') == race)
                synergy_score += tribal_count * 0.1
            
            # Keyword synergies
            for keyword, synergy_cards in self.keyword_synergies.items():
                if keyword in text:
                    keyword_count = sum(1 for drafted_id in deck_state.drafted_cards 
                                      if drafted_id in synergy_cards)
                    synergy_score += keyword_count * 0.08
            
            # Hero-specific synergy bonuses
            hero_bonus = self._get_hero_synergy_bonus(deck_state.hero_class, card_id, deck_state.drafted_cards)
            synergy_score += hero_bonus
            
            return max(-1.0, min(1.0, synergy_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating synergy score for {card_id}: {e}")
            return 0.0
    
    def _calculate_curve_score(self, card_id: str, deck_state: DeckState) -> float:
        """Calculate mana curve optimization with hero-specific curve preferences."""
        try:
            card_cost = self.cards_loader.get_card_cost(card_id) or 0
            
            if not deck_state.drafted_cards:
                # Early draft - prioritize 2-4 cost cards
                if 2 <= card_cost <= 4:
                    return 0.3
                elif card_cost == 1 or card_cost == 5:
                    return 0.1
                else:
                    return -0.1
            
            # Analyze current curve distribution
            curve_counts = {}
            for drafted_id in deck_state.drafted_cards:
                cost = self.cards_loader.get_card_cost(drafted_id) or 0
                curve_counts[cost] = curve_counts.get(cost, 0) + 1
            
            # Get hero-specific ideal curve
            ideal_curve = self._get_hero_ideal_curve(deck_state.hero_class)
            
            # Calculate how much this card improves the curve
            current_count = curve_counts.get(card_cost, 0)
            total_cards = len(deck_state.drafted_cards)
            current_ratio = current_count / total_cards if total_cards > 0 else 0
            ideal_ratio = ideal_curve.get(card_cost, 0.1)
            
            # Score based on how close we are to ideal
            if current_ratio < ideal_ratio:
                curve_score = 0.5 * (ideal_ratio - current_ratio)
            else:
                curve_score = -0.3 * (current_ratio - ideal_ratio)
            
            return max(-1.0, min(1.0, curve_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating curve score for {card_id}: {e}")
            return 0.0
    
    def _calculate_re_draftability_score(self, card_id: str) -> float:
        """Calculate Underground Arena re-draft potential with hero-specific considerations."""
        try:
            rarity = self.cards_loader.get_card_rarity(card_id)
            card_type = self.cards_loader.get_card_type(card_id)
            
            # Base re-draftability by rarity
            rarity_scores = {
                'COMMON': 0.8,      # High chance to see again
                'RARE': 0.5,        # Medium chance
                'EPIC': 0.2,        # Low chance
                'LEGENDARY': -0.3   # Very low chance, risky for re-drafts
            }
            
            base_score = rarity_scores.get(rarity, 0.0)
            
            # Type modifiers
            if card_type == 'SPELL':
                base_score += 0.1  # Spells often have multiple copies
            
            # Unique effects penalty
            text = self.cards_loader.get_card_attribute(card_id, 'text') or ''
            if 'legendary' in text.lower() or len(text) > 100:  # Complex effects
                base_score -= 0.2
            
            return max(-1.0, min(1.0, base_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating re-draftability score for {card_id}: {e}")
            return 0.0
    
    def _calculate_greed_score(self, dimensional_values: List[float]) -> float:
        """Calculate risk/specialization score using variance of dimensional scores."""
        try:
            if len(dimensional_values) < 2:
                return 0.0
            
            # Calculate standard deviation of dimensional scores
            std_dev = statistics.stdev(dimensional_values)
            
            # High variance = specialized/greedy card
            # Low variance = safe/balanced card
            # Normalize to -1 to 1 scale (higher = greedier)
            greed_score = (std_dev - 0.3) / 0.7  # Assuming 0.3 is "balanced" variance
            
            return max(-1.0, min(1.0, greed_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating greed score: {e}")
            return 0.0
    
    # === Hero-Specific Modifier Methods ===
    
    def _apply_hero_base_value_modifier(self, card_id: str, base_value: float, hero_class: str) -> float:
        """Apply hero-specific adjustments to base card value."""
        card_class = self.cards_loader.get_card_class(card_id)
        
        # Class cards get bonus for their hero
        if card_class == hero_class:
            return base_value * 1.15
        
        # Neutral cards - no modification
        return base_value
    
    def _get_hero_tempo_modifier(self, hero_class: str, card_id: str) -> float:
        """Get hero-specific tempo modifier (Hunter favors early tempo, Priest favors late value)."""
        modifiers = {
            'HUNTER': 1.2,      # Aggressive early tempo
            'DEMONHUNTER': 1.15,
            'PALADIN': 1.1,
            'ROGUE': 1.05,
            'WARRIOR': 1.0,     # Balanced
            'MAGE': 1.0,
            'SHAMAN': 1.0,
            'WARLOCK': 0.95,
            'DRUID': 0.9,       # Prefers value over tempo
            'PRIEST': 0.85      # Control-oriented
        }
        return modifiers.get(hero_class, 1.0)
    
    def _get_hero_value_modifier(self, hero_class: str, card_id: str, card_text: str) -> float:
        """Get hero-specific value modifier (Warlock loves card draw, Priest loves healing)."""
        base_modifier = 1.0
        
        # Warlock values card draw highly
        if hero_class == 'WARLOCK' and 'draw' in card_text:
            base_modifier = 1.5
        
        # Priest values healing and value generation
        elif hero_class == 'PRIEST' and ('heal' in card_text or 'discover' in card_text):
            base_modifier = 1.3
        
        # Mage values spell generation
        elif hero_class == 'MAGE' and ('spell' in card_text or 'discover' in card_text):
            base_modifier = 1.2
        
        # Hunter cares less about long-term value
        elif hero_class == 'HUNTER':
            base_modifier = 0.8
        
        return base_modifier
    
    def _get_hero_synergy_bonus(self, hero_class: str, card_id: str, drafted_cards: List[str]) -> float:
        """Calculate hero-specific synergy bonuses."""
        bonus = 0.0
        card_text = self.cards_loader.get_card_attribute(card_id, 'text') or ''
        
        # Warrior weapon synergy
        if hero_class == 'WARRIOR':
            if 'weapon' in card_text.lower():
                weapon_count = sum(1 for drafted_id in drafted_cards 
                                 if 'weapon' in (self.cards_loader.get_card_attribute(drafted_id, 'text') or '').lower())
                bonus += weapon_count * 0.1
        
        # Shaman elemental synergy
        elif hero_class == 'SHAMAN':
            if 'elemental' in card_text.lower():
                elemental_count = sum(1 for drafted_id in drafted_cards
                                    if self.cards_loader.get_card_attribute(drafted_id, 'race') == 'ELEMENTAL')
                bonus += elemental_count * 0.12
        
        return bonus
    
    def _get_hero_ideal_curve(self, hero_class: str) -> Dict[int, float]:
        """Get hero-specific ideal mana curve distributions."""
        curves = {
            'HUNTER': {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05, 7: 0.05},      # Aggressive
            'DEMONHUNTER': {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05, 7: 0.05},
            'PALADIN': {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.10, 7: 0.00},     # Midrange
            'WARRIOR': {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.15, 7: 0.05},     # Balanced
            'MAGE': {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.20, 6: 0.10, 7: 0.05},       # Tempo/Control
            'ROGUE': {1: 0.10, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.05, 7: 0.00},      # Tempo
            'SHAMAN': {1: 0.10, 2: 0.20, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.10, 7: 0.05},     # Balanced
            'WARLOCK': {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.20, 6: 0.10, 7: 0.05},    # Control
            'PRIEST': {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.15, 7: 0.10},     # Control
            'DRUID': {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.15, 5: 0.20, 6: 0.20, 7: 0.15}       # Ramp
        }
        
        return curves.get(hero_class, curves['WARRIOR'])  # Default to balanced
    
    # === Data Management Methods ===
    
    def _ensure_fresh_data(self) -> None:
        """Ensure HSReplay data is fresh and available with connection validation."""
        try:
            # Check if HSReplay scraper is available and connected
            if not self._validate_hsreplay_connection():
                self.logger.warning("HSReplay connection not available, using cached data")
                return
            
            # Check if we need to refresh HSReplay card data
            if self._should_refresh_hsreplay_data():
                self.logger.debug("Refreshing HSReplay card data...")
                
                # Add timeout and retry logic for HSReplay API calls
                fresh_stats = self._fetch_hsreplay_stats_with_retry()
                
                if fresh_stats:
                    self.hsreplay_stats = fresh_stats
                    self.last_stats_fetch = datetime.now()
                    self.logger.info(f"HSReplay card data refreshed: {len(fresh_stats)} cards")
                else:
                    self.logger.warning("Failed to refresh HSReplay data, using cached data")
            else:
                self.cache_hits += 1
                
        except Exception as e:
            self.logger.error(f"Error ensuring fresh data: {e}, degrading to cached data")
    
    def _should_refresh_hsreplay_data(self) -> bool:
        """Check if HSReplay data needs refreshing."""
        if not self.last_stats_fetch or not self.hsreplay_stats:
            return True
        
        # Refresh every 24 hours
        age = datetime.now() - self.last_stats_fetch
        return age.total_seconds() > (24 * 3600)
    
    def _validate_hsreplay_connection(self) -> bool:
        """Validate HSReplay connection and API availability."""
        try:
            # Check if hsreplay_scraper exists and is not None
            if not self.hsreplay_scraper:
                return False
            
            # Check if scraper has required methods
            if not hasattr(self.hsreplay_scraper, 'get_underground_arena_stats'):
                self.logger.error("HSReplay scraper missing required methods")
                return False
            
            # Try to get API status if available
            if hasattr(self.hsreplay_scraper, 'get_api_status'):
                status = self.hsreplay_scraper.get_api_status()  
                if not status or not status.get('session_active', False):
                    self.logger.warning("HSReplay API session not active")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"HSReplay connection validation failed: {e}")
            return False
    
    def _fetch_hsreplay_stats_with_retry(self, max_retries: int = 3, timeout_seconds: float = 10.0) -> dict:
        """Fetch HSReplay stats with retry logic and timeout protection."""
        import threading
        import queue
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"HSReplay fetch attempt {attempt + 1}/{max_retries}")
                
                # Use threading to add timeout protection
                result_queue = queue.Queue()
                exception_queue = queue.Queue()
                
                def fetch_worker():
                    try:
                        stats = self.hsreplay_scraper.get_underground_arena_stats()
                        result_queue.put(stats)
                    except Exception as e:
                        exception_queue.put(e)
                
                # Start fetch in separate thread with timeout
                thread = threading.Thread(target=fetch_worker, daemon=True)
                thread.start()
                thread.join(timeout=timeout_seconds)
                
                if thread.is_alive():
                    self.logger.warning(f"HSReplay fetch timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None
                
                # Check for exceptions
                if not exception_queue.empty():
                    exception = exception_queue.get()
                    self.logger.warning(f"HSReplay fetch error on attempt {attempt + 1}: {exception}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None
                
                # Get result
                if not result_queue.empty():
                    result = result_queue.get()
                    if result:  # Valid result
                        self.logger.debug(f"HSReplay fetch successful on attempt {attempt + 1}")
                        return result
                    else:
                        self.logger.warning(f"HSReplay fetch returned empty data on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return None
                            
            except Exception as e:
                self.logger.error(f"Unexpected error in HSReplay fetch attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
        
        return None
    
    def _estimate_base_value_fallback(self, card_id: str, hero_class: Optional[str]) -> float:
        """Fallback base value estimation when HSReplay data unavailable."""
        try:
            # Simple fallback based on rarity and cost efficiency
            rarity = self.cards_loader.get_card_rarity(card_id)
            cost = self.cards_loader.get_card_cost(card_id) or 0
            card_type = self.cards_loader.get_card_type(card_id)
            
            # Base score by rarity
            rarity_scores = {
                'COMMON': 0.0,
                'RARE': 0.1,
                'EPIC': 0.2,
                'LEGENDARY': 0.3
            }
            
            base_score = rarity_scores.get(rarity, 0.0)
            
            # Adjust for cost efficiency (minions only)
            if card_type == 'MINION' and cost > 0:
                attack = self.cards_loader.get_card_attribute(card_id, 'attack') or 0
                health = self.cards_loader.get_card_attribute(card_id, 'health') or 0
                stat_total = attack + health
                efficiency = stat_total / cost
                
                if efficiency > 3.0:
                    base_score += 0.2
                elif efficiency < 2.0:
                    base_score -= 0.2
            
            return max(-0.5, min(0.5, base_score))  # Conservative range
            
        except Exception as e:
            self.logger.warning(f"Error in fallback estimation for {card_id}: {e}")
            return 0.0
    
    def _calculate_evaluation_confidence(self, card_id: str) -> float:
        """Calculate confidence in evaluation based on data quality."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence if we have HSReplay data
        if card_id in self.hsreplay_stats:
            card_stats = self.hsreplay_stats[card_id]
            times_played = card_stats.get('times_played', 0)
            
            if times_played > 1000:
                confidence = 0.95
            elif times_played > 100:
                confidence = 0.85
            else:
                confidence = 0.75
        else:
            confidence = 0.6  # Lower confidence without statistical data
        
        # Reduce confidence for very new/unknown cards
        if not self.cards_loader.cards_data.get(card_id):
            confidence = 0.3
        
        return confidence
    
    def _build_tribal_synergies(self) -> Dict[str, List[str]]:
        """Build tribal synergy mappings."""
        return {
            'BEAST': [],
            'DRAGON': [],
            'ELEMENTAL': [],
            'MURLOC': [],
            'PIRATE': [],
            'MECH': [],
            'DEMON': []
        }
    
    def _build_keyword_synergies(self) -> Dict[str, List[str]]:
        """Build keyword synergy mappings.""" 
        return {
            'weapon': [],
            'spell': [],
            'secret': [],
            'overload': []
        }
    
    # === Statistics and Performance Methods ===
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        return {
            "evaluations_performed": self.evaluation_count,
            "cache_hits": self.cache_hits,
            "hsreplay_cards_available": len(self.hsreplay_stats),
            "last_stats_fetch": self.last_stats_fetch.isoformat() if self.last_stats_fetch else None,
            "data_age_hours": (
                (datetime.now() - self.last_stats_fetch).total_seconds() / 3600 
                if self.last_stats_fetch else None
            ),
            "cache_hit_rate": (self.cache_hits / self.evaluation_count * 100) if self.evaluation_count > 0 else 0
        }
    
    # === HERO-SPECIFIC SYNERGY ANALYSIS METHODS ===
    
    def analyze_hero_specific_synergies(self, hero_class: str, available_cards: List[str] = None) -> Dict[str, Any]:
        """
        Analyze hero-specific synergies using HSReplay co-occurrence data.
        
        Identifies cards that have strong synergistic relationships when drafted together
        by specific heroes, based on statistical co-occurrence patterns.
        """
        try:
            synergy_analysis = {
                'hero_class': hero_class,
                'timestamp': datetime.now().isoformat(),
                'synergy_clusters': {},
                'card_recommendations': {},
                'meta_insights': [],
                'confidence_level': 0.8
            }
            
            # Get hero-specific co-occurrence data
            co_occurrence_data = self._get_hero_cooccurrence_data(hero_class)
            
            if not co_occurrence_data:
                return self._get_fallback_hero_synergies(hero_class)
            
            # Analyze synergy clusters
            synergy_clusters = self._identify_synergy_clusters(co_occurrence_data, hero_class)
            synergy_analysis['synergy_clusters'] = synergy_clusters
            
            # Generate card recommendations based on synergies
            if available_cards:
                card_recommendations = self._generate_synergy_recommendations(
                    available_cards, synergy_clusters, hero_class
                )
                synergy_analysis['card_recommendations'] = card_recommendations
            
            # Generate meta insights
            meta_insights = self._generate_synergy_meta_insights(synergy_clusters, hero_class)
            synergy_analysis['meta_insights'] = meta_insights
            
            return synergy_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing hero synergies for {hero_class}: {e}")
            return self._get_fallback_hero_synergies(hero_class)
    
    def get_card_synergy_partners(self, card_id: str, hero_class: str, deck_cards: List[str] = None) -> Dict[str, Any]:
        """
        Get specific synergy partners for a card with a given hero.
        
        Returns cards that have high co-occurrence rates and performance bonuses
        when drafted together with the specified card by the hero class.
        """
        try:
            # Get co-occurrence data for this specific card
            card_cooccurrence = self._get_card_cooccurrence_data(card_id, hero_class)
            
            if not card_cooccurrence:
                return self._get_fallback_card_synergies(card_id, hero_class)
            
            # Analyze synergy strength with potential partners
            synergy_partners = []
            
            for partner_card, cooccurrence_stats in card_cooccurrence.items():
                synergy_strength = self._calculate_synergy_strength(
                    card_id, partner_card, cooccurrence_stats, hero_class
                )
                
                if synergy_strength > 0.3:  # Minimum threshold for meaningful synergy
                    partner_analysis = {
                        'partner_card': partner_card,
                        'synergy_strength': round(synergy_strength, 3),
                        'cooccurrence_rate': cooccurrence_stats.get('cooccurrence_rate', 0),
                        'performance_bonus': cooccurrence_stats.get('performance_bonus', 0),
                        'synergy_type': self._classify_synergy_type(card_id, partner_card, hero_class),
                        'explanation': self._explain_card_synergy(card_id, partner_card, hero_class)
                    }
                    synergy_partners.append(partner_analysis)
            
            # Sort by synergy strength
            synergy_partners.sort(key=lambda x: x['synergy_strength'], reverse=True)
            
            # Check for existing synergies in current deck
            existing_synergies = []
            if deck_cards:
                for deck_card in deck_cards:
                    for partner in synergy_partners:
                        if partner['partner_card'] == deck_card:
                            existing_synergies.append(partner)
            
            return {
                'card_id': card_id,
                'hero_class': hero_class,
                'top_synergy_partners': synergy_partners[:10],
                'existing_deck_synergies': existing_synergies,
                'synergy_potential_score': self._calculate_overall_synergy_potential(synergy_partners),
                'draft_recommendations': self._generate_synergy_draft_advice(synergy_partners, existing_synergies)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting synergy partners for {card_id}: {e}")
            return self._get_fallback_card_synergies(card_id, hero_class)
    
    def _get_hero_cooccurrence_data(self, hero_class: str) -> Dict[str, Any]:
        """Get hero-specific card co-occurrence data from HSReplay."""
        try:
            # Try to get from HSReplay scraper if available
            if hasattr(self.hsreplay_scraper, 'get_hero_cooccurrence_data'):
                return self.hsreplay_scraper.get_hero_cooccurrence_data(hero_class)
            
            # Fallback: simulate from general card data with hero modifiers
            return self._simulate_hero_cooccurrence_data(hero_class)
            
        except Exception as e:
            self.logger.warning(f"Error getting co-occurrence data for {hero_class}: {e}")
            return {}
    
    def _simulate_hero_cooccurrence_data(self, hero_class: str) -> Dict[str, Any]:
        """Simulate hero-specific co-occurrence data from general patterns."""
        try:
            # Hero-specific card preferences and synergy patterns
            hero_synergy_patterns = {
                'HUNTER': {
                    'primary_synergies': ['beast', 'weapon', 'face_damage'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [1, 2, 3]
                },
                'MAGE': {
                    'primary_synergies': ['spell_damage', 'freeze', 'secret'],
                    'card_types': ['spell', 'minion'],
                    'cost_preferences': [2, 3, 4, 5]
                },
                'PRIEST': {
                    'primary_synergies': ['heal', 'high_health', 'deathrattle'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [3, 4, 5, 6]
                },
                'WARLOCK': {
                    'primary_synergies': ['demon', 'discard', 'life_tap'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [1, 2, 4, 5]
                },
                'WARRIOR': {
                    'primary_synergies': ['weapon', 'armor', 'enrage'],
                    'card_types': ['minion', 'weapon', 'spell'],
                    'cost_preferences': [2, 3, 4, 5]
                },
                'PALADIN': {
                    'primary_synergies': ['divine_shield', 'buff', 'weapon'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [1, 2, 3, 4]
                },
                'ROGUE': {
                    'primary_synergies': ['combo', 'weapon', 'stealth'],
                    'card_types': ['minion', 'spell', 'weapon'],
                    'cost_preferences': [0, 1, 2, 3]
                },
                'SHAMAN': {
                    'primary_synergies': ['elemental', 'overload', 'spell_damage'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [1, 2, 3, 4]
                },
                'DRUID': {
                    'primary_synergies': ['choose_one', 'ramp', 'beast'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [2, 3, 6, 7, 8]
                },
                'DEMONHUNTER': {
                    'primary_synergies': ['outcast', 'demon', 'attack'],
                    'card_types': ['minion', 'spell'],
                    'cost_preferences': [1, 2, 3]
                }
            }
            
            return hero_synergy_patterns.get(hero_class, {})
            
        except Exception as e:
            self.logger.warning(f"Error simulating co-occurrence data: {e}")
            return {}
    
    def _get_card_cooccurrence_data(self, card_id: str, hero_class: str) -> Dict[str, Dict[str, float]]:
        """Get co-occurrence data for a specific card with a hero."""
        try:
            # Attempt to get real data from HSReplay
            if hasattr(self.hsreplay_scraper, 'get_card_cooccurrence_data'):
                return self.hsreplay_scraper.get_card_cooccurrence_data(card_id, hero_class)
            
            # Fallback: analyze based on card properties and hero synergies
            return self._analyze_card_synergies_heuristic(card_id, hero_class)
            
        except Exception as e:
            self.logger.warning(f"Error getting card co-occurrence data: {e}")
            return {}
    
    def _analyze_card_synergies_heuristic(self, card_id: str, hero_class: str) -> Dict[str, Dict[str, float]]:
        """Analyze card synergies using heuristic approach."""
        try:
            synergies = {}
            
            # Get card properties
            card_cost = self.cards_loader.get_card_cost(card_id) or 0
            card_type = self.cards_loader.get_card_type(card_id) or 'MINION'
            card_class = self.cards_loader.get_card_class(card_id) or 'NEUTRAL'
            
            # Generate synergy candidates based on card properties
            if card_type == 'SPELL':
                # Spell synergies
                if hero_class == 'MAGE':
                    synergies['spell_damage_minions'] = {
                        'cooccurrence_rate': 0.4,
                        'performance_bonus': 1.5
                    }
                elif hero_class == 'PRIEST':
                    synergies['heal_targets'] = {
                        'cooccurrence_rate': 0.3,
                        'performance_bonus': 1.2
                    }
            
            elif card_type == 'MINION':
                # Minion synergies
                if hero_class == 'HUNTER' and card_cost <= 3:
                    synergies['aggressive_minions'] = {
                        'cooccurrence_rate': 0.5,
                        'performance_bonus': 1.3
                    }
                elif hero_class == 'PRIEST' and card_cost >= 4:
                    synergies['high_health_minions'] = {
                        'cooccurrence_rate': 0.4,
                        'performance_bonus': 1.1
                    }
            
            return synergies
            
        except Exception as e:
            self.logger.warning(f"Error in heuristic synergy analysis: {e}")
            return {}
    
    def _identify_synergy_clusters(self, cooccurrence_data: Dict, hero_class: str) -> Dict[str, Any]:
        """Identify synergy clusters from co-occurrence data."""
        try:
            clusters = {}
            
            # Analyze synergy patterns from the data
            synergy_patterns = cooccurrence_data.get('primary_synergies', [])
            
            for i, synergy_type in enumerate(synergy_patterns):
                cluster = {
                    'synergy_type': synergy_type,
                    'strength': 0.8 - (i * 0.1),  # Decreasing strength by priority
                    'description': self._get_synergy_description(synergy_type, hero_class),
                    'key_cards': self._get_key_cards_for_synergy(synergy_type, hero_class),
                    'draft_priority': 'High' if i == 0 else 'Medium' if i == 1 else 'Low'
                }
                clusters[synergy_type] = cluster
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Error identifying synergy clusters: {e}")
            return {}
    
    def _calculate_synergy_strength(self, card1: str, card2: str, stats: Dict, hero_class: str) -> float:
        """Calculate synergy strength between two cards."""
        try:
            base_strength = stats.get('cooccurrence_rate', 0)
            performance_bonus = stats.get('performance_bonus', 1.0)
            
            # Normalize performance bonus to synergy strength scale
            bonus_strength = (performance_bonus - 1.0) * 0.5
            
            # Combine factors
            total_strength = base_strength + bonus_strength
            
            # Cap at maximum strength
            return min(1.0, max(0.0, total_strength))
            
        except Exception:
            return 0.0
    
    def _classify_synergy_type(self, card1: str, card2: str, hero_class: str) -> str:
        """Classify the type of synergy between two cards."""
        try:
            # Get card properties
            card1_type = self.cards_loader.get_card_type(card1) or 'UNKNOWN'
            card2_type = self.cards_loader.get_card_type(card2) or 'UNKNOWN'
            card1_cost = self.cards_loader.get_card_cost(card1) or 0
            card2_cost = self.cards_loader.get_card_cost(card2) or 0
            
            # Classify synergy type
            if card1_type == 'SPELL' and card2_type == 'SPELL':
                return 'Spell Combo'
            elif card1_type == 'MINION' and card2_type == 'MINION':
                if abs(card1_cost - card2_cost) <= 1:
                    return 'Curve Synergy'
                else:
                    return 'Board Synergy'
            elif 'SPELL' in [card1_type, card2_type] and 'MINION' in [card1_type, card2_type]:
                return 'Spell-Minion Synergy'
            else:
                return 'General Synergy'
                
        except Exception:
            return 'Unknown Synergy'
    
    def _explain_card_synergy(self, card1: str, card2: str, hero_class: str) -> str:
        """Generate explanation for card synergy."""
        try:
            synergy_type = self._classify_synergy_type(card1, card2, hero_class)
            
            explanations = {
                'Spell Combo': f"Spell combination that works well together in {hero_class}",
                'Curve Synergy': f"Cards that form effective mana curve for {hero_class}",
                'Board Synergy': f"Minions that support each other on board",
                'Spell-Minion Synergy': f"Spell and minion that enhance each other's effectiveness",
                'General Synergy': f"Cards that complement {hero_class} strategy"
            }
            
            return explanations.get(synergy_type, "Cards work well together")
            
        except Exception:
            return "Synergistic combination"
    
    def _get_synergy_description(self, synergy_type: str, hero_class: str) -> str:
        """Get description for synergy type."""
        descriptions = {
            'beast': f"Beast tribal synergies for {hero_class}",
            'spell_damage': f"Spell damage synergies for {hero_class}", 
            'weapon': f"Weapon-based synergies for {hero_class}",
            'heal': f"Healing synergies for {hero_class}",
            'demon': f"Demon tribal synergies for {hero_class}",
            'divine_shield': f"Divine shield synergies for {hero_class}",
            'combo': f"Combo synergies for {hero_class}",
            'elemental': f"Elemental tribal synergies for {hero_class}",
            'choose_one': f"Choose One synergies for {hero_class}",
            'outcast': f"Outcast synergies for {hero_class}"
        }
        return descriptions.get(synergy_type, f"{synergy_type} synergies for {hero_class}")
    
    def _get_key_cards_for_synergy(self, synergy_type: str, hero_class: str) -> List[str]:
        """Get key cards that enable or benefit from synergy type."""
        # This would ideally come from actual HSReplay data
        # For now, return placeholder examples
        return [f"Key card 1 for {synergy_type}", f"Key card 2 for {synergy_type}"]
    
    def _generate_synergy_recommendations(self, available_cards: List[str], 
                                        clusters: Dict, hero_class: str) -> Dict[str, Any]:
        """Generate synergy-based card recommendations."""
        recommendations = {}
        
        try:
            for card_id in available_cards:
                synergy_score = 0.0
                synergy_details = []
                
                # Check each synergy cluster
                for synergy_type, cluster in clusters.items():
                    card_fit = self._evaluate_card_cluster_fit(card_id, cluster, hero_class)
                    if card_fit > 0.3:
                        synergy_score += card_fit * cluster['strength']
                        synergy_details.append({
                            'synergy_type': synergy_type,
                            'fit_score': round(card_fit, 3),
                            'cluster_strength': cluster['strength']
                        })
                
                if synergy_score > 0.2:  # Minimum threshold
                    recommendations[card_id] = {
                        'overall_synergy_score': round(synergy_score, 3),
                        'synergy_details': synergy_details,
                        'recommendation_strength': 'High' if synergy_score > 0.6 else 'Medium' if synergy_score > 0.4 else 'Low'
                    }
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating synergy recommendations: {e}")
            return {}
    
    def _evaluate_card_cluster_fit(self, card_id: str, cluster: Dict, hero_class: str) -> float:
        """Evaluate how well a card fits into a synergy cluster."""
        try:
            # Simple heuristic based on card properties
            synergy_type = cluster['synergy_type']
            
            # Get card properties
            card_type = self.cards_loader.get_card_type(card_id) or 'UNKNOWN'
            card_cost = self.cards_loader.get_card_cost(card_id) or 0
            
            # Basic fit evaluation
            fit_score = 0.0
            
            if synergy_type == 'beast' and card_type == 'MINION':
                fit_score = 0.7
            elif synergy_type == 'spell_damage' and card_type == 'SPELL':
                fit_score = 0.8
            elif synergy_type == 'weapon' and card_type == 'WEAPON':
                fit_score = 0.9
            elif synergy_type in ['heal', 'combo', 'outcast'] and card_type == 'SPELL':
                fit_score = 0.6
            elif card_type == 'MINION':
                fit_score = 0.3  # Base fit for minions
            
            return fit_score
            
        except Exception:
            return 0.0
    
    def _generate_synergy_meta_insights(self, clusters: Dict, hero_class: str) -> List[str]:
        """Generate meta insights about hero synergies."""
        insights = []
        
        try:
            if not clusters:
                insights.append(f"No strong synergy patterns identified for {hero_class}")
                return insights
            
            # Analyze cluster strengths
            strong_clusters = [name for name, data in clusters.items() if data['strength'] > 0.7]
            medium_clusters = [name for name, data in clusters.items() if 0.5 <= data['strength'] <= 0.7]
            
            if strong_clusters:
                insights.append(f"{hero_class} excels with {', '.join(strong_clusters)} synergies")
            
            if medium_clusters:
                insights.append(f"Secondary synergies: {', '.join(medium_clusters)}")
            
            # Draft advice
            if len(clusters) >= 3:
                insights.append(f"{hero_class} has diverse synergy options - adapt to draft flow")
            elif len(clusters) >= 1:
                primary = list(clusters.keys())[0]
                insights.append(f"Focus on {primary} synergy package for {hero_class}")
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error generating synergy insights: {e}")
            return [f"Synergy analysis completed for {hero_class}"]
    
    def _calculate_overall_synergy_potential(self, partners: List[Dict]) -> float:
        """Calculate overall synergy potential score."""
        if not partners:
            return 0.0
        
        # Weight by strength and count
        total_strength = sum(partner['synergy_strength'] for partner in partners[:5])  # Top 5
        potential_score = min(1.0, total_strength / 2.0)  # Normalize
        
        return round(potential_score, 3)
    
    def _generate_synergy_draft_advice(self, partners: List[Dict], existing: List[Dict]) -> List[str]:
        """Generate draft advice based on synergy analysis."""
        advice = []
        
        try:
            if existing:
                advice.append(f"Activated synergies: {len(existing)} found in current deck")
                top_existing = existing[0] if existing else None
                if top_existing:
                    advice.append(f"Strongest active synergy: {top_existing['partner_card']}")
            
            if partners:
                top_potential = partners[0]
                advice.append(f"Best synergy target: {top_potential['partner_card']} (strength: {top_potential['synergy_strength']:.2f})")
                
                high_synergy_count = len([p for p in partners if p['synergy_strength'] > 0.6])
                if high_synergy_count >= 3:
                    advice.append("Multiple strong synergy options available")
                elif high_synergy_count >= 1:
                    advice.append("Focus on high-synergy picks")
            
            return advice
            
        except Exception:
            return ["Synergy analysis completed"]
    
    def _get_fallback_hero_synergies(self, hero_class: str) -> Dict[str, Any]:
        """Fallback synergy analysis when data unavailable."""
        fallback_synergies = {
            'HUNTER': ['beast', 'weapon', 'aggressive'],
            'MAGE': ['spell_damage', 'freeze', 'tempo'],
            'PRIEST': ['heal', 'control', 'high_health'],
            'WARLOCK': ['demon', 'value', 'life_tap'],
            'WARRIOR': ['weapon', 'armor', 'control'],
            'PALADIN': ['divine_shield', 'buff', 'board'],
            'ROGUE': ['combo', 'weapon', 'tempo'],
            'SHAMAN': ['elemental', 'overload', 'spell'],
            'DRUID': ['ramp', 'choose_one', 'value'],
            'DEMONHUNTER': ['outcast', 'aggressive', 'demon']
        }
        
        hero_synergies = fallback_synergies.get(hero_class, ['general'])
        
        return {
            'hero_class': hero_class,
            'timestamp': datetime.now().isoformat(),
            'synergy_clusters': {
                synergy: {
                    'synergy_type': synergy,
                    'strength': 0.6,
                    'description': f"Fallback {synergy} synergy for {hero_class}",
                    'draft_priority': 'Medium'
                }
                for synergy in hero_synergies
            },
            'card_recommendations': {},
            'meta_insights': [
                f"Fallback synergy analysis for {hero_class}",
                f"Focus on {', '.join(hero_synergies)} themes",
                "Statistical synergy data unavailable"
            ],
            'confidence_level': 0.4
        }
    
    def _get_fallback_card_synergies(self, card_id: str, hero_class: str) -> Dict[str, Any]:
        """Fallback card synergy analysis."""
        return {
            'card_id': card_id,
            'hero_class': hero_class,
            'top_synergy_partners': [],
            'existing_deck_synergies': [],
            'synergy_potential_score': 0.3,
            'draft_recommendations': [
                f"Statistical synergy data unavailable for {card_id}",
                "Consider general card quality and curve fit"
            ]
        }
    
    # === HERO-SPECIFIC CURVE OPTIMIZATION METHODS ===
    
    def analyze_hero_specific_curve_optimization(self, hero_class: str, current_deck: List[str], 
                                               available_cards: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive hero-specific curve optimization analysis.
        
        Analyzes current deck curve against hero's optimal distribution and provides
        detailed recommendations for improving mana curve balance.
        """
        try:
            curve_analysis = {
                'hero_class': hero_class,
                'current_curve': self._analyze_current_curve(current_deck),
                'ideal_curve': self._get_enhanced_hero_ideal_curve(hero_class),
                'curve_gaps': self._identify_curve_gaps(current_deck, hero_class),
                'curve_optimization_score': 0.0,
                'recommendations': [],
                'priority_costs': [],
                'archetype_adjustments': self._get_archetype_curve_adjustments(hero_class)
            }
            
            # Calculate optimization score
            optimization_score = self._calculate_curve_optimization_score(current_deck, hero_class)
            curve_analysis['curve_optimization_score'] = optimization_score
            
            # Generate recommendations
            recommendations = self._generate_curve_recommendations(current_deck, hero_class, available_cards)
            curve_analysis['recommendations'] = recommendations
            
            # Identify priority costs
            priority_costs = self._identify_priority_mana_costs(current_deck, hero_class)
            curve_analysis['priority_costs'] = priority_costs
            
            return curve_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing curve optimization for {hero_class}: {e}")
            return self._get_fallback_curve_analysis(hero_class, current_deck)
    
    def get_card_curve_impact_analysis(self, card_id: str, hero_class: str, 
                                     current_deck: List[str]) -> Dict[str, Any]:
        """
        Analyze the curve impact of adding a specific card to the deck.
        
        Evaluates how a card would affect the overall mana curve balance
        and provides curve-specific scoring.
        """
        try:
            card_cost = self.cards_loader.get_card_cost(card_id) or 0
            
            # Current curve state
            current_curve = self._analyze_current_curve(current_deck)
            ideal_curve = self._get_enhanced_hero_ideal_curve(hero_class)
            
            # Projected curve after adding card
            projected_deck = current_deck + [card_id]
            projected_curve = self._analyze_current_curve(projected_deck)
            
            # Calculate improvement
            current_curve_score = self._score_curve_distribution(current_curve, ideal_curve)
            projected_curve_score = self._score_curve_distribution(projected_curve, ideal_curve)
            curve_improvement = projected_curve_score - current_curve_score
            
            # Analyze specific impact areas
            impact_analysis = {
                'card_id': card_id,
                'card_cost': card_cost,
                'curve_improvement': round(curve_improvement, 4),
                'fills_gap': self._card_fills_curve_gap(card_cost, current_curve, ideal_curve),
                'creates_overflow': self._card_creates_overflow(card_cost, current_curve, ideal_curve),
                'curve_balance_rating': self._rate_curve_balance_impact(curve_improvement),
                'timing_priority': self._get_curve_timing_priority(card_cost, hero_class),
                'archetype_fit': self._evaluate_card_archetype_curve_fit(card_id, hero_class),
                'explanation': self._explain_curve_impact(card_cost, curve_improvement, hero_class)
            }
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing curve impact for {card_id}: {e}")
            return self._get_fallback_curve_impact(card_id)
    
    def _get_enhanced_hero_ideal_curve(self, hero_class: str) -> Dict[str, Any]:
        """Get enhanced hero-specific ideal curve with archetype variations."""
        try:
            # Base curves with archetype variations
            hero_curves = {
                'HUNTER': {
                    'base': {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05, 7: 0.05},
                    'archetype_modifiers': {
                        'Aggro': {1: 0.20, 2: 0.30, 3: 0.25, 4: 0.15, 5: 0.05, 6: 0.03, 7: 0.02},
                        'Tempo': {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.10, 6: 0.05, 7: 0.00}
                    },
                    'key_costs': [1, 2, 3],
                    'playstyle': 'aggressive',
                    'curve_philosophy': 'Early game dominance with efficient threats'
                },
                'MAGE': {
                    'base': {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.20, 6: 0.10, 7: 0.05},
                    'archetype_modifiers': {
                        'Tempo': {1: 0.08, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.15, 6: 0.05, 7: 0.02},
                        'Control': {1: 0.02, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.18, 7: 0.10}
                    },
                    'key_costs': [3, 4, 5],
                    'playstyle': 'flexible',
                    'curve_philosophy': 'Mid-game value with spell synergy'
                },
                'PRIEST': {
                    'base': {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.15, 7: 0.10},
                    'archetype_modifiers': {
                        'Control': {1: 0.02, 2: 0.08, 3: 0.12, 4: 0.18, 5: 0.25, 6: 0.20, 7: 0.15},
                        'Value': {1: 0.05, 2: 0.12, 3: 0.18, 4: 0.22, 5: 0.23, 6: 0.15, 7: 0.05}
                    },
                    'key_costs': [4, 5, 6],
                    'playstyle': 'control',
                    'curve_philosophy': 'Late game inevitability with healing support'
                },
                'WARLOCK': {
                    'base': {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.20, 6: 0.10, 7: 0.05},
                    'archetype_modifiers': {
                        'Aggro': {1: 0.18, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.08, 6: 0.03, 7: 0.01},
                        'Control': {1: 0.02, 2: 0.10, 3: 0.15, 4: 0.22, 5: 0.25, 6: 0.16, 7: 0.10}
                    },
                    'key_costs': [2, 3, 4],
                    'playstyle': 'flexible',
                    'curve_philosophy': 'Life tap value engine with flexible curve'
                },
                'WARRIOR': {
                    'base': {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.15, 7: 0.05},
                    'archetype_modifiers': {
                        'Control': {1: 0.05, 2: 0.12, 3: 0.18, 4: 0.20, 5: 0.20, 6: 0.15, 7: 0.10},
                        'Tempo': {1: 0.12, 2: 0.20, 3: 0.23, 4: 0.22, 5: 0.15, 6: 0.08, 7: 0.00}
                    },
                    'key_costs': [3, 4, 5],
                    'playstyle': 'balanced',
                    'curve_philosophy': 'Weapon synergy with balanced threats'
                },
                'PALADIN': {
                    'base': {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.10, 7: 0.00},
                    'archetype_modifiers': {
                        'Aggro': {1: 0.15, 2: 0.28, 3: 0.25, 4: 0.20, 5: 0.10, 6: 0.02, 7: 0.00},
                        'Midrange': {1: 0.08, 2: 0.18, 3: 0.25, 4: 0.22, 5: 0.18, 6: 0.09, 7: 0.00}
                    },
                    'key_costs': [2, 3, 4],
                    'playstyle': 'midrange',
                    'curve_philosophy': 'Board control with buff synergy'
                },
                'ROGUE': {
                    'base': {1: 0.10, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.05, 7: 0.00},
                    'archetype_modifiers': {
                        'Tempo': {1: 0.12, 2: 0.28, 3: 0.28, 4: 0.20, 5: 0.10, 6: 0.02, 7: 0.00},
                        'Combo': {1: 0.08, 2: 0.22, 3: 0.25, 4: 0.25, 5: 0.15, 6: 0.05, 7: 0.00}
                    },
                    'key_costs': [2, 3, 4],
                    'playstyle': 'tempo',
                    'curve_philosophy': 'Efficient tempo with combo potential'
                },
                'SHAMAN': {
                    'base': {1: 0.10, 2: 0.20, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.10, 7: 0.05},
                    'archetype_modifiers': {
                        'Elemental': {1: 0.12, 2: 0.22, 3: 0.22, 4: 0.22, 5: 0.15, 6: 0.07, 7: 0.00},
                        'Overload': {1: 0.08, 2: 0.18, 3: 0.25, 4: 0.20, 5: 0.18, 6: 0.08, 7: 0.03}
                    },
                    'key_costs': [2, 3, 4],
                    'playstyle': 'balanced',
                    'curve_philosophy': 'Elemental synergy with overload timing'
                },
                'DRUID': {
                    'base': {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.15, 5: 0.20, 6: 0.20, 7: 0.15},
                    'archetype_modifiers': {
                        'Ramp': {1: 0.02, 2: 0.08, 3: 0.12, 4: 0.15, 5: 0.18, 6: 0.25, 7: 0.20},
                        'Tempo': {1: 0.08, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.20, 6: 0.12, 7: 0.03}
                    },
                    'key_costs': [5, 6, 7],
                    'playstyle': 'ramp',
                    'curve_philosophy': 'Ramp acceleration to expensive threats'
                },
                'DEMONHUNTER': {
                    'base': {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05, 7: 0.05},
                    'archetype_modifiers': {
                        'Aggro': {1: 0.20, 2: 0.30, 3: 0.25, 4: 0.15, 5: 0.08, 6: 0.02, 7: 0.00},
                        'Tempo': {1: 0.12, 2: 0.22, 3: 0.28, 4: 0.20, 5: 0.13, 6: 0.05, 7: 0.00}
                    },
                    'key_costs': [1, 2, 3],
                    'playstyle': 'aggressive',
                    'curve_philosophy': 'Early pressure with outcast synergy'
                }
            }
            
            return hero_curves.get(hero_class, hero_curves['WARRIOR'])
            
        except Exception as e:
            self.logger.warning(f"Error getting enhanced curve for {hero_class}: {e}")
            return self._get_hero_ideal_curve(hero_class)
    
    def _analyze_current_curve(self, deck_cards: List[str]) -> Dict[int, Dict[str, Any]]:
        """Analyze current deck's mana curve distribution."""
        try:
            curve_analysis = {}
            total_cards = len(deck_cards)
            
            if total_cards == 0:
                return {}
            
            # Count cards by cost
            cost_counts = {}
            for card_id in deck_cards:
                cost = self.cards_loader.get_card_cost(card_id) or 0
                cost_counts[cost] = cost_counts.get(cost, 0) + 1
            
            # Analyze each cost slot
            for cost in range(0, 8):  # 0-7+ mana
                count = cost_counts.get(cost, 0)
                percentage = (count / total_cards) * 100
                
                curve_analysis[cost] = {
                    'count': count,
                    'percentage': round(percentage, 2),
                    'ratio': round(count / total_cards, 3) if total_cards > 0 else 0,
                    'cards': [card for card in deck_cards 
                             if (self.cards_loader.get_card_cost(card) or 0) == cost]
                }
            
            return curve_analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing current curve: {e}")
            return {}
    
    def _identify_curve_gaps(self, deck_cards: List[str], hero_class: str) -> List[Dict[str, Any]]:
        """Identify gaps in the mana curve compared to hero's ideal."""
        try:
            current_curve = self._analyze_current_curve(deck_cards)
            ideal_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            ideal_curve = ideal_curve_data.get('base', {})
            
            gaps = []
            
            for cost, ideal_ratio in ideal_curve.items():
                current_ratio = current_curve.get(cost, {}).get('ratio', 0)
                difference = ideal_ratio - current_ratio
                
                if difference > 0.05:  # Significant gap threshold
                    gap_severity = 'Critical' if difference > 0.15 else 'Major' if difference > 0.10 else 'Minor'
                    
                    gaps.append({
                        'cost': cost,
                        'ideal_ratio': round(ideal_ratio, 3),
                        'current_ratio': round(current_ratio, 3),
                        'deficit': round(difference, 3),
                        'severity': gap_severity,
                        'cards_needed': max(1, round(difference * 30)),  # Approximate cards needed
                        'priority': self._calculate_gap_priority(cost, difference, hero_class)
                    })
            
            # Sort by priority
            gaps.sort(key=lambda x: x['priority'], reverse=True)
            
            return gaps
            
        except Exception as e:
            self.logger.warning(f"Error identifying curve gaps: {e}")
            return []
    
    def _calculate_curve_optimization_score(self, deck_cards: List[str], hero_class: str) -> float:
        """Calculate overall curve optimization score."""
        try:
            if not deck_cards:
                return 0.5  # Neutral score for empty deck
            
            current_curve = self._analyze_current_curve(deck_cards)
            ideal_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            ideal_curve = ideal_curve_data.get('base', {})
            
            return self._score_curve_distribution(current_curve, ideal_curve)
            
        except Exception as e:
            self.logger.warning(f"Error calculating curve optimization score: {e}")
            return 0.5
    
    def _score_curve_distribution(self, current_curve: Dict, ideal_curve: Dict) -> float:
        """Score how well current curve matches ideal distribution."""
        try:
            total_deviation = 0.0
            total_weight = 0.0
            
            for cost, ideal_ratio in ideal_curve.items():
                current_ratio = current_curve.get(cost, {}).get('ratio', 0)
                deviation = abs(ideal_ratio - current_ratio)
                weight = ideal_ratio + 0.1  # Weight by importance
                
                total_deviation += deviation * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.5
            
            # Convert deviation to score (lower deviation = higher score)
            normalized_deviation = total_deviation / total_weight
            score = max(0.0, 1.0 - (normalized_deviation * 2))  # Scale appropriately
            
            return round(score, 3)
            
        except Exception:
            return 0.5
    
    def _generate_curve_recommendations(self, deck_cards: List[str], hero_class: str, 
                                      available_cards: List[str] = None) -> List[str]:
        """Generate specific curve optimization recommendations."""
        recommendations = []
        
        try:
            gaps = self._identify_curve_gaps(deck_cards, hero_class)
            curve_score = self._calculate_curve_optimization_score(deck_cards, hero_class)
            
            # Overall assessment
            if curve_score >= 0.8:
                recommendations.append("Excellent curve distribution - maintain balance")
            elif curve_score >= 0.6:
                recommendations.append("Good curve shape - minor adjustments needed")
            elif curve_score >= 0.4:
                recommendations.append("Curve needs improvement - focus on identified gaps")
            else:
                recommendations.append("Poor curve distribution - significant restructuring needed")
            
            # Specific gap recommendations
            if gaps:
                critical_gaps = [g for g in gaps if g['severity'] == 'Critical']
                major_gaps = [g for g in gaps if g['severity'] == 'Major']
                
                if critical_gaps:
                    costs = [str(g['cost']) for g in critical_gaps[:2]]
                    recommendations.append(f"Critical gaps at {', '.join(costs)} mana - highest priority")
                
                if major_gaps:
                    costs = [str(g['cost']) for g in major_gaps[:2]]
                    recommendations.append(f"Major gaps at {', '.join(costs)} mana - high priority")
            
            # Hero-specific advice
            hero_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            key_costs = hero_curve_data.get('key_costs', [])
            philosophy = hero_curve_data.get('curve_philosophy', '')
            
            if key_costs:
                recommendations.append(f"Focus on {key_costs} mana costs for {hero_class}")
            
            if philosophy:
                recommendations.append(f"Curve philosophy: {philosophy}")
            
            # Available cards analysis
            if available_cards:
                helpful_cards = self._find_curve_helpful_cards(deck_cards, hero_class, available_cards)
                if helpful_cards:
                    recommendations.append(f"Available curve improvements: {', '.join(helpful_cards[:3])}")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating curve recommendations: {e}")
            return [f"Curve analysis completed for {hero_class}"]
    
    def _identify_priority_mana_costs(self, deck_cards: List[str], hero_class: str) -> List[Dict[str, Any]]:
        """Identify priority mana costs for drafting."""
        try:
            gaps = self._identify_curve_gaps(deck_cards, hero_class)
            hero_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            key_costs = hero_curve_data.get('key_costs', [])
            
            priorities = []
            
            # Add gaps as priorities
            for gap in gaps[:3]:  # Top 3 gaps
                priorities.append({
                    'cost': gap['cost'],
                    'reason': f"Gap: need {gap['cards_needed']} more cards",
                    'priority_level': gap['severity'],
                    'urgency': 'High' if gap['severity'] == 'Critical' else 'Medium'
                })
            
            # Add hero key costs if not already covered
            for cost in key_costs:
                if not any(p['cost'] == cost for p in priorities):
                    current_curve = self._analyze_current_curve(deck_cards)
                    current_count = current_curve.get(cost, {}).get('count', 0)
                    
                    if current_count < 3:  # Need more cards at this cost
                        priorities.append({
                            'cost': cost,
                            'reason': f"Hero key cost: {hero_class} strength",
                            'priority_level': 'Major',
                            'urgency': 'Medium'
                        })
            
            return priorities[:5]  # Return top 5 priorities
            
        except Exception as e:
            self.logger.warning(f"Error identifying priority costs: {e}")
            return []
    
    def _get_archetype_curve_adjustments(self, hero_class: str) -> Dict[str, Any]:
        """Get archetype-specific curve adjustments for hero."""
        try:
            hero_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            archetype_modifiers = hero_curve_data.get('archetype_modifiers', {})
            
            adjustments = {}
            
            for archetype, curve in archetype_modifiers.items():
                base_curve = hero_curve_data.get('base', {})
                
                # Calculate major differences
                significant_changes = {}
                for cost, ratio in curve.items():
                    base_ratio = base_curve.get(cost, 0)
                    difference = ratio - base_ratio
                    
                    if abs(difference) > 0.03:  # Significant change threshold
                        change_type = 'increase' if difference > 0 else 'decrease'
                        significant_changes[cost] = {
                            'change': round(difference, 3),
                            'type': change_type,
                            'magnitude': 'major' if abs(difference) > 0.08 else 'moderate'
                        }
                
                if significant_changes:
                    adjustments[archetype] = {
                        'curve_changes': significant_changes,
                        'description': self._describe_archetype_curve_change(archetype, significant_changes)
                    }
            
            return adjustments
            
        except Exception as e:
            self.logger.warning(f"Error getting archetype adjustments: {e}")
            return {}
    
    def _calculate_gap_priority(self, cost: int, deficit: float, hero_class: str) -> float:
        """Calculate priority score for filling a curve gap."""
        try:
            # Base priority from deficit size
            base_priority = deficit * 10
            
            # Hero key cost bonus
            hero_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            key_costs = hero_curve_data.get('key_costs', [])
            
            if cost in key_costs:
                base_priority *= 1.5
            
            # Early game bonus (costs 1-3 are generally important)
            if 1 <= cost <= 3:
                base_priority *= 1.2
            
            # Mid game stability (costs 4-5)
            elif 4 <= cost <= 5:
                base_priority *= 1.1
            
            return round(base_priority, 2)
            
        except Exception:
            return 1.0
    
    def _card_fills_curve_gap(self, card_cost: int, current_curve: Dict, ideal_curve: Dict) -> bool:
        """Check if card fills a significant curve gap."""
        try:
            current_ratio = current_curve.get(card_cost, {}).get('ratio', 0)
            ideal_ratio = ideal_curve.get(card_cost, 0)
            
            return (ideal_ratio - current_ratio) > 0.05
            
        except Exception:
            return False
    
    def _card_creates_overflow(self, card_cost: int, current_curve: Dict, ideal_curve: Dict) -> bool:
        """Check if card creates overflow at this cost."""
        try:
            current_ratio = current_curve.get(card_cost, {}).get('ratio', 0)
            ideal_ratio = ideal_curve.get(card_cost, 0)
            
            # After adding this card
            total_cards = sum(slot.get('count', 0) for slot in current_curve.values())
            projected_ratio = (current_curve.get(card_cost, {}).get('count', 0) + 1) / (total_cards + 1)
            
            return projected_ratio > (ideal_ratio * 1.3)  # 30% over ideal
            
        except Exception:
            return False
    
    def _rate_curve_balance_impact(self, improvement: float) -> str:
        """Rate the curve balance impact of adding a card."""
        if improvement > 0.05:
            return "Excellent"
        elif improvement > 0.02:
            return "Good"
        elif improvement > -0.02:
            return "Neutral"
        elif improvement > -0.05:
            return "Poor"
        else:
            return "Harmful"
    
    def _get_curve_timing_priority(self, cost: int, hero_class: str) -> str:
        """Get timing priority for this cost slot."""
        try:
            hero_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            key_costs = hero_curve_data.get('key_costs', [])
            playstyle = hero_curve_data.get('playstyle', 'balanced')
            
            if cost in key_costs:
                return "Critical"
            elif playstyle == 'aggressive' and cost <= 3:
                return "High"
            elif playstyle == 'control' and cost >= 5:
                return "High"
            elif 2 <= cost <= 4:  # Generally important
                return "Medium"
            else:
                return "Low"
                
        except Exception:
            return "Medium"
    
    def _evaluate_card_archetype_curve_fit(self, card_id: str, hero_class: str) -> str:
        """Evaluate how well card fits hero's archetype curve needs."""
        try:
            card_cost = self.cards_loader.get_card_cost(card_id) or 0
            hero_curve_data = self._get_enhanced_hero_ideal_curve(hero_class)
            playstyle = hero_curve_data.get('playstyle', 'balanced')
            
            if playstyle == 'aggressive' and card_cost <= 3:
                return "Excellent fit"
            elif playstyle == 'control' and card_cost >= 5:
                return "Excellent fit"
            elif playstyle == 'tempo' and 2 <= card_cost <= 4:
                return "Excellent fit"
            elif playstyle == 'ramp' and card_cost >= 6:
                return "Excellent fit"
            else:
                return "Decent fit"
                
        except Exception:
            return "Unknown fit"
    
    def _explain_curve_impact(self, card_cost: int, improvement: float, hero_class: str) -> str:
        """Generate explanation for curve impact."""
        try:
            if improvement > 0.03:
                return f"Significantly improves {hero_class} curve at {card_cost} mana"
            elif improvement > 0.01:
                return f"Helps balance {hero_class} curve at {card_cost} mana"
            elif improvement > -0.01:
                return f"Neutral curve impact at {card_cost} mana"
            elif improvement > -0.03:
                return f"Slightly overloads {card_cost} mana slot"
            else:
                return f"Creates significant overflow at {card_cost} mana"
                
        except Exception:
            return f"Curve impact analysis for {card_cost} mana"
    
    def _find_curve_helpful_cards(self, deck_cards: List[str], hero_class: str, 
                                available_cards: List[str]) -> List[str]:
        """Find available cards that would help curve balance."""
        try:
            gaps = self._identify_curve_gaps(deck_cards, hero_class)
            helpful_cards = []
            
            # Get costs where we have gaps
            gap_costs = [gap['cost'] for gap in gaps if gap['severity'] in ['Critical', 'Major']]
            
            for card_id in available_cards:
                card_cost = self.cards_loader.get_card_cost(card_id) or 0
                if card_cost in gap_costs:
                    helpful_cards.append(card_id)
            
            return helpful_cards[:5]  # Return top 5
            
        except Exception:
            return []
    
    def _describe_archetype_curve_change(self, archetype: str, changes: Dict) -> str:
        """Describe how archetype changes the curve."""
        try:
            descriptions = []
            
            for cost, change_data in changes.items():
                change_type = change_data['type']
                magnitude = change_data['magnitude']
                
                if change_type == 'increase':
                    descriptions.append(f"More {cost}-cost cards")
                else:
                    descriptions.append(f"Fewer {cost}-cost cards")
            
            return f"{archetype}: " + ", ".join(descriptions[:3])
            
        except Exception:
            return f"{archetype} curve adjustments"
    
    def _get_fallback_curve_analysis(self, hero_class: str, deck_cards: List[str]) -> Dict[str, Any]:
        """Fallback curve analysis when errors occur."""
        return {
            'hero_class': hero_class,
            'current_curve': {},
            'ideal_curve': self._get_hero_ideal_curve(hero_class),
            'curve_gaps': [],
            'curve_optimization_score': 0.5,
            'recommendations': [
                f"Curve analysis unavailable for {hero_class}",
                "Focus on balanced mana distribution",
                "Prioritize 2-4 mana cards for stability"
            ],
            'priority_costs': [],
            'archetype_adjustments': {}
        }
    
    def _get_fallback_curve_impact(self, card_id: str) -> Dict[str, Any]:
        """Fallback curve impact analysis."""
        card_cost = self.cards_loader.get_card_cost(card_id) or 0
        
        return {
            'card_id': card_id,
            'card_cost': card_cost,
            'curve_improvement': 0.0,
            'fills_gap': False,
            'creates_overflow': False,
            'curve_balance_rating': 'Unknown',
            'timing_priority': 'Medium',
            'archetype_fit': 'Unknown',
            'explanation': f"Curve impact analysis unavailable for {card_id}"
        }
    
    def _create_fallback_scores(self, error_message: str) -> DimensionalScores:
        """Create fallback DimensionalScores when evaluation fails."""
        return DimensionalScores(
            card_id="fallback_card",
            base_value=0.5,
            tempo_score=0.5,
            value_score=0.5,
            synergy_score=0.5,
            curve_score=0.5,
            re_draftability_score=0.5,
            greed_score=0.5,
            confidence=0.1,  # Low confidence for fallback
            data_quality_score=0.1
        )
    
    def _prewarm_caches(self) -> None:
        """Pre-warm caches with common card data for better performance."""
        try:
            # Pre-populate card data cache with frequently accessed data
            if hasattr(self.cards_loader, 'cards_data') and self.cards_loader.cards_data:
                common_cards = list(self.cards_loader.cards_data.keys())[:1000]  # Top 1000 cards
                for card_id in common_cards:
                    self._get_cached_card_data(card_id)
                
                self.logger.info(f"Pre-warmed card data cache with {len(common_cards)} cards")
            
            # Pre-analyze card text for common patterns
            self._preanalyze_card_texts()
            
        except Exception as e:
            self.logger.warning(f"Cache pre-warming failed: {e}")
    
    def _preanalyze_card_texts(self) -> None:
        """Pre-analyze card texts for common keywords and effects."""
        if not hasattr(self.cards_loader, 'cards_data'):
            return
        
        keywords_to_analyze = [
            'charge', 'rush', 'battlecry', 'taunt', 'draw', 'discover', 
            'lifesteal', 'deathrattle', 'damage', 'destroy', 'create'
        ]
        
        analyzed_count = 0
        for card_id, card_data in self.cards_loader.cards_data.items():
            if not isinstance(card_data, dict):
                continue
                
            text = card_data.get('text', '').lower()
            if not text:
                continue
            
            text_features = {}
            for keyword in keywords_to_analyze:
                text_features[keyword] = keyword in text
            
            # Additional analysis
            text_features['draw_count'] = text.count('draw')
            text_features['has_generation'] = any(gen in text for gen in ['add a random', 'create', 'discover'])
            text_features['is_removal'] = any(rem in text for rem in ['damage', 'destroy', 'silence'])
            
            self.text_analysis_cache[card_id] = text_features
            analyzed_count += 1
            
            # Limit cache size
            if analyzed_count >= 5000:
                break
        
        self.logger.info(f"Pre-analyzed {analyzed_count} card texts for keywords")
    
    def _generate_cache_key(self, card_id: str, deck_state: DeckState) -> str:
        """Generate cache key for evaluation caching."""
        # Include key factors that affect evaluation
        hero_class = deck_state.hero_class if deck_state else 'NEUTRAL'
        archetype = getattr(deck_state, 'archetype', 'Unknown')
        pick_number = getattr(deck_state, 'pick_number', 0)
        
        # Group pick numbers for cache efficiency (0-10, 11-20, 21-30)
        pick_group = min(pick_number // 10, 2)
        
        return f"{card_id}|{hero_class}|{archetype}|{pick_group}"
    
    def _get_cached_evaluation(self, cache_key: str) -> Optional[DimensionalScores]:
        """Get cached evaluation result if available and fresh."""
        if cache_key not in self.evaluation_cache:
            return None
        
        cached_entry = self.evaluation_cache[cache_key]
        cache_time = cached_entry.get('timestamp', datetime.min)
        
        # Check if cache is still fresh
        age_minutes = (datetime.now() - cache_time).total_seconds() / 60
        if age_minutes > self.cache_ttl_minutes:
            # Remove expired entry
            del self.evaluation_cache[cache_key]
            return None
        
        return cached_entry.get('result')
    
    def _cache_evaluation(self, cache_key: str, result: DimensionalScores) -> None:
        """Cache evaluation result with timestamp."""
        # Implement cache size management
        if len(self.evaluation_cache) >= self.max_cache_size:
            self._cleanup_old_cache_entries()
        
        self.evaluation_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
    
    def _cleanup_old_cache_entries(self) -> None:
        """Remove old cache entries to manage memory usage."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.evaluation_cache.items():
            cache_time = entry.get('timestamp', datetime.min)
            age_minutes = (current_time - cache_time).total_seconds() / 60
            
            if age_minutes > self.cache_ttl_minutes:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.evaluation_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.evaluation_cache) >= self.max_cache_size:
            # Sort by timestamp and keep newest half
            entries_by_time = sorted(
                self.evaluation_cache.items(),
                key=lambda x: x[1].get('timestamp', datetime.min),
                reverse=True
            )
            
            keep_count = self.max_cache_size // 2
            self.evaluation_cache = dict(entries_by_time[:keep_count])
        
        self.logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    
    def _get_cached_card_data(self, card_id: str) -> Dict[str, Any]:
        """Get cached card data for performance optimization."""
        if card_id in self.card_data_cache:
            return self.card_data_cache[card_id]
        
        # Fetch and cache card data
        if hasattr(self.cards_loader, 'cards_data') and self.cards_loader.cards_data:
            card_data = self.cards_loader.cards_data.get(card_id, {})
        else:
            card_data = {}
        
        # Cache with reasonable size limit
        if len(self.card_data_cache) < 10000:
            self.card_data_cache[card_id] = card_data
        
        return card_data
    
    def _get_cached_text_analysis(self, card_id: str) -> Dict[str, Any]:
        """Get cached text analysis for performance optimization."""
        if card_id in self.text_analysis_cache:
            return self.text_analysis_cache[card_id]
        
        # Fallback to on-demand analysis
        card_data = self._get_cached_card_data(card_id)
        text = card_data.get('text', '').lower()
        
        text_features = {
            'charge': 'charge' in text,
            'rush': 'rush' in text,
            'battlecry': 'battlecry' in text,
            'taunt': 'taunt' in text,
            'draw': 'draw' in text,
            'discover': 'discover' in text,
            'lifesteal': 'lifesteal' in text,
            'deathrattle': 'deathrattle' in text,
            'draw_count': text.count('draw'),
            'has_generation': any(gen in text for gen in ['add a random', 'create', 'discover']),
            'is_removal': any(rem in text for rem in ['damage', 'destroy', 'silence'])
        }
        
        # Cache if space available
        if len(self.text_analysis_cache) < 5000:
            self.text_analysis_cache[card_id] = text_features
        
        return text_features
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_miss_count
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'evaluation_cache_size': len(self.evaluation_cache),
            'card_data_cache_size': len(self.card_data_cache),
            'text_analysis_cache_size': len(self.text_analysis_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_miss_count,
            'hit_rate_percentage': round(hit_rate, 2),
            'total_evaluations': self.evaluation_count
        }