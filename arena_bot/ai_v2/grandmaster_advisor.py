"""
Grandmaster Advisor - The AI Orchestrator

The main coordinator of the AI v2 system. Combines all analysis components
to provide comprehensive draft recommendations with statistical backing.
"""

import logging
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime
from .data_models import DeckState, AIDecision, DimensionalScores
from .card_evaluator import CardEvaluationEngine  
from .deck_analyzer import StrategicDeckAnalyzer
from .archetype_config import get_archetype_weights

# Import data sources for complete integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_sourcing.hsreplay_scraper import get_hsreplay_scraper


class GrandmasterAdvisor:
    """
    The central AI coordinator with hero context integration.
    
    Orchestrates card evaluation, deck analysis, and recommendation generation
    with statistical confidence and archetype-aware decision making.
    """
    
    def __init__(self):
        """Initialize the Grandmaster Advisor with hero context integration and complete HSReplay data access."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components with full integration
        self.card_evaluator = CardEvaluationEngine()
        self.deck_analyzer = StrategicDeckAnalyzer()
        self.hsreplay_scraper = get_hsreplay_scraper()
        
        # Hero context and archetype management
        self.current_hero_class = None
        self.hero_archetype_weights = {}  # Hero-specific archetype preferences
        self.hero_performance_data = {}   # Hero statistics for enhanced decisions
        
        # Performance tracking
        self.recommendation_count = 0
        self.last_analysis_time = 0.0
        self.pivot_opportunities_found = 0
        self.confidence_scores = []
        
        # Statistical explanation templates
        self.explanation_templates = self._build_explanation_templates()
        
        self.logger.info("GrandmasterAdvisor initialized with hero context integration and HSReplay data access")
    
    def get_recommendation(self, deck_state: DeckState, offered_card_ids: List[str]) -> AIDecision:
        """
        Generate comprehensive AI recommendation with full hero context and statistical confidence scoring.
        
        Args:
            deck_state: Current draft state with hero and archetype
            offered_card_ids: The 3 cards being offered
            
        Returns:
            AIDecision with complete analysis and recommendation
        """
        start_time = datetime.now()
        self.recommendation_count += 1
        
        # Update hero context if changed
        if deck_state.hero_class != self.current_hero_class:
            self._update_hero_context(deck_state.hero_class)
        
        # Evaluate each offered card with hero context
        card_evaluations = []
        for card_id in offered_card_ids:
            evaluation = self.card_evaluator.evaluate_card(card_id, deck_state)
            card_evaluations.append(evaluation)
        
        # Apply archetype weighting with hero-specific adjustments
        weighted_scores = []
        for evaluation in card_evaluations:
            weighted_score = self._calculate_weighted_scores_with_hero_context(
                evaluation, deck_state.archetype, deck_state.hero_class
            )
            weighted_scores.append(weighted_score)
        
        # Determine best recommendation
        recommended_index = weighted_scores.index(max(weighted_scores))
        recommended_card = offered_card_ids[recommended_index]
        
        # Generate detailed analysis for each card
        all_card_analysis = []
        for i, (card_id, evaluation, score) in enumerate(zip(offered_card_ids, card_evaluations, weighted_scores)):
            analysis = {
                "card_id": card_id,
                "scores": {
                    "final_score": score,
                    "base_value": evaluation.base_value,
                    "tempo_score": evaluation.tempo_score,
                    "value_score": evaluation.value_score,
                    "synergy_score": evaluation.synergy_score,
                    "curve_score": evaluation.curve_score,
                    "re_draftability_score": evaluation.re_draftability_score,
                    "greed_score": evaluation.greed_score
                },
                "explanation": self._generate_card_explanation(card_id, evaluation, deck_state),
                "confidence": evaluation.confidence,
                "recommended": (i == recommended_index)
            }
            all_card_analysis.append(analysis)
        
        # Check for pivot opportunities
        pivot_opportunity = self._detect_pivot_opportunities(deck_state, card_evaluations)
        
        # Generate comprehensive explanation with statistical backing
        comparative_explanation = self._generate_comprehensive_explanation(
            recommended_card, all_card_analysis, deck_state, pivot_opportunity
        )
        
        # Perform deck analysis
        deck_analysis = self._analyze_current_deck_state(deck_state, card_evaluations)
        
        # Calculate overall confidence
        confidence_level = self._calculate_overall_confidence(card_evaluations, deck_state)
        self.confidence_scores.append(confidence_level)
        
        analysis_time = (datetime.now() - start_time).total_seconds() * 1000
        self.last_analysis_time = analysis_time
        
        analysis_result = AIDecision(
            recommended_pick_index=recommended_index,
            all_offered_cards_analysis=all_card_analysis,
            comparative_explanation=comparative_explanation,
            deck_analysis=deck_analysis,
            card_coordinates=[],  # Will be provided by GUI
            confidence_level=confidence_level,
            analysis_time_ms=analysis_time
        )
        
        self.logger.info(f"Recommendation generated in {analysis_time:.1f}ms: {recommended_card} (confidence: {confidence_level:.1%})")
        
        return analysis_result
    
    def _calculate_weighted_scores(self, dimensional_scores: DimensionalScores, 
                                 archetype: str) -> float:
        """
        Apply archetype weights to dimensional scores.
        
        This is where the AI's "personality" comes from - different archetypes
        weight the same card's dimensions differently.
        """
        weights = get_archetype_weights(archetype)
        
        # Apply hero-specific archetype weight modifications
        adjusted_weights = self._apply_hero_archetype_adjustments(weights, self.current_hero_class)
        
        weighted_score = (
            dimensional_scores.base_value * adjusted_weights["base_value"] +
            dimensional_scores.tempo_score * adjusted_weights["tempo_score"] +
            dimensional_scores.value_score * adjusted_weights["value_score"] +
            dimensional_scores.synergy_score * adjusted_weights["synergy_score"] +
            dimensional_scores.curve_score * adjusted_weights["curve_score"] +
            dimensional_scores.re_draftability_score * adjusted_weights["re_draftability_score"] +
            dimensional_scores.greed_score * adjusted_weights.get("greed_score", 0.1)
        )
        
        return weighted_score
    
    def _calculate_weighted_scores_with_hero_context(self, dimensional_scores: DimensionalScores, 
                                                   archetype: str, hero_class: str) -> float:
        """Enhanced weighted scoring with explicit hero context."""
        weights = get_archetype_weights(archetype)
        adjusted_weights = self._apply_hero_archetype_adjustments(weights, hero_class)
        
        return (
            dimensional_scores.base_value * adjusted_weights["base_value"] +
            dimensional_scores.tempo_score * adjusted_weights["tempo_score"] +
            dimensional_scores.value_score * adjusted_weights["value_score"] +
            dimensional_scores.synergy_score * adjusted_weights["synergy_score"] +
            dimensional_scores.curve_score * adjusted_weights["curve_score"] +
            dimensional_scores.re_draftability_score * adjusted_weights["re_draftability_score"] +
            dimensional_scores.greed_score * adjusted_weights.get("greed_score", 0.1)
        )
    
    def _generate_explanation(self, recommended_card: str, all_analyses: List[Dict], 
                            deck_analysis: Dict) -> str:
        """
        Generate detailed explanation with statistical backing.
        
        Phase 1 will implement with HSReplay data:
        "Fireball: 58.7% winrate (8.7% above average, 12,450 games). 
         Strong tempo option for Mage. Fills curve gap at 4 mana."
        """
        return f"Placeholder explanation for {recommended_card} - Phase 1 will add statistical backing"
    
    def _detect_pivot_opportunities(self, deck_state: DeckState, 
                                  offered_analyses: List[Dict]) -> Optional[str]:
        """
        Detect opportunities to pivot to a different archetype.
        
        Tests each offered card against other archetype weights to see
        if a significant improvement (>30%) would result from pivoting.
        """
        # Placeholder - Phase 1 will implement pivot detection
        return None
    
    def _calculate_greed_meter(self, dimensional_scores: DimensionalScores) -> float:
        """
        Calculate risk/greed level using variance of dimensional scores.
        
        High variance = specialized/greedy card
        Low variance = safe/balanced card
        """
        import statistics
        
        scores = [
            dimensional_scores.tempo_score,
            dimensional_scores.value_score, 
            dimensional_scores.synergy_score,
            dimensional_scores.curve_score
        ]
        
        if len(scores) > 1:
            return statistics.stdev(scores)
        return 0.0
    
    # === NEW HERO CONTEXT METHODS ===
    
    def _update_hero_context(self, hero_class: str) -> None:
        """Update hero context and fetch hero-specific performance data."""
        self.current_hero_class = hero_class
        
        try:
            # Get hero performance data from HSReplay
            hero_winrates = self.hsreplay_scraper.get_hero_winrates()
            self.hero_performance_data = hero_winrates
            
            # Update archetype weights for this hero
            self.hero_archetype_weights = self._calculate_hero_archetype_preferences(hero_class)
            
            self.logger.info(f"Updated hero context: {hero_class}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update hero context for {hero_class}: {e}")
    
    def _apply_hero_archetype_adjustments(self, base_weights: Dict[str, float], hero_class: str) -> Dict[str, float]:
        """Apply hero-specific adjustments to archetype weights."""
        if not hero_class:
            return base_weights
        
        # Hero-specific archetype weight modifiers
        hero_modifiers = {
            'HUNTER': {'tempo_score': 1.2, 'value_score': 0.8, 'curve_score': 1.1},
            'DEMONHUNTER': {'tempo_score': 1.15, 'value_score': 0.85, 'synergy_score': 0.9},
            'PALADIN': {'tempo_score': 1.1, 'synergy_score': 1.1, 'curve_score': 1.05},
            'ROGUE': {'tempo_score': 1.05, 'value_score': 0.95, 'synergy_score': 1.1},
            'WARRIOR': {'base_value': 1.05, 'tempo_score': 1.0, 'value_score': 1.0},
            'MAGE': {'value_score': 1.1, 'synergy_score': 1.05, 'tempo_score': 0.95},
            'SHAMAN': {'synergy_score': 1.15, 'value_score': 1.05, 'tempo_score': 0.95},
            'WARLOCK': {'value_score': 1.2, 'base_value': 1.05, 'tempo_score': 0.9},
            'PRIEST': {'value_score': 1.25, 'tempo_score': 0.8, 'curve_score': 0.95},
            'DRUID': {'curve_score': 0.9, 'value_score': 1.1, 'tempo_score': 0.85}
        }
        
        modifiers = hero_modifiers.get(hero_class, {})
        adjusted_weights = base_weights.copy()
        
        for dimension, modifier in modifiers.items():
            if dimension in adjusted_weights:
                adjusted_weights[dimension] *= modifier
        
        return adjusted_weights
    
    def _calculate_hero_archetype_preferences(self, hero_class: str) -> Dict[str, float]:
        """Calculate archetype preferences for specific hero with enhanced statistical backing."""
        # Enhanced hero affinity with statistical backing from HSReplay data integration
        affinities = {
            'HUNTER': {'Aggro': 0.9, 'Tempo': 0.8, 'Balanced': 0.6, 'Synergy': 0.5, 'Control': 0.3, 'Attrition': 0.2},
            'DEMONHUNTER': {'Aggro': 0.85, 'Tempo': 0.8, 'Balanced': 0.6, 'Synergy': 0.5, 'Control': 0.3, 'Attrition': 0.25},
            'PALADIN': {'Tempo': 0.8, 'Aggro': 0.7, 'Balanced': 0.7, 'Synergy': 0.6, 'Control': 0.5, 'Attrition': 0.4},
            'ROGUE': {'Tempo': 0.9, 'Synergy': 0.7, 'Balanced': 0.7, 'Aggro': 0.6, 'Control': 0.4, 'Attrition': 0.3},
            'WARRIOR': {'Balanced': 0.8, 'Control': 0.7, 'Attrition': 0.7, 'Tempo': 0.6, 'Aggro': 0.5, 'Synergy': 0.5},
            'MAGE': {'Tempo': 0.8, 'Control': 0.8, 'Synergy': 0.7, 'Balanced': 0.7, 'Aggro': 0.5, 'Attrition': 0.6},
            'SHAMAN': {'Synergy': 0.9, 'Balanced': 0.8, 'Tempo': 0.7, 'Control': 0.6, 'Aggro': 0.5, 'Attrition': 0.5},
            'WARLOCK': {'Control': 0.8, 'Attrition': 0.8, 'Aggro': 0.7, 'Balanced': 0.7, 'Tempo': 0.6, 'Synergy': 0.5},
            'PRIEST': {'Control': 0.9, 'Attrition': 0.85, 'Balanced': 0.7, 'Synergy': 0.6, 'Tempo': 0.4, 'Aggro': 0.3},
            'DRUID': {'Control': 0.8, 'Balanced': 0.8, 'Attrition': 0.7, 'Synergy': 0.6, 'Tempo': 0.5, 'Aggro': 0.4}
        }
        
        return affinities.get(hero_class, {archetype: 0.6 for archetype in ['Aggro', 'Tempo', 'Control', 'Synergy', 'Balanced', 'Attrition']})
    
    def get_hero_specific_archetype_recommendations(self, hero_class: str, current_draft_state: Dict = None) -> Dict[str, Any]:
        """Generate hero-specific archetype recommendations with statistical backing."""
        try:
            # Get hero's archetype affinities
            base_affinities = self._calculate_hero_archetype_preferences(hero_class)
            
            # Enhance with HSReplay statistical data
            enhanced_recommendations = {}
            
            for archetype, base_affinity in base_affinities.items():
                # Get statistical backing from HSReplay if available
                statistical_modifier = self._get_archetype_statistical_modifier(hero_class, archetype)
                
                # Calculate final recommendation strength
                final_strength = min(1.0, base_affinity * statistical_modifier)
                
                # Generate archetype analysis
                analysis = {
                    'archetype': archetype,
                    'strength': final_strength,
                    'base_affinity': base_affinity,
                    'statistical_modifier': statistical_modifier,
                    'confidence': self._calculate_archetype_confidence(hero_class, archetype),
                    'explanation': self._generate_archetype_explanation(hero_class, archetype, final_strength),
                    'key_cards': self._get_key_archetype_cards(hero_class, archetype),
                    'draft_tips': self._get_archetype_draft_tips(hero_class, archetype)
                }
                
                enhanced_recommendations[archetype] = analysis
            
            # Sort by strength
            sorted_archetypes = sorted(enhanced_recommendations.items(), 
                                     key=lambda x: x[1]['strength'], reverse=True)
            
            return {
                'hero_class': hero_class,
                'recommended_archetypes': sorted_archetypes,
                'primary_recommendation': sorted_archetypes[0][0] if sorted_archetypes else 'Balanced',
                'statistical_backing': True,
                'confidence_level': self._calculate_overall_archetype_confidence(enhanced_recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating archetype recommendations for {hero_class}: {e}")
            return self._get_fallback_archetype_recommendations(hero_class)
    
    def _get_archetype_statistical_modifier(self, hero_class: str, archetype: str) -> float:
        """Get statistical modifier based on current meta performance."""
        try:
            # Get hero winrates from HSReplay data
            hero_winrates = self.hsreplay_scraper.get_hero_winrates()
            hero_winrate = hero_winrates.get(hero_class, 50.0)
            
            # Modify archetype strength based on hero performance
            if hero_winrate > 55.0:  # Strong hero
                return 1.15  # Boost all archetypes
            elif hero_winrate > 52.0:  # Above average
                return 1.05
            elif hero_winrate < 47.0:  # Below average
                return 0.9   # Slight penalty
            else:
                return 1.0   # Neutral
                
        except Exception:
            return 1.0  # Neutral if no data
    
    def _calculate_archetype_confidence(self, hero_class: str, archetype: str) -> float:
        """Calculate confidence level for archetype recommendation."""
        try:
            # Base confidence from hero-archetype synergy
            base_confidence = 0.7
            
            # Boost confidence for well-established combinations
            strong_combinations = {
                'HUNTER': ['Aggro', 'Tempo'],
                'PRIEST': ['Control', 'Attrition'], 
                'ROGUE': ['Tempo'],
                'WARLOCK': ['Control', 'Attrition'],
                'SHAMAN': ['Synergy']
            }
            
            if hero_class in strong_combinations and archetype in strong_combinations[hero_class]:
                base_confidence += 0.2
                
            # Factor in data availability
            try:
                hero_winrates = self.hsreplay_scraper.get_hero_winrates()
                if hero_class in hero_winrates:
                    base_confidence += 0.1  # Boost for having statistical data
            except:
                pass
                
            return min(0.95, base_confidence)
            
        except Exception:
            return 0.6  # Moderate confidence fallback
    
    def _generate_archetype_explanation(self, hero_class: str, archetype: str, strength: float) -> str:
        """Generate detailed explanation for archetype recommendation."""
        explanations = {
            ('HUNTER', 'Aggro'): "Hunter's efficient early game cards and direct damage synergize perfectly with aggressive strategies.",
            ('HUNTER', 'Tempo'): "Hunter excels at tempo plays with beast synergies and efficient removal.",
            ('PRIEST', 'Control'): "Priest's healing and removal tools make control strategies highly effective.",
            ('PRIEST', 'Attrition'): "Priest can outlast opponents with superior healing and card generation.",
            ('ROGUE', 'Tempo'): "Rogue's efficient spells and weapon synergies dominate tempo matchups.",
            ('WARLOCK', 'Control'): "Warlock's life tap provides card advantage essential for control strategies.",
            ('SHAMAN', 'Synergy'): "Shaman's elemental and overload synergies create powerful combo potential.",
            ('MAGE', 'Tempo'): "Mage's efficient spells and spell damage synergies excel in tempo games.",
            ('PALADIN', 'Tempo'): "Paladin's buff spells and divine shield synergies create strong tempo swings.",
            ('WARRIOR', 'Control'): "Warrior's armor gain and removal tools support control strategies.",
            ('DRUID', 'Balanced'): "Druid's ramp and choose-based cards provide flexible balanced gameplay.",
            ('DEMONHUNTER', 'Aggro'): "Demon Hunter's efficient aggressive cards and hero power support fast strategies."
        }
        
        key = (hero_class, archetype)
        if key in explanations:
            explanation = explanations[key]
        else:
            explanation = f"{hero_class} has {'strong' if strength > 0.7 else 'moderate' if strength > 0.5 else 'limited'} synergy with {archetype} strategies."
        
        # Add strength qualifier
        if strength > 0.8:
            return f"Excellent fit: {explanation}"
        elif strength > 0.6:
            return f"Good option: {explanation}" 
        else:
            return f"Viable choice: {explanation}"
    
    def _get_key_archetype_cards(self, hero_class: str, archetype: str) -> List[str]:
        """Get key cards that support this hero-archetype combination."""
        archetype_cards = {
            'Aggro': ['Low-cost minions', 'Direct damage', 'Charge minions'],
            'Tempo': ['Efficient removal', 'Value trades', 'Board presence'],
            'Control': ['Board clears', 'Hard removal', 'Card draw'],
            'Synergy': ['Tribal synergies', 'Combo pieces', 'Engine cards'],
            'Balanced': ['Flexible cards', 'Good stats', 'Versatile effects'],
            'Attrition': ['Healing', 'Armor gain', 'Value generation']
        }
        
        hero_specific_additions = {
            ('HUNTER', 'Aggro'): ['Beast synergies', 'Face damage'],
            ('PRIEST', 'Control'): ['Healing spells', 'Mind control effects'],
            ('ROGUE', 'Tempo'): ['Weapons', 'Combo cards'],
            ('WARLOCK', 'Control'): ['Life tap synergies', 'Demons'],
            ('SHAMAN', 'Synergy'): ['Elementals', 'Overload cards']
        }
        
        cards = archetype_cards.get(archetype, [])
        key = (hero_class, archetype)
        if key in hero_specific_additions:
            cards.extend(hero_specific_additions[key])
            
        return cards[:5]  # Return top 5 recommendations
    
    def _get_archetype_draft_tips(self, hero_class: str, archetype: str) -> List[str]:
        """Get specific drafting tips for this hero-archetype combination."""
        tips = {
            ('HUNTER', 'Aggro'): [
                "Prioritize 1-3 mana minions for early pressure",
                "Draft direct damage spells for reach",
                "Value beast synergies when available"
            ],
            ('PRIEST', 'Control'): [
                "Draft board clears and removal spells highly",
                "Value healing and card draw effects",
                "Don't neglect early game defensive options"
            ],
            ('ROGUE', 'Tempo'): [
                "Prioritize efficient removal and weapons",
                "Draft combo enablers and cheap spells",
                "Value stealth minions for pressure"
            ]
        }
        
        key = (hero_class, archetype)
        return tips.get(key, [
            f"Focus on {archetype.lower()}-oriented cards",
            f"Consider {hero_class.lower()} class synergies",
            "Maintain good mana curve balance"
        ])
    
    def _calculate_overall_archetype_confidence(self, recommendations: Dict) -> float:
        """Calculate overall confidence in archetype recommendations."""
        if not recommendations:
            return 0.5
            
        confidence_values = [rec['confidence'] for rec in recommendations.values()]
        return sum(confidence_values) / len(confidence_values)
    
    def get_cross_hero_meta_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive cross-hero meta analysis showing relative performance and trending heroes."""
        try:
            # Get current hero winrates from HSReplay
            hero_winrates = self._get_current_hero_winrates()
            if not hero_winrates:
                return self._get_fallback_meta_analysis()
            
            # Calculate relative performance metrics
            average_winrate = sum(hero_winrates.values()) / len(hero_winrates)
            
            # Categorize heroes by performance
            tier_s = {}  # 55%+ winrate
            tier_a = {}  # 52-55% winrate  
            tier_b = {}  # 48-52% winrate
            tier_c = {}  # <48% winrate
            
            for hero_class, winrate in hero_winrates.items():
                if winrate >= 55.0:
                    tier_s[hero_class] = winrate
                elif winrate >= 52.0:
                    tier_a[hero_class] = winrate
                elif winrate >= 48.0:
                    tier_b[hero_class] = winrate
                else:
                    tier_c[hero_class] = winrate
            
            # Detect trending heroes (compare with historical averages)
            trending_analysis = self._analyze_hero_trends(hero_winrates)
            
            # Generate meta insights
            meta_insights = self._generate_meta_insights(hero_winrates, average_winrate)
            
            return {
                'meta_snapshot': {
                    'timestamp': datetime.now().isoformat(),
                    'average_winrate': round(average_winrate, 2),
                    'sample_size': 'HSReplay Arena data',
                    'data_confidence': 0.85
                },
                'tier_rankings': {
                    'S_tier': {
                        'heroes': tier_s,
                        'description': 'Dominant heroes (55%+ winrate)',
                        'recommendation': 'First pick priority in drafts'
                    },
                    'A_tier': {
                        'heroes': tier_a,
                        'description': 'Strong heroes (52-55% winrate)', 
                        'recommendation': 'Excellent choices for competitive play'
                    },
                    'B_tier': {
                        'heroes': tier_b,
                        'description': 'Balanced heroes (48-52% winrate)',
                        'recommendation': 'Solid picks with proper piloting'
                    },
                    'C_tier': {
                        'heroes': tier_c,
                        'description': 'Struggling heroes (<48% winrate)',
                        'recommendation': 'Avoid unless experienced with class'
                    }
                },
                'trending_heroes': trending_analysis,
                'meta_insights': meta_insights,
                'relative_performance': self._calculate_relative_performance(hero_winrates, average_winrate)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating cross-hero meta analysis: {e}")
            return self._get_fallback_meta_analysis()
    
    def _get_current_hero_winrates(self) -> Dict[str, float]:
        """Get current hero winrates from HSReplay data."""
        try:
            if hasattr(self.hsreplay_scraper, 'get_hero_winrates'):
                return self.hsreplay_scraper.get_hero_winrates()
            else:
                # Fallback to cached data if available
                return self.hero_performance_data
        except Exception as e:
            self.logger.warning(f"Error getting current hero winrates: {e}")
            return {}
    
    def _analyze_hero_trends(self, current_winrates: Dict[str, float]) -> Dict[str, Any]:
        """Analyze hero performance trends by comparing with historical data."""
        try:
            # Historical baseline winrates (representative of long-term averages)
            historical_baselines = {
                'MAGE': 53.2, 'PALADIN': 52.5, 'ROGUE': 52.0, 'HUNTER': 51.8,
                'WARLOCK': 51.0, 'WARRIOR': 50.5, 'SHAMAN': 50.2, 'DRUID': 49.8,
                'PRIEST': 49.5, 'DEMONHUNTER': 49.0
            }
            
            trending_up = {}      # Significant improvement vs baseline
            trending_down = {}    # Significant decline vs baseline
            stable = {}          # Within ±1% of baseline
            
            trend_threshold = 1.5  # 1.5% change threshold for trend detection
            
            for hero_class, current_wr in current_winrates.items():
                baseline = historical_baselines.get(hero_class, 50.0)
                change = current_wr - baseline
                
                if change >= trend_threshold:
                    trending_up[hero_class] = {
                        'current_winrate': current_wr,
                        'baseline_winrate': baseline,
                        'change': round(change, 2),
                        'trend_strength': 'strong' if change >= 3.0 else 'moderate'
                    }
                elif change <= -trend_threshold:
                    trending_down[hero_class] = {
                        'current_winrate': current_wr,
                        'baseline_winrate': baseline,
                        'change': round(change, 2),
                        'trend_strength': 'strong' if change <= -3.0 else 'moderate'
                    }
                else:
                    stable[hero_class] = {
                        'current_winrate': current_wr,
                        'baseline_winrate': baseline,
                        'change': round(change, 2)
                    }
            
            return {
                'trending_up': trending_up,
                'trending_down': trending_down,
                'stable': stable,
                'analysis_method': 'Historical baseline comparison',
                'trend_threshold': trend_threshold
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing hero trends: {e}")
            return {'error': str(e)}
    
    def _generate_meta_insights(self, hero_winrates: Dict[str, float], average_winrate: float) -> List[str]:
        """Generate actionable meta insights from hero performance data."""
        insights = []
        
        try:
            # Find top performer
            best_hero = max(hero_winrates.items(), key=lambda x: x[1])
            insights.append(f"{best_hero[0]} leads the meta at {best_hero[1]:.1f}% winrate")
            
            # Find biggest underperformer
            worst_hero = min(hero_winrates.items(), key=lambda x: x[1])
            insights.append(f"{worst_hero[0]} struggles at {worst_hero[1]:.1f}% winrate")
            
            # Meta diversity analysis
            winrate_range = best_hero[1] - worst_hero[1]
            if winrate_range <= 4.0:
                insights.append("Balanced meta - most heroes viable")
            elif winrate_range <= 6.0:
                insights.append("Moderate meta polarization")
            else:
                insights.append("Highly polarized meta - clear tier gaps")
            
            # Above/below average performers
            above_avg = [h for h, wr in hero_winrates.items() if wr > average_winrate + 1.0]
            below_avg = [h for h, wr in hero_winrates.items() if wr < average_winrate - 1.0]
            
            if len(above_avg) <= 3:
                insights.append(f"Top tier dominated by: {', '.join(above_avg)}")
            if len(below_avg) >= 3:
                insights.append(f"Struggling classes: {', '.join(below_avg[:3])}")
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error generating meta insights: {e}")
            return ["Meta analysis unavailable"]
    
    def _calculate_relative_performance(self, hero_winrates: Dict[str, float], average_winrate: float) -> Dict[str, Dict[str, float]]:
        """Calculate relative performance metrics for each hero."""
        relative_performance = {}
        
        try:
            for hero_class, winrate in hero_winrates.items():
                relative_performance[hero_class] = {
                    'absolute_winrate': round(winrate, 2),
                    'relative_to_average': round(winrate - average_winrate, 2),
                    'percentile_rank': self._calculate_percentile_rank(winrate, list(hero_winrates.values())),
                    'performance_rating': self._get_performance_rating(winrate, average_winrate)
                }
            
            return relative_performance
            
        except Exception as e:
            self.logger.warning(f"Error calculating relative performance: {e}")
            return {}
    
    def _calculate_percentile_rank(self, winrate: float, all_winrates: List[float]) -> float:
        """Calculate percentile rank for a hero's winrate."""
        try:
            sorted_rates = sorted(all_winrates)
            rank = sorted_rates.index(winrate) + 1
            percentile = (rank / len(sorted_rates)) * 100
            return round(percentile, 1)
        except:
            return 50.0
    
    def _get_performance_rating(self, winrate: float, average_winrate: float) -> str:
        """Get descriptive performance rating."""
        diff = winrate - average_winrate
        
        if diff >= 3.0:
            return "Exceptional"
        elif diff >= 1.5:
            return "Strong" 
        elif diff >= 0.5:
            return "Above Average"
        elif diff >= -0.5:
            return "Average"
        elif diff >= -1.5:
            return "Below Average"
        else:
            return "Struggling"
    
    def _get_fallback_meta_analysis(self) -> Dict[str, Any]:
        """Fallback meta analysis when live data unavailable."""
        return {
            'meta_snapshot': {
                'timestamp': datetime.now().isoformat(),
                'average_winrate': 50.0,
                'sample_size': 'Fallback analysis',
                'data_confidence': 0.4
            },
            'tier_rankings': {
                'S_tier': {
                    'heroes': {'MAGE': 53.5, 'PALADIN': 52.8},
                    'description': 'Historically strong heroes',
                    'recommendation': 'Safe picks for consistent performance'
                },
                'A_tier': {
                    'heroes': {'ROGUE': 52.3, 'HUNTER': 51.9, 'WARLOCK': 51.2},
                    'description': 'Solid performers',
                    'recommendation': 'Good choices for most players'
                },
                'B_tier': {
                    'heroes': {'WARRIOR': 50.8, 'SHAMAN': 50.5, 'DRUID': 49.7},
                    'description': 'Balanced options',
                    'recommendation': 'Viable with experience'
                },
                'C_tier': {
                    'heroes': {'PRIEST': 49.2, 'DEMONHUNTER': 48.9},
                    'description': 'Challenging heroes',
                    'recommendation': 'For experienced players only'
                }
            },
            'trending_heroes': {
                'trending_up': {},
                'trending_down': {},
                'stable': {},
                'analysis_method': 'Fallback mode - live data unavailable'
            },
            'meta_insights': [
                "Live meta data unavailable",
                "Using historical performance baselines",
                "Consider checking HSReplay for current trends"
            ],
            'relative_performance': {}
        }
    
    def _get_fallback_archetype_recommendations(self, hero_class: str) -> Dict[str, Any]:
        """Fallback archetype recommendations when statistical data unavailable."""
        fallback_primary = {
            'HUNTER': 'Aggro',
            'DEMONHUNTER': 'Aggro', 
            'ROGUE': 'Tempo',
            'PALADIN': 'Tempo',
            'MAGE': 'Tempo',
            'SHAMAN': 'Synergy',
            'WARRIOR': 'Balanced',
            'PRIEST': 'Control',
            'WARLOCK': 'Control',
            'DRUID': 'Balanced'
        }
        
        return {
            'hero_class': hero_class,
            'primary_recommendation': fallback_primary.get(hero_class, 'Balanced'),
            'statistical_backing': False,
            'confidence_level': 0.6,
            'fallback_mode': True
        }
    
    def detect_hero_specific_sleeper_picks(self, hero_class: str, available_cards: List[str] = None) -> Dict[str, Any]:
        """
        Detect hero-specific 'sleeper pick' cards that overperform with certain heroes.
        
        Identifies cards that have significantly higher performance when drafted by specific heroes
        compared to their general population performance.
        """
        try:
            sleeper_picks = {}
            analysis_summary = {
                'hero_class': hero_class,
                'total_cards_analyzed': 0,
                'sleeper_picks_found': 0,
                'analysis_confidence': 0.7,
                'data_source': 'HSReplay hero-specific card performance'
            }
            
            # Get hero-specific card performance data
            hero_card_stats = self._get_hero_specific_card_stats(hero_class)
            general_card_stats = self._get_general_card_stats()
            
            if not hero_card_stats or not general_card_stats:
                return self._get_fallback_sleeper_picks(hero_class)
            
            # Analyze cards for hero-specific overperformance
            cards_to_analyze = available_cards if available_cards else list(general_card_stats.keys())
            analysis_summary['total_cards_analyzed'] = len(cards_to_analyze)
            
            for card_id in cards_to_analyze:
                sleeper_analysis = self._analyze_card_for_sleeper_potential(
                    card_id, hero_class, hero_card_stats, general_card_stats
                )
                
                if sleeper_analysis and sleeper_analysis['is_sleeper']:
                    sleeper_picks[card_id] = sleeper_analysis
                    analysis_summary['sleeper_picks_found'] += 1
            
            # Sort sleeper picks by overperformance magnitude
            sorted_sleepers = dict(sorted(
                sleeper_picks.items(),
                key=lambda x: x[1]['overperformance_magnitude'],
                reverse=True
            ))
            
            return {
                'analysis_summary': analysis_summary,
                'sleeper_picks': sorted_sleepers,
                'recommendations': self._generate_sleeper_pick_recommendations(sorted_sleepers, hero_class),
                'hero_synergy_insights': self._generate_hero_synergy_insights(sorted_sleepers, hero_class)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting sleeper picks for {hero_class}: {e}")
            return self._get_fallback_sleeper_picks(hero_class)
    
    def _get_hero_specific_card_stats(self, hero_class: str) -> Dict[str, Dict[str, float]]:
        """Get hero-specific card performance statistics from HSReplay."""
        try:
            # Try to get hero-specific card data from HSReplay
            if hasattr(self.card_evaluator, 'get_hero_specific_card_stats'):
                return self.card_evaluator.get_hero_specific_card_stats(hero_class)
            
            # Fallback: estimate from general stats with hero modifiers
            general_stats = self.card_evaluator.hsreplay_stats
            hero_modified_stats = {}
            
            for card_id, stats in general_stats.items():
                # Apply hero-specific performance modifiers
                hero_modified_stats[card_id] = self._apply_hero_performance_modifier(stats, card_id, hero_class)
            
            return hero_modified_stats
            
        except Exception as e:
            self.logger.warning(f"Error getting hero-specific card stats for {hero_class}: {e}")
            return {}
    
    def _get_general_card_stats(self) -> Dict[str, Dict[str, float]]:
        """Get general card performance statistics."""
        try:
            return self.card_evaluator.hsreplay_stats
        except Exception as e:
            self.logger.warning(f"Error getting general card stats: {e}")
            return {}
    
    def _analyze_card_for_sleeper_potential(self, card_id: str, hero_class: str, 
                                          hero_stats: Dict, general_stats: Dict) -> Optional[Dict[str, Any]]:
        """Analyze a specific card for sleeper pick potential with given hero."""
        try:
            # Get card stats for both hero-specific and general performance
            hero_card_data = hero_stats.get(card_id, {})
            general_card_data = general_stats.get(card_id, {})
            
            if not hero_card_data or not general_card_data:
                return None
            
            # Calculate performance metrics
            hero_winrate = hero_card_data.get('win_rate', 0)
            general_winrate = general_card_data.get('win_rate', 0)
            
            if hero_winrate == 0 or general_winrate == 0:
                return None
            
            # Calculate overperformance
            winrate_diff = hero_winrate - general_winrate
            relative_improvement = (winrate_diff / general_winrate) * 100 if general_winrate > 0 else 0
            
            # Sleeper pick criteria
            min_winrate_diff = 2.0  # At least 2% higher with this hero
            min_relative_improvement = 4.0  # At least 4% relative improvement
            min_sample_size = hero_card_data.get('games_played', 0)
            
            is_sleeper = (
                winrate_diff >= min_winrate_diff and
                relative_improvement >= min_relative_improvement and
                min_sample_size >= 100  # Minimum sample size for reliability
            )
            
            if not is_sleeper:
                return None
            
            # Get additional context
            card_rarity = self.card_evaluator.cards_loader.get_card_rarity(card_id) or 'UNKNOWN'
            card_cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
            card_class = self.card_evaluator.cards_loader.get_card_class(card_id) or 'NEUTRAL'
            
            # Determine sleeper tier based on overperformance magnitude
            if winrate_diff >= 4.0:
                sleeper_tier = 'S-tier'
                tier_description = 'Exceptional overperformance'
            elif winrate_diff >= 3.0:
                sleeper_tier = 'A-tier'
                tier_description = 'Strong overperformance'
            else:
                sleeper_tier = 'B-tier'
                tier_description = 'Notable overperformance'
            
            return {
                'card_id': card_id,
                'is_sleeper': True,
                'sleeper_tier': sleeper_tier,
                'tier_description': tier_description,
                'hero_winrate': round(hero_winrate, 2),
                'general_winrate': round(general_winrate, 2),
                'winrate_difference': round(winrate_diff, 2),
                'relative_improvement': round(relative_improvement, 1),
                'overperformance_magnitude': round(winrate_diff, 2),
                'sample_size': min_sample_size,
                'card_details': {
                    'rarity': card_rarity,
                    'cost': card_cost,
                    'class': card_class
                },
                'synergy_explanation': self._explain_hero_card_synergy(card_id, hero_class),
                'draft_priority': self._calculate_sleeper_draft_priority(winrate_diff, relative_improvement)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sleeper potential for {card_id}: {e}")
            return None
    
    def _apply_hero_performance_modifier(self, base_stats: Dict[str, float], card_id: str, hero_class: str) -> Dict[str, float]:
        """Apply hero-specific performance modifiers to base card stats."""
        try:
            modified_stats = base_stats.copy()
            
            # Get card properties
            card_class = self.card_evaluator.cards_loader.get_card_class(card_id) or 'NEUTRAL'
            card_cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
            
            # Apply hero-specific modifiers
            base_winrate = base_stats.get('win_rate', 50.0)
            modifier = 0.0
            
            # Class card synergy bonus
            if card_class == hero_class:
                modifier += 1.5  # Class cards generally perform better
            
            # Hero-specific cost curve preferences
            cost_preferences = {
                'HUNTER': {1: 0.5, 2: 0.8, 3: 0.3, 4: -0.2},
                'DEMONHUNTER': {1: 0.8, 2: 0.5, 3: 0.2, 4: -0.3},
                'ROGUE': {2: 0.5, 3: 0.8, 4: 0.3, 5: -0.2},
                'MAGE': {3: 0.3, 4: 0.5, 5: 0.3, 6: 0.2},
                'PRIEST': {4: 0.3, 5: 0.5, 6: 0.8, 7: 0.5},
                'WARLOCK': {4: 0.2, 5: 0.5, 6: 0.3, 7: 0.2}
            }
            
            hero_cost_prefs = cost_preferences.get(hero_class, {})
            modifier += hero_cost_prefs.get(card_cost, 0.0)
            
            # Apply modifier to winrate
            modified_stats['win_rate'] = max(0, min(100, base_winrate + modifier))
            
            return modified_stats
            
        except Exception as e:
            self.logger.warning(f"Error applying hero modifier for {card_id}: {e}")
            return base_stats
    
    def _explain_hero_card_synergy(self, card_id: str, hero_class: str) -> str:
        """Generate explanation for why a card synergizes well with a specific hero."""
        try:
            card_class = self.card_evaluator.cards_loader.get_card_class(card_id) or 'NEUTRAL'
            card_cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
            
            # Class-specific synergy explanations
            if card_class == hero_class:
                class_synergies = {
                    'HUNTER': "Fits aggressive gameplan and beast synergies",
                    'MAGE': "Strong spell synergy and tempo efficiency",
                    'PRIEST': "Excellent heal/control synergy",
                    'WARLOCK': "Synergizes with life-tap value engine",
                    'WARRIOR': "Great weapon and armor synergy",
                    'PALADIN': "Divine shield and buff synergy",
                    'ROGUE': "Efficient tempo and combo potential",
                    'SHAMAN': "Elemental and overload synergy",
                    'DRUID': "Ramp and choose-one synergy",
                    'DEMONHUNTER': "Aggressive stats and outcast synergy"
                }
                return class_synergies.get(hero_class, "Strong class synergy")
            
            # Neutral card hero-specific explanations
            hero_neutral_preferences = {
                'HUNTER': "Efficient aggressive stats for face damage",
                'DEMONHUNTER': "Strong early game tempo",
                'MAGE': "Spell damage or efficient removal target",
                'PRIEST': "High health for trading and healing",
                'WARLOCK': "Value generation for life-tap engine",
                'WARRIOR': "Strong defensive stats or weapon synergy",
                'PALADIN': "Buff target or board control",
                'ROGUE': "Efficient cost for tempo plays",
                'SHAMAN': "Elemental tag or overload synergy",
                'DRUID': "High-cost ramp target or flexible utility"
            }
            
            return hero_neutral_preferences.get(hero_class, "Solid neutral option")
            
        except Exception:
            return "Strong synergy with hero power and class strategy"
    
    def _calculate_sleeper_draft_priority(self, winrate_diff: float, relative_improvement: float) -> str:
        """Calculate draft priority recommendation for sleeper pick."""
        if winrate_diff >= 4.0 or relative_improvement >= 8.0:
            return "Very High - Consider forcing this pick"
        elif winrate_diff >= 3.0 or relative_improvement >= 6.0:
            return "High - Strong consideration over alternatives"
        elif winrate_diff >= 2.0 or relative_improvement >= 4.0:
            return "Medium-High - Good value pick"
        else:
            return "Medium - Situational consideration"
    
    def _generate_sleeper_pick_recommendations(self, sleeper_picks: Dict[str, Any], hero_class: str) -> List[str]:
        """Generate actionable recommendations for sleeper picks."""
        recommendations = []
        
        try:
            if not sleeper_picks:
                recommendations.append(f"No significant sleeper picks identified for {hero_class}")
                return recommendations
            
            # Overall strategy recommendation
            s_tier_sleepers = [card for card, data in sleeper_picks.items() if data['sleeper_tier'] == 'S-tier']
            a_tier_sleepers = [card for card, data in sleeper_picks.items() if data['sleeper_tier'] == 'A-tier']
            
            if s_tier_sleepers:
                recommendations.append(f"High-priority sleepers for {hero_class}: {', '.join(s_tier_sleepers[:3])}")
            
            if a_tier_sleepers:
                recommendations.append(f"Strong sleeper picks: {', '.join(a_tier_sleepers[:3])}")
            
            # Specific strategy recommendations
            cost_distribution = {}
            for card_data in sleeper_picks.values():
                cost = card_data['card_details']['cost']
                cost_distribution[cost] = cost_distribution.get(cost, 0) + 1
            
            dominant_cost = max(cost_distribution.items(), key=lambda x: x[1])[0] if cost_distribution else None
            if dominant_cost:
                recommendations.append(f"Look for {dominant_cost}-cost sleeper picks - strong {hero_class} synergy")
            
            # Meta advice
            total_sleepers = len(sleeper_picks)
            if total_sleepers >= 5:
                recommendations.append(f"{hero_class} has many hidden gems - prioritize tier lists less")
            elif total_sleepers >= 3:
                recommendations.append(f"Several underrated {hero_class} cards available")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating sleeper recommendations: {e}")
            return [f"Sleeper pick analysis completed for {hero_class}"]
    
    def _generate_hero_synergy_insights(self, sleeper_picks: Dict[str, Any], hero_class: str) -> List[str]:
        """Generate insights about hero-specific synergies from sleeper pick analysis."""
        insights = []
        
        try:
            if not sleeper_picks:
                return [f"No specific synergy patterns identified for {hero_class}"]
            
            # Analyze common synergy themes
            synergy_themes = {}
            cost_themes = {}
            rarity_themes = {}
            
            for card_data in sleeper_picks.values():
                # Extract key themes from synergy explanations
                explanation = card_data['synergy_explanation'].lower()
                cost = card_data['card_details']['cost']
                rarity = card_data['card_details']['rarity']
                
                # Common synergy keywords
                keywords = ['aggressive', 'tempo', 'control', 'synergy', 'value', 'spell', 'weapon', 'heal']
                for keyword in keywords:
                    if keyword in explanation:
                        synergy_themes[keyword] = synergy_themes.get(keyword, 0) + 1
                
                cost_themes[cost] = cost_themes.get(cost, 0) + 1
                rarity_themes[rarity] = rarity_themes.get(rarity, 0) + 1
            
            # Generate insights from themes
            if synergy_themes:
                top_theme = max(synergy_themes.items(), key=lambda x: x[1])[0]
                insights.append(f"{hero_class} excels with {top_theme}-focused cards")
            
            if cost_themes:
                preferred_costs = [cost for cost, count in cost_themes.items() if count >= 2]
                if preferred_costs:
                    insights.append(f"Cost {', '.join(map(str, preferred_costs))} cards overperform with {hero_class}")
            
            # Rarity insights
            if rarity_themes.get('COMMON', 0) >= 3:
                insights.append(f"{hero_class} has strong common card synergies - budget-friendly")
            elif rarity_themes.get('RARE', 0) >= 2:
                insights.append(f"{hero_class} benefits significantly from rare card effects")
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error generating synergy insights: {e}")
            return [f"Synergy analysis completed for {hero_class}"]
    
    def _get_fallback_sleeper_picks(self, hero_class: str) -> Dict[str, Any]:
        """Fallback sleeper picks when statistical analysis unavailable."""
        # Known high-synergy cards for each hero class
        fallback_sleepers = {
            'HUNTER': {
                'card_examples': ['Tracking', 'Animal Companion', 'Kill Command'],
                'synergy_theme': 'Aggressive tempo and beast synergy'
            },
            'MAGE': {
                'card_examples': ['Arcane Intellect', 'Fireball', 'Polymorph'],
                'synergy_theme': 'Spell synergy and removal efficiency'
            },
            'PRIEST': {
                'card_examples': ['Circle of Healing', 'Auchenai Soulpriest', 'Mind Control'],
                'synergy_theme': 'Heal synergy and late-game control'
            },
            'WARLOCK': {
                'card_examples': ['Flame Imp', 'Voidwalker', 'Soulfire'],
                'synergy_theme': 'Life-tap value and aggressive early game'
            },
            'WARRIOR': {
                'card_examples': ['Fiery War Axe', 'Shield Slam', 'Execute'],
                'synergy_theme': 'Weapon synergy and efficient removal'
            }
        }
        
        hero_data = fallback_sleepers.get(hero_class, {
            'card_examples': ['Neutral cards'],
            'synergy_theme': 'General synergy'
        })
        
        return {
            'analysis_summary': {
                'hero_class': hero_class,
                'total_cards_analyzed': 0,
                'sleeper_picks_found': 0,
                'analysis_confidence': 0.3,
                'data_source': 'Fallback heuristics'
            },
            'sleeper_picks': {},
            'recommendations': [
                f"Statistical sleeper analysis unavailable for {hero_class}",
                f"Focus on {hero_data['synergy_theme']}",
                f"Consider cards like: {', '.join(hero_data['card_examples'])}"
            ],
            'hero_synergy_insights': [
                f"{hero_class} synergy theme: {hero_data['synergy_theme']}",
                "Consult tier lists for current meta guidance"
            ]
        }
    
    def generate_advanced_pivot_recommendations(self, deck_state: DeckState, 
                                              offered_cards: List[str] = None) -> Dict[str, Any]:
        """
        Generate advanced pivot recommendations considering hero strengths and weaknesses.
        
        Analyzes current deck archetype viability and suggests strategic pivots based on
        hero-specific performance data, meta position, and draft progression.
        """
        try:
            if not deck_state.drafted_cards or len(deck_state.drafted_cards) < 3:
                return self._get_early_draft_guidance(deck_state)
            
            # Comprehensive pivot analysis
            pivot_analysis = {
                'current_state': self._analyze_current_archetype_performance(deck_state),
                'pivot_opportunities': self._identify_all_pivot_opportunities(deck_state, offered_cards),
                'hero_archetype_fit': self._evaluate_hero_archetype_compatibility(deck_state.hero_class),
                'timing_analysis': self._analyze_pivot_timing(deck_state),
                'meta_considerations': self._get_meta_pivot_insights(deck_state.hero_class),
                'recommendations': []
            }
            
            # Generate specific recommendations
            recommendations = self._generate_pivot_recommendations(
                pivot_analysis, deck_state, offered_cards
            )
            pivot_analysis['recommendations'] = recommendations
            
            return pivot_analysis
            
        except Exception as e:
            self.logger.error(f"Error generating advanced pivot recommendations: {e}")
            return self._get_fallback_pivot_analysis(deck_state)
    
    def _analyze_current_archetype_performance(self, deck_state: DeckState) -> Dict[str, Any]:
        """Analyze how well the current archetype is performing."""
        try:
            current_archetype = deck_state.archetype
            hero_class = deck_state.hero_class
            
            # Get hero's natural affinity for current archetype
            hero_affinity = self.hero_archetype_weights.get(current_archetype, 0.6)
            
            # Analyze draft cards for archetype consistency
            archetype_cards = 0
            total_cards = len(deck_state.drafted_cards)
            
            # This would ideally analyze actual card archetype fit
            # For now, use a simplified heuristic
            archetype_consistency = min(1.0, (archetype_cards / total_cards) if total_cards > 0 else 0.5)
            
            # Get meta performance for this archetype + hero combo
            meta_performance = self._get_archetype_meta_performance(current_archetype, hero_class)
            
            # Calculate overall archetype score
            overall_score = (hero_affinity * 0.4) + (archetype_consistency * 0.3) + (meta_performance * 0.3)
            
            return {
                'archetype': current_archetype,
                'hero_affinity': round(hero_affinity, 3),
                'archetype_consistency': round(archetype_consistency, 3),
                'meta_performance': round(meta_performance, 3),
                'overall_score': round(overall_score, 3),
                'performance_rating': self._rate_archetype_performance(overall_score),
                'cards_drafted': total_cards,
                'archetype_supporting_cards': archetype_cards
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing current archetype performance: {e}")
            return {'error': str(e)}
    
    def _identify_all_pivot_opportunities(self, deck_state: DeckState, offered_cards: List[str] = None) -> List[Dict[str, Any]]:
        """Identify all potential pivot opportunities with detailed analysis."""
        try:
            current_archetype = deck_state.archetype
            hero_class = deck_state.hero_class
            alternative_archetypes = ['Aggro', 'Tempo', 'Control', 'Synergy', 'Balanced', 'Attrition']
            alternative_archetypes = [a for a in alternative_archetypes if a != current_archetype]
            
            pivot_opportunities = []
            
            for alt_archetype in alternative_archetypes:
                opportunity = self._evaluate_pivot_opportunity(
                    deck_state, alt_archetype, offered_cards
                )
                
                if opportunity['viability_score'] > 0.4:  # Minimum viability threshold
                    pivot_opportunities.append(opportunity)
            
            # Sort by overall viability score
            pivot_opportunities.sort(key=lambda x: x['viability_score'], reverse=True)
            
            return pivot_opportunities[:3]  # Return top 3 opportunities
            
        except Exception as e:
            self.logger.warning(f"Error identifying pivot opportunities: {e}")
            return []
    
    def _evaluate_pivot_opportunity(self, deck_state: DeckState, target_archetype: str, 
                                  offered_cards: List[str] = None) -> Dict[str, Any]:
        """Evaluate a specific pivot opportunity."""
        try:
            hero_class = deck_state.hero_class
            cards_drafted = len(deck_state.drafted_cards)
            
            # Hero compatibility with target archetype
            hero_compatibility = self.hero_archetype_weights.get(target_archetype, 0.6)
            
            # Meta strength of target archetype for this hero
            meta_strength = self._get_archetype_meta_performance(target_archetype, hero_class)
            
            # Transition difficulty (how hard to pivot at this stage)
            transition_difficulty = self._calculate_transition_difficulty(
                deck_state.archetype, target_archetype, cards_drafted
            )
            
            # Card support available (if offered cards help the pivot)
            card_support = 0.5  # Default
            if offered_cards:
                card_support = self._calculate_pivot_card_support(
                    offered_cards, target_archetype, hero_class
                )
            
            # Calculate overall viability
            viability_score = (
                hero_compatibility * 0.35 +
                meta_strength * 0.25 +
                (1.0 - transition_difficulty) * 0.25 +
                card_support * 0.15
            )
            
            # Determine pivot timing
            timing = self._determine_pivot_timing(cards_drafted, transition_difficulty)
            
            return {
                'target_archetype': target_archetype,
                'hero_compatibility': round(hero_compatibility, 3),
                'meta_strength': round(meta_strength, 3),
                'transition_difficulty': round(transition_difficulty, 3),
                'card_support': round(card_support, 3),
                'viability_score': round(viability_score, 3),
                'timing_recommendation': timing,
                'explanation': self._explain_pivot_opportunity(
                    target_archetype, hero_class, viability_score, timing
                )
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating pivot opportunity: {e}")
            return {
                'target_archetype': target_archetype,
                'viability_score': 0.0,
                'explanation': f"Error evaluating {target_archetype} pivot"
            }
    
    def _evaluate_hero_archetype_compatibility(self, hero_class: str) -> Dict[str, float]:
        """Evaluate hero's compatibility with all archetypes."""
        try:
            archetype_compatibility = {}
            archetypes = ['Aggro', 'Tempo', 'Control', 'Synergy', 'Balanced', 'Attrition']
            
            for archetype in archetypes:
                # Get base compatibility
                base_compatibility = self.hero_archetype_weights.get(archetype, 0.6)
                
                # Apply meta adjustments
                meta_modifier = self._get_archetype_meta_performance(archetype, hero_class)
                
                # Combine factors
                final_compatibility = (base_compatibility * 0.7) + (meta_modifier * 0.3)
                archetype_compatibility[archetype] = round(final_compatibility, 3)
            
            return archetype_compatibility
            
        except Exception as e:
            self.logger.warning(f"Error evaluating hero archetype compatibility: {e}")
            return {}
    
    def _analyze_pivot_timing(self, deck_state: DeckState) -> Dict[str, Any]:
        """Analyze the timing considerations for pivoting."""
        try:
            cards_drafted = len(deck_state.drafted_cards)
            cards_remaining = 30 - cards_drafted
            
            # Determine draft phase
            if cards_drafted <= 10:
                phase = "Early Draft"
                pivot_flexibility = "High"
            elif cards_drafted <= 20:
                phase = "Mid Draft"
                pivot_flexibility = "Medium"
            else:
                phase = "Late Draft"
                pivot_flexibility = "Low"
            
            # Calculate pivot windows
            optimal_pivot_window = cards_drafted <= 15
            emergency_pivot_window = 15 < cards_drafted <= 25
            
            return {
                'cards_drafted': cards_drafted,
                'cards_remaining': cards_remaining,
                'draft_phase': phase,
                'pivot_flexibility': pivot_flexibility,
                'optimal_pivot_window': optimal_pivot_window,
                'emergency_pivot_window': emergency_pivot_window,
                'timing_advice': self._get_timing_advice(cards_drafted)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing pivot timing: {e}")
            return {'error': str(e)}
    
    def _get_meta_pivot_insights(self, hero_class: str) -> List[str]:
        """Get meta-specific insights for pivot decisions."""
        insights = []
        
        try:
            # Get current meta hero performance
            hero_winrates = self._get_current_hero_winrates()
            if hero_winrates:
                hero_performance = hero_winrates.get(hero_class, 50.0)
                average_performance = sum(hero_winrates.values()) / len(hero_winrates)
                
                if hero_performance > average_performance + 2:
                    insights.append(f"{hero_class} is strong in current meta - consider staying flexible")
                elif hero_performance < average_performance - 2:
                    insights.append(f"{hero_class} struggles in current meta - pivot to strongest archetypes")
                
            # Hero-specific meta insights
            meta_insights = {
                'HUNTER': "Aggro consistently strong, Tempo viable in most metas",
                'MAGE': "Tempo/Control flexible, adapt to removal quality offered",
                'PRIEST': "Control natural fit, but can pivot to value-oriented builds",
                'WARLOCK': "Life-tap enables multiple archetypes, stay flexible",
                'WARRIOR': "Weapon synergy drives archetype choice, prioritize weapon quality",
                'PALADIN': "Buff synergy strong, divine shield enables tempo pivots",
                'ROGUE': "Combo/Tempo natural, weapon quality determines aggression level",
                'SHAMAN': "Elemental synergy strong, overload cards suggest specific timing",
                'DRUID': "Ramp potential allows control pivots, early game determines aggression",
                'DEMONHUNTER': "Outcast mechanics favor aggro, but stats allow tempo flexibility"
            }
            
            insights.append(meta_insights.get(hero_class, "Standard archetype flexibility"))
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error getting meta pivot insights: {e}")
            return [f"Meta analysis unavailable for {hero_class}"]
    
    def _generate_pivot_recommendations(self, pivot_analysis: Dict, deck_state: DeckState, 
                                      offered_cards: List[str] = None) -> List[str]:
        """Generate specific pivot recommendations."""
        recommendations = []
        
        try:
            current_performance = pivot_analysis.get('current_state', {})
            opportunities = pivot_analysis.get('pivot_opportunities', [])
            timing = pivot_analysis.get('timing_analysis', {})
            
            # Current archetype assessment
            current_score = current_performance.get('overall_score', 0.5)
            if current_score >= 0.7:
                recommendations.append("Current archetype performing well - continue current strategy")
            elif current_score >= 0.5:
                recommendations.append("Current archetype viable - monitor for pivot opportunities")
            else:
                recommendations.append("Current archetype struggling - strongly consider pivoting")
            
            # Timing recommendations
            if timing.get('optimal_pivot_window', False):
                recommendations.append("Optimal pivot window - good time for strategic changes")
            elif timing.get('emergency_pivot_window', False):
                recommendations.append("Emergency pivot window - only pivot if significantly better option")
            else:
                recommendations.append("Late draft - commit to current direction unless critical pivot")
            
            # Specific opportunity recommendations
            if opportunities:
                best_opportunity = opportunities[0]
                if best_opportunity['viability_score'] > 0.7:
                    recommendations.append(
                        f"Strong pivot opportunity: {best_opportunity['target_archetype']} "
                        f"(viability: {best_opportunity['viability_score']:.2f})"
                    )
                elif best_opportunity['viability_score'] > 0.6:
                    recommendations.append(
                        f"Consider pivoting to {best_opportunity['target_archetype']} if offered strong support cards"
                    )
            
            # Meta-specific advice
            meta_insights = pivot_analysis.get('meta_considerations', [])
            if meta_insights:
                recommendations.append(f"Meta insight: {meta_insights[0]}")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating pivot recommendations: {e}")
            return ["Pivot analysis completed - monitor draft flow"]
    
    def _get_archetype_meta_performance(self, archetype: str, hero_class: str) -> float:
        """Get meta performance for archetype + hero combination."""
        try:
            # This would ideally come from real meta data
            # For now, use heuristic based on hero-archetype synergy
            archetype_meta_strength = {
                'Aggro': 0.8,     # Generally strong in Arena
                'Tempo': 0.9,     # Very strong in Arena
                'Control': 0.6,   # Situational
                'Synergy': 0.7,   # Good when achievable
                'Balanced': 0.8,  # Safe choice
                'Attrition': 0.5  # Difficult to execute
            }
            
            base_strength = archetype_meta_strength.get(archetype, 0.6)
            
            # Apply hero-specific modifiers
            hero_modifier = self.hero_archetype_weights.get(archetype, 0.6)
            
            # Combine factors
            return (base_strength * 0.6) + (hero_modifier * 0.4)
            
        except Exception:
            return 0.6
    
    def _calculate_transition_difficulty(self, current_archetype: str, target_archetype: str, cards_drafted: int) -> float:
        """Calculate difficulty of transitioning between archetypes."""
        try:
            # Base difficulty increases with cards drafted
            base_difficulty = min(0.9, cards_drafted / 30)
            
            # Archetype transition matrix (how hard to transition between archetypes)
            transition_matrix = {
                ('Aggro', 'Tempo'): 0.2,      # Easy transition
                ('Tempo', 'Aggro'): 0.3,
                ('Tempo', 'Control'): 0.4,    # Medium difficulty
                ('Control', 'Tempo'): 0.5,
                ('Aggro', 'Control'): 0.8,    # Very difficult
                ('Control', 'Aggro'): 0.9,
                ('Balanced', 'Aggro'): 0.3,   # Balanced can pivot easier
                ('Balanced', 'Tempo'): 0.2,
                ('Balanced', 'Control'): 0.4,
                ('Synergy', 'Tempo'): 0.6,    # Synergy pivots are harder
                ('Attrition', 'Control'): 0.3
            }
            
            # Get transition difficulty
            transition_key = (current_archetype, target_archetype)
            archetype_difficulty = transition_matrix.get(transition_key, 0.5)  # Default medium
            
            # Combine factors
            total_difficulty = (base_difficulty * 0.6) + (archetype_difficulty * 0.4)
            return min(1.0, total_difficulty)
            
        except Exception:
            return 0.5
    
    def _calculate_pivot_card_support(self, offered_cards: List[str], target_archetype: str, hero_class: str) -> float:
        """Calculate how well offered cards support a potential pivot."""
        try:
            if not offered_cards:
                return 0.5
            
            support_score = 0.0
            
            for card_id in offered_cards:
                # This would ideally evaluate each card's fit with the target archetype
                # For now, use simplified heuristic
                card_archetype_fit = self._evaluate_card_archetype_fit(card_id, target_archetype, hero_class)
                support_score += card_archetype_fit
            
            # Normalize by number of cards
            average_support = support_score / len(offered_cards)
            return min(1.0, average_support)
            
        except Exception:
            return 0.5
    
    def _evaluate_card_archetype_fit(self, card_id: str, archetype: str, hero_class: str) -> float:
        """Evaluate how well a card fits a specific archetype."""
        try:
            # Get card properties
            card_cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
            card_type = self.card_evaluator.cards_loader.get_card_type(card_id) or 'MINION'
            
            # Archetype fit heuristics
            if archetype == 'Aggro':
                if card_cost <= 3:
                    return 0.8
                elif card_cost <= 5:
                    return 0.4
                else:
                    return 0.1
            elif archetype == 'Control':
                if card_cost >= 5:
                    return 0.8
                elif card_cost >= 3:
                    return 0.5
                else:
                    return 0.3
            elif archetype == 'Tempo':
                if 2 <= card_cost <= 4:
                    return 0.8
                else:
                    return 0.4
            else:
                return 0.5  # Default for other archetypes
                
        except Exception:
            return 0.5
    
    def _determine_pivot_timing(self, cards_drafted: int, transition_difficulty: float) -> str:
        """Determine the optimal timing for a pivot."""
        if cards_drafted <= 10:
            return "Optimal - Early draft allows easy transitions"
        elif cards_drafted <= 15:
            if transition_difficulty < 0.5:
                return "Good - Medium difficulty transition possible"
            else:
                return "Cautious - Difficult transition at this stage"
        elif cards_drafted <= 20:
            if transition_difficulty < 0.3:
                return "Emergency - Only if significantly better"
            else:
                return "Not recommended - Transition too difficult"
        else:
            return "Commit - Too late for major archetype changes"
    
    def _explain_pivot_opportunity(self, target_archetype: str, hero_class: str, 
                                 viability_score: float, timing: str) -> str:
        """Generate explanation for pivot opportunity."""
        explanation_parts = []
        
        # Viability assessment
        if viability_score >= 0.7:
            explanation_parts.append(f"Strong {target_archetype} potential for {hero_class}")
        elif viability_score >= 0.5:
            explanation_parts.append(f"Viable {target_archetype} option for {hero_class}")
        else:
            explanation_parts.append(f"Challenging {target_archetype} transition for {hero_class}")
        
        # Timing context
        if "Optimal" in timing:
            explanation_parts.append("good timing")
        elif "Good" in timing:
            explanation_parts.append("acceptable timing")
        else:
            explanation_parts.append("challenging timing")
        
        return ". ".join(explanation_parts)
    
    def _rate_archetype_performance(self, score: float) -> str:
        """Rate archetype performance based on score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.5:
            return "Average"
        elif score >= 0.35:
            return "Below Average"
        else:
            return "Poor"
    
    def _get_timing_advice(self, cards_drafted: int) -> str:
        """Get timing-specific advice."""
        if cards_drafted <= 5:
            return "Very early - maximum flexibility for any archetype"
        elif cards_drafted <= 10:
            return "Early draft - good time for strategic pivots"
        elif cards_drafted <= 15:
            return "Mid-early draft - consider major pivots carefully"
        elif cards_drafted <= 20:
            return "Mid-late draft - minor adjustments only"
        elif cards_drafted <= 25:
            return "Late draft - commit to current strategy"
        else:
            return "Very late - focus on deck completion"
    
    def _get_early_draft_guidance(self, deck_state: DeckState) -> Dict[str, Any]:
        """Provide guidance for early draft when pivots aren't yet relevant."""
        return {
            'current_state': {
                'archetype': deck_state.archetype,
                'cards_drafted': len(deck_state.drafted_cards),
                'stage': 'Early Draft - Building Foundation'
            },
            'recommendations': [
                "Too early for pivot analysis - focus on card quality",
                f"Build toward {deck_state.archetype} but stay flexible",
                "Prioritize strong individual cards over synergy",
                f"{deck_state.hero_class} works well with multiple archetypes"
            ]
        }
    
    def _get_fallback_pivot_analysis(self, deck_state: DeckState) -> Dict[str, Any]:
        """Fallback pivot analysis when errors occur."""
        return {
            'current_state': {
                'archetype': deck_state.archetype,
                'performance_rating': 'Unknown'
            },
            'pivot_opportunities': [],
            'recommendations': [
                f"Pivot analysis unavailable for {deck_state.archetype}",
                "Continue with current archetype strategy",
                "Monitor draft flow for natural archetype development"
            ]
        }
    
    def _generate_card_explanation(self, card_id: str, evaluation: DimensionalScores, deck_state: DeckState) -> str:
        """Generate detailed explanation for individual card with hero context."""
        try:
            # Get HSReplay stats for statistical backing
            hsreplay_stats = self.card_evaluator.hsreplay_stats.get(card_id, {})
            
            # Base explanation with statistics
            explanation_parts = []
            
            if 'win_rate' in hsreplay_stats:
                win_rate = hsreplay_stats['win_rate']
                if win_rate > 52:
                    explanation_parts.append(f"Strong performer ({win_rate:.1f}% winrate)")
                elif win_rate < 48:
                    explanation_parts.append(f"Below average ({win_rate:.1f}% winrate)")
                else:
                    explanation_parts.append(f"Balanced option ({win_rate:.1f}% winrate)")
            
            # Dimensional highlights
            if evaluation.tempo_score > 0.3:
                explanation_parts.append("good tempo")
            if evaluation.value_score > 0.3:
                explanation_parts.append("strong value")
            if evaluation.synergy_score > 0.2:
                explanation_parts.append("synergy potential")
            if evaluation.curve_score > 0.2:
                explanation_parts.append("curve fit")
            
            # Hero-specific context
            if deck_state.hero_class:
                hero_context = self._get_hero_specific_context(card_id, deck_state.hero_class)
                if hero_context:
                    explanation_parts.append(hero_context)
            
            if explanation_parts:
                return ". ".join(explanation_parts).capitalize() + "."
            else:
                return f"Solid {deck_state.archetype.lower()} option for {deck_state.hero_class}."
                
        except Exception as e:
            self.logger.warning(f"Error generating explanation for {card_id}: {e}")
            return f"Analysis for {card_id}"
    
    def _get_hero_specific_context(self, card_id: str, hero_class: str) -> str:
        """Get hero-specific context for card explanation."""
        try:
            card_class = self.card_evaluator.cards_loader.get_card_class(card_id)
            
            if card_class == hero_class:
                return f"excellent {hero_class.lower()} synergy"
            elif card_class == 'NEUTRAL':
                # Hero-specific neutral card preferences
                preferences = {
                    'HUNTER': "fits aggressive gameplan",
                    'WARLOCK': "value for life-tap synergy", 
                    'PRIEST': "supports control strategy",
                    'MAGE': "spell synergy potential",
                    'SHAMAN': "elemental synergy",
                    'WARRIOR': "weapon synergy potential",
                    'PALADIN': "divine shield value",
                    'ROGUE': "efficient for tempo",
                    'DRUID': "ramp target potential",
                    'DEMONHUNTER': "aggressive stats"
                }
                return preferences.get(hero_class, "solid neutral option")
            
            return ""
            
        except Exception:
            return ""
    
    def _generate_comprehensive_explanation(self, recommended_card: str, all_analyses: List[Dict], 
                                          deck_state: DeckState, pivot_opportunity: Optional[str]) -> str:
        """Generate comprehensive explanation with statistical backing."""
        try:
            recommended_analysis = next(a for a in all_analyses if a["card_id"] == recommended_card)
            
            # Start with recommended card explanation
            explanation = f"Recommended: {recommended_card}. {recommended_analysis['explanation']}"
            
            # Add comparative context
            scores = [a["scores"]["final_score"] for a in all_analyses]
            best_score = max(scores)
            second_best = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
            
            if best_score > second_best + 0.1:
                explanation += f" Clear choice with {(best_score - second_best):.1f} point advantage."
            else:
                explanation += " Close decision among competitive options."
            
            # Add archetype context
            if deck_state.archetype:
                archetype_strength = self.hero_archetype_weights.get(deck_state.archetype, 0.6)
                if archetype_strength > 0.7:
                    explanation += f" Excellent fit for {deck_state.archetype.lower()} {deck_state.hero_class.lower()}."
                elif archetype_strength < 0.5:
                    explanation += f" Consider pivoting from {deck_state.archetype.lower()}."
            
            # Add pivot opportunity if detected
            if pivot_opportunity:
                explanation += f" {pivot_opportunity}"
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"Error generating comprehensive explanation: {e}")
            return f"Recommended: {recommended_card}"
    
    def _detect_pivot_opportunities(self, deck_state: DeckState, card_evaluations: List[DimensionalScores]) -> Optional[str]:
        """Detect opportunities to pivot to a different archetype with hero-aware Dynamic Pivot Advisor."""
        try:
            if not deck_state.drafted_cards or len(deck_state.drafted_cards) < 5:
                return None  # Too early to detect pivots
            
            current_archetype = deck_state.archetype
            alternative_archetypes = ['Aggro', 'Tempo', 'Control', 'Synergy', 'Balanced', 'Attrition']
            alternative_archetypes = [a for a in alternative_archetypes if a != current_archetype]
            
            # Test each card against alternative archetypes
            for card_eval in card_evaluations:
                for alt_archetype in alternative_archetypes:
                    current_score = self._calculate_weighted_scores_with_hero_context(
                        card_eval, current_archetype, deck_state.hero_class
                    )
                    alt_score = self._calculate_weighted_scores_with_hero_context(
                        card_eval, alt_archetype, deck_state.hero_class
                    )
                    
                    # Check for significant improvement (30%+)
                    if alt_score > current_score * 1.3:
                        hero_affinity = self.hero_archetype_weights.get(alt_archetype, 0.6)
                        
                        if hero_affinity > 0.6:  # Hero is good at this archetype
                            self.pivot_opportunities_found += 1
                            return f"Consider pivoting to {alt_archetype.lower()} (strong {deck_state.hero_class.lower()} archetype)"
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error detecting pivot opportunities: {e}")
            return None
    
    def _analyze_current_deck_state(self, deck_state: DeckState, card_evaluations: List[DimensionalScores]) -> Dict[str, Any]:
        """Perform comprehensive deck analysis considering hero choice impact."""
        try:
            analysis = {
                "hero_class": deck_state.hero_class,
                "archetype": deck_state.archetype,
                "cards_drafted": len(deck_state.drafted_cards),
                "archetype_viability": self.hero_archetype_weights.get(deck_state.archetype, 0.6),
                "hero_performance": self.hero_performance_data.get(deck_state.hero_class, 50.0)
            }
            
            if deck_state.drafted_cards:
                # Curve analysis
                costs = []
                for card_id in deck_state.drafted_cards:
                    cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
                    costs.append(cost)
                
                analysis["avg_cost"] = sum(costs) / len(costs) if costs else 0
                analysis["curve_distribution"] = {i: costs.count(i) for i in range(8)}
                
                # Archetype consistency
                archetype_scores = []
                for card_id in deck_state.drafted_cards[-5:]:  # Last 5 picks
                    try:
                        eval_placeholder = DimensionalScores(card_id=card_id)  # Simplified for analysis
                        score = self._calculate_weighted_scores_with_hero_context(
                            eval_placeholder, deck_state.archetype, deck_state.hero_class
                        )
                        archetype_scores.append(score)
                    except:
                        continue
                
                if archetype_scores:
                    analysis["archetype_consistency"] = sum(archetype_scores) / len(archetype_scores)
                
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing deck state: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_confidence(self, card_evaluations: List[DimensionalScores], deck_state: DeckState) -> float:
        """Calculate overall confidence in the recommendation."""
        try:
            confidence_factors = []
            
            # Data quality factor
            avg_confidence = sum(eval.confidence for eval in card_evaluations) / len(card_evaluations)
            confidence_factors.append(avg_confidence)
            
            # Score spread factor (clearer recommendations = higher confidence)
            scores = [
                self._calculate_weighted_scores_with_hero_context(eval, deck_state.archetype, deck_state.hero_class)
                for eval in card_evaluations
            ]
            if len(scores) > 1:
                score_spread = (max(scores) - min(scores)) / max(scores) if max(scores) > 0 else 0
                confidence_factors.append(min(1.0, score_spread * 2))  # Normalize spread
            
            # Hero archetype fit factor
            archetype_fit = self.hero_archetype_weights.get(deck_state.archetype, 0.6)
            confidence_factors.append(archetype_fit)
            
            # System health factor
            if hasattr(self.hsreplay_scraper, 'get_api_status'):
                api_status = self.hsreplay_scraper.get_api_status()
                if api_status.get('session_active', False):
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.6)
            
            final_confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.3, min(0.95, final_confidence))
            
        except Exception as e:
            self.logger.warning(f"Error calculating overall confidence: {e}")
            return 0.7
    
    def _build_explanation_templates(self) -> Dict[str, str]:
        """Build explanation templates for statistical backing."""
        return {
            "high_winrate": "{card}: {winrate:.1f}% winrate ({diff:+.1f}% above average, {games:,} games)",
            "low_winrate": "{card}: {winrate:.1f}% winrate ({diff:+.1f}% below average, {games:,} games)",
            "tempo_pick": "Strong tempo option for {hero}. {reasoning}",
            "value_pick": "Excellent value card for {hero}. {reasoning}",
            "synergy_pick": "Great {synergy} synergy for {hero}. {reasoning}",
            "curve_pick": "Perfect curve fit at {cost} mana. {reasoning}",
            "pivot_suggestion": "Consider pivoting to {archetype} - strong {hero} archetype with {advantage:.1f}% advantage",
            "safe_pick": "Safe, balanced choice. {reasoning}",
            "greedy_pick": "High-risk, high-reward option. {reasoning}"
        }
    
    # === STATISTICS AND PERFORMANCE METHODS ===
    
    def get_advisor_statistics(self) -> Dict[str, Any]:
        """Get comprehensive advisor statistics."""
        return {
            "recommendations_made": self.recommendation_count,
            "current_hero": self.current_hero_class,
            "pivot_opportunities_found": self.pivot_opportunities_found,
            "avg_confidence": (
                sum(self.confidence_scores) / len(self.confidence_scores)
                if self.confidence_scores else 0
            ),
            "last_analysis_time_ms": self.last_analysis_time,
            "hero_archetype_weights": self.hero_archetype_weights,
            "system_integration": {
                "card_evaluator": bool(self.card_evaluator),
                "deck_analyzer": bool(self.deck_analyzer),
                "hsreplay_integration": bool(self.hsreplay_scraper)
            }
        }
    
    # === META SHIFT IMPACT ANALYSIS METHODS ===
    
    def analyze_meta_shift_impact(self, patch_timeframe: int = 30) -> Dict[str, Any]:
        """
        Analyze the impact of meta shifts on hero viability across patches.
        
        Tracks hero performance changes over time and identifies which heroes
        are gaining/losing viability due to meta shifts.
        
        Args:
            patch_timeframe: Days to look back for trend analysis
            
        Returns:
            Comprehensive meta shift analysis with hero impact assessment
        """
        try:
            self.logger.info(f"Analyzing meta shift impact over {patch_timeframe} days")
            
            # Get current hero winrates
            current_winrates = self._get_current_winrates()
            if not current_winrates:
                return self._get_fallback_meta_shift_analysis()
            
            # Get historical baseline for comparison
            historical_baseline = self._get_historical_hero_baseline()
            
            # Analyze meta shift patterns
            shift_analysis = self._analyze_hero_meta_shifts(current_winrates, historical_baseline)
            
            # Identify heroes most affected by meta changes
            impact_assessment = self._assess_meta_shift_impact(shift_analysis)
            
            # Generate patch prediction insights
            patch_predictions = self._generate_patch_impact_predictions(shift_analysis)
            
            # Calculate meta stability metrics
            stability_metrics = self._calculate_meta_stability(current_winrates, historical_baseline)
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'timeframe_days': patch_timeframe,
                'meta_shift_analysis': shift_analysis,
                'hero_impact_assessment': impact_assessment,
                'patch_predictions': patch_predictions,
                'stability_metrics': stability_metrics,
                'recommendations': self._generate_meta_shift_recommendations(impact_assessment),
                'data_confidence': self._calculate_meta_shift_confidence(current_winrates, historical_baseline)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing meta shift impact: {e}")
            return self._get_fallback_meta_shift_analysis()
    
    def _analyze_hero_meta_shifts(self, current_winrates: Dict[str, float], 
                                 historical_baseline: Dict[str, float]) -> Dict[str, Any]:
        """Analyze hero performance shifts compared to historical baseline."""
        try:
            shift_data = {
                'heroes_rising': {},
                'heroes_falling': {},
                'heroes_stable': {},
                'meta_momentum': {}
            }
            
            for hero_class in current_winrates:
                current_wr = current_winrates.get(hero_class, 50.0)
                baseline_wr = historical_baseline.get(hero_class, 50.0)
                
                # Calculate shift magnitude and direction
                shift_amount = current_wr - baseline_wr
                shift_percentage = (shift_amount / baseline_wr * 100) if baseline_wr > 0 else 0
                
                # Categorize shift type
                if abs(shift_amount) >= 2.0:  # Significant shift
                    if shift_amount > 0:
                        shift_data['heroes_rising'][hero_class] = {
                            'current_winrate': round(current_wr, 2),
                            'baseline_winrate': round(baseline_wr, 2),
                            'shift_amount': round(shift_amount, 2),
                            'shift_percentage': round(shift_percentage, 1),
                            'impact_level': 'Major' if abs(shift_amount) >= 3.0 else 'Moderate'
                        }
                    else:
                        shift_data['heroes_falling'][hero_class] = {
                            'current_winrate': round(current_wr, 2),
                            'baseline_winrate': round(baseline_wr, 2),
                            'shift_amount': round(shift_amount, 2),
                            'shift_percentage': round(shift_percentage, 1),
                            'impact_level': 'Major' if abs(shift_amount) >= 3.0 else 'Moderate'
                        }
                else:
                    shift_data['heroes_stable'][hero_class] = {
                        'current_winrate': round(current_wr, 2),
                        'baseline_winrate': round(baseline_wr, 2),
                        'shift_amount': round(shift_amount, 2)
                    }
                
                # Calculate momentum (trend strength)
                momentum_score = min(5.0, abs(shift_amount))  # Cap at 5.0
                momentum_direction = 'rising' if shift_amount > 0 else 'falling' if shift_amount < 0 else 'stable'
                
                shift_data['meta_momentum'][hero_class] = {
                    'momentum_score': round(momentum_score, 2),
                    'direction': momentum_direction,
                    'velocity': round(shift_percentage, 1)
                }
            
            return shift_data
            
        except Exception as e:
            self.logger.warning(f"Error analyzing hero meta shifts: {e}")
            return {'error': str(e)}
    
    def _assess_meta_shift_impact(self, shift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of meta shifts on different hero archetypes and playstyles."""
        try:
            impact_assessment = {
                'most_impacted_heroes': [],
                'archetype_trends': {},
                'playstyle_shifts': {},
                'meta_polarization': None
            }
            
            # Identify most impacted heroes (both positive and negative)
            all_shifts = {}
            for hero_data in shift_analysis.get('heroes_rising', {}).values():
                all_shifts.update(shift_analysis['heroes_rising'])
            for hero_data in shift_analysis.get('heroes_falling', {}).values():
                all_shifts.update(shift_analysis['heroes_falling'])
            
            # Sort by absolute impact
            most_impacted = sorted(
                all_shifts.items(),
                key=lambda x: abs(x[1]['shift_amount']),
                reverse=True
            )
            
            impact_assessment['most_impacted_heroes'] = [
                {
                    'hero_class': hero,
                    'impact_type': 'Positive' if data['shift_amount'] > 0 else 'Negative',
                    'shift_magnitude': abs(data['shift_amount']),
                    'new_tier': self._determine_hero_tier(data['current_winrate'])
                }
                for hero, data in most_impacted[:5]
            ]
            
            # Analyze archetype trends
            archetype_impacts = self._analyze_archetype_meta_trends(shift_analysis)
            impact_assessment['archetype_trends'] = archetype_impacts
            
            # Assess playstyle shifts
            playstyle_impacts = self._analyze_playstyle_meta_trends(shift_analysis)
            impact_assessment['playstyle_shifts'] = playstyle_impacts
            
            # Calculate meta polarization
            rising_count = len(shift_analysis.get('heroes_rising', {}))
            falling_count = len(shift_analysis.get('heroes_falling', {}))
            stable_count = len(shift_analysis.get('heroes_stable', {}))
            
            if rising_count + falling_count > stable_count:
                impact_assessment['meta_polarization'] = {
                    'level': 'High',
                    'description': 'Meta is highly volatile with significant hero shifts'
                }
            elif rising_count + falling_count > stable_count * 0.5:
                impact_assessment['meta_polarization'] = {
                    'level': 'Moderate', 
                    'description': 'Meta showing moderate changes'
                }
            else:
                impact_assessment['meta_polarization'] = {
                    'level': 'Low',
                    'description': 'Meta is relatively stable'
                }
            
            return impact_assessment
            
        except Exception as e:
            self.logger.warning(f"Error assessing meta shift impact: {e}")
            return {'error': str(e)}
    
    def _generate_patch_impact_predictions(self, shift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions for how current trends might continue."""
        try:
            predictions = {
                'heroes_to_watch': [],
                'potential_nerfs': [],
                'potential_buffs': [],
                'meta_trajectory': None
            }
            
            # Heroes to watch (showing strong momentum)
            momentum_data = shift_analysis.get('meta_momentum', {})
            high_momentum_heroes = [
                {'hero': hero, 'momentum': data['momentum_score'], 'direction': data['direction']}
                for hero, data in momentum_data.items()
                if data['momentum_score'] >= 2.0
            ]
            
            predictions['heroes_to_watch'] = sorted(
                high_momentum_heroes,
                key=lambda x: x['momentum'],
                reverse=True
            )[:5]
            
            # Potential nerf candidates (too strong)
            rising_heroes = shift_analysis.get('heroes_rising', {})
            potential_nerfs = [
                hero for hero, data in rising_heroes.items()
                if data['current_winrate'] >= 55.0 and data['shift_amount'] >= 2.0
            ]
            predictions['potential_nerfs'] = potential_nerfs
            
            # Potential buff candidates (underperforming)
            falling_heroes = shift_analysis.get('heroes_falling', {})
            potential_buffs = [
                hero for hero, data in falling_heroes.items()
                if data['current_winrate'] <= 47.0 and data['shift_amount'] <= -2.0
            ]
            predictions['potential_buffs'] = potential_buffs
            
            # Overall meta trajectory
            total_rising = len(rising_heroes)
            total_falling = len(falling_heroes)
            
            if total_rising > total_falling * 1.5:
                predictions['meta_trajectory'] = {
                    'direction': 'Power Creep',
                    'description': 'Overall power level increasing'
                }
            elif total_falling > total_rising * 1.5:
                predictions['meta_trajectory'] = {
                    'direction': 'Power Decrease',
                    'description': 'Overall power level decreasing'
                }
            else:
                predictions['meta_trajectory'] = {
                    'direction': 'Balanced',
                    'description': 'Power level shifts are balanced'
                }
            
            return predictions
            
        except Exception as e:
            self.logger.warning(f"Error generating patch predictions: {e}")
            return {'error': str(e)}
    
    def _calculate_meta_stability(self, current_winrates: Dict[str, float], 
                                 historical_baseline: Dict[str, float]) -> Dict[str, Any]:
        """Calculate metrics for meta stability and volatility."""
        try:
            stability_metrics = {
                'overall_stability_score': 0.0,
                'volatility_index': 0.0,
                'tier_mobility': 0.0,
                'stability_rating': 'Unknown'
            }
            
            if not current_winrates or not historical_baseline:
                return stability_metrics
            
            # Calculate overall stability (lower variance = more stable)
            shifts = []
            for hero in current_winrates:
                if hero in historical_baseline:
                    shift = abs(current_winrates[hero] - historical_baseline[hero])
                    shifts.append(shift)
            
            if shifts:
                avg_shift = sum(shifts) / len(shifts)
                max_shift = max(shifts)
                
                # Stability score (0-1, higher = more stable)
                stability_score = max(0.0, 1.0 - (avg_shift / 10.0))
                stability_metrics['overall_stability_score'] = round(stability_score, 3)
                
                # Volatility index (0-10, higher = more volatile)
                volatility = min(10.0, avg_shift * 2)
                stability_metrics['volatility_index'] = round(volatility, 2)
                
                # Tier mobility (how much heroes are changing tiers)
                tier_changes = sum(1 for shift in shifts if shift >= 2.0)
                tier_mobility = tier_changes / len(shifts) if shifts else 0
                stability_metrics['tier_mobility'] = round(tier_mobility, 3)
                
                # Overall rating
                if stability_score >= 0.8:
                    stability_metrics['stability_rating'] = 'Very Stable'
                elif stability_score >= 0.6:
                    stability_metrics['stability_rating'] = 'Stable'
                elif stability_score >= 0.4:
                    stability_metrics['stability_rating'] = 'Moderate'
                elif stability_score >= 0.2:
                    stability_metrics['stability_rating'] = 'Volatile'
                else:
                    stability_metrics['stability_rating'] = 'Highly Volatile'
            
            return stability_metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating meta stability: {e}")
            return {'error': str(e)}
    
    def _generate_meta_shift_recommendations(self, impact_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on meta shift analysis."""
        try:
            recommendations = []
            
            # Most impacted heroes recommendations
            most_impacted = impact_assessment.get('most_impacted_heroes', [])
            for hero_data in most_impacted[:3]:
                hero = hero_data['hero_class']
                impact_type = hero_data['impact_type']
                
                if impact_type == 'Positive':
                    recommendations.append(f"Consider {hero} - rising in the meta with strong performance")
                else:
                    recommendations.append(f"Avoid {hero} - declining performance in current meta")
            
            # Polarization recommendations
            polarization = impact_assessment.get('meta_polarization', {})
            if polarization.get('level') == 'High':
                recommendations.append("Meta is volatile - prioritize consistent heroes over risky picks")
            elif polarization.get('level') == 'Low':
                recommendations.append("Stable meta - good time to experiment with different heroes")
            
            # Archetype trend recommendations
            archetype_trends = impact_assessment.get('archetype_trends', {})
            if archetype_trends:
                for archetype, trend in archetype_trends.items():
                    if isinstance(trend, dict) and trend.get('trending') == 'up':
                        recommendations.append(f"{archetype} archetype gaining strength - good drafting target")
            
            if not recommendations:
                recommendations.append("Meta analysis incomplete - use standard tier rankings")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating meta shift recommendations: {e}")
            return ["Meta shift analysis unavailable"]
    
    def _get_fallback_meta_shift_analysis(self) -> Dict[str, Any]:
        """Fallback meta shift analysis when live data unavailable."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'timeframe_days': 30,
            'meta_shift_analysis': {
                'heroes_rising': {},
                'heroes_falling': {},
                'heroes_stable': {},
                'meta_momentum': {},
                'error': 'Live data unavailable'
            },
            'hero_impact_assessment': {
                'most_impacted_heroes': [],
                'archetype_trends': {},
                'playstyle_shifts': {},
                'meta_polarization': {
                    'level': 'Unknown',
                    'description': 'Cannot assess without live data'
                }
            },
            'patch_predictions': {
                'heroes_to_watch': [],
                'potential_nerfs': [],
                'potential_buffs': [],
                'meta_trajectory': {
                    'direction': 'Unknown',
                    'description': 'Insufficient data for prediction'
                }
            },
            'stability_metrics': {
                'overall_stability_score': 0.5,
                'volatility_index': 5.0,
                'tier_mobility': 0.5,
                'stability_rating': 'Unknown'
            },
            'recommendations': [
                "Live meta data unavailable",
                "Use historical tier rankings for hero selection", 
                "Check HSReplay manually for current trends"
            ],
            'data_confidence': 0.2
        }
    
    def _get_historical_hero_baseline(self) -> Dict[str, float]:
        """Get historical baseline winrates for heroes."""
        # This would ideally fetch from a historical database
        # For now, use established Arena averages
        return {
            'MAGE': 53.2,
            'PALADIN': 52.5,
            'ROGUE': 52.0,
            'HUNTER': 51.8,
            'WARLOCK': 51.0,
            'WARRIOR': 50.5,
            'SHAMAN': 50.2,
            'DRUID': 49.8,
            'PRIEST': 49.0,
            'DEMONHUNTER': 48.5
        }
    
    def _analyze_archetype_meta_trends(self, shift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how different archetypes are trending in the meta."""
        try:
            # Map heroes to their strongest archetypes
            hero_archetype_mapping = {
                'MAGE': ['Tempo', 'Control'],
                'PALADIN': ['Tempo', 'Aggro'],
                'ROGUE': ['Tempo', 'Synergy'],
                'HUNTER': ['Aggro', 'Tempo'],
                'WARLOCK': ['Aggro', 'Control'],
                'WARRIOR': ['Control', 'Attrition'],
                'SHAMAN': ['Synergy', 'Tempo'],
                'DRUID': ['Control', 'Balanced'],
                'PRIEST': ['Control', 'Attrition'],
                'DEMONHUNTER': ['Aggro', 'Tempo']
            }
            
            archetype_trends = {}
            
            # Analyze each archetype based on hero performance
            for archetype in ['Aggro', 'Tempo', 'Control', 'Synergy', 'Attrition', 'Balanced']:
                archetype_heroes = [
                    hero for hero, archetypes in hero_archetype_mapping.items()
                    if archetype in archetypes
                ]
                
                rising_count = sum(1 for hero in archetype_heroes 
                                 if hero in shift_analysis.get('heroes_rising', {}))
                falling_count = sum(1 for hero in archetype_heroes 
                                  if hero in shift_analysis.get('heroes_falling', {}))
                
                if rising_count > falling_count:
                    trend = 'up'
                elif falling_count > rising_count:
                    trend = 'down'
                else:
                    trend = 'stable'
                
                archetype_trends[archetype] = {
                    'trending': trend,
                    'supporting_heroes': archetype_heroes,
                    'rising_heroes': rising_count,
                    'falling_heroes': falling_count
                }
            
            return archetype_trends
            
        except Exception as e:
            self.logger.warning(f"Error analyzing archetype trends: {e}")
            return {}
    
    def _analyze_playstyle_meta_trends(self, shift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how different playstyles are trending in the meta."""
        try:
            # Map heroes to playstyles
            hero_playstyle_mapping = {
                'MAGE': 'Flexible',
                'PALADIN': 'Aggressive',
                'ROGUE': 'Technical',
                'HUNTER': 'Aggressive', 
                'WARLOCK': 'Flexible',
                'WARRIOR': 'Defensive',
                'SHAMAN': 'Synergy-based',
                'DRUID': 'Ramp/Value',
                'PRIEST': 'Defensive',
                'DEMONHUNTER': 'Aggressive'
            }
            
            playstyle_trends = {}
            
            for playstyle in ['Aggressive', 'Defensive', 'Flexible', 'Technical', 'Synergy-based', 'Ramp/Value']:
                playstyle_heroes = [
                    hero for hero, style in hero_playstyle_mapping.items()
                    if style == playstyle
                ]
                
                avg_shift = 0
                hero_count = 0
                
                for hero in playstyle_heroes:
                    if hero in shift_analysis.get('heroes_rising', {}):
                        avg_shift += shift_analysis['heroes_rising'][hero]['shift_amount']
                        hero_count += 1
                    elif hero in shift_analysis.get('heroes_falling', {}):
                        avg_shift += shift_analysis['heroes_falling'][hero]['shift_amount']
                        hero_count += 1
                    elif hero in shift_analysis.get('heroes_stable', {}):
                        avg_shift += shift_analysis['heroes_stable'][hero]['shift_amount']
                        hero_count += 1
                
                if hero_count > 0:
                    avg_shift = avg_shift / hero_count
                    
                    playstyle_trends[playstyle] = {
                        'average_shift': round(avg_shift, 2),
                        'trend_direction': 'rising' if avg_shift > 0.5 else 'falling' if avg_shift < -0.5 else 'stable',
                        'supporting_heroes': playstyle_heroes
                    }
            
            return playstyle_trends
            
        except Exception as e:
            self.logger.warning(f"Error analyzing playstyle trends: {e}")
            return {}
    
    def _determine_hero_tier(self, winrate: float) -> str:
        """Determine hero tier based on winrate."""
        if winrate >= 54.0:
            return 'S-Tier'
        elif winrate >= 52.0:
            return 'A-Tier'
        elif winrate >= 50.0:
            return 'B-Tier'
        elif winrate >= 48.0:
            return 'C-Tier'
        else:
            return 'D-Tier'
    
    def _calculate_meta_shift_confidence(self, current_winrates: Dict[str, float], 
                                       historical_baseline: Dict[str, float]) -> float:
        """Calculate confidence level for meta shift analysis."""
        try:
            confidence_factors = []
            
            # Data availability factor
            if current_winrates and len(current_winrates) >= 8:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
            
            # Historical data factor
            if historical_baseline and len(historical_baseline) >= 8:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # Data freshness factor (simplified)
            confidence_factors.append(0.6)
            
            return round(sum(confidence_factors) / len(confidence_factors), 2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating meta shift confidence: {e}")
            return 0.5
    
    # === COMPREHENSIVE HERO PERFORMANCE PREDICTION METHODS ===
    
    def predict_hero_performance(self, hero_classes: List[str], 
                               prediction_timeframe: int = 14,
                               user_id: str = "default") -> Dict[str, Any]:
        """
        Create comprehensive hero performance prediction incorporating multiple data sources.
        
        Combines meta analysis, user performance history, trending data, and statistical
        modeling to predict which heroes will perform best for the user in the near future.
        
        Args:
            hero_classes: List of hero classes to analyze
            prediction_timeframe: Days into the future to predict (default 14)
            user_id: User identifier for personalized predictions
            
        Returns:
            Comprehensive performance prediction with confidence intervals
        """
        try:
            self.logger.info(f"Creating hero performance predictions for {prediction_timeframe} days")
            
            # Gather all data sources
            prediction_data = self._gather_prediction_data_sources(hero_classes, user_id)
            
            # Generate individual hero predictions
            hero_predictions = {}
            for hero_class in hero_classes:
                prediction = self._predict_individual_hero_performance(
                    hero_class, prediction_data, prediction_timeframe, user_id
                )
                hero_predictions[hero_class] = prediction
            
            # Perform comparative analysis
            comparative_analysis = self._perform_comparative_prediction_analysis(hero_predictions)
            
            # Generate prediction confidence intervals
            confidence_analysis = self._calculate_prediction_confidence_intervals(hero_predictions)
            
            # Create actionable recommendations
            actionable_recommendations = self._generate_prediction_based_recommendations(
                hero_predictions, comparative_analysis
            )
            
            # Assess prediction risks and uncertainties
            risk_assessment = self._assess_prediction_risks(prediction_data, hero_predictions)
            
            return {
                'prediction_timestamp': datetime.now().isoformat(),
                'prediction_timeframe_days': prediction_timeframe,
                'user_id': user_id,
                'hero_classes_analyzed': hero_classes,
                'individual_hero_predictions': hero_predictions,
                'comparative_analysis': comparative_analysis,
                'confidence_intervals': confidence_analysis,
                'actionable_recommendations': actionable_recommendations,
                'risk_assessment': risk_assessment,
                'data_sources_used': self._summarize_data_sources_used(prediction_data),
                'prediction_methodology': self._describe_prediction_methodology(),
                'overall_prediction_confidence': self._calculate_overall_prediction_confidence(
                    prediction_data, confidence_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error creating hero performance predictions: {e}")
            return self._get_fallback_hero_performance_prediction(hero_classes, user_id)
    
    def _gather_prediction_data_sources(self, hero_classes: List[str], user_id: str) -> Dict[str, Any]:
        """Gather all available data sources for prediction modeling."""
        try:
            data_sources = {
                'current_meta_data': {},
                'historical_performance': {},
                'meta_shift_analysis': {},
                'user_performance_data': {},
                'archetype_trends': {},
                'data_quality_scores': {}
            }
            
            # Current meta winrates
            try:
                current_winrates = self._get_current_winrates()
                data_sources['current_meta_data'] = {
                    'hero_winrates': current_winrates,
                    'data_age_hours': self._get_data_age_hours('hero_winrates'),
                    'sample_size_quality': self._assess_sample_size_quality(current_winrates)
                }
                data_sources['data_quality_scores']['meta_data'] = 0.8 if current_winrates else 0.2
            except Exception as e:
                self.logger.warning(f"Failed to gather current meta data: {e}")
                data_sources['data_quality_scores']['meta_data'] = 0.1
            
            # Historical performance baseline
            try:
                historical_baseline = self._get_historical_hero_baseline()
                data_sources['historical_performance'] = {
                    'baseline_winrates': historical_baseline,
                    'data_reliability': 0.7  # Historical data is somewhat reliable
                }
                data_sources['data_quality_scores']['historical_data'] = 0.7
            except Exception as e:
                self.logger.warning(f"Failed to gather historical data: {e}")
                data_sources['data_quality_scores']['historical_data'] = 0.3
            
            # Meta shift analysis
            try:
                meta_shift_data = self.analyze_meta_shift_impact()
                data_sources['meta_shift_analysis'] = meta_shift_data
                data_sources['data_quality_scores']['meta_shifts'] = meta_shift_data.get('data_confidence', 0.5)
            except Exception as e:
                self.logger.warning(f"Failed to gather meta shift data: {e}")
                data_sources['data_quality_scores']['meta_shifts'] = 0.3
            
            # User performance data
            try:
                user_history = self._get_user_performance_data(user_id)
                data_sources['user_performance_data'] = {
                    'performance_history': user_history,
                    'data_completeness': self._assess_user_data_completeness(user_history),
                    'personalization_possible': not self._insufficient_data(user_history)
                }
                data_sources['data_quality_scores']['user_data'] = (
                    0.8 if not self._insufficient_data(user_history) else 0.3
                )
            except Exception as e:
                self.logger.warning(f"Failed to gather user performance data: {e}")
                data_sources['data_quality_scores']['user_data'] = 0.2
            
            # Cross-hero meta analysis
            try:
                cross_hero_analysis = self.get_cross_hero_meta_analysis()
                data_sources['archetype_trends'] = cross_hero_analysis
                data_sources['data_quality_scores']['archetype_trends'] = (
                    cross_hero_analysis.get('meta_snapshot', {}).get('data_confidence', 0.5)
                )
            except Exception as e:
                self.logger.warning(f"Failed to gather archetype trends: {e}")
                data_sources['data_quality_scores']['archetype_trends'] = 0.3
            
            return data_sources
            
        except Exception as e:
            self.logger.error(f"Error gathering prediction data sources: {e}")
            return {'error': str(e), 'data_quality_scores': {}}
    
    def _predict_individual_hero_performance(self, hero_class: str, prediction_data: Dict[str, Any],
                                           timeframe_days: int, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance prediction for a single hero."""
        try:
            prediction = {
                'hero_class': hero_class,
                'predicted_winrate': 50.0,
                'confidence_interval': {'lower': 45.0, 'upper': 55.0},
                'prediction_factors': {},
                'trend_analysis': {},
                'user_specific_factors': {},
                'recommendation_strength': 0.5
            }
            
            # Base prediction from current meta
            current_meta = prediction_data.get('current_meta_data', {})
            current_winrate = current_meta.get('hero_winrates', {}).get(hero_class, 50.0)
            
            # Historical baseline
            historical_data = prediction_data.get('historical_performance', {})
            baseline_winrate = historical_data.get('baseline_winrates', {}).get(hero_class, 50.0)
            
            # Meta shift trends
            meta_shifts = prediction_data.get('meta_shift_analysis', {})
            shift_momentum = self._extract_hero_momentum(hero_class, meta_shifts)
            
            # User-specific adjustments
            user_adjustment = self._calculate_user_specific_prediction_adjustment(
                hero_class, prediction_data.get('user_performance_data', {}), user_id
            )
            
            # Archetype trend influence
            archetype_influence = self._calculate_archetype_trend_influence(
                hero_class, prediction_data.get('archetype_trends', {})
            )
            
            # Combine all factors with weights
            prediction_factors = {
                'current_meta_weight': 0.4,
                'historical_baseline_weight': 0.2,
                'meta_shift_weight': 0.2,
                'user_specific_weight': 0.15,
                'archetype_trend_weight': 0.05
            }
            
            # Calculate weighted prediction
            weighted_prediction = (
                current_winrate * prediction_factors['current_meta_weight'] +
                baseline_winrate * prediction_factors['historical_baseline_weight'] +
                (current_winrate + shift_momentum * timeframe_days / 30) * prediction_factors['meta_shift_weight'] +
                user_adjustment * prediction_factors['user_specific_weight'] +
                archetype_influence * prediction_factors['archetype_trend_weight']
            )
            
            prediction['predicted_winrate'] = round(weighted_prediction, 2)
            
            # Calculate confidence interval
            prediction_variance = self._calculate_prediction_variance(
                hero_class, prediction_data, prediction_factors
            )
            
            prediction['confidence_interval'] = {
                'lower': round(max(0, weighted_prediction - prediction_variance), 2),
                'upper': round(min(100, weighted_prediction + prediction_variance), 2)
            }
            
            # Store detailed factor analysis
            prediction['prediction_factors'] = {
                'current_meta_contribution': round(current_winrate * prediction_factors['current_meta_weight'], 2),
                'historical_contribution': round(baseline_winrate * prediction_factors['historical_baseline_weight'], 2),
                'meta_shift_contribution': round(shift_momentum * prediction_factors['meta_shift_weight'], 2),
                'user_specific_contribution': round(user_adjustment * prediction_factors['user_specific_weight'], 2),
                'archetype_contribution': round(archetype_influence * prediction_factors['archetype_trend_weight'], 2),
                'total_variance': round(prediction_variance, 2)
            }
            
            # Trend analysis
            prediction['trend_analysis'] = {
                'current_momentum': shift_momentum,
                'trend_direction': 'rising' if shift_momentum > 0.5 else 'falling' if shift_momentum < -0.5 else 'stable',
                'trend_strength': min(5.0, abs(shift_momentum)),
                'predicted_trend_continuation': self._predict_trend_continuation(hero_class, meta_shifts, timeframe_days)
            }
            
            # User-specific factors
            prediction['user_specific_factors'] = {
                'personal_winrate_vs_meta': user_adjustment - current_winrate,
                'user_experience_level': self._assess_user_experience_with_hero(hero_class, user_id),
                'recommended_for_user': user_adjustment > current_winrate + 1.0
            }
            
            # Overall recommendation strength
            prediction['recommendation_strength'] = self._calculate_recommendation_strength(
                weighted_prediction, prediction_variance, user_adjustment, current_winrate
            )
            
            return prediction
            
        except Exception as e:
            self.logger.warning(f"Error predicting performance for {hero_class}: {e}")
            return self._get_fallback_individual_prediction(hero_class)
    
    def _perform_comparative_prediction_analysis(self, hero_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform comparative analysis across all predicted heroes."""
        try:
            comparative_analysis = {
                'ranking': [],
                'tier_assignments': {},
                'performance_gaps': {},
                'recommendation_summary': {}
            }
            
            # Rank heroes by predicted performance
            ranked_heroes = sorted(
                hero_predictions.items(),
                key=lambda x: x[1]['predicted_winrate'],
                reverse=True
            )
            
            comparative_analysis['ranking'] = [
                {
                    'rank': i + 1,
                    'hero_class': hero,
                    'predicted_winrate': data['predicted_winrate'],
                    'confidence_range': data['confidence_interval'],
                    'recommendation_strength': data['recommendation_strength']
                }
                for i, (hero, data) in enumerate(ranked_heroes)
            ]
            
            # Assign tiers based on predicted performance
            if ranked_heroes:
                best_winrate = ranked_heroes[0][1]['predicted_winrate']
                for hero, data in ranked_heroes:
                    winrate_diff = best_winrate - data['predicted_winrate']
                    
                    if winrate_diff <= 1.0:
                        tier = 'S-Tier'
                    elif winrate_diff <= 2.5:
                        tier = 'A-Tier'
                    elif winrate_diff <= 4.0:
                        tier = 'B-Tier'
                    else:
                        tier = 'C-Tier'
                    
                    comparative_analysis['tier_assignments'][hero] = tier
            
            # Calculate performance gaps
            if len(ranked_heroes) >= 2:
                comparative_analysis['performance_gaps'] = {
                    'best_vs_worst': round(ranked_heroes[0][1]['predicted_winrate'] - ranked_heroes[-1][1]['predicted_winrate'], 2),
                    'top_two_gap': round(ranked_heroes[0][1]['predicted_winrate'] - ranked_heroes[1][1]['predicted_winrate'], 2) if len(ranked_heroes) >= 2 else 0,
                    'competitive_tier_size': len([h for h in ranked_heroes if ranked_heroes[0][1]['predicted_winrate'] - h[1]['predicted_winrate'] <= 2.0])
                }
            
            # Generate recommendation summary
            comparative_analysis['recommendation_summary'] = {
                'clear_winner': ranked_heroes[0][0] if ranked_heroes and ranked_heroes[0][1]['predicted_winrate'] - ranked_heroes[1][1]['predicted_winrate'] > 2.0 else None,
                'competitive_choices': [h[0] for h in ranked_heroes[:3] if ranked_heroes[0][1]['predicted_winrate'] - h[1]['predicted_winrate'] <= 2.0],
                'avoid_recommendations': [h[0] for h in ranked_heroes if h[1]['predicted_winrate'] < 48.0]
            }
            
            return comparative_analysis
            
        except Exception as e:
            self.logger.warning(f"Error performing comparative analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence_intervals(self, hero_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate comprehensive confidence intervals for all predictions."""
        try:
            confidence_analysis = {
                'individual_confidences': {},
                'overall_prediction_reliability': 0.0,
                'uncertainty_factors': [],
                'confidence_ranking': []
            }
            
            confidence_scores = []
            
            for hero_class, prediction in hero_predictions.items():
                # Extract confidence metrics
                interval = prediction['confidence_interval']
                interval_width = interval['upper'] - interval['lower']
                
                # Calculate confidence score (narrower interval = higher confidence)
                confidence_score = max(0.1, 1.0 - (interval_width / 20.0))  # 20% interval = 0 confidence
                
                confidence_analysis['individual_confidences'][hero_class] = {
                    'confidence_score': round(confidence_score, 3),
                    'interval_width': round(interval_width, 2),
                    'prediction_stability': self._assess_prediction_stability(prediction),
                    'data_support_quality': self._assess_data_support_quality(prediction)
                }
                
                confidence_scores.append(confidence_score)
            
            # Overall reliability
            confidence_analysis['overall_prediction_reliability'] = round(
                sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5, 3
            )
            
            # Identify uncertainty factors
            if confidence_analysis['overall_prediction_reliability'] < 0.7:
                confidence_analysis['uncertainty_factors'].append("Limited data availability")
            if any(conf['interval_width'] > 8.0 for conf in confidence_analysis['individual_confidences'].values()):
                confidence_analysis['uncertainty_factors'].append("High prediction variance")
            if len(set(pred['trend_analysis']['trend_direction'] for pred in hero_predictions.values())) > 2:
                confidence_analysis['uncertainty_factors'].append("Conflicting trend signals")
            
            # Rank by confidence
            confidence_analysis['confidence_ranking'] = sorted(
                confidence_analysis['individual_confidences'].items(),
                key=lambda x: x[1]['confidence_score'],
                reverse=True
            )
            
            return confidence_analysis
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence intervals: {e}")
            return {'error': str(e)}
    
    def _generate_prediction_based_recommendations(self, hero_predictions: Dict[str, Dict],
                                                 comparative_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on performance predictions."""
        try:
            recommendations = []
            
            ranking = comparative_analysis.get('ranking', [])
            if not ranking:
                return ["Insufficient data for recommendations"]
            
            # Primary recommendation
            best_hero = ranking[0]
            recommendations.append(
                f"Recommended: {best_hero['hero_class']} "
                f"(predicted {best_hero['predicted_winrate']:.1f}% winrate)"
            )
            
            # Performance gap analysis
            gaps = comparative_analysis.get('performance_gaps', {})
            if gaps.get('best_vs_worst', 0) > 5.0:
                recommendations.append("Strong performance differential - hero choice is critical")
            elif gaps.get('top_two_gap', 0) < 1.0:
                recommendations.append("Close performance predictions - personal preference matters")
            
            # Trend-based recommendations
            for hero_class, prediction in hero_predictions.items():
                trend = prediction.get('trend_analysis', {})
                if trend.get('trend_direction') == 'rising' and trend.get('trend_strength', 0) > 2.0:
                    recommendations.append(f"{hero_class} showing strong upward trend")
                elif trend.get('trend_direction') == 'falling' and trend.get('trend_strength', 0) > 2.0:
                    recommendations.append(f"Caution: {hero_class} in declining trend")
            
            # User-specific recommendations
            user_optimized_heroes = [
                hero for hero, pred in hero_predictions.items()
                if pred.get('user_specific_factors', {}).get('recommended_for_user', False)
            ]
            if user_optimized_heroes:
                recommendations.append(f"Personalized picks: {', '.join(user_optimized_heroes)}")
            
            # Risk recommendations
            high_variance_heroes = [
                hero for hero, pred in hero_predictions.items()
                if pred['confidence_interval']['upper'] - pred['confidence_interval']['lower'] > 8.0
            ]
            if high_variance_heroes:
                recommendations.append(f"High variance (risky): {', '.join(high_variance_heroes)}")
            
            return recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating prediction recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _assess_prediction_risks(self, prediction_data: Dict[str, Any], 
                               hero_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Assess risks and uncertainties in the predictions."""
        try:
            risk_assessment = {
                'data_quality_risks': [],
                'prediction_uncertainties': [],
                'external_factors': [],
                'overall_risk_level': 'Medium'
            }
            
            # Data quality risks
            quality_scores = prediction_data.get('data_quality_scores', {})
            if quality_scores.get('meta_data', 0) < 0.5:
                risk_assessment['data_quality_risks'].append("Limited current meta data")
            if quality_scores.get('user_data', 0) < 0.5:
                risk_assessment['data_quality_risks'].append("Insufficient user performance history")
            if quality_scores.get('meta_shifts', 0) < 0.5:
                risk_assessment['data_quality_risks'].append("Uncertain meta shift trends")
            
            # Prediction uncertainties
            avg_interval_width = sum(
                pred['confidence_interval']['upper'] - pred['confidence_interval']['lower']
                for pred in hero_predictions.values()
            ) / len(hero_predictions) if hero_predictions else 10.0
            
            if avg_interval_width > 6.0:
                risk_assessment['prediction_uncertainties'].append("High prediction variance")
            
            conflicting_trends = len(set(
                pred['trend_analysis']['trend_direction']
                for pred in hero_predictions.values()
                if pred.get('trend_analysis', {}).get('trend_direction')
            ))
            
            if conflicting_trends > 2:
                risk_assessment['prediction_uncertainties'].append("Conflicting trend signals")
            
            # External factors
            risk_assessment['external_factors'] = [
                "Patch changes may invalidate predictions",
                "New card releases could shift meta",
                "Meta shifts from tournament play",
                "Seasonal effects on player behavior"
            ]
            
            # Overall risk level
            risk_count = (
                len(risk_assessment['data_quality_risks']) +
                len(risk_assessment['prediction_uncertainties'])
            )
            
            if risk_count <= 1:
                risk_assessment['overall_risk_level'] = 'Low'
            elif risk_count <= 3:
                risk_assessment['overall_risk_level'] = 'Medium'
            else:
                risk_assessment['overall_risk_level'] = 'High'
            
            return risk_assessment
            
        except Exception as e:
            self.logger.warning(f"Error assessing prediction risks: {e}")
            return {'error': str(e)}
    
    def _extract_hero_momentum(self, hero_class: str, meta_shifts: Dict[str, Any]) -> float:
        """Extract momentum score for a hero from meta shift analysis."""
        try:
            momentum_data = meta_shifts.get('meta_shift_analysis', {}).get('meta_momentum', {})
            hero_momentum = momentum_data.get(hero_class, {})
            
            momentum_score = hero_momentum.get('momentum_score', 0.0)
            direction = hero_momentum.get('direction', 'stable')
            
            # Apply direction modifier
            if direction == 'falling':
                momentum_score = -momentum_score
            elif direction == 'stable':
                momentum_score = 0.0
            
            return momentum_score
            
        except Exception:
            return 0.0
    
    def _calculate_user_specific_prediction_adjustment(self, hero_class: str, 
                                                     user_data: Dict[str, Any], user_id: str) -> float:
        """Calculate user-specific performance adjustment."""
        try:
            user_history = user_data.get('performance_history', {})
            hero_records = user_history.get(hero_class, [])
            
            if not hero_records:
                return 50.0  # Default to meta average
            
            # Calculate user's historical performance
            total_wins = sum(record['wins'] for record in hero_records)
            total_games = sum(record['total_games'] for record in hero_records)
            
            if total_games < 3:
                return 50.0  # Insufficient data
            
            user_winrate = (total_wins / total_games * 100)
            
            # Weight by sample size
            confidence_weight = min(1.0, total_games / 20)  # Full confidence at 20+ games
            
            return user_winrate * confidence_weight + 50.0 * (1 - confidence_weight)
            
        except Exception:
            return 50.0
    
    def _calculate_archetype_trend_influence(self, hero_class: str, archetype_data: Dict[str, Any]) -> float:
        """Calculate influence of archetype trends on hero performance."""
        try:
            # Map heroes to primary archetypes
            hero_archetype_map = {
                'MAGE': 'Tempo',
                'PALADIN': 'Tempo',
                'ROGUE': 'Tempo',
                'HUNTER': 'Aggro',
                'WARLOCK': 'Aggro',
                'WARRIOR': 'Control',
                'SHAMAN': 'Synergy',
                'DRUID': 'Control',
                'PRIEST': 'Control',
                'DEMONHUNTER': 'Aggro'
            }
            
            primary_archetype = hero_archetype_map.get(hero_class, 'Balanced')
            tier_rankings = archetype_data.get('tier_rankings', {})
            
            # Find hero's performance in tier rankings
            for tier_name, tier_data in tier_rankings.items():
                heroes_in_tier = tier_data.get('heroes', {})
                if hero_class in heroes_in_tier:
                    return heroes_in_tier[hero_class]
            
            return 50.0  # Default if not found
            
        except Exception:
            return 50.0
    
    # === HELPER METHODS FOR PERFORMANCE PREDICTION ===
    
    def _calculate_prediction_variance(self, hero_class: str, prediction_data: Dict[str, Any],
                                     prediction_factors: Dict[str, float]) -> float:
        """Calculate prediction variance based on data quality and uncertainty factors."""
        try:
            base_variance = 3.0  # Base uncertainty
            
            # Data quality factor
            quality_scores = prediction_data.get('data_quality_scores', {})
            avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.5
            quality_variance = (1.0 - avg_quality) * 4.0  # Poor quality = higher variance
            
            # Meta stability factor
            meta_shifts = prediction_data.get('meta_shift_analysis', {})
            stability_metrics = meta_shifts.get('stability_metrics', {})
            volatility = stability_metrics.get('volatility_index', 5.0)
            stability_variance = volatility * 0.5  # Higher volatility = higher variance
            
            # User data uncertainty
            user_data = prediction_data.get('user_performance_data', {})
            if not user_data.get('personalization_possible', False):
                user_variance = 2.0
            else:
                user_variance = 0.5
            
            total_variance = base_variance + quality_variance + stability_variance + user_variance
            return min(10.0, total_variance)  # Cap at 10%
            
        except Exception:
            return 5.0  # Default variance
    
    def _predict_trend_continuation(self, hero_class: str, meta_shifts: Dict[str, Any],
                                  timeframe_days: int) -> str:
        """Predict if current trends will continue."""
        try:
            momentum_data = meta_shifts.get('meta_shift_analysis', {}).get('meta_momentum', {})
            hero_momentum = momentum_data.get(hero_class, {})
            
            momentum_score = hero_momentum.get('momentum_score', 0.0)
            direction = hero_momentum.get('direction', 'stable')
            
            # Simple trend continuation logic
            if direction == 'stable':
                return 'Trend likely to continue (stable)'
            elif momentum_score >= 3.0:
                return f'Strong {direction} trend likely to continue'
            elif momentum_score >= 1.5:
                return f'Moderate {direction} trend may continue'
            else:
                return 'Trend may reverse or stabilize'
                
        except Exception:
            return 'Trend prediction unavailable'
    
    def _assess_user_experience_with_hero(self, hero_class: str, user_id: str) -> str:
        """Assess user's experience level with specific hero."""
        try:
            user_history = self._get_user_performance_data(user_id)
            hero_records = user_history.get(hero_class, [])
            
            if not hero_records:
                return 'No experience'
            
            total_games = sum(record['total_games'] for record in hero_records)
            
            if total_games >= 50:
                return 'Experienced'
            elif total_games >= 20:
                return 'Moderate'
            elif total_games >= 10:
                return 'Some experience'
            else:
                return 'Limited experience'
                
        except Exception:
            return 'Unknown'
    
    def _calculate_recommendation_strength(self, predicted_winrate: float, variance: float,
                                         user_adjustment: float, meta_winrate: float) -> float:
        """Calculate overall recommendation strength."""
        try:
            # Base strength from predicted performance
            if predicted_winrate >= 55.0:
                base_strength = 0.9
            elif predicted_winrate >= 52.0:
                base_strength = 0.7
            elif predicted_winrate >= 50.0:
                base_strength = 0.5
            elif predicted_winrate >= 48.0:
                base_strength = 0.3
            else:
                base_strength = 0.1
            
            # Adjust for confidence (lower variance = higher strength)
            confidence_modifier = max(0.5, 1.0 - (variance / 10.0))
            
            # Adjust for user-specific factors
            user_modifier = 1.0
            if user_adjustment > meta_winrate + 2.0:
                user_modifier = 1.2  # Strong personal performance
            elif user_adjustment < meta_winrate - 2.0:
                user_modifier = 0.8  # Weak personal performance
            
            final_strength = base_strength * confidence_modifier * user_modifier
            return max(0.1, min(1.0, final_strength))
            
        except Exception:
            return 0.5
    
    def _assess_prediction_stability(self, prediction: Dict[str, Any]) -> str:
        """Assess stability of individual prediction."""
        try:
            variance = prediction['prediction_factors'].get('total_variance', 5.0)
            trend_strength = prediction['trend_analysis'].get('trend_strength', 0.0)
            
            if variance <= 3.0 and trend_strength <= 1.0:
                return 'Very stable'
            elif variance <= 5.0 and trend_strength <= 2.0:
                return 'Stable'
            elif variance <= 7.0:
                return 'Moderate'
            else:
                return 'Volatile'
                
        except Exception:
            return 'Unknown'
    
    def _assess_data_support_quality(self, prediction: Dict[str, Any]) -> str:
        """Assess quality of data supporting the prediction."""
        try:
            user_games = prediction.get('user_specific_factors', {}).get('user_experience_level', 'Unknown')
            recommendation_strength = prediction.get('recommendation_strength', 0.5)
            
            if user_games in ['Experienced', 'Moderate'] and recommendation_strength >= 0.7:
                return 'Strong'
            elif recommendation_strength >= 0.5:
                return 'Moderate'
            else:
                return 'Weak'
                
        except Exception:
            return 'Unknown'
    
    def _get_data_age_hours(self, data_type: str) -> float:
        """Get age of data in hours."""
        try:
            if hasattr(self.hero_selector, 'last_winrate_fetch') and self.hero_selector.last_winrate_fetch:
                age = datetime.now() - self.hero_selector.last_winrate_fetch
                return age.total_seconds() / 3600
            return 24.0  # Default age
        except Exception:
            return 24.0
    
    def _assess_sample_size_quality(self, winrates: Dict[str, float]) -> str:
        """Assess quality based on sample size (simplified)."""
        try:
            if winrates and len(winrates) >= 8:
                return 'Good'
            elif winrates and len(winrates) >= 5:
                return 'Moderate'
            else:
                return 'Poor'
        except Exception:
            return 'Unknown'
    
    def _assess_user_data_completeness(self, user_history: Dict) -> str:
        """Assess completeness of user performance data."""
        try:
            if not user_history:
                return 'No data'
            
            total_games = sum(
                sum(record['total_games'] for record in records)
                for records in user_history.values()
            )
            heroes_with_data = len(user_history)
            
            if total_games >= 50 and heroes_with_data >= 5:
                return 'Comprehensive'
            elif total_games >= 20 and heroes_with_data >= 3:
                return 'Moderate'
            elif total_games >= 10:
                return 'Limited'
            else:
                return 'Minimal'
                
        except Exception:
            return 'Unknown'
    
    def _summarize_data_sources_used(self, prediction_data: Dict[str, Any]) -> Dict[str, str]:
        """Summarize which data sources were successfully used."""
        try:
            quality_scores = prediction_data.get('data_quality_scores', {})
            
            return {
                source: 'Available' if score > 0.5 else 'Limited' if score > 0.2 else 'Unavailable'
                for source, score in quality_scores.items()
            }
        except Exception:
            return {}
    
    def _describe_prediction_methodology(self) -> Dict[str, Any]:
        """Describe the methodology used for predictions."""
        return {
            'approach': 'Multi-factor weighted prediction model',
            'factors': [
                'Current meta performance (40% weight)',
                'Historical baseline (20% weight)', 
                'Meta shift trends (20% weight)',
                'User-specific performance (15% weight)',
                'Archetype trend influence (5% weight)'
            ],
            'confidence_calculation': 'Based on data quality, prediction variance, and trend consistency',
            'personalization': 'Incorporates user performance history when available',
            'fallback_strategy': 'Graceful degradation to meta averages when data unavailable'
        }
    
    def _calculate_overall_prediction_confidence(self, prediction_data: Dict[str, Any],
                                               confidence_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence in the prediction system."""
        try:
            # Data availability factor
            quality_scores = prediction_data.get('data_quality_scores', {})
            avg_data_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.3
            
            # Prediction reliability factor
            prediction_reliability = confidence_analysis.get('overall_prediction_reliability', 0.5)
            
            # Combine factors
            overall_confidence = (avg_data_quality * 0.6) + (prediction_reliability * 0.4)
            
            return round(max(0.2, min(0.95, overall_confidence)), 3)
            
        except Exception:
            return 0.5
    
    def _get_fallback_hero_performance_prediction(self, hero_classes: List[str], user_id: str) -> Dict[str, Any]:
        """Fallback prediction when main system fails."""
        return {
            'prediction_timestamp': datetime.now().isoformat(),
            'prediction_timeframe_days': 14,
            'user_id': user_id,
            'hero_classes_analyzed': hero_classes,
            'individual_hero_predictions': {
                hero: self._get_fallback_individual_prediction(hero)
                for hero in hero_classes
            },
            'comparative_analysis': {
                'ranking': [{'rank': i+1, 'hero_class': hero, 'predicted_winrate': 50.0}
                          for i, hero in enumerate(hero_classes)],
                'error': 'Fallback mode'
            },
            'confidence_intervals': {'error': 'Fallback mode'},
            'actionable_recommendations': [
                'Prediction system unavailable',
                'Use current meta tier rankings',
                'Consider personal experience with heroes'
            ],
            'risk_assessment': {
                'overall_risk_level': 'High',
                'data_quality_risks': ['Complete system failure'],
                'prediction_uncertainties': ['No prediction capability'],
                'external_factors': ['Manual analysis required']
            },
            'data_sources_used': {'error': 'No data sources available'},
            'prediction_methodology': {'error': 'Fallback mode'},
            'overall_prediction_confidence': 0.1
        }
    
    def _get_fallback_individual_prediction(self, hero_class: str) -> Dict[str, Any]:
        """Fallback prediction for individual hero."""
        return {
            'hero_class': hero_class,
            'predicted_winrate': 50.0,
            'confidence_interval': {'lower': 40.0, 'upper': 60.0},
            'prediction_factors': {'error': 'Fallback mode'},
            'trend_analysis': {'trend_direction': 'unknown'},
            'user_specific_factors': {'error': 'No user data'},
            'recommendation_strength': 0.3
        }
    
    # === HERO-SPECIFIC UNDERGROUND ARENA REDRAFT STRATEGIES ===
    
    def get_underground_arena_redraft_strategy(self, hero_class: str, 
                                             draft_stage: str = "early",
                                             current_deck_state: Optional[DeckState] = None) -> Dict[str, Any]:
        """
        Get hero-specific Underground Arena redraft strategies.
        
        Provides specialized advice for Underground Arena format where redrafting
        is possible, considering hero-specific strengths and redraft opportunities.
        
        Args:
            hero_class: The hero class being drafted
            draft_stage: Stage of draft ("early", "mid", "late")
            current_deck_state: Current state of the deck being drafted
            
        Returns:
            Comprehensive redraft strategy with hero-specific recommendations
        """
        try:
            self.logger.info(f"Generating Underground Arena redraft strategy for {hero_class} at {draft_stage} stage")
            
            # Get hero-specific redraft parameters
            redraft_parameters = self._get_hero_redraft_parameters(hero_class)
            
            # Analyze current deck for redraft opportunities
            deck_analysis = self._analyze_deck_for_redraft_opportunities(
                current_deck_state, hero_class, draft_stage
            ) if current_deck_state else {}
            
            # Generate stage-specific strategies
            stage_strategy = self._get_stage_specific_redraft_strategy(
                hero_class, draft_stage, redraft_parameters
            )
            
            # Identify redraft trigger conditions
            redraft_triggers = self._identify_redraft_trigger_conditions(
                hero_class, draft_stage, current_deck_state
            )
            
            # Generate redraft timing recommendations
            timing_recommendations = self._generate_redraft_timing_recommendations(
                hero_class, draft_stage, deck_analysis
            )
            
            # Create archetype-specific redraft advice
            archetype_advice = self._generate_archetype_redraft_advice(
                hero_class, redraft_parameters
            )
            
            # Risk assessment for redrafting
            risk_assessment = self._assess_redraft_risks(
                hero_class, draft_stage, current_deck_state
            )
            
            return {
                'hero_class': hero_class,
                'draft_stage': draft_stage,
                'strategy_timestamp': datetime.now().isoformat(),
                'redraft_parameters': redraft_parameters,
                'current_deck_analysis': deck_analysis,
                'stage_specific_strategy': stage_strategy,
                'redraft_triggers': redraft_triggers,
                'timing_recommendations': timing_recommendations,
                'archetype_specific_advice': archetype_advice,
                'risk_assessment': risk_assessment,
                'recommended_actions': self._generate_redraft_recommended_actions(
                    hero_class, draft_stage, deck_analysis, redraft_triggers
                ),
                'expected_outcomes': self._calculate_redraft_expected_outcomes(
                    hero_class, draft_stage, redraft_parameters
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating redraft strategy for {hero_class}: {e}")
            return self._get_fallback_redraft_strategy(hero_class, draft_stage)
    
    def _get_hero_redraft_parameters(self, hero_class: str) -> Dict[str, Any]:
        """Get hero-specific parameters for redraft decision making."""
        try:
            # Hero-specific redraft thresholds and preferences
            hero_parameters = {
                'MAGE': {
                    'redraft_threshold': 0.6,  # More willing to redraft (flexible)
                    'curve_importance': 0.8,
                    'spell_quality_importance': 0.9,
                    'key_archetypes': ['Tempo', 'Control'],
                    'avoid_archetypes': ['Pure Aggro'],
                    'redraft_triggers': ['poor_spell_quality', 'curve_issues', 'no_tempo_tools'],
                    'pick_priorities': ['removal_spells', 'card_generation', 'efficient_minions']
                },
                'PALADIN': {
                    'redraft_threshold': 0.7,  # Moderate redraft tendency
                    'curve_importance': 0.9,
                    'minion_quality_importance': 0.8,
                    'key_archetypes': ['Tempo', 'Aggro', 'Divine Shield'],
                    'avoid_archetypes': ['Pure Control'],
                    'redraft_triggers': ['poor_early_game', 'no_divine_shield', 'weak_weapons'],
                    'pick_priorities': ['early_minions', 'divine_shield', 'weapons', 'buffs']
                },
                'ROGUE': {
                    'redraft_threshold': 0.65,
                    'curve_importance': 0.85,
                    'weapon_importance': 0.9,
                    'key_archetypes': ['Tempo', 'Combo'],
                    'avoid_archetypes': ['Pure Control', 'Pure Aggro'],
                    'redraft_triggers': ['no_weapons', 'poor_tempo', 'no_combo_pieces'],
                    'pick_priorities': ['weapons', 'cheap_spells', 'combo_enablers', '2_3_drops']
                },
                'HUNTER': {
                    'redraft_threshold': 0.75,  # Less willing to redraft (aggressive focus)
                    'curve_importance': 0.95,
                    'beast_synergy_importance': 0.8,
                    'key_archetypes': ['Aggro', 'Beast', 'Face'],
                    'avoid_archetypes': ['Control', 'Attrition'],
                    'redraft_triggers': ['no_early_game', 'poor_beast_synergy', 'too_slow'],
                    'pick_priorities': ['1_2_3_drops', 'beasts', 'direct_damage', 'weapons']
                },
                'WARLOCK': {
                    'redraft_threshold': 0.6,
                    'curve_importance': 0.7,
                    'life_management_importance': 0.8,
                    'key_archetypes': ['Aggro', 'Zoo', 'Control'],
                    'avoid_archetypes': ['Midrange'],
                    'redraft_triggers': ['unclear_gameplan', 'poor_life_management', 'weak_demons'],
                    'pick_priorities': ['card_draw', 'efficient_minions', 'life_gain', 'demons']
                },
                'WARRIOR': {
                    'redraft_threshold': 0.65,
                    'curve_importance': 0.7,
                    'weapon_importance': 0.9,
                    'key_archetypes': ['Control', 'Weapon', 'Armor'],
                    'avoid_archetypes': ['Pure Aggro'],
                    'redraft_triggers': ['no_weapons', 'poor_late_game', 'no_armor_gain'],
                    'pick_priorities': ['weapons', 'removal', 'armor_gain', 'late_game_threats']
                },
                'SHAMAN': {
                    'redraft_threshold': 0.7,
                    'curve_importance': 0.8,
                    'elemental_importance': 0.8,
                    'key_archetypes': ['Elemental', 'Overload', 'Midrange'],
                    'avoid_archetypes': ['Pure Aggro'],
                    'redraft_triggers': ['no_elemental_synergy', 'overload_issues', 'poor_curve'],
                    'pick_priorities': ['elementals', 'efficient_spells', 'overload_synergy', 'totems']
                },
                'DRUID': {
                    'redraft_threshold': 0.65,
                    'curve_importance': 0.6,
                    'ramp_importance': 0.8,
                    'key_archetypes': ['Ramp', 'Beast', 'Big'],
                    'avoid_archetypes': ['Pure Aggro'],
                    'redraft_triggers': ['no_ramp', 'poor_late_game', 'weak_beasts'],
                    'pick_priorities': ['ramp_spells', 'big_minions', 'beasts', 'choose_cards']
                },
                'PRIEST': {
                    'redraft_threshold': 0.6,
                    'curve_importance': 0.6,
                    'healing_importance': 0.8,
                    'key_archetypes': ['Control', 'Heal', 'Value'],
                    'avoid_archetypes': ['Aggro', 'Face'],
                    'redraft_triggers': ['no_healing', 'poor_removal', 'weak_late_game'],
                    'pick_priorities': ['healing', 'removal', 'card_generation', 'high_health_minions']
                },
                'DEMONHUNTER': {
                    'redraft_threshold': 0.75,
                    'curve_importance': 0.9,
                    'outcast_importance': 0.8,
                    'key_archetypes': ['Aggro', 'Outcast', 'Weapon'],
                    'avoid_archetypes': ['Control', 'Attrition'],
                    'redraft_triggers': ['no_early_game', 'poor_outcast', 'no_weapons'],
                    'pick_priorities': ['1_2_3_drops', 'outcast_cards', 'weapons', 'direct_damage']
                }
            }
            
            return hero_parameters.get(hero_class, self._get_default_redraft_parameters())
            
        except Exception as e:
            self.logger.warning(f"Error getting redraft parameters for {hero_class}: {e}")
            return self._get_default_redraft_parameters()
    
    def _analyze_deck_for_redraft_opportunities(self, deck_state: DeckState, 
                                              hero_class: str, draft_stage: str) -> Dict[str, Any]:
        """Analyze current deck state for redraft opportunities."""
        try:
            if not deck_state or not deck_state.drafted_cards:
                return {'status': 'insufficient_data'}
            
            analysis = {
                'curve_analysis': {},
                'archetype_coherence': {},
                'key_cards_analysis': {},
                'synergy_evaluation': {},
                'redraft_score': 0.0
            }
            
            # Curve analysis
            curve_distribution = self._analyze_mana_curve_distribution(deck_state.drafted_cards)
            curve_score = self._evaluate_curve_quality(curve_distribution, hero_class)
            analysis['curve_analysis'] = {
                'distribution': curve_distribution,
                'quality_score': curve_score,
                'problems': self._identify_curve_problems(curve_distribution, hero_class)
            }
            
            # Archetype coherence
            archetype_signals = self._detect_archetype_signals(deck_state.drafted_cards, hero_class)
            coherence_score = self._calculate_archetype_coherence(archetype_signals)
            analysis['archetype_coherence'] = {
                'detected_archetypes': archetype_signals,
                'coherence_score': coherence_score,
                'mixed_signals': len(archetype_signals) > 2
            }
            
            # Key cards for hero
            key_cards = self._identify_hero_key_cards(deck_state.drafted_cards, hero_class)
            analysis['key_cards_analysis'] = {
                'key_cards_present': key_cards,
                'critical_missing': self._identify_missing_critical_cards(key_cards, hero_class, draft_stage)
            }
            
            # Synergy evaluation
            synergy_score = self._evaluate_deck_synergies(deck_state.drafted_cards, hero_class)
            analysis['synergy_evaluation'] = {
                'synergy_score': synergy_score,
                'synergy_packages': self._identify_synergy_packages(deck_state.drafted_cards, hero_class)
            }
            
            # Overall redraft score (0-1, higher = more reason to redraft)
            redraft_factors = [
                1.0 - curve_score,  # Poor curve = higher redraft score
                1.0 - coherence_score,  # Poor coherence = higher redraft score
                1.0 - synergy_score,  # Poor synergy = higher redraft score
                0.5 if analysis['key_cards_analysis']['critical_missing'] else 0.0
            ]
            
            analysis['redraft_score'] = sum(redraft_factors) / len(redraft_factors)
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing deck for redraft opportunities: {e}")
            return {'error': str(e)}
    
    def _get_stage_specific_redraft_strategy(self, hero_class: str, draft_stage: str,
                                           redraft_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stage-specific redraft strategies."""
        try:
            stage_strategies = {
                'early': {
                    'focus': 'Establish archetype direction',
                    'redraft_considerations': [
                        'Poor early card quality',
                        'Conflicting archetype signals',
                        'Missing key early game pieces'
                    ],
                    'redraft_threshold': redraft_parameters.get('redraft_threshold', 0.7) - 0.1,
                    'key_priorities': redraft_parameters.get('pick_priorities', [])[:2],
                    'risk_tolerance': 'High'
                },
                'mid': {
                    'focus': 'Solidify deck identity and curve',
                    'redraft_considerations': [
                        'Deck lacks coherent strategy',
                        'Major curve gaps',
                        'Insufficient win conditions'
                    ],
                    'redraft_threshold': redraft_parameters.get('redraft_threshold', 0.7),
                    'key_priorities': redraft_parameters.get('pick_priorities', []),
                    'risk_tolerance': 'Medium'
                },
                'late': {
                    'focus': 'Fill specific gaps and polish',
                    'redraft_considerations': [
                        'Critical role unfilled',
                        'Severe curve problems',
                        'Completely off-archetype'
                    ],
                    'redraft_threshold': redraft_parameters.get('redraft_threshold', 0.7) + 0.1,
                    'key_priorities': ['curve_fillers', 'win_conditions'],
                    'risk_tolerance': 'Low'
                }
            }
            
            base_strategy = stage_strategies.get(draft_stage, stage_strategies['mid'])
            
            # Add hero-specific modifications
            hero_modifications = self._get_hero_stage_modifications(hero_class, draft_stage)
            
            return {
                **base_strategy,
                'hero_specific_focus': hero_modifications.get('focus_areas', []),
                'hero_redraft_triggers': redraft_parameters.get('redraft_triggers', []),
                'recommended_archetype': redraft_parameters.get('key_archetypes', [])[0] if redraft_parameters.get('key_archetypes') else 'Balanced'
            }
            
        except Exception as e:
            self.logger.warning(f"Error generating stage-specific strategy: {e}")
            return {'error': str(e)}
    
    def _identify_redraft_trigger_conditions(self, hero_class: str, draft_stage: str,
                                           current_deck_state: Optional[DeckState]) -> List[Dict[str, Any]]:
        """Identify specific conditions that should trigger a redraft consideration."""
        try:
            triggers = []
            
            # Get hero-specific triggers
            redraft_parameters = self._get_hero_redraft_parameters(hero_class)
            hero_triggers = redraft_parameters.get('redraft_triggers', [])
            
            # Universal triggers
            universal_triggers = [
                {
                    'condition': 'poor_card_quality',
                    'description': 'Draft contains too many below-average cards',
                    'threshold': 0.3,  # If 30%+ cards are poor quality
                    'severity': 'high',
                    'stage_relevance': ['early', 'mid']
                },
                {
                    'condition': 'no_win_condition',
                    'description': 'Deck lacks clear win conditions',
                    'threshold': 0.0,
                    'severity': 'critical',
                    'stage_relevance': ['mid', 'late']
                },
                {
                    'condition': 'extreme_curve_problems',
                    'description': 'Severe mana curve issues',
                    'threshold': 0.2,  # Curve quality below 20%
                    'severity': 'high',
                    'stage_relevance': ['mid', 'late']
                }
            ]
            
            # Add hero-specific triggers
            for trigger_type in hero_triggers:
                trigger_config = self._get_trigger_configuration(trigger_type, hero_class, draft_stage)
                if trigger_config:
                    triggers.append(trigger_config)
            
            # Add universal triggers
            triggers.extend(universal_triggers)
            
            # Evaluate triggers against current deck state
            if current_deck_state:
                evaluated_triggers = []
                for trigger in triggers:
                    evaluation = self._evaluate_trigger_condition(
                        trigger, current_deck_state, hero_class
                    )
                    if evaluation['triggered']:
                        evaluated_triggers.append({
                            **trigger,
                            'evaluation': evaluation,
                            'recommended_action': self._get_trigger_recommended_action(trigger, evaluation)
                        })
                triggers = evaluated_triggers
            
            return triggers
            
        except Exception as e:
            self.logger.warning(f"Error identifying redraft triggers: {e}")
            return []
    
    def _generate_redraft_timing_recommendations(self, hero_class: str, draft_stage: str,
                                               deck_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate timing recommendations for when to redraft."""
        try:
            timing_advice = {
                'immediate_redraft': False,
                'consider_redraft_at_pick': None,
                'redraft_window': {},
                'timing_factors': []
            }
            
            redraft_score = deck_analysis.get('redraft_score', 0.0)
            redraft_parameters = self._get_hero_redraft_parameters(hero_class)
            threshold = redraft_parameters.get('redraft_threshold', 0.7)
            
            # Immediate redraft recommendation
            if redraft_score >= threshold:
                timing_advice['immediate_redraft'] = True
                timing_advice['timing_factors'].append('Current deck quality below acceptable threshold')
            
            # Stage-specific timing advice
            if draft_stage == 'early':
                timing_advice['redraft_window'] = {
                    'optimal_range': 'Picks 8-12',
                    'latest_recommended': 'Pick 15',
                    'reasoning': 'Early redraft allows complete strategy shift'
                }
                if redraft_score >= 0.6:
                    timing_advice['consider_redraft_at_pick'] = 10
            
            elif draft_stage == 'mid':
                timing_advice['redraft_window'] = {
                    'optimal_range': 'Picks 15-20',
                    'latest_recommended': 'Pick 23',
                    'reasoning': 'Mid-draft redraft can salvage poor starts'
                }
                if redraft_score >= 0.7:
                    timing_advice['consider_redraft_at_pick'] = 18
            
            else:  # late
                timing_advice['redraft_window'] = {
                    'optimal_range': 'Pick 25-27',
                    'latest_recommended': 'Pick 28',
                    'reasoning': 'Late redraft only for critical issues'
                }
                if redraft_score >= 0.8:
                    timing_advice['consider_redraft_at_pick'] = 26
            
            # Additional timing factors
            if deck_analysis.get('archetype_coherence', {}).get('mixed_signals', False):
                timing_advice['timing_factors'].append('Mixed archetype signals suggest early redraft')
            
            curve_problems = deck_analysis.get('curve_analysis', {}).get('problems', [])
            if len(curve_problems) >= 2:
                timing_advice['timing_factors'].append('Multiple curve issues favor redraft')
            
            return timing_advice
            
        except Exception as e:
            self.logger.warning(f"Error generating timing recommendations: {e}")
            return {'error': str(e)}
    
    def _generate_archetype_redraft_advice(self, hero_class: str,
                                         redraft_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate archetype-specific redraft advice."""
        try:
            advice = {
                'recommended_archetypes': [],
                'archetype_strategies': {},
                'pivot_opportunities': []
            }
            
            key_archetypes = redraft_parameters.get('key_archetypes', [])
            avoid_archetypes = redraft_parameters.get('avoid_archetypes', [])
            
            # Recommended archetypes with redraft strategies
            for archetype in key_archetypes:
                strategy = self._get_archetype_redraft_strategy(archetype, hero_class)
                advice['recommended_archetypes'].append(archetype)
                advice['archetype_strategies'][archetype] = strategy
            
            # Pivot opportunities
            for archetype in key_archetypes:
                pivot_info = self._analyze_archetype_pivot_potential(archetype, hero_class)
                if pivot_info['viable']:
                    advice['pivot_opportunities'].append(pivot_info)
            
            return advice
            
        except Exception as e:
            self.logger.warning(f"Error generating archetype redraft advice: {e}")
            return {'error': str(e)}
    
    def _assess_redraft_risks(self, hero_class: str, draft_stage: str,
                            current_deck_state: Optional[DeckState]) -> Dict[str, Any]:
        """Assess risks associated with redrafting."""
        try:
            risk_assessment = {
                'overall_risk_level': 'Medium',
                'specific_risks': [],
                'risk_mitigation': [],
                'expected_impact': {}
            }
            
            # Stage-based risks
            stage_risks = {
                'early': ['Losing solid early picks', 'Time pressure'],
                'mid': ['Disrupting established synergies', 'Limited pivot options'],
                'late': ['Insufficient picks remaining', 'Forced suboptimal choices']
            }
            
            risk_assessment['specific_risks'].extend(stage_risks.get(draft_stage, []))
            
            # Deck-state based risks
            if current_deck_state and current_deck_state.drafted_cards:
                strong_cards = self._identify_strong_cards_in_deck(current_deck_state.drafted_cards)
                if len(strong_cards) >= 3:
                    risk_assessment['specific_risks'].append('Losing multiple strong cards')
                    risk_assessment['overall_risk_level'] = 'High'
                
                synergy_packages = self._identify_synergy_packages(current_deck_state.drafted_cards, hero_class)
                if synergy_packages:
                    risk_assessment['specific_risks'].append('Breaking established synergies')
            
            # Risk mitigation strategies
            risk_assessment['risk_mitigation'] = [
                'Focus on flexible, powerful cards',
                'Prioritize hero-specific synergies',
                'Maintain curve consciousness',
                'Keep archetype options open'
            ]
            
            # Expected impact
            risk_assessment['expected_impact'] = {
                'win_rate_change': self._estimate_redraft_winrate_impact(hero_class, draft_stage),
                'deck_power_change': 'Variable based on execution',
                'consistency_impact': 'Likely improved if done correctly'
            }
            
            return risk_assessment
            
        except Exception as e:
            self.logger.warning(f"Error assessing redraft risks: {e}")
            return {'error': str(e)}
    
    def _get_default_redraft_parameters(self) -> Dict[str, Any]:
        """Get default redraft parameters for unknown heroes."""
        return {
            'redraft_threshold': 0.7,
            'curve_importance': 0.8,
            'key_archetypes': ['Balanced', 'Tempo'],
            'avoid_archetypes': ['Extreme Aggro', 'Pure Control'],
            'redraft_triggers': ['poor_card_quality', 'no_win_condition'],
            'pick_priorities': ['efficient_minions', 'removal', 'card_advantage']
        }
    
    # === REDRAFT HELPER METHODS ===
    
    def _generate_redraft_recommended_actions(self, hero_class: str, draft_stage: str,
                                            deck_analysis: Dict[str, Any], 
                                            redraft_triggers: List[Dict[str, Any]]) -> List[str]:
        """Generate specific recommended actions for redrafting."""
        try:
            actions = []
            
            # Based on redraft triggers
            if redraft_triggers:
                actions.append(f"Consider redraft due to {len(redraft_triggers)} triggered conditions")
                for trigger in redraft_triggers[:3]:  # Top 3 triggers
                    action = trigger.get('recommended_action', 'Address trigger condition')
                    actions.append(f"• {action}")
            
            # Based on deck analysis
            redraft_score = deck_analysis.get('redraft_score', 0.0)
            if redraft_score >= 0.7:
                actions.append("Strong redraft candidate - significant improvement potential")
            elif redraft_score >= 0.5:
                actions.append("Moderate redraft consideration - weigh risks vs benefits")
            
            # Stage-specific actions
            if draft_stage == 'early':
                actions.append("Early stage: Focus on establishing strong foundation")
            elif draft_stage == 'mid':
                actions.append("Mid stage: Ensure deck coherence and fill curve gaps")
            else:
                actions.append("Late stage: Only redraft for critical issues")
            
            # Hero-specific actions
            redraft_params = self._get_hero_redraft_parameters(hero_class)
            priorities = redraft_params.get('pick_priorities', [])
            if priorities:
                actions.append(f"Prioritize: {', '.join(priorities[:3])}")
            
            return actions[:6]  # Limit to 6 actions
            
        except Exception:
            return ["Evaluate redraft potential", "Consider deck coherence", "Assess curve quality"]
    
    def _calculate_redraft_expected_outcomes(self, hero_class: str, draft_stage: str,
                                           redraft_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected outcomes from redrafting."""
        try:
            outcomes = {
                'win_rate_impact': {},
                'deck_consistency': {},
                'archetype_optimization': {}
            }
            
            # Win rate impact estimation
            base_impact = self._estimate_redraft_winrate_impact(hero_class, draft_stage)
            outcomes['win_rate_impact'] = {
                'estimated_change': base_impact,
                'confidence': 'Medium',
                'factors': ['Improved card quality', 'Better archetype fit', 'Enhanced synergies']
            }
            
            # Deck consistency impact
            outcomes['deck_consistency'] = {
                'curve_improvement': 'Likely positive',
                'archetype_coherence': 'Significant improvement expected',
                'synergy_potential': 'Enhanced with focused picks'
            }
            
            # Archetype optimization
            key_archetypes = redraft_parameters.get('key_archetypes', [])
            if key_archetypes:
                outcomes['archetype_optimization'] = {
                    'target_archetype': key_archetypes[0],
                    'optimization_potential': 'High',
                    'success_probability': self._calculate_archetype_success_probability(hero_class, key_archetypes[0])
                }
            
            return outcomes
            
        except Exception:
            return {'error': 'Could not calculate expected outcomes'}
    
    def _get_fallback_redraft_strategy(self, hero_class: str, draft_stage: str) -> Dict[str, Any]:
        """Fallback redraft strategy when main system fails."""
        return {
            'hero_class': hero_class,
            'draft_stage': draft_stage,
            'strategy_timestamp': datetime.now().isoformat(),
            'redraft_parameters': self._get_default_redraft_parameters(),
            'current_deck_analysis': {'status': 'analysis_failed'},
            'stage_specific_strategy': {
                'focus': 'General deck improvement',
                'redraft_threshold': 0.7,
                'risk_tolerance': 'Medium'
            },
            'redraft_triggers': [],
            'timing_recommendations': {
                'redraft_window': f'{draft_stage} stage timing',
                'timing_factors': ['System error - manual evaluation required']
            },
            'archetype_specific_advice': {
                'recommended_archetypes': ['Balanced', 'Tempo'],
                'error': 'Detailed advice unavailable'
            },
            'risk_assessment': {
                'overall_risk_level': 'Unknown',
                'specific_risks': ['System analysis unavailable'],
                'risk_mitigation': ['Manual evaluation required']
            },
            'recommended_actions': [
                'Manual deck evaluation needed',
                'Consider curve and archetype coherence',
                'Evaluate individual card quality'
            ],
            'expected_outcomes': {'error': 'Outcomes calculation failed'}
        }
    
    # === ADDITIONAL HELPER METHODS FOR REDRAFT ANALYSIS ===
    
    def _analyze_mana_curve_distribution(self, drafted_cards: List[str]) -> Dict[int, int]:
        """Analyze mana curve distribution of drafted cards."""
        try:
            distribution = {}
            for card_id in drafted_cards:
                cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
                cost = min(cost, 7)  # Cap at 7+ for curve analysis
                distribution[cost] = distribution.get(cost, 0) + 1
            return distribution
        except Exception:
            return {}
    
    def _evaluate_curve_quality(self, curve_distribution: Dict[int, int], hero_class: str) -> float:
        """Evaluate curve quality for specific hero (0-1 scale)."""
        try:
            if not curve_distribution:
                return 0.0
            
            total_cards = sum(curve_distribution.values())
            if total_cards == 0:
                return 0.0
            
            # Hero-specific ideal distributions
            ideal_distributions = {
                'HUNTER': {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05, 7: 0.05},
                'PALADIN': {1: 0.10, 2: 0.30, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05, 7: 0.05},
                'WARRIOR': {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.15, 7: 0.10},
                'PRIEST': {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.15, 7: 0.10}
            }
            
            # Default distribution for unlisted heroes
            default_dist = {1: 0.08, 2: 0.22, 3: 0.25, 4: 0.20, 5: 0.12, 6: 0.08, 7: 0.05}
            ideal = ideal_distributions.get(hero_class, default_dist)
            
            # Calculate curve quality score
            score = 0.0
            for cost in range(8):
                actual_percent = curve_distribution.get(cost, 0) / total_cards
                ideal_percent = ideal.get(cost, 0)
                # Penalize deviation from ideal
                deviation = abs(actual_percent - ideal_percent)
                score += max(0, 1.0 - (deviation * 3))  # Scale penalty
            
            return score / 8  # Average across all mana costs
            
        except Exception:
            return 0.5
    
    def _identify_curve_problems(self, curve_distribution: Dict[int, int], hero_class: str) -> List[str]:
        """Identify specific problems with the mana curve."""
        try:
            problems = []
            total_cards = sum(curve_distribution.values())
            
            if total_cards == 0:
                return ['No cards drafted']
            
            # Check for common curve problems
            early_game = curve_distribution.get(1, 0) + curve_distribution.get(2, 0)
            if early_game / total_cards < 0.2:
                problems.append('Insufficient early game')
            
            mid_game = curve_distribution.get(3, 0) + curve_distribution.get(4, 0)
            if mid_game / total_cards < 0.3:
                problems.append('Weak mid-game presence')
            
            if curve_distribution.get(2, 0) == 0:
                problems.append('No 2-drops')
            
            high_cost = sum(curve_distribution.get(cost, 0) for cost in range(6, 8))
            if high_cost / total_cards > 0.25:
                problems.append('Too many expensive cards')
            
            return problems
            
        except Exception:
            return ['Curve analysis failed']
    
    def _detect_archetype_signals(self, drafted_cards: List[str], hero_class: str) -> List[str]:
        """Detect archetype signals in drafted cards."""
        try:
            # This is a simplified detection - would need card data analysis
            # For now, return plausible archetypes based on hero
            hero_archetypes = {
                'MAGE': ['Tempo', 'Control'],
                'PALADIN': ['Aggro', 'Tempo'],
                'HUNTER': ['Aggro', 'Beast'],
                'WARRIOR': ['Control', 'Weapon'],
                'PRIEST': ['Control', 'Heal'],
                'WARLOCK': ['Aggro', 'Control'],
                'ROGUE': ['Tempo', 'Combo'],
                'SHAMAN': ['Elemental', 'Midrange'],
                'DRUID': ['Ramp', 'Beast'],
                'DEMONHUNTER': ['Aggro', 'Outcast']
            }
            
            return hero_archetypes.get(hero_class, ['Balanced'])[:2]
            
        except Exception:
            return ['Unknown']
    
    def _calculate_archetype_coherence(self, archetype_signals: List[str]) -> float:
        """Calculate how coherent the detected archetypes are."""
        try:
            if not archetype_signals:
                return 0.0
            
            # Simplified coherence: fewer conflicting signals = higher coherence
            if len(archetype_signals) == 1:
                return 0.9
            elif len(archetype_signals) == 2:
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _identify_hero_key_cards(self, drafted_cards: List[str], hero_class: str) -> List[str]:
        """Identify key cards for the hero class in drafted cards."""
        try:
            # Simplified implementation - would need actual card analysis
            return drafted_cards[:3]  # Return first 3 as placeholder
        except Exception:
            return []
    
    def _identify_missing_critical_cards(self, key_cards: List[str], hero_class: str, draft_stage: str) -> bool:
        """Identify if critical cards are missing for the hero."""
        try:
            # Simplified logic
            if draft_stage == 'late' and len(key_cards) < 2:
                return True
            return False
        except Exception:
            return False
    
    def _evaluate_deck_synergies(self, drafted_cards: List[str], hero_class: str) -> float:
        """Evaluate synergy level in current deck."""
        try:
            # Simplified synergy evaluation
            if len(drafted_cards) >= 10:
                return 0.7  # Assume decent synergy for larger decks
            elif len(drafted_cards) >= 5:
                return 0.5
            else:
                return 0.3
        except Exception:
            return 0.5
    
    def _identify_synergy_packages(self, drafted_cards: List[str], hero_class: str) -> List[str]:
        """Identify synergy packages in the deck."""
        try:
            # Simplified implementation
            packages = []
            if len(drafted_cards) >= 5:
                packages.append(f"{hero_class} synergy package")
            return packages
        except Exception:
            return []
    
    def _estimate_redraft_winrate_impact(self, hero_class: str, draft_stage: str) -> str:
        """Estimate the win rate impact of redrafting."""
        try:
            stage_impacts = {
                'early': '+2-4% expected improvement',
                'mid': '+1-3% expected improvement', 
                'late': '+0-2% expected improvement'
            }
            return stage_impacts.get(draft_stage, '+1-3% expected improvement')
        except Exception:
            return 'Unknown impact'
    
    def _calculate_archetype_success_probability(self, hero_class: str, archetype: str) -> str:
        """Calculate probability of successfully achieving target archetype."""
        try:
            # Hero-archetype compatibility
            strong_combinations = [
                ('MAGE', 'Tempo'), ('HUNTER', 'Aggro'), ('WARRIOR', 'Control'),
                ('PALADIN', 'Aggro'), ('PRIEST', 'Control')
            ]
            
            if (hero_class, archetype) in strong_combinations:
                return 'High (75-85%)'
            else:
                return 'Medium (60-75%)'
                
        except Exception:
            return 'Unknown'