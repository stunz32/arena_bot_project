"""
System Integrator - Comprehensive Error Recovery & Performance Monitoring

Handles system integration, error recovery with separate fallback paths,
and performance monitoring for all API calls and components.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import all AI v2 components
from .card_evaluator import CardEvaluationEngine
from .hero_selector import HeroSelectionAdvisor
from .grandmaster_advisor import GrandmasterAdvisor
from .data_models import DeckState, AIDecision, HeroRecommendation

# Import data sources
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_sourcing.hsreplay_scraper import get_hsreplay_scraper


class SystemStatus(Enum):
    """System component status levels."""
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class ComponentHealth:
    """Health status for individual system components."""
    name: str
    status: SystemStatus
    last_check: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    response_time_ms: float = 0.0
    fallback_active: bool = False


class SystemIntegrator:
    """
    Comprehensive system integration with error recovery and monitoring.
    
    Manages separate fallback paths for card data vs hero data failures,
    performance monitoring, and graceful degradation strategies.
    """
    
    def __init__(self):
        """Initialize system integrator with all components."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all AI v2 components
        self.card_evaluator = CardEvaluationEngine()
        self.hero_selector = HeroSelectionAdvisor()
        self.grandmaster_advisor = GrandmasterAdvisor()
        self.hsreplay_scraper = get_hsreplay_scraper()
        
        # System health tracking
        self.component_health = self._initialize_health_tracking()
        self.system_start_time = datetime.now()
        
        # Error recovery settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Performance tracking
        self.performance_metrics = {
            'hero_recommendations': [],
            'card_evaluations': [],
            'api_calls': [],
            'error_counts': {},
            'fallback_activations': {}
        }
        
        self.logger.info("SystemIntegrator initialized with comprehensive error recovery")
    
    def get_hero_recommendation_with_recovery(self, hero_classes: List[str]) -> HeroRecommendation:
        """
        Get hero recommendation with comprehensive error recovery.
        
        Fallback path: HSReplay → Cached → Qualitative-only analysis
        """
        start_time = time.time()
        component = "hero_selector"
        
        try:
            # Attempt primary recommendation
            recommendation = self._execute_with_retry(
                lambda: self.hero_selector.recommend_hero(hero_classes),
                component
            )
            
            # Update health status
            response_time = (time.time() - start_time) * 1000
            self._update_component_health(component, SystemStatus.ONLINE, response_time)
            
            # Track performance
            self.performance_metrics['hero_recommendations'].append({
                'timestamp': datetime.now(),
                'response_time_ms': response_time,
                'hero_classes': hero_classes,
                'fallback_used': False
            })
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Hero recommendation failed: {e}")
            return self._hero_recommendation_fallback(hero_classes, str(e))
    
    def get_card_evaluation_with_recovery(self, card_id: str, deck_state: DeckState) -> Any:
        """
        Get card evaluation with comprehensive error recovery.
        
        Fallback path: HSReplay+Hero → HSReplay-only → Heuristic-only → Basic scoring
        """
        start_time = time.time()
        component = "card_evaluator"
        
        try:
            # Attempt primary evaluation
            evaluation = self._execute_with_retry(
                lambda: self.card_evaluator.evaluate_card(card_id, deck_state),
                component
            )
            
            # Update health status
            response_time = (time.time() - start_time) * 1000
            self._update_component_health(component, SystemStatus.ONLINE, response_time)
            
            # Track performance
            self.performance_metrics['card_evaluations'].append({
                'timestamp': datetime.now(),
                'response_time_ms': response_time,
                'card_id': card_id,
                'hero_class': deck_state.hero_class,
                'fallback_used': False
            })
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Card evaluation failed for {card_id}: {e}")
            return self._card_evaluation_fallback(card_id, deck_state, str(e))
    
    def get_ai_decision_with_recovery(self, deck_state: DeckState, offered_cards: List[str]) -> AIDecision:
        """
        Get AI decision with comprehensive error recovery.
        
        Combines card evaluation and decision logic with multiple fallback levels.
        """
        start_time = time.time()
        component = "grandmaster_advisor"
        
        try:
            # Attempt primary decision
            decision = self._execute_with_retry(
                lambda: self.grandmaster_advisor.get_recommendation(deck_state, offered_cards),
                component
            )
            
            # Update health status
            response_time = (time.time() - start_time) * 1000
            self._update_component_health(component, SystemStatus.ONLINE, response_time)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"AI decision failed: {e}")
            return self._ai_decision_fallback(deck_state, offered_cards, str(e))
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health check.
        
        Tests all components and data sources individually.
        """
        health_report = {
            'overall_status': SystemStatus.ONLINE,
            'components': {},
            'data_sources': {},
            'performance': self._get_performance_summary(),
            'uptime_hours': (datetime.now() - self.system_start_time).total_seconds() / 3600
        }
        
        # Check each component
        for component_name in self.component_health:
            try:
                health = self._check_component_health(component_name)
                health_report['components'][component_name] = {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'error_count': health.error_count,
                    'response_time_ms': health.response_time_ms,
                    'fallback_active': health.fallback_active
                }
                
                # Update overall status
                if health.status in [SystemStatus.ERROR, SystemStatus.OFFLINE]:
                    health_report['overall_status'] = SystemStatus.DEGRADED
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                health_report['components'][component_name] = {
                    'status': SystemStatus.ERROR.value,
                    'error': str(e)
                }
        
        # Check data sources
        health_report['data_sources'] = self._check_data_sources_health()
        
        return health_report
    
    def _execute_with_retry(self, operation, component_name: str):
        """Execute operation with retry logic and circuit breaker."""
        for attempt in range(self.max_retries):
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open(component_name):
                    raise Exception(f"Circuit breaker open for {component_name}")
                
                return operation()
                
            except Exception as e:
                self._record_error(component_name, str(e))
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise e
    
    def _hero_recommendation_fallback(self, hero_classes: List[str], error: str) -> HeroRecommendation:
        """Hero recommendation fallback with qualitative-only analysis."""
        self.logger.warning(f"Using hero recommendation fallback: {error}")
        self._activate_fallback("hero_selector")
        
        try:
            # Fallback to qualitative analysis only
            fallback_analysis = []
            fallback_winrates = {}
            
            for i, hero_class in enumerate(hero_classes):
                # Use historical averages as fallback
                fallback_winrate = self._get_fallback_hero_winrate(hero_class)
                fallback_winrates[hero_class] = fallback_winrate
                
                analysis = {
                    "class": hero_class,
                    "winrate": fallback_winrate,
                    "profile": self.hero_selector.class_profiles.get(hero_class, {}),
                    "confidence": 0.4,  # Low confidence for fallback
                    "explanation": f"Fallback analysis for {hero_class} (HSReplay unavailable)",
                    "score": fallback_winrate,
                    "meta_position": "Unknown"
                }
                fallback_analysis.append(analysis)
            
            # Simple recommendation logic
            best_index = max(range(len(fallback_analysis)), 
                           key=lambda i: fallback_analysis[i]["winrate"])
            
            return HeroRecommendation(
                recommended_hero_index=best_index,
                hero_classes=hero_classes,
                hero_analysis=fallback_analysis,
                explanation=f"Fallback recommendation (data unavailable): {hero_classes[best_index]}",
                winrates=fallback_winrates,
                confidence_level=0.4
            )
            
        except Exception as e:
            self.logger.error(f"Hero fallback failed: {e}")
            # Ultimate fallback - recommend first hero
            return self._create_emergency_hero_recommendation(hero_classes)
    
    def _card_evaluation_fallback(self, card_id: str, deck_state: DeckState, error: str) -> Any:
        """Card evaluation fallback with heuristic-only scoring."""
        self.logger.warning(f"Using card evaluation fallback for {card_id}: {error}")
        self._activate_fallback("card_evaluator")
        
        try:
            # Import data model here to avoid circular imports
            from .data_models import DimensionalScores
            
            # Simple heuristic-based evaluation
            cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
            rarity = self.card_evaluator.cards_loader.get_card_rarity(card_id)
            
            # Basic fallback scoring
            rarity_bonus = {'COMMON': 0.0, 'RARE': 0.1, 'EPIC': 0.2, 'LEGENDARY': 0.3}.get(rarity, 0.0)
            curve_bonus = 0.2 if 2 <= cost <= 4 else 0.0
            
            fallback_score = rarity_bonus + curve_bonus
            
            return DimensionalScores(
                card_id=card_id,
                base_value=fallback_score,
                tempo_score=fallback_score,
                value_score=fallback_score,
                synergy_score=0.0,  # No synergy analysis in fallback
                curve_score=curve_bonus,
                re_draftability_score=0.5,
                greed_score=0.0,
                confidence=0.3  # Very low confidence
            )
            
        except Exception as e:
            self.logger.error(f"Card evaluation fallback failed: {e}")
            # Emergency fallback
            from .data_models import DimensionalScores
            return DimensionalScores(card_id=card_id, confidence=0.1)
    
    def _ai_decision_fallback(self, deck_state: DeckState, offered_cards: List[str], error: str) -> AIDecision:
        """AI decision fallback with simple recommendation logic."""
        self.logger.warning(f"Using AI decision fallback: {error}")
        self._activate_fallback("grandmaster_advisor")
        
        try:
            # Simple fallback: pick middle card
            recommended_index = 1 if len(offered_cards) >= 2 else 0
            
            # Basic analysis for each card
            fallback_analysis = []
            for i, card_id in enumerate(offered_cards):
                analysis = {
                    "card_id": card_id,
                    "scores": {"fallback": 0.5},
                    "explanation": f"Fallback analysis for {card_id}"
                }
                fallback_analysis.append(analysis)
            
            return AIDecision(
                recommended_pick_index=recommended_index,
                all_offered_cards_analysis=fallback_analysis,
                comparative_explanation="Fallback recommendation (system degraded)",
                deck_analysis={"status": "fallback_mode"},
                card_coordinates=[],
                confidence_level=0.3,
                analysis_time_ms=50.0
            )
            
        except Exception as e:
            self.logger.error(f"AI decision fallback failed: {e}")
            # Emergency fallback
            return AIDecision(
                recommended_pick_index=0,
                all_offered_cards_analysis=[],
                comparative_explanation="Emergency fallback",
                deck_analysis={},
                card_coordinates=[],
                confidence_level=0.1,
                analysis_time_ms=10.0
            )
    
    def _initialize_health_tracking(self) -> Dict[str, ComponentHealth]:
        """Initialize health tracking for all components."""
        components = ['card_evaluator', 'hero_selector', 'grandmaster_advisor', 'hsreplay_scraper']
        
        return {
            component: ComponentHealth(
                name=component,
                status=SystemStatus.ONLINE,
                last_check=datetime.now()
            )
            for component in components
        }
    
    def _check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of individual component."""
        health = self.component_health[component_name]
        
        try:
            start_time = time.time()
            
            # Component-specific health checks
            if component_name == 'hsreplay_scraper':
                status = self.hsreplay_scraper.get_api_status()
                if status.get('session_active', False):
                    health.status = SystemStatus.ONLINE
                else:
                    health.status = SystemStatus.DEGRADED
                    
            elif component_name == 'card_evaluator':
                stats = self.card_evaluator.get_evaluation_statistics()
                if stats.get('hsreplay_cards_available', 0) > 0:
                    health.status = SystemStatus.ONLINE
                else:
                    health.status = SystemStatus.DEGRADED
                    
            elif component_name == 'hero_selector':
                stats = self.hero_selector.get_hero_statistics()
                if stats.get('cached_winrates', 0) > 0:
                    health.status = SystemStatus.ONLINE
                else:
                    health.status = SystemStatus.DEGRADED
            
            health.response_time_ms = (time.time() - start_time) * 1000
            health.last_check = datetime.now()
            
        except Exception as e:
            health.status = SystemStatus.ERROR
            health.last_error = str(e)
            health.error_count += 1
        
        return health
    
    def _check_data_sources_health(self) -> Dict[str, Any]:
        """Check health of external data sources."""
        data_sources = {}
        
        try:
            # HSReplay API status
            hsreplay_status = self.hsreplay_scraper.get_api_status()
            data_sources['hsreplay'] = {
                'status': 'online' if hsreplay_status.get('session_active') else 'degraded',
                'api_calls': hsreplay_status.get('api_calls_made', 0),
                'card_cache_age_hours': hsreplay_status.get('card_cache_age_hours'),
                'hero_cache_age_hours': hsreplay_status.get('hero_cache_age_hours')
            }
            
            # Cards database status
            card_stats = self.card_evaluator.cards_loader.get_id_mapping_statistics()
            data_sources['cards_database'] = {
                'status': card_stats.get('mapping_status', 'unknown'),
                'total_cards': card_stats.get('total_cards_loaded', 0),
                'dbf_mappings': card_stats.get('dbf_id_mappings', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Data source health check failed: {e}")
            data_sources['error'] = str(e)
        
        return data_sources
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        
        # Filter recent metrics
        recent_hero_recs = [m for m in self.performance_metrics['hero_recommendations'] 
                           if m['timestamp'] > last_hour]
        recent_card_evals = [m for m in self.performance_metrics['card_evaluations'] 
                            if m['timestamp'] > last_hour]
        
        return {
            'hero_recommendations_last_hour': len(recent_hero_recs),
            'card_evaluations_last_hour': len(recent_card_evals),
            'avg_hero_response_time_ms': (
                sum(m['response_time_ms'] for m in recent_hero_recs) / len(recent_hero_recs)
                if recent_hero_recs else 0
            ),
            'avg_card_eval_time_ms': (
                sum(m['response_time_ms'] for m in recent_card_evals) / len(recent_card_evals)
                if recent_card_evals else 0
            ),
            'total_errors_last_hour': sum(
                1 for error_list in self.performance_metrics['error_counts'].values()
                for error_time in error_list if error_time > last_hour
            )
        }
    
    def _get_fallback_hero_winrate(self, hero_class: str) -> float:
        """Get fallback hero winrate when data unavailable."""
        fallback_rates = {
            'MAGE': 53.5, 'PALADIN': 52.8, 'ROGUE': 52.3, 'HUNTER': 51.9,
            'WARLOCK': 51.2, 'WARRIOR': 50.8, 'SHAMAN': 50.5, 'DRUID': 49.7,
            'PRIEST': 49.2, 'DEMONHUNTER': 48.9
        }
        return fallback_rates.get(hero_class, 50.0)
    
    def _create_emergency_hero_recommendation(self, hero_classes: List[str]) -> HeroRecommendation:
        """Create emergency hero recommendation when all else fails."""
        return HeroRecommendation(
            recommended_hero_index=0,
            hero_classes=hero_classes,
            hero_analysis=[],
            explanation="Emergency recommendation - system offline",
            winrates={cls: 50.0 for cls in hero_classes},
            confidence_level=0.1
        )
    
    def _record_error(self, component: str, error: str) -> None:
        """Record error for circuit breaker logic."""
        if component not in self.performance_metrics['error_counts']:
            self.performance_metrics['error_counts'][component] = []
        
        self.performance_metrics['error_counts'][component].append(datetime.now())
        
        # Clean old errors (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.performance_metrics['error_counts'][component] = [
            error_time for error_time in self.performance_metrics['error_counts'][component]
            if error_time > cutoff
        ]
    
    def _is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker should be open for component."""
        recent_errors = self.performance_metrics['error_counts'].get(component, [])
        return len(recent_errors) >= self.circuit_breaker_threshold
    
    def _activate_fallback(self, component: str) -> None:
        """Activate fallback mode for component."""
        if component not in self.performance_metrics['fallback_activations']:
            self.performance_metrics['fallback_activations'][component] = []
        
        self.performance_metrics['fallback_activations'][component].append(datetime.now())
        
        if component in self.component_health:
            self.component_health[component].fallback_active = True
            self.component_health[component].status = SystemStatus.DEGRADED
    
    def _update_component_health(self, component: str, status: SystemStatus, response_time: float) -> None:
        """Update component health status."""
        if component in self.component_health:
            health = self.component_health[component]
            health.status = status
            health.response_time_ms = response_time
            health.last_check = datetime.now()
            health.fallback_active = False  # Reset fallback when successful