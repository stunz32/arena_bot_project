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

# Import standardized logging
from .logging_utils import (
    get_structured_logger, 
    LogCategory, 
    log_component_initialization,
    log_performance_warning,
    log_circuit_breaker_event
)

# Import all AI v2 components with defensive handling
try:
    from .card_evaluator import CardEvaluationEngine
except ImportError as e:
    logging.error(f"Failed to import CardEvaluationEngine: {e}")
    CardEvaluationEngine = None

try:
    from .hero_selector import HeroSelectionAdvisor
except ImportError as e:
    logging.error(f"Failed to import HeroSelectionAdvisor: {e}")
    HeroSelectionAdvisor = None

try:
    from .grandmaster_advisor import GrandmasterAdvisor
except ImportError as e:
    logging.error(f"Failed to import GrandmasterAdvisor: {e}")
    GrandmasterAdvisor = None

from .data_models import DeckState, AIDecision, HeroRecommendation

# Defensive import for validation utils to handle circular dependencies
try:
    from .validation_utils import (
        validate_system_boundary_input, 
        CardDataValidator, 
        DeckStateValidator,
        AIDecisionValidator,
        ValidationError
    )
except ImportError as e:
    logging.warning(f"Validation utils import failed, using fallback: {e}")
    # Create minimal fallback classes
    class ValidationError(Exception):
        pass
    
    class CardDataValidator:
        @staticmethod
        def validate_card_id(card_id, field_name):
            return card_id or "unknown_card"
    
    class DeckStateValidator:
        @staticmethod
        def validate_deck_state(deck_state, field_name):
            if not deck_state or not hasattr(deck_state, 'hero_class'):
                from .data_models import DeckState
                return DeckState(hero_class="NEUTRAL", archetype="Unknown")
            return deck_state
    
    class AIDecisionValidator:
        pass
    
    def validate_system_boundary_input(data, operation_type):
        return data

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
        # Use structured logging for better debugging and monitoring
        self.logger = get_structured_logger("system_integrator")
        init_start = self.logger.log_operation_start("system_integrator_init")
        
        # Initialize all AI v2 components with defensive programming
        # Check if classes are available before instantiating
        if CardEvaluationEngine is not None:
            try:
                self.card_evaluator = CardEvaluationEngine()
            except Exception as e:
                self.logger.error(
                    "Failed to initialize CardEvaluationEngine",
                    category=LogCategory.INITIALIZATION,
                    exception=e,
                    component="CardEvaluationEngine"
                )
                self.card_evaluator = None
        else:
            self.logger.error("CardEvaluationEngine class not available")
            self.card_evaluator = None
        
        if HeroSelectionAdvisor is not None:
            try:
                self.hero_selector = HeroSelectionAdvisor()
            except Exception as e:
                self.logger.error(
                    "Failed to initialize HeroSelectionAdvisor",
                    category=LogCategory.INITIALIZATION,
                    exception=e,
                    component="HeroSelectionAdvisor"
                )
                self.hero_selector = None
        else:
            self.logger.error("HeroSelectionAdvisor class not available")
            self.hero_selector = None
            
        if GrandmasterAdvisor is not None:
            try:
                self.grandmaster_advisor = GrandmasterAdvisor()
            except Exception as e:
                self.logger.error(
                    "Failed to initialize GrandmasterAdvisor",
                    category=LogCategory.INITIALIZATION,
                    exception=e,
                    component="GrandmasterAdvisor"
                )
                self.grandmaster_advisor = None
        else:
            self.logger.error("GrandmasterAdvisor class not available")
            self.grandmaster_advisor = None
            
        try:
            self.hsreplay_scraper = get_hsreplay_scraper()
        except Exception as e:
            self.logger.error(
                "Failed to initialize HSReplay scraper",
                category=LogCategory.INITIALIZATION,
                exception=e,
                component="HSReplayScaper"
            )
            self.hsreplay_scraper = None
        
        # System health tracking
        self.component_health = self._initialize_health_tracking()
        self.system_start_time = datetime.now()
        
        # Report initialization status for better error propagation
        failed_components = []
        available_components = []
        
        components = [('CardEvaluationEngine', self.card_evaluator),
                     ('HeroSelectionAdvisor', self.hero_selector), 
                     ('GrandmasterAdvisor', self.grandmaster_advisor),
                     ('HSReplayScaper', self.hsreplay_scraper)]
        
        for name, component in components:
            if component is None:
                failed_components.append(name)
            else:
                available_components.append(name)
        
        if failed_components:
            self.logger.warning(
                f"SystemIntegrator initialized with {len(failed_components)} failed components",
                category=LogCategory.INITIALIZATION,
                failed_components=failed_components,
                failed_count=len(failed_components)
            )
            self.logger.info(
                "Available components initialized successfully",
                category=LogCategory.INITIALIZATION,
                available_components=available_components,
                available_count=len(available_components)
            )
            
            # If critical components failed, this is a serious issue
            critical_components = ['CardEvaluationEngine', 'GrandmasterAdvisor']
            failed_critical = [c for c in failed_components if c in critical_components]
            if failed_critical:
                error_msg = f"Critical AI v2 components failed to initialize: {', '.join(failed_critical)}"
                self.logger.critical(
                    "Critical AI v2 components failed to initialize",
                    category=LogCategory.INITIALIZATION,
                    failed_critical_components=failed_critical,
                    system_degraded=True
                )
                # Don't raise an exception here - let the system degrade gracefully
                # But make sure the error is visible for debugging
        
        # Error recovery settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Enhanced circuit breaker state management
        self.circuit_breaker_states = {}  # Component -> CircuitBreakerState
        self.failure_patterns = {}  # Component -> FailurePattern analysis
        self.recovery_attempts = {}  # Component -> recovery tracking
        
        # Enhanced fallback data caching
        self.known_good_data_cache = {
            'hero_winrates': {},
            'card_evaluations': {},
            'last_good_response': {},
            'cache_timestamps': {}
        }
        self.cache_ttl_hours = 24  # Cache good data for 24 hours
        
        # Performance tracking
        self.performance_metrics = {
            'hero_recommendations': [],
            'card_evaluations': [],
            'api_calls': [],
            'error_counts': {},
            'fallback_activations': {}
        }
        
        # Log initialization completion
        self.logger.log_operation_end(init_start, success=True, 
                                     result_summary=f"{len(available_components)} components available")
        
        log_component_initialization(
            self.logger, "SystemIntegrator", True,
            available_components=available_components,
            failed_components=failed_components
        )
    
    def get_hero_recommendation_with_recovery(self, hero_classes: List[str]) -> HeroRecommendation:
        """
        Get hero recommendation with comprehensive error recovery and input validation.
        
        Fallback path: HSReplay → Cached → Qualitative-only analysis
        """
        start_time = time.time()
        component = "hero_selector"
        
        try:
            # Input validation at system boundary
            try:
                validated_input = validate_system_boundary_input(
                    {'hero_classes': hero_classes}, 
                    'hero_recommendation'
                )
                hero_classes = validated_input['hero_classes']
                self.logger.log_validation_result(
                    "hero_classes", True, hero_count=len(hero_classes)
                )
            except ValidationError as e:
                self.logger.log_validation_result(
                    "hero_classes", False, str(e), hero_classes
                )
                return self._hero_recommendation_fallback(hero_classes or [], str(e))
            
            # Defensive programming: check if component is available
            if not self.hero_selector:
                self.logger.error("❌ HeroSelectionAdvisor not available, using fallback immediately")
                return self._hero_recommendation_fallback(hero_classes, "Component not initialized")
            
            # Attempt primary recommendation
            recommendation = self._execute_with_retry(
                lambda: self.hero_selector.recommend_hero(hero_classes),
                component
            )
            
            # Update health status
            response_time = (time.time() - start_time) * 1000
            self._update_component_health(component, SystemStatus.ONLINE, response_time)
            
            # Cache successful recommendation for fallback
            if hasattr(recommendation, 'winrates') and recommendation.winrates:
                for hero_class, winrate in recommendation.winrates.items():
                    self._cache_successful_response('hero_winrates', hero_class, winrate)
            
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
        Get card evaluation with comprehensive error recovery and input validation.
        
        Fallback path: HSReplay+Hero → HSReplay-only → Heuristic-only → Basic scoring
        """
        start_time = time.time()
        component = "card_evaluator"
        
        try:
            # Input validation at system boundary
            try:
                validated_input = validate_system_boundary_input(
                    {'card_id': card_id, 'deck_state': deck_state}, 
                    'card_evaluation'
                )
                card_id = validated_input['card_id']
                deck_state = validated_input['deck_state']
                self.logger.log_validation_result(
                    "card_evaluation_input", True, 
                    card_id=card_id, hero_class=deck_state.hero_class
                )
            except ValidationError as e:
                self.logger.log_validation_result(
                    "card_evaluation_input", False, str(e), {"card_id": card_id, "deck_state": deck_state}
                )
                # Create safe fallback inputs
                card_id = card_id or "unknown_card"
                if not deck_state:
                    from .data_models import DeckState
                    deck_state = DeckState(hero_class="NEUTRAL", archetype="Unknown")
                return self._card_evaluation_fallback(card_id, deck_state, str(e))
            
            # Defensive programming: check if component is available
            if not self.card_evaluator:
                self.logger.error("❌ CardEvaluationEngine not available, using fallback immediately")
                return self._card_evaluation_fallback(card_id, deck_state, "Component not initialized")
            
            # Attempt primary evaluation
            evaluation = self._execute_with_retry(
                lambda: self.card_evaluator.evaluate_card(card_id, deck_state),
                component
            )
            
            # Update health status
            response_time = (time.time() - start_time) * 1000
            self._update_component_health(component, SystemStatus.ONLINE, response_time)
            
            # Cache successful evaluation for fallback
            cache_key = f"{card_id}_{deck_state.hero_class}_{deck_state.pick_number}"
            self._cache_successful_response('card_evaluations', cache_key, evaluation)
            
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
        Get AI decision with comprehensive error recovery and input validation.
        
        Combines card evaluation and decision logic with multiple fallback levels.
        """
        start_time = time.time()
        component = "grandmaster_advisor"
        
        try:
            # Input validation at system boundary
            try:
                validated_input = validate_system_boundary_input(
                    {'deck_state': deck_state, 'offered_cards': offered_cards}, 
                    'ai_decision'
                )
                deck_state = validated_input['deck_state']
                offered_cards = validated_input['offered_cards']
                self.logger.log_validation_result(
                    "ai_decision_input", True, 
                    card_count=len(offered_cards), hero_class=deck_state.hero_class
                )
            except ValidationError as e:
                self.logger.log_validation_result(
                    "ai_decision_input", False, str(e), 
                    {"deck_state": deck_state, "offered_cards": offered_cards}
                )
                # Create safe fallback inputs
                if not deck_state:
                    from .data_models import DeckState
                    deck_state = DeckState(hero_class="NEUTRAL", archetype="Unknown")
                offered_cards = offered_cards or ["unknown_card"]
                return self._ai_decision_fallback(deck_state, offered_cards, str(e))
            
            # Defensive programming: check if component is available
            if not self.grandmaster_advisor:
                self.logger.error("❌ GrandmasterAdvisor not available, using fallback immediately")
                return self._ai_decision_fallback(deck_state, offered_cards, "Component not initialized")
            
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
            'overall_status': SystemStatus.ONLINE.value,
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
                    health_report['overall_status'] = SystemStatus.DEGRADED.value
                    
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
                
                # Classify error type to determine if retry is appropriate
                error_classification = self._classify_error(e, component_name)
                
                if error_classification == 'FATAL':
                    # Fatal errors should not be retried - fail fast
                    self.logger.error(f"❌ Fatal error in {component_name}: {e}")
                    raise e
                elif error_classification == 'CIRCUIT_BREAK':
                    # Circuit breaker errors should open the circuit immediately
                    self.logger.error(f"❌ Circuit breaker triggered for {component_name}: {e}")
                    self._open_circuit_breaker(component_name)
                    raise e
                elif error_classification == 'RECOVERABLE':
                    # Only retry recoverable errors
                    if attempt < self.max_retries - 1:
                        retry_delay = self.retry_delay * (attempt + 1)
                        self.logger.warning(f"⚠️ Recoverable error in {component_name} (attempt {attempt + 1}/{self.max_retries}): {e}, retrying in {retry_delay}s")
                        time.sleep(retry_delay)  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"❌ Max retries exceeded for {component_name}: {e}")
                        raise e
                else:
                    # Unknown error type - default to retry behavior but log warning
                    self.logger.warning(f"⚠️ Unknown error type in {component_name}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise e
    
    def _hero_recommendation_fallback(self, hero_classes: List[str], error: str) -> HeroRecommendation:
        """Hero recommendation fallback with qualitative-only analysis."""
        self.logger.warning(f"Using hero recommendation fallback: {error}")
        self._activate_fallback("hero_selector")
        
        try:
            # Defensive programming: handle None/empty inputs
            if not hero_classes:
                self.logger.error("❌ hero_classes is None or empty in fallback")
                hero_classes = ['MAGE', 'HUNTER', 'PALADIN']  # Default fallback heroes
            
            # Fallback to qualitative analysis only
            fallback_analysis = []
            fallback_winrates = {}
            
            for i, hero_class in enumerate(hero_classes):
                if not hero_class:
                    hero_class = f'UNKNOWN_HERO_{i}'
                    self.logger.warning(f"⚠️ Empty hero_class at index {i}, using fallback")
                
                # Use historical averages as fallback
                fallback_winrate = self._get_fallback_hero_winrate(hero_class)
                fallback_winrates[hero_class] = fallback_winrate
                
                # Safe profile access with None check
                profile = {}
                if hasattr(self, 'hero_selector') and self.hero_selector and hasattr(self.hero_selector, 'class_profiles'):
                    profile = self.hero_selector.class_profiles.get(hero_class, {})
                
                analysis = {
                    "class": hero_class,
                    "winrate": fallback_winrate,
                    "profile": profile,
                    "confidence": 0.4,  # Low confidence for fallback
                    "explanation": f"Fallback analysis for {hero_class} (HSReplay unavailable)",
                    "score": fallback_winrate,
                    "meta_position": "Unknown"
                }
                fallback_analysis.append(analysis)
            
            # Simple recommendation logic with safety checks
            if not fallback_analysis:
                self.logger.error("❌ No fallback analysis available")
                return self._create_emergency_hero_recommendation(hero_classes)
                
            best_index = max(range(len(fallback_analysis)), 
                           key=lambda i: fallback_analysis[i]["winrate"])
            
            # Validate index is within bounds
            if best_index >= len(hero_classes):
                self.logger.warning(f"⚠️ Invalid best_index {best_index}, using 0")
                best_index = 0
            
            return HeroRecommendation(
                recommended_hero_index=best_index,
                hero_classes=hero_classes,
                hero_analysis=fallback_analysis,
                explanation=f"Fallback recommendation (data unavailable): {hero_classes[best_index] if best_index < len(hero_classes) else 'Unknown'}",
                winrates=fallback_winrates,
                confidence_level=0.4
            )
            
        except Exception as e:
            self.logger.error(f"Hero fallback failed: {e}")
            # Ultimate fallback - recommend first hero
            return self._create_emergency_hero_recommendation(hero_classes)
    
    def _card_evaluation_fallback(self, card_id: str, deck_state: DeckState, error: str) -> Any:
        """Enhanced card evaluation fallback with intelligent heuristics and caching."""
        self.logger.warning(f"Using card evaluation fallback for {card_id}: {error}")
        self._activate_fallback("card_evaluator")
        
        try:
            # Try cached known-good evaluation first
            cached_eval = self._get_cached_card_evaluation(card_id, deck_state)
            if cached_eval is not None:
                self.logger.info(f"Using cached evaluation for {card_id}")
                return cached_eval
            
            # Import data model here to avoid circular imports
            from .data_models import DimensionalScores
            
            # Enhanced heuristic evaluation with defensive programming
            cost = 0
            rarity = 'COMMON'
            card_type = 'MINION'
            
            if self.card_evaluator and self.card_evaluator.cards_loader:
                try:
                    cost = self.card_evaluator.cards_loader.get_card_cost(card_id) or 0
                    rarity = self.card_evaluator.cards_loader.get_card_rarity(card_id) or 'COMMON'
                    card_type = self.card_evaluator.cards_loader.get_card_type(card_id) or 'MINION'
                except Exception as e:
                    self.logger.warning(f"Failed to get card data for {card_id}: {e}")
            
            # Enhanced scoring algorithm
            base_scores = self._calculate_enhanced_fallback_scores(cost, rarity, card_type, deck_state)
            
            return DimensionalScores(
                card_id=card_id,
                base_value=base_scores['base_value'],
                tempo_score=base_scores['tempo_score'],
                value_score=base_scores['value_score'],
                synergy_score=base_scores['synergy_score'],
                curve_score=base_scores['curve_score'],
                re_draftability_score=base_scores['re_draftability_score'],
                greed_score=base_scores['greed_score'],
                confidence=0.4  # Improved confidence with better heuristics
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
            # Defensive programming: handle None/empty inputs
            if not offered_cards:
                self.logger.error("❌ offered_cards is None or empty in fallback")
                offered_cards = ['unknown_card']
            
            if not deck_state:
                self.logger.error("❌ deck_state is None in fallback")
            
            # Simple fallback: pick middle card
            recommended_index = 1 if len(offered_cards) >= 2 else 0
            
            # Basic analysis for each card
            fallback_analysis = []
            for i, card_id in enumerate(offered_cards):
                if not card_id:
                    card_id = f"unknown_card_{i}"
                    self.logger.warning(f"⚠️ Empty card_id at index {i}, using fallback")
                
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
        """Get intelligent fallback hero winrate using cached data and heuristics."""
        # Try cached known-good data first
        cached_rate = self._get_cached_hero_winrate(hero_class)
        if cached_rate is not None:
            self.logger.info(f"Using cached winrate for {hero_class}: {cached_rate}%")
            return cached_rate
        
        # Enhanced heuristics based on class characteristics
        current_meta_rates = {
            # Tier 1 classes (typically strong)
            'MAGE': 53.8, 'PALADIN': 53.2, 'ROGUE': 52.9,
            # Tier 2 classes (solid picks)
            'HUNTER': 52.1, 'WARLOCK': 51.8, 'WARRIOR': 51.4,
            # Tier 3 classes (situational)
            'SHAMAN': 50.9, 'DRUID': 50.3, 'PRIEST': 49.8,
            # Tier 4 classes (challenging)
            'DEMONHUNTER': 49.4
        }
        
        # Apply meta adjustment based on class complexity
        base_rate = current_meta_rates.get(hero_class, 50.0)
        
        # Adjust based on draft format complexity (simpler classes get bonus in fallback)
        complexity_adjustments = {
            'MAGE': 0.5,    # Straightforward, good fallback
            'PALADIN': 0.3, # Value-based, reliable
            'HUNTER': 0.4,  # Tempo-focused, consistent
            'WARLOCK': -0.2, # Complex resource management
            'PRIEST': -0.3,  # Reactive playstyle, harder
            'ROGUE': 0.1,   # Tempo but complex combos
            'WARRIOR': 0.0, # Balanced complexity
            'SHAMAN': -0.1, # RNG dependent
            'DRUID': -0.2,  # Ramp complexity
            'DEMONHUNTER': -0.4  # Newest class, less stable
        }
        
        adjustment = complexity_adjustments.get(hero_class, 0.0)
        adjusted_rate = base_rate + adjustment
        
        self.logger.info(f"Using heuristic winrate for {hero_class}: {adjusted_rate}% (base: {base_rate}%, adj: {adjustment})")
        return max(45.0, min(60.0, adjusted_rate))  # Clamp to reasonable bounds
    
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
        """Enhanced error recording with circuit breaker state management."""
        current_time = datetime.now()
        
        # Record error timestamp
        if component not in self.performance_metrics['error_counts']:
            self.performance_metrics['error_counts'][component] = []
        
        self.performance_metrics['error_counts'][component].append(current_time)
        
        # Update circuit breaker state
        if component not in self.circuit_breaker_states:
            self.circuit_breaker_states[component] = {
                'state': 'CLOSED',
                'failure_count': 0,
                'last_failure_time': None,
                'last_success_time': datetime.now(),
                'consecutive_failures': 0,
                'failure_rate': 0.0,
                'recovery_probe_time': None
            }
        
        breaker_state = self.circuit_breaker_states[component]
        breaker_state['failure_count'] += 1
        breaker_state['last_failure_time'] = current_time
        
        # Update consecutive failures (reset if we had recent success)
        if (breaker_state.get('last_success_time') and 
            current_time - breaker_state['last_success_time'] < timedelta(minutes=5)):
            breaker_state['consecutive_failures'] = 1  # Reset count
        else:
            breaker_state['consecutive_failures'] += 1
        
        # If in HALF_OPEN state and we get an error, go back to OPEN
        if breaker_state['state'] == 'HALF_OPEN':
            breaker_state['state'] = 'OPEN'
            breaker_state['recovery_probe_time'] = current_time + timedelta(seconds=self.circuit_breaker_timeout)
            log_circuit_breaker_event(
                self.logger, component, "PROBE_FAILED", "OPEN",
                message="recovery probe failed"
            )
        
        # Clean old errors (older than 1 hour)
        cutoff = current_time - timedelta(hours=1)
        self.performance_metrics['error_counts'][component] = [
            error_time for error_time in self.performance_metrics['error_counts'][component]
            if error_time > cutoff
        ]
    
    def _is_circuit_breaker_open(self, component: str) -> bool:
        """Enhanced circuit breaker with failure pattern analysis and recovery states."""
        # Initialize circuit breaker state if not exists
        if component not in self.circuit_breaker_states:
            self.circuit_breaker_states[component] = {
                'state': 'CLOSED',  # CLOSED, HALF_OPEN, OPEN
                'failure_count': 0,
                'last_failure_time': None,
                'last_success_time': datetime.now(),
                'consecutive_failures': 0,
                'failure_rate': 0.0,
                'recovery_probe_time': None
            }
        
        breaker_state = self.circuit_breaker_states[component]
        current_time = datetime.now()
        
        # Analyze recent failure patterns
        failure_analysis = self._analyze_failure_patterns(component)
        
        # State machine logic
        if breaker_state['state'] == 'CLOSED':
            # Normal operation - check if we should open
            if self._should_open_circuit(component, failure_analysis):
                breaker_state['state'] = 'OPEN'
                breaker_state['recovery_probe_time'] = current_time + timedelta(seconds=self.circuit_breaker_timeout)
                log_circuit_breaker_event(
                    self.logger, component, "OPENED", "OPEN",
                    reason=failure_analysis['reason'],
                    error_rate=failure_analysis.get('error_rate', 0),
                    consecutive_failures=failure_analysis.get('consecutive_failures', 0)
                )
                return True
            return False
            
        elif breaker_state['state'] == 'OPEN':
            # Circuit is open - check if we should try recovery
            if current_time >= breaker_state['recovery_probe_time']:
                breaker_state['state'] = 'HALF_OPEN'
                log_circuit_breaker_event(
                    self.logger, component, "HALF_OPEN", "HALF_OPEN",
                    message="attempting recovery probe"
                )
                return False  # Allow one test request
            return True  # Still open
            
        elif breaker_state['state'] == 'HALF_OPEN':
            # Testing recovery - one request allowed to test health
            return False
        
        return False
    
    def _should_open_circuit(self, component: str, failure_analysis: Dict[str, Any]) -> bool:
        """Determine if circuit should open based on failure pattern analysis."""
        # Multiple criteria for opening circuit
        criteria = {
            'high_error_rate': failure_analysis['error_rate'] > 0.5,
            'consecutive_failures': failure_analysis['consecutive_failures'] >= 3,
            'critical_error_pattern': failure_analysis['has_critical_errors'],
            'rapid_failures': failure_analysis['failures_per_minute'] > 10,
            'timeout_cascade': failure_analysis['timeout_cascade_detected']
        }
        
        # Log decision reasoning
        active_criteria = [k for k, v in criteria.items() if v]
        if active_criteria:
            failure_analysis['reason'] = f"Failure criteria met: {', '.join(active_criteria)}"
            return len(active_criteria) >= 2  # Require at least 2 criteria
        
        return False
    
    def _analyze_failure_patterns(self, component: str) -> Dict[str, Any]:
        """Analyze failure patterns to make intelligent circuit breaker decisions."""
        recent_errors = self.performance_metrics['error_counts'].get(component, [])
        current_time = datetime.now()
        
        # Time windows for analysis
        last_minute = current_time - timedelta(minutes=1)
        last_5_minutes = current_time - timedelta(minutes=5)
        last_hour = current_time - timedelta(hours=1)
        
        # Filter errors by time windows
        errors_last_minute = [e for e in recent_errors if e > last_minute]
        errors_last_5min = [e for e in recent_errors if e > last_5_minutes]
        errors_last_hour = [e for e in recent_errors if e > last_hour]
        
        # Calculate rates and patterns
        total_requests_estimate = max(1, len(errors_last_hour) * 10)  # Rough estimate
        error_rate = len(errors_last_hour) / total_requests_estimate
        failures_per_minute = len(errors_last_minute)
        
        # Check for consecutive failures (no successes between errors)
        breaker_state = self.circuit_breaker_states.get(component, {})
        last_success = breaker_state.get('last_success_time')
        consecutive_failures = 0
        
        if last_success:
            consecutive_failures = len([e for e in recent_errors if e > last_success])
        else:
            consecutive_failures = len(errors_last_5min)
        
        # Detect specific failure patterns
        has_critical_errors = self._has_critical_error_pattern(component)
        timeout_cascade = len(errors_last_minute) > 3 and len(errors_last_5min) > errors_last_minute * 3
        
        return {
            'error_rate': error_rate,
            'failures_per_minute': failures_per_minute,
            'consecutive_failures': consecutive_failures,
            'has_critical_errors': has_critical_errors,
            'timeout_cascade_detected': timeout_cascade,
            'total_recent_errors': len(errors_last_hour),
            'reason': None  # Will be set by _should_open_circuit if needed
        }
    
    def _has_critical_error_pattern(self, component: str) -> bool:
        """Detect critical error patterns that should immediately trigger circuit breaker."""
        # Check recent error types - this would need to be enhanced with actual error tracking
        # For now, we'll use a simple heuristic based on error frequency
        recent_errors = self.performance_metrics['error_counts'].get(component, [])
        last_5_minutes = datetime.now() - timedelta(minutes=5)
        
        recent_error_count = len([e for e in recent_errors if e > last_5_minutes])
        
        # Critical patterns:
        # 1. Rapid error accumulation (>5 errors in 5 minutes)
        # 2. Component health shows ERROR state
        critical_error_rate = recent_error_count > 5
        component_in_error_state = (
            component in self.component_health and 
            self.component_health[component].status == SystemStatus.ERROR
        )
        
        return critical_error_rate or component_in_error_state
    
    def _activate_fallback(self, component: str) -> None:
        """Activate fallback mode for component."""
        if component not in self.performance_metrics['fallback_activations']:
            self.performance_metrics['fallback_activations'][component] = []
        
        self.performance_metrics['fallback_activations'][component].append(datetime.now())
        
        if component in self.component_health:
            self.component_health[component].fallback_active = True
            self.component_health[component].status = SystemStatus.DEGRADED
    
    def _update_component_health(self, component: str, status: SystemStatus, response_time: float) -> None:
        """Update component health status and circuit breaker recovery."""
        if component in self.component_health:
            health = self.component_health[component]
            health.status = status
            health.response_time_ms = response_time
            health.last_check = datetime.now()
            health.fallback_active = False  # Reset fallback when successful
            
            # Update circuit breaker state on successful operation
            if status == SystemStatus.ONLINE and component in self.circuit_breaker_states:
                breaker_state = self.circuit_breaker_states[component]
                breaker_state['last_success_time'] = datetime.now()
                
                # If we were in HALF_OPEN and got success, close the circuit
                if breaker_state['state'] == 'HALF_OPEN':
                    breaker_state['state'] = 'CLOSED'
                    breaker_state['consecutive_failures'] = 0
                    log_circuit_breaker_event(
                        self.logger, component, "RECOVERY_SUCCESS", "CLOSED",
                        message="recovery successful"
                    )
                
                # Reset consecutive failures counter on success
                elif breaker_state['state'] == 'CLOSED':
                    breaker_state['consecutive_failures'] = 0
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status for all components."""
        status = {}
        
        for component in ['card_evaluator', 'hero_selector', 'grandmaster_advisor', 'hsreplay_scraper']:
            breaker_state = self.circuit_breaker_states.get(component, {})
            failure_analysis = self._analyze_failure_patterns(component)
            
            status[component] = {
                'state': breaker_state.get('state', 'CLOSED'),
                'failure_count': breaker_state.get('failure_count', 0),
                'consecutive_failures': breaker_state.get('consecutive_failures', 0),
                'error_rate': failure_analysis.get('error_rate', 0.0),
                'last_failure_time': breaker_state.get('last_failure_time'),
                'last_success_time': breaker_state.get('last_success_time'),
                'recovery_probe_time': breaker_state.get('recovery_probe_time'),
                'health_status': self.component_health.get(component, {}).get('status', 'unknown')
            }
        
        return status
    
    def _classify_error(self, error: Exception, component_name: str) -> str:
        """
        Classify error to determine appropriate handling strategy.
        
        Returns:
            'FATAL': Should not be retried, fail immediately
            'RECOVERABLE': Can be retried with exponential backoff  
            'CIRCUIT_BREAK': Should trigger circuit breaker
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Fatal errors - should not be retried
        fatal_indicators = [
            'importerror', 'modulenotfounderror', 'attributeerror',
            'no module named', 'cannot import', 'missing dependency',
            'configuration error', 'invalid configuration',
            'file not found', 'directory not found', 'path does not exist',
            'permission denied', 'access denied',
            'invalid credentials', 'authentication failed permanently',
            'schema validation failed', 'invalid data structure'
        ]
        
        for indicator in fatal_indicators:
            if indicator in error_str or indicator in error_type.lower():
                return 'FATAL'
        
        # Circuit breaker triggers - persistent failures requiring circuit break
        circuit_break_indicators = [
            'connection refused', 'service unavailable permanently',
            'api quota exceeded', 'rate limit exceeded permanently',
            'server error 5', 'internal server error',
            'bad gateway', 'service unavailable',
            'upstream connect error', 'upstream request timeout'
        ]
        
        for indicator in circuit_break_indicators:
            if indicator in error_str:
                return 'CIRCUIT_BREAK'
        
        # Recoverable errors - can be retried
        recoverable_indicators = [
            'timeout', 'connection timeout', 'read timeout',
            'temporary failure', 'try again', 'retry',
            'network error', 'connection error',
            'http error 429', 'too many requests',  # Rate limiting
            'http error 502', 'http error 503', 'http error 504',  # Temporary server issues
            'dns resolution failed', 'name resolution failed'
        ]
        
        for indicator in recoverable_indicators:
            if indicator in error_str:
                return 'RECOVERABLE'
        
        # Component-specific error classification
        if component_name == 'hsreplay_scraper':
            if 'captcha' in error_str or 'blocked' in error_str:
                return 'CIRCUIT_BREAK'
            elif 'parsing' in error_str or 'data format' in error_str:
                return 'RECOVERABLE'
        
        elif component_name == 'card_evaluator':
            if 'card not found' in error_str:
                return 'RECOVERABLE'  # Might be temporary data issue
            elif 'evaluation method' in error_str:
                return 'FATAL'  # Code structure issue
        
        elif component_name == 'system_integrator':
            if 'initialization' in error_str:
                return 'FATAL'  # Startup issues should fail fast
        
        # Default to recoverable for unknown errors (conservative approach)
        return 'RECOVERABLE'
    
    def _open_circuit_breaker(self, component: str) -> None:
        """Force circuit breaker open for component."""
        # Add enough recent errors to trigger circuit breaker
        if component not in self.performance_metrics['error_counts']:
            self.performance_metrics['error_counts'][component] = []
        
        # Add circuit breaker threshold number of recent errors
        now = datetime.now()
        for i in range(self.circuit_breaker_threshold):
            self.performance_metrics['error_counts'][component].append(now)
        
        # Update component health to error state  
        if component in self.component_health:
            self.component_health[component].status = SystemStatus.ERROR
            self.component_health[component].last_check = now
        
        self.logger.error(f"🚨 Circuit breaker opened for {component}")
        self._activate_fallback(component)
    
    def _cache_successful_response(self, component: str, key: str, data: Any) -> None:
        """Cache successful response for fallback use."""
        try:
            if component not in self.known_good_data_cache:
                self.known_good_data_cache[component] = {}
            
            self.known_good_data_cache[component][key] = {
                'data': data,
                'timestamp': datetime.now(),
                'quality_score': self._assess_data_quality(data)
            }
            
            # Cleanup old cache entries
            self._cleanup_expired_cache(component)
            
        except Exception as e:
            self.logger.warning(f"Failed to cache response: {e}")
    
    def _get_cached_hero_winrate(self, hero_class: str) -> Optional[float]:
        """Get cached hero winrate if available and fresh."""
        try:
            cache = self.known_good_data_cache.get('hero_winrates', {})
            entry = cache.get(hero_class)
            
            if not entry:
                return None
            
            # Check if cache is still fresh
            age_hours = (datetime.now() - entry['timestamp']).total_seconds() / 3600
            if age_hours > self.cache_ttl_hours:
                return None
            
            # Return cached winrate if quality is good
            if entry.get('quality_score', 0) > 0.7:
                return entry['data']
                
        except Exception as e:
            self.logger.warning(f"Error accessing cached winrate: {e}")
        
        return None
    
    def _get_cached_card_evaluation(self, card_id: str, deck_state: DeckState) -> Optional[Any]:
        """Get cached card evaluation if available and fresh."""
        try:
            cache_key = f"{card_id}_{deck_state.hero_class}_{deck_state.pick_number}"
            cache = self.known_good_data_cache.get('card_evaluations', {})
            entry = cache.get(cache_key)
            
            if not entry:
                return None
            
            # Check if cache is still fresh
            age_hours = (datetime.now() - entry['timestamp']).total_seconds() / 3600
            if age_hours > self.cache_ttl_hours:
                return None
            
            # Return cached evaluation if quality is good
            if entry.get('quality_score', 0) > 0.6:
                return entry['data']
                
        except Exception as e:
            self.logger.warning(f"Error accessing cached evaluation: {e}")
        
        return None
    
    def _calculate_enhanced_fallback_scores(self, cost: int, rarity: str, card_type: str, deck_state: DeckState) -> Dict[str, float]:
        """Calculate enhanced fallback scores using intelligent heuristics."""
        scores = {}
        
        # Base value calculation with rarity and cost considerations
        rarity_values = {'COMMON': 0.45, 'RARE': 0.55, 'EPIC': 0.65, 'LEGENDARY': 0.75}
        base_rarity_score = rarity_values.get(rarity, 0.45)
        
        # Cost curve optimization (Arena favors 2-4 cost)
        if 2 <= cost <= 4:
            curve_multiplier = 1.2
        elif cost == 1 or cost == 5:
            curve_multiplier = 1.0
        elif cost == 6 or cost == 7:
            curve_multiplier = 0.8
        else:
            curve_multiplier = 0.6
        
        scores['base_value'] = base_rarity_score * curve_multiplier
        
        # Tempo score based on cost and type
        if card_type == 'MINION':
            scores['tempo_score'] = min(0.8, (7 - cost) / 7.0 + 0.2)  # Lower cost = higher tempo
        elif card_type == 'SPELL':
            scores['tempo_score'] = 0.6 if cost <= 3 else 0.4
        else:
            scores['tempo_score'] = 0.5
        
        # Value score based on cost and rarity
        scores['value_score'] = base_rarity_score + (0.1 if cost >= 5 else 0.0)
        
        # Curve score based on deck state
        current_curve = getattr(deck_state, 'mana_curve', {})
        pick_number = getattr(deck_state, 'pick_number', 15)
        
        if current_curve and cost in current_curve:
            # Penalize if we already have many cards at this cost
            curve_saturation = current_curve[cost] / max(1, pick_number / 10)
            scores['curve_score'] = max(0.2, 0.8 - curve_saturation * 0.3)
        else:
            scores['curve_score'] = 0.6
        
        # Basic synergy estimation
        scores['synergy_score'] = 0.3  # Conservative default
        
        # Re-draftability based on flexibility
        if rarity in ['COMMON', 'RARE'] and 2 <= cost <= 5:
            scores['re_draftability_score'] = 0.7
        else:
            scores['re_draftability_score'] = 0.4
        
        # Greed score (higher cost/rarity = more greedy)
        scores['greed_score'] = (cost / 10.0) + (rarity_values.get(rarity, 0.45) - 0.45) * 2
        
        return scores
    
    def _assess_data_quality(self, data: Any) -> float:
        """Assess the quality of cached data for fallback purposes."""
        try:
            if data is None:
                return 0.0
            
            # For numeric data (winrates)
            if isinstance(data, (int, float)):
                if 30 <= data <= 80:  # Reasonable winrate range
                    return 0.9
                elif 20 <= data <= 90:
                    return 0.7
                else:
                    return 0.3
            
            # For DimensionalScores objects
            if hasattr(data, 'confidence'):
                return min(0.9, data.confidence * 1.2)
            
            # For complex objects, basic existence check
            return 0.6
            
        except Exception:
            return 0.1
    
    def _cleanup_expired_cache(self, component: str) -> None:
        """Remove expired entries from cache."""
        try:
            if component not in self.known_good_data_cache:
                return
            
            cache = self.known_good_data_cache[component]
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in cache.items():
                age_hours = (current_time - entry['timestamp']).total_seconds() / 3600
                if age_hours > self.cache_ttl_hours:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries for {component}")
                
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")