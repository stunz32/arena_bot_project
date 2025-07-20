"""
Settings Integration - AI v2 System Configuration Integration

Integrates the comprehensive settings system with all AI v2 components,
providing seamless configuration propagation and preference application.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import AI v2 components
from .settings_manager import get_settings_manager
from .grandmaster_advisor import GrandmasterAdvisor
from .hero_selector import HeroSelectionAdvisor
from .conversational_coach import ConversationalCoach
from .card_evaluator import CardEvaluationEngine


class SettingsIntegrator:
    """
    Integration layer between settings and AI v2 components.
    
    Ensures settings are properly applied across all system components
    and handles dynamic configuration updates.
    """
    
    def __init__(self):
        """Initialize settings integrator."""
        self.logger = logging.getLogger(__name__)
        self.settings_manager = get_settings_manager()
        
        # Track component instances for configuration updates
        self.registered_components = {
            'grandmaster_advisor': [],
            'hero_selector': [],
            'conversational_coach': [],
            'card_evaluator': []
        }
        
        # Settings change tracking
        self.last_settings_update = datetime.now()
        
        self.logger.info("SettingsIntegrator initialized")
    
    def register_component(self, component_type: str, component_instance) -> None:
        """Register component instance for settings updates."""
        if component_type in self.registered_components:
            self.registered_components[component_type].append(component_instance)
            self._apply_settings_to_component(component_type, component_instance)
            self.logger.debug(f"Registered {component_type} component for settings integration")
    
    def apply_settings_to_all_components(self) -> None:
        """Apply current settings to all registered components."""
        self.logger.info("Applying settings to all components")
        
        for component_type, instances in self.registered_components.items():
            for instance in instances:
                self._apply_settings_to_component(component_type, instance)
        
        self.last_settings_update = datetime.now()
    
    def get_hero_evaluation_config(self, hero_class: str) -> Dict[str, Any]:
        """Get hero-specific evaluation configuration."""
        preference = self.settings_manager.get_hero_preference(hero_class)
        advanced = self.settings_manager.get_advanced_settings()
        thresholds = self.settings_manager.get_statistical_thresholds()
        
        return {
            'hero_class': hero_class,
            'preferred_archetypes': preference.preferred_archetypes,
            'archetype_weights': self.settings_manager.get_hero_archetype_weights(hero_class),
            'playstyle_weight': preference.playstyle_weight,
            'complexity_tolerance': preference.complexity_tolerance,
            'avoid_hero': preference.avoid_hero,
            'auto_select_threshold': preference.auto_select_threshold,
            'enable_personalization': advanced.enable_hero_personalization,
            'enable_meta_analysis': advanced.enable_meta_analysis,
            'enable_curve_optimization': advanced.enable_curve_optimization,
            'enable_synergy_detection': advanced.enable_synergy_detection,
            'confidence_minimum': thresholds.confidence_minimum,
            'winrate_significance': thresholds.winrate_significance,
            'meta_stability_threshold': thresholds.meta_stability_threshold
        }
    
    def get_recommendation_config(self) -> Dict[str, Any]:
        """Get general recommendation configuration."""
        advanced = self.settings_manager.get_advanced_settings()
        thresholds = self.settings_manager.get_statistical_thresholds()
        ui_prefs = self.settings_manager.get_ui_preferences()
        
        return {
            'enable_conversational_coach': advanced.enable_conversational_coach,
            'enable_underground_arena_mode': advanced.enable_underground_arena_mode,
            'verbose_explanations': advanced.verbose_explanations,
            'experimental_features': advanced.experimental_features,
            'cache_max_age_hours': thresholds.cache_max_age_hours,
            'api_timeout_seconds': thresholds.api_timeout_seconds,
            'fallback_activation_threshold': thresholds.fallback_activation_threshold,
            'show_confidence_indicators': ui_prefs.show_confidence_indicators,
            'show_winrate_comparisons': ui_prefs.show_winrate_comparisons,
            'show_meta_position': ui_prefs.show_meta_position,
            'highlight_recommended_picks': ui_prefs.highlight_recommended_picks
        }
    
    def should_auto_select_hero(self, hero_class: str, winrate_advantage: float) -> bool:
        """Check if hero should be auto-selected based on settings."""
        return self.settings_manager.should_auto_select_hero(hero_class, winrate_advantage)
    
    def get_ui_display_config(self) -> Dict[str, Any]:
        """Get UI display configuration."""
        ui_prefs = self.settings_manager.get_ui_preferences()
        advanced = self.settings_manager.get_advanced_settings()
        
        return {
            'show_confidence_indicators': ui_prefs.show_confidence_indicators,
            'show_winrate_comparisons': ui_prefs.show_winrate_comparisons,
            'show_meta_position': ui_prefs.show_meta_position,
            'show_archetype_suggestions': ui_prefs.show_archetype_suggestions,
            'highlight_recommended_picks': ui_prefs.highlight_recommended_picks,
            'enable_hover_questions': ui_prefs.enable_hover_questions,
            'compact_display_mode': ui_prefs.compact_display_mode,
            'color_code_recommendations': ui_prefs.color_code_recommendations,
            'verbose_explanations': advanced.verbose_explanations
        }
    
    def get_conversational_coach_config(self, hero_class: Optional[str] = None) -> Dict[str, Any]:
        """Get conversational coach configuration."""
        advanced = self.settings_manager.get_advanced_settings()
        ui_prefs = self.settings_manager.get_ui_preferences()
        
        config = {
            'enabled': advanced.enable_conversational_coach,
            'enable_hover_questions': ui_prefs.enable_hover_questions,
            'verbose_explanations': advanced.verbose_explanations,
            'experimental_features': advanced.experimental_features
        }
        
        # Add hero-specific config if provided
        if hero_class:
            preference = self.settings_manager.get_hero_preference(hero_class)
            config.update({
                'hero_class': hero_class,
                'preferred_archetypes': preference.preferred_archetypes,
                'complexity_tolerance': preference.complexity_tolerance,
                'custom_notes': preference.custom_notes
            })
        
        return config
    
    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source and caching configuration."""
        thresholds = self.settings_manager.get_statistical_thresholds()
        advanced = self.settings_manager.get_advanced_settings()
        
        return {
            'cache_max_age_hours': thresholds.cache_max_age_hours,
            'api_timeout_seconds': thresholds.api_timeout_seconds,
            'fallback_activation_threshold': thresholds.fallback_activation_threshold,
            'enable_meta_analysis': advanced.enable_meta_analysis,
            'experimental_features': advanced.experimental_features,
            'personalization_min_games': thresholds.personalization_min_games
        }
    
    def update_component_settings(self, component_type: str) -> None:
        """Update settings for specific component type."""
        if component_type in self.registered_components:
            for instance in self.registered_components[component_type]:
                self._apply_settings_to_component(component_type, instance)
    
    def _apply_settings_to_component(self, component_type: str, instance) -> None:
        """Apply settings to a specific component instance."""
        try:
            if component_type == 'grandmaster_advisor':
                self._configure_grandmaster_advisor(instance)
            elif component_type == 'hero_selector':
                self._configure_hero_selector(instance)
            elif component_type == 'conversational_coach':
                self._configure_conversational_coach(instance)
            elif component_type == 'card_evaluator':
                self._configure_card_evaluator(instance)
            
        except Exception as e:
            self.logger.error(f"Error applying settings to {component_type}: {e}")
    
    def _configure_grandmaster_advisor(self, advisor: GrandmasterAdvisor) -> None:
        """Configure GrandmasterAdvisor with current settings."""
        config = self.get_recommendation_config()
        
        # Update advisor configuration
        if hasattr(advisor, 'set_configuration'):
            advisor.set_configuration(config)
        
        # Update archetype weights for all heroes
        for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                          'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
            weights = self.settings_manager.get_hero_archetype_weights(hero_class)
            if hasattr(advisor, 'update_hero_archetype_weights'):
                advisor.update_hero_archetype_weights(hero_class, weights)
    
    def _configure_hero_selector(self, selector: HeroSelectionAdvisor) -> None:
        """Configure HeroSelectionAdvisor with current settings."""
        advanced = self.settings_manager.get_advanced_settings()
        thresholds = self.settings_manager.get_statistical_thresholds()
        
        # Update personalization settings
        if hasattr(selector, 'set_personalization_enabled'):
            selector.set_personalization_enabled(advanced.enable_hero_personalization)
        
        # Update cache settings
        if hasattr(selector, 'set_cache_config'):
            selector.set_cache_config({
                'max_age_hours': thresholds.cache_max_age_hours,
                'api_timeout': thresholds.api_timeout_seconds
            })
        
        # Update hero preferences
        for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                          'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
            preference = self.settings_manager.get_hero_preference(hero_class)
            if hasattr(selector, 'set_hero_preference'):
                selector.set_hero_preference(hero_class, preference)
    
    def _configure_conversational_coach(self, coach: ConversationalCoach) -> None:
        """Configure ConversationalCoach with current settings."""
        advanced = self.settings_manager.get_advanced_settings()
        ui_prefs = self.settings_manager.get_ui_preferences()
        
        # Update coach configuration
        if hasattr(coach, 'set_configuration'):
            coach.set_configuration({
                'enabled': advanced.enable_conversational_coach,
                'enable_hover_questions': ui_prefs.enable_hover_questions,
                'verbose_explanations': advanced.verbose_explanations
            })
        
        # Update hero preferences for contextual coaching
        for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                          'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
            preference = self.settings_manager.get_hero_preference(hero_class)
            if hasattr(coach, 'set_hero_coaching_context'):
                coach.set_hero_coaching_context(hero_class, {
                    'preferred_archetypes': preference.preferred_archetypes,
                    'complexity_tolerance': preference.complexity_tolerance,
                    'custom_notes': preference.custom_notes
                })
    
    def _configure_card_evaluator(self, evaluator: CardEvaluationEngine) -> None:
        """Configure CardEvaluationEngine with current settings."""
        advanced = self.settings_manager.get_advanced_settings()
        thresholds = self.settings_manager.get_statistical_thresholds()
        
        # Update evaluator configuration
        if hasattr(evaluator, 'set_configuration'):
            evaluator.set_configuration({
                'enable_synergy_detection': advanced.enable_synergy_detection,
                'enable_curve_optimization': advanced.enable_curve_optimization,
                'enable_underground_arena_mode': advanced.enable_underground_arena_mode,
                'confidence_minimum': thresholds.confidence_minimum,
                'cache_max_age_hours': thresholds.cache_max_age_hours
            })
        
        # Update hero-specific evaluation weights
        for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                          'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
            weights = self.settings_manager.get_hero_archetype_weights(hero_class)
            preference = self.settings_manager.get_hero_preference(hero_class)
            
            if hasattr(evaluator, 'set_hero_evaluation_weights'):
                evaluator.set_hero_evaluation_weights(hero_class, {
                    'archetype_weights': weights,
                    'playstyle_weight': preference.playstyle_weight,
                    'complexity_tolerance': preference.complexity_tolerance
                })
    
    def validate_all_settings(self) -> List[str]:
        """Validate all settings across the system."""
        return self.settings_manager.validate_settings()
    
    def get_settings_summary_for_display(self) -> Dict[str, Any]:
        """Get formatted settings summary for UI display."""
        summary = self.settings_manager.get_settings_summary()
        
        # Add integration-specific information
        summary.update({
            'registered_components': {
                component_type: len(instances) 
                for component_type, instances in self.registered_components.items()
            },
            'last_settings_update': self.last_settings_update.isoformat(),
            'integration_status': 'active'
        })
        
        return summary
    
    def export_component_config(self, component_type: str) -> Dict[str, Any]:
        """Export configuration for specific component type."""
        if component_type == 'grandmaster_advisor':
            return self.get_recommendation_config()
        elif component_type == 'hero_selector':
            return {
                'hero_preferences': {
                    hero_class: self.get_hero_evaluation_config(hero_class)
                    for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                                      'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']
                },
                'data_source_config': self.get_data_source_config()
            }
        elif component_type == 'conversational_coach':
            return self.get_conversational_coach_config()
        elif component_type == 'card_evaluator':
            return {
                'evaluation_config': self.get_recommendation_config(),
                'hero_configs': {
                    hero_class: self.get_hero_evaluation_config(hero_class)
                    for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                                      'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']
                }
            }
        else:
            return {}


# Global settings integrator instance
_settings_integrator = None

def get_settings_integrator() -> SettingsIntegrator:
    """Get global settings integrator instance."""
    global _settings_integrator
    if _settings_integrator is None:
        _settings_integrator = SettingsIntegrator()
    return _settings_integrator


def register_component_for_settings(component_type: str, component_instance) -> None:
    """Register component for automatic settings integration."""
    integrator = get_settings_integrator()
    integrator.register_component(component_type, component_instance)


def apply_settings_to_all_components() -> None:
    """Apply current settings to all registered components."""
    integrator = get_settings_integrator()
    integrator.apply_settings_to_all_components()


def get_hero_config_for_evaluation(hero_class: str) -> Dict[str, Any]:
    """Get hero-specific configuration for evaluation components."""
    integrator = get_settings_integrator()
    return integrator.get_hero_evaluation_config(hero_class)


def get_ui_config_for_display() -> Dict[str, Any]:
    """Get UI configuration for display components."""
    integrator = get_settings_integrator()
    return integrator.get_ui_display_config()