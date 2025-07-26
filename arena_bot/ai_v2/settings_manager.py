"""
Settings Manager - Comprehensive Configuration System

Provides advanced settings management including hero preference profiles,
statistical thresholds, and personalization options for the AI v2 system.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Import for validation - using relative import to avoid circular dependencies
from .data_models import DimensionalScores


@dataclass
class HeroPreferenceProfile:
    """Individual hero preference configuration."""
    hero_class: str
    preferred_archetypes: List[str]
    playstyle_weight: float  # 0.0-2.0, 1.0 = normal
    complexity_tolerance: str  # "low", "medium", "high"
    auto_select_threshold: float  # Auto-select if winrate advantage > threshold
    avoid_hero: bool = False
    custom_notes: str = ""


@dataclass
class StatisticalThresholds:
    """Configuration for statistical analysis thresholds."""
    confidence_minimum: float = 0.3  # Minimum confidence to trust recommendations
    winrate_significance: float = 1.0  # Minimum winrate difference to consider significant
    meta_stability_threshold: float = 0.8  # Threshold for considering meta stable
    personalization_min_games: int = 10  # Minimum games for personalization
    cache_max_age_hours: int = 12  # Maximum cache age before refresh
    api_timeout_seconds: int = 10  # API call timeout
    fallback_activation_threshold: int = 3  # Errors before fallback activation


@dataclass
class AdvancedSettings:
    """Advanced system configuration options."""
    enable_hero_personalization: bool = True
    enable_conversational_coach: bool = True
    enable_meta_analysis: bool = True
    enable_curve_optimization: bool = True
    enable_synergy_detection: bool = True
    enable_underground_arena_mode: bool = True
    verbose_explanations: bool = True
    auto_update_preferences: bool = True
    experimental_features: bool = False


@dataclass
class UIPreferences:
    """User interface and display preferences."""
    show_confidence_indicators: bool = True
    show_winrate_comparisons: bool = True
    show_meta_position: bool = True
    show_archetype_suggestions: bool = True
    highlight_recommended_picks: bool = True
    enable_hover_questions: bool = True
    compact_display_mode: bool = False
    color_code_recommendations: bool = True


class SettingsManager:
    """
    Comprehensive settings management system.
    
    Handles hero preferences, statistical thresholds, advanced options,
    and user interface preferences with persistent storage and validation.
    """
    
    def __init__(self, settings_file: Optional[str] = None):
        """Initialize settings manager with optional custom settings file."""
        self.logger = logging.getLogger(__name__)
        
        # Settings file path
        if settings_file:
            self.settings_file = Path(settings_file)
        else:
            self.settings_file = Path(__file__).parent.parent.parent / "config" / "ai_v2_settings.json"
        
        # Ensure config directory exists
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize default settings
        self.hero_preferences = self._create_default_hero_preferences()
        self.statistical_thresholds = StatisticalThresholds()
        self.advanced_settings = AdvancedSettings()
        self.ui_preferences = UIPreferences()
        
        # Load existing settings
        self.load_settings()
        
        # Track changes for auto-save
        self.has_unsaved_changes = False
        self.last_save_time = datetime.now()
        
        self.logger.info(f"SettingsManager initialized with file: {self.settings_file}")
    
    def get_hero_preference(self, hero_class: str) -> HeroPreferenceProfile:
        """Get preference profile for specific hero class."""
        return self.hero_preferences.get(hero_class, self._create_default_hero_preference(hero_class))
    
    def set_hero_preference(self, hero_class: str, preference: HeroPreferenceProfile) -> None:
        """Set preference profile for specific hero class."""
        self.hero_preferences[hero_class] = preference
        self.has_unsaved_changes = True
        self.logger.debug(f"Updated hero preference for {hero_class}")
    
    def get_statistical_thresholds(self) -> StatisticalThresholds:
        """Get current statistical thresholds configuration."""
        return self.statistical_thresholds
    
    def update_statistical_thresholds(self, **kwargs) -> None:
        """Update statistical thresholds with provided values."""
        for key, value in kwargs.items():
            if hasattr(self.statistical_thresholds, key):
                setattr(self.statistical_thresholds, key, value)
                self.has_unsaved_changes = True
    
    def get_advanced_settings(self) -> AdvancedSettings:
        """Get current advanced settings configuration."""
        return self.advanced_settings
    
    def update_advanced_settings(self, **kwargs) -> None:
        """Update advanced settings with provided values."""
        for key, value in kwargs.items():
            if hasattr(self.advanced_settings, key):
                setattr(self.advanced_settings, key, value)
                self.has_unsaved_changes = True
    
    def get_ui_preferences(self) -> UIPreferences:
        """Get current UI preferences configuration."""
        return self.ui_preferences
    
    def update_ui_preferences(self, **kwargs) -> None:
        """Update UI preferences with provided values."""
        for key, value in kwargs.items():
            if hasattr(self.ui_preferences, key):
                setattr(self.ui_preferences, key, value)
                self.has_unsaved_changes = True
    
    def should_auto_select_hero(self, hero_class: str, winrate_advantage: float) -> bool:
        """Check if hero should be auto-selected based on preferences."""
        preference = self.get_hero_preference(hero_class)
        
        # Never auto-select avoided heroes
        if preference.avoid_hero:
            return False
        
        # Check if winrate advantage meets threshold
        return winrate_advantage >= preference.auto_select_threshold
    
    def get_hero_archetype_weights(self, hero_class: str) -> Dict[str, float]:
        """Get archetype weights adjusted for hero preferences."""
        preference = self.get_hero_preference(hero_class)
        base_weights = {
            'Aggro': 1.0, 'Tempo': 1.0, 'Control': 1.0, 
            'Attrition': 1.0, 'Synergy': 1.0, 'Balanced': 1.0
        }
        
        # Boost preferred archetypes
        for archetype in preference.preferred_archetypes:
            if archetype in base_weights:
                base_weights[archetype] *= preference.playstyle_weight
        
        return base_weights
    
    def get_complexity_filter(self, hero_class: str) -> Optional[str]:
        """Get complexity filter based on hero preferences."""
        preference = self.get_hero_preference(hero_class)
        return preference.complexity_tolerance
    
    def save_settings(self, force: bool = False) -> bool:
        """Save current settings to file."""
        try:
            if not force and not self.has_unsaved_changes:
                return True
            
            settings_data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'hero_preferences': {
                    hero_class: asdict(preference) 
                    for hero_class, preference in self.hero_preferences.items()
                },
                'statistical_thresholds': asdict(self.statistical_thresholds),
                'advanced_settings': asdict(self.advanced_settings),
                'ui_preferences': asdict(self.ui_preferences)
            }
            
            # Write to file with backup
            backup_file = self.settings_file.with_suffix('.json.backup')
            if self.settings_file.exists():
                self.settings_file.rename(backup_file)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, indent=2, ensure_ascii=False)
            
            # Remove backup on successful write
            if backup_file.exists():
                backup_file.unlink()
            
            self.has_unsaved_changes = False
            self.last_save_time = datetime.now()
            self.logger.info("Settings saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            return False
    
    def load_settings(self) -> bool:
        """Load settings from file."""
        try:
            if not self.settings_file.exists():
                self.logger.info("No settings file found, using defaults")
                return True
            
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)
            
            # Load hero preferences
            if 'hero_preferences' in settings_data:
                for hero_class, pref_data in settings_data['hero_preferences'].items():
                    self.hero_preferences[hero_class] = HeroPreferenceProfile(**pref_data)
            
            # Load statistical thresholds
            if 'statistical_thresholds' in settings_data:
                self.statistical_thresholds = StatisticalThresholds(**settings_data['statistical_thresholds'])
            
            # Load advanced settings
            if 'advanced_settings' in settings_data:
                self.advanced_settings = AdvancedSettings(**settings_data['advanced_settings'])
            
            # Load UI preferences
            if 'ui_preferences' in settings_data:
                self.ui_preferences = UIPreferences(**settings_data['ui_preferences'])
            
            self.has_unsaved_changes = False
            self.logger.info("Settings loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            self.logger.info("Using default settings")
            return False
    
    def reset_to_defaults(self, component: Optional[str] = None) -> None:
        """Reset settings to defaults."""
        if component is None or component == 'hero_preferences':
            self.hero_preferences = self._create_default_hero_preferences()
        
        if component is None or component == 'statistical_thresholds':
            self.statistical_thresholds = StatisticalThresholds()
        
        if component is None or component == 'advanced_settings':
            self.advanced_settings = AdvancedSettings()
        
        if component is None or component == 'ui_preferences':
            self.ui_preferences = UIPreferences()
        
        self.has_unsaved_changes = True
        self.logger.info(f"Reset {component or 'all settings'} to defaults")
    
    def export_settings(self, export_path: str) -> bool:
        """Export current settings to specified file."""
        try:
            export_file = Path(export_path)
            temp_file = self.settings_file
            self.settings_file = export_file
            
            result = self.save_settings(force=True)
            self.settings_file = temp_file
            
            if result:
                self.logger.info(f"Settings exported to {export_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, import_path: str, merge: bool = False) -> bool:
        """Import settings from specified file."""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            if not merge:
                # Full replacement
                backup_file = self.settings_file
                self.settings_file = import_file
                result = self.load_settings()
                self.settings_file = backup_file
            else:
                # Merge with existing settings
                with open(import_file, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                # Selective merge logic here
                # For now, just full replacement for simplicity
                result = self._merge_imported_settings(import_data)
            
            if result:
                self.has_unsaved_changes = True
                self.logger.info(f"Settings imported from {import_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            return False
    
    def validate_settings(self) -> List[str]:
        """Validate current settings and return list of issues."""
        issues = []
        
        # Validate statistical thresholds
        thresholds = self.statistical_thresholds
        
        if not 0.0 <= thresholds.confidence_minimum <= 1.0:
            issues.append("Confidence minimum must be between 0.0 and 1.0")
        
        if thresholds.winrate_significance < 0:
            issues.append("Winrate significance must be non-negative")
        
        if not 0.0 <= thresholds.meta_stability_threshold <= 1.0:
            issues.append("Meta stability threshold must be between 0.0 and 1.0")
        
        if thresholds.personalization_min_games < 1:
            issues.append("Personalization minimum games must be at least 1")
        
        if thresholds.cache_max_age_hours < 1:
            issues.append("Cache max age must be at least 1 hour")
        
        if thresholds.api_timeout_seconds < 1:
            issues.append("API timeout must be at least 1 second")
        
        # Validate hero preferences
        for hero_class, preference in self.hero_preferences.items():
            if not 0.0 <= preference.playstyle_weight <= 2.0:
                issues.append(f"{hero_class}: Playstyle weight must be between 0.0 and 2.0")
            
            if preference.complexity_tolerance not in ['low', 'medium', 'high']:
                issues.append(f"{hero_class}: Invalid complexity tolerance")
            
            if not 0.0 <= preference.auto_select_threshold <= 50.0:
                issues.append(f"{hero_class}: Auto-select threshold must be between 0.0 and 50.0")
        
        return issues
    
    def get_settings_summary(self) -> Dict[str, Any]:
        """Get summary of current settings for display."""
        return {
            'hero_preferences_count': len(self.hero_preferences),
            'avoided_heroes': [
                hero for hero, pref in self.hero_preferences.items() if pref.avoid_hero
            ],
            'personalization_enabled': self.advanced_settings.enable_hero_personalization,
            'confidence_threshold': self.statistical_thresholds.confidence_minimum,
            'cache_age_hours': self.statistical_thresholds.cache_max_age_hours,
            'experimental_features': self.advanced_settings.experimental_features,
            'last_save': self.last_save_time.isoformat() if self.last_save_time else None,
            'has_unsaved_changes': self.has_unsaved_changes,
            'validation_issues': len(self.validate_settings())
        }
    
    def _create_default_hero_preferences(self) -> Dict[str, HeroPreferenceProfile]:
        """Create default hero preference profiles."""
        hero_classes = [
            'MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
            'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER'
        ]
        
        default_archetypes = {
            'MAGE': ['Tempo', 'Control'],
            'PALADIN': ['Aggro', 'Tempo'], 
            'HUNTER': ['Aggro', 'Tempo'],
            'WARRIOR': ['Control', 'Tempo'],
            'PRIEST': ['Control', 'Attrition'],
            'WARLOCK': ['Aggro', 'Control'],
            'ROGUE': ['Tempo', 'Synergy'],
            'SHAMAN': ['Synergy', 'Tempo'],
            'DRUID': ['Control', 'Balanced'],
            'DEMONHUNTER': ['Aggro', 'Tempo']
        }
        
        preferences = {}
        for hero_class in hero_classes:
            preferences[hero_class] = HeroPreferenceProfile(
                hero_class=hero_class,
                preferred_archetypes=default_archetypes.get(hero_class, ['Balanced']),
                playstyle_weight=1.0,
                complexity_tolerance='medium',
                auto_select_threshold=3.0,  # Auto-select if 3%+ winrate advantage
                avoid_hero=False,
                custom_notes=""
            )
        
        return preferences
    
    def _create_default_hero_preference(self, hero_class: str) -> HeroPreferenceProfile:
        """Create default preference for a single hero class."""
        return HeroPreferenceProfile(
            hero_class=hero_class,
            preferred_archetypes=['Balanced'],
            playstyle_weight=1.0,
            complexity_tolerance='medium',
            auto_select_threshold=3.0,
            avoid_hero=False,
            custom_notes=""
        )
    
    def _merge_imported_settings(self, import_data: Dict[str, Any]) -> bool:
        """Merge imported settings with current settings."""
        try:
            # For now, implement simple replacement
            # In future, could add more sophisticated merging logic
            
            if 'hero_preferences' in import_data:
                for hero_class, pref_data in import_data['hero_preferences'].items():
                    self.hero_preferences[hero_class] = HeroPreferenceProfile(**pref_data)
            
            if 'statistical_thresholds' in import_data:
                for key, value in import_data['statistical_thresholds'].items():
                    if hasattr(self.statistical_thresholds, key):
                        setattr(self.statistical_thresholds, key, value)
            
            if 'advanced_settings' in import_data:
                for key, value in import_data['advanced_settings'].items():
                    if hasattr(self.advanced_settings, key):
                        setattr(self.advanced_settings, key, value)
            
            if 'ui_preferences' in import_data:
                for key, value in import_data['ui_preferences'].items():
                    if hasattr(self.ui_preferences, key):
                        setattr(self.ui_preferences, key, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging imported settings: {e}")
            return False


# Global settings manager instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager