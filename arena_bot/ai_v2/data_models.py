"""
AI v2 Data Models - The Core API Contract

Defines the data structures that ensure seamless integration between
all AI v2 components. These dataclasses serve as the "universal language"
for communication between the detection, analysis, and UI systems.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime


@dataclass
class DimensionalScores:
    """
    Multi-dimensional scoring for individual cards.
    Each dimension represents a different strategic aspect.
    """
    card_id: str
    base_value: float = 0.0          # HSReplay winrate + HearthArena tier
    tempo_score: float = 0.0         # Immediate board impact
    value_score: float = 0.0         # Long-term resource advantage  
    synergy_score: float = 0.0       # Synergy with drafted cards
    curve_score: float = 0.0         # Mana curve optimization
    re_draftability_score: float = 0.0  # Underground Arena cut potential
    greed_score: float = 0.0         # Risk/reward specialization
    
    # Statistical confidence (0.0 to 1.0)
    confidence: float = 1.0
    
    # Source data quality indicators
    hsreplay_games_played: int = 0
    data_freshness_hours: float = 0.0


@dataclass  
class DeckState:
    """
    Complete current state of the draft.
    Contains all information needed for intelligent recommendations.
    """
    hero_class: str                  # Selected hero class (e.g. "MAGE") 
    archetype: str                   # User's selected archetype
    drafted_cards: List[str] = field(default_factory=list)  # List of card IDs
    mana_curve: Dict[int, int] = field(default_factory=dict)  # {cost: count}
    pick_number: int = 0            # Current pick (1-30)
    
    # Backward compatibility properties
    @property
    def hero(self) -> str:
        """Backward compatibility for hero field."""
        return self.hero_class
    
    @property 
    def chosen_archetype(self) -> str:
        """Backward compatibility for chosen_archetype field."""
        return self.archetype
    
    # Advanced state tracking
    archetype_conformance: float = 0.0  # How well deck matches archetype
    strategic_gaps: List[str] = field(default_factory=list)  # Missing components
    cut_candidates: List[str] = field(default_factory=list)  # Re-draft targets


@dataclass
class AIDecision:
    """
    Complete AI recommendation package.
    Contains everything needed to display and explain the recommendation.
    """
    recommended_pick_index: int     # Which of the 3 cards (0, 1, or 2)
    all_offered_cards_analysis: List[Dict[str, Any]]  # Full analysis for all 3
    comparative_explanation: str    # Detailed reasoning
    deck_analysis: Dict[str, Any]   # Overall deck state analysis
    card_coordinates: List[Tuple[int, int, int, int]]  # UI positioning
    
    # Advanced features
    pivot_suggestion: Optional[str] = None  # Alternative archetype advice
    confidence_level: float = 1.0    # Overall recommendation confidence
    statistical_backing: Dict[str, Any] = field(default_factory=dict)
    greed_meter: float = 0.0         # Risk level of recommendation
    
    # Meta information
    analysis_time_ms: float = 0.0
    data_sources_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HeroRecommendation:
    """
    Hero selection recommendation with statistical backing.
    """
    recommended_hero_index: int     # Which of the 3 heroes (0, 1, or 2)
    hero_classes: List[str]         # The 3 offered hero classes
    hero_analysis: List[Dict[str, Any]]  # Analysis for each hero
    explanation: str                # Detailed reasoning
    
    # Statistical data
    winrates: Dict[str, float] = field(default_factory=dict)  # {class: winrate}
    meta_trends: Dict[str, str] = field(default_factory=dict)  # Recent changes
    confidence_level: float = 1.0
    
    # Archetype recommendations per hero
    suggested_archetypes: Dict[str, str] = field(default_factory=dict)
    
    # Meta information
    data_freshness_hours: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """
    Real-time performance monitoring data.
    """
    analysis_time_ms: float = 0.0
    api_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Component-specific timings
    coordinate_detection_ms: float = 0.0
    card_identification_ms: float = 0.0
    ai_analysis_ms: float = 0.0
    overlay_render_ms: float = 0.0
    
    # Data quality metrics
    hsreplay_data_age_hours: float = 0.0
    id_translation_success_rate: float = 1.0
    fallback_mode_active: bool = False
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealth:
    """
    Overall system health and status indicators.
    """
    ai_v2_enabled: bool = True
    hsreplay_api_available: bool = True
    heartharena_data_available: bool = True
    overlay_functional: bool = True
    
    # Error tracking
    recent_errors: List[str] = field(default_factory=list)
    recovery_actions_taken: List[str] = field(default_factory=list)
    
    # Performance status
    performance_degraded: bool = False
    memory_pressure: bool = False
    
    # Data freshness
    last_hsreplay_update: Optional[datetime] = None
    last_heartharena_update: Optional[datetime] = None
    
    timestamp: datetime = field(default_factory=datetime.now)