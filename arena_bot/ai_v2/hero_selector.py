"""
Hero Selection Advisor

Provides statistical hero recommendations using HSReplay data
combined with qualitative analysis of playstyle and complexity.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .data_models import HeroRecommendation

# Import HSReplay integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_sourcing.hsreplay_scraper import get_hsreplay_scraper


class HeroSelectionAdvisor:
    """
    Hero selection recommendation system.
    
    Combines HSReplay hero winrate data with qualitative analysis
    to provide comprehensive hero choice guidance.
    """
    
    def __init__(self):
        """Initialize the hero selection advisor."""
        self.logger = logging.getLogger(__name__)
        
        # HSReplay integration
        self.hsreplay_scraper = get_hsreplay_scraper()
        self.hero_winrates = {}  # Cached {CLASS_NAME: winrate}
        self.last_winrate_fetch = None
        
        # Qualitative hero profiles
        self.class_profiles = self._build_class_profiles()
        
        # Performance tracking
        self.recommendation_count = 0
        
        # User performance history for personalization
        self.user_performance_history = {}  # {hero_class: [performance_records]}
        self.user_preferences = {}  # User-specific preferences and patterns
        self.personalization_enabled = True
        
        self.logger.info("HeroSelectionAdvisor initialized with HSReplay integration and personalization")
    
    def recommend_hero(self, hero_classes: List[str]) -> HeroRecommendation:
        """
        Generate hero recommendation with statistical backing.
        
        Args:
            hero_classes: List of 3 offered hero classes (e.g., ["WARRIOR", "MAGE", "PALADIN"])
            
        Returns:
            HeroRecommendation with analysis and statistical data
        """
        start_time = datetime.now()
        self.recommendation_count += 1
        
        # Get fresh hero winrates from HSReplay
        winrates = self._get_current_winrates()
        
        # Analyze each hero option
        hero_analysis = []
        hero_scores = []
        
        for i, hero_class in enumerate(hero_classes):
            profile = self.class_profiles.get(hero_class, {})
            winrate = winrates.get(hero_class, 50.0)
            
            # Calculate comprehensive score
            score = self._calculate_hero_score(hero_class, winrate, profile)
            hero_scores.append(score)
            
            # Generate detailed analysis
            analysis = {
                "class": hero_class,
                "winrate": winrate,
                "profile": profile,
                "confidence": self._calculate_confidence(winrate, winrates),
                "explanation": self._generate_hero_explanation(hero_class, winrate, profile, winrates),
                "score": score,
                "meta_position": self._assess_meta_position(hero_class, winrate, winrates)
            }
            hero_analysis.append(analysis)
        
        # Determine best recommendation
        recommended_index = hero_scores.index(max(hero_scores))
        recommended_class = hero_classes[recommended_index]
        
        # Generate comprehensive explanation
        explanation = self._generate_recommendation_explanation(
            recommended_class, hero_classes, hero_analysis, winrates
        )
        
        # Calculate overall confidence
        confidence_level = self._calculate_overall_confidence(hero_analysis, winrates)
        
        recommendation = HeroRecommendation(
            recommended_hero_index=recommended_index,
            hero_classes=hero_classes,
            hero_analysis=hero_analysis,
            explanation=explanation,
            winrates={cls: winrates.get(cls, 50.0) for cls in hero_classes},
            confidence_level=confidence_level
        )
        
        analysis_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"Hero recommendation generated in {analysis_time:.1f}ms: {recommended_class}")
        
        return recommendation
    
    def _build_class_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Build qualitative profiles for each hero class.
        
        Contains playstyle, complexity, strengths, and recommended archetypes.
        """
        return {
            "WARRIOR": {
                "playstyle": "Control/Midrange",
                "complexity": "Medium", 
                "description": "Durable class with weapon synergies and armor gain",
                "strengths": ["Weapons", "Armor", "Board clears"],
                "weaknesses": ["Card draw", "Direct damage"],
                "recommended_archetypes": ["Control", "Attrition", "Tempo"]
            },
            "MAGE": {
                "playstyle": "Tempo/Control",
                "complexity": "Medium-High",
                "description": "Versatile spellcaster with powerful removal and card generation",
                "strengths": ["Spells", "Card generation", "Flexibility"],
                "weaknesses": ["Health management", "Minion quality"],
                "recommended_archetypes": ["Tempo", "Control", "Synergy"]
            },
            "HUNTER": {
                "playstyle": "Aggro/Tempo", 
                "complexity": "Low-Medium",
                "description": "Fast aggressive class with beast synergies",
                "strengths": ["Early pressure", "Beast synergy", "Direct damage"],
                "weaknesses": ["Card draw", "Late game"],
                "recommended_archetypes": ["Aggro", "Tempo", "Synergy"]
            },
            "PRIEST": {
                "playstyle": "Control/Attrition",
                "complexity": "High",
                "description": "Defensive class focused on healing and value generation", 
                "strengths": ["Healing", "Value generation", "Board clears"],
                "weaknesses": ["Early pressure", "Tempo"],
                "recommended_archetypes": ["Control", "Attrition", "Balanced"]
            },
            "WARLOCK": {
                "playstyle": "Flexible",
                "complexity": "Medium",
                "description": "Life-for-power class with strong card draw",
                "strengths": ["Card draw", "Flexibility", "Powerful effects"],
                "weaknesses": ["Health management", "Self-damage"],
                "recommended_archetypes": ["Aggro", "Control", "Attrition"]
            },
            "ROGUE": {
                "playstyle": "Tempo/Combo",
                "complexity": "High", 
                "description": "Efficient class with weapon synergies and combo mechanics",
                "strengths": ["Efficiency", "Weapons", "Card generation"],
                "weaknesses": ["Health management", "AOE"],
                "recommended_archetypes": ["Tempo", "Synergy", "Balanced"]
            },
            "SHAMAN": {
                "playstyle": "Synergy/Midrange",
                "complexity": "Medium-High",
                "description": "Elemental and Overload synergies with versatile spells",
                "strengths": ["Elemental synergy", "Versatile spells", "Value"],
                "weaknesses": ["Overload management", "Consistency"],
                "recommended_archetypes": ["Synergy", "Tempo", "Balanced"]
            },
            "PALADIN": {
                "playstyle": "Tempo/Aggro",
                "complexity": "Medium",
                "description": "Divine Shield and weapon synergies with consistent pressure",
                "strengths": ["Divine Shield", "Weapons", "Board presence"],
                "weaknesses": ["Card draw", "Removal"],
                "recommended_archetypes": ["Tempo", "Aggro", "Balanced"]
            },
            "DRUID": {
                "playstyle": "Ramp/Midrange", 
                "complexity": "Medium",
                "description": "Mana acceleration and choose mechanics with big minions",
                "strengths": ["Mana acceleration", "Big minions", "Flexibility"],
                "weaknesses": ["Early game", "Removal"],
                "recommended_archetypes": ["Control", "Balanced", "Attrition"]
            },
            "DEMONHUNTER": {
                "playstyle": "Aggro/Tempo",
                "complexity": "Low-Medium", 
                "description": "Fast aggressive class with outcast mechanics",
                "strengths": ["Early pressure", "Efficient spells", "Weapons"],
                "weaknesses": ["Card draw", "Late game"],
                "recommended_archetypes": ["Aggro", "Tempo", "Balanced"]
            }
        }
    
    def _get_current_winrates(self, force_refresh: bool = False) -> Dict[str, float]:
        """Get current hero winrates with caching and enhanced reliability."""
        try:
            # Check if we need to refresh data
            if force_refresh or not self.hero_winrates or self._should_refresh_winrates():
                self.logger.debug("Fetching fresh hero winrates from HSReplay")
                
                # Validate HSReplay connection before attempting data fetch
                if not self._validate_hsreplay_connection():
                    self.logger.warning("HSReplay connection validation failed, using cached/fallback data")
                    return self.hero_winrates if self.hero_winrates else self._get_fallback_winrates()
                
                # Fetch winrates with retry logic
                fresh_winrates = self._fetch_winrates_with_retry(force_refresh=force_refresh)
                
                if fresh_winrates:
                    self.hero_winrates = fresh_winrates
                    self.last_winrate_fetch = datetime.now()
                    self.logger.info(f"✅ Updated hero winrates: {len(fresh_winrates)} classes")
                else:
                    self.logger.warning("❌ Failed to fetch fresh winrates, using cached/fallback data")
            
            return self.hero_winrates if self.hero_winrates else self._get_fallback_winrates()
            
        except Exception as e:
            self.logger.error(f"❌ Error getting hero winrates: {e}")
            return self._get_fallback_winrates()
    
    def _should_refresh_winrates(self) -> bool:
        """Check if winrates need refreshing based on age."""
        if not self.last_winrate_fetch:
            return True
        
        # Refresh every 12 hours
        age = datetime.now() - self.last_winrate_fetch
        return age.total_seconds() > (12 * 3600)
    
    def _calculate_hero_score(self, hero_class: str, winrate: float, profile: Dict[str, Any]) -> float:
        """Calculate comprehensive hero score combining statistical and qualitative factors."""
        # Base score from winrate (0-100 scale)
        winrate_score = winrate
        
        # Complexity penalty for new players (simplified for now)
        complexity = profile.get("complexity", "Medium")
        complexity_modifier = {
            "Low": 1.02,
            "Low-Medium": 1.01,
            "Medium": 1.0,
            "Medium-High": 0.99,
            "High": 0.98
        }.get(complexity, 1.0)
        
        # Meta stability bonus (consistent performers get small bonus)
        meta_stability = 1.0  # Placeholder - future: analyze winrate variance
        
        final_score = winrate_score * complexity_modifier * meta_stability
        
        return final_score
    
    def _calculate_confidence(self, winrate: float, all_winrates: Dict[str, float]) -> float:
        """Calculate confidence level for individual hero recommendation."""
        if not all_winrates:
            return 0.5  # Low confidence with no data
        
        # Calculate average winrate
        avg_winrate = sum(all_winrates.values()) / len(all_winrates)
        
        # Higher confidence for winrates further from average
        deviation = abs(winrate - avg_winrate)
        confidence = min(0.95, 0.7 + (deviation / 10))  # Scale deviation to confidence
        
        return confidence
    
    def _generate_hero_explanation(self, hero_class: str, winrate: float, 
                                 profile: Dict[str, Any], all_winrates: Dict[str, float]) -> str:
        """Generate detailed explanation for individual hero choice."""
        avg_winrate = sum(all_winrates.values()) / len(all_winrates) if all_winrates else 50.0
        winrate_diff = winrate - avg_winrate
        
        # Performance assessment
        if winrate_diff > 3:
            performance = f"{winrate:.1f}% winrate ({winrate_diff:+.1f}% above average)"
            performance_tier = "Strong"
        elif winrate_diff > 1:
            performance = f"{winrate:.1f}% winrate ({winrate_diff:+.1f}% above average)" 
            performance_tier = "Good"
        elif winrate_diff > -1:
            performance = f"{winrate:.1f}% winrate (near average)"
            performance_tier = "Balanced"
        else:
            performance = f"{winrate:.1f}% winrate ({winrate_diff:+.1f}% below average)"
            performance_tier = "Challenging"
        
        # Get profile details
        playstyle = profile.get("playstyle", "Unknown")
        complexity = profile.get("complexity", "Unknown")
        description = profile.get("description", "")
        
        explanation = f"{hero_class}: {performance}. {playstyle} playstyle, {complexity} complexity. {description}"
        
        return explanation
    
    def _assess_meta_position(self, hero_class: str, winrate: float, all_winrates: Dict[str, float]) -> str:
        """Assess hero's position in current meta."""
        if not all_winrates:
            return "Unknown"
        
        # Rank among all heroes
        sorted_winrates = sorted(all_winrates.values(), reverse=True)
        position = sorted_winrates.index(winrate) + 1
        total_heroes = len(sorted_winrates)
        
        if position <= total_heroes * 0.2:
            return "Meta Dominator"
        elif position <= total_heroes * 0.4:
            return "Meta Strong"
        elif position <= total_heroes * 0.6:
            return "Meta Viable"
        elif position <= total_heroes * 0.8:
            return "Meta Weak"
        else:
            return "Off-Meta"
    
    def _generate_recommendation_explanation(self, recommended_class: str, 
                                           hero_classes: List[str], 
                                           hero_analysis: List[Dict],
                                           winrates: Dict[str, float]) -> str:
        """Generate comprehensive recommendation explanation."""
        recommended_analysis = next(h for h in hero_analysis if h["class"] == recommended_class)
        
        winrate = recommended_analysis["winrate"]
        meta_position = recommended_analysis["meta_position"] 
        profile = recommended_analysis["profile"]
        
        # Get comparison context
        other_classes = [cls for cls in hero_classes if cls != recommended_class]
        other_winrates = [winrates.get(cls, 50.0) for cls in other_classes]
        
        if other_winrates:
            avg_other = sum(other_winrates) / len(other_winrates)
            advantage = winrate - avg_other
            
            if advantage > 2:
                comparison = f"Clear advantage over alternatives (+{advantage:.1f}%)"
            elif advantage > 0.5:
                comparison = f"Slight edge over alternatives (+{advantage:.1f}%)"
            else:
                comparison = f"Competitive option among close choices"
        else:
            comparison = "Single option analysis"
        
        # Get recommended archetypes
        archetypes = profile.get("recommended_archetypes", [])
        archetype_text = f"Best archetypes: {', '.join(archetypes[:3])}" if archetypes else ""
        
        explanation = (f"Recommended: {recommended_class} ({winrate:.1f}% winrate, {meta_position}). "
                      f"{comparison}. {archetype_text}")
        
        return explanation
    
    def _calculate_overall_confidence(self, hero_analysis: List[Dict], winrates: Dict[str, float]) -> float:
        """Calculate overall confidence in the recommendation."""
        if not hero_analysis or not winrates:
            return 0.3  # Low confidence with insufficient data
        
        # Factor 1: Data quality (number of heroes with data)
        heroes_with_data = len([h for h in hero_analysis if h["winrate"] != 50.0])
        data_quality = heroes_with_data / len(hero_analysis)
        
        # Factor 2: Recommendation clarity (how clear the winner is)
        winrates_list = [h["winrate"] for h in hero_analysis]
        winrate_spread = max(winrates_list) - min(winrates_list)
        clarity = min(1.0, winrate_spread / 10)  # 10% spread = full clarity
        
        # Factor 3: Meta stability (simplified - assume stable for now)
        stability = 0.8  # Placeholder
        
        # Combine factors
        confidence = (data_quality * 0.4 + clarity * 0.4 + stability * 0.2)
        
        return max(0.3, min(0.95, confidence))  # Clamp between 30% and 95%
    
    def _get_fallback_winrates(self) -> Dict[str, float]:
        """Provide fallback winrates when HSReplay data unavailable."""
        # Conservative fallback based on general Arena performance
        return {
            "MAGE": 53.5,
            "PALADIN": 52.8, 
            "ROGUE": 52.3,
            "HUNTER": 51.9,
            "WARLOCK": 51.2,
            "WARRIOR": 50.8,
            "SHAMAN": 50.5,
            "DRUID": 49.7,
            "PRIEST": 49.2,
            "DEMONHUNTER": 48.9
        }
    
    def detect_meta_shifts(self) -> List[Dict[str, Any]]:
        """Detect significant changes in hero performance (future enhancement)."""
        # Placeholder for meta shift detection
        # Would compare current winrates with historical data
        return []
    
    def get_hero_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hero selection statistics."""
        return {
            "recommendations_made": self.recommendation_count,
            "last_winrate_fetch": self.last_winrate_fetch.isoformat() if self.last_winrate_fetch else None,
            "cached_winrates": len(self.hero_winrates),
            "winrate_data_age_hours": (
                (datetime.now() - self.last_winrate_fetch).total_seconds() / 3600 
                if self.last_winrate_fetch else None
            ),
            "api_status": self.hsreplay_scraper.get_api_status() if self.hsreplay_scraper else {},
            "personalization_stats": {
                "heroes_tracked": len(self.user_performance_history),
                "total_games_recorded": sum(len(records) for records in self.user_performance_history.values()),
                "personalization_enabled": self.personalization_enabled
            }
        }
    
    # === PERSONALIZED RECOMMENDATION METHODS ===
    
    def get_personalized_hero_recommendations(self, hero_classes: List[str], 
                                            user_id: str = "default") -> Dict[str, Any]:
        """
        Generate personalized hero recommendations based on user's historical performance.
        
        Combines general meta data with user-specific performance patterns to provide
        tailored recommendations that account for individual skill with each hero class.
        """
        try:
            if not self.personalization_enabled:
                # Fall back to standard recommendations
                standard_rec = self.recommend_hero(hero_classes)
                return self._convert_to_personalized_format(standard_rec, "Personalization disabled")
            
            # Get user's historical performance
            user_performance = self._get_user_performance_data(user_id)
            
            if not user_performance or self._insufficient_data(user_performance):
                # Not enough data for personalization, use standard with learning mode
                standard_rec = self.recommend_hero(hero_classes)
                return self._convert_to_personalized_format(standard_rec, "Learning mode - building performance history")
            
            # Get general meta winrates
            meta_winrates = self._get_current_winrates()
            
            # Generate personalized analysis for each offered hero
            personalized_analysis = []
            for hero_class in hero_classes:
                analysis = self._analyze_hero_for_user(hero_class, user_performance, meta_winrates, user_id)
                personalized_analysis.append(analysis)
            
            # Determine best recommendation based on personalized scores
            best_hero_index = max(range(len(personalized_analysis)), 
                                key=lambda i: personalized_analysis[i]['personalized_score'])
            
            # Generate comprehensive personalized recommendation
            return {
                'user_id': user_id,
                'recommended_hero_index': best_hero_index,
                'hero_classes': hero_classes,
                'personalized_analysis': personalized_analysis,
                'personalization_factors': self._get_personalization_factors(user_performance),
                'confidence_level': self._calculate_personalized_confidence(personalized_analysis),
                'learning_insights': self._generate_learning_insights(user_performance, hero_classes),
                'recommendation_explanation': self._generate_personalized_explanation(
                    hero_classes[best_hero_index], personalized_analysis[best_hero_index], user_performance
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating personalized recommendations: {e}")
            # Fallback to standard recommendation
            standard_rec = self.recommend_hero(hero_classes)
            return self._convert_to_personalized_format(standard_rec, f"Error in personalization: {str(e)}")
    
    def record_user_performance(self, user_id: str, hero_class: str, 
                              wins: int, total_games: int, additional_stats: Dict = None) -> None:
        """Record user's performance with a specific hero for future personalization."""
        try:
            if user_id not in self.user_performance_history:
                self.user_performance_history[user_id] = {}
            
            if hero_class not in self.user_performance_history[user_id]:
                self.user_performance_history[user_id][hero_class] = []
            
            performance_record = {
                'timestamp': datetime.now(),
                'wins': wins,
                'total_games': total_games,
                'winrate': (wins / total_games * 100) if total_games > 0 else 0,
                'additional_stats': additional_stats or {}
            }
            
            self.user_performance_history[user_id][hero_class].append(performance_record)
            
            # Keep only last 50 records per hero to avoid memory bloat
            if len(self.user_performance_history[user_id][hero_class]) > 50:
                self.user_performance_history[user_id][hero_class] = \
                    self.user_performance_history[user_id][hero_class][-50:]
            
            # Update user preferences based on performance
            self._update_user_preferences(user_id, hero_class, performance_record)
            
            self.logger.debug(f"Recorded performance for user {user_id}, hero {hero_class}: {wins}/{total_games}")
            
        except Exception as e:
            self.logger.error(f"Error recording user performance: {e}")
    
    def _get_user_performance_data(self, user_id: str) -> Dict[str, List[Dict]]:
        """Get user's performance history across all heroes."""
        return self.user_performance_history.get(user_id, {})
    
    def _insufficient_data(self, user_performance: Dict) -> bool:
        """Check if user has insufficient data for reliable personalization."""
        # Need at least 3 heroes with 5+ games each for basic personalization
        sufficient_heroes = 0
        for hero_class, records in user_performance.items():
            total_games = sum(record['total_games'] for record in records)
            if total_games >= 5:
                sufficient_heroes += 1
        
        return sufficient_heroes < 3
    
    def _analyze_hero_for_user(self, hero_class: str, user_performance: Dict, 
                             meta_winrates: Dict, user_id: str) -> Dict[str, Any]:
        """Analyze a specific hero for the user's personalized recommendation."""
        try:
            # Get meta winrate
            meta_winrate = meta_winrates.get(hero_class, 50.0)
            
            # Get user's historical performance with this hero
            user_hero_records = user_performance.get(hero_class, [])
            
            if user_hero_records:
                # Calculate user's personal winrate with this hero
                total_wins = sum(record['wins'] for record in user_hero_records)
                total_games = sum(record['total_games'] for record in user_hero_records)
                user_winrate = (total_wins / total_games * 100) if total_games > 0 else 0
                
                # Calculate performance vs meta
                performance_vs_meta = user_winrate - meta_winrate
                
                # Confidence based on sample size
                data_confidence = min(1.0, total_games / 20)  # Full confidence at 20+ games
                
                # Recent performance trend (last 5 records)
                recent_records = user_hero_records[-5:]
                recent_wins = sum(record['wins'] for record in recent_records)
                recent_games = sum(record['total_games'] for record in recent_records)
                recent_winrate = (recent_wins / recent_games * 100) if recent_games > 0 else user_winrate
                
                # Calculate trend
                trend = recent_winrate - user_winrate if len(user_hero_records) > 5 else 0
                
            else:
                # No user data for this hero - use meta data with low confidence
                user_winrate = meta_winrate
                performance_vs_meta = 0
                data_confidence = 0.1
                trend = 0
                total_games = 0
            
            # Calculate personalized score
            personalized_score = self._calculate_personalized_score(
                meta_winrate, user_winrate, performance_vs_meta, data_confidence, trend
            )
            
            # Get class profile for additional context
            profile = self.class_profiles.get(hero_class, {})
            
            return {
                'hero_class': hero_class,
                'meta_winrate': round(meta_winrate, 2),
                'user_winrate': round(user_winrate, 2),
                'user_total_games': total_games,
                'performance_vs_meta': round(performance_vs_meta, 2),
                'recent_trend': round(trend, 2),
                'data_confidence': round(data_confidence, 2),
                'personalized_score': round(personalized_score, 2),
                'recommendation_reason': self._generate_hero_recommendation_reason(
                    hero_class, performance_vs_meta, trend, data_confidence, profile
                ),
                'skill_assessment': self._assess_user_skill_with_hero(performance_vs_meta, total_games),
                'improvement_potential': self._assess_improvement_potential(trend, data_confidence)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing hero {hero_class} for user: {e}")
            # Fallback analysis
            return {
                'hero_class': hero_class,
                'meta_winrate': meta_winrates.get(hero_class, 50.0),
                'user_winrate': 0,
                'user_total_games': 0,
                'performance_vs_meta': 0,
                'personalized_score': meta_winrates.get(hero_class, 50.0),
                'recommendation_reason': f"Insufficient data for {hero_class}",
                'data_confidence': 0.1
            }
    
    def _calculate_personalized_score(self, meta_winrate: float, user_winrate: float, 
                                    performance_vs_meta: float, data_confidence: float, trend: float) -> float:
        """Calculate personalized score combining meta and user performance."""
        # Base score from meta winrate
        base_score = meta_winrate
        
        # User performance adjustment (weighted by confidence)
        user_adjustment = performance_vs_meta * data_confidence
        
        # Trend bonus/penalty
        trend_adjustment = trend * 0.3  # Trend has less impact than overall performance
        
        # Combine all factors
        personalized_score = base_score + user_adjustment + trend_adjustment
        
        # Ensure score stays within reasonable bounds
        return max(0, min(100, personalized_score))
    
    def _generate_hero_recommendation_reason(self, hero_class: str, performance_vs_meta: float, 
                                           trend: float, confidence: float, profile: Dict) -> str:
        """Generate explanation for hero recommendation."""
        reasons = []
        
        # Performance vs meta
        if confidence > 0.5:  # Only mention if we have decent data
            if performance_vs_meta > 3:
                reasons.append(f"You excel with {hero_class} (+{performance_vs_meta:.1f}% vs meta)")
            elif performance_vs_meta > 1:
                reasons.append(f"Above-average performance with {hero_class}")
            elif performance_vs_meta < -3:
                reasons.append(f"Below your usual performance with {hero_class}")
            elif performance_vs_meta < -1:
                reasons.append(f"Room for improvement with {hero_class}")
        
        # Trend analysis
        if abs(trend) > 2 and confidence > 0.3:
            if trend > 0:
                reasons.append(f"Recent upward trend (+{trend:.1f}%)")
            else:
                reasons.append(f"Recent struggles ({trend:.1f}%)")
        
        # Class characteristics
        playstyle = profile.get('playstyle', 'Unknown')
        if playstyle != 'Unknown':
            reasons.append(f"{playstyle} playstyle")
        
        if not reasons:
            reasons.append(f"Solid meta choice")
        
        return ". ".join(reasons)
    
    def _assess_user_skill_with_hero(self, performance_vs_meta: float, total_games: int) -> str:
        """Assess user's skill level with specific hero."""
        if total_games < 5:
            return "Insufficient data"
        elif performance_vs_meta > 5:
            return "Expert"
        elif performance_vs_meta > 2:
            return "Proficient" 
        elif performance_vs_meta > -2:
            return "Average"
        elif performance_vs_meta > -5:
            return "Developing"
        else:
            return "Struggling"
    
    def _assess_improvement_potential(self, trend: float, confidence: float) -> str:
        """Assess potential for improvement with hero."""
        if confidence < 0.3:
            return "Unknown"
        elif trend > 3:
            return "Rapidly improving"
        elif trend > 1:
            return "Improving"
        elif trend > -1:
            return "Stable"
        elif trend > -3:
            return "Declining"
        else:
            return "Significantly declining"
    
    def _get_personalization_factors(self, user_performance: Dict) -> Dict[str, Any]:
        """Get factors that influenced the personalization."""
        total_games = sum(
            sum(record['total_games'] for record in records)
            for records in user_performance.values()
        )
        
        heroes_played = len(user_performance)
        
        # Calculate overall user winrate
        total_wins = sum(
            sum(record['wins'] for record in records)
            for records in user_performance.values()
        )
        overall_winrate = (total_wins / total_games * 100) if total_games > 0 else 0
        
        # Find user's best and worst heroes
        hero_winrates = {}
        for hero_class, records in user_performance.items():
            wins = sum(record['wins'] for record in records)
            games = sum(record['total_games'] for record in records)
            if games >= 3:  # Only consider heroes with meaningful data
                hero_winrates[hero_class] = (wins / games * 100)
        
        best_hero = max(hero_winrates.items(), key=lambda x: x[1]) if hero_winrates else None
        worst_hero = min(hero_winrates.items(), key=lambda x: x[1]) if hero_winrates else None
        
        return {
            'total_games_tracked': total_games,
            'heroes_played': heroes_played,
            'overall_winrate': round(overall_winrate, 1),
            'best_hero': best_hero[0] if best_hero else None,
            'best_hero_winrate': round(best_hero[1], 1) if best_hero else None,
            'worst_hero': worst_hero[0] if worst_hero else None,
            'worst_hero_winrate': round(worst_hero[1], 1) if worst_hero else None,
            'data_quality': 'Good' if total_games >= 50 else 'Moderate' if total_games >= 20 else 'Limited'
        }
    
    def _calculate_personalized_confidence(self, analysis_list: List[Dict]) -> float:
        """Calculate overall confidence in personalized recommendation."""
        confidences = [analysis['data_confidence'] for analysis in analysis_list]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Factor in score separation
        scores = [analysis['personalized_score'] for analysis in analysis_list]
        max_score = max(scores)
        second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
        score_separation = (max_score - second_max) / max_score if max_score > 0 else 0
        
        # Combine factors
        final_confidence = (avg_confidence * 0.7) + (score_separation * 0.3)
        return round(final_confidence, 2)
    
    def _generate_learning_insights(self, user_performance: Dict, offered_heroes: List[str]) -> List[str]:
        """Generate insights to help user improve."""
        insights = []
        
        try:
            # Analyze offered heroes against user's history
            for hero_class in offered_heroes:
                records = user_performance.get(hero_class, [])
                if records:
                    total_games = sum(record['total_games'] for record in records)
                    if total_games < 10:
                        insights.append(f"More practice needed with {hero_class} (only {total_games} games)")
                
            # Overall insights
            if len(user_performance) < 5:
                insights.append("Try different hero classes to build comprehensive experience")
            
            # Find patterns
            hero_winrates = {}
            for hero_class, records in user_performance.items():
                total_wins = sum(record['wins'] for record in records)
                total_games = sum(record['total_games'] for record in records)
                if total_games >= 5:
                    hero_winrates[hero_class] = total_wins / total_games * 100
            
            if len(hero_winrates) >= 3:
                avg_winrate = sum(hero_winrates.values()) / len(hero_winrates)
                if avg_winrate < 45:
                    insights.append("Focus on fundamentals - draft basics and curve management")
                elif avg_winrate > 65:
                    insights.append("Strong player - experiment with challenging archetypes")
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error generating learning insights: {e}")
            return ["Continue practicing to build performance history"]
    
    def _generate_personalized_explanation(self, recommended_hero: str, analysis: Dict, 
                                         user_performance: Dict) -> str:
        """Generate comprehensive explanation for personalized recommendation."""
        explanation_parts = []
        
        # Main recommendation
        explanation_parts.append(f"Recommended: {recommended_hero}")
        
        # Personal performance context
        if analysis['user_total_games'] >= 5:
            explanation_parts.append(
                f"Your {recommended_hero} winrate: {analysis['user_winrate']:.1f}% "
                f"({analysis['performance_vs_meta']:+.1f}% vs meta)"
            )
        else:
            explanation_parts.append(f"Meta winrate: {analysis['meta_winrate']:.1f}%")
        
        # Trend information
        if abs(analysis.get('recent_trend', 0)) > 1:
            trend = analysis['recent_trend']
            trend_desc = "improving" if trend > 0 else "declining"
            explanation_parts.append(f"Recent trend: {trend_desc} ({trend:+.1f}%)")
        
        # Skill assessment
        skill = analysis.get('skill_assessment', '')
        if skill and skill != 'Insufficient data':
            explanation_parts.append(f"Skill level: {skill}")
        
        return ". ".join(explanation_parts) + "."
    
    def _convert_to_personalized_format(self, standard_rec: HeroRecommendation, reason: str) -> Dict[str, Any]:
        """Convert standard recommendation to personalized format."""
        return {
            'user_id': 'default',
            'recommended_hero_index': standard_rec.recommended_hero_index,
            'hero_classes': standard_rec.hero_classes,
            'personalized_analysis': [],
            'personalization_factors': {},
            'confidence_level': standard_rec.confidence_level,
            'learning_insights': [reason],
            'recommendation_explanation': standard_rec.explanation,
            'fallback_mode': True
        }
    
    def _update_user_preferences(self, user_id: str, hero_class: str, performance_record: Dict) -> None:
        """Update user preferences based on performance."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Track preferred complexity levels based on performance
        profile = self.class_profiles.get(hero_class, {})
        complexity = profile.get('complexity', 'Medium')
        winrate = performance_record['winrate']
        
        if 'complexity_performance' not in self.user_preferences[user_id]:
            self.user_preferences[user_id]['complexity_performance'] = {}
        
        if complexity not in self.user_preferences[user_id]['complexity_performance']:
            self.user_preferences[user_id]['complexity_performance'][complexity] = []
        
        self.user_preferences[user_id]['complexity_performance'][complexity].append(winrate)
    
    # === HSReplay RELIABILITY METHODS ===
    
    def _validate_hsreplay_connection(self) -> bool:
        """Validate HSReplay connection before making API calls."""
        try:
            if not self.hsreplay_scraper:
                self.logger.warning("HSReplay scraper not initialized")
                return False
            
            # Check if scraper has basic connectivity
            if hasattr(self.hsreplay_scraper, 'get_api_status'):
                status = self.hsreplay_scraper.get_api_status()
                if not status.get('available', False):
                    self.logger.warning(f"HSReplay API not available: {status}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"HSReplay connection validation failed: {e}")
            return False
    
    def _fetch_winrates_with_retry(self, force_refresh: bool = False, max_retries: int = 3) -> Optional[Dict[str, float]]:
        """Fetch hero winrates with retry logic and timeout protection."""
        import threading
        import time
        
        def fetch_with_timeout():
            try:
                return self.hsreplay_scraper.get_hero_winrates(force_refresh=force_refresh)
            except Exception as e:
                self.logger.warning(f"HSReplay winrates fetch error: {e}")
                return None
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"HSReplay winrates fetch attempt {attempt + 1}/{max_retries}")
                
                # Use threading for timeout protection
                result = [None]
                thread = threading.Thread(target=lambda: result.__setitem__(0, fetch_with_timeout()))
                thread.daemon = True
                thread.start()
                thread.join(timeout=8.0)  # 8 second timeout
                
                if thread.is_alive():
                    self.logger.warning(f"HSReplay winrates fetch timeout on attempt {attempt + 1}")
                    continue
                
                if result[0]:
                    self.logger.info(f"✅ HSReplay winrates fetched successfully on attempt {attempt + 1}")
                    return result[0]
                else:
                    self.logger.warning(f"❌ HSReplay winrates fetch returned empty on attempt {attempt + 1}")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))  # Progressive backoff
                    
            except Exception as e:
                self.logger.warning(f"HSReplay winrates fetch exception on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
        
        self.logger.error("❌ All HSReplay winrates fetch attempts failed, using fallback")
        return None