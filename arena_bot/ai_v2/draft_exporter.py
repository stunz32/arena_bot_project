"""
Draft Exporter - Complete Draft Analysis Export System

Provides comprehensive draft export functionality including hero choice reasoning,
card-by-card analysis, strategic insights, and performance tracking with multiple
output formats (JSON, CSV, HTML, PDF).
"""

import logging
import json
import csv
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import base64

# Import for data models
from .data_models import DeckState, AIDecision, HeroRecommendation, DimensionalScores


@dataclass
class CardPickAnalysis:
    """Detailed analysis for a single card pick."""
    pick_number: int
    offered_cards: List[str]
    recommended_card: str
    recommended_index: int
    user_selected_card: str
    user_selected_index: int
    followed_recommendation: bool
    pick_reasoning: str
    alternative_explanations: List[str]
    dimensional_scores: Dict[str, DimensionalScores]
    deck_state_before: Dict[str, Any]
    deck_state_after: Dict[str, Any]
    archetype_impact: Dict[str, Any]
    curve_impact: Dict[str, Any]
    synergy_analysis: Dict[str, Any]
    meta_considerations: Dict[str, Any]
    confidence_level: float
    analysis_time_ms: float
    timestamp: datetime


@dataclass
class HeroChoiceAnalysis:
    """Detailed analysis for hero selection."""
    offered_heroes: List[str]
    recommended_hero: str
    recommended_index: int
    user_selected_hero: str
    user_selected_index: int
    followed_recommendation: bool
    hero_winrates: Dict[str, float]
    hero_analysis: List[Dict[str, Any]]
    selection_reasoning: str
    meta_context: Dict[str, Any]
    confidence_level: float
    timestamp: datetime


@dataclass
class DraftSummary:
    """Complete draft summary and statistics."""
    draft_id: str
    start_time: datetime
    end_time: datetime
    total_duration_minutes: float
    hero_choice: HeroChoiceAnalysis
    card_picks: List[CardPickAnalysis]
    final_deck: Dict[str, Any]
    archetype_evolution: List[Dict[str, Any]]
    curve_analysis: Dict[str, Any]
    synergy_analysis: Dict[str, Any]
    recommendations_followed: int
    total_recommendations: int
    follow_rate_percentage: float
    average_confidence: float
    performance_prediction: Dict[str, Any]
    export_timestamp: datetime
    export_version: str


class DraftExporter:
    """
    Comprehensive draft export system.
    
    Captures complete draft analysis including hero selection, all card picks,
    strategic reasoning, and performance insights with multiple export formats.
    """
    
    def __init__(self):
        """Initialize draft exporter."""
        self.logger = logging.getLogger(__name__)
        
        # Current draft tracking
        self.current_draft = None
        self.current_draft_id = None
        self.draft_start_time = None
        self.hero_analysis = None
        self.card_picks = []
        
        # Export settings
        self.export_formats = ['json', 'csv', 'html', 'txt']
        self.default_export_dir = Path("draft_exports")
        self.default_export_dir.mkdir(exist_ok=True)
        
        # Template loading
        self.html_template = self._load_html_template()
        
        self.logger.info("DraftExporter initialized")
    
    def start_new_draft(self, draft_id: Optional[str] = None) -> str:
        """Start tracking a new draft session."""
        self.current_draft_id = draft_id or f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.draft_start_time = datetime.now()
        self.hero_analysis = None
        self.card_picks = []
        
        self.logger.info(f"Started tracking new draft: {self.current_draft_id}")
        return self.current_draft_id
    
    def record_hero_selection(self, hero_recommendation: HeroRecommendation, 
                            user_selected_index: int) -> None:
        """Record hero selection analysis."""
        try:
            selected_hero = hero_recommendation.hero_classes[user_selected_index]
            recommended_hero = hero_recommendation.hero_classes[hero_recommendation.recommended_hero_index]
            
            self.hero_analysis = HeroChoiceAnalysis(
                offered_heroes=hero_recommendation.hero_classes,
                recommended_hero=recommended_hero,
                recommended_index=hero_recommendation.recommended_hero_index,
                user_selected_hero=selected_hero,
                user_selected_index=user_selected_index,
                followed_recommendation=(user_selected_index == hero_recommendation.recommended_hero_index),
                hero_winrates=hero_recommendation.winrates,
                hero_analysis=hero_recommendation.hero_analysis,
                selection_reasoning=hero_recommendation.explanation,
                meta_context=self._extract_meta_context(hero_recommendation),
                confidence_level=hero_recommendation.confidence_level,
                timestamp=datetime.now()
            )
            
            self.logger.debug(f"Recorded hero selection: {selected_hero}")
            
        except Exception as e:
            self.logger.error(f"Error recording hero selection: {e}")
    
    def record_card_pick(self, pick_number: int, ai_decision: AIDecision, 
                        deck_state_before: DeckState, deck_state_after: DeckState,
                        user_selected_index: int) -> None:
        """Record detailed card pick analysis."""
        try:
            offered_cards = [analysis.get("card_id", f"Card_{i}") for i, analysis in 
                           enumerate(ai_decision.all_offered_cards_analysis)]
            
            recommended_card = offered_cards[ai_decision.recommended_pick_index] if ai_decision.recommended_pick_index < len(offered_cards) else "Unknown"
            user_selected_card = offered_cards[user_selected_index] if user_selected_index < len(offered_cards) else "Unknown"
            
            # Extract dimensional scores for all cards
            dimensional_scores = {}
            for i, analysis in enumerate(ai_decision.all_offered_cards_analysis):
                card_id = analysis.get("card_id", f"Card_{i}")
                scores = analysis.get("scores", {})
                
                # Convert to DimensionalScores if needed
                if isinstance(scores, dict):
                    dimensional_scores[card_id] = DimensionalScores(
                        card_id=card_id,
                        base_value=scores.get("base_value", 0.0),
                        tempo_score=scores.get("tempo_score", 0.0),
                        value_score=scores.get("value_score", 0.0),
                        synergy_score=scores.get("synergy_score", 0.0),
                        curve_score=scores.get("curve_score", 0.0),
                        re_draftability_score=scores.get("re_draftability_score", 0.0),
                        greed_score=scores.get("greed_score", 0.0),
                        confidence=scores.get("confidence", 0.5)
                    )
                else:
                    dimensional_scores[card_id] = scores
            
            # Generate alternative explanations
            alternative_explanations = []
            for i, analysis in enumerate(ai_decision.all_offered_cards_analysis):
                if i != ai_decision.recommended_pick_index:
                    card_id = analysis.get("card_id", f"Card_{i}")
                    explanation = analysis.get("explanation", f"Alternative choice: {card_id}")
                    alternative_explanations.append(explanation)
            
            card_pick = CardPickAnalysis(
                pick_number=pick_number,
                offered_cards=offered_cards,
                recommended_card=recommended_card,
                recommended_index=ai_decision.recommended_pick_index,
                user_selected_card=user_selected_card,
                user_selected_index=user_selected_index,
                followed_recommendation=(user_selected_index == ai_decision.recommended_pick_index),
                pick_reasoning=ai_decision.comparative_explanation,
                alternative_explanations=alternative_explanations,
                dimensional_scores=dimensional_scores,
                deck_state_before=self._serialize_deck_state(deck_state_before),
                deck_state_after=self._serialize_deck_state(deck_state_after),
                archetype_impact=self._analyze_archetype_impact(deck_state_before, deck_state_after),
                curve_impact=self._analyze_curve_impact(deck_state_before, deck_state_after),
                synergy_analysis=self._analyze_synergy_impact(deck_state_before, deck_state_after),
                meta_considerations=self._extract_meta_considerations(ai_decision),
                confidence_level=ai_decision.confidence_level,
                analysis_time_ms=ai_decision.analysis_time_ms,
                timestamp=datetime.now()
            )
            
            self.card_picks.append(card_pick)
            self.logger.debug(f"Recorded card pick {pick_number}: {user_selected_card}")
            
        except Exception as e:
            self.logger.error(f"Error recording card pick {pick_number}: {e}")
    
    def complete_draft(self, final_deck_state: DeckState) -> DraftSummary:
        """Complete the draft and generate comprehensive summary."""
        if not self.current_draft_id:
            raise ValueError("No active draft to complete")
        
        end_time = datetime.now()
        duration = (end_time - self.draft_start_time).total_seconds() / 60
        
        # Calculate statistics
        recommendations_followed = sum(1 for pick in self.card_picks if pick.followed_recommendation)
        if self.hero_analysis and self.hero_analysis.followed_recommendation:
            recommendations_followed += 1
        
        total_recommendations = len(self.card_picks) + (1 if self.hero_analysis else 0)
        follow_rate = (recommendations_followed / total_recommendations * 100) if total_recommendations > 0 else 0
        
        average_confidence = self._calculate_average_confidence()
        
        # Generate comprehensive summary
        draft_summary = DraftSummary(
            draft_id=self.current_draft_id,
            start_time=self.draft_start_time,
            end_time=end_time,
            total_duration_minutes=duration,
            hero_choice=self.hero_analysis,
            card_picks=self.card_picks,
            final_deck=self._serialize_deck_state(final_deck_state),
            archetype_evolution=self._analyze_archetype_evolution(),
            curve_analysis=self._analyze_final_curve(final_deck_state),
            synergy_analysis=self._analyze_final_synergies(final_deck_state),
            recommendations_followed=recommendations_followed,
            total_recommendations=total_recommendations,
            follow_rate_percentage=follow_rate,
            average_confidence=average_confidence,
            performance_prediction=self._generate_performance_prediction(final_deck_state),
            export_timestamp=datetime.now(),
            export_version="1.0"
        )
        
        self.logger.info(f"Completed draft {self.current_draft_id}: {follow_rate:.1f}% follow rate")
        return draft_summary
    
    def export_draft(self, draft_summary: DraftSummary, formats: Optional[List[str]] = None,
                    output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export draft summary in specified formats."""
        export_formats = formats or ['json', 'html']
        output_directory = Path(output_dir) if output_dir else self.default_export_dir
        output_directory.mkdir(exist_ok=True)
        
        exported_files = {}
        
        for format_type in export_formats:
            try:
                if format_type == 'json':
                    file_path = self._export_json(draft_summary, output_directory)
                elif format_type == 'csv':
                    file_path = self._export_csv(draft_summary, output_directory)
                elif format_type == 'html':
                    file_path = self._export_html(draft_summary, output_directory)
                elif format_type == 'txt':
                    file_path = self._export_txt(draft_summary, output_directory)
                else:
                    self.logger.warning(f"Unsupported export format: {format_type}")
                    continue
                
                exported_files[format_type] = str(file_path)
                self.logger.info(f"Exported {format_type}: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error exporting {format_type}: {e}")
        
        return exported_files
    
    def _export_json(self, draft_summary: DraftSummary, output_dir: Path) -> Path:
        """Export draft summary as JSON."""
        file_path = output_dir / f"{draft_summary.draft_id}_complete.json"
        
        # Convert dataclass to dict with datetime serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(draft_summary), f, indent=2, default=serialize_datetime, ensure_ascii=False)
        
        return file_path
    
    def _export_csv(self, draft_summary: DraftSummary, output_dir: Path) -> Path:
        """Export draft summary as CSV (card picks table)."""
        file_path = output_dir / f"{draft_summary.draft_id}_picks.csv"
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Pick Number', 'Recommended Card', 'Selected Card', 'Followed Rec',
                'Confidence', 'Reasoning', 'Base Value', 'Tempo', 'Value', 'Synergy'
            ])
            
            # Card picks
            for pick in draft_summary.card_picks:
                recommended_scores = pick.dimensional_scores.get(pick.recommended_card)
                writer.writerow([
                    pick.pick_number,
                    pick.recommended_card,
                    pick.user_selected_card,
                    'Yes' if pick.followed_recommendation else 'No',
                    f"{pick.confidence_level:.2f}",
                    pick.pick_reasoning[:100] + "..." if len(pick.pick_reasoning) > 100 else pick.pick_reasoning,
                    f"{recommended_scores.base_value:.2f}" if recommended_scores else "N/A",
                    f"{recommended_scores.tempo_score:.2f}" if recommended_scores else "N/A",
                    f"{recommended_scores.value_score:.2f}" if recommended_scores else "N/A",
                    f"{recommended_scores.synergy_score:.2f}" if recommended_scores else "N/A"
                ])
        
        return file_path
    
    def _export_html(self, draft_summary: DraftSummary, output_dir: Path) -> Path:
        """Export draft summary as HTML report."""
        file_path = output_dir / f"{draft_summary.draft_id}_report.html"
        
        # Generate HTML content
        html_content = self._generate_html_report(draft_summary)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    def _export_txt(self, draft_summary: DraftSummary, output_dir: Path) -> Path:
        """Export draft summary as plain text report."""
        file_path = output_dir / f"{draft_summary.draft_id}_summary.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_report(draft_summary))
        
        return file_path
    
    def _generate_html_report(self, draft_summary: DraftSummary) -> str:
        """Generate comprehensive HTML report."""
        # Hero section
        hero_section = ""
        if draft_summary.hero_choice:
            hero = draft_summary.hero_choice
            hero_section = f"""
            <div class="hero-section">
                <h2>Hero Selection</h2>
                <div class="hero-choice">
                    <p><strong>Selected:</strong> {hero.user_selected_hero} {'✓' if hero.followed_recommendation else '✗'}</p>
                    <p><strong>Recommended:</strong> {hero.recommended_hero}</p>
                    <p><strong>Confidence:</strong> {hero.confidence_level:.1%}</p>
                    <p><strong>Reasoning:</strong> {hero.selection_reasoning}</p>
                </div>
                <div class="hero-winrates">
                    <h3>Hero Winrates</h3>
                    <ul>
                    {''.join(f'<li>{hero_class}: {winrate:.1f}%</li>' for hero_class, winrate in hero.hero_winrates.items())}
                    </ul>
                </div>
            </div>
            """
        
        # Card picks section
        picks_section = ""
        for pick in draft_summary.card_picks:
            pick_class = "followed" if pick.followed_recommendation else "not-followed"
            scores = pick.dimensional_scores.get(pick.recommended_card)
            scores_html = ""
            if scores:
                scores_html = f"""
                <div class="scores">
                    <span>Base: {scores.base_value:.2f}</span>
                    <span>Tempo: {scores.tempo_score:.2f}</span>
                    <span>Value: {scores.value_score:.2f}</span>
                    <span>Synergy: {scores.synergy_score:.2f}</span>
                </div>
                """
            
            picks_section += f"""
            <div class="card-pick {pick_class}">
                <h3>Pick {pick.pick_number}</h3>
                <p><strong>Recommended:</strong> {pick.recommended_card}</p>
                <p><strong>Selected:</strong> {pick.user_selected_card} {'✓' if pick.followed_recommendation else '✗'}</p>
                <p><strong>Confidence:</strong> {pick.confidence_level:.1%}</p>
                {scores_html}
                <p class="reasoning">{pick.pick_reasoning}</p>
            </div>
            """
        
        # Statistics section
        stats_section = f"""
        <div class="statistics">
            <h2>Draft Statistics</h2>
            <div class="stats-grid">
                <div class="stat">
                    <h3>Follow Rate</h3>
                    <p class="stat-value">{draft_summary.follow_rate_percentage:.1f}%</p>
                </div>
                <div class="stat">
                    <h3>Average Confidence</h3>
                    <p class="stat-value">{draft_summary.average_confidence:.1%}</p>
                </div>
                <div class="stat">
                    <h3>Duration</h3>
                    <p class="stat-value">{draft_summary.total_duration_minutes:.1f} min</p>
                </div>
                <div class="stat">
                    <h3>Total Picks</h3>
                    <p class="stat-value">{len(draft_summary.card_picks)}</p>
                </div>
            </div>
        </div>
        """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arena Bot Draft Report - {draft_summary.draft_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .hero-section {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .card-pick {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .card-pick.followed {{ border-left: 5px solid #27ae60; }}
                .card-pick.not-followed {{ border-left: 5px solid #e74c3c; }}
                .scores {{ display: flex; gap: 15px; margin: 10px 0; }}
                .scores span {{ background: #3498db; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }}
                .reasoning {{ font-style: italic; color: #7f8c8d; margin-top: 10px; }}
                .statistics {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-top: 15px; }}
                .stat {{ text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 5px 0; }}
                .hero-winrates ul {{ columns: 2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Arena Bot Draft Report</h1>
                <p><strong>Draft ID:</strong> {draft_summary.draft_id}</p>
                <p><strong>Date:</strong> {draft_summary.start_time.strftime('%Y-%m-%d %H:%M')}</p>
                
                {hero_section}
                {stats_section}
                
                <div class="card-picks-section">
                    <h2>Card Pick Analysis</h2>
                    {picks_section}
                </div>
            </div>
        </body>
        </html>
        """
    
    def _generate_text_report(self, draft_summary: DraftSummary) -> str:
        """Generate plain text summary report."""
        lines = [
            "=" * 60,
            f"ARENA BOT DRAFT REPORT - {draft_summary.draft_id}",
            "=" * 60,
            f"Date: {draft_summary.start_time.strftime('%Y-%m-%d %H:%M')}",
            f"Duration: {draft_summary.total_duration_minutes:.1f} minutes",
            ""
        ]
        
        # Hero selection
        if draft_summary.hero_choice:
            hero = draft_summary.hero_choice
            lines.extend([
                "HERO SELECTION:",
                f"  Selected: {hero.user_selected_hero} {'(Recommended)' if hero.followed_recommendation else '(Not Recommended)'}",
                f"  Recommended: {hero.recommended_hero}",
                f"  Confidence: {hero.confidence_level:.1%}",
                f"  Reasoning: {hero.selection_reasoning}",
                ""
            ])
        
        # Statistics
        lines.extend([
            "DRAFT STATISTICS:",
            f"  Recommendations Followed: {draft_summary.recommendations_followed}/{draft_summary.total_recommendations} ({draft_summary.follow_rate_percentage:.1f}%)",
            f"  Average Confidence: {draft_summary.average_confidence:.1%}",
            f"  Total Card Picks: {len(draft_summary.card_picks)}",
            ""
        ])
        
        # Card picks summary
        lines.append("CARD PICK SUMMARY:")
        for pick in draft_summary.card_picks:
            status = "✓" if pick.followed_recommendation else "✗"
            lines.append(f"  Pick {pick.pick_number}: {pick.user_selected_card} {status} (Rec: {pick.recommended_card}, Conf: {pick.confidence_level:.1%})")
        
        lines.extend([
            "",
            "=" * 60,
            f"Report generated: {draft_summary.export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "Arena Bot AI v2 System"
        ])
        
        return "\n".join(lines)
    
    def _serialize_deck_state(self, deck_state: DeckState) -> Dict[str, Any]:
        """Serialize deck state for export."""
        return {
            'hero_class': deck_state.hero_class,
            'drafted_cards': deck_state.drafted_cards,
            'mana_curve': getattr(deck_state, 'mana_curve', {}),
            'archetype_scores': getattr(deck_state, 'archetype_scores', {}),
            'total_cards': len(deck_state.drafted_cards)
        }
    
    def _analyze_archetype_impact(self, before: DeckState, after: DeckState) -> Dict[str, Any]:
        """Analyze how the card pick impacted archetype scores."""
        before_scores = getattr(before, 'archetype_scores', {})
        after_scores = getattr(after, 'archetype_scores', {})
        
        impact = {}
        for archetype in ['Aggro', 'Tempo', 'Control', 'Attrition', 'Synergy', 'Balanced']:
            before_val = before_scores.get(archetype, 0.0)
            after_val = after_scores.get(archetype, 0.0)
            impact[archetype] = after_val - before_val
        
        return impact
    
    def _analyze_curve_impact(self, before: DeckState, after: DeckState) -> Dict[str, Any]:
        """Analyze how the card pick impacted mana curve."""
        before_curve = getattr(before, 'mana_curve', {})
        after_curve = getattr(after, 'mana_curve', {})
        
        return {
            'before': before_curve,
            'after': after_curve,
            'change': {
                cost: after_curve.get(str(cost), 0) - before_curve.get(str(cost), 0)
                for cost in range(0, 8)
            }
        }
    
    def _analyze_synergy_impact(self, before: DeckState, after: DeckState) -> Dict[str, Any]:
        """Analyze synergy impact of card pick."""
        # Placeholder for synergy analysis
        return {
            'tribal_synergies': {},
            'mechanic_synergies': {},
            'combo_potential': 0.0
        }
    
    def _extract_meta_context(self, hero_recommendation: HeroRecommendation) -> Dict[str, Any]:
        """Extract meta context from hero recommendation."""
        return {
            'winrate_spread': max(hero_recommendation.winrates.values()) - min(hero_recommendation.winrates.values()),
            'meta_stability': 'stable',  # Placeholder
            'sample_size': 'adequate'    # Placeholder
        }
    
    def _extract_meta_considerations(self, ai_decision: AIDecision) -> Dict[str, Any]:
        """Extract meta considerations from AI decision."""
        return {
            'meta_tier': 'unknown',
            'popularity': 'unknown',
            'counter_potential': 'unknown'
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all decisions."""
        confidences = []
        
        if self.hero_analysis:
            confidences.append(self.hero_analysis.confidence_level)
        
        for pick in self.card_picks:
            confidences.append(pick.confidence_level)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _analyze_archetype_evolution(self) -> List[Dict[str, Any]]:
        """Analyze how deck archetype evolved throughout draft."""
        evolution = []
        
        for i, pick in enumerate(self.card_picks):
            if 'archetype_scores' in pick.deck_state_after:
                evolution.append({
                    'pick_number': pick.pick_number,
                    'archetype_scores': pick.deck_state_after['archetype_scores']
                })
        
        return evolution
    
    def _analyze_final_curve(self, final_deck: DeckState) -> Dict[str, Any]:
        """Analyze final mana curve."""
        curve = getattr(final_deck, 'mana_curve', {})
        total_cards = sum(curve.values()) if curve else 0
        
        return {
            'curve': curve,
            'total_cards': total_cards,
            'curve_quality': 'good' if total_cards >= 25 else 'incomplete',
            'early_game': sum(curve.get(str(i), 0) for i in range(1, 4)),
            'mid_game': sum(curve.get(str(i), 0) for i in range(4, 7)),
            'late_game': sum(curve.get(str(i), 0) for i in range(7, 10))
        }
    
    def _analyze_final_synergies(self, final_deck: DeckState) -> Dict[str, Any]:
        """Analyze final deck synergies."""
        return {
            'tribal_synergies': {},
            'mechanical_synergies': {},
            'combo_pieces': 0,
            'synergy_rating': 'moderate'
        }
    
    def _generate_performance_prediction(self, final_deck: DeckState) -> Dict[str, Any]:
        """Generate performance prediction for final deck."""
        return {
            'predicted_winrate': 'unknown',
            'archetype_strength': 'moderate',
            'meta_positioning': 'neutral',
            'key_strengths': [],
            'potential_weaknesses': []
        }
    
    def _load_html_template(self) -> str:
        """Load HTML template for reports."""
        # Return basic template - could be loaded from file
        return ""
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export system statistics."""
        return {
            'current_draft_id': self.current_draft_id,
            'current_draft_active': self.current_draft_id is not None,
            'card_picks_recorded': len(self.card_picks),
            'hero_selection_recorded': self.hero_analysis is not None,
            'supported_formats': self.export_formats,
            'export_directory': str(self.default_export_dir)
        }


# Global draft exporter instance
_draft_exporter = None

def get_draft_exporter() -> DraftExporter:
    """Get global draft exporter instance."""
    global _draft_exporter
    if _draft_exporter is None:
        _draft_exporter = DraftExporter()
    return _draft_exporter