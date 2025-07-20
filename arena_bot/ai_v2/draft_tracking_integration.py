"""
Draft Tracking Integration - Seamless Draft Export Integration

Integrates the draft export system with the main Arena Bot GUI and AI v2 components,
providing automatic draft tracking, data collection, and export management.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

# Import AI v2 components
from .draft_exporter import get_draft_exporter, DraftSummary
from .data_models import DeckState, AIDecision, HeroRecommendation
from .settings_integration import get_settings_integrator

# Import GUI components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gui.draft_export_dialog import show_draft_export_dialog


class DraftTrackingIntegrator:
    """
    Integration layer for automatic draft tracking and export.
    
    Seamlessly integrates with the main GUI to automatically capture
    draft decisions and provide export functionality.
    """
    
    def __init__(self):
        """Initialize draft tracking integrator."""
        self.logger = logging.getLogger(__name__)
        
        # Get components
        self.draft_exporter = get_draft_exporter()
        self.settings_integrator = get_settings_integrator()
        
        # Tracking state
        self.current_draft_active = False
        self.current_draft_id = None
        self.auto_tracking_enabled = True
        self.auto_export_enabled = False
        
        # Draft session data
        self.session_start_time = None
        self.hero_selection_completed = False
        self.card_picks_count = 0
        self.last_deck_state = None
        
        # Callbacks for GUI integration
        self.on_draft_started_callbacks = []
        self.on_draft_completed_callbacks = []
        self.on_export_ready_callbacks = []
        
        # Export settings
        self.auto_export_formats = ['json', 'html']
        self.export_location = None
        
        self.logger.info("DraftTrackingIntegrator initialized")
    
    def register_draft_started_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for when draft starts."""
        self.on_draft_started_callbacks.append(callback)
    
    def register_draft_completed_callback(self, callback: Callable[[DraftSummary], None]) -> None:
        """Register callback for when draft completes."""
        self.on_draft_completed_callbacks.append(callback)
    
    def register_export_ready_callback(self, callback: Callable[[DraftSummary], None]) -> None:
        """Register callback for when export is ready."""
        self.on_export_ready_callbacks.append(callback)
    
    def start_draft_tracking(self, draft_id: Optional[str] = None) -> str:
        """Start tracking a new draft session."""
        if self.current_draft_active:
            self.logger.warning("Draft already active, completing previous draft")
            self._force_complete_current_draft()
        
        # Start new draft
        self.current_draft_id = self.draft_exporter.start_new_draft(draft_id)
        self.current_draft_active = True
        self.session_start_time = datetime.now()
        self.hero_selection_completed = False
        self.card_picks_count = 0
        self.last_deck_state = None
        
        # Notify callbacks
        for callback in self.on_draft_started_callbacks:
            try:
                callback(self.current_draft_id)
            except Exception as e:
                self.logger.error(f"Error in draft started callback: {e}")
        
        self.logger.info(f"Started draft tracking: {self.current_draft_id}")
        return self.current_draft_id
    
    def track_hero_selection(self, hero_recommendation: HeroRecommendation, 
                           user_selected_index: int) -> None:
        """Track hero selection decision."""
        if not self.current_draft_active:
            if self.auto_tracking_enabled:
                self.start_draft_tracking()
            else:
                self.logger.warning("No active draft for hero selection tracking")
                return
        
        try:
            self.draft_exporter.record_hero_selection(hero_recommendation, user_selected_index)
            self.hero_selection_completed = True
            
            self.logger.debug(f"Tracked hero selection: {hero_recommendation.hero_classes[user_selected_index]}")
            
        except Exception as e:
            self.logger.error(f"Error tracking hero selection: {e}")
    
    def track_card_pick(self, ai_decision: AIDecision, deck_state_before: DeckState,
                       deck_state_after: DeckState, user_selected_index: int) -> None:
        """Track card pick decision."""
        if not self.current_draft_active:
            if self.auto_tracking_enabled:
                self.start_draft_tracking()
            else:
                self.logger.warning("No active draft for card pick tracking")
                return
        
        try:
            self.card_picks_count += 1
            
            self.draft_exporter.record_card_pick(
                pick_number=self.card_picks_count,
                ai_decision=ai_decision,
                deck_state_before=deck_state_before,
                deck_state_after=deck_state_after,
                user_selected_index=user_selected_index
            )
            
            self.last_deck_state = deck_state_after
            
            self.logger.debug(f"Tracked card pick {self.card_picks_count}")
            
            # Check if draft is complete (typically 30 cards)
            if self.card_picks_count >= 30:
                self._auto_complete_draft()
            
        except Exception as e:
            self.logger.error(f"Error tracking card pick: {e}")
    
    def complete_draft_tracking(self, final_deck_state: Optional[DeckState] = None) -> Optional[DraftSummary]:
        """Complete the current draft tracking session."""
        if not self.current_draft_active:
            self.logger.warning("No active draft to complete")
            return None
        
        try:
            # Use last known deck state if not provided
            if final_deck_state is None:
                final_deck_state = self.last_deck_state
            
            if final_deck_state is None:
                self.logger.error("No deck state available for draft completion")
                return None
            
            # Complete draft
            draft_summary = self.draft_exporter.complete_draft(final_deck_state)
            
            # Reset tracking state
            self.current_draft_active = False
            self.current_draft_id = None
            
            # Notify callbacks
            for callback in self.on_draft_completed_callbacks:
                try:
                    callback(draft_summary)
                except Exception as e:
                    self.logger.error(f"Error in draft completed callback: {e}")
            
            # Auto-export if enabled
            if self.auto_export_enabled:
                self._auto_export_draft(draft_summary)
            
            # Notify export ready callbacks
            for callback in self.on_export_ready_callbacks:
                try:
                    callback(draft_summary)
                except Exception as e:
                    self.logger.error(f"Error in export ready callback: {e}")
            
            self.logger.info(f"Completed draft tracking: {draft_summary.draft_id}")
            return draft_summary
            
        except Exception as e:
            self.logger.error(f"Error completing draft: {e}")
            return None
    
    def show_export_dialog(self, parent=None, draft_summary: Optional[DraftSummary] = None) -> None:
        """Show the draft export dialog."""
        try:
            # Use current draft summary if none provided
            if draft_summary is None and self.current_draft_active:
                # Get current state for preview
                if self.last_deck_state:
                    draft_summary = self._create_preview_summary()
            
            show_draft_export_dialog(parent, draft_summary)
            
        except Exception as e:
            self.logger.error(f"Error showing export dialog: {e}")
    
    def export_draft_programmatically(self, draft_summary: DraftSummary,
                                    formats: Optional[List[str]] = None,
                                    output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export draft programmatically without GUI."""
        try:
            export_formats = formats or self.auto_export_formats
            export_location = output_dir or self.export_location or "draft_exports"
            
            exported_files = self.draft_exporter.export_draft(
                draft_summary, 
                formats=export_formats,
                output_dir=export_location
            )
            
            self.logger.info(f"Programmatically exported draft: {len(exported_files)} files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error in programmatic export: {e}")
            return {}
    
    def get_current_draft_status(self) -> Dict[str, Any]:
        """Get current draft tracking status."""
        return {
            'draft_active': self.current_draft_active,
            'draft_id': self.current_draft_id,
            'session_start_time': self.session_start_time.isoformat() if self.session_start_time else None,
            'hero_selection_completed': self.hero_selection_completed,
            'card_picks_count': self.card_picks_count,
            'auto_tracking_enabled': self.auto_tracking_enabled,
            'auto_export_enabled': self.auto_export_enabled,
            'export_formats': self.auto_export_formats,
            'export_location': self.export_location
        }
    
    def configure_auto_tracking(self, enabled: bool = True, auto_export: bool = False,
                              export_formats: Optional[List[str]] = None,
                              export_location: Optional[str] = None) -> None:
        """Configure automatic tracking settings."""
        self.auto_tracking_enabled = enabled
        self.auto_export_enabled = auto_export
        
        if export_formats:
            self.auto_export_formats = export_formats
        
        if export_location:
            self.export_location = export_location
        
        self.logger.info(f"Auto-tracking configured: enabled={enabled}, auto_export={auto_export}")
    
    def _auto_complete_draft(self) -> None:
        """Automatically complete draft when conditions are met."""
        try:
            if self.card_picks_count >= 30 and self.last_deck_state:
                self.logger.info("Auto-completing draft (30 cards reached)")
                self.complete_draft_tracking()
                
        except Exception as e:
            self.logger.error(f"Error in auto-complete: {e}")
    
    def _force_complete_current_draft(self) -> None:
        """Force complete current draft (emergency cleanup)."""
        try:
            if self.last_deck_state:
                self.complete_draft_tracking()
            else:
                # Reset state without completion
                self.current_draft_active = False
                self.current_draft_id = None
                self.logger.warning("Force-completed draft without deck state")
                
        except Exception as e:
            self.logger.error(f"Error force-completing draft: {e}")
    
    def _auto_export_draft(self, draft_summary: DraftSummary) -> None:
        """Automatically export completed draft."""
        try:
            exported_files = self.export_draft_programmatically(draft_summary)
            
            if exported_files:
                self.logger.info(f"Auto-exported {len(exported_files)} files")
            else:
                self.logger.warning("Auto-export failed")
                
        except Exception as e:
            self.logger.error(f"Error in auto-export: {e}")
    
    def _create_preview_summary(self) -> Optional[DraftSummary]:
        """Create a preview summary for current draft state."""
        try:
            if not self.current_draft_active or not self.last_deck_state:
                return None
            
            # This would be a partial summary for preview purposes
            # In a real implementation, you'd create a temporary summary
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating preview summary: {e}")
            return None
    
    def get_draft_statistics(self) -> Dict[str, Any]:
        """Get overall draft tracking statistics."""
        stats = self.draft_exporter.get_export_statistics()
        
        stats.update({
            'integration_status': 'active',
            'auto_tracking_enabled': self.auto_tracking_enabled,
            'auto_export_enabled': self.auto_export_enabled,
            'current_session': self.get_current_draft_status()
        })
        
        return stats


# Global draft tracking integrator instance
_draft_tracking_integrator = None

def get_draft_tracking_integrator() -> DraftTrackingIntegrator:
    """Get global draft tracking integrator instance."""
    global _draft_tracking_integrator
    if _draft_tracking_integrator is None:
        _draft_tracking_integrator = DraftTrackingIntegrator()
    return _draft_tracking_integrator


# Convenience functions for GUI integration
def start_draft_session(draft_id: Optional[str] = None) -> str:
    """Start a new draft tracking session."""
    integrator = get_draft_tracking_integrator()
    return integrator.start_draft_tracking(draft_id)


def track_hero_choice(hero_recommendation: HeroRecommendation, user_choice: int) -> None:
    """Track hero selection decision."""
    integrator = get_draft_tracking_integrator()
    integrator.track_hero_selection(hero_recommendation, user_choice)


def track_card_choice(ai_decision: AIDecision, deck_before: DeckState, 
                     deck_after: DeckState, user_choice: int) -> None:
    """Track card pick decision."""
    integrator = get_draft_tracking_integrator()
    integrator.track_card_pick(ai_decision, deck_before, deck_after, user_choice)


def complete_draft_session(final_deck: Optional[DeckState] = None) -> Optional[DraftSummary]:
    """Complete current draft tracking session."""
    integrator = get_draft_tracking_integrator()
    return integrator.complete_draft_tracking(final_deck)


def show_export_interface(parent=None, draft_summary=None) -> None:
    """Show the draft export interface."""
    integrator = get_draft_tracking_integrator()
    integrator.show_export_dialog(parent, draft_summary)


def configure_draft_tracking(auto_track: bool = True, auto_export: bool = False,
                           formats: Optional[List[str]] = None, location: Optional[str] = None) -> None:
    """Configure draft tracking settings."""
    integrator = get_draft_tracking_integrator()
    integrator.configure_auto_tracking(auto_track, auto_export, formats, location)


def get_current_draft_info() -> Dict[str, Any]:
    """Get information about current draft session."""
    integrator = get_draft_tracking_integrator()
    return integrator.get_current_draft_status()