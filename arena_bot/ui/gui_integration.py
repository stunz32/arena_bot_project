"""
AI Helper v2 - GUI Integration Module
Integration layer for ConversationalCoach and SettingsDialog in the main GUI

This module provides the integration methods to incorporate the ConversationalCoach
and SettingsDialog into the existing IntegratedArenaBotGUI system.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
from typing import Dict, Any, Optional
import threading
import time
from datetime import datetime

from ..ai_v2.conversational_coach import (
    ConversationalCoach, UserProfile, ConversationContext,
    UserSkillLevel, ConversationTone, MessageType
)
from ..ai_v2.data_models import DeckState, ArchetypePreference
from .settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)

class GUIIntegrationMixin:
    """
    Mixin class to add AI Helper integration to the main GUI
    
    This provides methods to integrate ConversationalCoach and SettingsDialog
    into the existing IntegratedArenaBotGUI without breaking existing functionality.
    """
    
    def init_ai_helper_gui_components(self):
        """
        Initialize AI Helper GUI components (Phase 4.3)
        Call this from the main GUI __init__ method
        """
        try:
            # Initialize ConversationalCoach
            self.conversational_coach = ConversationalCoach()
            self.current_user_profile = UserProfile()
            self.chat_session_id = None
            
            # Initialize settings system
            self.settings_manager = None  # Will be created when needed
            self.current_ai_settings = self._get_default_ai_settings()
            
            # GUI state for chat
            self.chat_window = None
            self.chat_history_widget = None
            self.chat_input_widget = None
            self.chat_send_button = None
            self.chat_visible = False
            
            logger.info("AI Helper GUI components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI Helper GUI components: {str(e)}")
            # Don't fail the entire GUI if AI Helper components fail
            self.conversational_coach = None
            self.current_user_profile = None
    
    def _get_default_ai_settings(self) -> Dict[str, Any]:
        """Get default AI Helper settings"""
        return {
            "general": {
                "startup_mode": "Normal",
                "auto_start_monitoring": True,
                "theme": "Dark",
                "language": "English",
                "check_updates": True
            },
            "ai_coaching": {
                "assistance_level": "Balanced",
                "preferred_tone": "Friendly",
                "skill_level": "Intermediate",
                "enable_proactive_tips": True,
                "show_advanced_analysis": False,
                "favorite_archetype": "Balanced",
                "remember_preferences": True,
                "learn_from_feedback": True
            },
            "visual_overlay": {
                "enable_overlay": True,
                "overlay_opacity": 0.8,
                "show_card_scores": True,
                "show_explanations": True,
                "overlay_position": "Top-Right",
                "enable_hover_detection": True,
                "hover_sensitivity": 0.5
            },
            "performance": {
                "max_memory_usage": 100,
                "cpu_usage_limit": 30,
                "enable_performance_monitoring": True,
                "ai_response_timeout": 5
            }
        }
    
    def _create_ai_helper_controls_enhanced(self, parent_frame):
        """
        Enhanced version of the _create_ai_helper_controls method
        Replace the existing method in the main GUI with this
        """
        try:
            # Archetype selection (existing functionality)
            archetype_label = tk.Label(
                parent_frame,
                text="🎯 Archetype:",
                fg='#ECF0F1',
                bg='#2C3E50',
                font=('Arial', 9)
            )
            archetype_label.pack(side='left', padx=(10, 5))
            
            self.archetype_var = tk.StringVar(value="Balanced")
            archetype_options = ["Balanced", "Aggressive", "Control", "Tempo", "Value"]
            archetype_combo = ttk.Combobox(
                parent_frame,
                textvariable=self.archetype_var,
                values=archetype_options,
                state="readonly",
                width=10
            )
            archetype_combo.pack(side='left', padx=5)
            archetype_combo.bind('<<ComboboxSelected>>', self._on_archetype_changed)
            
            # Settings button (enhanced)
            self.settings_btn = tk.Button(
                parent_frame,
                text="⚙️ AI Settings",
                command=self._open_enhanced_settings_dialog,
                bg='#9B59B6',
                fg='white',
                font=('Arial', 9),
                relief='raised',
                bd=2
            )
            self.settings_btn.pack(side='left', padx=5)
            
            # NEW: Chat Coach button
            self.chat_btn = tk.Button(
                parent_frame,
                text="💬 AI Coach",
                command=self._toggle_chat_window,
                bg='#E67E22',
                fg='white',
                font=('Arial', 9),
                relief='raised',
                bd=2
            )
            self.chat_btn.pack(side='left', padx=5)
            
            # NEW: Quick coaching tips button
            self.tips_btn = tk.Button(
                parent_frame,
                text="💡 Tips",
                command=self._show_quick_tips,
                bg='#F39C12',
                fg='white',
                font=('Arial', 9),
                relief='raised',
                bd=2
            )
            self.tips_btn.pack(side='left', padx=5)
            
            logger.info("Enhanced AI Helper controls created")
            
        except Exception as e:
            logger.error(f"Error creating enhanced AI Helper controls: {str(e)}")
    
    def _on_archetype_changed(self, event=None):
        """
        Handle archetype selection change with ConversationalCoach integration
        """
        try:
            new_archetype = self.archetype_var.get()
            
            # Update user profile
            if self.current_user_profile:
                archetype_map = {
                    "Balanced": ArchetypePreference.BALANCED,
                    "Aggressive": ArchetypePreference.AGGRESSIVE,
                    "Control": ArchetypePreference.CONTROL,
                    "Tempo": ArchetypePreference.TEMPO,
                    "Value": ArchetypePreference.VALUE
                }
                
                if new_archetype in archetype_map:
                    if archetype_map[new_archetype] not in self.current_user_profile.favorite_archetypes:
                        self.current_user_profile.favorite_archetypes.append(archetype_map[new_archetype])
                
                # Update preference in AI systems
                if hasattr(self, 'current_deck_state') and self.current_deck_state:
                    self.current_deck_state.archetype_preference = archetype_map.get(new_archetype, ArchetypePreference.BALANCED)
            
            self.log_text(f"🎯 Archetype preference changed to: {new_archetype}")
            
            # Get contextual advice for the new archetype
            if self.conversational_coach and hasattr(self, 'current_deck_state') and self.current_deck_state:
                advice = self.conversational_coach.get_coaching_suggestions(
                    self.current_deck_state, 
                    self.current_user_profile
                )
                if advice:
                    self.log_text(f"💡 Tip: {advice[0]}")
            
        except Exception as e:
            logger.error(f"Error handling archetype change: {str(e)}")
    
    def _open_enhanced_settings_dialog(self):
        """
        Open the enhanced SettingsDialog (Phase 4.2 integration)
        """
        try:
            if not hasattr(self, 'settings_manager') or self.settings_manager is None:
                # Create settings manager on first use
                config_path = "config/ai_helper"
                self.settings_manager = SettingsDialog(
                    parent=self.root,
                    current_settings=self.current_ai_settings,
                    config_path=config_path
                )
            
            self.log_text("⚙️ Opening AI Helper settings...")
            
            # Show settings dialog
            result = self.settings_manager.show()
            
            if result.get('success', False):
                # Apply new settings
                self.current_ai_settings = result['settings']
                self._apply_settings_changes(result['settings'])
                self.log_text("✅ Settings applied successfully!")
            elif result.get('cancelled', False):
                self.log_text("ℹ️ Settings dialog cancelled")
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log_text(f"❌ Settings error: {error_msg}")
                messagebox.showerror("Settings Error", f"Failed to apply settings: {error_msg}")
            
        except Exception as e:
            logger.error(f"Error opening enhanced settings dialog: {str(e)}")
            self.log_text(f"❌ Settings dialog error: {str(e)}")
            messagebox.showerror("Error", f"Failed to open settings: {str(e)}")
    
    def _apply_settings_changes(self, new_settings: Dict[str, Any]):
        """
        Apply settings changes to the AI Helper system
        """
        try:
            # Update user profile based on settings
            if self.current_user_profile:
                ai_settings = new_settings.get('ai_coaching', {})
                
                # Update skill level
                skill_level_str = ai_settings.get('skill_level', 'Intermediate')
                skill_level_map = {
                    'Beginner': UserSkillLevel.BEGINNER,
                    'Intermediate': UserSkillLevel.INTERMEDIATE,
                    'Advanced': UserSkillLevel.ADVANCED,
                    'Expert': UserSkillLevel.EXPERT
                }
                self.current_user_profile.skill_level = skill_level_map.get(skill_level_str, UserSkillLevel.INTERMEDIATE)
                
                # Update conversation tone
                tone_str = ai_settings.get('preferred_tone', 'Friendly')
                tone_map = {
                    'Professional': ConversationTone.PROFESSIONAL,
                    'Friendly': ConversationTone.FRIENDLY,
                    'Casual': ConversationTone.CASUAL,
                    'Encouraging': ConversationTone.ENCOURAGING
                }
                self.current_user_profile.preferred_tone = tone_map.get(tone_str, ConversationTone.FRIENDLY)
                
                # Update other preferences
                self.current_user_profile.show_advanced_explanations = ai_settings.get('show_advanced_analysis', False)
                self.current_user_profile.coaching_frequency = ai_settings.get('assistance_level', 'balanced').lower()
            
            # Update visual overlay settings
            visual_settings = new_settings.get('visual_overlay', {})
            if hasattr(self, 'visual_overlay') and self.visual_overlay:
                # Update overlay opacity, position, etc.
                pass  # Would integrate with actual visual overlay system
            
            # Update ConversationalCoach config
            if self.conversational_coach:
                coach_config = {
                    'max_memory_mb': new_settings.get('performance', {}).get('max_memory_usage', 100),
                    'response_timeout_seconds': new_settings.get('performance', {}).get('ai_response_timeout', 5.0),
                    'enable_learning': new_settings.get('ai_coaching', {}).get('learn_from_feedback', True)
                }
                # Update coach configuration
                self.conversational_coach.config.update(coach_config)
            
            logger.info("Settings changes applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying settings changes: {str(e)}")
    
    def _toggle_chat_window(self):
        """
        Toggle the ConversationalCoach chat window (Phase 4.1 integration)
        """
        try:
            if self.chat_visible and self.chat_window:
                # Hide chat window
                self.chat_window.withdraw()
                self.chat_visible = False
                self.chat_btn.config(text="💬 AI Coach", bg='#E67E22')
                self.log_text("💬 AI Coach chat hidden")
            else:
                # Show or create chat window
                if not self.chat_window:
                    self._create_chat_window()
                else:
                    self.chat_window.deiconify()
                
                self.chat_visible = True
                self.chat_btn.config(text="💬 Hide Coach", bg='#27AE60')
                self.log_text("💬 AI Coach chat opened")
                
                # Show welcome message if first time
                if not hasattr(self, '_chat_initialized'):
                    self._show_welcome_message()
                    self._chat_initialized = True
            
        except Exception as e:
            logger.error(f"Error toggling chat window: {str(e)}")
            self.log_text(f"❌ Chat error: {str(e)}")
    
    def _create_chat_window(self):
        """
        Create the ConversationalCoach chat window
        """
        try:
            # Create chat window
            self.chat_window = tk.Toplevel(self.root)
            self.chat_window.title("💬 AI Draft Coach")
            self.chat_window.geometry("500x600")
            self.chat_window.configure(bg='#2C3E50')
            
            # Make it stay on top but not modal
            self.chat_window.attributes('-topmost', True)
            
            # Position it to the right of main window
            root_x = self.root.winfo_x()
            root_width = self.root.winfo_width()
            self.chat_window.geometry(f"500x600+{root_x + root_width + 10}+100")
            
            # Title bar
            title_frame = tk.Frame(self.chat_window, bg='#34495E', height=40)
            title_frame.pack(fill='x')
            title_frame.pack_propagate(False)
            
            tk.Label(
                title_frame,
                text="🧠 AI Draft Coach",
                font=('Arial', 14, 'bold'),
                fg='#ECF0F1',
                bg='#34495E'
            ).pack(side='left', padx=15, pady=10)
            
            # User skill level indicator
            skill_text = f"Skill: {self.current_user_profile.skill_level.value.title()}" if self.current_user_profile else "Skill: Intermediate"
            tk.Label(
                title_frame,
                text=skill_text,
                font=('Arial', 9),
                fg='#BDC3C7',
                bg='#34495E'
            ).pack(side='right', padx=15, pady=10)
            
            # Chat history area
            history_frame = tk.Frame(self.chat_window, bg='#2C3E50')
            history_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
            
            self.chat_history_widget = scrolledtext.ScrolledText(
                history_frame,
                bg='#1C1C1C',
                fg='#ECF0F1',
                font=('Arial', 10),
                wrap=tk.WORD,
                state='disabled'
            )
            self.chat_history_widget.pack(fill='both', expand=True)
            
            # Input area
            input_frame = tk.Frame(self.chat_window, bg='#2C3E50')
            input_frame.pack(fill='x', padx=10, pady=(0, 10))
            
            # Input text widget
            self.chat_input_widget = tk.Text(
                input_frame,
                height=3,
                bg='#34495E',
                fg='#ECF0F1',
                font=('Arial', 10),
                wrap=tk.WORD
            )
            self.chat_input_widget.pack(fill='x', pady=(0, 5))
            
            # Button frame
            button_frame = tk.Frame(input_frame, bg='#2C3E50')
            button_frame.pack(fill='x')
            
            # Send button
            self.chat_send_button = tk.Button(
                button_frame,
                text="Send 📨",
                command=self._send_chat_message,
                bg='#27AE60',
                fg='white',
                font=('Arial', 10),
                padx=20
            )
            self.chat_send_button.pack(side='right')
            
            # Quick suggestion buttons
            suggestions_frame = tk.Frame(button_frame, bg='#2C3E50')
            suggestions_frame.pack(side='left')
            
            quick_suggestions = [
                ("💡", "Give me tips for this draft phase"),
                ("🤔", "Which card should I pick?"),
                ("📊", "Analyze my deck so far")
            ]
            
            for icon, suggestion in quick_suggestions:
                btn = tk.Button(
                    suggestions_frame,
                    text=icon,
                    command=lambda s=suggestion: self._insert_quick_message(s),
                    bg='#3498DB',
                    fg='white',
                    font=('Arial', 8),
                    width=3
                )
                btn.pack(side='left', padx=2)
            
            # Bind Enter key to send
            self.chat_input_widget.bind('<Control-Return>', lambda e: self._send_chat_message())
            
            # Handle window close
            self.chat_window.protocol("WM_DELETE_WINDOW", self._on_chat_window_close)
            
            logger.info("Chat window created successfully")
            
        except Exception as e:
            logger.error(f"Error creating chat window: {str(e)}")
    
    def _show_welcome_message(self):
        """Show welcome message in chat"""
        welcome_msg = f"""👋 Hello! I'm your AI Draft Coach.

I'm here to help you draft better Arena decks! I can:
• Analyze your current draft choices
• Provide strategic advice based on your deck
• Answer questions about card synergies
• Help you understand draft priorities

Your current skill level is set to: {self.current_user_profile.skill_level.value.title()}

Feel free to ask me anything about your draft! 🎯"""
        
        self._add_chat_message("AI Coach", welcome_msg, MessageType.COACH_RESPONSE)
    
    def _insert_quick_message(self, message: str):
        """Insert a quick suggestion into the chat input"""
        self.chat_input_widget.delete('1.0', tk.END)
        self.chat_input_widget.insert('1.0', message)
        self.chat_input_widget.focus_set()
    
    def _send_chat_message(self):
        """
        Send message to ConversationalCoach and display response
        """
        try:
            # Get user input
            user_input = self.chat_input_widget.get('1.0', tk.END).strip()
            if not user_input:
                return
            
            # Clear input
            self.chat_input_widget.delete('1.0', tk.END)
            
            # Add user message to chat
            self._add_chat_message("You", user_input, MessageType.USER_QUESTION)
            
            # Disable send button while processing
            self.chat_send_button.config(state='disabled', text="Thinking...")
            
            # Process in background thread
            threading.Thread(
                target=self._process_chat_message,
                args=(user_input,),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"Error sending chat message: {str(e)}")
            self._add_chat_message("System", f"Error: {str(e)}", MessageType.SYSTEM_NOTIFICATION)
    
    def _process_chat_message(self, user_input: str):
        """
        Process chat message in background thread
        """
        try:
            if not self.conversational_coach:
                self._add_chat_message("System", "AI Coach is not available", MessageType.SYSTEM_NOTIFICATION)
                return
            
            # Determine conversation context
            context = ConversationContext.GENERAL_COACHING
            if any(word in user_input.lower() for word in ['pick', 'choose', 'better', 'compare']):
                context = ConversationContext.CARD_COMPARISON
            elif any(word in user_input.lower() for word in ['strategy', 'deck', 'archetype']):
                context = ConversationContext.STRATEGY_DISCUSSION
            elif any(word in user_input.lower() for word in ['learn', 'explain', 'why', 'how']):
                context = ConversationContext.LEARNING
            
            # Process with ConversationalCoach
            response = self.conversational_coach.process_user_input(
                user_input=user_input,
                session_id=self.chat_session_id,
                deck_state=getattr(self, 'current_deck_state', None),
                user_profile=self.current_user_profile,
                context=context
            )
            
            # Update session ID if needed
            if not self.chat_session_id:
                self.chat_session_id = response.get('session_id')
            
            # Add response to chat
            response_text = response.get('response', 'Sorry, I couldn\'t process that.')
            message_type = MessageType(response.get('message_type', 'coach_response'))
            
            # Schedule UI update on main thread
            self.root.after(0, self._handle_chat_response, response_text, message_type, response)
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.root.after(0, self._handle_chat_response, error_msg, MessageType.SYSTEM_NOTIFICATION, {})
    
    def _handle_chat_response(self, response_text: str, message_type: MessageType, full_response: Dict[str, Any]):
        """
        Handle chat response on main thread
        """
        try:
            # Add response to chat
            self._add_chat_message("AI Coach", response_text, message_type)
            
            # Add suggestions if available
            suggestions = full_response.get('suggestions', [])
            if suggestions:
                suggestions_text = "💡 You might also ask:\n" + "\n".join(f"• {s}" for s in suggestions[:3])
                self._add_chat_message("AI Coach", suggestions_text, MessageType.SUGGESTION)
            
            # Show any knowledge gaps as learning opportunities
            knowledge_gaps = full_response.get('knowledge_gaps', [])
            if knowledge_gaps:
                gaps_text = "🎓 Learning opportunity:\n" + "\n".join(f"• {gap}" for gap in knowledge_gaps[:2])
                self._add_chat_message("AI Coach", gaps_text, MessageType.EXPLANATION)
            
            # Re-enable send button
            self.chat_send_button.config(state='normal', text="Send 📨")
            
        except Exception as e:
            logger.error(f"Error handling chat response: {str(e)}")
            self.chat_send_button.config(state='normal', text="Send 📨")
    
    def _add_chat_message(self, sender: str, message: str, message_type: MessageType):
        """
        Add a message to the chat history
        """
        try:
            if not self.chat_history_widget:
                return
            
            # Enable editing
            self.chat_history_widget.config(state='normal')
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M")
            
            # Determine message color and icon based on type
            if message_type == MessageType.USER_QUESTION:
                color = '#3498DB'
                icon = '👤'
            elif message_type == MessageType.COACH_RESPONSE:
                color = '#27AE60'
                icon = '🧠'
            elif message_type == MessageType.SUGGESTION:
                color = '#F39C12'
                icon = '💡'
            elif message_type == MessageType.EXPLANATION:
                color = '#9B59B6'
                icon = '🎓'
            else:  # SYSTEM_NOTIFICATION
                color = '#E74C3C'
                icon = '⚠️'
            
            # Add message
            header = f"\n{icon} {sender} ({timestamp})\n"
            self.chat_history_widget.insert(tk.END, header)
            self.chat_history_widget.insert(tk.END, f"{message}\n")
            
            # Add separator
            self.chat_history_widget.insert(tk.END, "─" * 50 + "\n")
            
            # Disable editing and scroll to bottom
            self.chat_history_widget.config(state='disabled')
            self.chat_history_widget.see(tk.END)
            
        except Exception as e:
            logger.error(f"Error adding chat message: {str(e)}")
    
    def _on_chat_window_close(self):
        """Handle chat window close event"""
        self.chat_window.withdraw()
        self.chat_visible = False
        self.chat_btn.config(text="💬 AI Coach", bg='#E67E22')
    
    def _show_quick_tips(self):
        """
        Show quick coaching tips based on current context
        """
        try:
            if not self.conversational_coach:
                messagebox.showinfo("Tips", "AI Coach is not available")
                return
            
            # Get current deck state for contextual tips
            deck_state = getattr(self, 'current_deck_state', None)
            tips = self.conversational_coach.get_coaching_suggestions(deck_state, self.current_user_profile)
            
            if tips:
                tips_text = "💡 Quick Tips:\n\n" + "\n\n".join(f"• {tip}" for tip in tips[:3])
                messagebox.showinfo("AI Coach Tips", tips_text)
            else:
                messagebox.showinfo("Tips", "No specific tips available right now. Try analyzing a draft first!")
            
        except Exception as e:
            logger.error(f"Error showing quick tips: {str(e)}")
            messagebox.showerror("Error", f"Failed to get tips: {str(e)}")
    
    def _integrate_coach_with_analysis(self, analysis_result: Dict[str, Any]):
        """
        Integrate ConversationalCoach with card analysis results
        Call this method when card analysis is completed
        """
        try:
            if not self.conversational_coach or not analysis_result:
                return
            
            # Update current deck state if available
            if hasattr(self, 'current_deck_state') and self.current_deck_state:
                # Get proactive coaching based on the analysis
                suggestions = self.conversational_coach.get_coaching_suggestions(
                    self.current_deck_state,
                    self.current_user_profile
                )
                
                # Show the most relevant suggestion in the log
                if suggestions:
                    self.log_text(f"🧠 AI Coach: {suggestions[0]}")
                
                # If chat is open, add contextual message
                if self.chat_visible and self.chat_history_widget:
                    context_msg = f"I see you're analyzing cards. {suggestions[0] if suggestions else 'Feel free to ask me about your choices!'}"
                    self._add_chat_message("AI Coach", context_msg, MessageType.COACH_RESPONSE)
            
        except Exception as e:
            logger.error(f"Error integrating coach with analysis: {str(e)}")
    
    def update_user_profile_from_interaction(self, feedback_rating: int = None):
        """
        Update user profile based on interactions
        Call this when user provides feedback or makes decisions
        """
        try:
            if not self.current_user_profile:
                return
            
            self.current_user_profile.total_interactions += 1
            self.current_user_profile.last_interaction = datetime.now()
            
            if feedback_rating is not None:
                self.current_user_profile.feedback_ratings.append(feedback_rating)
                
                # Keep only last 50 ratings
                if len(self.current_user_profile.feedback_ratings) > 50:
                    self.current_user_profile.feedback_ratings.pop(0)
            
            # Adaptive skill level adjustment
            self.current_user_profile.update_skill_level_adaptive()
            
            # Update ConversationalCoach with feedback
            if self.conversational_coach and self.chat_session_id:
                self.conversational_coach.update_user_feedback(
                    self.chat_session_id,
                    feedback_rating or 3,
                    "User interaction feedback"
                )
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")


# Helper function to integrate into existing GUI
def integrate_ai_helper_into_gui(gui_instance):
    """
    Helper function to integrate AI Helper components into existing GUI
    
    Args:
        gui_instance: Instance of IntegratedArenaBotGUI
    """
    try:
        # Add mixin methods to the GUI instance
        for method_name in dir(GUIIntegrationMixin):
            if not method_name.startswith('_') or method_name.startswith('_create_') or method_name.startswith('_on_'):
                method = getattr(GUIIntegrationMixin, method_name)
                if callable(method):
                    setattr(gui_instance, method_name, method.__get__(gui_instance))
        
        # Initialize AI Helper components
        gui_instance.init_ai_helper_gui_components()
        
        logger.info("AI Helper integration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error integrating AI Helper into GUI: {str(e)}")
        return False

# Export the integration components
__all__ = [
    'GUIIntegrationMixin',
    'integrate_ai_helper_into_gui'
]