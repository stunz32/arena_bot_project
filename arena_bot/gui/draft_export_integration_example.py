"""
Draft Export Integration Example - Main GUI Integration

Example implementation showing how to integrate the complete draft export
functionality with the main Arena Bot GUI system.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import AI v2 components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ai_v2.draft_tracking_integration import (
    get_draft_tracking_integrator, start_draft_session, track_hero_choice,
    track_card_choice, complete_draft_session, show_export_interface,
    configure_draft_tracking, get_current_draft_info
)
from ai_v2.data_models import DeckState, AIDecision, HeroRecommendation
from gui.draft_export_dialog import show_draft_export_dialog


class DraftExportIntegratedGUI:
    """
    Example implementation of main Arena Bot GUI with integrated draft export.
    
    Shows how to seamlessly integrate draft tracking and export functionality
    into the existing Arena Bot interface.
    """
    
    def __init__(self):
        """Initialize the integrated GUI."""
        self.logger = logging.getLogger(__name__)
        
        # Get draft tracking integrator
        self.draft_integrator = get_draft_tracking_integrator()
        
        # Setup callbacks
        self._setup_draft_callbacks()
        
        # Configure draft tracking
        configure_draft_tracking(
            auto_track=True,
            auto_export=False,
            formats=['json', 'html'],
            location='draft_exports'
        )
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Arena Bot AI v2 with Draft Export")
        self.root.geometry("900x700")
        
        # Setup GUI
        self._setup_gui()
        
        # Draft state
        self.current_draft_summary = None
        self.draft_in_progress = False
        
        self.logger.info("Draft Export Integrated GUI initialized")
    
    def _setup_draft_callbacks(self):
        """Setup callbacks for draft tracking events."""
        self.draft_integrator.register_draft_started_callback(self._on_draft_started)
        self.draft_integrator.register_draft_completed_callback(self._on_draft_completed)
        self.draft_integrator.register_export_ready_callback(self._on_export_ready)
    
    def _setup_gui(self):
        """Setup the main GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Setup sections
        self._create_header_section(main_frame)
        self._create_main_content_section(main_frame)
        self._create_draft_tracking_section(main_frame)
        self._create_export_section(main_frame)
        self._create_status_section(main_frame)
    
    def _create_header_section(self, parent):
        """Create header section with draft controls."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(header_frame, text="Arena Bot AI v2", 
                               font=("TkDefaultFont", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="w")
        
        # Draft controls
        draft_controls = ttk.Frame(header_frame)
        draft_controls.grid(row=0, column=1, sticky="e")
        
        ttk.Button(draft_controls, text="Start New Draft", 
                  command=self._start_new_draft).pack(side="left", padx=(0, 5))
        
        ttk.Button(draft_controls, text="Complete Draft", 
                  command=self._complete_draft).pack(side="left", padx=5)
        
        ttk.Button(draft_controls, text="Export Draft", 
                  command=self._show_export_dialog).pack(side="left", padx=(5, 0))
    
    def _create_main_content_section(self, parent):
        """Create main content area."""
        content_frame = ttk.LabelFrame(parent, text="Draft Session", padding="10")
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Hero selection tab
        self._create_hero_selection_tab()
        
        # Card selection tab
        self._create_card_selection_tab()
        
        # Deck view tab
        self._create_deck_view_tab()
    
    def _create_hero_selection_tab(self):
        """Create hero selection interface."""
        hero_frame = ttk.Frame(self.notebook)
        self.notebook.add(hero_frame, text="Hero Selection")
        
        # Hero selection simulation
        ttk.Label(hero_frame, text="Hero Selection (Simulation)", 
                 font=("TkDefaultFont", 12, "bold")).pack(pady=10)
        
        # Example hero choices
        self.hero_vars = {}
        hero_classes = ['MAGE', 'WARRIOR', 'PALADIN']
        
        for i, hero_class in enumerate(hero_classes):
            var = tk.BooleanVar()
            self.hero_vars[hero_class] = var
            
            frame = ttk.Frame(hero_frame)
            frame.pack(fill="x", padx=20, pady=5)
            
            ttk.Radiobutton(frame, text=f"{hero_class} (52.{i+1}% winrate)", 
                           variable=var, value=True).pack(side="left")
            
            if i == 0:  # Default selection
                var.set(True)
        
        ttk.Button(hero_frame, text="Confirm Hero Selection", 
                  command=self._simulate_hero_selection).pack(pady=20)
    
    def _create_card_selection_tab(self):
        """Create card selection interface."""
        card_frame = ttk.Frame(self.notebook)
        self.notebook.add(card_frame, text="Card Selection")
        
        # Card selection simulation
        ttk.Label(card_frame, text="Card Selection (Simulation)", 
                 font=("TkDefaultFont", 12, "bold")).pack(pady=10)
        
        # Pick counter
        self.pick_counter_var = tk.StringVar()
        self.pick_counter_var.set("Pick 1 of 30")
        ttk.Label(card_frame, textvariable=self.pick_counter_var, 
                 font=("TkDefaultFont", 10)).pack(pady=5)
        
        # Example card choices
        self.card_vars = {}
        example_cards = ['Fireball', 'Flamestrike', 'Arcane Intellect']
        
        for i, card_name in enumerate(example_cards):
            var = tk.BooleanVar()
            self.card_vars[card_name] = var
            
            frame = ttk.Frame(card_frame)
            frame.pack(fill="x", padx=20, pady=5)
            
            ttk.Radiobutton(frame, text=f"{card_name} (Score: {75-i*5})", 
                           variable=var, value=True).pack(side="left")
            
            if i == 0:  # Default selection
                var.set(True)
        
        ttk.Button(card_frame, text="Confirm Card Pick", 
                  command=self._simulate_card_pick).pack(pady=20)
        
        # Quick draft button for testing
        ttk.Button(card_frame, text="Quick Draft (30 cards)", 
                  command=self._simulate_full_draft).pack(pady=10)
    
    def _create_deck_view_tab(self):
        """Create deck view interface."""
        deck_frame = ttk.Frame(self.notebook)
        self.notebook.add(deck_frame, text="Deck View")
        
        # Deck display
        ttk.Label(deck_frame, text="Current Deck", 
                 font=("TkDefaultFont", 12, "bold")).pack(pady=10)
        
        self.deck_listbox = tk.Listbox(deck_frame, height=15)
        self.deck_listbox.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Deck stats
        stats_frame = ttk.Frame(deck_frame)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        self.deck_stats_var = tk.StringVar()
        self.deck_stats_var.set("Deck: 0/30 cards")
        ttk.Label(stats_frame, textvariable=self.deck_stats_var).pack()
    
    def _create_draft_tracking_section(self, parent):
        """Create draft tracking status section."""
        tracking_frame = ttk.LabelFrame(parent, text="Draft Tracking", padding="10")
        tracking_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        tracking_frame.columnconfigure(1, weight=1)
        
        # Status indicators
        ttk.Label(tracking_frame, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.tracking_status_var = tk.StringVar()
        self.tracking_status_var.set("No active draft")
        ttk.Label(tracking_frame, textvariable=self.tracking_status_var).grid(row=0, column=1, sticky="w")
        
        ttk.Label(tracking_frame, text="Progress:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.tracking_progress_var = tk.StringVar()
        self.tracking_progress_var.set("0 decisions tracked")
        ttk.Label(tracking_frame, textvariable=self.tracking_progress_var).grid(row=1, column=1, sticky="w")
        
        # Tracking controls
        tracking_controls = ttk.Frame(tracking_frame)
        tracking_controls.grid(row=0, column=2, rowspan=2, sticky="e")
        
        ttk.Button(tracking_controls, text="View Progress", 
                  command=self._show_tracking_progress).pack(side="left", padx=(0, 5))
        
        ttk.Button(tracking_controls, text="Settings", 
                  command=self._show_tracking_settings).pack(side="left")
    
    def _create_export_section(self, parent):
        """Create export controls section."""
        export_frame = ttk.LabelFrame(parent, text="Draft Export", padding="10")
        export_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        export_frame.columnconfigure(1, weight=1)
        
        # Export status
        ttk.Label(export_frame, text="Last Export:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.export_status_var = tk.StringVar()
        self.export_status_var.set("No exports yet")
        ttk.Label(export_frame, textvariable=self.export_status_var).grid(row=0, column=1, sticky="w")
        
        # Export controls
        export_controls = ttk.Frame(export_frame)
        export_controls.grid(row=0, column=2, sticky="e")
        
        ttk.Button(export_controls, text="Quick Export", 
                  command=self._quick_export).pack(side="left", padx=(0, 5))
        
        ttk.Button(export_controls, text="Export Options", 
                  command=self._show_export_dialog).pack(side="left")
    
    def _create_status_section(self, parent):
        """Create status bar."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E))
    
    def _start_new_draft(self):
        """Start a new draft session."""
        try:
            if self.draft_in_progress:
                result = messagebox.askyesno("Active Draft", 
                    "A draft is already in progress. Start a new one?")
                if not result:
                    return
            
            draft_id = start_draft_session()
            self.draft_in_progress = True
            self.notebook.select(0)  # Switch to hero selection
            
            # Clear previous data
            self.deck_listbox.delete(0, tk.END)
            self._update_deck_stats(0)
            
            self.status_var.set(f"Started new draft: {draft_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting draft: {e}")
            messagebox.showerror("Error", f"Failed to start draft: {e}")
    
    def _complete_draft(self):
        """Complete the current draft."""
        try:
            if not self.draft_in_progress:
                messagebox.showinfo("No Draft", "No draft in progress to complete.")
                return
            
            # Create a simple deck state for completion
            deck_state = DeckState(
                hero_class="MAGE",  # Example
                drafted_cards=["card_" + str(i) for i in range(30)]  # Example
            )
            
            draft_summary = complete_draft_session(deck_state)
            
            if draft_summary:
                self.current_draft_summary = draft_summary
                self.draft_in_progress = False
                self.status_var.set("Draft completed successfully")
                
                # Ask if user wants to export
                result = messagebox.askyesno("Draft Complete", 
                    "Draft completed! Would you like to export the analysis?")
                if result:
                    self._show_export_dialog()
            else:
                messagebox.showerror("Error", "Failed to complete draft")
                
        except Exception as e:
            self.logger.error(f"Error completing draft: {e}")
            messagebox.showerror("Error", f"Failed to complete draft: {e}")
    
    def _simulate_hero_selection(self):
        """Simulate hero selection for demonstration."""
        try:
            # Find selected hero
            selected_hero = None
            selected_index = 0
            
            for i, (hero_class, var) in enumerate(self.hero_vars.items()):
                if var.get():
                    selected_hero = hero_class
                    selected_index = i
                    break
            
            if selected_hero:
                # Create mock hero recommendation
                hero_recommendation = HeroRecommendation(
                    recommended_hero_index=0,
                    hero_classes=list(self.hero_vars.keys()),
                    hero_analysis=[],
                    explanation=f"Recommended {selected_hero} based on meta analysis",
                    winrates={hero: 52.0 + i for i, hero in enumerate(self.hero_vars.keys())},
                    confidence_level=0.85
                )
                
                # Track the selection
                track_hero_choice(hero_recommendation, selected_index)
                
                # Move to card selection
                self.notebook.select(1)
                self.status_var.set(f"Selected hero: {selected_hero}")
            
        except Exception as e:
            self.logger.error(f"Error in hero selection: {e}")
            messagebox.showerror("Error", f"Hero selection failed: {e}")
    
    def _simulate_card_pick(self):
        """Simulate a single card pick."""
        try:
            # Find selected card
            selected_card = None
            selected_index = 0
            
            for i, (card_name, var) in enumerate(self.card_vars.items()):
                if var.get():
                    selected_card = card_name
                    selected_index = i
                    break
                var.set(False)  # Reset for next pick
            
            if selected_card:
                # Create mock AI decision
                ai_decision = AIDecision(
                    recommended_pick_index=0,
                    all_offered_cards_analysis=[
                        {"card_id": card, "scores": {"base_value": 70 - i*5}}
                        for i, card in enumerate(self.card_vars.keys())
                    ],
                    comparative_explanation=f"Recommended {selected_card} for tempo value",
                    deck_analysis={},
                    card_coordinates=[],
                    confidence_level=0.78,
                    analysis_time_ms=150.0
                )
                
                # Create mock deck states
                current_cards = self.deck_listbox.size()
                deck_before = DeckState(
                    hero_class="MAGE",
                    drafted_cards=[f"card_{i}" for i in range(current_cards)]
                )
                deck_after = DeckState(
                    hero_class="MAGE", 
                    drafted_cards=[f"card_{i}" for i in range(current_cards + 1)]
                )
                
                # Track the pick
                track_card_choice(ai_decision, deck_before, deck_after, selected_index)
                
                # Update UI
                self.deck_listbox.insert(tk.END, f"Pick {current_cards + 1}: {selected_card}")
                self._update_deck_stats(current_cards + 1)
                self.pick_counter_var.set(f"Pick {current_cards + 2} of 30")
                
                # Reset card selection
                list(self.card_vars.values())[0].set(True)
                
                self.status_var.set(f"Picked: {selected_card}")
                
                # Auto-complete if 30 cards
                if current_cards + 1 >= 30:
                    self._complete_draft()
            
        except Exception as e:
            self.logger.error(f"Error in card pick: {e}")
            messagebox.showerror("Error", f"Card pick failed: {e}")
    
    def _simulate_full_draft(self):
        """Simulate a complete 30-card draft quickly."""
        try:
            if not self.draft_in_progress:
                self._start_new_draft()
                self._simulate_hero_selection()
            
            # Simulate 30 card picks
            for i in range(30):
                self._simulate_card_pick()
                self.root.update()  # Update UI
            
            self.status_var.set("Quick draft completed")
            
        except Exception as e:
            self.logger.error(f"Error in quick draft: {e}")
            messagebox.showerror("Error", f"Quick draft failed: {e}")
    
    def _update_deck_stats(self, card_count: int):
        """Update deck statistics display."""
        self.deck_stats_var.set(f"Deck: {card_count}/30 cards")
    
    def _show_export_dialog(self):
        """Show the draft export dialog."""
        try:
            show_export_interface(self.root, self.current_draft_summary)
            
        except Exception as e:
            self.logger.error(f"Error showing export dialog: {e}")
            messagebox.showerror("Error", f"Failed to show export dialog: {e}")
    
    def _quick_export(self):
        """Perform a quick export with default settings."""
        try:
            if not self.current_draft_summary:
                messagebox.showinfo("No Data", "No completed draft available for export.")
                return
            
            exported_files = self.draft_integrator.export_draft_programmatically(
                self.current_draft_summary,
                formats=['json', 'html'],
                output_dir='draft_exports'
            )
            
            if exported_files:
                self.export_status_var.set(f"Exported {len(exported_files)} files")
                self.status_var.set("Quick export completed")
                messagebox.showinfo("Export Complete", 
                    f"Exported {len(exported_files)} files to draft_exports/")
            else:
                messagebox.showerror("Export Failed", "Failed to export draft")
                
        except Exception as e:
            self.logger.error(f"Error in quick export: {e}")
            messagebox.showerror("Error", f"Quick export failed: {e}")
    
    def _show_tracking_progress(self):
        """Show draft tracking progress."""
        try:
            status = get_current_draft_info()
            
            progress_text = f"""Draft Tracking Progress:
            
Draft Active: {status['draft_active']}
Draft ID: {status['draft_id'] or 'None'}
Hero Selection: {'Complete' if status['hero_selection_completed'] else 'Pending'}
Card Picks: {status['card_picks_count']}
Auto Tracking: {'Enabled' if status['auto_tracking_enabled'] else 'Disabled'}
Auto Export: {'Enabled' if status['auto_export_enabled'] else 'Disabled'}
"""
            
            messagebox.showinfo("Tracking Progress", progress_text)
            
        except Exception as e:
            self.logger.error(f"Error showing tracking progress: {e}")
            messagebox.showerror("Error", f"Failed to show progress: {e}")
    
    def _show_tracking_settings(self):
        """Show tracking settings dialog."""
        messagebox.showinfo("Settings", "Tracking settings dialog would open here")
    
    def _on_draft_started(self, draft_id: str):
        """Callback for when draft starts."""
        self.tracking_status_var.set(f"Draft active: {draft_id}")
        self._update_tracking_display()
    
    def _on_draft_completed(self, draft_summary):
        """Callback for when draft completes."""
        self.tracking_status_var.set("Draft completed")
        self.current_draft_summary = draft_summary
        self._update_tracking_display()
    
    def _on_export_ready(self, draft_summary):
        """Callback for when export is ready."""
        self.export_status_var.set("Ready for export")
    
    def _update_tracking_display(self):
        """Update tracking progress display."""
        status = get_current_draft_info()
        if status['draft_active']:
            total_decisions = 1 if status['hero_selection_completed'] else 0
            total_decisions += status['card_picks_count']
            self.tracking_progress_var.set(f"{total_decisions} decisions tracked")
        else:
            self.tracking_progress_var.set("0 decisions tracked")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the integrated GUI example."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run GUI
    app = DraftExportIntegratedGUI()
    app.run()


if __name__ == "__main__":
    main()