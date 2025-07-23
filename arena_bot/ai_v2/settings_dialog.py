"""
Settings Dialog - Complete GUI Settings Interface

Provides comprehensive settings dialog for configuring all AI v2 system preferences
including hero preferences, statistical thresholds, and advanced options.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Any, Optional, Callable
from .settings_manager import get_settings_manager, HeroPreferenceProfile


class SettingsDialog(tk.Toplevel):
    """
    Complete settings configuration dialog.
    
    Provides tabbed interface for configuring hero preferences,
    statistical thresholds, advanced options, and UI preferences.
    """
    
    def __init__(self, parent, on_settings_changed: Optional[Callable] = None):
        """Initialize settings dialog."""
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.settings_manager = get_settings_manager()
        self.on_settings_changed = on_settings_changed
        
        # Track changes for save/cancel functionality
        self.original_settings = None
        self.changes_made = False
        
        self.setup_dialog()
        self.load_settings()
        
    def setup_dialog(self):
        """Setup the dialog window and UI components."""
        self.title("AI v2 Settings Configuration")
        self.geometry("800x600")
        self.configure(bg='#2C3E50')
        
        # Make dialog modal
        self.transient(self.master)
        self.grab_set()
        
        # Center the dialog
        self.center_window()
        
        # Create main frame
        main_frame = tk.Frame(self, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="⚙️ AI v2 Settings Configuration",
            font=('Arial', 16, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        title_label.pack(pady=(0, 20))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, pady=(0, 20))
        
        # Create tabs
        self.create_hero_preferences_tab()
        self.create_statistical_thresholds_tab()
        self.create_advanced_settings_tab()
        self.create_ui_preferences_tab()
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#2C3E50')
        button_frame.pack(fill='x')
        
        # Buttons
        self.create_buttons(button_frame)
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def create_hero_preferences_tab(self):
        """Create hero preferences configuration tab."""
        frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(frame, text="Hero Preferences")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame, bg='#34495E')
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#34495E')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Hero classes
        hero_classes = [
            'MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST',
            'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER'
        ]
        
        self.hero_widgets = {}
        
        for i, hero_class in enumerate(hero_classes):
            hero_frame = tk.LabelFrame(
                scrollable_frame,
                text=f"{hero_class.title()} Preferences",
                font=('Arial', 10, 'bold'),
                fg='#ECF0F1',
                bg='#34495E'
            )
            hero_frame.pack(fill='x', padx=10, pady=5)
            
            # Create widgets for this hero
            widgets = {}
            
            # Preferred archetypes
            tk.Label(
                hero_frame,
                text="Preferred Archetypes:",
                fg='#ECF0F1',
                bg='#34495E'
            ).grid(row=0, column=0, sticky='w', padx=5, pady=2)
            
            archetype_frame = tk.Frame(hero_frame, bg='#34495E')
            archetype_frame.grid(row=0, column=1, columnspan=2, sticky='w', padx=5)
            
            widgets['archetypes'] = {}
            archetypes = ['Aggro', 'Tempo', 'Control', 'Attrition', 'Synergy', 'Balanced']
            for j, archetype in enumerate(archetypes):
                var = tk.BooleanVar()
                cb = tk.Checkbutton(
                    archetype_frame,
                    text=archetype,
                    variable=var,
                    fg='#ECF0F1',
                    bg='#34495E',
                    selectcolor='#2C3E50'
                )
                cb.grid(row=j//3, column=j%3, sticky='w', padx=5)
                widgets['archetypes'][archetype] = var
            
            # Playstyle weight
            tk.Label(
                hero_frame,
                text="Playstyle Weight:",
                fg='#ECF0F1',
                bg='#34495E'
            ).grid(row=1, column=0, sticky='w', padx=5, pady=2)
            
            widgets['playstyle_weight'] = tk.Scale(
                hero_frame,
                from_=0.5,
                to=2.0,
                resolution=0.1,
                orient='horizontal',
                fg='#ECF0F1',
                bg='#34495E'
            )
            widgets['playstyle_weight'].grid(row=1, column=1, sticky='ew', padx=5)
            
            # Complexity tolerance
            tk.Label(
                hero_frame,
                text="Complexity Tolerance:",
                fg='#ECF0F1',
                bg='#34495E'
            ).grid(row=2, column=0, sticky='w', padx=5, pady=2)
            
            widgets['complexity_tolerance'] = ttk.Combobox(
                hero_frame,
                values=['low', 'medium', 'high'],
                state='readonly'
            )
            widgets['complexity_tolerance'].grid(row=2, column=1, sticky='ew', padx=5)
            
            # Auto-select threshold
            tk.Label(
                hero_frame,
                text="Auto-select Threshold (%):",
                fg='#ECF0F1',
                bg='#34495E'
            ).grid(row=3, column=0, sticky='w', padx=5, pady=2)
            
            widgets['auto_select_threshold'] = tk.Scale(
                hero_frame,
                from_=0.0,
                to=10.0,
                resolution=0.5,
                orient='horizontal',
                fg='#ECF0F1',
                bg='#34495E'
            )
            widgets['auto_select_threshold'].grid(row=3, column=1, sticky='ew', padx=5)
            
            # Avoid hero checkbox
            widgets['avoid_hero'] = tk.BooleanVar()
            avoid_cb = tk.Checkbutton(
                hero_frame,
                text="Avoid this hero",
                variable=widgets['avoid_hero'],
                fg='#E74C3C',
                bg='#34495E',
                selectcolor='#2C3E50'
            )
            avoid_cb.grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=2)
            
            self.hero_widgets[hero_class] = widgets
            
    def create_statistical_thresholds_tab(self):
        """Create statistical thresholds configuration tab."""
        frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(frame, text="Statistical Thresholds")
        
        # Main container
        container = tk.Frame(frame, bg='#34495E')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.threshold_widgets = {}
        
        # Confidence minimum
        tk.Label(
            container,
            text="Minimum Confidence (0.0-1.0):",
            fg='#ECF0F1',
            bg='#34495E',
            font=('Arial', 10)
        ).grid(row=0, column=0, sticky='w', pady=5)
        
        self.threshold_widgets['confidence_minimum'] = tk.Scale(
            container,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient='horizontal',
            fg='#ECF0F1',
            bg='#34495E'
        )
        self.threshold_widgets['confidence_minimum'].grid(row=0, column=1, sticky='ew', padx=10)
        
        # Winrate significance
        tk.Label(
            container,
            text="Winrate Significance (%):",
            fg='#ECF0F1',
            bg='#34495E',
            font=('Arial', 10)
        ).grid(row=1, column=0, sticky='w', pady=5)
        
        self.threshold_widgets['winrate_significance'] = tk.Scale(
            container,
            from_=0.0,
            to=5.0,
            resolution=0.1,
            orient='horizontal',
            fg='#ECF0F1',
            bg='#34495E'
        )
        self.threshold_widgets['winrate_significance'].grid(row=1, column=1, sticky='ew', padx=10)
        
        # Add more threshold controls...
        
        container.columnconfigure(1, weight=1)
        
    def create_advanced_settings_tab(self):
        """Create advanced settings configuration tab."""
        frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(frame, text="Advanced Settings")
        
        container = tk.Frame(frame, bg='#34495E')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.advanced_widgets = {}
        
        # Feature toggles
        features = [
            ('enable_hero_personalization', 'Enable Hero Personalization'),
            ('enable_conversational_coach', 'Enable Conversational Coach'),
            ('enable_meta_analysis', 'Enable Meta Analysis'),
            ('enable_curve_optimization', 'Enable Curve Optimization'),
            ('enable_synergy_detection', 'Enable Synergy Detection'),
            ('enable_underground_arena_mode', 'Enable Underground Arena Mode'),
            ('verbose_explanations', 'Verbose Explanations'),
            ('auto_update_preferences', 'Auto-update Preferences'),
            ('experimental_features', 'Experimental Features')
        ]
        
        for i, (key, label) in enumerate(features):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(
                container,
                text=label,
                variable=var,
                fg='#ECF0F1',
                bg='#34495E',
                selectcolor='#2C3E50',
                font=('Arial', 10)
            )
            cb.grid(row=i, column=0, sticky='w', pady=2)
            self.advanced_widgets[key] = var
            
    def create_ui_preferences_tab(self):
        """Create UI preferences configuration tab."""
        frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(frame, text="UI Preferences")
        
        container = tk.Frame(frame, bg='#34495E')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.ui_widgets = {}
        
        # UI preference toggles
        preferences = [
            ('show_confidence_indicators', 'Show Confidence Indicators'),
            ('show_winrate_comparisons', 'Show Winrate Comparisons'),
            ('show_meta_position', 'Show Meta Position'),
            ('show_archetype_suggestions', 'Show Archetype Suggestions'),
            ('highlight_recommended_picks', 'Highlight Recommended Picks'),
            ('enable_hover_questions', 'Enable Hover Questions'),
            ('compact_display_mode', 'Compact Display Mode'),
            ('color_code_recommendations', 'Color Code Recommendations')
        ]
        
        for i, (key, label) in enumerate(preferences):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(
                container,
                text=label,
                variable=var,
                fg='#ECF0F1',
                bg='#34495E',
                selectcolor='#2C3E50',
                font=('Arial', 10)
            )
            cb.grid(row=i, column=0, sticky='w', pady=2)
            self.ui_widgets[key] = var
            
    def create_buttons(self, parent):
        """Create dialog buttons."""
        # Save button
        save_btn = tk.Button(
            parent,
            text="💾 Save Settings",
            font=('Arial', 10, 'bold'),
            bg='#27AE60',
            fg='white',
            command=self.on_save,
            padx=20
        )
        save_btn.pack(side='right', padx=5)
        
        # Cancel button
        cancel_btn = tk.Button(
            parent,
            text="❌ Cancel",
            font=('Arial', 10),
            bg='#E74C3C',
            fg='white',
            command=self.on_cancel,
            padx=20
        )
        cancel_btn.pack(side='right', padx=5)
        
        # Reset to defaults button
        reset_btn = tk.Button(
            parent,
            text="🔄 Reset to Defaults",
            font=('Arial', 10),
            bg='#F39C12',
            fg='white',
            command=self.on_reset,
            padx=20
        )
        reset_btn.pack(side='left', padx=5)
        
    def load_settings(self):
        """Load current settings into the dialog."""
        try:
            # Load hero preferences
            for hero_class, widgets in self.hero_widgets.items():
                preference = self.settings_manager.get_hero_preference(hero_class)
                
                # Set archetype checkboxes
                for archetype, var in widgets['archetypes'].items():
                    var.set(archetype in preference.preferred_archetypes)
                
                # Set other values
                widgets['playstyle_weight'].set(preference.playstyle_weight)
                widgets['complexity_tolerance'].set(preference.complexity_tolerance)
                widgets['auto_select_threshold'].set(preference.auto_select_threshold)
                widgets['avoid_hero'].set(preference.avoid_hero)
            
            # Load statistical thresholds
            thresholds = self.settings_manager.get_statistical_thresholds()
            self.threshold_widgets['confidence_minimum'].set(thresholds.confidence_minimum)
            self.threshold_widgets['winrate_significance'].set(thresholds.winrate_significance)
            
            # Load advanced settings
            advanced = self.settings_manager.get_advanced_settings()
            for key, widget in self.advanced_widgets.items():
                widget.set(getattr(advanced, key))
            
            # Load UI preferences
            ui_prefs = self.settings_manager.get_ui_preferences()
            for key, widget in self.ui_widgets.items():
                widget.set(getattr(ui_prefs, key))
                
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            messagebox.showerror("Error", f"Failed to load settings: {e}")
            
    def on_save(self):
        """Save settings and close dialog."""
        try:
            # Save hero preferences
            for hero_class, widgets in self.hero_widgets.items():
                # Get selected archetypes
                selected_archetypes = [
                    archetype for archetype, var in widgets['archetypes'].items()
                    if var.get()
                ]
                
                # Create preference profile
                preference = HeroPreferenceProfile(
                    hero_class=hero_class,
                    preferred_archetypes=selected_archetypes,
                    playstyle_weight=widgets['playstyle_weight'].get(),
                    complexity_tolerance=widgets['complexity_tolerance'].get(),
                    auto_select_threshold=widgets['auto_select_threshold'].get(),
                    avoid_hero=widgets['avoid_hero'].get()
                )
                
                self.settings_manager.set_hero_preference(hero_class, preference)
            
            # Save statistical thresholds
            self.settings_manager.update_statistical_thresholds(
                confidence_minimum=self.threshold_widgets['confidence_minimum'].get(),
                winrate_significance=self.threshold_widgets['winrate_significance'].get()
            )
            
            # Save advanced settings
            advanced_values = {
                key: widget.get() for key, widget in self.advanced_widgets.items()
            }
            self.settings_manager.update_advanced_settings(**advanced_values)
            
            # Save UI preferences
            ui_values = {
                key: widget.get() for key, widget in self.ui_widgets.items()
            }
            self.settings_manager.update_ui_preferences(**ui_values)
            
            # Save to file
            if self.settings_manager.save_settings():
                messagebox.showinfo("Success", "Settings saved successfully!")
                
                # Notify parent about changes
                if self.on_settings_changed:
                    self.on_settings_changed()
                
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save settings to file.")
                
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            
    def on_cancel(self):
        """Cancel changes and close dialog."""
        if self.changes_made:
            result = messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to cancel?"
            )
            if not result:
                return
        
        self.destroy()
        
    def on_reset(self):
        """Reset settings to defaults."""
        result = messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults? This cannot be undone."
        )
        
        if result:
            self.settings_manager.reset_to_defaults()
            self.load_settings()
            messagebox.showinfo("Reset Complete", "All settings have been reset to defaults.")
            
    def center_window(self):
        """Center the dialog window on screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")