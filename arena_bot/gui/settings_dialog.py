"""
Settings Dialog - Comprehensive Configuration Interface

Provides a full-featured settings dialog for configuring hero preferences,
statistical thresholds, and all AI v2 system options with real-time validation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

# Import settings manager
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ai_v2.settings_manager import get_settings_manager, HeroPreferenceProfile


class SettingsDialog:
    """
    Comprehensive settings dialog with tabbed interface.
    
    Provides intuitive configuration for all AI v2 system settings including
    hero preferences, thresholds, advanced options, and UI preferences.
    """
    
    def __init__(self, parent=None, on_settings_changed: Optional[Callable] = None):
        """Initialize settings dialog."""
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.on_settings_changed = on_settings_changed
        
        # Get settings manager
        self.settings_manager = get_settings_manager()
        
        # Track changes
        self.has_changes = False
        self.validation_errors = []
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent) if parent else tk.Tk()
        self.dialog.title("Arena Bot AI v2 Settings")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        
        # Variables for form controls
        self.control_vars = {}
        self.hero_preference_frames = {}
        
        # Setup dialog
        self._setup_dialog()
        self._load_current_settings()
        self._setup_validation()
        
        # Center dialog
        self._center_dialog()
        
        self.logger.info("Settings dialog initialized")
    
    def _setup_dialog(self):
        """Setup the main dialog interface."""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create tabs
        self._create_hero_preferences_tab()
        self._create_statistical_thresholds_tab()
        self._create_advanced_settings_tab()
        self._create_ui_preferences_tab()
        self._create_import_export_tab()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(1, weight=1)
        
        # Buttons
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self._reset_to_defaults).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(button_frame, text="Validate Settings", 
                  command=self._validate_settings).grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="Cancel", 
                  command=self._cancel).grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="Apply", 
                  command=self._apply_settings).grid(row=0, column=3, padx=5)
        
        ttk.Button(button_frame, text="OK", 
                  command=self._ok).grid(row=0, column=4, padx=(5, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _create_hero_preferences_tab(self):
        """Create hero preferences configuration tab."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Hero Preferences")
        
        # Scrollable frame for hero preferences
        canvas = tk.Canvas(tab_frame)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(header_frame, text="Hero Preference Profiles", 
                 font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        
        ttk.Label(header_frame, text="Configure individual hero preferences, archetypes, and auto-selection settings.").pack(anchor="w")
        
        # Hero classes
        hero_classes = ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                       'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']
        
        for hero_class in hero_classes:
            self._create_hero_preference_frame(scrollable_frame, hero_class)
    
    def _create_hero_preference_frame(self, parent, hero_class: str):
        """Create preference configuration frame for a specific hero."""
        # Main frame for this hero
        hero_frame = ttk.LabelFrame(parent, text=f"{hero_class} Preferences", padding="10")
        hero_frame.pack(fill="x", padx=10, pady=5)
        
        self.hero_preference_frames[hero_class] = hero_frame
        
        # Grid layout
        row = 0
        
        # Avoid hero checkbox
        avoid_var = tk.BooleanVar()
        self.control_vars[f"{hero_class}_avoid"] = avoid_var
        ttk.Checkbutton(hero_frame, text="Avoid this hero", 
                       variable=avoid_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        
        # Preferred archetypes
        ttk.Label(hero_frame, text="Preferred Archetypes:").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        
        archetype_frame = ttk.Frame(hero_frame)
        archetype_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        archetypes = ['Aggro', 'Tempo', 'Control', 'Attrition', 'Synergy', 'Balanced']
        archetype_vars = {}
        for i, archetype in enumerate(archetypes):
            var = tk.BooleanVar()
            archetype_vars[archetype] = var
            ttk.Checkbutton(archetype_frame, text=archetype, variable=var).grid(
                row=i//3, column=i%3, sticky="w", padx=(0, 10)
            )
        self.control_vars[f"{hero_class}_archetypes"] = archetype_vars
        row += 1
        
        # Playstyle weight
        ttk.Label(hero_frame, text="Playstyle Weight:").grid(row=row, column=0, sticky="w", pady=(10, 0))
        weight_var = tk.DoubleVar()
        self.control_vars[f"{hero_class}_weight"] = weight_var
        weight_scale = ttk.Scale(hero_frame, from_=0.0, to=2.0, orient="horizontal", 
                               variable=weight_var, length=200)
        weight_scale.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))
        
        weight_label = ttk.Label(hero_frame, text="1.0")
        weight_label.grid(row=row, column=2, padx=(5, 0), pady=(10, 0))
        
        # Update label when scale changes
        def update_weight_label(*args):
            weight_label.config(text=f"{weight_var.get():.1f}")
        weight_var.trace('w', update_weight_label)
        row += 1
        
        # Complexity tolerance
        ttk.Label(hero_frame, text="Complexity Tolerance:").grid(row=row, column=0, sticky="w", pady=(5, 0))
        complexity_var = tk.StringVar()
        self.control_vars[f"{hero_class}_complexity"] = complexity_var
        complexity_combo = ttk.Combobox(hero_frame, textvariable=complexity_var, 
                                      values=["low", "medium", "high"], state="readonly")
        complexity_combo.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(5, 0))
        row += 1
        
        # Auto-select threshold
        ttk.Label(hero_frame, text="Auto-select Threshold (%):").grid(row=row, column=0, sticky="w", pady=(5, 0))
        threshold_var = tk.DoubleVar()
        self.control_vars[f"{hero_class}_threshold"] = threshold_var
        threshold_spin = ttk.Spinbox(hero_frame, from_=0.0, to=20.0, increment=0.5, 
                                   textvariable=threshold_var, width=10)
        threshold_spin.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(5, 0))
        row += 1
        
        # Custom notes
        ttk.Label(hero_frame, text="Custom Notes:").grid(row=row, column=0, sticky="nw", pady=(5, 0))
        notes_var = tk.StringVar()
        self.control_vars[f"{hero_class}_notes"] = notes_var
        notes_entry = ttk.Entry(hero_frame, textvariable=notes_var, width=40)
        notes_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(5, 0))
        
        # Configure column weights
        hero_frame.columnconfigure(1, weight=1)
    
    def _create_statistical_thresholds_tab(self):
        """Create statistical thresholds configuration tab."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Statistical Thresholds")
        
        # Main frame with padding
        main_frame = ttk.Frame(tab_frame, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main_frame, text="Statistical Analysis Thresholds", 
                 font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(main_frame, text="Configure thresholds for statistical analysis and system behavior.").pack(anchor="w", pady=(0, 20))
        
        # Threshold settings
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill="x")
        
        row = 0
        
        # Confidence minimum
        ttk.Label(settings_frame, text="Minimum Confidence Level:").grid(row=row, column=0, sticky="w", pady=5)
        confidence_var = tk.DoubleVar()
        self.control_vars["confidence_minimum"] = confidence_var
        ttk.Scale(settings_frame, from_=0.0, to=1.0, orient="horizontal", 
                 variable=confidence_var, length=200).grid(row=row, column=1, padx=(10, 5), pady=5)
        confidence_label = ttk.Label(settings_frame, text="0.3")
        confidence_label.grid(row=row, column=2, pady=5)
        
        def update_confidence_label(*args):
            confidence_label.config(text=f"{confidence_var.get():.2f}")
        confidence_var.trace('w', update_confidence_label)
        row += 1
        
        # Winrate significance
        ttk.Label(settings_frame, text="Winrate Significance Threshold (%):").grid(row=row, column=0, sticky="w", pady=5)
        winrate_var = tk.DoubleVar()
        self.control_vars["winrate_significance"] = winrate_var
        ttk.Spinbox(settings_frame, from_=0.0, to=10.0, increment=0.1, 
                   textvariable=winrate_var, width=10).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=5)
        row += 1
        
        # Meta stability
        ttk.Label(settings_frame, text="Meta Stability Threshold:").grid(row=row, column=0, sticky="w", pady=5)
        meta_var = tk.DoubleVar()
        self.control_vars["meta_stability_threshold"] = meta_var
        ttk.Scale(settings_frame, from_=0.0, to=1.0, orient="horizontal", 
                 variable=meta_var, length=200).grid(row=row, column=1, padx=(10, 5), pady=5)
        meta_label = ttk.Label(settings_frame, text="0.8")
        meta_label.grid(row=row, column=2, pady=5)
        
        def update_meta_label(*args):
            meta_label.config(text=f"{meta_var.get():.2f}")
        meta_var.trace('w', update_meta_label)
        row += 1
        
        # Personalization minimum games
        ttk.Label(settings_frame, text="Personalization Minimum Games:").grid(row=row, column=0, sticky="w", pady=5)
        games_var = tk.IntVar()
        self.control_vars["personalization_min_games"] = games_var
        ttk.Spinbox(settings_frame, from_=1, to=100, increment=1, 
                   textvariable=games_var, width=10).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=5)
        row += 1
        
        # Cache max age
        ttk.Label(settings_frame, text="Cache Maximum Age (hours):").grid(row=row, column=0, sticky="w", pady=5)
        cache_var = tk.IntVar()
        self.control_vars["cache_max_age_hours"] = cache_var
        ttk.Spinbox(settings_frame, from_=1, to=72, increment=1, 
                   textvariable=cache_var, width=10).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=5)
        row += 1
        
        # API timeout
        ttk.Label(settings_frame, text="API Timeout (seconds):").grid(row=row, column=0, sticky="w", pady=5)
        timeout_var = tk.IntVar()
        self.control_vars["api_timeout_seconds"] = timeout_var
        ttk.Spinbox(settings_frame, from_=1, to=60, increment=1, 
                   textvariable=timeout_var, width=10).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=5)
        row += 1
        
        # Fallback threshold
        ttk.Label(settings_frame, text="Fallback Activation Threshold:").grid(row=row, column=0, sticky="w", pady=5)
        fallback_var = tk.IntVar()
        self.control_vars["fallback_activation_threshold"] = fallback_var
        ttk.Spinbox(settings_frame, from_=1, to=10, increment=1, 
                   textvariable=fallback_var, width=10).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=5)
        
        # Configure column weights
        settings_frame.columnconfigure(1, weight=1)
    
    def _create_advanced_settings_tab(self):
        """Create advanced settings configuration tab."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Advanced Settings")
        
        # Main frame with padding
        main_frame = ttk.Frame(tab_frame, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main_frame, text="Advanced System Settings", 
                 font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(main_frame, text="Configure advanced AI v2 system features and experimental options.").pack(anchor="w", pady=(0, 20))
        
        # Settings frame
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill="x")
        
        # Feature toggles
        features = [
            ("enable_hero_personalization", "Enable Hero Personalization"),
            ("enable_conversational_coach", "Enable Conversational Coach"),
            ("enable_meta_analysis", "Enable Meta Analysis"),
            ("enable_curve_optimization", "Enable Curve Optimization"),
            ("enable_synergy_detection", "Enable Synergy Detection"),
            ("enable_underground_arena_mode", "Enable Underground Arena Mode"),
            ("verbose_explanations", "Verbose Explanations"),
            ("auto_update_preferences", "Auto-Update Preferences"),
            ("experimental_features", "Experimental Features")
        ]
        
        for i, (var_name, label) in enumerate(features):
            var = tk.BooleanVar()
            self.control_vars[var_name] = var
            ttk.Checkbutton(settings_frame, text=label, variable=var).grid(
                row=i//2, column=i%2, sticky="w", padx=(0, 20), pady=5
            )
        
        # Configure column weights
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)
    
    def _create_ui_preferences_tab(self):
        """Create UI preferences configuration tab."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="UI Preferences")
        
        # Main frame with padding
        main_frame = ttk.Frame(tab_frame, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main_frame, text="User Interface Preferences", 
                 font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(main_frame, text="Configure display options and interface behavior.").pack(anchor="w", pady=(0, 20))
        
        # Settings frame
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill="x")
        
        # UI preferences
        ui_options = [
            ("show_confidence_indicators", "Show Confidence Indicators"),
            ("show_winrate_comparisons", "Show Winrate Comparisons"),
            ("show_meta_position", "Show Meta Position"),
            ("show_archetype_suggestions", "Show Archetype Suggestions"),
            ("highlight_recommended_picks", "Highlight Recommended Picks"),
            ("enable_hover_questions", "Enable Hover Questions"),
            ("compact_display_mode", "Compact Display Mode"),
            ("color_code_recommendations", "Color Code Recommendations")
        ]
        
        for i, (var_name, label) in enumerate(ui_options):
            var = tk.BooleanVar()
            self.control_vars[var_name] = var
            ttk.Checkbutton(settings_frame, text=label, variable=var).grid(
                row=i//2, column=i%2, sticky="w", padx=(0, 20), pady=5
            )
        
        # Configure column weights
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)
    
    def _create_import_export_tab(self):
        """Create import/export configuration tab."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Import/Export")
        
        # Main frame with padding
        main_frame = ttk.Frame(tab_frame, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main_frame, text="Settings Import/Export", 
                 font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(main_frame, text="Backup, restore, and share your settings configuration.").pack(anchor="w", pady=(0, 20))
        
        # Export section
        export_frame = ttk.LabelFrame(main_frame, text="Export Settings", padding="10")
        export_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(export_frame, text="Save your current settings to a file for backup or sharing.").pack(anchor="w", pady=(0, 10))
        
        export_button_frame = ttk.Frame(export_frame)
        export_button_frame.pack(fill="x")
        
        ttk.Button(export_button_frame, text="Export All Settings", 
                  command=self._export_settings).pack(side="left", padx=(0, 10))
        
        ttk.Button(export_button_frame, text="Export Hero Preferences Only", 
                  command=lambda: self._export_settings("hero_preferences")).pack(side="left")
        
        # Import section
        import_frame = ttk.LabelFrame(main_frame, text="Import Settings", padding="10")
        import_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(import_frame, text="Load settings from a previously exported file.").pack(anchor="w", pady=(0, 10))
        
        import_button_frame = ttk.Frame(import_frame)
        import_button_frame.pack(fill="x")
        
        ttk.Button(import_button_frame, text="Import Settings (Replace)", 
                  command=lambda: self._import_settings(False)).pack(side="left", padx=(0, 10))
        
        ttk.Button(import_button_frame, text="Import Settings (Merge)", 
                  command=lambda: self._import_settings(True)).pack(side="left")
        
        # Settings summary
        summary_frame = ttk.LabelFrame(main_frame, text="Current Settings Summary", padding="10")
        summary_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        self.summary_text = tk.Text(summary_frame, height=10, wrap="word")
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scrollbar.pack(side="right", fill="y")
        
        # Update summary button
        ttk.Button(summary_frame, text="Refresh Summary", 
                  command=self._update_summary).pack(pady=(10, 0))
        
        # Load initial summary
        self._update_summary()
    
    def _load_current_settings(self):
        """Load current settings into the dialog controls."""
        try:
            # Load hero preferences
            for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                              'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
                preference = self.settings_manager.get_hero_preference(hero_class)
                
                # Avoid hero
                if f"{hero_class}_avoid" in self.control_vars:
                    self.control_vars[f"{hero_class}_avoid"].set(preference.avoid_hero)
                
                # Preferred archetypes
                if f"{hero_class}_archetypes" in self.control_vars:
                    archetype_vars = self.control_vars[f"{hero_class}_archetypes"]
                    for archetype, var in archetype_vars.items():
                        var.set(archetype in preference.preferred_archetypes)
                
                # Other preferences
                if f"{hero_class}_weight" in self.control_vars:
                    self.control_vars[f"{hero_class}_weight"].set(preference.playstyle_weight)
                if f"{hero_class}_complexity" in self.control_vars:
                    self.control_vars[f"{hero_class}_complexity"].set(preference.complexity_tolerance)
                if f"{hero_class}_threshold" in self.control_vars:
                    self.control_vars[f"{hero_class}_threshold"].set(preference.auto_select_threshold)
                if f"{hero_class}_notes" in self.control_vars:
                    self.control_vars[f"{hero_class}_notes"].set(preference.custom_notes)
            
            # Load statistical thresholds
            thresholds = self.settings_manager.get_statistical_thresholds()
            for attr in ['confidence_minimum', 'winrate_significance', 'meta_stability_threshold',
                        'personalization_min_games', 'cache_max_age_hours', 'api_timeout_seconds',
                        'fallback_activation_threshold']:
                if attr in self.control_vars:
                    self.control_vars[attr].set(getattr(thresholds, attr))
            
            # Load advanced settings
            advanced = self.settings_manager.get_advanced_settings()
            for attr in ['enable_hero_personalization', 'enable_conversational_coach', 'enable_meta_analysis',
                        'enable_curve_optimization', 'enable_synergy_detection', 'enable_underground_arena_mode',
                        'verbose_explanations', 'auto_update_preferences', 'experimental_features']:
                if attr in self.control_vars:
                    self.control_vars[attr].set(getattr(advanced, attr))
            
            # Load UI preferences
            ui_prefs = self.settings_manager.get_ui_preferences()
            for attr in ['show_confidence_indicators', 'show_winrate_comparisons', 'show_meta_position',
                        'show_archetype_suggestions', 'highlight_recommended_picks', 'enable_hover_questions',
                        'compact_display_mode', 'color_code_recommendations']:
                if attr in self.control_vars:
                    self.control_vars[attr].set(getattr(ui_prefs, attr))
            
            self.logger.debug("Settings loaded into dialog")
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            messagebox.showerror("Error", f"Failed to load settings: {e}")
    
    def _setup_validation(self):
        """Setup real-time validation for settings."""
        # Add validation traces to relevant variables
        for var_name, var in self.control_vars.items():
            if 'threshold' in var_name or 'weight' in var_name or var_name in [
                'confidence_minimum', 'winrate_significance', 'meta_stability_threshold'
            ]:
                var.trace('w', self._validate_on_change)
    
    def _validate_on_change(self, *args):
        """Validate settings when they change."""
        try:
            errors = self._get_current_validation_errors()
            if errors:
                self.status_var.set(f"Validation issues: {len(errors)}")
            else:
                self.status_var.set("Settings valid")
        except Exception:
            pass  # Don't break UI on validation errors
    
    def _get_current_validation_errors(self) -> List[str]:
        """Get validation errors for current settings."""
        errors = []
        
        try:
            # Check statistical thresholds
            if 'confidence_minimum' in self.control_vars:
                conf = self.control_vars['confidence_minimum'].get()
                if not 0.0 <= conf <= 1.0:
                    errors.append("Confidence minimum must be 0.0-1.0")
            
            if 'winrate_significance' in self.control_vars:
                winrate = self.control_vars['winrate_significance'].get()
                if winrate < 0:
                    errors.append("Winrate significance must be non-negative")
            
            # Check hero preferences
            for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                              'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
                if f"{hero_class}_weight" in self.control_vars:
                    weight = self.control_vars[f"{hero_class}_weight"].get()
                    if not 0.0 <= weight <= 2.0:
                        errors.append(f"{hero_class}: Weight must be 0.0-2.0")
                
                if f"{hero_class}_threshold" in self.control_vars:
                    threshold = self.control_vars[f"{hero_class}_threshold"].get()
                    if not 0.0 <= threshold <= 50.0:
                        errors.append(f"{hero_class}: Threshold must be 0.0-50.0%")
        
        except Exception:
            pass
        
        return errors
    
    def _apply_settings(self):
        """Apply current settings without closing dialog."""
        if self._save_settings():
            self.status_var.set("Settings applied successfully")
            if self.on_settings_changed:
                self.on_settings_changed()
    
    def _save_settings(self) -> bool:
        """Save current dialog settings to settings manager."""
        try:
            # Validate first
            errors = self._get_current_validation_errors()
            if errors:
                messagebox.showerror("Validation Error", 
                                   "Please fix the following issues:\n\n" + "\n".join(errors))
                return False
            
            # Save hero preferences
            for hero_class in ['MAGE', 'PALADIN', 'HUNTER', 'WARRIOR', 'PRIEST', 
                              'WARLOCK', 'ROGUE', 'SHAMAN', 'DRUID', 'DEMONHUNTER']:
                
                # Get current preference
                current_pref = self.settings_manager.get_hero_preference(hero_class)
                
                # Update from dialog
                if f"{hero_class}_avoid" in self.control_vars:
                    current_pref.avoid_hero = self.control_vars[f"{hero_class}_avoid"].get()
                
                if f"{hero_class}_archetypes" in self.control_vars:
                    archetype_vars = self.control_vars[f"{hero_class}_archetypes"]
                    current_pref.preferred_archetypes = [
                        archetype for archetype, var in archetype_vars.items() if var.get()
                    ]
                
                if f"{hero_class}_weight" in self.control_vars:
                    current_pref.playstyle_weight = self.control_vars[f"{hero_class}_weight"].get()
                
                if f"{hero_class}_complexity" in self.control_vars:
                    current_pref.complexity_tolerance = self.control_vars[f"{hero_class}_complexity"].get()
                
                if f"{hero_class}_threshold" in self.control_vars:
                    current_pref.auto_select_threshold = self.control_vars[f"{hero_class}_threshold"].get()
                
                if f"{hero_class}_notes" in self.control_vars:
                    current_pref.custom_notes = self.control_vars[f"{hero_class}_notes"].get()
                
                # Save updated preference
                self.settings_manager.set_hero_preference(hero_class, current_pref)
            
            # Save statistical thresholds
            threshold_updates = {}
            for attr in ['confidence_minimum', 'winrate_significance', 'meta_stability_threshold',
                        'personalization_min_games', 'cache_max_age_hours', 'api_timeout_seconds',
                        'fallback_activation_threshold']:
                if attr in self.control_vars:
                    threshold_updates[attr] = self.control_vars[attr].get()
            
            self.settings_manager.update_statistical_thresholds(**threshold_updates)
            
            # Save advanced settings
            advanced_updates = {}
            for attr in ['enable_hero_personalization', 'enable_conversational_coach', 'enable_meta_analysis',
                        'enable_curve_optimization', 'enable_synergy_detection', 'enable_underground_arena_mode',
                        'verbose_explanations', 'auto_update_preferences', 'experimental_features']:
                if attr in self.control_vars:
                    advanced_updates[attr] = self.control_vars[attr].get()
            
            self.settings_manager.update_advanced_settings(**advanced_updates)
            
            # Save UI preferences
            ui_updates = {}
            for attr in ['show_confidence_indicators', 'show_winrate_comparisons', 'show_meta_position',
                        'show_archetype_suggestions', 'highlight_recommended_picks', 'enable_hover_questions',
                        'compact_display_mode', 'color_code_recommendations']:
                if attr in self.control_vars:
                    ui_updates[attr] = self.control_vars[attr].get()
            
            self.settings_manager.update_ui_preferences(**ui_updates)
            
            # Save to file
            if self.settings_manager.save_settings():
                self.has_changes = False
                return True
            else:
                messagebox.showerror("Error", "Failed to save settings to file")
                return False
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            return False
    
    def _validate_settings(self):
        """Validate current settings and show results."""
        errors = self._get_current_validation_errors()
        
        if errors:
            messagebox.showwarning("Validation Issues", 
                                 f"Found {len(errors)} validation issues:\n\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Validation", "All settings are valid!")
    
    def _reset_to_defaults(self):
        """Reset settings to defaults with confirmation."""
        if messagebox.askyesno("Reset Settings", 
                              "Reset all settings to defaults? This cannot be undone."):
            self.settings_manager.reset_to_defaults()
            self._load_current_settings()
            self.status_var.set("Settings reset to defaults")
    
    def _export_settings(self, component: Optional[str] = None):
        """Export settings to file."""
        try:
            filename = f"arena_bot_settings_{component or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = filedialog.asksaveasfilename(
                title="Export Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialvalue=filename
            )
            
            if file_path:
                if self.settings_manager.export_settings(file_path):
                    messagebox.showinfo("Export Success", f"Settings exported to {file_path}")
                else:
                    messagebox.showerror("Export Error", "Failed to export settings")
                    
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export settings: {e}")
    
    def _import_settings(self, merge: bool):
        """Import settings from file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Import Settings",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                action = "merge with" if merge else "replace"
                if messagebox.askyesno("Import Settings", 
                                     f"Import settings will {action} current settings. Continue?"):
                    
                    if self.settings_manager.import_settings(file_path, merge):
                        self._load_current_settings()
                        self.status_var.set("Settings imported successfully")
                        messagebox.showinfo("Import Success", "Settings imported successfully")
                    else:
                        messagebox.showerror("Import Error", "Failed to import settings")
                        
        except Exception as e:
            self.logger.error(f"Import error: {e}")
            messagebox.showerror("Import Error", f"Failed to import settings: {e}")
    
    def _update_summary(self):
        """Update settings summary display."""
        try:
            summary = self.settings_manager.get_settings_summary()
            
            summary_text = f"""Settings Summary:
            
Hero Preferences: {summary['hero_preferences_count']} configured
Avoided Heroes: {', '.join(summary['avoided_heroes']) if summary['avoided_heroes'] else 'None'}
Personalization: {'Enabled' if summary['personalization_enabled'] else 'Disabled'}
Confidence Threshold: {summary['confidence_threshold']:.2f}
Cache Age: {summary['cache_age_hours']} hours
Experimental Features: {'Enabled' if summary['experimental_features'] else 'Disabled'}

Last Saved: {summary['last_save'] or 'Never'}
Unsaved Changes: {'Yes' if summary['has_unsaved_changes'] else 'No'}
Validation Issues: {summary['validation_issues']}
"""
            
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary_text)
            
        except Exception as e:
            self.logger.error(f"Error updating summary: {e}")
    
    def _center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    def _ok(self):
        """OK button handler - save and close."""
        if self._save_settings():
            self.dialog.destroy()
    
    def _cancel(self):
        """Cancel button handler - close without saving."""
        if self.has_changes:
            if messagebox.askyesno("Unsaved Changes", 
                                 "You have unsaved changes. Close without saving?"):
                self.dialog.destroy()
        else:
            self.dialog.destroy()
    
    def show(self):
        """Show the dialog."""
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.wait_window()


def show_settings_dialog(parent=None, on_settings_changed=None):
    """Show the settings dialog."""
    dialog = SettingsDialog(parent, on_settings_changed)
    dialog.show()