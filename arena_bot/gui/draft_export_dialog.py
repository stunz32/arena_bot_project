"""
Draft Export Dialog - User Interface for Draft Analysis Export

Provides an intuitive interface for exporting complete draft analysis with
format selection, preview functionality, and batch export capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import webbrowser
import subprocess
import sys

# Import draft exporter
sys.path.append(str(Path(__file__).parent.parent))
from ai_v2.draft_exporter import get_draft_exporter, DraftSummary


class DraftExportDialog:
    """
    Comprehensive draft export interface.
    
    Provides format selection, preview options, export location selection,
    and post-export actions for complete draft analysis.
    """
    
    def __init__(self, parent=None, draft_summary: Optional[DraftSummary] = None):
        """Initialize draft export dialog."""
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.draft_summary = draft_summary
        
        # Get draft exporter
        self.draft_exporter = get_draft_exporter()
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent) if parent else tk.Tk()
        self.dialog.title("Export Draft Analysis")
        self.dialog.geometry("700x500")
        self.dialog.resizable(True, True)
        
        # Export settings
        self.selected_formats = []
        self.export_location = tk.StringVar()
        self.include_hero_analysis = tk.BooleanVar(value=True)
        self.include_card_analysis = tk.BooleanVar(value=True)
        self.include_statistics = tk.BooleanVar(value=True)
        self.include_performance_prediction = tk.BooleanVar(value=True)
        
        # Export results
        self.exported_files = {}
        
        # Setup dialog
        self._setup_dialog()
        self._load_default_settings()
        
        # Center dialog
        self._center_dialog()
        
        self.logger.info("Draft export dialog initialized")
    
    def _setup_dialog(self):
        """Setup the main dialog interface."""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Create sections
        self._create_draft_info_section(main_frame)
        self._create_export_options_section(main_frame)
        self._create_preview_section(main_frame)
        self._create_button_section(main_frame)
    
    def _create_draft_info_section(self, parent):
        """Create draft information display section."""
        info_frame = ttk.LabelFrame(parent, text="Draft Information", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        if self.draft_summary:
            # Draft ID
            ttk.Label(info_frame, text="Draft ID:").grid(row=0, column=0, sticky="w", padx=(0, 10))
            ttk.Label(info_frame, text=self.draft_summary.draft_id).grid(row=0, column=1, sticky="w")
            
            # Date and duration
            ttk.Label(info_frame, text="Date:").grid(row=1, column=0, sticky="w", padx=(0, 10))
            date_text = self.draft_summary.start_time.strftime("%Y-%m-%d %H:%M")
            ttk.Label(info_frame, text=date_text).grid(row=1, column=1, sticky="w")
            
            ttk.Label(info_frame, text="Duration:").grid(row=2, column=0, sticky="w", padx=(0, 10))
            duration_text = f"{self.draft_summary.total_duration_minutes:.1f} minutes"
            ttk.Label(info_frame, text=duration_text).grid(row=2, column=1, sticky="w")
            
            # Hero and recommendations
            if self.draft_summary.hero_choice:
                ttk.Label(info_frame, text="Hero:").grid(row=3, column=0, sticky="w", padx=(0, 10))
                ttk.Label(info_frame, text=self.draft_summary.hero_choice.user_selected_hero).grid(row=3, column=1, sticky="w")
            
            ttk.Label(info_frame, text="Follow Rate:").grid(row=4, column=0, sticky="w", padx=(0, 10))
            follow_text = f"{self.draft_summary.follow_rate_percentage:.1f}% ({self.draft_summary.recommendations_followed}/{self.draft_summary.total_recommendations})"
            ttk.Label(info_frame, text=follow_text).grid(row=4, column=1, sticky="w")
        else:
            ttk.Label(info_frame, text="No draft data available", 
                     font=("TkDefaultFont", 10, "italic")).grid(row=0, column=0, columnspan=2)
    
    def _create_export_options_section(self, parent):
        """Create export options configuration section."""
        options_frame = ttk.LabelFrame(parent, text="Export Options", padding="10")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        
        # Format selection
        ttk.Label(options_frame, text="Export Formats:").grid(row=0, column=0, sticky="nw", padx=(0, 10))
        
        format_frame = ttk.Frame(options_frame)
        format_frame.grid(row=0, column=1, sticky="w")
        
        self.format_vars = {}
        formats = [
            ("json", "JSON (Complete Data)"),
            ("html", "HTML (Visual Report)"),
            ("csv", "CSV (Card Picks Table)"),
            ("txt", "Text (Summary)")
        ]
        
        for i, (format_key, format_label) in enumerate(formats):
            var = tk.BooleanVar()
            self.format_vars[format_key] = var
            ttk.Checkbutton(format_frame, text=format_label, variable=var).grid(
                row=i//2, column=i%2, sticky="w", padx=(0, 20), pady=2
            )
        
        # Set default selections
        self.format_vars["json"].set(True)
        self.format_vars["html"].set(True)
        
        # Content options
        ttk.Label(options_frame, text="Include Content:").grid(row=1, column=0, sticky="nw", padx=(0, 10), pady=(10, 0))
        
        content_frame = ttk.Frame(options_frame)
        content_frame.grid(row=1, column=1, sticky="w", pady=(10, 0))
        
        ttk.Checkbutton(content_frame, text="Hero Selection Analysis", 
                       variable=self.include_hero_analysis).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Checkbutton(content_frame, text="Card Pick Analysis", 
                       variable=self.include_card_analysis).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Checkbutton(content_frame, text="Draft Statistics", 
                       variable=self.include_statistics).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Checkbutton(content_frame, text="Performance Prediction", 
                       variable=self.include_performance_prediction).grid(row=3, column=0, sticky="w", pady=2)
        
        # Export location
        ttk.Label(options_frame, text="Export Location:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        
        location_frame = ttk.Frame(options_frame)
        location_frame.grid(row=2, column=1, sticky="ew", pady=(10, 0))
        location_frame.columnconfigure(0, weight=1)
        
        self.location_entry = ttk.Entry(location_frame, textvariable=self.export_location)
        self.location_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(location_frame, text="Browse", 
                  command=self._browse_export_location).grid(row=0, column=1)
    
    def _create_preview_section(self, parent):
        """Create preview section."""
        preview_frame = ttk.LabelFrame(parent, text="Export Preview", padding="10")
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        
        # Preview controls
        preview_controls = ttk.Frame(preview_frame)
        preview_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(preview_controls, text="Preview Format:").pack(side="left", padx=(0, 10))
        
        self.preview_format = tk.StringVar(value="summary")
        preview_combo = ttk.Combobox(preview_controls, textvariable=self.preview_format,
                                   values=["summary", "hero_analysis", "card_picks", "statistics"],
                                   state="readonly", width=15)
        preview_combo.pack(side="left", padx=(0, 10))
        preview_combo.bind('<<ComboboxSelected>>', self._update_preview)
        
        ttk.Button(preview_controls, text="Refresh Preview", 
                  command=self._update_preview).pack(side="left", padx=(10, 0))
        
        # Preview text area
        preview_text_frame = ttk.Frame(preview_frame)
        preview_text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_text_frame.columnconfigure(0, weight=1)
        preview_text_frame.rowconfigure(0, weight=1)
        
        self.preview_text = tk.Text(preview_text_frame, wrap="word", height=10)
        preview_scrollbar = ttk.Scrollbar(preview_text_frame, orient="vertical", 
                                        command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=preview_scrollbar.set)
        
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Load initial preview
        self._update_preview()
    
    def _create_button_section(self, parent):
        """Create dialog buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, sticky="ew")
        button_frame.columnconfigure(1, weight=1)
        
        # Export info
        self.export_status = tk.StringVar()
        self.export_status.set("Ready to export")
        ttk.Label(button_frame, textvariable=self.export_status).grid(row=0, column=0, sticky="w")
        
        # Buttons
        button_container = ttk.Frame(button_frame)
        button_container.grid(row=0, column=1, sticky="e")
        
        ttk.Button(button_container, text="Preview Export", 
                  command=self._preview_export).pack(side="left", padx=(0, 5))
        
        ttk.Button(button_container, text="Export", 
                  command=self._export_draft).pack(side="left", padx=5)
        
        ttk.Button(button_container, text="Close", 
                  command=self._close_dialog).pack(side="left", padx=(5, 0))
    
    def _load_default_settings(self):
        """Load default export settings."""
        # Set default export location
        default_location = Path("draft_exports")
        self.export_location.set(str(default_location))
    
    def _browse_export_location(self):
        """Browse for export location."""
        directory = filedialog.askdirectory(
            title="Select Export Directory",
            initialdir=self.export_location.get()
        )
        
        if directory:
            self.export_location.set(directory)
    
    def _update_preview(self, *args):
        """Update preview content."""
        if not self.draft_summary:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, "No draft data available for preview")
            return
        
        preview_type = self.preview_format.get()
        content = self._generate_preview_content(preview_type)
        
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(1.0, content)
    
    def _generate_preview_content(self, preview_type: str) -> str:
        """Generate preview content for specified type."""
        if not self.draft_summary:
            return "No draft data available"
        
        if preview_type == "summary":
            return self._generate_summary_preview()
        elif preview_type == "hero_analysis":
            return self._generate_hero_preview()
        elif preview_type == "card_picks":
            return self._generate_card_picks_preview()
        elif preview_type == "statistics":
            return self._generate_statistics_preview()
        else:
            return "Unknown preview type"
    
    def _generate_summary_preview(self) -> str:
        """Generate summary preview."""
        lines = [
            f"Draft ID: {self.draft_summary.draft_id}",
            f"Date: {self.draft_summary.start_time.strftime('%Y-%m-%d %H:%M')}",
            f"Duration: {self.draft_summary.total_duration_minutes:.1f} minutes",
            ""
        ]
        
        if self.draft_summary.hero_choice:
            hero = self.draft_summary.hero_choice
            lines.extend([
                f"Hero Selected: {hero.user_selected_hero}",
                f"Hero Recommended: {hero.recommended_hero}",
                f"Followed Hero Recommendation: {'Yes' if hero.followed_recommendation else 'No'}",
                ""
            ])
        
        lines.extend([
            f"Total Picks: {len(self.draft_summary.card_picks)}",
            f"Recommendations Followed: {self.draft_summary.recommendations_followed}/{self.draft_summary.total_recommendations}",
            f"Follow Rate: {self.draft_summary.follow_rate_percentage:.1f}%",
            f"Average Confidence: {self.draft_summary.average_confidence:.1%}",
            "",
            "Export will include complete analysis with detailed reasoning for each decision."
        ])
        
        return "\n".join(lines)
    
    def _generate_hero_preview(self) -> str:
        """Generate hero analysis preview."""
        if not self.draft_summary.hero_choice:
            return "No hero selection data available"
        
        hero = self.draft_summary.hero_choice
        lines = [
            "HERO SELECTION ANALYSIS",
            "=" * 30,
            f"Offered Heroes: {', '.join(hero.offered_heroes)}",
            f"Recommended: {hero.recommended_hero} (Position {hero.recommended_index + 1})",
            f"Selected: {hero.user_selected_hero} (Position {hero.user_selected_index + 1})",
            f"Followed Recommendation: {'Yes' if hero.followed_recommendation else 'No'}",
            f"Confidence: {hero.confidence_level:.1%}",
            "",
            "Reasoning:",
            hero.selection_reasoning,
            "",
            "Hero Winrates:"
        ]
        
        for hero_class, winrate in hero.hero_winrates.items():
            lines.append(f"  {hero_class}: {winrate:.1f}%")
        
        return "\n".join(lines)
    
    def _generate_card_picks_preview(self) -> str:
        """Generate card picks preview."""
        if not self.draft_summary.card_picks:
            return "No card pick data available"
        
        lines = [
            "CARD PICK ANALYSIS (First 5 picks)",
            "=" * 40
        ]
        
        for pick in self.draft_summary.card_picks[:5]:
            status = "✓" if pick.followed_recommendation else "✗"
            lines.extend([
                f"Pick {pick.pick_number}: {pick.user_selected_card} {status}",
                f"  Recommended: {pick.recommended_card}",
                f"  Confidence: {pick.confidence_level:.1%}",
                f"  Reasoning: {pick.pick_reasoning[:100]}{'...' if len(pick.pick_reasoning) > 100 else ''}",
                ""
            ])
        
        if len(self.draft_summary.card_picks) > 5:
            lines.append(f"... and {len(self.draft_summary.card_picks) - 5} more picks")
        
        return "\n".join(lines)
    
    def _generate_statistics_preview(self) -> str:
        """Generate statistics preview."""
        lines = [
            "DRAFT STATISTICS",
            "=" * 20,
            f"Total Duration: {self.draft_summary.total_duration_minutes:.1f} minutes",
            f"Total Decisions: {self.draft_summary.total_recommendations}",
            f"Recommendations Followed: {self.draft_summary.recommendations_followed}",
            f"Follow Rate: {self.draft_summary.follow_rate_percentage:.1f}%",
            f"Average Confidence: {self.draft_summary.average_confidence:.1%}",
            "",
            "Decision Breakdown:"
        ]
        
        if self.draft_summary.hero_choice:
            hero_status = "Followed" if self.draft_summary.hero_choice.followed_recommendation else "Not Followed"
            lines.append(f"  Hero Selection: {hero_status}")
        
        followed_picks = sum(1 for pick in self.draft_summary.card_picks if pick.followed_recommendation)
        total_picks = len(self.draft_summary.card_picks)
        lines.extend([
            f"  Card Picks: {followed_picks}/{total_picks} followed",
            "",
            "Confidence Distribution:"
        ])
        
        # Simple confidence distribution
        high_conf = sum(1 for pick in self.draft_summary.card_picks if pick.confidence_level > 0.8)
        med_conf = sum(1 for pick in self.draft_summary.card_picks if 0.5 <= pick.confidence_level <= 0.8)
        low_conf = sum(1 for pick in self.draft_summary.card_picks if pick.confidence_level < 0.5)
        
        lines.extend([
            f"  High (>80%): {high_conf} picks",
            f"  Medium (50-80%): {med_conf} picks", 
            f"  Low (<50%): {low_conf} picks"
        ])
        
        return "\n".join(lines)
    
    def _preview_export(self):
        """Preview what will be exported."""
        selected_formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        
        if not selected_formats:
            messagebox.showwarning("No Formats Selected", "Please select at least one export format.")
            return
        
        preview_text = f"Export Preview:\n\n"
        preview_text += f"Formats: {', '.join(selected_formats).upper()}\n"
        preview_text += f"Location: {self.export_location.get()}\n\n"
        
        preview_text += "Content Included:\n"
        if self.include_hero_analysis.get():
            preview_text += "• Hero Selection Analysis\n"
        if self.include_card_analysis.get():
            preview_text += "• Card Pick Analysis\n"
        if self.include_statistics.get():
            preview_text += "• Draft Statistics\n"
        if self.include_performance_prediction.get():
            preview_text += "• Performance Prediction\n"
        
        if self.draft_summary:
            estimated_size = len(selected_formats) * len(self.draft_summary.card_picks) * 0.5  # KB estimate
            preview_text += f"\nEstimated Size: {estimated_size:.1f} KB"
        
        messagebox.showinfo("Export Preview", preview_text)
    
    def _export_draft(self):
        """Perform the actual export."""
        try:
            # Validate inputs
            if not self.draft_summary:
                messagebox.showerror("No Data", "No draft data available to export.")
                return
            
            selected_formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
            if not selected_formats:
                messagebox.showwarning("No Formats", "Please select at least one export format.")
                return
            
            export_dir = self.export_location.get()
            if not export_dir:
                messagebox.showwarning("No Location", "Please select an export location.")
                return
            
            # Create export directory
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            
            # Update status
            self.export_status.set("Exporting...")
            self.dialog.update()
            
            # Perform export
            exported_files = self.draft_exporter.export_draft(
                self.draft_summary, 
                formats=selected_formats,
                output_dir=export_dir
            )
            
            self.exported_files = exported_files
            
            # Update status
            self.export_status.set(f"Exported {len(exported_files)} files successfully")
            
            # Show success message with options
            self._show_export_success(exported_files)
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export draft: {e}")
            self.export_status.set("Export failed")
    
    def _show_export_success(self, exported_files: Dict[str, str]):
        """Show export success dialog with actions."""
        message = f"Successfully exported {len(exported_files)} files:\n\n"
        for format_type, file_path in exported_files.items():
            filename = Path(file_path).name
            message += f"• {format_type.upper()}: {filename}\n"
        
        message += f"\nLocation: {self.export_location.get()}"
        
        result = messagebox.askyesnocancel(
            "Export Complete", 
            message + "\n\nWould you like to open the export folder?",
            icon='question'
        )
        
        if result is True:  # Yes - open folder
            self._open_export_folder()
        elif result is False:  # No - show file actions
            self._show_file_actions(exported_files)
    
    def _open_export_folder(self):
        """Open the export folder in file manager."""
        try:
            export_path = Path(self.export_location.get())
            
            if sys.platform == "win32":
                subprocess.run(["explorer", str(export_path)])
            elif sys.platform == "darwin":
                subprocess.run(["open", str(export_path)])
            else:
                subprocess.run(["xdg-open", str(export_path)])
                
        except Exception as e:
            self.logger.error(f"Failed to open export folder: {e}")
            messagebox.showerror("Error", f"Failed to open folder: {e}")
    
    def _show_file_actions(self, exported_files: Dict[str, str]):
        """Show individual file action options."""
        # Create a simple dialog for file actions
        action_dialog = tk.Toplevel(self.dialog)
        action_dialog.title("Exported Files")
        action_dialog.geometry("400x300")
        action_dialog.transient(self.dialog)
        
        frame = ttk.Frame(action_dialog, padding="10")
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Exported Files:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        
        for format_type, file_path in exported_files.items():
            file_frame = ttk.Frame(frame)
            file_frame.pack(fill="x", pady=5)
            
            filename = Path(file_path).name
            ttk.Label(file_frame, text=f"{format_type.upper()}: {filename}").pack(side="left")
            
            if format_type == "html":
                ttk.Button(file_frame, text="Open", 
                          command=lambda fp=file_path: self._open_file(fp)).pack(side="right")
        
        ttk.Button(frame, text="Close", command=action_dialog.destroy).pack(pady=(10, 0))
    
    def _open_file(self, file_path: str):
        """Open exported file."""
        try:
            if sys.platform == "win32":
                subprocess.run(["start", file_path], shell=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            self.logger.error(f"Failed to open file: {e}")
            messagebox.showerror("Error", f"Failed to open file: {e}")
    
    def _center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    def _close_dialog(self):
        """Close the dialog."""
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog."""
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.wait_window()


def show_draft_export_dialog(parent=None, draft_summary=None):
    """Show the draft export dialog."""
    dialog = DraftExportDialog(parent, draft_summary)
    dialog.show()