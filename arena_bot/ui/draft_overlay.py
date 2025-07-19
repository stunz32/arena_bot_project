#!/usr/bin/env python3
"""
Real-time draft overlay for Arena Bot.
Displays pick recommendations directly over the Hearthstone window.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import cv2

logger = logging.getLogger(__name__)

@dataclass
class OverlayConfig:
    """Configuration for the draft overlay."""
    opacity: float = 0.85
    update_interval: float = 2.0  # seconds
    show_tier_scores: bool = True
    show_win_rates: bool = True
    font_size: int = 12
    background_color: str = "#2c3e50"
    text_color: str = "#ecf0f1"
    highlight_color: str = "#e74c3c"
    success_color: str = "#27ae60"

class DraftOverlay:
    """
    Real-time overlay window for draft recommendations.
    """
    
    def __init__(self, config: OverlayConfig = None):
        """Initialize the draft overlay."""
        self.config = config or OverlayConfig()
        self.logger = logging.getLogger(__name__)
        
        # Overlay state
        self.root = None
        self.running = False
        self.current_analysis = None
        self.update_thread = None
        
        # UI elements
        self.status_label = None
        self.recommendation_frame = None
        self.card_frames = []
        
        self.logger.info("DraftOverlay initialized")
    
    def create_overlay_window(self) -> tk.Tk:
        """Create the overlay window."""
        root = tk.Tk()
        root.title("Arena Bot - Draft Assistant")
        root.configure(bg=self.config.background_color)
        
        # Make window stay on top
        root.attributes('-topmost', True)
        
        # Set window transparency
        root.attributes('-alpha', self.config.opacity)
        
        # Set window size and position
        window_width = 400
        window_height = 300
        
        # Position overlay in top-right corner
        screen_width = root.winfo_screenwidth()
        x = screen_width - window_width - 50
        y = 50
        
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Prevent window from being resized
        root.resizable(False, False)
        
        return root
    
    def create_ui_elements(self):
        """Create the UI elements for the overlay."""
        # Title
        title_label = tk.Label(
            self.root,
            text="🎯 Arena Draft Assistant",
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="🔍 Waiting for draft...",
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)
        
        # Recommendation frame
        self.recommendation_frame = tk.Frame(
            self.root,
            bg=self.config.background_color
        )
        self.recommendation_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg=self.config.background_color)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Manual update button
        update_btn = tk.Button(
            control_frame,
            text="🔄 Update",
            command=self.manual_update,
            bg="#3498db",
            fg="white",
            font=("Arial", 9)
        )
        update_btn.pack(side="left", padx=5)
        
        # Settings button
        settings_btn = tk.Button(
            control_frame,
            text="⚙️ Settings",
            command=self.show_settings,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 9)
        )
        settings_btn.pack(side="left", padx=5)
        
        # Close button
        close_btn = tk.Button(
            control_frame,
            text="❌ Close",
            command=self.stop,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 9)
        )
        close_btn.pack(side="right", padx=5)
    
    def update_recommendation_display(self, analysis: Dict):
        """Update the recommendation display with new analysis."""
        # Clear existing content
        for widget in self.recommendation_frame.winfo_children():
            widget.destroy()
        
        if not analysis or not analysis.get('success'):
            error_label = tk.Label(
                self.recommendation_frame,
                text="❌ No draft detected",
                bg=self.config.background_color,
                fg=self.config.highlight_color,
                font=("Arial", 10)
            )
            error_label.pack(pady=20)
            return
        
        # Recommendation header
        rec_card = analysis['recommended_card']
        rec_level = analysis['recommendation_level'].upper()
        
        header_text = f"👑 PICK: {rec_card}"
        header_label = tk.Label(
            self.recommendation_frame,
            text=header_text,
            bg=self.config.background_color,
            fg=self.config.success_color,
            font=("Arial", 12, "bold")
        )
        header_label.pack(pady=(0, 10))
        
        # Confidence level
        confidence_label = tk.Label(
            self.recommendation_frame,
            text=f"Confidence: {rec_level}",
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=("Arial", 10)
        )
        confidence_label.pack()
        
        # Card details
        for i, card in enumerate(analysis['card_details']):
            is_recommended = (i == analysis['recommended_pick'] - 1)
            
            # Card frame
            card_frame = tk.Frame(
                self.recommendation_frame,
                bg=self.config.success_color if is_recommended else self.config.background_color,
                relief="raised" if is_recommended else "flat",
                bd=2 if is_recommended else 0
            )
            card_frame.pack(fill="x", pady=2, padx=5)
            
            # Card name and tier
            name_text = f"{i+1}. {card['card_code']} ({card['tier']}-tier)"
            name_label = tk.Label(
                card_frame,
                text=name_text,
                bg=self.config.success_color if is_recommended else self.config.background_color,
                fg="white" if is_recommended else self.config.text_color,
                font=("Arial", 9, "bold" if is_recommended else "normal")
            )
            name_label.pack(anchor="w")
            
            # Stats
            if self.config.show_win_rates:
                stats_text = f"   Win Rate: {card['win_rate']:.1%}"
                if self.config.show_tier_scores:
                    stats_text += f" | Score: {card['tier_score']:.1f}"
                
                stats_label = tk.Label(
                    card_frame,
                    text=stats_text,
                    bg=self.config.success_color if is_recommended else self.config.background_color,
                    fg="white" if is_recommended else "#bdc3c7",
                    font=("Arial", 8)
                )
                stats_label.pack(anchor="w")
    
    def manual_update(self):
        """Manually trigger an update."""
        self.logger.info("Manual update triggered")
        self.status_label.config(text="🔄 Updating...")
        
        # Import here to avoid circular imports
        from ..complete_arena_bot import CompleteArenaBot
        
        try:
            bot = CompleteArenaBot()
            analysis = bot.analyze_draft("screenshot.png", "warrior")
            
            if analysis['success']:
                self.current_analysis = analysis
                self.update_recommendation_display(analysis)
                self.status_label.config(
                    text=f"✅ Updated - {len(analysis['detected_cards'])} cards detected"
                )
            else:
                self.status_label.config(text="❌ No draft found")
                
        except Exception as e:
            self.logger.error(f"Update error: {e}")
            self.status_label.config(text="❌ Update failed")
    
    def show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Arena Bot Settings")
        settings_window.configure(bg=self.config.background_color)
        settings_window.geometry("300x200")
        settings_window.attributes('-topmost', True)
        
        # Opacity setting
        tk.Label(
            settings_window,
            text="Overlay Opacity:",
            bg=self.config.background_color,
            fg=self.config.text_color
        ).pack(pady=5)
        
        opacity_var = tk.DoubleVar(value=self.config.opacity)
        opacity_scale = tk.Scale(
            settings_window,
            from_=0.3,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=opacity_var,
            bg=self.config.background_color,
            fg=self.config.text_color
        )
        opacity_scale.pack(pady=5)
        
        # Update interval setting
        tk.Label(
            settings_window,
            text="Update Interval (seconds):",
            bg=self.config.background_color,
            fg=self.config.text_color
        ).pack(pady=5)
        
        interval_var = tk.DoubleVar(value=self.config.update_interval)
        interval_scale = tk.Scale(
            settings_window,
            from_=1.0,
            to=10.0,
            resolution=0.5,
            orient="horizontal",
            variable=interval_var,
            bg=self.config.background_color,
            fg=self.config.text_color
        )
        interval_scale.pack(pady=5)
        
        # Apply button
        def apply_settings():
            self.config.opacity = opacity_var.get()
            self.config.update_interval = interval_var.get()
            self.root.attributes('-alpha', self.config.opacity)
            settings_window.destroy()
        
        apply_btn = tk.Button(
            settings_window,
            text="Apply",
            command=apply_settings,
            bg="#27ae60",
            fg="white"
        )
        apply_btn.pack(pady=10)
    
    def auto_update_loop(self):
        """Automatic update loop running in separate thread."""
        while self.running:
            try:
                # Sleep first to avoid immediate update on start
                time.sleep(self.config.update_interval)
                
                if not self.running:
                    break
                
                # Update in main thread
                self.root.after(0, self.manual_update)
                
            except Exception as e:
                self.logger.error(f"Auto-update error: {e}")
    
    def start(self):
        """Start the overlay interface."""
        self.logger.info("Starting draft overlay")
        
        # Create and setup window
        self.root = self.create_overlay_window()
        self.create_ui_elements()
        
        # Start running
        self.running = True
        
        # Start auto-update thread
        self.update_thread = threading.Thread(target=self.auto_update_loop, daemon=True)
        self.update_thread.start()
        
        # Do initial update
        self.manual_update()
        
        # Start the GUI main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the overlay interface."""
        self.logger.info("Stopping draft overlay")
        self.running = False
        
        if self.root:
            self.root.quit()
            self.root.destroy()


def create_draft_overlay(config: OverlayConfig = None) -> DraftOverlay:
    """Create a new draft overlay instance."""
    return DraftOverlay(config)


def main():
    """Demo the draft overlay."""
    print("=== Draft Overlay Demo ===")
    print("This will create a real-time overlay window.")
    print("Press Ctrl+C or close the window to exit.")
    
    # Create and start overlay
    overlay = create_draft_overlay()
    
    try:
        overlay.start()
    except KeyboardInterrupt:
        print("\nOverlay stopped by user")


if __name__ == "__main__":
    main()