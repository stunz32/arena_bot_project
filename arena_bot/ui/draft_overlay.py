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

# Enhanced tooltip class for comprehensive statistical information
class StatisticalTooltip:
    """
    Advanced tooltip system that displays comprehensive statistical information
    incorporating both hero and card context for AI v2 system.
    """
    
    def __init__(self, widget, text: str, hero_context: Dict = None, card_context: Dict = None):
        self.widget = widget
        self.text = text
        self.hero_context = hero_context or {}
        self.card_context = card_context or {}
        self.tooltip_window = None
        
        # Bind events
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Motion>", self.on_motion)
    
    def on_enter(self, event=None):
        """Show tooltip on mouse enter."""
        self.show_tooltip(event)
    
    def on_leave(self, event=None):
        """Hide tooltip on mouse leave."""
        self.hide_tooltip()
    
    def on_motion(self, event=None):
        """Update tooltip position on mouse motion."""
        if self.tooltip_window:
            self.update_tooltip_position(event)
    
    def show_tooltip(self, event=None):
        """Display the comprehensive statistical tooltip."""
        if self.tooltip_window or not self.text:
            return
        
        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.configure(bg="#1a1a2e", relief="solid", bd=1)
        
        # Create comprehensive tooltip content
        self.create_tooltip_content()
        
        # Position tooltip
        self.update_tooltip_position(event)
        
        # Make tooltip stay on top
        self.tooltip_window.attributes('-topmost', True)
    
    def create_tooltip_content(self):
        """Create comprehensive tooltip content with statistical information."""
        main_frame = tk.Frame(self.tooltip_window, bg="#1a1a2e", padx=8, pady=6)
        main_frame.pack()
        
        # Main text
        main_label = tk.Label(
            main_frame,
            text=self.text,
            bg="#1a1a2e",
            fg="#ffd700",
            font=("Arial", 10, "bold"),
            justify="left"
        )
        main_label.pack(anchor="w")
        
        # Separator
        separator = tk.Frame(main_frame, height=1, bg="#ffd700")
        separator.pack(fill="x", pady=3)
        
        # Hero context information
        if self.hero_context:
            self.add_hero_context_info(main_frame)
        
        # Card context information
        if self.card_context:
            self.add_card_context_info(main_frame)
        
        # AI v2 system information
        self.add_ai_system_info(main_frame)
    
    def add_hero_context_info(self, parent):
        """Add hero-specific context information to tooltip."""
        hero_frame = tk.LabelFrame(
            parent,
            text="👑 Hero Context",
            bg="#1a1a2e",
            fg="#f39c12",
            font=("Arial", 9, "bold")
        )
        hero_frame.pack(fill="x", pady=2)
        
        hero_class = self.hero_context.get('class', 'Unknown')
        winrate = self.hero_context.get('winrate', 0)
        profile = self.hero_context.get('profile', {})
        
        # Hero basic info
        info_text = f"Class: {hero_class}"
        if winrate > 0:
            info_text += f"\nWin Rate: {winrate:.1f}%"
            
        if profile.get('playstyle'):
            info_text += f"\nPlaystyle: {profile['playstyle']}"
        if profile.get('complexity'):
            info_text += f"\nComplexity: {profile['complexity']}"
        
        hero_info_label = tk.Label(
            hero_frame,
            text=info_text,
            bg="#1a1a2e",
            fg="#ecf0f1",
            font=("Arial", 8),
            justify="left"
        )
        hero_info_label.pack(anchor="w", padx=5, pady=2)
        
        # Meta analysis if available
        if self.hero_context.get('meta_position'):
            meta_text = f"Meta Position: {self.hero_context['meta_position']}"
            meta_label = tk.Label(
                hero_frame,
                text=meta_text,
                bg="#1a1a2e",
                fg="#3498db",
                font=("Arial", 8, "italic")
            )
            meta_label.pack(anchor="w", padx=5)
    
    def add_card_context_info(self, parent):
        """Add card-specific context information to tooltip."""
        card_frame = tk.LabelFrame(
            parent,
            text="🃏 Card Analysis",
            bg="#1a1a2e",
            fg="#3498db",
            font=("Arial", 9, "bold")
        )
        card_frame.pack(fill="x", pady=2)
        
        # Basic card stats
        basic_stats = []
        if self.card_context.get('win_rate'):
            basic_stats.append(f"Win Rate: {self.card_context['win_rate']:.1%}")
        if self.card_context.get('tier_score'):
            basic_stats.append(f"Tier Score: {self.card_context['tier_score']:.1f}")
        if self.card_context.get('deck_win_rate'):
            basic_stats.append(f"Deck Win Rate: {self.card_context['deck_win_rate']:.1f}%")
        
        if basic_stats:
            stats_text = " | ".join(basic_stats)
            stats_label = tk.Label(
                card_frame,
                text=stats_text,
                bg="#1a1a2e",
                fg="#ecf0f1",
                font=("Arial", 8)
            )
            stats_label.pack(anchor="w", padx=5, pady=2)
        
        # Dimensional scores if available
        dimensional_scores = self.card_context.get('dimensional_scores', {})
        if dimensional_scores:
            dimensions_text = "Dimensional Analysis:"
            for dimension, score in dimensional_scores.items():
                if score > 0:
                    dimensions_text += f"\n  • {dimension.replace('_', ' ').title()}: {score:.2f}"
            
            dimensions_label = tk.Label(
                card_frame,
                text=dimensions_text,
                bg="#1a1a2e",
                fg="#9b59b6",
                font=("Arial", 8),
                justify="left"
            )
            dimensions_label.pack(anchor="w", padx=5)
        
        # Hero synergy information
        if self.card_context.get('hero_synergy_score'):
            synergy_text = f"Hero Synergy: {self.card_context['hero_synergy_score']:.1f}/10"
            synergy_label = tk.Label(
                card_frame,
                text=synergy_text,
                bg="#1a1a2e",
                fg="#e74c3c",
                font=("Arial", 8, "bold")
            )
            synergy_label.pack(anchor="w", padx=5)
    
    def add_ai_system_info(self, parent):
        """Add AI v2 system information to tooltip."""
        ai_frame = tk.LabelFrame(
            parent,
            text="🤖 AI v2 System",
            bg="#1a1a2e",
            fg="#9b59b6",
            font=("Arial", 9, "bold")
        )
        ai_frame.pack(fill="x", pady=2)
        
        # Analysis confidence
        confidence = self.card_context.get('confidence', 0.8)
        confidence_text = f"Analysis Confidence: {confidence:.1%}"
        
        confidence_color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"
        confidence_label = tk.Label(
            ai_frame,
            text=confidence_text,
            bg="#1a1a2e",
            fg=confidence_color,
            font=("Arial", 8, "bold")
        )
        confidence_label.pack(anchor="w", padx=5, pady=2)
        
        # Data sources
        data_sources = []
        if self.card_context.get('hsreplay_data'):
            data_sources.append("HSReplay")
        if self.card_context.get('hearthareana_data'):
            data_sources.append("HearthArena")
        if self.card_context.get('ai_analysis'):
            data_sources.append("AI Analysis")
        
        if data_sources:
            sources_text = f"Data Sources: {', '.join(data_sources)}"
            sources_label = tk.Label(
                ai_frame,
                text=sources_text,
                bg="#1a1a2e",
                fg="#95a5a6",
                font=("Arial", 8)
            )
            sources_label.pack(anchor="w", padx=5)
        
        # Analysis timestamp
        timestamp = self.card_context.get('analysis_time', 'Unknown')
        if timestamp != 'Unknown':
            time_text = f"Analysis Time: {timestamp:.1f}ms"
            time_label = tk.Label(
                ai_frame,
                text=time_text,
                bg="#1a1a2e",
                fg="#7f8c8d",
                font=("Arial", 8)
            )
            time_label.pack(anchor="w", padx=5)
    
    def update_tooltip_position(self, event=None):
        """Update tooltip position relative to mouse."""
        if not self.tooltip_window:
            return
        
        # Get mouse position
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        
        if event:
            x = event.x_root + 20
            y = event.y_root + 20
        
        # Adjust position to keep tooltip on screen
        screen_width = self.tooltip_window.winfo_screenwidth()
        screen_height = self.tooltip_window.winfo_screenheight()
        
        # Update tooltip window size
        self.tooltip_window.update_idletasks()
        tooltip_width = self.tooltip_window.winfo_reqwidth()
        tooltip_height = self.tooltip_window.winfo_reqheight()
        
        # Adjust x position
        if x + tooltip_width > screen_width:
            x = screen_width - tooltip_width - 10
        
        # Adjust y position
        if y + tooltip_height > screen_height:
            y = y - tooltip_height - 40
        
        self.tooltip_window.geometry(f"+{x}+{y}")
    
    def hide_tooltip(self):
        """Hide the tooltip window."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class PerformanceMonitor:
    """
    Performance monitoring and optimization for dual-mode overlay rendering.
    Tracks rendering times, memory usage, and provides optimization recommendations.
    """
    
    def __init__(self):
        self.render_times = []
        self.memory_usage = []
        self.frame_drops = 0
        self.total_renders = 0
        self.optimization_active = False
        
    def start_render(self):
        """Start timing a render operation."""
        return time.time()
        
    def end_render(self, start_time: float):
        """End timing a render operation and record metrics."""
        render_time = (time.time() - start_time) * 1000  # Convert to ms
        self.render_times.append(render_time)
        self.total_renders += 1
        
        # Keep only recent render times (last 100)
        if len(self.render_times) > 100:
            self.render_times.pop(0)
            
        # Detect frame drops (>50ms render time)
        if render_time > 50:
            self.frame_drops += 1
            
        return render_time
        
    def get_average_render_time(self) -> float:
        """Get average render time in milliseconds."""
        return sum(self.render_times) / len(self.render_times) if self.render_times else 0
        
    def is_performance_degraded(self) -> bool:
        """Check if performance is degraded and optimization is needed."""
        avg_time = self.get_average_render_time()
        frame_drop_rate = self.frame_drops / max(1, self.total_renders)
        
        return avg_time > 30 or frame_drop_rate > 0.1  # 30ms or 10% frame drops
        
    def get_optimization_recommendations(self) -> list:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if self.get_average_render_time() > 25:
            recommendations.append("Enable batch rendering")
            recommendations.append("Reduce tooltip complexity")
            
        if self.frame_drops > 10:
            recommendations.append("Enable UI caching")
            recommendations.append("Disable smooth animations")
            
        return recommendations


logger = logging.getLogger(__name__)

@dataclass
class OverlayConfig:
    """Configuration for the draft overlay."""
    opacity: float = 0.85
    update_interval: float = 2.0  # seconds
    show_tier_scores: bool = True
    show_win_rates: bool = True
    show_hero_winrates: bool = True  # NEW: Show hero winrates
    show_confidence_indicators: bool = True  # NEW: Show confidence levels
    font_size: int = 12
    background_color: str = "#2c3e50"
    text_color: str = "#ecf0f1"
    highlight_color: str = "#e74c3c"
    success_color: str = "#27ae60"
    hero_color: str = "#f39c12"  # NEW: Hero selection color
    ai_v2_color: str = "#9b59b6"  # NEW: AI v2 system color

class DraftOverlay:
    """
    Enhanced real-time overlay window for both hero selection and card draft recommendations.
    Supports AI v2 system with statistical backing and hero-aware analysis.
    """
    
    def __init__(self, config: OverlayConfig = None):
        """Initialize the enhanced draft overlay."""
        self.config = config or OverlayConfig()
        self.logger = logging.getLogger(__name__)
        
        # Overlay state
        self.root = None
        self.running = False
        self.current_analysis = None
        self.current_hero_analysis = None  # NEW: Hero selection analysis
        self.update_thread = None
        
        # Draft phase tracking
        self.draft_phase = "waiting"  # "waiting", "hero_selection", "card_picks"
        self.selected_hero = None
        
        # UI elements
        self.status_label = None
        self.phase_label = None  # NEW: Draft phase indicator
        self.recommendation_frame = None
        self.hero_frame = None  # NEW: Hero selection frame
        self.card_frames = []
        
        # AI v2 integration
        self.ai_v2_enabled = True
        self.system_health_indicator = None  # NEW: System health display
        
        # Performance optimization
        self.performance_monitor = PerformanceMonitor()
        self.ui_cache = {}  # Cache for UI elements
        self.render_queue = []  # Queue for batch rendering
        self.last_render_time = 0
        self.min_render_interval = 100  # Minimum ms between renders
        
        # Optimization flags
        self.batch_rendering_enabled = True
        self.ui_caching_enabled = True
        self.smooth_animations_enabled = True
        
        self.logger.info("Enhanced DraftOverlay initialized with AI v2 support and performance optimizations")
    
    def create_overlay_window(self) -> tk.Tk:
        """Create the adaptive overlay window that adjusts based on draft phase."""
        root = tk.Tk()
        root.title("Arena AI v2 - Grandmaster Coach")
        root.configure(bg=self.config.background_color)
        
        # Make window stay on top
        root.attributes('-topmost', True)
        
        # Set window transparency
        root.attributes('-alpha', self.config.opacity)
        
        # Dynamic window sizing based on phase
        self.update_window_size_for_phase("waiting")
        
        # Position overlay intelligently
        self.position_overlay_intelligently()
        
        # Prevent window from being resized manually
        root.resizable(False, False)
        
        return root
    
    def update_window_size_for_phase(self, phase: str):
        """Dynamically adjust window size based on current draft phase."""
        phase_configs = {
            "waiting": {"width": 350, "height": 200},
            "hero_selection": {"width": 450, "height": 600},
            "card_picks": {"width": 400, "height": 400}
        }
        
        config = phase_configs.get(phase, phase_configs["waiting"])
        self.window_width = config["width"]
        self.window_height = config["height"]
        
        if hasattr(self, 'root') and self.root:
            # Smoothly resize window
            self.animate_window_resize(config["width"], config["height"])
    
    def position_overlay_intelligently(self):
        """Position overlay intelligently to avoid interfering with game UI."""
        if not hasattr(self, 'root') or not self.root:
            return
            
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Default position (top-right corner)
        x = screen_width - self.window_width - 50
        y = 50
        
        # Adjust based on phase for optimal positioning
        if self.draft_phase == "hero_selection":
            # Position closer to center for hero selection
            x = screen_width - self.window_width - 30
            y = 30
        elif self.draft_phase == "card_picks":
            # Standard position for card picks
            x = screen_width - self.window_width - 40
            y = 100
        
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
    
    def animate_window_resize(self, target_width: int, target_height: int):
        """Smoothly animate window resize for better user experience."""
        if not self.root:
            return
            
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        
        # Simple resize (can be enhanced with smooth animation)
        self.root.geometry(f"{target_width}x{target_height}")
        
        # Update position after resize
        self.position_overlay_intelligently()
    
    def create_ui_elements(self):
        """Create the enhanced UI elements for the overlay."""
        # Title with AI v2 branding
        title_label = tk.Label(
            self.root,
            text="🎯 Arena AI v2 - Grandmaster Coach",
            bg=self.config.background_color,
            fg=self.config.ai_v2_color,
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # System health indicator
        self.system_health_indicator = tk.Label(
            self.root,
            text="🟢 AI v2 Systems Online",
            bg=self.config.background_color,
            fg=self.config.success_color,
            font=("Arial", 9)
        )
        self.system_health_indicator.pack()
        
        # Draft phase indicator
        self.phase_label = tk.Label(
            self.root,
            text="📋 Phase: Waiting for Draft",
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=("Arial", 10, "bold")
        )
        self.phase_label.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="🔍 Monitoring for hero selection or card picks...",
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)
        
        # Hero selection frame (initially hidden)
        self.hero_frame = tk.Frame(
            self.root,
            bg=self.config.background_color
        )
        
        # Recommendation frame for cards
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
        """Update the recommendation display with new analysis and hero-specific visual cues."""
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
        
        # Enhanced header with AI v2 and hero context
        rec_card = analysis['recommended_card']
        rec_level = analysis['recommendation_level'].upper()
        
        header_text = f"🎯 AI v2 PICK: {rec_card}"
        if self.selected_hero:
            header_text += f" (for {self.selected_hero})"
            
        header_label = tk.Label(
            self.recommendation_frame,
            text=header_text,
            bg=self.config.background_color,
            fg=self.config.ai_v2_color,
            font=("Arial", 12, "bold")
        )
        header_label.pack(pady=(0, 5))
        
        # Confidence level with hero context
        confidence_text = f"📊 Confidence: {rec_level}"
        if self.selected_hero:
            confidence_text += f" | Hero Synergy: Optimized for {self.selected_hero}"
            
        confidence_label = tk.Label(
            self.recommendation_frame,
            text=confidence_text,
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=("Arial", 10)
        )
        confidence_label.pack()
        
        # Card details with hero-specific visual cues
        for i, card in enumerate(analysis['card_details']):
            is_recommended = (i == analysis['recommended_pick'] - 1)
            
            # Determine hero synergy level for color coding
            hero_synergy_level = self._calculate_hero_synergy_level(card, self.selected_hero)
            card_bg_color, card_border = self._get_hero_synergy_colors(hero_synergy_level, is_recommended)
            
            # Card frame with hero-specific styling
            card_frame = tk.Frame(
                self.recommendation_frame,
                bg=card_bg_color,
                relief="raised" if is_recommended else card_border,
                bd=3 if is_recommended else 2
            )
            card_frame.pack(fill="x", pady=2, padx=5)
            
            # Card name with hero synergy indicator
            name_text = f"{i+1}. {card['card_code']}"
            if card.get('tier'):
                name_text += f" ({card['tier']}-tier)"
            
            # Add hero synergy visual indicator
            synergy_indicator = self._get_synergy_indicator(hero_synergy_level)
            if synergy_indicator:
                name_text = f"{synergy_indicator} {name_text}"
            
            name_label = tk.Label(
                card_frame,
                text=name_text,
                bg=card_bg_color,
                fg="white" if is_recommended else self.config.text_color,
                font=("Arial", 9, "bold" if is_recommended else "normal")
            )
            name_label.pack(anchor="w", padx=5, pady=2)
            
            # Add comprehensive statistical tooltip
            tooltip_text = f"Detailed Analysis: {card['card_code']}"
            card_context = {
                'win_rate': card.get('win_rate'),
                'tier_score': card.get('tier_score'),
                'deck_win_rate': card.get('deck_win_rate'),
                'dimensional_scores': card.get('dimensional_scores', {}),
                'hero_synergy_score': card.get('hero_synergy_score'),
                'confidence': card.get('confidence', 0.8),
                'hsreplay_data': True,
                'ai_analysis': True,
                'analysis_time': card.get('analysis_time_ms', 50.0)
            }
            
            hero_context = {}
            if self.selected_hero:
                hero_context = {
                    'class': self.selected_hero,
                    'winrate': 50.0,  # Default, should come from hero analysis
                    'profile': {'playstyle': 'Unknown', 'complexity': 'Medium'}
                }
            
            StatisticalTooltip(card_frame, tooltip_text, hero_context, card_context)
            
            # Enhanced stats with hero context
            if self.config.show_win_rates:
                stats_parts = []
                
                if card.get('win_rate'):
                    stats_parts.append(f"Win Rate: {card['win_rate']:.1%}")
                    
                if self.config.show_tier_scores and card.get('tier_score'):
                    stats_parts.append(f"Score: {card['tier_score']:.1f}")
                
                # Add hero-specific stats if available
                if self.selected_hero and card.get('hero_synergy_score'):
                    stats_parts.append(f"Hero Synergy: {card['hero_synergy_score']:.1f}")
                
                if stats_parts:
                    stats_text = "   " + " | ".join(stats_parts)
                    stats_label = tk.Label(
                        card_frame,
                        text=stats_text,
                        bg=card_bg_color,
                        fg="white" if is_recommended else "#bdc3c7",
                        font=("Arial", 8)
                    )
                    stats_label.pack(anchor="w", padx=5)
                
                # Hero-specific explanation if available
                if self.selected_hero and card.get('hero_explanation'):
                    explanation_text = f"🎯 {card['hero_explanation']}"
                    if len(explanation_text) > 60:
                        explanation_text = explanation_text[:60] + "..."
                        
                    explanation_label = tk.Label(
                        card_frame,
                        text=explanation_text,
                        bg=card_bg_color,
                        fg="white" if is_recommended else "#95a5a6",
                        font=("Arial", 8),
                        wraplength=280,
                        justify="left"
                    )
                    explanation_label.pack(anchor="w", padx=5, pady=(0, 2))
    
    def _calculate_hero_synergy_level(self, card: Dict, hero_class: str) -> str:
        """Calculate the hero synergy level for visual cue color coding."""
        if not hero_class:
            return "neutral"
            
        # Check card class vs hero class
        card_class = card.get('card_class', '').upper()
        if card_class == hero_class.upper():
            return "excellent"  # Same class = excellent synergy
        elif card_class == 'NEUTRAL':
            # For neutral cards, check for specific synergies
            card_name = card.get('card_code', '').lower()
            
            # Hero-specific synergy detection
            hero_synergies = {
                'WARRIOR': ['weapon', 'armor', 'rush', 'taunt'],
                'PALADIN': ['divine', 'shield', 'heal', 'buff'],
                'HUNTER': ['beast', 'secret', 'face', 'damage'],
                'ROGUE': ['stealth', 'combo', 'weapon', 'secret'],
                'PRIEST': ['heal', 'mind', 'control', 'divine'],
                'SHAMAN': ['elemental', 'overload', 'totem', 'lightning'],
                'MAGE': ['spell', 'frost', 'fire', 'arcane', 'secret'],
                'WARLOCK': ['demon', 'discard', 'sacrifice', 'life'],
                'DRUID': ['beast', 'nature', 'ramp', 'choose'],
                'DEMONHUNTER': ['demon', 'outcast', 'attack', 'aggro']
            }
            
            synergy_keywords = hero_synergies.get(hero_class.upper(), [])
            for keyword in synergy_keywords:
                if keyword in card_name:
                    return "good"
                    
            return "neutral"
        else:
            return "poor"  # Different class = poor synergy
    
    def _get_hero_synergy_colors(self, synergy_level: str, is_recommended: bool) -> tuple:
        """Get background color and border style based on hero synergy level."""
        if is_recommended:
            # Recommended cards always use success color
            return self.config.success_color, "raised"
        
        color_mapping = {
            "excellent": "#8e44ad",  # Purple for class cards
            "good": "#3498db",       # Blue for good synergy
            "neutral": self.config.background_color,  # Default
            "poor": "#7f8c8d"        # Gray for poor synergy
        }
        
        border_mapping = {
            "excellent": "raised",
            "good": "ridge", 
            "neutral": "flat",
            "poor": "sunken"
        }
        
        return color_mapping.get(synergy_level, self.config.background_color), border_mapping.get(synergy_level, "flat")
    
    def _get_synergy_indicator(self, synergy_level: str) -> str:
        """Get emoji indicator for hero synergy level."""
        indicators = {
            "excellent": "⭐",  # Star for class cards
            "good": "🔹",       # Blue diamond for good synergy
            "neutral": "",      # No indicator for neutral
            "poor": "❓"        # Question mark for poor synergy
        }
        return indicators.get(synergy_level, "")
    
    def update_hero_selection_display(self, hero_analysis: Dict):
        """Update the display for hero selection with AI v2 statistical backing."""
        # Switch to hero selection mode with dynamic adaptation
        self.transition_to_phase("hero_selection")
        self.phase_label.config(
            text="👑 Phase: Hero Selection",
            fg=self.config.hero_color
        )
        
        # Hide card frame and show hero frame
        self.recommendation_frame.pack_forget()
        self.hero_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Clear existing content
        for widget in self.hero_frame.winfo_children():
            widget.destroy()
        
        if not hero_analysis or not hero_analysis.get('hero_classes'):
            error_label = tk.Label(
                self.hero_frame,
                text="❌ No hero choices detected",
                bg=self.config.background_color,
                fg=self.config.highlight_color,
                font=("Arial", 10)
            )
            error_label.pack(pady=20)
            return
        
        # Header with confidence
        confidence = hero_analysis.get('confidence_level', 0.0)
        confidence_color = (
            self.config.success_color if confidence > 0.7 
            else self.config.hero_color if confidence > 0.5 
            else self.config.highlight_color
        )
        
        header_text = f"👑 AI v2 HERO RECOMMENDATION"
        header_label = tk.Label(
            self.hero_frame,
            text=header_text,
            bg=self.config.background_color,
            fg=self.config.hero_color,
            font=("Arial", 12, "bold")
        )
        header_label.pack(pady=(0, 5))
        
        # Enhanced confidence visualization with statistical significance
        if self.config.show_confidence_indicators:
            self.create_confidence_visualization(confidence, hero_analysis)
        
        # Recommended hero highlight
        recommended_index = hero_analysis.get('recommended_hero_index', 0)
        hero_classes = hero_analysis.get('hero_classes', [])
        hero_winrates = hero_analysis.get('winrates', {})
        
        if recommended_index < len(hero_classes):
            recommended_hero = hero_classes[recommended_index]
            recommended_winrate = hero_winrates.get(recommended_hero, 50.0)
            
            # Prominent recommendation display
            rec_frame = tk.Frame(
                self.hero_frame,
                bg=self.config.hero_color,
                relief="raised",
                bd=3
            )
            rec_frame.pack(fill="x", pady=10, padx=5)
            
            rec_text = f"PICK: {recommended_hero}"
            rec_label = tk.Label(
                rec_frame,
                text=rec_text,
                bg=self.config.hero_color,
                fg="white",
                font=("Arial", 14, "bold")
            )
            rec_label.pack(pady=5)
            
            if self.config.show_hero_winrates and recommended_winrate > 0:
                winrate_text = f"Win Rate: {recommended_winrate:.1f}%"
                winrate_label = tk.Label(
                    rec_frame,
                    text=winrate_text,
                    bg=self.config.hero_color,
                    fg="white",
                    font=("Arial", 11)
                )
                winrate_label.pack()
        
        # All hero options with detailed analysis
        analysis_data = hero_analysis.get('hero_analysis', [])
        for i, hero_class in enumerate(hero_classes):
            is_recommended = (i == recommended_index)
            winrate = hero_winrates.get(hero_class, 50.0)
            
            # Individual hero frame
            hero_option_frame = tk.Frame(
                self.hero_frame,
                bg=self.config.success_color if is_recommended else self.config.background_color,
                relief="raised" if is_recommended else "flat",
                bd=2 if is_recommended else 1
            )
            hero_option_frame.pack(fill="x", pady=2, padx=5)
            
            # Hero name and ranking
            rank_text = f"#{i+1}. {hero_class}"
            if is_recommended:
                rank_text += " ⭐"
            
            name_label = tk.Label(
                hero_option_frame,
                text=rank_text,
                bg=self.config.success_color if is_recommended else self.config.background_color,
                fg="white" if is_recommended else self.config.text_color,
                font=("Arial", 10, "bold" if is_recommended else "normal")
            )
            name_label.pack(anchor="w", padx=5, pady=2)
            
            # Add comprehensive hero tooltip
            hero_tooltip_text = f"Hero Analysis: {hero_class}"
            hero_context_data = {
                'class': hero_class,
                'winrate': winrate,
                'profile': hero_info.get('profile', {}) if hero_info else {},
                'meta_position': hero_info.get('meta_position', 'Unknown') if hero_info else 'Unknown'
            }
            
            card_context_data = {
                'confidence': hero_info.get('confidence', 0.8) if hero_info else 0.8,
                'hsreplay_data': True,
                'ai_analysis': True,
                'analysis_time': 45.0  # Hero analysis is typically faster
            }
            
            StatisticalTooltip(hero_option_frame, hero_tooltip_text, hero_context_data, card_context_data)
            
            # Statistical information
            if self.config.show_hero_winrates and winrate > 0:
                # Find specific analysis for this hero
                hero_info = None
                for analysis in analysis_data:
                    if analysis.get('class') == hero_class:
                        hero_info = analysis
                        break
                
                stats_parts = [f"Win Rate: {winrate:.1f}%"]
                
                if hero_info:
                    profile = hero_info.get('profile', {})
                    if profile.get('playstyle'):
                        stats_parts.append(f"Style: {profile['playstyle']}")
                    if profile.get('complexity'):
                        stats_parts.append(f"Complexity: {profile['complexity']}")
                
                stats_text = " | ".join(stats_parts)
                stats_label = tk.Label(
                    hero_option_frame,
                    text=stats_text,
                    bg=self.config.success_color if is_recommended else self.config.background_color,
                    fg="white" if is_recommended else "#bdc3c7",
                    font=("Arial", 8)
                )
                stats_label.pack(anchor="w", padx=5)
                
                # AI explanation if available
                if hero_info and hero_info.get('explanation'):
                    explanation_text = hero_info['explanation'][:80] + "..." if len(hero_info['explanation']) > 80 else hero_info['explanation']
                    explanation_label = tk.Label(
                        hero_option_frame,
                        text=f"📝 {explanation_text}",
                        bg=self.config.success_color if is_recommended else self.config.background_color,
                        fg="white" if is_recommended else "#95a5a6",
                        font=("Arial", 8),
                        wraplength=300,
                        justify="left"
                    )
                    explanation_label.pack(anchor="w", padx=5, pady=(0, 2))
        
        # Update status
        self.status_label.config(text=f"👑 Hero analysis complete - {len(hero_classes)} options")
    
    def create_confidence_visualization(self, confidence: float, hero_analysis: Dict):
        """Create advanced confidence visualization with statistical significance indicators."""
        # Main confidence frame
        confidence_frame = tk.Frame(self.hero_frame, bg=self.config.background_color)
        confidence_frame.pack(fill="x", padx=10, pady=5)
        
        # Confidence level indicator
        confidence_color = (
            self.config.success_color if confidence > 0.7 
            else self.config.hero_color if confidence > 0.5 
            else self.config.highlight_color
        )
        
        confidence_text = f"📊 Analysis Confidence: {confidence:.1%}"
        confidence_label = tk.Label(
            confidence_frame,
            text=confidence_text,
            bg=self.config.background_color,
            fg=confidence_color,
            font=("Arial", 10, "bold")
        )
        confidence_label.pack(anchor="w")
        
        # Confidence bar visualization
        bar_frame = tk.Frame(confidence_frame, bg=self.config.background_color)
        bar_frame.pack(fill="x", pady=2)
        
        # Create confidence bar with segments
        bar_width = 200
        bar_height = 8
        
        # Background bar
        bg_bar = tk.Frame(
            bar_frame,
            width=bar_width,
            height=bar_height,
            bg="#34495e",
            relief="sunken",
            bd=1
        )
        bg_bar.pack(side="left")
        bg_bar.pack_propagate(False)
        
        # Confidence fill bar
        fill_width = int(bar_width * confidence)
        fill_color = confidence_color
        
        fill_bar = tk.Frame(
            bg_bar,
            width=fill_width,
            height=bar_height-2,
            bg=fill_color,
            relief="flat"
        )
        fill_bar.place(x=1, y=1)
        
        # Confidence level text
        conf_level_text = self.get_confidence_level_description(confidence)
        level_label = tk.Label(
            bar_frame,
            text=conf_level_text,
            bg=self.config.background_color,
            fg=confidence_color,
            font=("Arial", 9)
        )
        level_label.pack(side="left", padx=10)
        
        # Statistical significance indicators
        self.create_statistical_significance_display(confidence_frame, hero_analysis)
    
    def create_statistical_significance_display(self, parent, hero_analysis: Dict):
        """Create display showing statistical significance of the hero recommendation."""
        sig_frame = tk.Frame(parent, bg=self.config.background_color)
        sig_frame.pack(fill="x", pady=3)
        
        # Get winrate data for significance calculation
        hero_classes = hero_analysis.get('hero_classes', [])
        winrates = hero_analysis.get('winrates', {})
        
        if len(hero_classes) >= 2 and len(winrates) >= 2:
            # Calculate winrate differences for significance
            winrate_values = [winrates.get(hero, 50.0) for hero in hero_classes]
            winrate_values.sort(reverse=True)
            
            if len(winrate_values) >= 2:
                top_winrate = winrate_values[0]
                second_winrate = winrate_values[1]
                winrate_diff = top_winrate - second_winrate
                
                # Determine statistical significance
                significance_level = self.calculate_statistical_significance(winrate_diff, len(hero_classes))
                
                # Display significance indicator
                sig_text, sig_color, sig_icon = self.get_significance_indicator(significance_level)
                
                sig_label = tk.Label(
                    sig_frame,
                    text=f"{sig_icon} Statistical Significance: {sig_text}",
                    bg=self.config.background_color,
                    fg=sig_color,
                    font=("Arial", 9)
                )
                sig_label.pack(anchor="w")
                
                # Additional statistical details
                details_text = f"📈 Win Rate Difference: {winrate_diff:.1f}% | Sample Reliability: {self.get_sample_reliability_text(hero_analysis)}"
                details_label = tk.Label(
                    sig_frame,
                    text=details_text,
                    bg=self.config.background_color,
                    fg="#95a5a6",
                    font=("Arial", 8)
                )
                details_label.pack(anchor="w")
    
    def get_confidence_level_description(self, confidence: float) -> str:
        """Get descriptive text for confidence level."""
        if confidence >= 0.9:
            return "Extremely High"
        elif confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def calculate_statistical_significance(self, winrate_diff: float, sample_size: int) -> str:
        """Calculate statistical significance level based on winrate difference and sample size."""
        # Simplified significance calculation
        # In a real implementation, this would use proper statistical tests
        
        if winrate_diff >= 5.0 and sample_size >= 3:
            return "high"
        elif winrate_diff >= 3.0 and sample_size >= 3:
            return "moderate"
        elif winrate_diff >= 1.0:
            return "low"
        else:
            return "none"
    
    def get_significance_indicator(self, significance_level: str) -> tuple:
        """Get indicator text, color, and icon for significance level."""
        indicators = {
            "high": ("Highly Significant", "#27ae60", "🟢"),
            "moderate": ("Moderately Significant", "#f39c12", "🟡"),
            "low": ("Low Significance", "#e67e22", "🟠"),
            "none": ("Not Significant", "#95a5a6", "⚪")
        }
        
        return indicators.get(significance_level, ("Unknown", "#95a5a6", "❓"))
    
    def get_sample_reliability_text(self, hero_analysis: Dict) -> str:
        """Get sample reliability description based on available data."""
        # Check data freshness and source reliability
        confidence = hero_analysis.get('confidence_level', 0.0)
        
        if confidence >= 0.8:
            return "Excellent"
        elif confidence >= 0.6:
            return "Good"
        elif confidence >= 0.4:
            return "Fair"
        else:
            return "Limited"
    
    def transition_to_phase(self, new_phase: str):
        """Smoothly transition to a new draft phase with dynamic overlay adaptation."""
        if new_phase == self.draft_phase:
            return  # No change needed
        
        old_phase = self.draft_phase
        self.draft_phase = new_phase
        
        self.logger.info(f"Transitioning overlay from {old_phase} to {new_phase}")
        
        # Update window size and position for new phase
        self.update_window_size_for_phase(new_phase)
        
        # Update UI elements based on phase
        self.adapt_ui_for_phase(new_phase)
        
        # Update system health indicator
        self.update_phase_specific_health_indicator(new_phase)
        
        # Log transition for debugging
        self.status_label.config(text=f"🔄 Transitioned to {new_phase.replace('_', ' ').title()} phase")
    
    def adapt_ui_for_phase(self, phase: str):
        """Adapt UI elements for the specific draft phase."""
        if phase == "waiting":
            # Minimal UI for waiting state
            self.hide_all_analysis_frames()
            self.phase_label.config(
                text="📋 Phase: Waiting for Draft",
                fg=self.config.text_color
            )
            
        elif phase == "hero_selection":
            # Optimize UI for hero selection
            self.hide_all_analysis_frames()
            self.phase_label.config(
                text="👑 Phase: Hero Selection",
                fg=self.config.hero_color
            )
            
        elif phase == "card_picks":
            # Optimize UI for card recommendations
            self.hide_all_analysis_frames()
            hero_text = f" ({self.selected_hero})" if self.selected_hero else ""
            self.phase_label.config(
                text=f"🃏 Phase: Card Picks{hero_text}",
                fg=self.config.success_color
            )
    
    def hide_all_analysis_frames(self):
        """Hide all analysis frames for clean transitions."""
        if hasattr(self, 'hero_frame') and self.hero_frame:
            self.hero_frame.pack_forget()
        if hasattr(self, 'recommendation_frame') and self.recommendation_frame:
            self.recommendation_frame.pack_forget()
    
    def update_phase_specific_health_indicator(self, phase: str):
        """Update system health indicator with phase-specific information."""
        if not self.system_health_indicator:
            return
        
        phase_indicators = {
            "waiting": "🔍 Monitoring for Draft Start",
            "hero_selection": "👑 Hero Analysis Active",
            "card_picks": "🃏 Card Analysis Active"
        }
        
        indicator_text = phase_indicators.get(phase, "🤖 AI v2 Active")
        
        # Update text while preserving health color
        current_fg = self.system_health_indicator.cget("fg")
        self.system_health_indicator.config(text=indicator_text, fg=current_fg)
    
    def get_optimal_opacity_for_phase(self, phase: str) -> float:
        """Get optimal opacity level for different draft phases."""
        opacity_configs = {
            "waiting": 0.7,        # Less prominent while waiting
            "hero_selection": 0.9,  # Highly visible for important hero choice
            "card_picks": 0.85     # Prominent but not overwhelming
        }
        
        return opacity_configs.get(phase, self.config.opacity)
    
    def adapt_opacity_for_phase(self, phase: str):
        """Dynamically adapt overlay opacity based on draft phase importance."""
        if not self.root:
            return
            
        optimal_opacity = self.get_optimal_opacity_for_phase(phase)
        self.root.attributes('-alpha', optimal_opacity)
        
        # Update config for consistency
        self.config.opacity = optimal_opacity
    
    # === PERFORMANCE OPTIMIZATION METHODS ===
    
    def optimized_update_display(self, analysis: Dict, display_type: str = "card"):
        """Performance-optimized display update with batch rendering and caching."""
        if not self.should_render_now():
            self.queue_render_operation(analysis, display_type)
            return
            
        render_start = self.performance_monitor.start_render()
        
        try:
            if display_type == "hero":
                self.update_hero_selection_display(analysis)
            else:
                self.update_card_recommendation_display(analysis)
                
            # Check if optimization is needed
            if self.performance_monitor.is_performance_degraded():
                self.apply_performance_optimizations()
                
        finally:
            render_time = self.performance_monitor.end_render(render_start)
            self.last_render_time = time.time() * 1000
            
            if render_time > 30:
                self.logger.warning(f"Slow render detected: {render_time:.1f}ms")
    
    def should_render_now(self) -> bool:
        """Check if enough time has passed to allow rendering."""
        if not self.batch_rendering_enabled:
            return True
            
        current_time = time.time() * 1000
        return (current_time - self.last_render_time) >= self.min_render_interval
    
    def queue_render_operation(self, analysis: Dict, display_type: str):
        """Queue a render operation for batch processing."""
        self.render_queue.append({
            'analysis': analysis,
            'display_type': display_type,
            'timestamp': time.time()
        })
        
        # Process queue if it gets too long
        if len(self.render_queue) > 5:
            self.process_render_queue()
    
    def process_render_queue(self):
        """Process queued render operations in batch."""
        if not self.render_queue:
            return
            
        # Take the most recent operation of each type
        latest_operations = {}
        for op in self.render_queue:
            display_type = op['display_type']
            if display_type not in latest_operations or op['timestamp'] > latest_operations[display_type]['timestamp']:
                latest_operations[display_type] = op
        
        # Execute latest operations
        for op in latest_operations.values():
            if op['display_type'] == "hero":
                self.update_hero_selection_display(op['analysis'])
            else:
                self.update_card_recommendation_display(op['analysis'])
        
        # Clear queue
        self.render_queue.clear()
    
    def get_cached_ui_element(self, cache_key: str, create_func):
        """Get UI element from cache or create and cache it."""
        if not self.ui_caching_enabled:
            return create_func()
            
        if cache_key in self.ui_cache:
            return self.ui_cache[cache_key]
        
        element = create_func()
        self.ui_cache[cache_key] = element
        return element
    
    def clear_ui_cache(self):
        """Clear UI element cache to free memory."""
        self.ui_cache.clear()
        self.logger.info("UI cache cleared for memory optimization")
    
    def apply_performance_optimizations(self):
        """Apply automatic performance optimizations when degradation is detected."""
        if self.performance_monitor.optimization_active:
            return  # Already optimizing
            
        self.performance_monitor.optimization_active = True
        self.logger.warning("Performance degradation detected, applying optimizations")
        
        # Get and apply recommendations
        recommendations = self.performance_monitor.get_optimization_recommendations()
        
        for recommendation in recommendations:
            if "batch rendering" in recommendation.lower():
                self.batch_rendering_enabled = True
                self.min_render_interval = 150  # Increase interval
                
            elif "ui caching" in recommendation.lower():
                self.ui_caching_enabled = True
                
            elif "smooth animations" in recommendation.lower():
                self.smooth_animations_enabled = False
                
            elif "tooltip complexity" in recommendation.lower():
                self.reduce_tooltip_complexity()
        
        # Update status to show optimization is active
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.config(text="⚡ Performance optimizations active")
    
    def reduce_tooltip_complexity(self):
        """Reduce tooltip complexity for performance."""
        # Simplify tooltips by reducing sections
        self.config.show_confidence_indicators = False
        self.logger.info("Reduced tooltip complexity for performance")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics for monitoring."""
        return {
            'average_render_time_ms': self.performance_monitor.get_average_render_time(),
            'total_renders': self.performance_monitor.total_renders,
            'frame_drops': self.performance_monitor.frame_drops,
            'frame_drop_rate': self.performance_monitor.frame_drops / max(1, self.performance_monitor.total_renders),
            'optimization_active': self.performance_monitor.optimization_active,
            'cache_size': len(self.ui_cache),
            'render_queue_size': len(self.render_queue),
            'batch_rendering_enabled': self.batch_rendering_enabled,
            'ui_caching_enabled': self.ui_caching_enabled,
            'smooth_animations_enabled': self.smooth_animations_enabled
        }
    
    def reset_performance_optimizations(self):
        """Reset performance optimizations to default settings."""
        self.batch_rendering_enabled = True
        self.ui_caching_enabled = True
        self.smooth_animations_enabled = True
        self.min_render_interval = 100
        self.config.show_confidence_indicators = True
        
        self.performance_monitor.optimization_active = False
        self.clear_ui_cache()
        
        self.logger.info("Performance optimizations reset to defaults")
    
    def optimize_for_hero_mode(self):
        """Apply specific optimizations for hero selection mode."""
        # Hero selection typically has fewer elements, can enable higher quality
        self.config.show_confidence_indicators = True
        self.config.show_hero_winrates = True
        self.min_render_interval = 80  # Faster updates for important hero choice
    
    def optimize_for_card_mode(self):
        """Apply specific optimizations for card selection mode."""
        # Card selection happens more frequently, optimize for speed
        if self.performance_monitor.is_performance_degraded():
            self.min_render_interval = 120  # Slower updates
            
    def schedule_periodic_optimization_check(self):
        """Schedule periodic performance optimization checks."""
        def check_performance():
            if self.running:
                metrics = self.get_performance_metrics()
                
                # Log performance status periodically
                if metrics['total_renders'] % 50 == 0:
                    self.logger.info(f"Performance: {metrics['average_render_time_ms']:.1f}ms avg, "
                                   f"{metrics['frame_drop_rate']:.1%} drops")
                
                # Schedule next check
                if self.root:
                    self.root.after(5000, check_performance)  # Check every 5 seconds
        
        if self.root:
            self.root.after(5000, check_performance)
    
    def update_card_recommendation_display(self, analysis: Dict):
        """Update the display for card recommendations with hero context."""
        # Switch to card selection mode with dynamic adaptation
        self.transition_to_phase("card_picks")
        self.phase_label.config(
            text=f"🃏 Phase: Card Picks{f' ({self.selected_hero})' if self.selected_hero else ''}",
            fg=self.config.success_color
        )
        
        # Hide hero frame and show card frame
        self.hero_frame.pack_forget()
        self.recommendation_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Use the existing card display logic with enhancements
        self.update_recommendation_display(analysis)
    
    def set_selected_hero(self, hero_class: str):
        """Set the selected hero for hero-aware card recommendations."""
        self.selected_hero = hero_class
        self.logger.info(f"Selected hero set to: {hero_class}")
    
    def update_system_health(self, health_status: Dict):
        """Update the AI v2 system health indicator."""
        if not self.system_health_indicator:
            return
            
        overall_status = health_status.get('overall_status', 'unknown')
        
        if overall_status == 'online':
            self.system_health_indicator.config(
                text="🟢 AI v2 Systems Online",
                fg=self.config.success_color
            )
        elif overall_status == 'degraded':
            self.system_health_indicator.config(
                text="🟡 AI v2 Systems Degraded",
                fg=self.config.hero_color
            )
        else:
            self.system_health_indicator.config(
                text="🔴 AI v2 Systems Offline",
                fg=self.config.highlight_color
            )
    
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
        
        # Start performance monitoring
        self.schedule_periodic_optimization_check()
        
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