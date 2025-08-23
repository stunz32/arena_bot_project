"""
UI Safe Demo Mode

Provides diagnostic UI rendering independent of CV/AI data.
Always renders visible elements to prevent blue screen issues.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime
from typing import Optional, List, Tuple
import math


class SafeDemoRenderer:
    """Renders diagnostic UI elements for Safe Demo mode."""
    
    def __init__(self, root_window: tk.Tk, ui_health_reporter):
        """
        Initialize Safe Demo renderer.
        
        Args:
            root_window: Main tkinter window
            ui_health_reporter: UI health reporter instance
        """
        self.root_window = root_window
        self.ui_health_reporter = ui_health_reporter
        self.demo_active = False
        self.animation_thread = None
        self.stop_animation = threading.Event()
        
        # Animation state
        self.frame_counter = 0
        self.start_time = time.time()
        self.fps_values = []
        
        # Demo widgets
        self.watermark_label = None
        self.animation_label = None
        self.fps_label = None
        self.guide_frame = None
        
    def start_demo_mode(self):
        """Start Safe Demo rendering mode."""
        if self.demo_active:
            return
            
        self.demo_active = True
        self.stop_animation.clear()
        
        # Create demo UI elements
        self._create_demo_elements()
        
        # Start animation using Tk's after() method for thread safety
        self._schedule_next_animation()
        
        print("🎨 UI Safe Demo mode activated")
        
    def stop_demo_mode(self):
        """Stop Safe Demo rendering mode."""
        if not self.demo_active:
            return
            
        self.demo_active = False
        self.stop_animation.set()
        
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=1.0)
        
        # Clean up demo elements
        self._cleanup_demo_elements()
        
        print("🎨 UI Safe Demo mode deactivated")
    
    def _create_demo_elements(self):
        """Create diagnostic UI elements."""
        try:
            # Watermark label
            self.watermark_label = tk.Label(
                self.root_window,
                text="🎯 ARENA ASSISTANT DEMO MODE",
                font=('Arial', 16, 'bold'),
                fg='#FF6B35',  # Bright orange for visibility
                bg='#2C3E50',
                relief='raised',
                bd=2
            )
            self.watermark_label.place(x=20, y=20)
            
            # Animated indicator
            self.animation_label = tk.Label(
                self.root_window,
                text="⚡ LIVE",
                font=('Arial', 12, 'bold'),
                fg='#27AE60',  # Green for active status
                bg='#2C3E50'
            )
            self.animation_label.place(x=20, y=60)
            
            # FPS counter
            self.fps_label = tk.Label(
                self.root_window,
                text="FPS: 0.0",
                font=('Arial', 10),
                fg='#3498DB',  # Blue for technical info
                bg='#2C3E50'
            )
            self.fps_label.place(x=20, y=90)
            
            # Guide rectangles for card slots
            self._create_guide_rectangles()
            
            # Force initial paint
            self.root_window.update_idletasks()
            
        except Exception as e:
            print(f"❌ Failed to create demo elements: {e}")
    
    def _create_guide_rectangles(self):
        """Create guide rectangles showing where card slots would be."""
        try:
            # Create frame for guides
            self.guide_frame = tk.Frame(self.root_window, bg='#2C3E50')
            self.guide_frame.place(x=300, y=150, width=800, height=400)
            
            # Card slot guides
            card_positions = [
                (50, 50, 200, 300),   # Card 1
                (300, 50, 200, 300),  # Card 2  
                (550, 50, 200, 300)   # Card 3
            ]
            
            for i, (x, y, w, h) in enumerate(card_positions):
                # Card outline
                card_frame = tk.Frame(
                    self.guide_frame,
                    bg='#34495E',
                    relief='solid',
                    bd=2,
                    width=w,
                    height=h
                )
                card_frame.place(x=x, y=y)
                card_frame.pack_propagate(False)
                
                # Card label
                card_label = tk.Label(
                    card_frame,
                    text=f"CARD SLOT {i+1}\\n\\n📄\\n\\nDemo Card\\nDetection Area",
                    font=('Arial', 10, 'bold'),
                    fg='#BDC3C7',
                    bg='#34495E',
                    justify='center'
                )
                card_label.pack(expand=True)
                
                # Animated border for visibility
                if i == 0:  # Animate first card slot
                    card_frame.config(highlightbackground='#E74C3C', highlightthickness=2)
            
        except Exception as e:
            print(f"❌ Failed to create guide rectangles: {e}")
    
    def _cleanup_demo_elements(self):
        """Clean up demo UI elements."""
        try:
            widgets_to_destroy = [
                self.watermark_label,
                self.animation_label, 
                self.fps_label,
                self.guide_frame
            ]
            
            for widget in widgets_to_destroy:
                if widget and widget.winfo_exists():
                    widget.destroy()
                    
            # Clear references
            self.watermark_label = None
            self.animation_label = None
            self.fps_label = None
            self.guide_frame = None
            
        except Exception as e:
            print(f"❌ Failed to cleanup demo elements: {e}")
    
    def _animation_loop(self):
        """Animation loop for indicators."""
        last_update = time.time()
        
        while not self.stop_animation.is_set():
            try:
                current_time = time.time()
                
                # Update frame counter and FPS
                self.frame_counter += 1
                if self.ui_health_reporter:
                    self.ui_health_reporter.increment_paint_counter()
                
                # Calculate FPS
                elapsed = current_time - self.start_time
                if elapsed > 0:
                    current_fps = self.frame_counter / elapsed
                    self.fps_values.append(current_fps)
                    
                    # Keep only recent FPS values
                    if len(self.fps_values) > 30:
                        self.fps_values = self.fps_values[-30:]
                    
                    avg_fps = sum(self.fps_values) / len(self.fps_values)
                else:
                    avg_fps = 0
                
                # Update UI elements on main thread
                self.root_window.after_idle(self._update_animation_elements, current_time, avg_fps)
                
                # Control animation framerate
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Animation loop error: {e}")
                break
    
    def _update_animation_elements(self, current_time: float, fps: float):
        """Update animated elements (called on main thread)."""
        try:
            if not self.demo_active:
                return
                
            # Animated dot indicator
            if self.animation_label and self.animation_label.winfo_exists():
                dot_cycle = int(current_time * 2) % 4
                dots = "." * dot_cycle + " " * (3 - dot_cycle)
                self.animation_label.config(text=f"⚡ LIVE{dots}")
            
            # FPS display
            if self.fps_label and self.fps_label.winfo_exists():
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            # Pulsing watermark
            if self.watermark_label and self.watermark_label.winfo_exists():
                pulse_cycle = (math.sin(current_time * 2) + 1) / 2  # 0 to 1
                if pulse_cycle > 0.5:
                    self.watermark_label.config(fg='#FF6B35')  # Orange
                else:
                    self.watermark_label.config(fg='#E74C3C')  # Red
            
        except Exception as e:
            print(f"❌ Failed to update animation elements: {e}")
    
    def force_paint_event(self):
        """Force a paint event to ensure visibility."""
        try:
            if self.demo_active and self.root_window:
                self.root_window.update_idletasks()
                if self.ui_health_reporter:
                    self.ui_health_reporter.increment_paint_counter()
        except Exception as e:
            print(f"❌ Failed to force paint event: {e}")
    
    def _schedule_next_animation(self):
        """Schedule the next animation update using Tk's after() method."""
        if not self.demo_active:
            return
            
        try:
            # Schedule next update in ~33ms (~30 FPS)
            self.root_window.after(33, self._tk_animation_update)
        except Exception as e:
            print(f"❌ Failed to schedule animation: {e}")
    
    def _tk_animation_update(self):
        """Animation update called via Tk's after() method - thread safe."""
        if not self.demo_active:
            return
            
        try:
            current_time = time.time()
            
            # Update frame counter and FPS
            self.frame_counter += 1
            if self.ui_health_reporter:
                self.ui_health_reporter.increment_paint_counter()
            
            # Calculate FPS
            elapsed = current_time - self.start_time
            if elapsed > 0:
                current_fps = self.frame_counter / elapsed
                self.fps_values.append(current_fps)
                
                # Keep only recent FPS values
                if len(self.fps_values) > 30:
                    self.fps_values = self.fps_values[-30:]
                
                avg_fps = sum(self.fps_values) / len(self.fps_values)
            else:
                avg_fps = 0
            
            # Update animation elements directly (we're on the main thread)
            self._update_animation_elements(current_time, avg_fps)
            
            # Schedule next update
            self._schedule_next_animation()
            
        except Exception as e:
            print(f"❌ Tk animation update error: {e}")
    
    def trigger_repaint(self):
        """Trigger a repaint for resize events."""
        self.force_paint_event()


class SafeDemoManager:
    """Manages Safe Demo mode for the entire application."""
    
    def __init__(self, root_window: tk.Tk = None):
        """
        Initialize Safe Demo manager.
        
        Args:
            root_window: Main application window
        """
        self.root_window = root_window
        self.renderer = None
        self.enabled = False
        
    def enable_safe_demo(self, ui_health_reporter=None):
        """Enable Safe Demo mode."""
        if self.enabled or not self.root_window:
            return
            
        self.enabled = True
        self.renderer = SafeDemoRenderer(self.root_window, ui_health_reporter)
        self.renderer.start_demo_mode()
        
        # Ensure immediate visibility
        self._ensure_visibility()
        
    def disable_safe_demo(self):
        """Disable Safe Demo mode."""
        if not self.enabled:
            return
            
        if self.renderer:
            self.renderer.stop_demo_mode()
            self.renderer = None
            
        self.enabled = False
    
    def _ensure_visibility(self):
        """Ensure the window has visible content."""
        try:
            if self.root_window:
                # Force window to front
                self.root_window.lift()
                self.root_window.focus_force()
                
                # Ensure non-transparent background
                self.root_window.config(bg='#2C3E50')
                
                # Force redraw
                self.root_window.update()
                
        except Exception as e:
            print(f"❌ Failed to ensure visibility: {e}")
    
    def is_enabled(self) -> bool:
        """Check if Safe Demo mode is enabled."""
        return self.enabled
    
    def force_paint(self):
        """Force a paint event."""
        if self.renderer:
            self.renderer.force_paint_event()
    
    def trigger_repaint(self):
        \"\"\"Trigger a repaint for resize events.\"\"\"
        if self.renderer:
            self.renderer.trigger_repaint()
    
    def start_demo_mode(self):
        \"\"\"Start Safe Demo mode if not already active.\"\"\"
        if self.renderer:
            self.renderer.start_demo_mode()


# Global demo manager instance
_demo_manager = None


def get_demo_manager(root_window: tk.Tk = None) -> SafeDemoManager:
    """Get or create the global demo manager."""
    global _demo_manager
    if _demo_manager is None:
        _demo_manager = SafeDemoManager(root_window)
    elif root_window and not _demo_manager.root_window:
        _demo_manager.root_window = root_window
    return _demo_manager


def enable_safe_demo_mode(root_window: tk.Tk, ui_health_reporter=None):
    """Enable Safe Demo mode globally."""
    demo_manager = get_demo_manager(root_window)
    demo_manager.enable_safe_demo(ui_health_reporter)


def disable_safe_demo_mode():
    """Disable Safe Demo mode globally."""
    demo_manager = get_demo_manager()
    demo_manager.disable_safe_demo()


def is_safe_demo_active() -> bool:
    """Check if Safe Demo mode is active."""
    demo_manager = get_demo_manager()
    return demo_manager.is_enabled()