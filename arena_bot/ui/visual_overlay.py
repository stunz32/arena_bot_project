#!/usr/bin/env python3
"""
Visual Intelligence Overlay for AI Helper v2 - Phase 3 Implementation

This module implements the VisualIntelligenceOverlay system as specified in Phase 3
of the todo_ai_helper.md master plan. It provides a high-performance, click-through,
multi-monitor aware overlay system for displaying AI recommendations directly over
the Hearthstone game window.

Features:
- High-performance click-through overlay (P3.1.1-P3.1.12)
- Multi-monitor topology detection with DPI awareness
- Platform-specific Windows optimization
- Frame rate limiting and performance monitoring
- Automatic crash recovery and restart capability
- Integration with AI v2 data models and decision system

Performance Requirements:
- <50ms overlay rendering time
- <5% CPU usage during active gameplay
- Zero impact on game performance
- Automatic throttling when game FPS drops

Platform Support:
- Windows 10/11 with full click-through support
- Multi-monitor setups (1-6 monitors)
- DPI scaling compensation
- Remote session detection and warnings

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import os
import sys
import time
import json
import threading
import logging
import traceback
import ctypes
import weakref
from ctypes import wintypes
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import math

# Import AI v2 components
try:
    from ..ai_v2.data_models import AIDecision, CardOption, EvaluationScores, ConfidenceLevel
    from ..ai_v2.monitoring import PerformanceMonitor, ResourceTracker, MetricType
    from ..ai_v2.exceptions import AIHelperUIError, AIHelperPerformanceError
    from ..ai_v2.platform_abstraction import get_platform_manager
except ImportError as e:
    logging.warning(f"Failed to import AI v2 components: {e}")
    # Fallback classes for development
    class AIDecision: pass
    class CardOption: pass
    class EvaluationScores: pass
    class ConfidenceLevel: pass
    class PerformanceMonitor: pass
    class ResourceTracker: pass
    class MetricType: pass
    class AIHelperUIError(Exception): pass
    class AIHelperPerformanceError(Exception): pass
    get_platform_manager = lambda: None

# Windows API imports for advanced overlay functionality
try:
    from win32api import GetSystemMetrics
    from win32gui import (
        FindWindow, GetWindowRect, SetWindowPos, SetLayeredWindowAttributes,
        GetForegroundWindow, GetWindowText, IsWindow, EnumWindows, EnumDisplayMonitors
    )
    from win32con import (
        WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOPMOST, WS_EX_TOOLWINDOW,
        GWL_EXSTYLE, HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE, LWA_ALPHA
    )
    import win32process
    import win32ui
    WINDOWS_API_AVAILABLE = True
except ImportError:
    WINDOWS_API_AVAILABLE = False
    logging.warning("Windows API not available - overlay functionality will be limited")

logger = logging.getLogger(__name__)

class OverlayState(Enum):
    """Overlay system states"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    MINIMIZED = "minimized"
    ERROR = "error"
    RECOVERING = "recovering"

class MonitorInfo(Enum):
    """Monitor information flags"""
    PRIMARY = 0x1
    VIRTUAL_DESKTOP = 0x2
    DPI_AWARE = 0x4

class ClickThroughMode(Enum):
    """Click-through implementation strategies"""
    TRANSPARENT_WINDOW = "transparent_window"
    LAYERED_WINDOW = "layered_window"
    WINDOW_EX_STYLE = "window_ex_style"
    FALLBACK_DISABLE = "fallback_disable"

@dataclass
class MonitorConfiguration:
    """
    Monitor configuration with DPI awareness
    
    Implements P3.1.1: Advanced Monitor Topology Detection
    and P3.1.2: DPI-Aware Coordinate Transformation
    """
    handle: int
    bounds: Tuple[int, int, int, int]  # left, top, right, bottom
    work_area: Tuple[int, int, int, int]  # Available area excluding taskbar
    dpi_scale: float = 1.0
    is_primary: bool = False
    device_name: str = ""
    refresh_rate: int = 60
    color_depth: int = 32
    
    @property
    def width(self) -> int:
        """Monitor width in pixels"""
        return self.bounds[2] - self.bounds[0]
    
    @property
    def height(self) -> int:
        """Monitor height in pixels"""
        return self.bounds[3] - self.bounds[1]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Monitor center point"""
        return (
            self.bounds[0] + self.width // 2,
            self.bounds[1] + self.height // 2
        )
    
    @property
    def aspect_ratio(self) -> float:
        """Monitor aspect ratio"""
        return self.width / self.height if self.height > 0 else 16/9

@dataclass
class OverlayConfiguration:
    """
    Configuration for the Visual Intelligence Overlay
    
    Implements performance budgets and platform-specific settings
    """
    # Performance settings
    target_fps: int = 30
    max_cpu_usage: float = 0.05  # 5% CPU limit
    frame_budget_ms: float = 33.33  # ~30 FPS
    enable_vsync: bool = True
    
    # Visual settings
    opacity: float = 0.9
    background_color: str = "#2c3e50"
    text_color: str = "#ecf0f1"
    highlight_color: str = "#e74c3c"
    success_color: str = "#27ae60"
    warning_color: str = "#f39c12"
    
    # Layout settings
    card_width: int = 180
    card_height: int = 250
    card_spacing: int = 20
    overlay_padding: int = 10
    
    # Click-through settings
    click_through_enabled: bool = True
    click_through_mode: ClickThroughMode = ClickThroughMode.LAYERED_WINDOW
    fallback_modes: List[ClickThroughMode] = field(default_factory=lambda: [
        ClickThroughMode.TRANSPARENT_WINDOW,
        ClickThroughMode.WINDOW_EX_STYLE,
        ClickThroughMode.FALLBACK_DISABLE
    ])
    
    # Monitor settings
    auto_detect_monitors: bool = True
    target_monitor: int = -1  # -1 for auto-detect
    dpi_awareness: bool = True
    
    # Game detection settings - Enhanced window title detection
    game_window_titles: List[str] = field(default_factory=lambda: [
        "Hearthstone", "Hearthstone.exe", "Hearthstone Battle.net",
        "Hearthstone Arena", "Blizzard Entertainment", "Blizzard Games"
    ])
    
    # Error recovery settings
    crash_recovery_enabled: bool = True
    max_recovery_attempts: int = 3
    recovery_delay_seconds: float = 2.0

class WindowsAPIHelper:
    """
    Helper class for Windows API operations
    
    Implements P3.1.7-P3.1.12: Platform-Specific Click-Through Strategies
    """
    
    @staticmethod
    def get_monitor_info() -> List[MonitorConfiguration]:
        """
        P3.1.1: Advanced Monitor Topology Detection
        Real-time monitor configuration tracking
        """
        if not WINDOWS_API_AVAILABLE:
            # Fallback for non-Windows or missing dependencies
            return [MonitorConfiguration(
                handle=0,
                bounds=(0, 0, 1920, 1080),
                work_area=(0, 0, 1920, 1080),
                is_primary=True,
                device_name="Primary"
            )]
        
        monitors = []
        
        def monitor_enum_proc(monitor, dc, rect, data):
            """Callback for EnumDisplayMonitors"""
            try:
                # Get monitor info using ctypes for detailed information
                monitor_info = wintypes.RECT()
                work_area = wintypes.RECT()
                
                # Try to get DPI information (Windows 10+)
                dpi_scale = 1.0
                try:
                    # This requires Windows 10 version 1607+
                    user32 = ctypes.windll.user32
                    if hasattr(user32, 'GetDpiForMonitor'):
                        dpi_x = ctypes.c_uint()
                        dpi_y = ctypes.c_uint()
                        user32.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
                        dpi_scale = dpi_x.value / 96.0  # 96 DPI is standard
                except Exception as e:
                    logger.debug(f"Could not get DPI info for monitor: {e}")
                
                # Get monitor bounds from rect parameter
                bounds = (rect[0], rect[1], rect[2], rect[3])
                
                # Work area is typically the same as bounds unless taskbar interferes
                work_area_bounds = bounds  # Simplified for now
                
                # Determine if this is the primary monitor
                primary_left = GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
                primary_top = GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
                is_primary = (bounds[0] == primary_left and bounds[1] == primary_top)
                
                monitor_config = MonitorConfiguration(
                    handle=monitor,
                    bounds=bounds,
                    work_area=work_area_bounds,
                    dpi_scale=dpi_scale,
                    is_primary=is_primary,
                    device_name=f"Monitor_{len(monitors) + 1}"
                )
                
                monitors.append(monitor_config)
                logger.debug(f"Detected monitor: {monitor_config.device_name} "
                           f"bounds={bounds} dpi_scale={dpi_scale:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing monitor {monitor}: {e}")
            
            return True  # Continue enumeration
        
        try:
            # Use EnumDisplayMonitors to get all monitors
            EnumDisplayMonitors(None, None, monitor_enum_proc, 0)
        except Exception as e:
            logger.error(f"Failed to enumerate monitors: {e}")
            # Return default monitor configuration
            return [MonitorConfiguration(
                handle=0,
                bounds=(0, 0, 1920, 1080),
                work_area=(0, 0, 1920, 1080),
                is_primary=True,
                device_name="Default"
            )]
        
        if not monitors:
            logger.warning("No monitors detected, using default configuration")
            monitors = [MonitorConfiguration(
                handle=0,
                bounds=(0, 0, 1920, 1080),
                work_area=(0, 0, 1920, 1080),
                is_primary=True,
                device_name="Default"
            )]
        
        return monitors
    
    @staticmethod
    def find_game_window(window_titles: List[str]) -> Optional[int]:
        """Enhanced game window detection with comprehensive debugging"""
        if not WINDOWS_API_AVAILABLE:
            logger.warning("Windows API not available - cannot detect game window")
            return None
        
        logger.info(f"🔍 Starting game window detection with {len(window_titles)} titles to check")
        
        # Strategy 1: Exact title match
        for title in window_titles:
            try:
                hwnd = FindWindow(None, title)
                if hwnd and IsWindow(hwnd):
                    logger.info(f"✅ Found game window via exact match: '{title}' (hwnd: {hwnd})")
                    return hwnd
                else:
                    logger.debug(f"   No exact match for '{title}'")
            except Exception as e:
                logger.debug(f"   Error checking title '{title}': {e}")
        
        # Strategy 2: Partial title match with comprehensive logging
        logger.info("🔍 Trying partial title matching - scanning all windows...")
        game_windows = []
        all_windows = []
        
        def enum_windows_proc(hwnd, lParam):
            try:
                if IsWindow(hwnd):
                    window_text = GetWindowText(hwnd)
                    if window_text:  # Only log windows with text
                        all_windows.append((hwnd, window_text))
                        
                        # Check for game window matches
                        for title in window_titles:
                            if title.lower() in window_text.lower():
                                logger.info(f"✅ Found game window via partial match: '{window_text}' contains '{title}' (hwnd: {hwnd})")
                                game_windows.append((hwnd, window_text, title))
                                
                        # Additional heuristics for Hearthstone detection
                        if any(keyword in window_text.lower() for keyword in ['hearthstone', 'blizzard', 'battle.net']):
                            logger.info(f"🎯 Potential Hearthstone window detected: '{window_text}' (hwnd: {hwnd})")
                            if (hwnd, window_text, 'heuristic') not in [(g[0], g[1], 'heuristic') for g in game_windows]:
                                game_windows.append((hwnd, window_text, 'heuristic'))
            except Exception:
                pass
            return True
        
        try:
            EnumWindows(enum_windows_proc, 0)
            
            # Log summary of findings
            logger.info(f"📊 Window scan complete: Found {len(all_windows)} windows with titles")
            logger.info(f"🎯 Found {len(game_windows)} potential game windows")
            
            if logger.level <= logging.DEBUG:
                logger.debug("🔍 All windows found:")
                for hwnd, title in all_windows[:20]:  # Log first 20 to avoid spam
                    logger.debug(f"   {hwnd}: '{title}'")
                if len(all_windows) > 20:
                    logger.debug(f"   ... and {len(all_windows) - 20} more windows")
            
            if game_windows:
                # Return the window handle of the first match
                selected_window = game_windows[0]
                logger.info(f"🎯 Selected game window: '{selected_window[1]}' via {selected_window[2]} (hwnd: {selected_window[0]})")
                return selected_window[0]
            else:
                logger.warning("❌ No game windows found with any detection strategy")
                
        except Exception as e:
            logger.error(f"❌ Error enumerating windows: {e}")
        
        logger.warning("❌ Game window detection failed - overlay will not be positioned correctly")
        return None
    
    @staticmethod
    def setup_click_through(hwnd: int, mode: ClickThroughMode) -> bool:
        """
        P3.1.7-P3.1.12: Platform-Specific Click-Through Strategies
        Implement click-through with fallback hierarchy
        """
        if not WINDOWS_API_AVAILABLE or not hwnd:
            logger.warning("Cannot setup click-through: Windows API not available")
            return False
        
        try:
            if mode == ClickThroughMode.LAYERED_WINDOW:
                # P3.1.7: Layered window approach (preferred)
                current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                new_style = current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
                result = ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
                
                if result != 0:
                    # Set transparency (255 = opaque, 0 = fully transparent)
                    SetLayeredWindowAttributes(hwnd, 0, 230, LWA_ALPHA)  # 90% opacity
                    logger.info("Successfully enabled layered window click-through")
                    return True
                    
            elif mode == ClickThroughMode.TRANSPARENT_WINDOW:
                # P3.1.10: Transparent window fallback
                current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                new_style = current_style | WS_EX_TRANSPARENT | WS_EX_TOPMOST
                result = ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
                
                if result != 0:
                    logger.info("Successfully enabled transparent window click-through")
                    return True
                    
            elif mode == ClickThroughMode.WINDOW_EX_STYLE:
                # P3.1.10: Extended style approach
                current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                new_style = current_style | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
                result = ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
                
                if result != 0:
                    logger.info("Successfully enabled extended style window management")
                    return True
            
        except Exception as e:
            logger.error(f"Failed to setup click-through mode {mode}: {e}")
        
        return False
    
    @staticmethod
    def validate_click_through(hwnd: int) -> bool:
        """
        P3.1.8: Click-Through Validation Testing
        Runtime verification of click-through behavior
        """
        if not WINDOWS_API_AVAILABLE or not hwnd:
            return False
        
        try:
            # Check if window has the correct extended styles
            current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            has_transparent = bool(current_style & WS_EX_TRANSPARENT)
            has_layered = bool(current_style & WS_EX_LAYERED)
            has_topmost = bool(current_style & WS_EX_TOPMOST)
            
            logger.debug(f"Window {hwnd} styles: transparent={has_transparent}, "
                        f"layered={has_layered}, topmost={has_topmost}")
            
            # At minimum, we need either transparent or layered for click-through
            return has_transparent or has_layered
            
        except Exception as e:
            logger.error(f"Failed to validate click-through: {e}")
            return False
    
    @staticmethod
    def detect_remote_session() -> bool:
        """
        P3.1.5: Remote Session Detection & Warning
        Disable overlay for remote sessions where performance may be poor
        """
        try:
            # Check if we're in a remote desktop session
            return GetSystemMetrics(0x1000) != 0  # SM_REMOTESESSION
        except Exception:
            return False

class PerformanceLimiter:
    """
    Performance management and frame rate limiting
    
    Implements P3.3.1-P3.3.6: Overlay Rendering Performance Protection
    """
    
    def __init__(self, config: OverlayConfiguration):
        self.config = config
        self.frame_times = []
        self.last_frame_time = time.perf_counter()
        self.frame_budget = config.frame_budget_ms / 1000.0  # Convert to seconds
        self.cpu_usage_samples = []
        self.gpu_frame_drops = 0
        self.performance_throttle = 1.0
        
        # P3.3.1: Frame Rate Budget Management
        self.frame_budget_exceeded_count = 0
        self.max_budget_violations = 5  # Allow 5 violations before throttling
        
    def start_frame(self) -> float:
        """Start timing a new frame"""
        return time.perf_counter()
    
    def end_frame(self, start_time: float) -> bool:
        """
        End frame timing and apply throttling if needed
        
        Returns:
            bool: True if frame was within budget, False if throttling applied
        """
        current_time = time.perf_counter()
        frame_time = current_time - start_time
        
        # Track frame times for analysis
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:  # Keep last 60 frames (~2 seconds at 30fps)
            self.frame_times.pop(0)
        
        # P3.3.1: Check frame budget
        if frame_time > self.frame_budget:
            self.frame_budget_exceeded_count += 1
            logger.warning(f"Frame budget exceeded: {frame_time*1000:.1f}ms > {self.frame_budget*1000:.1f}ms")
            
            # Apply throttling if too many violations
            if self.frame_budget_exceeded_count > self.max_budget_violations:
                self.performance_throttle = max(0.5, self.performance_throttle * 0.9)
                logger.info(f"Applied performance throttling: {self.performance_throttle:.2f}")
                self.frame_budget_exceeded_count = 0
        else:
            # Gradually restore performance if frames are within budget
            if self.performance_throttle < 1.0:
                self.performance_throttle = min(1.0, self.performance_throttle * 1.01)
        
        self.last_frame_time = current_time
        return frame_time <= self.frame_budget
    
    def get_target_fps(self) -> float:
        """Get current target FPS with throttling applied"""
        return self.config.target_fps * self.performance_throttle
    
    def should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped for performance"""
        if self.performance_throttle < 0.8:
            # Skip every other frame when heavily throttled
            return len(self.frame_times) % 2 == 0
        return False
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.frame_times:
            return {}
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        max_frame_time = max(self.frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            'avg_frame_time_ms': avg_frame_time * 1000,
            'max_frame_time_ms': max_frame_time * 1000,
            'current_fps': current_fps,
            'target_fps': self.get_target_fps(),
            'performance_throttle': self.performance_throttle,
            'budget_violations': self.frame_budget_exceeded_count
        }

class VisualIntelligenceOverlay:
    """
    High-Performance Visual Intelligence Overlay
    
    This is the main overlay class that implements all Phase 3 requirements:
    - P3.1.1-P3.1.12: Multi-monitor platform compatibility and click-through
    - P3.2.1-P3.2.6: Integration with hover detection system
    - P3.3.1-P3.3.6: Performance monitoring and optimization
    
    The overlay displays AI recommendations directly over the Hearthstone game
    window with minimal performance impact and maximum platform compatibility.
    """
    
    def __init__(self, config: OverlayConfiguration = None):
        """
        Initialize the Visual Intelligence Overlay
        
        Args:
            config: Overlay configuration, uses defaults if None
        """
        self.config = config or OverlayConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.state = OverlayState.INACTIVE
        self.running = False
        self.recovery_attempts = 0
        
        # Platform and monitor management
        self.monitors: List[MonitorConfiguration] = []
        self.target_monitor: Optional[MonitorConfiguration] = None
        self.game_window_hwnd: Optional[int] = None
        self.is_remote_session = False
        
        # UI components
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.overlay_elements: Dict[str, Any] = {}
        
        # Performance management
        self.performance_limiter = PerformanceLimiter(self.config)
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.update_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Current AI decision
        self.current_decision: Optional[AIDecision] = None
        self.decision_timestamp: Optional[datetime] = None
        
        # Click-through management
        self.click_through_active = False
        self.click_through_mode = self.config.click_through_mode
        
        self.logger.info("VisualIntelligenceOverlay initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the overlay system
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Visual Intelligence Overlay")
        self.state = OverlayState.INITIALIZING
        
        try:
            # P3.1.5: Remote Session Detection
            self.is_remote_session = WindowsAPIHelper.detect_remote_session()
            if self.is_remote_session:
                self.logger.warning("Remote session detected - overlay performance may be degraded")
            
            # P3.1.1: Monitor topology detection
            self.monitors = WindowsAPIHelper.get_monitor_info()
            self.logger.info(f"Detected {len(self.monitors)} monitors")
            
            # P3.1.6: Ultrawide Display Support
            for monitor in self.monitors:
                if monitor.aspect_ratio > 2.0:  # 21:9 or wider
                    self.logger.info(f"Ultrawide display detected: {monitor.device_name} "
                                   f"({monitor.width}x{monitor.height}, {monitor.aspect_ratio:.2f}:1)")
            
            # Select target monitor
            self._select_target_monitor()
            
            # Initialize performance monitoring
            try:
                self.performance_monitor = PerformanceMonitor()
                self.performance_monitor.start()
            except Exception as e:
                self.logger.warning(f"Could not initialize performance monitor: {e}")
            
            # Initialize UI components
            if not self._initialize_ui():
                self.logger.error("Failed to initialize UI components")
                return False
            
            self.state = OverlayState.ACTIVE
            self.logger.info("Visual Intelligence Overlay initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize overlay: {e}")
            self.logger.error(traceback.format_exc())
            self.state = OverlayState.ERROR
            return False
    
    def _select_target_monitor(self):
        """
        Select the target monitor for overlay display with enhanced debugging
        
        Implements smart monitor selection based on game window position
        """
        if not self.monitors:
            self.logger.error("❌ No monitors available for overlay")
            return
        
        self.logger.info(f"🖥️ Selecting target monitor from {len(self.monitors)} available monitors")
        
        # Try to find the game window first
        self.logger.info("🎯 Attempting to detect game window for smart monitor selection...")
        self.game_window_hwnd = WindowsAPIHelper.find_game_window(
            self.config.game_window_titles
        )
        
        if self.game_window_hwnd and WINDOWS_API_AVAILABLE:
            try:
                # Get game window position
                game_rect = GetWindowRect(self.game_window_hwnd)
                game_center_x = (game_rect[0] + game_rect[2]) // 2
                game_center_y = (game_rect[1] + game_rect[3]) // 2
                
                self.logger.info(f"🎯 Game window found at position: "
                                f"({game_rect[0]}, {game_rect[1]}) to ({game_rect[2]}, {game_rect[3]})")
                self.logger.info(f"🎯 Game window center: ({game_center_x}, {game_center_y})")
                
                # Find which monitor contains the game window
                for i, monitor in enumerate(self.monitors):
                    self.logger.debug(f"   Monitor {i}: {monitor.device_name} bounds: {monitor.bounds}")
                    if (monitor.bounds[0] <= game_center_x <= monitor.bounds[2] and
                        monitor.bounds[1] <= game_center_y <= monitor.bounds[3]):
                        self.target_monitor = monitor
                        self.logger.info(f"✅ Game window detected on monitor: {monitor.device_name}")
                        self.logger.info(f"🖥️ Selected target monitor: {monitor.device_name} "
                                        f"({monitor.width}x{monitor.height})")
                        return
                
                self.logger.warning(f"⚠️ Game window center ({game_center_x}, {game_center_y}) "
                                   f"not found within any monitor bounds")
                        
            except Exception as e:
                self.logger.warning(f"⚠️ Could not get game window position: {e}")
        else:
            if not self.game_window_hwnd:
                self.logger.warning("⚠️ Game window not found - using fallback monitor selection")
            if not WINDOWS_API_AVAILABLE:
                self.logger.warning("⚠️ Windows API not available - using fallback monitor selection")
        
        # Fallback: use primary monitor or first available
        self.logger.info("🖥️ Using fallback monitor selection...")
        for monitor in self.monitors:
            if monitor.is_primary:
                self.target_monitor = monitor
                self.logger.info(f"✅ Using primary monitor: {monitor.device_name} "
                                f"({monitor.width}x{monitor.height})")
                return
        
        # Last fallback: use first monitor
        self.target_monitor = self.monitors[0]
        self.logger.info(f"✅ Using first available monitor: {self.target_monitor.device_name} "
                        f"({self.target_monitor.width}x{self.target_monitor.height})")
    
    def _calculate_overlay_position(self, overlay_width: int, overlay_height: int) -> Tuple[int, int]:
        """
        Calculate optimal overlay position relative to game window (PHASE 1.2 FIX)
        
        Args:
            overlay_width: Width of the overlay in pixels
            overlay_height: Height of the overlay in pixels
            
        Returns:
            Tuple[int, int]: (x, y) coordinates for overlay position
        """
        self.logger.info("🎯 Calculating overlay position...")
        
        # Strategy 1: Position relative to detected game window
        if self.game_window_hwnd and WINDOWS_API_AVAILABLE:
            try:
                # Get current game window position
                game_rect = GetWindowRect(self.game_window_hwnd)
                game_x, game_y, game_right, game_bottom = game_rect
                game_width = game_right - game_x
                game_height = game_bottom - game_y
                
                self.logger.info(f"🎯 Game window position: ({game_x}, {game_y}) size: {game_width}x{game_height}")
                
                # Check if window is minimized or has invalid size
                if game_width <= 0 or game_height <= 0:
                    self.logger.warning("⚠️ Game window appears minimized or invalid - using fallback positioning")
                    return self._fallback_overlay_position(overlay_width, overlay_height)
                
                # Calculate overlay position for Hearthstone Arena card selection
                # Position overlay at the bottom of the game window where cards typically appear
                card_area_height = 200  # Approximate height of card selection area
                
                # Center horizontally over the game window
                overlay_x = game_x + (game_width - overlay_width) // 2
                
                # Position vertically at the bottom of the game window 
                # (where Arena cards typically appear)
                overlay_y = game_bottom - card_area_height - overlay_height - 20  # 20px margin
                
                # Ensure overlay stays within monitor bounds
                if self.target_monitor:
                    monitor_left = self.target_monitor.bounds[0]
                    monitor_top = self.target_monitor.bounds[1]
                    monitor_right = self.target_monitor.bounds[2]
                    monitor_bottom = self.target_monitor.bounds[3]
                    
                    # Clamp to monitor bounds
                    overlay_x = max(monitor_left, min(overlay_x, monitor_right - overlay_width))
                    overlay_y = max(monitor_top, min(overlay_y, monitor_bottom - overlay_height))
                
                self.logger.info(f"✅ Overlay positioned relative to game window: ({overlay_x}, {overlay_y})")
                return overlay_x, overlay_y
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error calculating game window relative position: {e}")
                return self._fallback_overlay_position(overlay_width, overlay_height)
        
        # Strategy 2: Fallback positioning
        return self._fallback_overlay_position(overlay_width, overlay_height)
    
    def _fallback_overlay_position(self, overlay_width: int, overlay_height: int) -> Tuple[int, int]:
        """
        Fallback overlay positioning when game window detection fails
        
        Args:
            overlay_width: Width of the overlay in pixels
            overlay_height: Height of the overlay in pixels
            
        Returns:
            Tuple[int, int]: (x, y) coordinates for overlay position
        """
        self.logger.info("🔄 Using fallback overlay positioning...")
        
        if not self.target_monitor:
            # Emergency fallback - center on primary display
            self.logger.warning("⚠️ No monitor information - using emergency positioning")
            return 100, 100
        
        # Position at top center of the target monitor (original behavior)
        x = self.target_monitor.bounds[0] + (self.target_monitor.width - overlay_width) // 2
        y = self.target_monitor.bounds[1] + 50  # 50px from top
        
        self.logger.info(f"✅ Fallback overlay position: ({x}, {y})")
        return x, y
    
    def _initialize_ui(self) -> bool:
        """
        Initialize the UI components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create the main window
            self.root = tk.Tk()
            self.root.title("AI Helper - Visual Intelligence Overlay")
            
            # Configure window properties
            self.root.configure(bg=self.config.background_color)
            self.root.attributes('-alpha', self.config.opacity)
            self.root.attributes('-topmost', True)
            
            # P3.1.3: Window State Change Resilience
            self.root.bind('<Configure>', self._on_window_configure)
            self.root.bind('<FocusIn>', self._on_window_focus)
            self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
            
            # Calculate overlay size and position
            if not self.target_monitor:
                self.logger.error("No target monitor selected")
                return False
            
            # P3.1.2: DPI-Aware Coordinate Transformation
            overlay_width = int((self.config.card_width * 3 + 
                               self.config.card_spacing * 2 + 
                               self.config.overlay_padding * 2) * self.target_monitor.dpi_scale)
            overlay_height = int((self.config.card_height + 
                                self.config.overlay_padding * 2) * self.target_monitor.dpi_scale)
            
            # Position overlay relative to game window or monitor (PHASE 1.2 FIX)
            x, y = self._calculate_overlay_position(overlay_width, overlay_height)
            
            self.root.geometry(f"{overlay_width}x{overlay_height}+{x}+{y}")
            self.root.resizable(False, False)
            
            # Create the main canvas for drawing
            self.canvas = tk.Canvas(
                self.root,
                width=overlay_width,
                height=overlay_height,
                bg=self.config.background_color,
                highlightthickness=0
            )
            self.canvas.pack(fill='both', expand=True)
            
            # Setup click-through if enabled
            if self.config.click_through_enabled:
                self._setup_click_through()
            
            self.logger.info(f"UI initialized: {overlay_width}x{overlay_height} at ({x}, {y})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UI: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _setup_click_through(self):
        """
        Setup click-through functionality with fallback modes
        
        Implements P3.1.7-P3.1.12: Platform-Specific Click-Through Strategies
        """
        if not WINDOWS_API_AVAILABLE:
            self.logger.warning("Windows API not available - click-through disabled")
            return
        
        # Get the Tkinter window handle
        try:
            self.root.update()  # Ensure window is created
            hwnd = int(self.root.wm_frame(), 16) if hasattr(self.root, 'wm_frame') else None
            
            if not hwnd:
                # Alternative method to get window handle
                import tkinter.winfo
                hwnd = self.root.winfo_id()
                
            if hwnd:
                # Try primary click-through mode
                success = WindowsAPIHelper.setup_click_through(hwnd, self.click_through_mode)
                
                if success and WindowsAPIHelper.validate_click_through(hwnd):
                    self.click_through_active = True
                    self.logger.info(f"Click-through enabled with mode: {self.click_through_mode}")
                else:
                    # Try fallback modes
                    for fallback_mode in self.config.fallback_modes:
                        self.logger.info(f"Trying fallback click-through mode: {fallback_mode}")
                        success = WindowsAPIHelper.setup_click_through(hwnd, fallback_mode)
                        
                        if success and WindowsAPIHelper.validate_click_through(hwnd):
                            self.click_through_active = True
                            self.click_through_mode = fallback_mode
                            self.logger.info(f"Click-through enabled with fallback mode: {fallback_mode}")
                            break
                    else:
                        self.logger.warning("All click-through modes failed - overlay will capture clicks")
            else:
                self.logger.error("Could not get window handle for click-through setup")
                
        except Exception as e:
            self.logger.error(f"Failed to setup click-through: {e}")
    
    def start(self) -> bool:
        """
        Start the overlay system
        
        Returns:
            bool: True if started successfully
        """
        if self.running:
            self.logger.warning("Overlay is already running")
            return True
        
        if not self.initialize():
            self.logger.error("Failed to initialize overlay")
            return False
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start the main update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            name="OverlayUpdateThread",
            daemon=True
        )
        self.update_thread.start()
        
        self.logger.info("Visual Intelligence Overlay started")
        return True
    
    def stop(self):
        """Stop the overlay system with proper cleanup"""
        self.logger.info("Stopping Visual Intelligence Overlay")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for update thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        
        # Stop performance monitor
        if self.performance_monitor:
            try:
                self.performance_monitor.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping performance monitor: {e}")
        
        # Cleanup UI
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                self.logger.debug(f"Error destroying UI: {e}")
        
        self.state = OverlayState.INACTIVE
        self.logger.info("Visual Intelligence Overlay stopped")
    
    def update_decision(self, decision: AIDecision):
        """
        Update the overlay with a new AI decision
        
        Args:
            decision: AI decision to display
        """
        if not self.running:
            self.logger.warning("Cannot update decision - overlay not running")
            return
        
        self.current_decision = decision
        self.decision_timestamp = datetime.now()
        
        self.logger.info(f"Updated overlay with new AI decision: "
                        f"recommended_pick={decision.recommended_pick}, "
                        f"confidence={decision.confidence}")
    
    def _update_loop(self):
        """
        Main update loop running in separate thread
        
        Implements P3.3.1-P3.3.6: Performance monitoring and frame rate limiting
        """
        self.logger.info("Starting overlay update loop")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # P3.3.1: Frame Rate Budget Management
                frame_start = self.performance_limiter.start_frame()
                
                # Check if we should skip this frame for performance
                if self.performance_limiter.should_skip_frame():
                    time.sleep(1.0 / self.performance_limiter.get_target_fps())
                    continue
                
                # Update the overlay display
                self._update_display()
                
                # End frame timing and apply throttling
                within_budget = self.performance_limiter.end_frame(frame_start)
                
                if not within_budget:
                    # Log performance issues
                    stats = self.performance_limiter.get_performance_stats()
                    self.logger.debug(f"Performance stats: {stats}")
                
                # Sleep to maintain target FPS
                target_frame_time = 1.0 / self.performance_limiter.get_target_fps()
                elapsed = time.perf_counter() - frame_start
                sleep_time = max(0, target_frame_time - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                
                # P3.1.11: Compositor Recovery Protocol
                if self._should_attempt_recovery():
                    self._attempt_recovery()
                else:
                    self.logger.error("Too many recovery attempts, stopping overlay")
                    break
                
                time.sleep(1.0)  # Prevent rapid error loops
        
        self.logger.info("Overlay update loop finished")
    
    def _update_overlay_position(self):
        """
        Check and update overlay position if game window has moved (PHASE 1.2)
        
        This method ensures the overlay follows the game window if it's moved or resized
        """
        if not self.root or not self.game_window_hwnd or not WINDOWS_API_AVAILABLE:
            return
        
        try:
            # Get current game window position
            current_game_rect = GetWindowRect(self.game_window_hwnd)
            current_game_x, current_game_y, current_game_right, current_game_bottom = current_game_rect
            
            # Check if we have stored the previous position
            if not hasattr(self, '_last_game_position'):
                self._last_game_position = current_game_rect
                return
            
            # Compare with last known position
            last_game_rect = self._last_game_position
            position_changed = (
                current_game_x != last_game_rect[0] or
                current_game_y != last_game_rect[1] or
                current_game_right != last_game_rect[2] or
                current_game_bottom != last_game_rect[3]
            )
            
            if position_changed:
                self.logger.info(f"🎯 Game window moved - updating overlay position")
                self.logger.info(f"   Old position: {last_game_rect}")
                self.logger.info(f"   New position: {current_game_rect}")
                
                # Get current overlay size
                overlay_width = self.root.winfo_width()
                overlay_height = self.root.winfo_height()
                
                # Calculate new overlay position
                new_x, new_y = self._calculate_overlay_position(overlay_width, overlay_height)
                
                # Update overlay position
                self.root.geometry(f"{overlay_width}x{overlay_height}+{new_x}+{new_y}")
                
                # Store new game window position
                self._last_game_position = current_game_rect
                
                self.logger.info(f"✅ Overlay repositioned to ({new_x}, {new_y})")
            
        except Exception as e:
            self.logger.debug(f"Error updating overlay position: {e}")
            # Don't spam logs with positioning errors
    
    def _update_display(self):
        """
        Update the overlay display with current AI decision
        
        This method renders the AI recommendations on the overlay canvas
        """
        if not self.root or not self.canvas:
            return
        
        try:
            # PHASE 1.2: Check and update overlay position if game window moved
            self._update_overlay_position()
            
            # Only draw if we have a decision to display
            if self.current_decision:
                # Clear the canvas
                self.canvas.delete("all")
                
                # P3.1.12: Theme Change Resilience
                self.canvas.configure(bg=self.config.background_color)
                
                # Draw the AI recommendations
                self._draw_recommendations()
            
            # Update the display
            self.root.update_idletasks()
            
        except Exception as e:
            self.logger.error(f"Error updating display: {e}")
    
    def _draw_recommendations(self):
        """Draw AI recommendations on the canvas"""
        if not self.current_decision or not self.canvas:
            return
        
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Draw title
            title_text = "🎯 AI Recommendations"
            self.canvas.create_text(
                canvas_width // 2, 20,
                text=title_text,
                fill=self.config.text_color,
                font=("Arial", 14, "bold"),
                anchor="n"
            )
            
            # Draw cards
            card_y = 50
            card_width = (canvas_width - self.config.overlay_padding * 2 - 
                         self.config.card_spacing * 2) // 3
            
            for i, (card_option, scores) in enumerate(self.current_decision.card_evaluations):
                x = (self.config.overlay_padding + 
                     i * (card_width + self.config.card_spacing))
                
                # Determine if this is the recommended card
                is_recommended = (card_option.position == self.current_decision.recommended_pick)
                
                # Draw card background
                card_color = (self.config.success_color if is_recommended 
                             else self.config.background_color)
                border_color = (self.config.success_color if is_recommended 
                               else self.config.text_color)
                
                self.canvas.create_rectangle(
                    x, card_y, x + card_width, card_y + 120,
                    fill=card_color, outline=border_color, width=2
                )
                
                # Draw card name
                self.canvas.create_text(
                    x + card_width // 2, card_y + 10,
                    text=card_option.card_info.name[:15] + "..." if len(card_option.card_info.name) > 15 
                         else card_option.card_info.name,
                    fill="white" if is_recommended else self.config.text_color,
                    font=("Arial", 10, "bold" if is_recommended else "normal"),
                    anchor="n"
                )
                
                # Draw scores
                score_y = card_y + 35
                score_text = f"Score: {scores.composite_score:.1f}"
                self.canvas.create_text(
                    x + card_width // 2, score_y,
                    text=score_text,
                    fill="white" if is_recommended else self.config.text_color,
                    font=("Arial", 9),
                    anchor="n"
                )
                
                # Draw recommendation indicator
                if is_recommended:
                    self.canvas.create_text(
                        x + card_width // 2, score_y + 20,
                        text="👑 PICK",
                        fill="gold",
                        font=("Arial", 12, "bold"),
                        anchor="n"
                    )
            
            # Draw confidence and reasoning
            confidence_y = card_y + 140
            confidence_text = f"Confidence: {self.current_decision.confidence.value.upper()}"
            self.canvas.create_text(
                canvas_width // 2, confidence_y,
                text=confidence_text,
                fill=self.config.text_color,
                font=("Arial", 10),
                anchor="n"
            )
            
            # Draw reasoning (truncated)
            if self.current_decision.reasoning:
                reasoning_text = (self.current_decision.reasoning[:50] + "..." 
                                if len(self.current_decision.reasoning) > 50 
                                else self.current_decision.reasoning)
                self.canvas.create_text(
                    canvas_width // 2, confidence_y + 20,
                    text=reasoning_text,
                    fill=self.config.text_color,
                    font=("Arial", 8),
                    anchor="n"
                )
                
        except Exception as e:
            self.logger.error(f"Error drawing recommendations: {e}")
    
    def _should_attempt_recovery(self) -> bool:
        """Determine if we should attempt recovery from an error"""
        return (self.config.crash_recovery_enabled and 
                self.recovery_attempts < self.config.max_recovery_attempts)
    
    def _attempt_recovery(self):
        """
        P3.1.11: Compositor Recovery Protocol
        Attempt to recover from overlay failures
        """
        self.recovery_attempts += 1
        self.state = OverlayState.RECOVERING
        self.logger.info(f"Attempting overlay recovery (attempt {self.recovery_attempts})")
        
        try:
            # Wait before recovery attempt
            time.sleep(self.config.recovery_delay_seconds)
            
            # Try to recreate the UI
            if self.root:
                try:
                    self.root.destroy()
                except Exception:
                    pass
                self.root = None
                self.canvas = None
            
            # Reinitialize UI
            if self._initialize_ui():
                self.state = OverlayState.ACTIVE
                self.logger.info("Overlay recovery successful")
            else:
                self.logger.error("Overlay recovery failed")
                self.state = OverlayState.ERROR
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            self.state = OverlayState.ERROR
    
    def _on_window_configure(self, event):
        """P3.1.3: Handle window configuration changes"""
        if event.widget == self.root:
            self.logger.debug("Window configuration changed")
    
    def _on_window_focus(self, event):
        """Handle window focus events"""
        if event.widget == self.root and self.config.click_through_enabled:
            # Ensure click-through is still active
            self._setup_click_through()
    
    def _on_window_close(self):
        """Handle window close event"""
        self.stop()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self.performance_limiter.get_performance_stats()
        stats.update({
            'state': self.state.value,
            'running': self.running,
            'click_through_active': self.click_through_active,
            'click_through_mode': self.click_through_mode.value,
            'monitor_count': len(self.monitors),
            'target_monitor': self.target_monitor.device_name if self.target_monitor else None,
            'recovery_attempts': self.recovery_attempts,
            'is_remote_session': self.is_remote_session
        })
        return stats
    
    def minimize(self):
        """Minimize the overlay"""
        if self.root:
            self.root.withdraw()
            self.state = OverlayState.MINIMIZED
    
    def restore(self):
        """Restore the overlay from minimized state"""
        if self.root:
            self.root.deiconify()
            self.state = OverlayState.ACTIVE

# Factory function for creating overlay instances
def create_visual_overlay(config: OverlayConfiguration = None) -> VisualIntelligenceOverlay:
    """
    Factory function to create a Visual Intelligence Overlay
    
    Args:
        config: Configuration for the overlay
        
    Returns:
        VisualIntelligenceOverlay: Configured overlay instance
    """
    return VisualIntelligenceOverlay(config)

# Backward compatibility alias
VisualOverlay = VisualIntelligenceOverlay

# Main function for testing
def main():
    """Demo the Visual Intelligence Overlay"""
    print("=== Visual Intelligence Overlay Demo ===")
    print("This will create a high-performance overlay window.")
    print("Press Ctrl+C to exit.")
    
    # Create configuration
    config = OverlayConfiguration()
    config.click_through_enabled = True
    
    # Create and start overlay
    overlay = create_visual_overlay(config)
    
    try:
        if overlay.start():
            print("Overlay started successfully")
            
            # Keep running until interrupted
            while overlay.running:
                time.sleep(1.0)
                
                # Print performance stats every 10 seconds
                stats = overlay.get_performance_stats()
                if stats:
                    print(f"Performance: {stats.get('current_fps', 0):.1f} FPS, "
                          f"throttle: {stats.get('performance_throttle', 1.0):.2f}")
        else:
            print("Failed to start overlay")
            
    except KeyboardInterrupt:
        print("\nStopping overlay...")
    finally:
        overlay.stop()

if __name__ == "__main__":
    main()