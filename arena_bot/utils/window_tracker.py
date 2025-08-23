"""
Hearthstone window locator and bounds tracking for Phase 4.

Provides robust window finding, bounds tracking, and debounced updates
for overlay positioning on Windows desktop.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Dict, Any
from collections import deque

from arena_bot.capture.capture_backend import AdaptiveCaptureManager, WindowInfo


logger = logging.getLogger(__name__)


@dataclass
class NormalizedBounds:
    """Normalized window bounds with DPI awareness."""
    x: int
    y: int
    width: int
    height: int
    dpi_scale: float
    timestamp: float
    window_handle: int
    
    def __post_init__(self):
        # Ensure bounds are reasonable
        self.x = max(0, self.x)
        self.y = max(0, self.y)
        self.width = max(100, self.width)  # Minimum reasonable width
        self.height = max(100, self.height)  # Minimum reasonable height
    
    def get_rect(self) -> Tuple[int, int, int, int]:
        """Get bounds as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def get_inset_rect(self, inset_px: int = 5) -> Tuple[int, int, int, int]:
        """Get bounds with inset applied."""
        return (
            self.x + inset_px,
            self.y + inset_px,
            max(100, self.width - 2 * inset_px),
            max(100, self.height - 2 * inset_px)
        )


@dataclass
class WindowTracker:
    """Tracks Hearthstone window bounds with debouncing and change detection."""
    
    # Configuration
    update_interval_ms: int = 500  # Check window bounds every 500ms
    debounce_threshold_px: int = 10  # Ignore changes smaller than 10px
    debounce_time_ms: int = 200  # Wait 200ms before confirming change
    min_window_size: Tuple[int, int] = (400, 300)  # Minimum reasonable window size
    
    # State
    _capture_manager: AdaptiveCaptureManager = field(default_factory=AdaptiveCaptureManager)
    _current_bounds: Optional[NormalizedBounds] = None
    _last_raw_bounds: Optional[Tuple[int, int, int, int]] = None
    _pending_bounds: Optional[NormalizedBounds] = None
    _pending_since: Optional[float] = None
    _window_handle: Optional[int] = None
    _tracking_thread: Optional[threading.Thread] = None
    _stop_tracking: threading.Event = field(default_factory=threading.Event)
    _bounds_callbacks: list = field(default_factory=list)
    _bounds_history: deque = field(default_factory=lambda: deque(maxlen=10))
    _last_error: Optional[str] = None
    
    def __post_init__(self):
        self._lock = threading.RLock()
    
    def add_bounds_callback(self, callback: Callable[[NormalizedBounds], None]):
        """Add a callback to be called when bounds change."""
        with self._lock:
            self._bounds_callbacks.append(callback)
    
    def remove_bounds_callback(self, callback: Callable[[NormalizedBounds], None]):
        """Remove a bounds callback."""
        with self._lock:
            if callback in self._bounds_callbacks:
                self._bounds_callbacks.remove(callback)
    
    def find_hearthstone_window(self) -> Optional[WindowInfo]:
        """Find the Hearthstone window using robust heuristics."""
        try:
            window = self._capture_manager.find_hearthstone_window()
            if window is None:
                self._last_error = "No Hearthstone window found - please launch Hearthstone in windowed/borderless mode"
                logger.warning(self._last_error)
                return None
            
            # Validate window size
            if window.width < self.min_window_size[0] or window.height < self.min_window_size[1]:
                self._last_error = f"Hearthstone window too small: {window.width}x{window.height} (min: {self.min_window_size[0]}x{self.min_window_size[1]})"
                logger.warning(self._last_error)
                return None
            
            self._last_error = None
            logger.info(f"Found Hearthstone window: '{window.title}' ({window.width}x{window.height})")
            return window
            
        except Exception as e:
            self._last_error = f"Error finding Hearthstone window: {e}"
            logger.error(self._last_error)
            return None
    
    def _get_dpi_scale(self, window_handle: int) -> float:
        """Get DPI scale for the window (placeholder implementation)."""
        # TODO(claude): Implement actual DPI detection for multi-monitor setups
        # For now, return 1.0 as default scale
        return 1.0
    
    def _normalize_bounds(self, window: WindowInfo) -> NormalizedBounds:
        """Convert window info to normalized bounds."""
        dpi_scale = self._get_dpi_scale(window.handle)
        
        return NormalizedBounds(
            x=window.x,
            y=window.y,
            width=window.width,
            height=window.height,
            dpi_scale=dpi_scale,
            timestamp=time.time(),
            window_handle=window.handle
        )
    
    def _bounds_changed_significantly(self, old_bounds: Optional[NormalizedBounds], 
                                    new_bounds: NormalizedBounds) -> bool:
        """Check if bounds changed beyond debounce threshold."""
        if old_bounds is None:
            return True
        
        # Check for significant position/size changes
        dx = abs(new_bounds.x - old_bounds.x)
        dy = abs(new_bounds.y - old_bounds.y)
        dw = abs(new_bounds.width - old_bounds.width)
        dh = abs(new_bounds.height - old_bounds.height)
        
        return (dx > self.debounce_threshold_px or 
                dy > self.debounce_threshold_px or
                dw > self.debounce_threshold_px or 
                dh > self.debounce_threshold_px or
                new_bounds.window_handle != old_bounds.window_handle)
    
    def _notify_bounds_changed(self, bounds: NormalizedBounds):
        """Notify all callbacks about bounds change."""
        with self._lock:
            for callback in self._bounds_callbacks:
                try:
                    callback(bounds)
                except Exception as e:
                    logger.error(f"Error in bounds callback: {e}")
    
    def _tracking_loop(self):
        """Main tracking loop that runs in background thread."""
        logger.info("Window tracking started")
        
        while not self._stop_tracking.is_set():
            try:
                # Find current window
                window = self.find_hearthstone_window()
                current_time = time.time()
                
                if window is None:
                    # Clear current bounds if window lost
                    with self._lock:
                        if self._current_bounds is not None:
                            logger.warning("Lost Hearthstone window")
                            self._current_bounds = None
                            self._window_handle = None
                            self._pending_bounds = None
                            self._pending_since = None
                    
                    # Wait and continue
                    self._stop_tracking.wait(self.update_interval_ms / 1000.0)
                    continue
                
                # Normalize bounds
                new_bounds = self._normalize_bounds(window)
                
                with self._lock:
                    # Check if bounds changed significantly
                    if self._bounds_changed_significantly(self._current_bounds, new_bounds):
                        
                        # If this is a new pending change, start debounce timer
                        if (self._pending_bounds is None or 
                            self._bounds_changed_significantly(self._pending_bounds, new_bounds)):
                            self._pending_bounds = new_bounds
                            self._pending_since = current_time
                            logger.debug(f"Pending bounds change: {new_bounds.get_rect()}")
                        
                        # Check if debounce time has elapsed
                        elif (self._pending_since is not None and 
                              current_time - self._pending_since >= self.debounce_time_ms / 1000.0):
                            
                            # Confirm the change
                            old_bounds = self._current_bounds
                            self._current_bounds = self._pending_bounds
                            self._window_handle = self._pending_bounds.window_handle
                            self._bounds_history.append(self._current_bounds)
                            
                            # Clear pending state
                            self._pending_bounds = None
                            self._pending_since = None
                            
                            logger.info(f"Hearthstone window bounds updated: {self._current_bounds.get_rect()}")
                            
                            # Notify callbacks
                            self._notify_bounds_changed(self._current_bounds)
                    
                    else:
                        # Bounds haven't changed significantly, clear pending if any
                        if self._pending_bounds is not None:
                            self._pending_bounds = None
                            self._pending_since = None
                
            except Exception as e:
                logger.error(f"Error in window tracking loop: {e}")
                self._last_error = f"Tracking loop error: {e}"
            
            # Wait for next update
            self._stop_tracking.wait(self.update_interval_ms / 1000.0)
        
        logger.info("Window tracking stopped")
    
    def start_tracking(self) -> bool:
        """Start background window tracking."""
        with self._lock:
            if self._tracking_thread is not None and self._tracking_thread.is_alive():
                logger.warning("Window tracking already running")
                return True
            
            # Find initial window
            window = self.find_hearthstone_window()
            if window is None:
                logger.error("Cannot start tracking - Hearthstone window not found")
                return False
            
            # Set initial bounds
            self._current_bounds = self._normalize_bounds(window)
            self._window_handle = window.handle
            self._bounds_history.append(self._current_bounds)
            
            # Start tracking thread
            self._stop_tracking.clear()
            self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._tracking_thread.start()
            
            logger.info(f"Window tracking started for: {window.title}")
            return True
    
    def stop_tracking(self):
        """Stop background window tracking."""
        with self._lock:
            if self._tracking_thread is None:
                return
            
            self._stop_tracking.set()
            
            # Wait for thread to finish
            if self._tracking_thread.is_alive():
                self._tracking_thread.join(timeout=2.0)
                if self._tracking_thread.is_alive():
                    logger.warning("Window tracking thread did not stop gracefully")
            
            self._tracking_thread = None
    
    def get_current_bounds(self) -> Optional[NormalizedBounds]:
        """Get current window bounds."""
        with self._lock:
            return self._current_bounds
    
    def get_window_handle(self) -> Optional[int]:
        """Get current window handle."""
        with self._lock:
            return self._window_handle
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get tracking status information."""
        with self._lock:
            return {
                'is_tracking': self._tracking_thread is not None and self._tracking_thread.is_alive(),
                'current_bounds': self._current_bounds.get_rect() if self._current_bounds else None,
                'window_handle': self._window_handle,
                'has_pending_change': self._pending_bounds is not None,
                'last_error': self._last_error,
                'bounds_history_count': len(self._bounds_history)
            }