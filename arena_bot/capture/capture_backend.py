"""
Capture backend interface and implementations for Windows.

Provides abstraction for screen capture with multiple backend implementations:
- DXGI Desktop Duplication (preferred, faster)
- BitBlt/MSS fallback (always available, slower but reliable)
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Information about a display monitor."""
    monitor_id: int
    width: int
    height: int
    x: int
    y: int
    dpi_scale: float
    is_primary: bool
    device_name: str


@dataclass
class WindowInfo:
    """Information about a window."""
    handle: int
    title: str
    x: int
    y: int
    width: int
    height: int
    pid: int
    is_visible: bool
    is_minimized: bool


@dataclass
class CaptureFrame:
    """Captured frame with metadata."""
    image: np.ndarray  # BGR format for OpenCV compatibility
    timestamp: float
    capture_duration_ms: float
    source_rect: Tuple[int, int, int, int]  # x, y, width, height
    backend_name: str
    dpi_scale: float
    metadata: Dict[str, Any]


class CaptureBackend(ABC):
    """Abstract base class for capture backends."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this capture backend."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def enumerate_monitors(self) -> List[MonitorInfo]:
        """Get information about all available monitors."""
        pass
    
    @abstractmethod
    def find_windows(self, title_substring: str = None, pid: int = None) -> List[WindowInfo]:
        """Find windows matching criteria."""
        pass
    
    @abstractmethod
    def get_window_rect(self, window_handle: int) -> Tuple[int, int, int, int]:
        """Get the bounding rectangle of a window."""
        pass
    
    @abstractmethod
    def capture_rect(self, x: int, y: int, width: int, height: int) -> CaptureFrame:
        """Capture a rectangular region of the screen."""
        pass
    
    @abstractmethod
    def capture_window(self, window_handle: int) -> CaptureFrame:
        """Capture a specific window."""
        pass


class DXGICaptureBackend(CaptureBackend):
    """DXGI Desktop Duplication capture backend (Windows 8+)."""
    
    def __init__(self):
        self._initialized = False
        self._available = None
        
    def get_name(self) -> str:
        return "DXGI Desktop Duplication"
    
    def is_available(self) -> bool:
        """Check if DXGI is available."""
        if self._available is not None:
            return self._available
            
        try:
            # Try to import required Windows modules
            import d3dshot
            import win32gui
            import win32con
            
            # Test if we can create a D3DShot instance
            d3d = d3dshot.create(capture_output="numpy")
            if d3d is None:
                self._available = False
                logger.info("D3DShot creation failed - DXGI not available")
                return False
                
            # Clean up test instance
            d3d = None
            self._available = True
            logger.info("DXGI Desktop Duplication backend available")
            return True
            
        except ImportError as e:
            logger.info(f"DXGI backend not available - missing dependencies: {e}")
            self._available = False
            return False
        except Exception as e:
            logger.info(f"DXGI backend not available - initialization failed: {e}")
            self._available = False
            return False
    
    def enumerate_monitors(self) -> List[MonitorInfo]:
        """Get monitor information using Win32 API."""
        monitors = []
        try:
            import win32api
            import win32con
            
            def monitor_enum_proc(hmonitor, hdc, rect, data):
                monitor_info = win32api.GetMonitorInfo(hmonitor)
                device = monitor_info['Device']
                monitor_rect = monitor_info['Monitor']
                is_primary = monitor_info['Flags'] & win32con.MONITORINFOF_PRIMARY != 0
                
                # TODO(claude): Get actual DPI scale - using 1.0 for now
                dpi_scale = 1.0
                
                monitors.append(MonitorInfo(
                    monitor_id=len(monitors),
                    width=monitor_rect[2] - monitor_rect[0],
                    height=monitor_rect[3] - monitor_rect[1],
                    x=monitor_rect[0],
                    y=monitor_rect[1],
                    dpi_scale=dpi_scale,
                    is_primary=is_primary,
                    device_name=device
                ))
                return True
            
            win32api.EnumDisplayMonitors(None, None, monitor_enum_proc, None)
            
        except ImportError:
            logger.warning("Win32 API not available for monitor enumeration")
        except Exception as e:
            logger.error(f"Failed to enumerate monitors: {e}")
            
        return monitors
    
    def find_windows(self, title_substring: str = None, pid: int = None) -> List[WindowInfo]:
        """Find windows using Win32 API."""
        windows = []
        try:
            import win32gui
            import win32process
            
            def enum_windows_callback(hwnd, param):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    
                    # Check title filter
                    if title_substring and title_substring.lower() not in title.lower():
                        return True
                    
                    # Get window rect and process info
                    rect = win32gui.GetWindowRect(hwnd)
                    _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
                    
                    # Check PID filter
                    if pid and window_pid != pid:
                        return True
                    
                    # Check if minimized
                    is_minimized = win32gui.IsIconic(hwnd)
                    
                    windows.append(WindowInfo(
                        handle=hwnd,
                        title=title,
                        x=rect[0],
                        y=rect[1],
                        width=rect[2] - rect[0],
                        height=rect[3] - rect[1],
                        pid=window_pid,
                        is_visible=True,
                        is_minimized=is_minimized
                    ))
                return True
            
            win32gui.EnumWindows(enum_windows_callback, None)
            
        except ImportError:
            logger.warning("Win32 API not available for window enumeration")
        except Exception as e:
            logger.error(f"Failed to enumerate windows: {e}")
            
        return windows
    
    def get_window_rect(self, window_handle: int) -> Tuple[int, int, int, int]:
        """Get window rectangle using Win32 API."""
        try:
            import win32gui
            rect = win32gui.GetWindowRect(window_handle)
            return rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        except ImportError:
            raise RuntimeError("Win32 API not available")
        except Exception as e:
            raise RuntimeError(f"Failed to get window rect: {e}")
    
    def capture_rect(self, x: int, y: int, width: int, height: int) -> CaptureFrame:
        """Capture screen region using DXGI."""
        start_time = time.time()
        
        try:
            import d3dshot
            
            # Create D3DShot instance
            d3d = d3dshot.create(capture_output="numpy")
            if d3d is None:
                raise RuntimeError("Failed to create D3DShot instance")
            
            # Capture the region
            region = (x, y, x + width, y + height)
            screenshot = d3d.screenshot(region=region)
            
            if screenshot is None:
                raise RuntimeError("DXGI capture returned None")
            
            # Convert RGBA to BGR for OpenCV
            if screenshot.shape[2] == 4:  # RGBA
                screenshot = screenshot[:, :, [2, 1, 0]]  # Convert to BGR
            elif screenshot.shape[2] == 3:  # RGB
                screenshot = screenshot[:, :, [2, 1, 0]]  # Convert to BGR
            
            capture_duration = (time.time() - start_time) * 1000
            
            return CaptureFrame(
                image=screenshot,
                timestamp=time.time(),
                capture_duration_ms=capture_duration,
                source_rect=(x, y, width, height),
                backend_name=self.get_name(),
                dpi_scale=1.0,  # TODO(claude): Get actual DPI scale
                metadata={'method': 'dxgi_desktop_duplication'}
            )
            
        except Exception as e:
            logger.error(f"DXGI capture failed: {e}")
            raise RuntimeError(f"DXGI capture failed: {e}")
    
    def capture_window(self, window_handle: int) -> CaptureFrame:
        """Capture a specific window using DXGI."""
        x, y, width, height = self.get_window_rect(window_handle)
        frame = self.capture_rect(x, y, width, height)
        frame.metadata['window_handle'] = window_handle
        return frame


class BitBltCaptureBackend(CaptureBackend):
    """BitBlt/MSS capture backend (fallback, always available on Windows)."""
    
    def get_name(self) -> str:
        return "BitBlt/MSS"
    
    def is_available(self) -> bool:
        """BitBlt via MSS should always be available on Windows."""
        try:
            import mss
            return True
        except ImportError:
            logger.warning("MSS library not available for BitBlt capture")
            return False
    
    def enumerate_monitors(self) -> List[MonitorInfo]:
        """Get monitor information using MSS."""
        monitors = []
        try:
            import mss
            
            with mss.mss() as sct:
                for i, monitor in enumerate(sct.monitors[1:], 1):  # Skip first (all monitors)
                    monitors.append(MonitorInfo(
                        monitor_id=i,
                        width=monitor['width'],
                        height=monitor['height'],
                        x=monitor['left'],
                        y=monitor['top'],
                        dpi_scale=1.0,  # TODO(claude): Get actual DPI scale
                        is_primary=(i == 1),  # Assume first monitor is primary
                        device_name=f"Monitor {i}"
                    ))
                    
        except ImportError:
            logger.warning("MSS not available for monitor enumeration")
        except Exception as e:
            logger.error(f"Failed to enumerate monitors with MSS: {e}")
            
        return monitors
    
    def find_windows(self, title_substring: str = None, pid: int = None) -> List[WindowInfo]:
        """Find windows using Win32 API (same as DXGI backend)."""
        return DXGICaptureBackend().find_windows(title_substring, pid)
    
    def get_window_rect(self, window_handle: int) -> Tuple[int, int, int, int]:
        """Get window rectangle using Win32 API (same as DXGI backend)."""
        return DXGICaptureBackend().get_window_rect(window_handle)
    
    def capture_rect(self, x: int, y: int, width: int, height: int) -> CaptureFrame:
        """Capture screen region using MSS BitBlt."""
        start_time = time.time()
        
        try:
            import mss
            import numpy as np
            
            # Define capture region
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            
            with mss.mss() as sct:
                # Capture the region
                screenshot = sct.grab(monitor)
                
                # Convert to numpy array and BGR format
                img_array = np.frombuffer(screenshot.rgb, dtype=np.uint8)
                img_array = img_array.reshape((screenshot.height, screenshot.width, 3))
                
                # MSS gives RGB, convert to BGR for OpenCV
                img_bgr = img_array[:, :, [2, 1, 0]]
                
                capture_duration = (time.time() - start_time) * 1000
                
                return CaptureFrame(
                    image=img_bgr,
                    timestamp=time.time(),
                    capture_duration_ms=capture_duration,
                    source_rect=(x, y, width, height),
                    backend_name=self.get_name(),
                    dpi_scale=1.0,  # TODO(claude): Get actual DPI scale
                    metadata={'method': 'mss_bitblt'}
                )
                
        except Exception as e:
            logger.error(f"BitBlt capture failed: {e}")
            raise RuntimeError(f"BitBlt capture failed: {e}")
    
    def capture_window(self, window_handle: int) -> CaptureFrame:
        """Capture a specific window using BitBlt."""
        x, y, width, height = self.get_window_rect(window_handle)
        frame = self.capture_rect(x, y, width, height)
        frame.metadata['window_handle'] = window_handle
        return frame


class AdaptiveCaptureManager:
    """Manages multiple capture backends with automatic fallback."""
    
    def __init__(self):
        self.backends = [
            DXGICaptureBackend(),
            BitBltCaptureBackend()
        ]
        self._active_backend = None
        self._backend_selection_attempted = False
    
    def _select_backend(self) -> CaptureBackend:
        """Select the best available backend."""
        if self._active_backend and not self._backend_selection_attempted:
            return self._active_backend
            
        self._backend_selection_attempted = True
        
        for backend in self.backends:
            try:
                if backend.is_available():
                    logger.info(f"Selected capture backend: {backend.get_name()}")
                    self._active_backend = backend
                    return backend
                else:
                    logger.info(f"Backend not available: {backend.get_name()}")
            except Exception as e:
                logger.warning(f"Backend availability check failed for {backend.get_name()}: {e}")
        
        raise RuntimeError("No capture backends available")
    
    def get_active_backend(self) -> CaptureBackend:
        """Get the currently active backend."""
        if self._active_backend is None:
            self._active_backend = self._select_backend()
        return self._active_backend
    
    def capture_rect(self, x: int, y: int, width: int, height: int) -> CaptureFrame:
        """Capture using the active backend with fallback."""
        backend = self.get_active_backend()
        
        try:
            return backend.capture_rect(x, y, width, height)
        except Exception as e:
            logger.error(f"Capture failed with {backend.get_name()}: {e}")
            
            # Try fallback backends
            for fallback_backend in self.backends:
                if fallback_backend != backend and fallback_backend.is_available():
                    try:
                        logger.info(f"Trying fallback backend: {fallback_backend.get_name()}")
                        frame = fallback_backend.capture_rect(x, y, width, height)
                        # Update active backend to the working one
                        self._active_backend = fallback_backend
                        logger.info(f"Switched to fallback backend: {fallback_backend.get_name()}")
                        return frame
                    except Exception as fallback_e:
                        logger.error(f"Fallback backend {fallback_backend.get_name()} also failed: {fallback_e}")
            
            # All backends failed
            raise RuntimeError(f"All capture backends failed. Last error: {e}")
    
    def find_hearthstone_window(self) -> Optional[WindowInfo]:
        """Find Hearthstone window using the active backend."""
        backend = self.get_active_backend()
        windows = backend.find_windows(title_substring="hearthstone")
        
        # Filter for reasonable window sizes (not minimized)
        viable_windows = [
            w for w in windows 
            if w.is_visible and not w.is_minimized and w.width > 100 and w.height > 100
        ]
        
        if viable_windows:
            # Return the largest window (likely the main game window)
            return max(viable_windows, key=lambda w: w.width * w.height)
        
        return None