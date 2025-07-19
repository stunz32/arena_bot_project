"""
Screen detection and capture functionality.

Based on Arena Tracker's proven screen detection methods.
Uses PyQt6 for cross-platform screen capture.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QScreen, QPixmap
from PyQt6.QtCore import QRect
import sys


class ScreenDetector:
    """
    Screen detection and capture system.
    
    Handles multi-monitor setups and provides screen capture functionality
    similar to Arena Tracker's approach.
    """
    
    def __init__(self):
        """Initialize screen detector."""
        self.logger = logging.getLogger(__name__)
        self.screens: List[QScreen] = []
        self.current_screen_index = -1
        
        # Initialize QApplication if not already done
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self._detect_screens()
    
    def _detect_screens(self):
        """Detect available screens."""
        self.screens = self.app.screens()
        self.logger.info(f"Detected {len(self.screens)} screen(s)")
        
        for i, screen in enumerate(self.screens):
            geometry = screen.geometry()
            self.logger.info(f"Screen {i}: {geometry.width()}x{geometry.height()} at ({geometry.x()}, {geometry.y()})")
    
    def get_screen_count(self) -> int:
        """Get the number of available screens."""
        return len(self.screens)
    
    def set_target_screen(self, screen_index: int) -> bool:
        """
        Set the target screen for capture.
        
        Args:
            screen_index: Index of the screen to use (-1 for auto-detect)
            
        Returns:
            True if successful, False otherwise
        """
        if screen_index == -1:
            self.current_screen_index = -1
            self.logger.info("Screen set to auto-detect mode")
            return True
        
        if 0 <= screen_index < len(self.screens):
            self.current_screen_index = screen_index
            geometry = self.screens[screen_index].geometry()
            self.logger.info(f"Screen set to index {screen_index}: {geometry.width()}x{geometry.height()}")
            return True
        
        self.logger.warning(f"Invalid screen index: {screen_index}")
        return False
    
    def capture_screen(self, screen_index: int = None) -> Optional[np.ndarray]:
        """
        Capture a full screen.
        
        Args:
            screen_index: Screen index to capture (None for current/auto)
            
        Returns:
            OpenCV image array or None if failed
        """
        if screen_index is None:
            screen_index = self.current_screen_index
        
        # Auto-detect screen if needed
        if screen_index == -1:
            screen_index = 0  # Default to primary screen
        
        if not (0 <= screen_index < len(self.screens)):
            self.logger.error(f"Invalid screen index: {screen_index}")
            return None
        
        try:
            screen = self.screens[screen_index]
            
            # Capture the screen
            pixmap = screen.grabWindow(0)
            
            # Convert QPixmap to OpenCV format
            image = self._qpixmap_to_opencv(pixmap)
            
            if image is not None:
                self.logger.debug(f"Screen captured: {image.shape}")
                return image
            else:
                self.logger.error("Failed to convert screen capture to OpenCV format")
                return None
                
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int, 
                      screen_index: int = None) -> Optional[np.ndarray]:
        """
        Capture a specific region of the screen.
        
        Args:
            x: Left coordinate
            y: Top coordinate  
            width: Region width
            height: Region height
            screen_index: Screen index to capture from
            
        Returns:
            OpenCV image array or None if failed
        """
        if screen_index is None:
            screen_index = self.current_screen_index
        
        if screen_index == -1:
            screen_index = 0
        
        if not (0 <= screen_index < len(self.screens)):
            self.logger.error(f"Invalid screen index: {screen_index}")
            return None
        
        try:
            screen = self.screens[screen_index]
            
            # Capture the specific region
            pixmap = screen.grabWindow(0, x, y, width, height)
            
            # Convert to OpenCV format
            image = self._qpixmap_to_opencv(pixmap)
            
            if image is not None:
                self.logger.debug(f"Region captured: {image.shape} at ({x}, {y})")
                return image
            else:
                self.logger.error("Failed to convert region capture to OpenCV format")
                return None
                
        except Exception as e:
            self.logger.error(f"Region capture failed: {e}")
            return None
    
    def _qpixmap_to_opencv(self, pixmap: QPixmap) -> Optional[np.ndarray]:
        """
        Convert QPixmap to OpenCV format.
        
        Args:
            pixmap: QPixmap to convert
            
        Returns:
            OpenCV image array or None if failed
        """
        try:
            # Convert QPixmap to QImage
            image = pixmap.toImage()
            
            # Convert to format compatible with OpenCV
            image = image.convertToFormat(image.Format.Format_RGB888)
            
            # Get image data
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            
            # Create numpy array
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            
            # Convert RGB to BGR (OpenCV format)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            return arr
            
        except Exception as e:
            self.logger.error(f"QPixmap to OpenCV conversion failed: {e}")
            return None
    
    def get_screen_geometry(self, screen_index: int = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Get screen geometry (x, y, width, height).
        
        Args:
            screen_index: Screen index (None for current)
            
        Returns:
            Tuple of (x, y, width, height) or None if invalid
        """
        if screen_index is None:
            screen_index = self.current_screen_index
        
        if screen_index == -1:
            screen_index = 0
        
        if not (0 <= screen_index < len(self.screens)):
            return None
        
        geometry = self.screens[screen_index].geometry()
        return (geometry.x(), geometry.y(), geometry.width(), geometry.height())
    
    def save_screenshot(self, filename: str, screen_index: int = None) -> bool:
        """
        Save a screenshot to file.
        
        Args:
            filename: Output filename
            screen_index: Screen index to capture
            
        Returns:
            True if successful, False otherwise
        """
        image = self.capture_screen(screen_index)
        
        if image is not None:
            try:
                cv2.imwrite(filename, image)
                self.logger.info(f"Screenshot saved: {filename}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save screenshot: {e}")
                return False
        
        return False


# Global screen detector instance
_screen_detector = None


def get_screen_detector() -> ScreenDetector:
    """
    Get the global screen detector instance.
    
    Returns:
        ScreenDetector instance
    """
    global _screen_detector
    if _screen_detector is None:
        _screen_detector = ScreenDetector()
    return _screen_detector