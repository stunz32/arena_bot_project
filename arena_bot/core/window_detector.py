"""
Hearthstone window detection system.

Automatically locates Hearthstone windows and arena UI elements.
Designed to work in headless environments without Qt dependencies.
"""

import cv2
import numpy as np
import logging
import subprocess
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class WindowInfo:
    """Information about a detected window."""
    process_name: str
    window_title: str
    x: int
    y: int
    width: int
    height: int
    process_id: int

@dataclass
class ArenaUIElements:
    """Detected arena UI element positions."""
    window_bounds: Tuple[int, int, int, int]  # x, y, width, height
    card_regions: List[Tuple[int, int, int, int]]  # List of (x, y, w, h) for each card
    confidence: float

class WindowDetector:
    """
    Hearthstone window detection and UI element location system.
    
    Automatically finds Hearthstone windows and calculates card positions
    based on UI template matching.
    """
    
    def __init__(self):
        """Initialize window detector."""
        self.logger = logging.getLogger(__name__)
        
        # Arena UI templates will be loaded here
        self.arena_templates: Dict[str, np.ndarray] = {}
        
        self.logger.info("WindowDetector initialized")
    
    def initialize(self) -> bool:
        """
        Initialize window detector by loading UI templates.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            from ..utils.asset_loader import get_asset_loader
            
            asset_loader = get_asset_loader()
            
            # Load arena UI templates
            template_names = [
                "arenaTemplate",
                "arenaTemplate2", 
                "collectionTemplate",
                "heroesTemplate",
                "heroesTemplate2"
            ]
            
            loaded_templates = 0
            for template_name in template_names:
                template = asset_loader.load_ui_template(template_name)
                if template is not None:
                    self.arena_templates[template_name] = template
                    loaded_templates += 1
                    self.logger.debug(f"Loaded UI template: {template_name}")
            
            if loaded_templates > 0:
                self.logger.info(f"WindowDetector initialized with {loaded_templates} UI templates")
                return True
            else:
                self.logger.warning("No UI templates loaded")
                return False
                
        except Exception as e:
            self.logger.error(f"WindowDetector initialization failed: {e}")
            return False
    
    def find_hearthstone_windows(self) -> List[WindowInfo]:
        """
        Find all Hearthstone windows on the system.
        
        Returns:
            List of detected Hearthstone windows
        """
        windows = []
        
        try:
            # For WSL/Linux environments, we'll use a different approach
            # than direct window enumeration since we're working with screenshots
            
            # For now, we'll implement a fallback method that assumes
            # the screenshot contains Hearthstone and tries to detect the UI
            self.logger.info("Searching for Hearthstone windows...")
            
            # TODO: Implement actual window enumeration for production
            # For testing purposes, we'll return a placeholder
            placeholder_window = WindowInfo(
                process_name="Hearthstone.exe",
                window_title="Hearthstone",
                x=0,
                y=0, 
                width=1920,
                height=1080,
                process_id=0
            )
            
            windows.append(placeholder_window)
            self.logger.info(f"Found {len(windows)} potential Hearthstone windows")
            
        except Exception as e:
            self.logger.error(f"Window detection failed: {e}")
        
        return windows
    
    def detect_arena_ui(self, screenshot: np.ndarray, 
                       window_bounds: Optional[Tuple[int, int, int, int]] = None) -> Optional[ArenaUIElements]:
        """
        Detect arena UI elements in a screenshot.
        
        Args:
            screenshot: Screenshot image
            window_bounds: Optional window bounds (x, y, width, height)
            
        Returns:
            ArenaUIElements if detected, None otherwise
        """
        try:
            self.logger.debug("Detecting arena UI elements...")
            
            if not self.arena_templates:
                self.logger.warning("No arena templates loaded for UI detection")
                return None
            
            height, width = screenshot.shape[:2]
            
            # Try to match arena templates at different scales
            best_match = None
            best_confidence = 0.0
            best_location = None
            best_scale = 1.0
            
            scales = [1.0, 0.8, 0.6, 1.2]  # Try different scales
            
            for template_name, template in self.arena_templates.items():
                for scale in scales:
                    # Skip if template would be too large
                    scaled_height = int(template.shape[0] * scale)
                    scaled_width = int(template.shape[1] * scale)
                    
                    if scaled_height > height or scaled_width > width:
                        continue
                    
                    # Resize template
                    if scale != 1.0:
                        scaled_template = cv2.resize(template, (scaled_width, scaled_height))
                    else:
                        scaled_template = template
                    
                    # Template matching
                    result = cv2.matchTemplate(screenshot, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    self.logger.debug(f"Template {template_name} @ {scale:.1f}x match confidence: {max_val:.3f}")
                    
                    if max_val > best_confidence:
                        best_confidence = max_val
                        best_match = template_name
                        best_location = max_loc
                        best_scale = scale
            
            # Check if we found a good match (lowered threshold since templates may not match exactly)
            if best_confidence > 0.3:  # More lenient threshold for arena UI detection
                self.logger.info(f"Arena UI detected using {best_match} @ {best_scale:.1f}x (confidence: {best_confidence:.3f})")
                
                # Calculate card regions based on detected UI
                scaled_template = self.arena_templates[best_match]
                if best_scale != 1.0:
                    scaled_height = int(scaled_template.shape[0] * best_scale)
                    scaled_width = int(scaled_template.shape[1] * best_scale)
                    scaled_template = cv2.resize(scaled_template, (scaled_width, scaled_height))
                
                card_regions = self._calculate_card_regions_from_ui(
                    best_location, 
                    scaled_template,
                    width, 
                    height
                )
                
                # Use provided window bounds or default to full screenshot
                if window_bounds is None:
                    window_bounds = (0, 0, width, height)
                
                return ArenaUIElements(
                    window_bounds=window_bounds,
                    card_regions=card_regions,
                    confidence=best_confidence
                )
            else:
                self.logger.debug(f"No arena UI detected (best confidence: {best_confidence:.3f})")
                return None
                
        except Exception as e:
            self.logger.error(f"Arena UI detection failed: {e}")
            return None
    
    def _calculate_card_regions_from_ui(self, ui_location: Tuple[int, int], 
                                       ui_template: np.ndarray,
                                       screen_width: int, screen_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate card regions based on detected UI template location.
        
        Args:
            ui_location: (x, y) location where UI template was found
            ui_template: The matched UI template
            screen_width: Screenshot width
            screen_height: Screenshot height
            
        Returns:
            List of card regions (x, y, width, height)
        """
        ui_x, ui_y = ui_location
        template_height, template_width = ui_template.shape[:2]
        
        # These offsets are based on Arena Tracker's UI analysis
        # Need to be adjusted based on actual arena interface layout
        
        # For now, use relative positioning similar to our manual regions
        # but calculated relative to the detected UI element
        
        card_width = 200
        card_height = 280
        
        # Estimate card positions relative to UI element
        # This is a simplified calculation - in practice would need precise measurements
        center_x = ui_x + template_width // 2
        center_y = ui_y + template_height // 2
        
        # Calculate positions for 3 cards
        card_regions = []
        
        # Left card
        left_x = max(0, center_x - int(screen_width * 0.25))
        card_y = max(0, center_y - card_height // 2)
        card_regions.append((left_x, card_y, card_width, card_height))
        
        # Center card  
        center_card_x = max(0, center_x - card_width // 2)
        card_regions.append((center_card_x, card_y, card_width, card_height))
        
        # Right card
        right_x = min(screen_width - card_width, center_x + int(screen_width * 0.25) - card_width)
        card_regions.append((right_x, card_y, card_width, card_height))
        
        self.logger.debug(f"Calculated card regions: {card_regions}")
        return card_regions
    
    def auto_detect_arena_cards(self, screenshot: np.ndarray) -> Optional[ArenaUIElements]:
        """
        Automatically detect arena interface and card positions.
        
        Args:
            screenshot: Screenshot to analyze
            
        Returns:
            ArenaUIElements if successful, None otherwise
        """
        try:
            # First try to detect arena UI
            ui_elements = self.detect_arena_ui(screenshot)
            
            if ui_elements is not None:
                self.logger.info(f"Auto-detected arena cards with confidence {ui_elements.confidence:.3f}")
                return ui_elements
            
            # If UI detection fails, fall back to manual positioning
            self.logger.info("UI detection failed, falling back to manual positioning")
            return self._fallback_manual_positioning(screenshot)
            
        except Exception as e:
            self.logger.error(f"Auto-detection failed: {e}")
            return None
    
    def _fallback_manual_positioning(self, screenshot: np.ndarray) -> ArenaUIElements:
        """
        Fallback to manual positioning when UI detection fails.
        
        Args:
            screenshot: Screenshot to analyze
            
        Returns:
            ArenaUIElements with manually calculated positions
        """
        height, width = screenshot.shape[:2]
        
        # Use our tested manual positioning logic as fallback
        if width > 3000:  # Ultrawide
            card_regions = [
                (int(width * 0.25 - 100), int(height * 0.35), 200, 280),   # Left card
                (int(width * 0.50 - 100), int(height * 0.35), 200, 280),   # Middle card  
                (int(width * 0.75 - 100), int(height * 0.35), 200, 280)    # Right card
            ]
        else:  # Standard resolution
            scale_x = width / 1920.0
            scale_y = height / 1080.0
            card_regions = [
                (int(480 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y)),
                (int(860 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y)),
                (int(1240 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y))
            ]
        
        self.logger.debug("Using fallback manual positioning")
        
        return ArenaUIElements(
            window_bounds=(0, 0, width, height),
            card_regions=card_regions,
            confidence=0.5  # Lower confidence for manual fallback
        )

# Global window detector instance
_window_detector = None

def get_window_detector() -> WindowDetector:
    """
    Get the global window detector instance.
    
    Returns:
        WindowDetector instance
    """
    global _window_detector
    if _window_detector is None:
        _window_detector = WindowDetector()
    return _window_detector