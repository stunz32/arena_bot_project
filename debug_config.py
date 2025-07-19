#!/usr/bin/env python3
"""
Arena Bot Debug Configuration System
Centralized configuration for visual debugging, metrics logging, and validation testing.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

class DebugConfig:
    """Centralized debug configuration with intelligent defaults."""
    
    def __init__(self):
        # Global debug toggle (environment variable or manual override)
        self.DEBUG = os.getenv('ARENA_DEBUG', 'False').lower() in ('true', '1', 'yes')
        
        # Directory setup
        self.project_root = Path(__file__).parent
        self.debug_frames_dir = self.project_root / "debug_frames"
        self.debug_data_dir = self.project_root / "debug_data"
        
        # Create directories if they don't exist
        self.debug_frames_dir.mkdir(exist_ok=True)
        self.debug_data_dir.mkdir(exist_ok=True)
        
        # Visual debugging settings
        self.VISUAL_DEBUG = {
            'save_annotated_images': True,
            'show_anchor_points': True,
            'show_detected_boxes': True,
            'show_ground_truth_boxes': True,
            'show_iou_overlaps': True,
            'show_confidence_heatmap': True,
            'box_colors': {
                'detected': (0, 255, 0),      # Green for detected boxes
                'ground_truth': (255, 0, 0),  # Red for ground truth
                'iou_overlap': (0, 255, 255), # Yellow for overlaps
                'anchor_points': (255, 0, 255), # Magenta for anchors
            },
            'text_color': (255, 255, 255),    # White text
            'text_font_scale': 0.7,
            'box_thickness': 2,
            'point_radius': 5,
        }
        
        # Performance thresholds (based on computer vision best practices)
        self.THRESHOLDS = {
            'min_iou': 0.92,              # 92% overlap required for "good" detection
            'max_miss_rate': 0.005,       # 0.5% miss rate maximum
            'min_confidence': 0.8,        # 80% confidence minimum for card ID
            'max_detection_time_ms': 100, # 100ms maximum per detection
            'min_box_area': 15000,        # Minimum pixels for valid card region
            'max_box_area': 200000,       # Maximum pixels (prevent oversized boxes)
            'aspect_ratio_tolerance': 0.2, # ±20% from expected card aspect ratio
        }
        
        # Card dimensions and ratios (Hearthstone-specific)
        self.CARD_SPECS = {
            'expected_aspect_ratio': 0.67,  # Height/Width for Hearthstone cards
            'min_width': 200,               # Minimum card width in pixels
            'min_height': 280,              # Minimum card height in pixels
            'reference_resolution': (3440, 1440),  # User's ultrawide resolution
            'reference_card_size': (447, 493),     # Working card size from logs
        }
        
        # Detection method comparison settings
        self.DETECTION_METHODS = [
            'simple_working',
            'hybrid_cascade', 
            'enhanced_auto',
            'static_scaling',
            'contour_detection',
            'anchor_detection'
        ]
        
        # Metrics logging
        self.METRICS = {
            'csv_file': self.debug_data_dir / 'detection_metrics.csv',
            'log_file': self.debug_data_dir / 'debug.log',
            'performance_report': self.debug_data_dir / 'performance_report.json',
            'fields': [
                'timestamp', 'screenshot_file', 'resolution', 'detection_method',
                'card1_iou', 'card2_iou', 'card3_iou', 'mean_iou',
                'card1_confidence', 'card2_confidence', 'card3_confidence', 'mean_confidence',
                'detection_time_ms', 'total_cards_detected', 'miss_rate',
                'anchor_score', 'box_accuracy_score', 'overall_grade'
            ]
        }
        
        # Ground truth file location
        self.GROUND_TRUTH_FILE = self.debug_data_dir / 'ground_truth.json'
        
        # Logging configuration
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup debug logging with appropriate level."""
        log_level = logging.DEBUG if self.DEBUG else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.METRICS['log_file']),
                logging.StreamHandler()
            ]
        )
    
    def enable_debug(self):
        """Manually enable debug mode."""
        self.DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        print("🐛 DEBUG MODE ENABLED - Visual debugging and metrics logging active")
    
    def disable_debug(self):
        """Manually disable debug mode."""
        self.DEBUG = False
        logging.getLogger().setLevel(logging.INFO)
        print("📊 DEBUG MODE DISABLED - Normal operation mode")
    
    def get_debug_image_path(self, base_name: str, method: str = None) -> Path:
        """Generate debug image file path with timestamp and method."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if method:
            filename = f"{timestamp}_{method}_{base_name}_debug.png"
        else:
            filename = f"{timestamp}_{base_name}_debug.png"
        return self.debug_frames_dir / filename
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: (x, y, width, height) tuples
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        inter_x = max(x1, x2)
        inter_y = max(y1, y2)
        inter_w = max(0, min(x1 + w1, x2 + w2) - inter_x)
        inter_h = max(0, min(y1 + h1, y2 + h2) - inter_y)
        
        intersection = inter_w * inter_h
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def validate_box(self, box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Validate a detected card box against expected specifications.
        
        Returns:
            Dictionary with validation results and scores
        """
        x, y, w, h = box
        
        # Calculate metrics
        area = w * h
        aspect_ratio = h / w if w > 0 else 0
        expected_ratio = self.CARD_SPECS['expected_aspect_ratio']
        
        # Validation checks
        area_valid = self.THRESHOLDS['min_box_area'] <= area <= self.THRESHOLDS['max_box_area']
        size_valid = w >= self.CARD_SPECS['min_width'] and h >= self.CARD_SPECS['min_height']
        ratio_valid = abs(aspect_ratio - expected_ratio) <= self.THRESHOLDS['aspect_ratio_tolerance']
        
        # Calculate quality score (0-1)
        area_score = min(1.0, area / self.THRESHOLDS['min_box_area']) if area_valid else 0.0
        ratio_score = max(0.0, 1.0 - abs(aspect_ratio - expected_ratio) / expected_ratio)
        quality_score = (area_score + ratio_score) / 2
        
        return {
            'valid': area_valid and size_valid and ratio_valid,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'quality_score': quality_score,
            'area_valid': area_valid,
            'size_valid': size_valid,
            'ratio_valid': ratio_valid,
        }

# Global debug configuration instance
debug_config = DebugConfig()

# Convenience functions for easy access
def is_debug_enabled() -> bool:
    """Check if debug mode is active."""
    return debug_config.DEBUG

def enable_debug():
    """Enable debug mode globally."""
    debug_config.enable_debug()

def disable_debug():
    """Disable debug mode globally."""
    debug_config.disable_debug()

def get_debug_config() -> DebugConfig:
    """Get the global debug configuration instance."""
    return debug_config