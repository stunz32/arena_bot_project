#!/usr/bin/env python3
"""
Visual Debugger for Arena Bot Detection System
Creates annotated images showing detection accuracy, IoU overlaps, and confidence metrics.
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from debug_config import get_debug_config, is_debug_enabled

logger = logging.getLogger(__name__)

class VisualDebugger:
    """
    Advanced visual debugging system for card detection validation.
    Generates annotated images with detection overlays, metrics, and comparisons.
    """
    
    def __init__(self):
        self.config = get_debug_config()
        self.colors = self.config.VISUAL_DEBUG['box_colors']
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = self.config.VISUAL_DEBUG['text_font_scale']
        self.thickness = self.config.VISUAL_DEBUG['box_thickness']
        
    def create_debug_visualization(self, 
                                 screenshot: np.ndarray,
                                 detected_boxes: List[Tuple[int, int, int, int]],
                                 ground_truth_boxes: List[Tuple[int, int, int, int]] = None,
                                 detection_method: str = "unknown",
                                 card_names: List[str] = None,
                                 confidences: List[float] = None,
                                 anchor_points: List[Tuple[int, int]] = None,
                                 timing_ms: float = None) -> np.ndarray:
        """
        Create comprehensive debug visualization with all overlays.
        
        Args:
            screenshot: Original screenshot
            detected_boxes: List of (x, y, w, h) detected card regions
            ground_truth_boxes: List of (x, y, w, h) ground truth regions
            detection_method: Name of detection method used
            card_names: Identified card names (optional)
            confidences: Detection confidence scores (optional)
            anchor_points: Detected anchor points (optional)
            timing_ms: Detection timing in milliseconds
            
        Returns:
            Annotated debug image
        """
        if not is_debug_enabled():
            return screenshot
            
        # Create working copy
        debug_img = screenshot.copy()
        
        # Add ground truth boxes (red)
        if ground_truth_boxes:
            debug_img = self._draw_boxes(debug_img, ground_truth_boxes, 
                                       self.colors['ground_truth'], "GT", thickness=3)
        
        # Add detected boxes (green)
        debug_img = self._draw_boxes(debug_img, detected_boxes, 
                                   self.colors['detected'], "DET")
        
        # Calculate and visualize IoU overlaps
        if ground_truth_boxes and detected_boxes:
            debug_img = self._draw_iou_overlaps(debug_img, detected_boxes, ground_truth_boxes)
        
        # Add anchor points if provided
        if anchor_points:
            debug_img = self._draw_anchor_points(debug_img, anchor_points)
        
        # Add card information text
        debug_img = self._add_card_info_text(debug_img, detected_boxes, card_names, confidences)
        
        # Add method and timing information
        debug_img = self._add_header_info(debug_img, detection_method, timing_ms, 
                                        len(detected_boxes), ground_truth_boxes)
        
        # Add IoU metrics table
        if ground_truth_boxes and detected_boxes:
            debug_img = self._add_metrics_table(debug_img, detected_boxes, ground_truth_boxes)
        
        return debug_img
    
    def _draw_boxes(self, img: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int], label: str, thickness: int = None) -> np.ndarray:
        """Draw bounding boxes with labels."""
        if thickness is None:
            thickness = self.thickness
            
        for i, (x, y, w, h) in enumerate(boxes):
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label_text = f"{label}{i+1}"
            label_size = cv2.getTextSize(label_text, self.font, self.font_scale, 1)[0]
            cv2.rectangle(img, (x, y - label_size[1] - 5), 
                         (x + label_size[0] + 5, y), color, -1)
            cv2.putText(img, label_text, (x + 2, y - 3), self.font, 
                       self.font_scale, (255, 255, 255), 1)
        
        return img
    
    def _draw_iou_overlaps(self, img: np.ndarray, 
                          detected_boxes: List[Tuple[int, int, int, int]],
                          ground_truth_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw IoU overlap regions and calculate intersection areas."""
        
        # Find best matches between detected and ground truth boxes
        matches = self._find_best_box_matches(detected_boxes, ground_truth_boxes)
        
        for det_idx, gt_idx, iou_score in matches:
            if iou_score > 0.1:  # Only show meaningful overlaps
                det_box = detected_boxes[det_idx]
                gt_box = ground_truth_boxes[gt_idx]
                
                # Calculate intersection rectangle
                inter_rect = self._calculate_intersection_rect(det_box, gt_box)
                if inter_rect:
                    x, y, w, h = inter_rect
                    
                    # Draw semi-transparent overlap region
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), 
                                self.colors['iou_overlap'], -1)
                    img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
                    
                    # Add IoU score text
                    iou_text = f"IoU: {iou_score:.3f}"
                    cv2.putText(img, iou_text, (x + 5, y + 20), self.font,
                              self.font_scale * 0.8, (0, 0, 0), 2)
                    cv2.putText(img, iou_text, (x + 5, y + 20), self.font,
                              self.font_scale * 0.8, (255, 255, 255), 1)
        
        return img
    
    def _draw_anchor_points(self, img: np.ndarray, 
                           anchor_points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw anchor points used for detection."""
        for i, (x, y) in enumerate(anchor_points):
            # Draw anchor point circle
            cv2.circle(img, (x, y), self.config.VISUAL_DEBUG['point_radius'], 
                      self.colors['anchor_points'], -1)
            
            # Add anchor label
            anchor_text = f"A{i+1}"
            cv2.putText(img, anchor_text, (x + 8, y - 8), self.font,
                       self.font_scale * 0.8, self.colors['anchor_points'], 2)
        
        return img
    
    def _add_card_info_text(self, img: np.ndarray, 
                           boxes: List[Tuple[int, int, int, int]],
                           card_names: List[str] = None,
                           confidences: List[float] = None) -> np.ndarray:
        """Add card identification information near each box."""
        
        for i, (x, y, w, h) in enumerate(boxes):
            info_lines = []
            
            # Add card name if available
            if card_names and i < len(card_names):
                info_lines.append(f"Card: {card_names[i]}")
            
            # Add confidence if available
            if confidences and i < len(confidences):
                conf_color = (0, 255, 0) if confidences[i] > 0.8 else (0, 165, 255) if confidences[i] > 0.5 else (0, 0, 255)
                info_lines.append(f"Conf: {confidences[i]:.3f}")
            
            # Add box dimensions
            info_lines.append(f"Size: {w}×{h}")
            
            # Draw info box background
            if info_lines:
                text_height = 20 * len(info_lines)
                text_width = max(len(line) * 8 for line in info_lines)
                
                # Position info box to the right of card box
                info_x = x + w + 10
                info_y = y
                
                # Ensure info box stays within image bounds
                if info_x + text_width > img.shape[1]:
                    info_x = x - text_width - 10
                
                cv2.rectangle(img, (info_x - 5, info_y - 5), 
                             (info_x + text_width, info_y + text_height), 
                             (0, 0, 0), -1)
                cv2.rectangle(img, (info_x - 5, info_y - 5), 
                             (info_x + text_width, info_y + text_height), 
                             (255, 255, 255), 1)
                
                # Draw text lines
                for j, line in enumerate(info_lines):
                    text_y = info_y + 15 + (j * 20)
                    color = conf_color if "Conf:" in line and confidences else (255, 255, 255)
                    cv2.putText(img, line, (info_x, text_y), self.font,
                              self.font_scale * 0.7, color, 1)
        
        return img
    
    def _add_header_info(self, img: np.ndarray, detection_method: str, 
                        timing_ms: float, num_detected: int,
                        ground_truth_boxes: List = None) -> np.ndarray:
        """Add header with detection method, timing, and summary stats."""
        
        header_lines = [
            f"Detection Method: {detection_method}",
            f"Cards Detected: {num_detected}",
        ]
        
        if timing_ms is not None:
            timing_color = (0, 255, 0) if timing_ms < 50 else (0, 165, 255) if timing_ms < 100 else (0, 0, 255)
            header_lines.append(f"Timing: {timing_ms:.1f}ms")
        
        if ground_truth_boxes:
            header_lines.append(f"Ground Truth: {len(ground_truth_boxes)} cards")
        
        # Draw header background
        header_height = 25 * len(header_lines) + 10
        cv2.rectangle(img, (10, 10), (500, header_height), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (500, header_height), (255, 255, 255), 2)
        
        # Draw header text
        for i, line in enumerate(header_lines):
            y_pos = 30 + (i * 25)
            color = timing_color if "Timing:" in line and timing_ms else (255, 255, 255)
            cv2.putText(img, line, (15, y_pos), self.font, self.font_scale, color, 1)
        
        return img
    
    def _add_metrics_table(self, img: np.ndarray,
                          detected_boxes: List[Tuple[int, int, int, int]],
                          ground_truth_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Add IoU metrics table in bottom corner."""
        
        matches = self._find_best_box_matches(detected_boxes, ground_truth_boxes)
        
        # Calculate metrics
        iou_scores = [iou for _, _, iou in matches]
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        
        # Create metrics table
        metrics_lines = [
            "IoU METRICS:",
            f"Mean IoU: {mean_iou:.3f}",
            f"Min IoU: {min(iou_scores):.3f}" if iou_scores else "Min IoU: N/A",
            f"Max IoU: {max(iou_scores):.3f}" if iou_scores else "Max IoU: N/A",
        ]
        
        # Add per-card IoU scores
        for i, (_, _, iou) in enumerate(matches):
            metrics_lines.append(f"Card {i+1}: {iou:.3f}")
        
        # Position in bottom right
        table_width = 200
        table_height = 25 * len(metrics_lines) + 10
        start_x = img.shape[1] - table_width - 10
        start_y = img.shape[0] - table_height - 10
        
        # Draw table background
        cv2.rectangle(img, (start_x, start_y), 
                     (start_x + table_width, start_y + table_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(img, (start_x, start_y), 
                     (start_x + table_width, start_y + table_height), 
                     (255, 255, 255), 1)
        
        # Draw metrics text
        for i, line in enumerate(metrics_lines):
            y_pos = start_y + 20 + (i * 25)
            color = (0, 255, 0) if "Mean IoU:" in line and mean_iou > 0.9 else (255, 255, 255)
            cv2.putText(img, line, (start_x + 5, y_pos), self.font, 
                       self.font_scale * 0.6, color, 1)
        
        return img
    
    def _find_best_box_matches(self, detected_boxes: List[Tuple[int, int, int, int]], 
                              ground_truth_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, float]]:
        """Find best IoU matches between detected and ground truth boxes."""
        matches = []
        
        for i, det_box in enumerate(detected_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(ground_truth_boxes):
                iou = self.config.calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matches.append((i, best_gt_idx, best_iou))
        
        return matches
    
    def _calculate_intersection_rect(self, box1: Tuple[int, int, int, int], 
                                   box2: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Calculate intersection rectangle between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        inter_x = max(x1, x2)
        inter_y = max(y1, y2)
        inter_w = max(0, min(x1 + w1, x2 + w2) - inter_x)
        inter_h = max(0, min(y1 + h1, y2 + h2) - inter_y)
        
        if inter_w > 0 and inter_h > 0:
            return (inter_x, inter_y, inter_w, inter_h)
        return None
    
    def save_debug_image(self, debug_img: np.ndarray, base_name: str, 
                        detection_method: str = None) -> str:
        """Save debug image with timestamp and method info."""
        if not is_debug_enabled():
            return ""
            
        try:
            output_path = self.config.get_debug_image_path(base_name, detection_method)
            cv2.imwrite(str(output_path), debug_img)
            logger.info(f"Debug image saved: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save debug image: {e}")
            return ""
    
    def create_comparison_grid(self, screenshot: np.ndarray,
                              detection_results: Dict[str, Any]) -> np.ndarray:
        """Create side-by-side comparison of multiple detection methods."""
        if not detection_results:
            return screenshot
        
        # Create grid layout (2x3 for up to 6 methods)
        num_methods = len(detection_results)
        grid_cols = min(3, num_methods)
        grid_rows = (num_methods + grid_cols - 1) // grid_cols
        
        # Resize images for grid
        cell_height = 400
        cell_width = int(cell_height * screenshot.shape[1] / screenshot.shape[0])
        
        # Create empty grid
        grid_img = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
        
        for i, (method_name, result) in enumerate(detection_results.items()):
            row = i // grid_cols
            col = i % grid_cols
            
            # Create debug visualization for this method
            method_img = self.create_debug_visualization(
                screenshot,
                result.get('detected_boxes', []),
                result.get('ground_truth_boxes'),
                method_name,
                result.get('card_names'),
                result.get('confidences'),
                result.get('anchor_points'),
                result.get('timing_ms')
            )
            
            # Resize and place in grid
            resized_img = cv2.resize(method_img, (cell_width, cell_height))
            
            start_y = row * cell_height
            start_x = col * cell_width
            grid_img[start_y:start_y + cell_height, start_x:start_x + cell_width] = resized_img
        
        return grid_img

# Global visual debugger instance
visual_debugger = VisualDebugger()

def create_debug_visualization(*args, **kwargs) -> np.ndarray:
    """Convenience function for creating debug visualizations."""
    return visual_debugger.create_debug_visualization(*args, **kwargs)

def save_debug_image(*args, **kwargs) -> str:
    """Convenience function for saving debug images."""
    return visual_debugger.save_debug_image(*args, **kwargs)