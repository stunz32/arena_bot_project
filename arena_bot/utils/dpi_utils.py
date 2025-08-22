"""
DPI and coordinate normalization utilities for cross-platform compatibility.

This module provides utilities for handling different DPI scaling factors across
Windows, Linux, and other platforms to ensure consistent coordinate detection.
"""

from typing import Tuple, Dict, Any, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class DPINormalizer:
    """
    DPI-aware coordinate normalization for Arena draft detection.
    
    Handles Windows scaling factors (100%, 125%, 150%, 200%) and provides
    normalized coordinates that are consistent across different scaling levels.
    """
    
    # Standard Windows DPI scaling factors
    STANDARD_SCALING_FACTORS = [1.0, 1.25, 1.5, 2.0]
    
    # Reference resolution (1920x1080 at 100% scaling)
    REFERENCE_RESOLUTION = (1920, 1080)
    REFERENCE_DPI = 96
    
    def __init__(self, detected_scaling: Optional[float] = None):
        """
        Initialize DPI normalizer.
        
        Args:
            detected_scaling: Override scaling factor, or None for auto-detection
        """
        self.detected_scaling = detected_scaling
        self.logger = logger
        
    def detect_scaling_factor(self, screen_width: int, screen_height: int) -> float:
        """
        Detect DPI scaling factor from screen dimensions.
        
        Args:
            screen_width: Current screen width in pixels
            screen_height: Current screen height in pixels
            
        Returns:
            Estimated scaling factor (1.0, 1.25, 1.5, 2.0)
        """
        if self.detected_scaling is not None:
            return self.detected_scaling
            
        # Calculate scaling factor based on how much smaller the screen is vs reference
        ref_width, ref_height = self.REFERENCE_RESOLUTION
        
        # For common scaling scenarios (smaller screens = higher DPI scaling):
        # 100%: 1920x1080, 125%: 1536x864, 150%: 1280x720, 200%: 960x540
        scale_x = ref_width / screen_width
        scale_y = ref_height / screen_height
        
        # Use the average - this gives us the DPI scaling factor
        avg_scale = (scale_x + scale_y) / 2
        
        # Find closest standard scaling factor
        closest_scale = min(self.STANDARD_SCALING_FACTORS, 
                          key=lambda x: abs(x - avg_scale))
        
        self.logger.info(f"DPI scaling detection: screen={screen_width}x{screen_height}, "
                        f"calculated={avg_scale:.3f}, rounded={closest_scale}")
        
        return closest_scale
    
    def normalize_coordinates(self, coordinates: List[Tuple[int, int, int, int]], 
                            screen_width: int, screen_height: int,
                            target_scaling: float = 1.0) -> List[Tuple[int, int, int, int]]:
        """
        Normalize coordinates from current DPI scaling to target scaling.
        
        Args:
            coordinates: List of (x, y, width, height) tuples
            screen_width: Current screen width
            screen_height: Current screen height
            target_scaling: Target scaling factor (default 1.0 = 100%)
            
        Returns:
            Normalized coordinates at target scaling
        """
        current_scaling = self.detect_scaling_factor(screen_width, screen_height)
        
        if abs(current_scaling - target_scaling) < 0.01:
            return coordinates  # No normalization needed
            
        # Calculate normalization factor
        # To go FROM current scaling TO target scaling, we need to multiply by the ratio
        norm_factor = current_scaling / target_scaling
        
        normalized = []
        for x, y, w, h in coordinates:
            norm_x = int(x * norm_factor)
            norm_y = int(y * norm_factor)
            norm_w = int(w * norm_factor)
            norm_h = int(h * norm_factor)
            normalized.append((norm_x, norm_y, norm_w, norm_h))
            
        self.logger.debug(f"Normalized {len(coordinates)} coordinates: "
                         f"{current_scaling:.2f} → {target_scaling:.2f} "
                         f"(factor: {norm_factor:.3f})")
        
        return normalized
    
    def denormalize_coordinates(self, normalized_coords: List[Tuple[int, int, int, int]], 
                              screen_width: int, screen_height: int,
                              source_scaling: float = 1.0) -> List[Tuple[int, int, int, int]]:
        """
        Convert normalized coordinates back to current DPI scaling.
        
        Args:
            normalized_coords: Coordinates at source scaling
            screen_width: Target screen width
            screen_height: Target screen height
            source_scaling: Source scaling factor (default 1.0 = 100%)
            
        Returns:
            Coordinates adjusted for current DPI scaling
        """
        current_scaling = self.detect_scaling_factor(screen_width, screen_height)
        
        if abs(current_scaling - source_scaling) < 0.01:
            return normalized_coords  # No denormalization needed
            
        # Calculate denormalization factor  
        denorm_factor = source_scaling / current_scaling
        
        denormalized = []
        for x, y, w, h in normalized_coords:
            denorm_x = int(x * denorm_factor)
            denorm_y = int(y * denorm_factor)
            denorm_w = int(w * denorm_factor)
            denorm_h = int(h * denorm_factor)
            denormalized.append((denorm_x, denorm_y, denorm_w, denorm_h))
            
        self.logger.debug(f"Denormalized {len(normalized_coords)} coordinates: "
                         f"{source_scaling:.2f} → {current_scaling:.2f} "
                         f"(factor: {denorm_factor:.3f})")
        
        return denormalized
    
    def validate_coordinate_accuracy(self, 
                                   detected_coords: List[Tuple[int, int, int, int]], 
                                   expected_coords: List[Tuple[int, int, int, int]], 
                                   tolerance: int = 2) -> Dict[str, Any]:
        """
        Validate coordinate detection accuracy within tolerance.
        
        Args:
            detected_coords: Actually detected coordinates
            expected_coords: Expected ground truth coordinates
            tolerance: Maximum pixel deviation allowed (default ±2px)
            
        Returns:
            Validation results with pass/fail status and detailed metrics
        """
        if len(detected_coords) != len(expected_coords):
            return {
                'passed': False,
                'error': f"Count mismatch: detected {len(detected_coords)}, expected {len(expected_coords)}",
                'detected_count': len(detected_coords),
                'expected_count': len(expected_coords)
            }
        
        results = []
        all_passed = True
        max_deviation = 0
        
        for i, ((det_x, det_y, det_w, det_h), (exp_x, exp_y, exp_w, exp_h)) in enumerate(
            zip(detected_coords, expected_coords)
        ):
            # Calculate deviations
            dx = abs(det_x - exp_x)
            dy = abs(det_y - exp_y)
            dw = abs(det_w - exp_w)
            dh = abs(det_h - exp_h)
            
            coord_max_dev = max(dx, dy, dw, dh)
            max_deviation = max(max_deviation, coord_max_dev)
            
            coord_passed = coord_max_dev <= tolerance
            if not coord_passed:
                all_passed = False
                
            results.append({
                'index': i,
                'detected': (det_x, det_y, det_w, det_h),
                'expected': (exp_x, exp_y, exp_w, exp_h),
                'deviations': (dx, dy, dw, dh),
                'max_deviation': coord_max_dev,
                'passed': coord_passed
            })
        
        return {
            'passed': all_passed,
            'tolerance': tolerance,
            'max_deviation': max_deviation,
            'results': results,
            'summary': f"{sum(1 for r in results if r['passed'])}/{len(results)} coordinates passed"
        }


def create_scaled_fixtures(base_image_path: str, output_dir: str, 
                          scaling_factors: List[float] = None) -> Dict[float, str]:
    """
    Create DPI-scaled versions of a base fixture image.
    
    Args:
        base_image_path: Path to base image (100% scaling reference)
        output_dir: Directory to save scaled versions
        scaling_factors: List of scaling factors to generate
        
    Returns:
        Dictionary mapping scaling factor to output file path
    """
    if scaling_factors is None:
        scaling_factors = DPINormalizer.STANDARD_SCALING_FACTORS
        
    try:
        import cv2
        import os
        
        # Load base image
        base_image = cv2.imread(base_image_path)
        if base_image is None:
            raise ValueError(f"Could not load base image: {base_image_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        created_files = {}
        base_name = os.path.splitext(os.path.basename(base_image_path))[0]
        
        for scale in scaling_factors:
            # Calculate new dimensions
            h, w = base_image.shape[:2]
            new_w = int(w / scale)  # Smaller image for higher DPI
            new_h = int(h / scale)
            
            # Resize image
            scaled_image = cv2.resize(base_image, (new_w, new_h), 
                                    interpolation=cv2.INTER_AREA)
            
            # Save scaled version
            scale_percent = int(scale * 100)
            output_path = os.path.join(output_dir, f"{base_name}_{scale_percent}pct.png")
            cv2.imwrite(output_path, scaled_image)
            
            created_files[scale] = output_path
            logger.info(f"Created DPI fixture: {scale*100:.0f}% scaling → {output_path}")
            
        return created_files
        
    except ImportError:
        logger.error("OpenCV not available for fixture creation")
        return {}
    except Exception as e:
        logger.error(f"Error creating scaled fixtures: {e}")
        return {}