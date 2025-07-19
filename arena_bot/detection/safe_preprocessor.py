"""
Safe Image Preprocessing for Enhanced Card Detection

Provides advanced image enhancement techniques with quality assessment
and graceful fallbacks to ensure preprocessing only improves detection.
Part of the Zero-Cost Detection Enhancement Plan.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ImageQualityMetrics:
    """Container for image quality assessment results."""
    brightness: float      # 0-255 range
    contrast: float        # Standard deviation of pixel values
    sharpness: float       # Laplacian variance (blur detection)
    overall_score: float   # 0-1 composite quality score


class SafeImagePreprocessor:
    """
    Safe image preprocessing with quality assessment and fallbacks.
    
    Only applies enhancements if they improve image quality.
    Always preserves original image as fallback option.
    """
    
    def __init__(self):
        """Initialize the safe image preprocessor."""
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds for enhancement decisions
        self.MIN_BRIGHTNESS = 30
        self.MAX_BRIGHTNESS = 220
        self.MIN_CONTRAST = 20
        self.MIN_SHARPNESS = 10
        
        # Enhancement parameters (conservative settings)
        self.CLAHE_CLIP_LIMIT = 2.0
        self.CLAHE_GRID_SIZE = (8, 8)
        self.BILATERAL_D = 9
        self.BILATERAL_SIGMA_COLOR = 75
        self.BILATERAL_SIGMA_SPACE = 75
        
        self.logger.info("SafeImagePreprocessor initialized with conservative settings")
    
    def assess_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Assess image quality across multiple metrics.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            ImageQualityMetrics object with quality scores
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Brightness assessment (mean pixel value)
            brightness = np.mean(gray)
            
            # Contrast assessment (standard deviation)
            contrast = np.std(gray)
            
            # Sharpness assessment (Laplacian variance - higher is sharper)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate overall quality score (0-1 range)
            brightness_score = self._normalize_brightness_score(brightness)
            contrast_score = min(1.0, contrast / 50.0)  # Normalize contrast
            sharpness_score = min(1.0, sharpness / 100.0)  # Normalize sharpness
            
            overall_score = (brightness_score + contrast_score + sharpness_score) / 3.0
            
            metrics = ImageQualityMetrics(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                overall_score=overall_score
            )
            
            self.logger.debug(f"Quality assessment - Brightness: {brightness:.1f}, "
                            f"Contrast: {contrast:.1f}, Sharpness: {sharpness:.1f}, "
                            f"Overall: {overall_score:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            # Return neutral quality metrics on error
            return ImageQualityMetrics(
                brightness=128.0,
                contrast=25.0,
                sharpness=50.0,
                overall_score=0.5
            )
    
    def _normalize_brightness_score(self, brightness: float) -> float:
        """Normalize brightness to 0-1 score (optimal around 100-150)."""
        if 100 <= brightness <= 150:
            return 1.0
        elif 80 <= brightness <= 180:
            return 0.8
        elif 50 <= brightness <= 200:
            return 0.6
        else:
            return 0.3
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Enhanced image with improved contrast
        """
        try:
            # Convert to LAB color space for better CLAHE results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to luminance channel
            clahe = cv2.createCLAHE(
                clipLimit=self.CLAHE_CLIP_LIMIT,
                tileGridSize=self.CLAHE_GRID_SIZE
            )
            l_enhanced = clahe.apply(l)
            
            # Merge back and convert to BGR
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            self.logger.warning(f"CLAHE enhancement failed: {e}")
            return image
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for noise reduction while preserving edges.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Filtered image with reduced noise
        """
        try:
            filtered = cv2.bilateralFilter(
                image,
                self.BILATERAL_D,
                self.BILATERAL_SIGMA_COLOR,
                self.BILATERAL_SIGMA_SPACE
            )
            return filtered
            
        except Exception as e:
            self.logger.warning(f"Bilateral filtering failed: {e}")
            return image
    
    def apply_unsharp_mask(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply unsharp masking for detail enhancement.
        
        Args:
            image: Input image (BGR format)
            strength: Enhancement strength (0.0-1.0)
            
        Returns:
            Sharpened image with enhanced details
        """
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            return unsharp_mask
            
        except Exception as e:
            self.logger.warning(f"Unsharp masking failed: {e}")
            return image
    
    def enhance_card_region(self, image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Apply complete enhancement pipeline with quality checking.
        
        Args:
            image: Input card region image
            aggressive: If True, apply stronger enhancements for poor quality images
            
        Returns:
            Enhanced image (or original if enhancement doesn't improve quality)
        """
        if image is None or image.size == 0:
            self.logger.warning("Invalid image provided for enhancement")
            return image
        
        try:
            # Assess original image quality
            original_quality = self.assess_image_quality(image)
            
            # Start with a copy to preserve original
            enhanced = image.copy()
            
            # Step 1: CLAHE for contrast enhancement
            if original_quality.contrast < self.MIN_CONTRAST or original_quality.brightness < 50:
                enhanced = self.apply_clahe(enhanced)
                self.logger.debug("Applied CLAHE for contrast enhancement")
            
            # Step 2: Bilateral filtering for noise reduction
            if original_quality.overall_score < 0.7:  # Apply if quality is below threshold
                enhanced = self.apply_bilateral_filter(enhanced)
                self.logger.debug("Applied bilateral filtering for noise reduction")
            
            # Step 3: Unsharp masking for detail enhancement
            sharpening_strength = 0.3 if not aggressive else 0.6
            if original_quality.sharpness < self.MIN_SHARPNESS:
                enhanced = self.apply_unsharp_mask(enhanced, strength=sharpening_strength)
                self.logger.debug(f"Applied unsharp masking with strength {sharpening_strength}")
            
            # Quality check: only use enhanced version if it's actually better
            enhanced_quality = self.assess_image_quality(enhanced)
            
            if enhanced_quality.overall_score > original_quality.overall_score:
                self.logger.debug(f"Enhancement successful: {original_quality.overall_score:.3f} → "
                                f"{enhanced_quality.overall_score:.3f}")
                return enhanced
            else:
                self.logger.debug(f"Enhancement did not improve quality: {original_quality.overall_score:.3f} → "
                                f"{enhanced_quality.overall_score:.3f}, using original")
                return image
                
        except Exception as e:
            self.logger.error(f"Enhancement pipeline failed: {e}")
            return image
    
    def prepare_multi_scale_regions(self, image: np.ndarray, scales: Tuple[float, ...] = (0.8, 1.0, 1.2)) -> Dict[str, np.ndarray]:
        """
        Prepare multiple scales of the same region for robust detection.
        
        Args:
            image: Input card region image
            scales: Tuple of scale factors to apply
            
        Returns:
            Dictionary mapping scale names to scaled images
        """
        scaled_regions = {}
        
        try:
            original_height, original_width = image.shape[:2]
            
            for scale in scales:
                scale_name = f"scale_{scale:.1f}".replace(".", "_")
                
                if scale == 1.0:
                    # Use original size
                    scaled_regions[scale_name] = image.copy()
                else:
                    # Calculate new dimensions
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    
                    # Resize with high-quality interpolation
                    if scale > 1.0:
                        # Upscaling: use cubic interpolation
                        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    else:
                        # Downscaling: use area interpolation
                        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    scaled_regions[scale_name] = scaled
                
                self.logger.debug(f"Prepared {scale_name}: {scaled_regions[scale_name].shape}")
            
            return scaled_regions
            
        except Exception as e:
            self.logger.error(f"Multi-scale preparation failed: {e}")
            return {"scale_1_0": image}
    
    def get_enhancement_recommendation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image and recommend specific enhancements.
        
        Args:
            image: Input image to analyze
            
        Returns:
            Dictionary with enhancement recommendations
        """
        try:
            quality = self.assess_image_quality(image)
            
            recommendations = {
                "apply_clahe": quality.contrast < self.MIN_CONTRAST or quality.brightness < 50,
                "apply_bilateral": quality.overall_score < 0.7,
                "apply_unsharp": quality.sharpness < self.MIN_SHARPNESS,
                "aggressive_mode": quality.overall_score < 0.4,
                "quality_metrics": quality,
                "enhancement_priority": []
            }
            
            # Priority ranking for enhancements
            if quality.brightness < 50 or quality.brightness > 200:
                recommendations["enhancement_priority"].append("brightness_correction")
            if quality.contrast < self.MIN_CONTRAST:
                recommendations["enhancement_priority"].append("contrast_enhancement")
            if quality.sharpness < self.MIN_SHARPNESS:
                recommendations["enhancement_priority"].append("sharpening")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Enhancement recommendation failed: {e}")
            return {"apply_clahe": False, "apply_bilateral": False, "apply_unsharp": False}


# Global instance for easy access
_safe_preprocessor = None


def get_safe_preprocessor() -> SafeImagePreprocessor:
    """
    Get the global SafeImagePreprocessor instance.
    
    Returns:
        SafeImagePreprocessor instance
    """
    global _safe_preprocessor
    if _safe_preprocessor is None:
        _safe_preprocessor = SafeImagePreprocessor()
    return _safe_preprocessor