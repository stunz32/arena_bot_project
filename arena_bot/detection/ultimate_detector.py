"""
Ultimate Detection Engine

Combines all enhancement components into a single, powerful detection system:
- SafeImagePreprocessor for image enhancement
- FreeAlgorithmEnsemble for multi-algorithm feature detection  
- AdvancedTemplateValidator for template-based validation
- Intelligent voting and confidence boosting
- Graceful fallbacks to ensure reliability

Part of the Zero-Cost Detection Enhancement Plan.
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class DetectionMode(Enum):
    """Detection mode options."""
    BASIC = "basic"                    # Basic histogram matching only
    ENHANCED = "enhanced"              # With preprocessing and templates
    ENSEMBLE = "ensemble"              # Multi-algorithm ensemble
    ULTIMATE = "ultimate"              # Full enhancement pipeline


@dataclass
class UltimateDetectionResult:
    """Comprehensive detection result with all enhancement metrics."""
    card_code: str
    confidence: float
    distance: float
    
    # Enhancement details
    algorithm_used: str
    preprocessing_applied: bool
    template_validated: bool
    template_validation_score: float
    consensus_level: int              # Number of algorithms that agreed
    processing_time: float
    
    # Quality metrics
    image_quality_score: float
    enhancement_applied: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    
    # Detailed breakdown
    algorithm_results: Dict[str, Any] = field(default_factory=dict)
    template_details: Dict[str, Any] = field(default_factory=dict)


class UltimateDetectionEngine:
    """
    Ultimate card detection engine combining all enhancement techniques.
    
    Provides multiple detection modes with graceful fallbacks and comprehensive
    enhancement capabilities while maintaining compatibility with existing systems.
    """
    
    def __init__(self, mode: DetectionMode = DetectionMode.ULTIMATE):
        """
        Initialize the ultimate detection engine.
        
        Args:
            mode: Detection mode to use
        """
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Configuration flags
        self.config = {
            'enable_preprocessing': True,
            'enable_feature_ensemble': True,
            'enable_template_validation': True,
            'enable_consensus_boosting': True,
            'enable_quality_assessment': True,
            'fallback_to_basic': True,
            'max_processing_time': 5.0,
            'min_confidence_threshold': 0.3,
            'template_validation_weight': 0.3,
            'consensus_boost_weight': 0.2
        }
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"UltimateDetectionEngine initialized in {mode.value} mode")
    
    def _initialize_components(self):
        """Initialize all detection components with error handling."""
        # Basic histogram matcher (always required as fallback)
        try:
            from .histogram_matcher import get_histogram_matcher
            self.histogram_matcher = get_histogram_matcher()
            self.logger.info("✅ Histogram matcher initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize histogram matcher: {e}")
            self.histogram_matcher = None
        
        # Safe image preprocessor
        self.preprocessor = None
        if self.config['enable_preprocessing']:
            try:
                from .safe_preprocessor import get_safe_preprocessor
                self.preprocessor = get_safe_preprocessor()
                self.logger.info("✅ Safe preprocessor initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Safe preprocessor not available: {e}")
        
        # Feature ensemble
        self.feature_ensemble = None
        if self.config['enable_feature_ensemble']:
            try:
                from .feature_ensemble import get_feature_ensemble
                self.feature_ensemble = get_feature_ensemble(use_cache=True)
                self.logger.info("✅ Feature ensemble initialized with caching")
            except Exception as e:
                self.logger.warning(f"⚠️ Feature ensemble not available: {e}")
        
        # Template validator
        self.template_validator = None
        if self.config['enable_template_validation']:
            try:
                from .template_validator import get_template_validator
                self.template_validator = get_template_validator()
                self.logger.info("✅ Template validator initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Template validator not available: {e}")
        
        # Asset loader for database
        try:
            from ..utils.asset_loader import get_asset_loader
            self.asset_loader = get_asset_loader()
            self.logger.info("✅ Asset loader initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Asset loader not available: {e}")
            self.asset_loader = None
        
        # Feature cache manager for intelligent loading
        try:
            from .feature_cache_manager import FeatureCacheManager
            self.feature_cache_manager = FeatureCacheManager()
            self.logger.info("✅ Feature Cache Manager initialized for Ultimate Engine")
        except Exception as e:
            self.logger.warning(f"⚠️ Ultimate Engine could not access Feature Cache Manager: {e}")
            self.feature_cache_manager = None
    
    def load_card_database(self, card_images: Optional[Dict[str, np.ndarray]] = None):
        """
        Load card database for all detection components.
        
        Args:
            card_images: Optional pre-loaded card images
        """
        self.logger.info("Loading card database for Ultimate Engine...")
        
        # CRITICAL: First, check if a valid cache already covers our needs
        if self.feature_cache_manager and self.feature_ensemble:
            try:
                available_algorithms = self.feature_ensemble.get_available_algorithms()
                if self.feature_cache_manager.is_cache_valid_for_ensemble(available_algorithms):
                    self.logger.info("✅ Valid feature cache found! Skipping redundant computation.")
                    self.logger.info("   Ultimate Engine will load features from cache on demand.")
                    # Set internal flag to prevent future loads (if it exists)
                    if hasattr(self, '_ultimate_db_loaded'):
                        self._ultimate_db_loaded = True
                    return  # EXIT EARLY
            except Exception as e:
                self.logger.warning(f"Cache validation failed: {e}")
        
        # If no valid cache, proceed with the original logic
        self.logger.info("⚠️ No valid feature cache found. Computing features now...")
        
        # Load card images if not provided
        if card_images is None and self.asset_loader:
            try:
                # Get available card codes
                card_codes = self.asset_loader.get_available_cards()
                self.logger.info(f"Loading {len(card_codes)} card images...")
                
                card_images = {}
                for i, card_code in enumerate(card_codes):
                    if i % 1000 == 0:
                        self.logger.info(f"Loading card images... {i}/{len(card_codes)}")
                    
                    # Load normal card
                    image = self.asset_loader.load_card_image(card_code, premium=False)
                    if image is not None:
                        card_images[card_code] = image
                    
                    # Load premium card
                    premium_image = self.asset_loader.load_card_image(card_code, premium=True)
                    if premium_image is not None:
                        card_images[f"{card_code}_premium"] = premium_image
                
                self.logger.info(f"Loaded {len(card_images)} card images")
            except Exception as e:
                self.logger.error(f"Failed to load card images: {e}")
                return
        
        if not card_images:
            self.logger.warning("No card images available for database loading")
            return
        
        # Load into histogram matcher
        if self.histogram_matcher:
            try:
                self.histogram_matcher.load_card_database(card_images)
                self.logger.info(f"✅ Histogram matcher loaded {len(card_images)} cards")
            except Exception as e:
                self.logger.error(f"❌ Failed to load histogram database: {e}")
        
        # Load into feature ensemble
        if self.feature_ensemble:
            try:
                start_time = time.time()
                self.feature_ensemble.load_card_database(card_images)
                load_time = time.time() - start_time
                
                stats = self.feature_ensemble.get_database_stats()
                self.logger.info(f"✅ Feature ensemble loaded in {load_time:.2f}s: {stats}")
                
                # Log cache performance if caching is enabled
                if hasattr(self.feature_ensemble, 'use_cache') and self.feature_ensemble.use_cache:
                    self.logger.info(f"⚡ Fast loading achieved with feature caching!")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load feature database: {e}")
    
    def detect_card_ultimate(self, card_region: np.ndarray, 
                           candidate_pool: Optional[List[str]] = None) -> UltimateDetectionResult:
        """
        Perform ultimate card detection with all enhancements.
        
        Args:
            card_region: Card region image to detect
            candidate_pool: Optional pre-filtered candidate list
            
        Returns:
            UltimateDetectionResult with comprehensive detection information
        """
        start_time = time.time()
        
        # Initialize result
        result = UltimateDetectionResult(
            card_code="UNKNOWN",
            confidence=0.0,
            distance=1.0,
            algorithm_used="none",
            preprocessing_applied=False,
            template_validated=False,
            template_validation_score=0.0,
            consensus_level=0,
            processing_time=0.0,
            image_quality_score=0.0
        )
        
        try:
            # Step 1: Image preprocessing and quality assessment
            enhanced_region, quality_score = self._apply_safe_preprocessing(card_region)
            result.preprocessing_applied = enhanced_region is not card_region
            result.image_quality_score = quality_score
            
            # Step 2: Template-based database pre-filtering
            if candidate_pool is None:
                candidate_pool = self._get_template_filtered_database(enhanced_region)
            
            # Step 3: Multi-algorithm ensemble detection
            algorithm_results = self._run_ensemble_detection(enhanced_region, candidate_pool)
            result.algorithm_results = algorithm_results
            
            # Step 4: Template validation and conflict resolution
            validated_result = self._apply_template_validation(enhanced_region, algorithm_results)
            
            # Step 5: Consensus analysis and confidence boosting
            final_result = self._apply_consensus_boosting(enhanced_region, validated_result)
            
            # Update result with final values
            if final_result:
                result.card_code = final_result.get('card_code', 'UNKNOWN')
                result.confidence = final_result.get('confidence', 0.0)
                result.distance = final_result.get('distance', 1.0)
                result.algorithm_used = final_result.get('algorithm', 'ensemble')
                result.template_validated = final_result.get('template_validated', False)
                result.template_validation_score = final_result.get('template_score', 0.0)
                result.consensus_level = final_result.get('consensus_level', 0)
                result.template_details = final_result.get('template_details', {})
            
            result.processing_time = time.time() - start_time
            
            self.logger.debug(f"Ultimate detection: {result.card_code} "
                            f"(conf: {result.confidence:.3f}, time: {result.processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ultimate detection failed: {e}")
            result.fallback_reason = f"Error: {str(e)}"
            result.processing_time = time.time() - start_time
            
            # Emergency fallback
            return self._emergency_fallback(card_region, result)
    
    def _apply_safe_preprocessing(self, card_region: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply safe image preprocessing with quality assessment.
        
        Args:
            card_region: Input card region
            
        Returns:
            Tuple of (enhanced_image, quality_score)
        """
        if not self.preprocessor:
            return card_region, 0.5
        
        try:
            # Assess original quality
            quality_metrics = self.preprocessor.assess_image_quality(card_region)
            
            # Apply enhancement if needed
            if quality_metrics.overall_score < 0.7:
                enhanced = self.preprocessor.enhance_card_region(card_region, aggressive=False)
                self.logger.debug(f"Preprocessing applied (quality: {quality_metrics.overall_score:.3f})")
                return enhanced, quality_metrics.overall_score
            else:
                self.logger.debug(f"Preprocessing skipped (quality: {quality_metrics.overall_score:.3f})")
                return card_region, quality_metrics.overall_score
                
        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {e}")
            return card_region, 0.5
    
    def _get_template_filtered_database(self, card_region: np.ndarray) -> List[str]:
        """
        Get database pre-filtered by template information.
        
        Args:
            card_region: Card region for template extraction
            
        Returns:
            List of template-compatible card codes
        """
        if not self.template_validator:
            return []  # Will fall back to full database in histogram matcher
        
        try:
            # Get all available cards
            if self.asset_loader:
                full_database = self.asset_loader.get_available_cards()
            else:
                return []
            
            # Filter by template compatibility
            filtered = self.template_validator.get_template_compatible_database(
                card_region, full_database
            )
            
            self.logger.debug(f"Template filtering: {len(full_database)} → {len(filtered)} cards")
            return filtered
            
        except Exception as e:
            self.logger.warning(f"Template filtering failed: {e}")
            return []
    
    def _run_ensemble_detection(self, card_region: np.ndarray, 
                               candidate_pool: List[str]) -> Dict[str, Any]:
        """
        Run ensemble detection with all available algorithms.
        
        Args:
            card_region: Enhanced card region
            candidate_pool: Filtered candidate pool
            
        Returns:
            Dictionary with results from all algorithms
        """
        results = {}
        
        # Always run histogram matching (primary algorithm)
        if self.histogram_matcher:
            try:
                histogram_result = self.histogram_matcher.match_card(card_region)
                if histogram_result:
                    results['histogram'] = {
                        'card_code': histogram_result.card_code,
                        'confidence': histogram_result.confidence,
                        'distance': histogram_result.distance,
                        'is_premium': histogram_result.is_premium
                    }
                    self.logger.debug(f"Histogram: {histogram_result.card_code} "
                                    f"(conf: {histogram_result.confidence:.3f})")
            except Exception as e:
                self.logger.warning(f"Histogram matching failed: {e}")
        
        # Run feature ensemble if available
        if self.feature_ensemble:
            try:
                feature_result = self.feature_ensemble.get_best_match(card_region)
                if feature_result:
                    results['features'] = {
                        'card_code': feature_result.card_code,
                        'confidence': feature_result.confidence,
                        'distance': feature_result.distance,
                        'algorithm': feature_result.algorithm,
                        'feature_count': feature_result.feature_count
                    }
                    self.logger.debug(f"Features ({feature_result.algorithm}): {feature_result.card_code} "
                                    f"(conf: {feature_result.confidence:.3f})")
            except Exception as e:
                self.logger.warning(f"Feature detection failed: {e}")
        
        return results
    
    def _apply_template_validation(self, card_region: np.ndarray, 
                                 algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply template validation to algorithm results.
        
        Args:
            card_region: Card region for validation
            algorithm_results: Results from detection algorithms
            
        Returns:
            Enhanced results with template validation
        """
        if not self.template_validator or not algorithm_results:
            return algorithm_results
        
        try:
            validated_results = {}
            
            for algorithm, result in algorithm_results.items():
                card_code = result.get('card_code')
                if not card_code:
                    continue
                
                # Perform template validation
                validation = self.template_validator.validate_card_comprehensive(
                    card_region, card_code
                )
                
                # Calculate template boost
                template_boost = 0.0
                if validation.validation_score > 0.8:
                    template_boost = 0.20  # Strong validation
                elif validation.validation_score > 0.6:
                    template_boost = 0.10  # Moderate validation
                
                # Update result with template information
                enhanced_result = result.copy()
                enhanced_result['template_validated'] = validation.validation_score > 0.6
                enhanced_result['template_score'] = validation.validation_score
                enhanced_result['template_boost'] = template_boost
                enhanced_result['original_confidence'] = result.get('confidence', 0.0)
                enhanced_result['confidence'] = min(0.99, 
                    result.get('confidence', 0.0) + template_boost)
                enhanced_result['template_details'] = {
                    'mana_match': validation.mana_cost_match,
                    'rarity_match': validation.rarity_match,
                    'detected_mana': validation.detected_mana_cost,
                    'detected_rarity': validation.detected_rarity
                }
                
                validated_results[algorithm] = enhanced_result
                
                self.logger.debug(f"Template validation for {card_code}: "
                                f"score={validation.validation_score:.3f}, "
                                f"boost=+{template_boost:.3f}")
            
            return validated_results
            
        except Exception as e:
            self.logger.warning(f"Template validation failed: {e}")
            return algorithm_results
    
    def _apply_consensus_boosting(self, card_region: np.ndarray, 
                                validated_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply consensus analysis and confidence boosting.
        
        Args:
            card_region: Card region for analysis
            validated_results: Template-validated results
            
        Returns:
            Final detection result with consensus boosting
        """
        if not validated_results:
            return None
        
        try:
            # Count consensus for each card
            card_votes = {}
            for algorithm, result in validated_results.items():
                card_code = result.get('card_code')
                confidence = result.get('confidence', 0.0)
                
                if confidence > self.config['min_confidence_threshold']:
                    if card_code not in card_votes:
                        card_votes[card_code] = []
                    card_votes[card_code].append((algorithm, result))
            
            if not card_votes:
                return None
            
            # Find card with highest consensus
            best_card = None
            best_score = 0.0
            best_consensus = 0
            
            for card_code, votes in card_votes.items():
                consensus_level = len(votes)
                
                # Calculate weighted confidence
                total_confidence = sum(vote[1].get('confidence', 0.0) for vote in votes)
                avg_confidence = total_confidence / consensus_level
                
                # Apply consensus boost
                consensus_boost = 0.0
                if consensus_level >= 3:
                    consensus_boost = 0.15  # Strong consensus
                elif consensus_level >= 2:
                    consensus_boost = 0.08  # Moderate consensus
                
                final_confidence = min(0.99, avg_confidence + consensus_boost)
                
                # Combine confidence and consensus for final score
                consensus_score = (final_confidence * 0.8) + (consensus_level * 0.05)
                
                if consensus_score > best_score:
                    best_score = consensus_score
                    best_card = card_code
                    best_consensus = consensus_level
                    
                    # Select best individual result for this card
                    best_individual = max(votes, key=lambda x: x[1].get('confidence', 0.0))
                    
                    final_result = best_individual[1].copy()
                    final_result['card_code'] = card_code
                    final_result['confidence'] = final_confidence
                    final_result['consensus_level'] = consensus_level
                    final_result['consensus_boost'] = consensus_boost
                    final_result['algorithm'] = f"ensemble_{best_individual[0]}"
            
            if best_card:
                self.logger.debug(f"Consensus result: {best_card} "
                                f"(consensus: {best_consensus}, final_conf: {final_result['confidence']:.3f})")
                return final_result
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Consensus boosting failed: {e}")
            # Return best individual result as fallback
            if validated_results:
                best_result = max(validated_results.values(), 
                                key=lambda x: x.get('confidence', 0.0))
                return best_result
            return None
    
    def _emergency_fallback(self, card_region: np.ndarray, 
                          result: UltimateDetectionResult) -> UltimateDetectionResult:
        """
        Emergency fallback to basic histogram matching.
        
        Args:
            card_region: Original card region
            result: Current result to update
            
        Returns:
            Updated result with fallback detection
        """
        if not self.histogram_matcher:
            result.fallback_reason = "No fallback available - histogram matcher failed"
            return result
        
        try:
            fallback_result = self.histogram_matcher.match_card(card_region)
            if fallback_result:
                result.card_code = fallback_result.card_code
                result.confidence = fallback_result.confidence
                result.distance = fallback_result.distance
                result.algorithm_used = "histogram_fallback"
                result.fallback_reason = "Emergency fallback to basic detection"
                
                self.logger.info(f"Emergency fallback successful: {result.card_code}")
            else:
                result.fallback_reason = "Emergency fallback failed - no detection possible"
                
        except Exception as e:
            result.fallback_reason = f"Emergency fallback error: {str(e)}"
            self.logger.error(f"Emergency fallback failed: {e}")
        
        return result
    
    def detect_card_simple(self, card_region: np.ndarray) -> Optional[Any]:
        """
        Simple detection interface compatible with existing code.
        
        Args:
            card_region: Card region to detect
            
        Returns:
            Simple detection result compatible with CardMatch format
        """
        result = self.detect_card_ultimate(card_region)
        
        if result.confidence > self.config['min_confidence_threshold']:
            # Return object compatible with CardMatch
            class SimpleResult:
                def __init__(self, card_code, distance, confidence, is_premium=False):
                    self.card_code = card_code
                    self.distance = distance
                    self.confidence = confidence
                    self.is_premium = is_premium
            
            return SimpleResult(
                card_code=result.card_code,
                distance=result.distance,
                confidence=result.confidence,
                is_premium=result.card_code.endswith('_premium')
            )
        
        return None
    
    def set_mode(self, mode: DetectionMode):
        """Change detection mode."""
        self.mode = mode
        self.logger.info(f"Detection mode changed to {mode.value}")
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration parameters."""
        self.config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all components."""
        status = {
            'mode': self.mode.value,
            'config': self.config.copy(),
            'components': {
                'histogram_matcher': self.histogram_matcher is not None,
                'preprocessor': self.preprocessor is not None,
                'feature_ensemble': self.feature_ensemble is not None,
                'template_validator': self.template_validator is not None,
                'asset_loader': self.asset_loader is not None
            }
        }
        
        # Add component-specific status
        if self.feature_ensemble:
            status['feature_algorithms'] = self.feature_ensemble.get_available_algorithms()
            status['feature_database_stats'] = self.feature_ensemble.get_database_stats()
        
        if self.template_validator:
            status['template_validation_stats'] = self.template_validator.get_validation_stats()
        
        return status


# Global ultimate detector instance
_ultimate_detector = None


def get_ultimate_detector(mode: DetectionMode = DetectionMode.ULTIMATE) -> UltimateDetectionEngine:
    """
    Get the global UltimateDetectionEngine instance.
    
    Args:
        mode: Detection mode to use
        
    Returns:
        UltimateDetectionEngine instance
    """
    global _ultimate_detector
    if _ultimate_detector is None:
        _ultimate_detector = UltimateDetectionEngine(mode)
    return _ultimate_detector