"""
Advanced Template Validation System

Enhances template matching with intelligent validation, candidate filtering,
and cross-validation capabilities. Integrates with existing template_matcher.py
to provide comprehensive template-based verification.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import time


@dataclass
class TemplateValidationResult:
    """Container for template validation results."""
    card_code: str
    validation_score: float      # 0-1 overall validation score
    mana_cost_match: bool
    detected_mana_cost: Optional[int]
    expected_mana_cost: Optional[int]
    rarity_match: bool
    detected_rarity: Optional[int]
    expected_rarity: Optional[int]
    validation_details: Dict[str, Any]


@dataclass
class TemplateInfo:
    """Container for extracted template information."""
    mana_cost: Optional[int] = None
    rarity: Optional[int] = None
    mana_detection_confidence: float = 0.0
    rarity_detection_confidence: float = 0.0
    extraction_quality: float = 0.0


class AdvancedTemplateValidator:
    """
    Advanced template validation system for enhanced card detection.
    
    Provides intelligent template-based validation, candidate filtering,
    and cross-validation to boost detection accuracy.
    """
    
    def __init__(self):
        """Initialize the advanced template validator."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize template matcher
        try:
            from .template_matcher import get_template_matcher
            self.template_matcher = get_template_matcher()
            self.template_matcher.initialize()
            self.logger.info("Template matcher initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize template matcher: {e}")
            self.template_matcher = None
        
        # Initialize card database for metadata lookup
        try:
            from ..data.cards_json_loader import get_cards_json_loader
            self.cards_loader = get_cards_json_loader()
            self.logger.info("Cards JSON loader initialized")
        except Exception as e:
            self.logger.warning(f"Cards JSON loader not available: {e}")
            self.cards_loader = None
        
        # Template validation weights
        self.MANA_COST_WEIGHT = 0.5      # High weight for mana cost
        self.RARITY_WEIGHT = 0.3         # Medium weight for rarity
        self.QUALITY_WEIGHT = 0.2        # Low weight for extraction quality
        
        # Confidence thresholds - further adjusted for better card detection
        self.MIN_TEMPLATE_CONFIDENCE = 0.20  # Lowered from 0.30 to 0.20 for maximum permissiveness
        self.HIGH_CONFIDENCE_THRESHOLD = 0.45  # Lowered from 0.65 to 0.45
        
        self.logger.info("AdvancedTemplateValidator initialized")
    
    def extract_template_regions(self, card_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract template regions from card image.
        
        Args:
            card_image: Full card image
            
        Returns:
            Dictionary with extracted regions for each template type
        """
        regions = {}
        
        try:
            height, width = card_image.shape[:2]
            
            # Mana cost region (top-left corner)
            mana_x = int(width * 0.02)  # 2% from left
            mana_y = int(height * 0.02)  # 2% from top
            mana_w = int(width * 0.15)   # 15% width
            mana_h = int(height * 0.20)  # 20% height
            
            regions['mana'] = card_image[mana_y:mana_y+mana_h, mana_x:mana_x+mana_w]
            
            # Rarity region (center-bottom area)
            rarity_x = int(width * 0.35)  # 35% from left
            rarity_y = int(height * 0.75)  # 75% from top
            rarity_w = int(width * 0.30)   # 30% width
            rarity_h = int(height * 0.20)  # 20% height
            
            regions['rarity'] = card_image[rarity_y:rarity_y+rarity_h, rarity_x:rarity_x+rarity_w]
            
            # Future: Add more template regions (class icon, set symbol, etc.)
            
            self.logger.debug(f"Extracted {len(regions)} template regions")
            return regions
            
        except Exception as e:
            self.logger.error(f"Template region extraction failed: {e}")
            return {}
    
    def extract_all_templates(self, card_image: np.ndarray) -> TemplateInfo:
        """
        Extract all template information from card image.
        
        Args:
            card_image: Full card image
            
        Returns:
            TemplateInfo object with all detected template data
        """
        if self.template_matcher is None:
            return TemplateInfo()
        
        try:
            # Extract template regions
            regions = self.extract_template_regions(card_image)
            
            template_info = TemplateInfo()
            
            # Detect mana cost
            if 'mana' in regions and regions['mana'].size > 0:
                mana_cost = self.template_matcher.detect_mana_cost(regions['mana'])
                template_info.mana_cost = mana_cost
                template_info.mana_detection_confidence = 0.8 if mana_cost is not None else 0.0
                
                self.logger.debug(f"Detected mana cost: {mana_cost}")
            
            # Detect rarity
            if 'rarity' in regions and regions['rarity'].size > 0:
                rarity = self.template_matcher.detect_rarity(regions['rarity'])
                template_info.rarity = rarity
                template_info.rarity_detection_confidence = 0.7 if rarity is not None else 0.0
                
                self.logger.debug(f"Detected rarity: {rarity}")
            
            # Assess overall extraction quality
            template_info.extraction_quality = self._assess_extraction_quality(regions)
            
            return template_info
            
        except Exception as e:
            self.logger.error(f"Template extraction failed: {e}")
            return TemplateInfo()
    
    def _assess_extraction_quality(self, regions: Dict[str, np.ndarray]) -> float:
        """
        Assess quality of template region extraction.
        
        Args:
            regions: Extracted template regions
            
        Returns:
            Quality score (0-1)
        """
        try:
            total_quality = 0.0
            region_count = 0
            
            for region_name, region in regions.items():
                if region.size == 0:
                    continue
                
                # Check region size
                min_size = 20 * 20  # Minimum useful size
                size_score = min(1.0, region.size / min_size) if region.size >= min_size else 0.5
                
                # Check contrast
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                contrast = np.std(gray)
                contrast_score = min(1.0, contrast / 30.0)  # Normalize contrast
                
                # Check brightness
                brightness = np.mean(gray)
                brightness_score = 1.0 if 50 < brightness < 200 else 0.7
                
                region_quality = (size_score + contrast_score + brightness_score) / 3.0
                total_quality += region_quality
                region_count += 1
            
            return total_quality / region_count if region_count > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return 0.5
    
    def get_card_metadata(self, card_code: str) -> Dict[str, Any]:
        """
        Get card metadata for validation.
        
        Args:
            card_code: Hearthstone card code
            
        Returns:
            Dictionary with card metadata
        """
        metadata = {
            'mana_cost': None,
            'rarity': None,
            'card_class': None,
            'card_set': None
        }
        
        try:
            if self.cards_loader:
                # Get card data from JSON loader
                card_data = self.cards_loader.cards_data.get(card_code)
                if card_data:
                    metadata['mana_cost'] = card_data.get('cost', None)
                    
                    # Convert rarity text to numeric
                    rarity_text = card_data.get('rarity', '').upper()
                    rarity_map = {
                        'COMMON': 0,
                        'RARE': 1,
                        'EPIC': 2,
                        'LEGENDARY': 3
                    }
                    metadata['rarity'] = rarity_map.get(rarity_text, None)
                    
                    metadata['card_class'] = card_data.get('cardClass', None)
                    metadata['card_set'] = card_data.get('set', None)
            
            return metadata
            
        except Exception as e:
            self.logger.debug(f"Failed to get metadata for {card_code}: {e}")
            return metadata
    
    def validate_card_comprehensive(self, card_image: np.ndarray, candidate_card: str) -> TemplateValidationResult:
        """
        Perform comprehensive template validation for a candidate card.
        
        Args:
            card_image: Card image to validate
            candidate_card: Candidate card code
            
        Returns:
            TemplateValidationResult with detailed validation information
        """
        try:
            # Extract template information from image
            template_info = self.extract_all_templates(card_image)
            
            # Get expected card metadata
            expected_metadata = self.get_card_metadata(candidate_card)
            
            # Validate mana cost
            mana_cost_match = False
            if (template_info.mana_cost is not None and 
                expected_metadata['mana_cost'] is not None):
                mana_cost_match = template_info.mana_cost == expected_metadata['mana_cost']
            
            # Validate rarity
            rarity_match = False
            if (template_info.rarity is not None and 
                expected_metadata['rarity'] is not None):
                rarity_match = template_info.rarity == expected_metadata['rarity']
            
            # Calculate overall validation score
            validation_score = self._calculate_validation_score(
                template_info, expected_metadata, mana_cost_match, rarity_match
            )
            
            # Prepare validation details
            validation_details = {
                'template_info': template_info,
                'expected_metadata': expected_metadata,
                'extraction_quality': template_info.extraction_quality,
                'mana_confidence': template_info.mana_detection_confidence,
                'rarity_confidence': template_info.rarity_detection_confidence
            }
            
            result = TemplateValidationResult(
                card_code=candidate_card,
                validation_score=validation_score,
                mana_cost_match=mana_cost_match,
                detected_mana_cost=template_info.mana_cost,
                expected_mana_cost=expected_metadata['mana_cost'],
                rarity_match=rarity_match,
                detected_rarity=template_info.rarity,
                expected_rarity=expected_metadata['rarity'],
                validation_details=validation_details
            )
            
            self.logger.debug(f"Validation for {candidate_card}: score={validation_score:.3f}, "
                            f"mana={mana_cost_match}, rarity={rarity_match}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Template validation failed for {candidate_card}: {e}")
            return TemplateValidationResult(
                card_code=candidate_card,
                validation_score=0.0,
                mana_cost_match=False,
                detected_mana_cost=None,
                expected_mana_cost=None,
                rarity_match=False,
                detected_rarity=None,
                expected_rarity=None,
                validation_details={}
            )
    
    def _calculate_validation_score(self, template_info: TemplateInfo, 
                                  expected_metadata: Dict[str, Any],
                                  mana_cost_match: bool, rarity_match: bool) -> float:
        """
        Calculate overall validation score.
        
        Args:
            template_info: Extracted template information
            expected_metadata: Expected card metadata
            mana_cost_match: Whether mana cost matches
            rarity_match: Whether rarity matches
            
        Returns:
            Validation score (0-1)
        """
        score = 0.0
        
        # Mana cost component
        if template_info.mana_cost is not None and expected_metadata['mana_cost'] is not None:
            if mana_cost_match:
                score += self.MANA_COST_WEIGHT * template_info.mana_detection_confidence
            # No penalty for mismatch - might be detection error
        else:
            # Partial score if we can't verify mana cost
            score += self.MANA_COST_WEIGHT * 0.5
        
        # Rarity component
        if template_info.rarity is not None and expected_metadata['rarity'] is not None:
            if rarity_match:
                score += self.RARITY_WEIGHT * template_info.rarity_detection_confidence
            # No penalty for mismatch - might be detection error
        else:
            # Partial score if we can't verify rarity
            score += self.RARITY_WEIGHT * 0.5
        
        # Quality component
        score += self.QUALITY_WEIGHT * template_info.extraction_quality
        
        return min(1.0, score)
    
    def filter_candidates_by_templates(self, card_image: np.ndarray, 
                                     candidates: List[str]) -> List[str]:
        """
        Filter candidate cards using template information.
        
        Args:
            card_image: Card image for template extraction
            candidates: List of candidate card codes
            
        Returns:
            Filtered list of candidates that pass template validation
        """
        if not candidates:
            return candidates
        
        try:
            # Extract template information
            template_info = self.extract_all_templates(card_image)
            
            # If no template info extracted, return all candidates
            if (template_info.mana_cost is None and template_info.rarity is None):
                self.logger.debug("No template info extracted, returning all candidates")
                return candidates
            
            filtered_candidates = []
            
            for candidate in candidates:
                metadata = self.get_card_metadata(candidate)
                
                # Check mana cost filter
                mana_pass = True
                if template_info.mana_cost is not None and metadata['mana_cost'] is not None:
                    mana_pass = template_info.mana_cost == metadata['mana_cost']
                
                # Check rarity filter
                rarity_pass = True
                if template_info.rarity is not None and metadata['rarity'] is not None:
                    rarity_pass = template_info.rarity == metadata['rarity']
                
                # Include candidate if it passes all available filters
                if mana_pass and rarity_pass:
                    filtered_candidates.append(candidate)
            
            self.logger.debug(f"Template filtering: {len(candidates)} → {len(filtered_candidates)} candidates")
            
            # If filtering removes all candidates, return original list (detection might be wrong)
            if not filtered_candidates:
                self.logger.warning("Template filtering removed all candidates, returning original list")
                return candidates
            
            return filtered_candidates
            
        except Exception as e:
            self.logger.error(f"Template filtering failed: {e}")
            return candidates
    
    def get_template_compatible_database(self, card_image: np.ndarray, 
                                       full_database: List[str]) -> List[str]:
        """
        Pre-filter card database using template information.
        
        Args:
            card_image: Card image for template extraction
            full_database: Full list of card codes in database
            
        Returns:
            Filtered database of template-compatible cards
        """
        return self.filter_candidates_by_templates(card_image, full_database)
    
    def resolve_detection_conflicts(self, card_image: np.ndarray, 
                                  conflicting_candidates: List[str]) -> Optional[str]:
        """
        Use template validation to resolve conflicts between detection algorithms.
        
        Args:
            card_image: Card image for validation
            conflicting_candidates: List of conflicting candidate cards
            
        Returns:
            Best candidate based on template validation, or None
        """
        if not conflicting_candidates:
            return None
        
        try:
            best_candidate = None
            best_score = 0.0
            
            for candidate in conflicting_candidates:
                validation_result = self.validate_card_comprehensive(card_image, candidate)
                
                if validation_result.validation_score > best_score:
                    best_score = validation_result.validation_score
                    best_candidate = candidate
            
            # Only return candidate if validation score is reasonable
            if best_score >= self.MIN_TEMPLATE_CONFIDENCE:
                self.logger.debug(f"Template resolution: {best_candidate} (score: {best_score:.3f})")
                return best_candidate
            else:
                self.logger.debug(f"No candidate passed template validation threshold ({best_score:.3f} < {self.MIN_TEMPLATE_CONFIDENCE})")
                return None
                
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return None
    
    def boost_confidence_with_templates(self, card_image: np.ndarray, 
                                      detection_result: Any) -> float:
        """
        Calculate confidence boost based on template validation.
        
        Args:
            card_image: Card image for validation
            detection_result: Detection result with card_code attribute
            
        Returns:
            Confidence boost amount (0.0-0.3)
        """
        try:
            if not hasattr(detection_result, 'card_code'):
                return 0.0
            
            validation_result = self.validate_card_comprehensive(
                card_image, detection_result.card_code
            )
            
            # Calculate boost based on validation score
            if validation_result.validation_score >= self.HIGH_CONFIDENCE_THRESHOLD:
                boost = 0.25  # Strong template validation
            elif validation_result.validation_score >= self.MIN_TEMPLATE_CONFIDENCE:
                boost = 0.15  # Moderate template validation
            elif validation_result.validation_score >= 0.4:
                boost = 0.05  # Weak template validation
            else:
                boost = 0.0   # No template validation
            
            self.logger.debug(f"Template confidence boost for {detection_result.card_code}: +{boost:.3f}")
            return boost
            
        except Exception as e:
            self.logger.error(f"Confidence boost calculation failed: {e}")
            return 0.0
    
    def is_template_matcher_available(self) -> bool:
        """Check if template matcher is available and working."""
        return self.template_matcher is not None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about template validation performance."""
        if self.template_matcher is None:
            return {"status": "unavailable"}
        
        try:
            mana_count, rarity_count = self.template_matcher.get_template_counts()
            return {
                "status": "available",
                "mana_templates": mana_count,
                "rarity_templates": rarity_count,
                "cards_loader": self.cards_loader is not None,
                "validation_weights": {
                    "mana_cost": self.MANA_COST_WEIGHT,
                    "rarity": self.RARITY_WEIGHT,
                    "quality": self.QUALITY_WEIGHT
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get validation stats: {e}")
            return {"status": "error", "error": str(e)}


# Global template validator instance
_template_validator = None


def get_template_validator() -> AdvancedTemplateValidator:
    """
    Get the global AdvancedTemplateValidator instance.
    
    Returns:
        AdvancedTemplateValidator instance
    """
    global _template_validator
    if _template_validator is None:
        _template_validator = AdvancedTemplateValidator()
    return _template_validator