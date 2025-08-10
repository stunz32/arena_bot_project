"""
Validation engine for card detection results.

Combines histogram matching with template matching validation.
Based on Arena Tracker's proven validation approach.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from .histogram_matcher import CardMatch
from .template_matcher import get_template_matcher


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    confidence: float
    mana_cost: Optional[int]
    rarity: Optional[int]
    validation_score: float


class ValidationEngine:
    """
    Validation engine for card detection results.
    
    Combines histogram matching with mana cost and rarity validation
    using Arena Tracker's proven approach.
    """
    
    def __init__(self):
        """Initialize validation engine."""
        self.logger = logging.getLogger(__name__)
        self.template_matcher = get_template_matcher()
        
        # Arena Tracker's validation parameters
        self.MANA_TOLERANCE = 1  # Allow ±1 mana cost difference
        self.MIN_VALIDATION_CONFIDENCE = 0.4  # Reduced from 0.6 to reduce false negatives
        
        self.logger.info("ValidationEngine initialized")
    
    def validate_card_detection(self, card_match: CardMatch, mana_region: Optional[bytes] = None,
                              rarity_region: Optional[bytes] = None, 
                              expected_mana: Optional[int] = None,
                              expected_rarity: Optional[int] = None) -> ValidationResult:
        """
        Validate a card detection result.
        
        Args:
            card_match: Histogram-based card match
            mana_region: Image region containing mana cost
            rarity_region: Image region containing rarity gem
            expected_mana: Expected mana cost from card database
            expected_rarity: Expected rarity from card database
            
        Returns:
            ValidationResult with validation outcome
        """
        validation_score = 0.0
        detected_mana = None
        detected_rarity = None
        
        # Start with base confidence from histogram matching
        base_confidence = card_match.confidence
        validation_score += base_confidence * 0.6  # 60% weight for histogram
        
        # Validate mana cost if region provided
        if mana_region is not None:
            detected_mana = self.template_matcher.detect_mana_cost(mana_region)
            
            if detected_mana is not None and expected_mana is not None:
                mana_diff = abs(detected_mana - expected_mana)
                
                if mana_diff <= self.MANA_TOLERANCE:
                    # Mana cost matches - boost confidence
                    validation_score += 0.2  # 20% boost for mana validation
                    self.logger.debug(f"Mana cost validated: {detected_mana} (expected: {expected_mana})")
                else:
                    # Mana cost mismatch - reduce confidence
                    validation_score -= 0.3  # 30% penalty for mana mismatch
                    self.logger.debug(f"Mana cost mismatch: {detected_mana} vs {expected_mana}")
            elif detected_mana is not None:
                # Mana detected but no expected value
                validation_score += 0.1  # Small boost for successful detection
        
        # Validate rarity if region provided
        if rarity_region is not None:
            detected_rarity = self.template_matcher.detect_rarity(rarity_region)
            
            if detected_rarity is not None and expected_rarity is not None:
                if detected_rarity == expected_rarity:
                    # Rarity matches - boost confidence
                    validation_score += 0.2  # 20% boost for rarity validation
                    self.logger.debug(f"Rarity validated: {detected_rarity}")
                else:
                    # Rarity mismatch - reduce confidence
                    validation_score -= 0.2  # 20% penalty for rarity mismatch
                    self.logger.debug(f"Rarity mismatch: {detected_rarity} vs {expected_rarity}")
            elif detected_rarity is not None:
                # Rarity detected but no expected value
                validation_score += 0.1  # Small boost for successful detection
        
        # Clamp validation score to [0, 1]
        validation_score = max(0.0, min(1.0, validation_score))
        
        # Determine if validation passes
        is_valid = validation_score >= self.MIN_VALIDATION_CONFIDENCE
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence=validation_score,
            mana_cost=detected_mana,
            rarity=detected_rarity,
            validation_score=validation_score
        )
        
        self.logger.debug(f"Validation result: {result}")
        return result
    
    def validate_card_consistency(self, card_matches: list[CardMatch], 
                                 mana_detections: list[Optional[int]],
                                 rarity_detections: list[Optional[int]]) -> Tuple[bool, float]:
        """
        Validate consistency across multiple card detections.
        
        Args:
            card_matches: List of card matches
            mana_detections: List of detected mana costs
            rarity_detections: List of detected rarities
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        if not card_matches:
            return False, 0.0
        
        consistency_score = 0.0
        total_checks = 0
        
        # Check that all cards have reasonable confidence
        confidence_sum = sum(match.confidence for match in card_matches)
        avg_confidence = confidence_sum / len(card_matches)
        consistency_score += avg_confidence * 0.4  # 40% weight for average confidence
        
        # Check mana cost consistency
        valid_mana_count = sum(1 for mana in mana_detections if mana is not None)
        if valid_mana_count > 0:
            mana_consistency = valid_mana_count / len(mana_detections)
            consistency_score += mana_consistency * 0.3  # 30% weight for mana consistency
        
        # Check rarity consistency
        valid_rarity_count = sum(1 for rarity in rarity_detections if rarity is not None)
        if valid_rarity_count > 0:
            rarity_consistency = valid_rarity_count / len(rarity_detections)
            consistency_score += rarity_consistency * 0.3  # 30% weight for rarity consistency
        
        # Clamp to [0, 1]
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        # Consider consistent if score is above threshold
        is_consistent = consistency_score >= self.MIN_VALIDATION_CONFIDENCE
        
        self.logger.debug(f"Consistency check: {is_consistent} (score: {consistency_score:.3f})")
        return is_consistent, consistency_score
    
    def set_validation_threshold(self, threshold: float):
        """
        Set the minimum validation confidence threshold.
        
        Args:
            threshold: Minimum confidence for validation (0.0 to 1.0)
        """
        self.MIN_VALIDATION_CONFIDENCE = max(0.0, min(1.0, threshold))
        self.logger.info(f"Validation threshold set to: {self.MIN_VALIDATION_CONFIDENCE}")
    
    def get_validation_stats(self) -> dict:
        """
        Get validation engine statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return {
            "mana_tolerance": self.MANA_TOLERANCE,
            "min_confidence": self.MIN_VALIDATION_CONFIDENCE,
            "template_counts": self.template_matcher.get_template_counts()
        }


# Global validation engine instance
_validation_engine = None


def get_validation_engine() -> ValidationEngine:
    """
    Get the global validation engine instance.
    
    Returns:
        ValidationEngine instance
    """
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = ValidationEngine()
    return _validation_engine