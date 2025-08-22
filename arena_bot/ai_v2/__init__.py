"""
AI v2 Package

Advanced AI analysis system for Arena drafting with comprehensive error handling
and data model support.
"""

# Import exception classes for easy access
from .errors import (
    AIEngineError, ModelLoadingError, InferenceTimeout, InvalidInputError,
    ValidationError, AnalysisError, ConfidenceError
)

# Import data models
from .data_models import (
    CardInstance, CardInfo, CardOption, DraftChoice, AIDecision,
    DeckState, EvaluationScores, StrategicContext
)

__all__ = [
    # Exceptions
    'AIEngineError', 'ModelLoadingError', 'InferenceTimeout', 'InvalidInputError',
    'ValidationError', 'AnalysisError', 'ConfidenceError',
    
    # Data Models
    'CardInstance', 'CardInfo', 'CardOption', 'DraftChoice', 'AIDecision',
    'DeckState', 'EvaluationScores', 'StrategicContext'
]