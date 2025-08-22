"""
AI v2 Exception Classes

Custom exception hierarchy for AI Engine v2 components.
Provides specific error types for different failure modes in AI analysis pipeline.
"""


class AIEngineError(Exception):
    """Base exception for AI Engine failures"""
    pass


class ModelLoadingError(AIEngineError):
    """Raised when AI models fail to load or initialize"""
    pass


class InferenceTimeout(AIEngineError):
    """Raised when AI inference takes too long"""
    pass


class InvalidInputError(AIEngineError):
    """Raised when input data is invalid for AI processing"""
    pass


class ValidationError(AIEngineError):
    """Raised when data validation fails"""
    pass


class AnalysisError(AIEngineError):
    """Raised when AI analysis fails"""
    pass


class ConfidenceError(AIEngineError):
    """Raised when confidence levels are insufficient"""
    pass


# Export all exception classes
__all__ = [
    'AIEngineError',
    'ModelLoadingError', 
    'InferenceTimeout',
    'InvalidInputError',
    'ValidationError',
    'AnalysisError',
    'ConfidenceError'
]