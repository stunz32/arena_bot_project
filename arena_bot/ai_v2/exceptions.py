"""
AI Helper v2 - Custom Exception Handling
Comprehensive exception hierarchy for the Grandmaster AI Coach system

This module defines all custom exceptions used throughout the AI Helper v2 system,
providing structured error handling with context preservation and recovery guidance.
"""

import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for structured error handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification and handling"""
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    DATA = "data"
    RESOURCE = "resource"
    AI_MODEL = "ai_model"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_INPUT = "user_input"
    SYSTEM = "system"

class AIHelperException(Exception):
    """Base exception class for all AI Helper v2 exceptions"""
    
    def __init__(
        self, 
        message: str,
        error_code: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Dict[str, Any] = None,
        recovery_suggestions: List[str] = None,
        correlation_id: str = None
    ):
        super().__init__(message)
        
        self.message = message
        self.severity = severity
        self.category = category
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now().isoformat()
        self.traceback_info = traceback.format_exc()
        
        # Log the error
        self._log_error()
        
    def _generate_error_code(self) -> str:
        """Generate a unique error code"""
        return f"AIH-{self.category.value.upper()}-{self.correlation_id}"
        
    def _log_error(self):
        """Log the error with appropriate level based on severity"""
        log_data = {
            "error_code": self.error_code,
            "correlation_id": self.correlation_id,
            "error_message": self.message,  # Renamed to avoid LogRecord conflict
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions
        }
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error {self.error_code}: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error {self.error_code}: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error {self.error_code}: {self.message}", extra=log_data)
        else:
            logger.info(f"Low severity error {self.error_code}: {self.message}", extra=log_data)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "exception_type": self.__class__.__name__
        }

# === Dependency-Related Exceptions ===

class DependencyError(AIHelperException):
    """Base class for dependency-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DEPENDENCY)
        super().__init__(message, **kwargs)

class MissingDependencyError(DependencyError):
    """Critical dependency is missing"""
    
    def __init__(self, package_name: str, **kwargs):
        message = f"Critical dependency '{package_name}' is missing"
        context = kwargs.get('context', {})
        context['package_name'] = package_name
        
        recovery_suggestions = [
            f"Install {package_name} using: pip install {package_name}",
            "Check if you're in the correct virtual environment",
            "Run dependency validation script: python scripts/validate_dependencies.py"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.HIGH
        })
        
        super().__init__(message, **kwargs)

class DependencyVersionError(DependencyError):
    """Dependency version mismatch"""
    
    def __init__(self, package_name: str, required_version: str, current_version: str, **kwargs):
        message = f"Version mismatch for '{package_name}': required {required_version}, found {current_version}"
        context = kwargs.get('context', {})
        context.update({
            'package_name': package_name,
            'required_version': required_version,
            'current_version': current_version
        })
        
        recovery_suggestions = [
            f"Update {package_name}: pip install {package_name}=={required_version}",
            "Check requirements.txt for correct versions",
            "Consider using a fresh virtual environment"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

# === Configuration-Related Exceptions ===

class ConfigurationError(AIHelperException):
    """Base class for configuration-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        super().__init__(message, **kwargs)

class InvalidConfigurationError(ConfigurationError):
    """Configuration is invalid or corrupted"""
    
    def __init__(self, config_key: str, config_value: Any = None, **kwargs):
        message = f"Invalid configuration for '{config_key}'"
        if config_value is not None:
            message += f": {config_value}"
            
        context = kwargs.get('context', {})
        context.update({
            'config_key': config_key,
            'config_value': config_value
        })
        
        recovery_suggestions = [
            f"Check configuration value for '{config_key}'",
            "Validate configuration schema",
            "Reset to default configuration if needed"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

class ConfigurationLoadError(ConfigurationError):
    """Failed to load configuration"""
    
    def __init__(self, config_path: str, **kwargs):
        message = f"Failed to load configuration from '{config_path}'"
        context = kwargs.get('context', {})
        context['config_path'] = config_path
        
        recovery_suggestions = [
            "Check if configuration file exists",
            "Verify file permissions",
            "Validate configuration file syntax"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.HIGH
        })
        
        super().__init__(message, **kwargs)

class AIHelperConfigurationError(ConfigurationError):
    """Legacy configuration error alias for backward compatibility"""
    pass

# === Data-Related Exceptions ===

class DataError(AIHelperException):
    """Base class for data-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA)
        super().__init__(message, **kwargs)

class DataValidationError(DataError):
    """Data validation failed"""
    
    def __init__(self, field_name: str, expected_type: str, actual_value: Any, **kwargs):
        message = f"Data validation failed for '{field_name}': expected {expected_type}, got {type(actual_value).__name__}"
        context = kwargs.get('context', {})
        context.update({
            'field_name': field_name,
            'expected_type': expected_type,
            'actual_value': str(actual_value),
            'actual_type': type(actual_value).__name__
        })
        
        recovery_suggestions = [
            f"Check data format for field '{field_name}'",
            "Validate input data schema",
            "Use data transformation utilities"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

class DataCorruptionError(DataError):
    """Data is corrupted or inconsistent"""
    
    def __init__(self, data_source: str, **kwargs):
        message = f"Data corruption detected in '{data_source}'"
        context = kwargs.get('context', {})
        context['data_source'] = data_source
        
        recovery_suggestions = [
            "Restore from backup if available",
            "Validate data integrity",
            "Clear corrupted cache/data"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.HIGH
        })
        
        super().__init__(message, **kwargs)

class AIHelperValidationError(DataError):
    """Legacy validation error alias for backward compatibility"""
    pass

class AIHelperSecurityError(AIHelperException):
    """Legacy security error alias for backward compatibility"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

# === Resource-Related Exceptions ===

class ResourceError(AIHelperException):
    """Base class for resource-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RESOURCE)
        super().__init__(message, **kwargs)

class ResourceExhaustionError(ResourceError):
    """System resources are exhausted"""
    
    def __init__(self, resource_type: str, current_usage: float, limit: float, **kwargs):
        message = f"Resource exhaustion: {resource_type} usage {current_usage:.1f} exceeds limit {limit:.1f}"
        context = kwargs.get('context', {})
        context.update({
            'resource_type': resource_type,
            'current_usage': current_usage,
            'limit': limit,
            'usage_percentage': (current_usage / limit) * 100 if limit > 0 else 0
        })
        
        recovery_suggestions = [
            f"Free up {resource_type} resources",
            "Restart the application to clear memory leaks",
            "Increase resource limits if possible"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.HIGH
        })
        
        super().__init__(message, **kwargs)

class ResourceUnavailableError(ResourceError):
    """Required resource is unavailable"""
    
    def __init__(self, resource_name: str, **kwargs):
        message = f"Required resource '{resource_name}' is unavailable"
        context = kwargs.get('context', {})
        context['resource_name'] = resource_name
        
        recovery_suggestions = [
            f"Check if {resource_name} is accessible",
            "Verify permissions and connectivity",
            "Use alternative resource if available"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

# === AI Model-Related Exceptions ===

class AIModelError(AIHelperException):
    """Base class for AI model-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AI_MODEL)
        super().__init__(message, **kwargs)

class ModelLoadError(AIModelError):
    """Failed to load AI model"""
    
    def __init__(self, model_name: str, model_path: str = None, **kwargs):
        message = f"Failed to load AI model '{model_name}'"
        if model_path:
            message += f" from '{model_path}'"
            
        context = kwargs.get('context', {})
        context.update({
            'model_name': model_name,
            'model_path': model_path
        })
        
        recovery_suggestions = [
            "Check if model file exists and is accessible",
            "Verify model file integrity",
            "Use fallback heuristic evaluation"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.HIGH
        })
        
        super().__init__(message, **kwargs)

class ModelPredictionError(AIModelError):
    """AI model prediction failed"""
    
    def __init__(self, model_name: str, input_data: Dict[str, Any] = None, **kwargs):
        message = f"Prediction failed for model '{model_name}'"
        context = kwargs.get('context', {})
        context.update({
            'model_name': model_name,
            'input_data_keys': list(input_data.keys()) if input_data else None
        })
        
        recovery_suggestions = [
            "Validate input data format",
            "Check model compatibility",
            "Use fallback prediction method"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

# === Integration-Related Exceptions ===

class IntegrationError(AIHelperException):
    """Base class for integration-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.INTEGRATION)
        super().__init__(message, **kwargs)

class ComponentCommunicationError(IntegrationError):
    """Communication between components failed"""
    
    def __init__(self, source_component: str, target_component: str, **kwargs):
        message = f"Communication failed between '{source_component}' and '{target_component}'"
        context = kwargs.get('context', {})
        context.update({
            'source_component': source_component,
            'target_component': target_component
        })
        
        recovery_suggestions = [
            "Check component initialization order",
            "Verify component interfaces",
            "Restart affected components"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

# === Performance-Related Exceptions ===

class PerformanceError(AIHelperException):
    """Base class for performance-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PERFORMANCE)
        super().__init__(message, **kwargs)

class PerformanceThresholdExceeded(PerformanceError):
    """Performance threshold exceeded"""
    
    def __init__(self, operation: str, duration: float, threshold: float, **kwargs):
        message = f"Performance threshold exceeded for '{operation}': {duration:.2f}s > {threshold:.2f}s"
        context = kwargs.get('context', {})
        context.update({
            'operation': operation,
            'duration': duration,
            'threshold': threshold,
            'overhead_percentage': ((duration - threshold) / threshold) * 100
        })
        
        recovery_suggestions = [
            "Optimize the operation implementation",
            "Increase performance threshold if appropriate",
            "Enable performance monitoring for debugging"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.LOW
        })
        
        super().__init__(message, **kwargs)

# === Security-Related Exceptions ===

class SecurityError(AIHelperException):
    """Base class for security-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

class UnauthorizedAccessError(SecurityError):
    """Unauthorized access attempt"""
    
    def __init__(self, resource: str, **kwargs):
        message = f"Unauthorized access attempt to '{resource}'"
        context = kwargs.get('context', {})
        context['resource'] = resource
        
        recovery_suggestions = [
            "Verify user permissions",
            "Check authentication status",
            "Review security policies"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.CRITICAL
        })
        
        super().__init__(message, **kwargs)

# === Circuit Breaker Exceptions ===

class CircuitBreakerError(AIHelperException):
    """Circuit breaker related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        super().__init__(message, **kwargs)

class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open, blocking requests"""
    
    def __init__(self, component_name: str, failure_count: int, **kwargs):
        message = f"Circuit breaker open for '{component_name}' after {failure_count} failures"
        context = kwargs.get('context', {})
        context.update({
            'component_name': component_name,
            'failure_count': failure_count
        })
        
        recovery_suggestions = [
            "Wait for circuit breaker to reset",
            "Check component health",
            "Use alternative implementation if available"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

# === Exception Handler Utility ===

class ExceptionHandler:
    """Utility class for consistent exception handling"""
    
    @staticmethod
    def handle_exception(
        exception: Exception,
        operation: str,
        context: Dict[str, Any] = None,
        reraise: bool = True
    ) -> Optional[AIHelperException]:
        """Handle an exception consistently"""
        
        if isinstance(exception, AIHelperException):
            # Already handled
            if reraise:
                raise
            return exception
            
        # Convert to AIHelperException
        wrapped_exception = AIHelperException(
            message=f"Unhandled exception in {operation}: {str(exception)}",
            context=context or {},
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                f"Check logs for details about {operation}",
                "Verify input parameters",
                "Contact support if issue persists"
            ]
        )
        
        if reraise:
            raise wrapped_exception
        return wrapped_exception
        
    @staticmethod
    def create_error_response(exception: AIHelperException) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "success": False,
            "error": exception.to_dict(),
            "correlation_id": exception.correlation_id
        }

# === UI-Related Exceptions ===

class UIError(AIHelperException):
    """Base class for UI-related errors"""
    
    def __init__(self, message: str, **kwargs):
        if 'category' not in kwargs:
            kwargs['category'] = ErrorCategory.USER_INPUT
        super().__init__(message, **kwargs)

class AIHelperUIError(UIError):
    """UI component error for visual overlay and hover detection systems"""
    
    def __init__(self, component: str, operation: str, **kwargs):
        message = f"UI component '{component}' failed during '{operation}'"
        context = kwargs.get('context', {})
        context.update({
            'component': component,
            'operation': operation
        })
        
        recovery_suggestions = [
            "Check if display drivers are up to date",
            "Verify Windows API compatibility", 
            "Try restarting the visual overlay system",
            "Check for conflicting overlay applications"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM
        })
        
        super().__init__(message, **kwargs)

class AIHelperPerformanceError(AIHelperException):
    """Performance-related error for AI operations exceeding thresholds"""
    
    def __init__(self, operation: str, threshold: float, actual: float, unit: str = "ms", **kwargs):
        message = f"Performance threshold exceeded for '{operation}': {actual}{unit} > {threshold}{unit}"
        context = kwargs.get('context', {})
        context.update({
            'operation': operation,
            'threshold': threshold,
            'actual': actual,
            'unit': unit,
            'performance_ratio': actual / threshold if threshold > 0 else float('inf')
        })
        
        recovery_suggestions = [
            "Check system resource availability",
            "Consider reducing operation complexity",
            "Verify no background tasks are consuming resources",
            "Try restarting the AI components"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.HIGH,
            'category': ErrorCategory.PERFORMANCE
        })
        
        super().__init__(message, **kwargs)

class AIHelperNotSupportedError(AIHelperException):
    """Feature not supported error"""
    
    def __init__(self, feature: str, **kwargs):
        message = f"Feature '{feature}' is not supported in this environment"
        context = kwargs.get('context', {})
        context['feature'] = feature
        
        kwargs.update({
            'context': context,
            'severity': ErrorSeverity.MEDIUM,
            'category': ErrorCategory.SYSTEM
        })
        
        super().__init__(message, **kwargs)

class AIHelperPlatformError(AIHelperException):
    """Platform-specific error for Windows/Linux compatibility issues"""
    
    def __init__(self, platform: str, feature: str, **kwargs):
        message = f"Platform '{platform}' does not support feature '{feature}'"
        context = kwargs.get('context', {})
        context.update({
            'platform': platform,
            'feature': feature
        })
        
        recovery_suggestions = [
            f"Check if feature '{feature}' is available on {platform}",
            "Update platform-specific dependencies",
            "Use alternative implementation for this platform",
            "Check platform documentation for compatibility"
        ]
        
        kwargs.update({
            'context': context,
            'recovery_suggestions': recovery_suggestions,
            'severity': ErrorSeverity.MEDIUM,
            'category': ErrorCategory.SYSTEM
        })
        
        super().__init__(message, **kwargs)

# === Legacy Compatibility Aliases ===

class AIHelperError(AIHelperException):
    """
    Legacy alias for AIHelperException - maintains backward compatibility.
    
    This class provides backward compatibility for code that references the legacy
    AIHelperError class name while leveraging the enhanced AIHelperException functionality.
    """
    pass

# Export list for public API
__all__ = [
    # Base exceptions
    'AIHelperException', 'AIHelperError', 'DependencyError', 
    # Configuration exceptions
    'ConfigurationError', 'InvalidConfigurationError', 'ConfigurationLoadError',
    'AIHelperConfigurationError', 'AIHelperValidationError', 'AIHelperSecurityError',
    # Data exceptions  
    'DataError', 'DataValidationError', 'DataCorruptionError',
    # AI exceptions
    'AIProcessingError', 'AIResponseError', 'AIModelError',
    # Network exceptions
    'NetworkError', 'APIError', 'AuthenticationError',
    # Resource exceptions
    'ResourceError', 'ResourceExhaustionError', 'PerformanceError',
    # UI exceptions
    'UIError', 'AIHelperUIError', 'AIHelperPerformanceError', 'AIHelperPlatformError', 'AIHelperNotSupportedError',
    # Utilities
    'ErrorHandler', 'ErrorSeverity', 'ErrorCategory'
]