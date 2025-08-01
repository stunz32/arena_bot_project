"""
Configuration Validation for S-Tier Logging System.

This module provides comprehensive validation for configuration models
with custom validation rules, security checks, and detailed error reporting
for enterprise-grade configuration management.

Features:
- Pydantic-based model validation with custom rules
- Security-focused validation for enterprise compliance
- Performance validation for resource constraints
- Cross-reference validation between configuration components
- Detailed error reporting with suggestions
- Validation rule composition and extension
"""

import re
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PydanticValidationError = Exception
    PYDANTIC_AVAILABLE = False


class ValidationSeverity(str, Enum):
    """Validation message severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationContext(str, Enum):
    """Validation context types."""
    STRUCTURE = "structure"           # Configuration structure validation
    SECURITY = "security"             # Security policy validation
    PERFORMANCE = "performance"       # Performance constraint validation
    COMPATIBILITY = "compatibility"   # Backward compatibility validation
    CONSISTENCY = "consistency"       # Cross-reference consistency validation


@dataclass
class ValidationMessage:
    """Validation message with context and suggestions."""
    
    severity: ValidationSeverity
    context: ValidationContext
    message: str
    field_path: str = ""
    field_value: Optional[Any] = None
    suggestion: Optional[str] = None
    rule_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "context": self.context.value,
            "message": self.message,
            "field_path": self.field_path,
            "field_value": self.field_value,
            "suggestion": self.suggestion,
            "rule_name": self.rule_name
        }
    
    def __str__(self) -> str:
        """String representation of validation message."""
        parts = [f"[{self.severity.value.upper()}]"]
        
        if self.field_path:
            parts.append(f"{self.field_path}:")
        
        parts.append(self.message)
        
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        
        return " ".join(parts)


class ValidationError(Exception):
    """Configuration validation error with detailed messages."""
    
    def __init__(self, messages: List[ValidationMessage]):
        self.messages = messages
        self.error_count = len([m for m in messages if m.severity == ValidationSeverity.ERROR])
        self.critical_count = len([m for m in messages if m.severity == ValidationSeverity.CRITICAL])
        
        # Create summary message
        summary = f"Configuration validation failed: {self.error_count} errors"
        if self.critical_count > 0:
            summary += f", {self.critical_count} critical issues"
        
        super().__init__(summary)
    
    def get_messages_by_severity(self, severity: ValidationSeverity) -> List[ValidationMessage]:
        """Get messages filtered by severity."""
        return [m for m in self.messages if m.severity == severity]
    
    def get_messages_by_context(self, context: ValidationContext) -> List[ValidationMessage]:
        """Get messages filtered by context."""
        return [m for m in self.messages if m.context == context]
    
    def has_errors(self) -> bool:
        """Check if validation has any errors or critical issues."""
        return self.error_count > 0 or self.critical_count > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "error_count": self.error_count,
            "critical_count": self.critical_count,
            "messages": [m.to_dict() for m in self.messages]
        }


class ValidationRule(ABC):
    """
    Abstract base class for validation rules.
    
    Validation rules can be composed and applied to configuration
    objects to enforce business logic, security policies, and
    performance constraints.
    """
    
    def __init__(self, 
                 name: str,
                 severity: ValidationSeverity = ValidationSeverity.ERROR,
                 context: ValidationContext = ValidationContext.STRUCTURE):
        self.name = name
        self.severity = severity
        self.context = context
    
    @abstractmethod
    def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
        """
        Validate configuration against this rule.
        
        Args:
            config: Configuration dictionary to validate
            field_path: Current field path for error reporting
            
        Returns:
            List of validation messages
        """
        pass
    
    def create_message(self, 
                      message: str,
                      field_path: str = "",
                      field_value: Optional[Any] = None,
                      suggestion: Optional[str] = None) -> ValidationMessage:
        """Create validation message for this rule."""
        return ValidationMessage(
            severity=self.severity,
            context=self.context,
            message=message,
            field_path=field_path,
            field_value=field_value,
            suggestion=suggestion,
            rule_name=self.name
        )


class PathExistsRule(ValidationRule):
    """Validation rule to check if file paths exist."""
    
    def __init__(self, 
                 path_fields: List[str],
                 create_if_missing: bool = False,
                 check_permissions: bool = True):
        super().__init__("path_exists", ValidationSeverity.ERROR, ValidationContext.STRUCTURE)
        self.path_fields = path_fields
        self.create_if_missing = create_if_missing
        self.check_permissions = check_permissions
    
    def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
        """Validate that specified paths exist and are accessible."""
        messages = []
        
        for path_field in self.path_fields:
            value = self._get_nested_value(config, path_field.split('.'))
            
            if value is None:
                continue
            
            if isinstance(value, (str, Path)):
                path_obj = Path(value)
                full_field_path = f"{field_path}.{path_field}" if field_path else path_field
                
                if not path_obj.exists():
                    if self.create_if_missing:
                        try:
                            if path_field.endswith('_path') or 'file' in path_field:
                                # Create parent directory for files
                                path_obj.parent.mkdir(parents=True, exist_ok=True)
                            else:
                                # Create directory
                                path_obj.mkdir(parents=True, exist_ok=True)
                        except Exception as e:
                            messages.append(self.create_message(
                                f"Cannot create path '{path_obj}': {e}",
                                full_field_path,
                                str(path_obj),
                                f"Ensure parent directory exists and has write permissions"
                            ))
                    else:
                        messages.append(self.create_message(
                            f"Path does not exist: {path_obj}",
                            full_field_path,
                            str(path_obj),
                            f"Create the path or set create_if_missing=True"
                        ))
                
                elif self.check_permissions:
                    # Check read/write permissions
                    if path_obj.is_file():
                        if not os.access(path_obj, os.R_OK):
                            messages.append(self.create_message(
                                f"File is not readable: {path_obj}",
                                full_field_path,
                                str(path_obj),
                                "Check file permissions"
                            ))
                        if 'log' in path_field and not os.access(path_obj, os.W_OK):
                            messages.append(self.create_message(
                                f"Log file is not writable: {path_obj}",
                                full_field_path,
                                str(path_obj),
                                "Check file permissions"
                            ))
                    elif path_obj.is_dir():
                        if not os.access(path_obj, os.R_OK | os.X_OK):
                            messages.append(self.create_message(
                                f"Directory is not accessible: {path_obj}",
                                full_field_path,
                                str(path_obj),
                                "Check directory permissions"
                            ))
        
        return messages
    
    def _get_nested_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """Get nested value from dictionary."""
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current


class SecurityRule(ValidationRule):
    """Security-focused validation rules."""
    
    def __init__(self):
        super().__init__("security_policy", ValidationSeverity.CRITICAL, ValidationContext.SECURITY)
    
    def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
        """Validate security configuration."""
        messages = []
        
        environment = config.get('environment', 'development')
        security = config.get('security', {})
        
        # Production environment security requirements
        if environment == 'production':
            messages.extend(self._validate_production_security(security, field_path))
        
        # General security validation
        messages.extend(self._validate_encryption_config(security.get('encryption', {}), field_path))
        messages.extend(self._validate_audit_config(security, field_path))
        messages.extend(self._validate_retention_policy(security.get('retention', {}), field_path))
        
        return messages
    
    def _validate_production_security(self, security: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate production environment security requirements."""
        messages = []
        security_path = f"{field_path}.security" if field_path else "security"
        
        if not security.get('enable_audit_trail', False):
            messages.append(self.create_message(
                "Production environment requires audit trail",
                f"{security_path}.enable_audit_trail",
                False,
                "Set enable_audit_trail=true for production"
            ))
        
        encryption = security.get('encryption', {})
        if not encryption.get('enabled', False):
            messages.append(self.create_message(
                "Production environment requires encryption",
                f"{security_path}.encryption.enabled",
                False,
                "Enable encryption for production environment"
            ))
        
        if not security.get('enable_pii_detection', False):
            messages.append(self.create_message(
                "Production environment should enable PII detection",
                f"{security_path}.enable_pii_detection",
                False,
                "Enable PII detection to prevent data leaks"
            ))
        
        return messages
    
    def _validate_encryption_config(self, encryption: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate encryption configuration."""
        messages = []
        
        if not encryption.get('enabled', False):
            return messages
        
        encryption_path = f"{field_path}.security.encryption" if field_path else "security.encryption"
        
        # Check encryption key configuration
        key_file_path = encryption.get('key_file_path')
        if not key_file_path:
            messages.append(self.create_message(
                "Encryption enabled but no key file path specified",
                f"{encryption_path}.key_file_path",
                None,
                "Specify path to encryption key file"
            ))
        
        # Check key rotation policy
        key_rotation_days = encryption.get('key_rotation_days', 0)
        if key_rotation_days > 365:
            messages.append(self.create_message(
                "Key rotation interval too long for security best practices",
                f"{encryption_path}.key_rotation_days",
                key_rotation_days,
                "Set key rotation to 90 days or less"
            ))
        
        return messages
    
    def _validate_audit_config(self, security: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate audit configuration."""
        messages = []
        
        if not security.get('enable_audit_trail', False):
            return messages
        
        security_path = f"{field_path}.security" if field_path else "security"
        
        audit_log_path = security.get('audit_log_path')
        if not audit_log_path:
            messages.append(self.create_message(
                "Audit trail enabled but no audit log path specified",
                f"{security_path}.audit_log_path",
                None,
                "Specify path for audit logs"
            ))
        
        return messages
    
    def _validate_retention_policy(self, retention: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate data retention policy.""" 
        messages = []
        
        if not retention:
            return messages
        
        retention_path = f"{field_path}.security.retention" if field_path else "security.retention"
        
        # Check retention periods are logical
        archive_days = retention.get('archive_after_days', 30)
        delete_days = retention.get('delete_after_days', 90)
        
        if archive_days >= delete_days:
            messages.append(self.create_message(
                "Archive period must be shorter than deletion period",
                f"{retention_path}.archive_after_days",
                archive_days,
                f"Set archive_after_days to less than {delete_days}"
            ))
        
        # Check maximum retention limits
        max_age_days = retention.get('max_age_days', 90)
        if max_age_days > 2555:  # ~7 years
            messages.append(self.create_message(
                "Retention period exceeds recommended maximum",
                f"{retention_path}.max_age_days",
                max_age_days,
                "Consider shorter retention for compliance and storage costs"
            ))
        
        return messages


class PerformanceRule(ValidationRule):
    """Performance constraint validation rules."""
    
    def __init__(self):
        super().__init__("performance_constraints", ValidationSeverity.WARNING, ValidationContext.PERFORMANCE)
    
    def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
        """Validate performance configuration."""
        messages = []
        
        performance = config.get('performance', {})
        environment = config.get('environment', 'development')
        
        # Environment-specific performance validation
        if environment == 'production':
            messages.extend(self._validate_production_performance(performance, field_path))
        
        # General performance validation
        messages.extend(self._validate_resource_limits(performance, field_path))
        messages.extend(self._validate_async_config(performance, field_path))
        
        return messages
    
    def _validate_production_performance(self, performance: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate production performance requirements."""
        messages = []
        performance_path = f"{field_path}.performance" if field_path else "performance"
        
        if not performance.get('enable_async_processing', False):
            messages.append(self.create_message(
                "Production environment should use async processing",
                f"{performance_path}.enable_async_processing",
                False,
                "Enable async processing for better performance"
            ))
        
        worker_threads = performance.get('worker_threads', 1)
        if worker_threads < 2:
            messages.append(self.create_message(
                "Production environment should use multiple worker threads",
                f"{performance_path}.worker_threads",
                worker_threads,
                "Set worker_threads to at least 4 for production"
            ))
        
        return messages
    
    def _validate_resource_limits(self, performance: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate resource limit configuration."""
        messages = []
        performance_path = f"{field_path}.performance" if field_path else "performance"
        
        # Memory limits
        max_memory_mb = performance.get('max_memory_mb', 512)
        if max_memory_mb > 8192:  # 8GB
            messages.append(self.create_message(
                "Memory limit very high, may impact system performance",
                f"{performance_path}.max_memory_mb",
                max_memory_mb,
                "Consider reducing memory limit or monitoring usage closely"
            ))
        
        # Queue size validation
        async_queue_size = performance.get('async_queue_size', 1000)
        if async_queue_size > 100000:
            messages.append(self.create_message(
                "Async queue size very large, may consume excessive memory",
                f"{performance_path}.async_queue_size",
                async_queue_size,
                "Consider reducing queue size or implementing backpressure"
            ))
        
        return messages
    
    def _validate_async_config(self, performance: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate async processing configuration."""
        messages = []
        
        if not performance.get('enable_async_processing', False):
            return messages
        
        performance_path = f"{field_path}.performance" if field_path else "performance"
        
        # Check worker thread configuration
        worker_threads = performance.get('worker_threads', 1)
        async_queue_size = performance.get('async_queue_size', 1000)
        
        # Warn if queue size per thread is too small
        queue_per_thread = async_queue_size / worker_threads
        if queue_per_thread < 100:
            messages.append(self.create_message(
                "Queue size per worker thread may be too small",
                f"{performance_path}.async_queue_size",
                async_queue_size,
                f"Consider increasing queue size or reducing worker threads"
            ))
        
        return messages


class ConsistencyRule(ValidationRule):
    """Cross-reference consistency validation rules."""
    
    def __init__(self):
        super().__init__("consistency_check", ValidationSeverity.ERROR, ValidationContext.CONSISTENCY)
    
    def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
        """Validate cross-reference consistency."""
        messages = []
        
        # Validate logger references
        messages.extend(self._validate_logger_references(config, field_path))
        
        # Validate sink references
        messages.extend(self._validate_sink_references(config, field_path))
        
        # Validate filter references
        messages.extend(self._validate_filter_references(config, field_path))
        
        return messages
    
    def _validate_logger_references(self, config: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate logger handler and sink references."""
        messages = []
        
        loggers = config.get('loggers', {})
        handlers = config.get('handlers', {})
        sinks = config.get('sinks', {})
        
        for logger_name, logger_config in loggers.items():
            logger_path = f"{field_path}.loggers.{logger_name}" if field_path else f"loggers.{logger_name}"
            
            # Check handler references
            for handler_name in logger_config.get('handlers', []):
                if handler_name not in handlers:
                    messages.append(self.create_message(
                        f"Logger references unknown handler '{handler_name}'",
                        f"{logger_path}.handlers",
                        handler_name,
                        f"Add handler '{handler_name}' to handlers configuration"
                    ))
            
            # Check sink references
            for sink_name in logger_config.get('sinks', []):
                if sink_name not in sinks:
                    messages.append(self.create_message(
                        f"Logger references unknown sink '{sink_name}'",
                        f"{logger_path}.sinks",
                        sink_name,
                        f"Add sink '{sink_name}' to sinks configuration"
                    ))
        
        return messages
    
    def _validate_sink_references(self, config: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate sink filter references."""
        messages = []
        
        sinks = config.get('sinks', {})
        filters = config.get('filters', {})
        
        for sink_name, sink_config in sinks.items():
            sink_path = f"{field_path}.sinks.{sink_name}" if field_path else f"sinks.{sink_name}"
            
            # Check filter references
            for filter_name in sink_config.get('filters', []):
                if filter_name not in filters:
                    messages.append(self.create_message(
                        f"Sink references unknown filter '{filter_name}'",
                        f"{sink_path}.filters",
                        filter_name,
                        f"Add filter '{filter_name}' to filters configuration"
                    ))
        
        return messages
    
    def _validate_filter_references(self, config: Dict[str, Any], field_path: str) -> List[ValidationMessage]:
        """Validate filter configuration consistency."""
        messages = []
        
        filters = config.get('filters', {})
        
        for filter_name, filter_config in filters.items():
            filter_path = f"{field_path}.filters.{filter_name}" if field_path else f"filters.{filter_name}"
            
            # Validate filter type and configuration
            filter_type = filter_config.get('type')
            filter_config_data = filter_config.get('config', {})
            
            if filter_type == 'level':
                min_level = filter_config_data.get('min_level')
                if min_level and min_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    messages.append(self.create_message(
                        f"Invalid log level '{min_level}' in level filter",
                        f"{filter_path}.config.min_level",
                        min_level,
                        "Use one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
                    ))
        
        return messages


class ConfigurationValidator:
    """
    Main configuration validator with rule composition.
    
    Combines multiple validation rules and provides comprehensive
    configuration validation with detailed error reporting.
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._default_rules_added = False
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name."""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < initial_count
    
    def add_default_rules(self) -> None:
        """Add default validation rules."""
        if self._default_rules_added:
            return
        
        # Path validation for common file paths
        path_fields = [
            'config_file_path',
            'security.audit_log_path',
            'security.encryption.key_file_path'
        ]
        
        # Add handlers and sinks with filename fields
        self.add_rule(PathExistsRule(path_fields, create_if_missing=True))
        self.add_rule(SecurityRule())
        self.add_rule(PerformanceRule())
        self.add_rule(ConsistencyRule())
        
        self._default_rules_added = True
    
    def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
        """
        Validate configuration using all rules.
        
        Args:
            config: Configuration dictionary to validate
            field_path: Current field path for nested validation
            
        Returns:
            List of validation messages
        """
        if not self._default_rules_added:
            self.add_default_rules()
        
        all_messages = []
        
        # Apply all validation rules
        for rule in self.rules:
            try:
                messages = rule.validate(config, field_path)
                all_messages.extend(messages)
            except Exception as e:
                # Create error message for failed validation rule
                error_message = ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    context=ValidationContext.STRUCTURE,
                    message=f"Validation rule '{rule.name}' failed: {e}",
                    field_path=field_path,
                    rule_name=rule.name
                )
                all_messages.append(error_message)
        
        return all_messages
    
    def validate_and_raise(self, config: Dict[str, Any], field_path: str = "") -> None:
        """
        Validate configuration and raise ValidationError if there are errors.
        
        Args:
            config: Configuration dictionary to validate
            field_path: Current field path for nested validation
            
        Raises:
            ValidationError: If validation fails with errors or critical issues
        """
        messages = self.validate(config, field_path)
        
        # Check for errors or critical issues
        has_errors = any(m.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                        for m in messages)
        
        if has_errors:
            raise ValidationError(messages)


class SecurityValidator:
    """
    Specialized validator for security configurations.
    
    Provides focused security validation with compliance
    framework support and detailed security recommendations.
    """
    
    def __init__(self, compliance_mode: str = "none"):
        self.compliance_mode = compliance_mode
        self.validator = ConfigurationValidator()
        self._add_security_rules()
    
    def _add_security_rules(self) -> None:
        """Add security-specific validation rules."""
        # Base security rule
        self.validator.add_rule(SecurityRule())
        
        # Compliance-specific rules
        if self.compliance_mode == "gdpr":
            self.validator.add_rule(self._create_gdpr_rule())
        elif self.compliance_mode == "hipaa":
            self.validator.add_rule(self._create_hipaa_rule())
        elif self.compliance_mode == "pci_dss":
            self.validator.add_rule(self._create_pci_dss_rule())
    
    def _create_gdpr_rule(self) -> ValidationRule:
        """Create GDPR compliance validation rule."""
        class GDPRRule(ValidationRule):
            def __init__(self):
                super().__init__("gdpr_compliance", ValidationSeverity.CRITICAL, ValidationContext.SECURITY)
            
            def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
                messages = []
                security = config.get('security', {})
                
                if not security.get('enable_pii_detection', False):
                    messages.append(self.create_message(
                        "GDPR requires PII detection and redaction",
                        "security.enable_pii_detection",
                        False,
                        "Enable PII detection for GDPR compliance"
                    ))
                
                retention = security.get('retention', {})
                if retention.get('max_age_days', 0) > 365:
                    messages.append(self.create_message(
                        "GDPR recommends data retention under 1 year without explicit consent",
                        "security.retention.max_age_days",
                        retention.get('max_age_days'),
                        "Consider shorter retention period for GDPR compliance"
                    ))
                
                return messages
        
        return GDPRRule()
    
    def _create_hipaa_rule(self) -> ValidationRule:
        """Create HIPAA compliance validation rule."""
        class HIPAARule(ValidationRule):
            def __init__(self):
                super().__init__("hipaa_compliance", ValidationSeverity.CRITICAL, ValidationContext.SECURITY)
            
            def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
                messages = []
                security = config.get('security', {})
                
                encryption = security.get('encryption', {})
                if not encryption.get('enabled', False):
                    messages.append(self.create_message(
                        "HIPAA requires encryption of PHI data",
                        "security.encryption.enabled",
                        False,
                        "Enable encryption for HIPAA compliance"
                    ))
                
                if not security.get('enable_audit_trail', False):
                    messages.append(self.create_message(
                        "HIPAA requires comprehensive audit trails",
                        "security.enable_audit_trail",
                        False,
                        "Enable audit trail for HIPAA compliance"
                    ))
                
                return messages
        
        return HIPAARule()
    
    def _create_pci_dss_rule(self) -> ValidationRule:
        """Create PCI DSS compliance validation rule."""
        class PCIDSSRule(ValidationRule):
            def __init__(self):
                super().__init__("pci_dss_compliance", ValidationSeverity.CRITICAL, ValidationContext.SECURITY)
            
            def validate(self, config: Dict[str, Any], field_path: str = "") -> List[ValidationMessage]:
                messages = []
                security = config.get('security', {})
                
                if not security.get('redact_credentials', False):
                    messages.append(self.create_message(
                        "PCI DSS requires credential redaction",
                        "security.redact_credentials",
                        False,
                        "Enable credential redaction for PCI DSS compliance"
                    ))
                
                retention = security.get('retention', {})
                if retention.get('max_age_days', 0) > 365:
                    messages.append(self.create_message(
                        "PCI DSS requires secure disposal of cardholder data after business need expires",
                        "security.retention.max_age_days",
                        retention.get('max_age_days'),
                        "Implement shorter retention for payment data"
                    ))
                
                return messages
        
        return PCIDSSRule()
    
    def validate_security(self, config: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate security configuration with compliance rules."""
        return self.validator.validate(config)
    
    def validate_security_and_raise(self, config: Dict[str, Any]) -> None:
        """Validate security configuration and raise on errors."""
        self.validator.validate_and_raise(config)


# Module exports
__all__ = [
    'ValidationSeverity',
    'ValidationContext',
    'ValidationMessage',
    'ValidationError',
    'ValidationRule',
    'PathExistsRule',
    'SecurityRule',
    'PerformanceRule',
    'ConsistencyRule',
    'ConfigurationValidator',
    'SecurityValidator'
]