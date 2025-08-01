"""
Security Filter for S-Tier Logging System.

This module provides comprehensive security filtering for log messages,
including PII detection and sanitization, credential scrubbing, sensitive
data pattern matching, and security policy enforcement.

Features:
- PII detection and sanitization (emails, SSNs, credit cards, etc.)
- Credential and API key scrubbing
- Custom sensitive data pattern matching
- Security policy enforcement
- Data classification and handling
- Audit trail for security actions
- GDPR and compliance support
- Thread-safe security operations
"""

import re
import time
import hashlib
import threading
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Set, Pattern
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# Import from our components
from .base_filter import BaseFilter, FilterResult
from ..core.hybrid_async_queue import LogMessage


class SecurityAction(Enum):
    """Security actions for sensitive data."""
    REDACT = "redact"                    # Replace with [REDACTED]
    MASK = "mask"                        # Partial masking (e.g., ****@email.com)
    HASH = "hash"                        # Replace with hash
    ENCRYPT = "encrypt"                  # Encrypt sensitive data
    REMOVE = "remove"                    # Remove field entirely
    BLOCK = "block"                      # Block entire message
    ALERT = "alert"                      # Allow but generate security alert


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"                    # Public data, no restrictions
    INTERNAL = "internal"                # Internal use only
    CONFIDENTIAL = "confidential"        # Confidential data
    RESTRICTED = "restricted"            # Highly restricted data
    SECRET = "secret"                    # Secret/classified data


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


@dataclass
class SecurityPattern:
    """Security pattern for sensitive data detection."""
    name: str
    pattern: Union[str, Pattern]         # Regex pattern
    pii_type: PIIType
    action: SecurityAction
    classification: DataClassification
    priority: int = 1                    # Pattern priority (higher wins)
    enabled: bool = True                 # Pattern enabled
    description: str = ""                # Pattern description
    false_positive_patterns: List[str] = field(default_factory=list)  # Patterns to exclude


@dataclass
class SecurityConfig:
    """Security filter configuration."""
    
    # Enable/disable security features
    enable_pii_detection: bool = True
    enable_credential_scrubbing: bool = True
    enable_custom_patterns: bool = True
    enable_context_analysis: bool = True
    
    # Default actions
    default_pii_action: SecurityAction = SecurityAction.REDACT
    default_credential_action: SecurityAction = SecurityAction.REDACT
    default_sensitive_action: SecurityAction = SecurityAction.MASK
    
    # PII detection sensitivity
    pii_detection_threshold: float = 0.8  # Confidence threshold for PII detection
    enable_name_detection: bool = False   # Disabled by default (high false positives)
    enable_address_detection: bool = False # Disabled by default (high false positives)
    
    # Credential patterns
    api_key_patterns: List[str] = field(default_factory=lambda: [
        r'api[_-]?key["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_-]{20,})',
        r'secret[_-]?key["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_-]{20,})',
        r'access[_-]?token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_-]{20,})',
        r'bearer["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_-]{20,})',
        r'authorization["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_-]{20,})'
    ])
    
    # Custom security patterns
    custom_patterns: List[SecurityPattern] = field(default_factory=list)
    
    # Security policy
    block_on_secret: bool = True          # Block messages containing secret data
    require_encryption_for_restricted: bool = True  # Require encryption for restricted data
    audit_all_security_actions: bool = True         # Audit all security actions
    
    # Performance and reliability
    max_pattern_matches: int = 100        # Maximum pattern matches per message
    pattern_timeout_seconds: float = 0.1  # Timeout for pattern matching
    enable_caching: bool = True           # Enable result caching
    cache_size_limit: int = 10000         # Cache size limit


@dataclass
class SecurityEvent:
    """Security event for audit trail."""
    timestamp: float
    message_id: str
    correlation_id: Optional[str]
    logger_name: str
    action: SecurityAction
    pii_type: PIIType
    pattern_name: str
    classification: DataClassification
    original_data: str                    # Hashed/encrypted original data
    sanitized_data: str                   # Sanitized version
    confidence: float                     # Detection confidence
    context: Dict[str, Any]              # Additional context


class SecurityFilter(BaseFilter):
    """
    Comprehensive security filter for sensitive data protection.
    
    Provides sophisticated security filtering including PII detection,
    credential scrubbing, custom pattern matching, and security policy
    enforcement to protect sensitive information in log messages.
    
    Features:
    - Multi-type PII detection and sanitization
    - API key and credential scrubbing  
    - Custom sensitive data patterns
    - Security policy enforcement
    - Data classification handling
    - Comprehensive audit trail
    - GDPR and compliance support
    """
    
    def __init__(self,
                 name: str = "security_filter",
                 config: Optional[SecurityConfig] = None,
                 enable_audit_trail: bool = True,
                 enable_performance_monitoring: bool = True,
                 encryption_key: Optional[bytes] = None,
                 **base_kwargs):
        """
        Initialize security filter.
        
        Args:
            name: Filter name for identification
            config: Security filter configuration
            enable_audit_trail: Enable security audit trail
            enable_performance_monitoring: Enable performance monitoring
            encryption_key: Key for encrypting sensitive data (32 bytes)
            **base_kwargs: Arguments for BaseFilter
        """
        super().__init__(name=name, **base_kwargs)
        
        self.config = config or SecurityConfig()
        self.enable_audit_trail = enable_audit_trail
        self.enable_performance_monitoring = enable_performance_monitoring
        self.encryption_key = encryption_key
        
        # Compile security patterns
        self.compiled_patterns: List[tuple] = []
        self._compile_security_patterns()
        
        # Audit trail
        self.security_events: deque = deque(maxlen=10000)
        self.audit_lock = threading.RLock()
        
        # Performance caching
        self.pattern_cache: Dict[str, List[tuple]] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.detection_stats: Dict[str, int] = defaultdict(int)
        self.action_stats: Dict[str, int] = defaultdict(int)
        self.stats_lock = threading.RLock()
        
        # Encryption setup
        if self.encryption_key:
            self._setup_encryption()
        
        self._logger.info(f"SecurityFilter '{name}' initialized",
                         extra={
                             'pii_detection': self.config.enable_pii_detection,
                             'credential_scrubbing': self.config.enable_credential_scrubbing,
                             'custom_patterns': len(self.config.custom_patterns),
                             'audit_trail': enable_audit_trail,
                             'encryption_enabled': bool(self.encryption_key),
                             'compiled_patterns': len(self.compiled_patterns)
                         })
    
    def _apply_filter(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """Apply security filtering logic."""
        start_time = time.time()
        
        # Create working copy of message
        sanitized_message = self._create_message_copy(message)
        security_events = []
        
        try:
            # Apply security patterns
            for pattern_info in self.compiled_patterns:
                compiled_pattern, pattern_config = pattern_info
                
                # Check timeout
                if time.time() - start_time > self.config.pattern_timeout_seconds:
                    self._logger.warning(f"Security pattern timeout exceeded")
                    break
                
                # Apply pattern
                events = self._apply_security_pattern(
                    sanitized_message, 
                    compiled_pattern, 
                    pattern_config
                )
                security_events.extend(events)
                
                # Check if we should block the message
                if any(event.action == SecurityAction.BLOCK for event in events):
                    self._record_security_events(security_events)
                    return FilterResult.REJECT, None
                
                # Check max matches limit
                if len(security_events) >= self.config.max_pattern_matches:
                    break
            
            # Record security events
            if security_events:
                self._record_security_events(security_events)
                self._update_security_stats(security_events)
            
            # Return sanitized message if any changes were made
            if security_events:
                return FilterResult.MODIFY, sanitized_message
            else:
                return FilterResult.ACCEPT, message
                
        except Exception as e:
            self._logger.error(f"Security filtering error: {e}")
            # Return original message on error to avoid losing data
            return FilterResult.ACCEPT, message
    
    def _create_message_copy(self, message: LogMessage) -> LogMessage:
        """Create a copy of the message for sanitization."""
        return LogMessage(
            timestamp=message.timestamp,
            level=message.level,
            logger_name=message.logger_name,
            message=message.message,
            correlation_id=message.correlation_id,
            thread_id=message.thread_id,
            process_id=message.process_id,
            context=message.context.copy() if message.context else {},
            error=message.error.copy() if message.error else None
        )
    
    def _apply_security_pattern(self, 
                              message: LogMessage, 
                              pattern: Pattern, 
                              config: SecurityPattern) -> List[SecurityEvent]:
        """Apply single security pattern to message."""
        events = []
        
        if not config.enabled:
            return events
        
        # Check message content
        content = message.message
        matches = pattern.finditer(content)
        
        for match in matches:
            # Check for false positives
            if self._is_false_positive(match.group(), config.false_positive_patterns):
                continue
            
            # Calculate confidence
            confidence = self._calculate_confidence(match, config)
            
            if confidence < self.config.pii_detection_threshold:
                continue
            
            # Apply security action
            original_data = match.group()
            sanitized_data = self._apply_security_action(
                original_data, 
                config.action, 
                config.pii_type
            )
            
            # Replace in message
            if sanitized_data != original_data:
                message.message = message.message.replace(original_data, sanitized_data)
            
            # Create security event
            event = SecurityEvent(
                timestamp=time.time(),
                message_id=f"{message.timestamp}_{hash(message.message)}",
                correlation_id=message.correlation_id,
                logger_name=message.logger_name,
                action=config.action,
                pii_type=config.pii_type,
                pattern_name=config.name,
                classification=config.classification,
                original_data=self._hash_sensitive_data(original_data),
                sanitized_data=sanitized_data,
                confidence=confidence,
                context={'match_start': match.start(), 'match_end': match.end()}
            )
            
            events.append(event)
        
        # Also check context data if enabled
        if self.config.enable_context_analysis and message.context:
            context_events = self._scan_context_data(message, pattern, config)
            events.extend(context_events)
        
        return events
    
    def _is_false_positive(self, match_text: str, false_positive_patterns: List[str]) -> bool:
        """Check if match is a false positive."""
        if not false_positive_patterns:
            return False
        
        for fp_pattern in false_positive_patterns:
            try:
                if re.search(fp_pattern, match_text, re.IGNORECASE):
                    return True
            except re.error:
                continue
        
        return False
    
    def _calculate_confidence(self, match: re.Match, config: SecurityPattern) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = 0.9  # Base confidence for regex match
        
        # Adjust based on pattern type
        if config.pii_type == PIIType.EMAIL:
            # Check for valid email structure
            email = match.group()
            if "@" in email and "." in email.split("@")[1]:
                base_confidence = 0.95
            else:
                base_confidence = 0.7
        
        elif config.pii_type == PIIType.CREDIT_CARD:
            # Check Luhn algorithm for credit cards
            cc_number = re.sub(r'\D', '', match.group())
            if self._luhn_check(cc_number):
                base_confidence = 0.98
            else:
                base_confidence = 0.6
        
        elif config.pii_type == PIIType.SSN:
            # Check SSN format
            ssn = re.sub(r'\D', '', match.group())
            if len(ssn) == 9 and ssn != "000000000":
                base_confidence = 0.9
            else:
                base_confidence = 0.5
        
        elif config.pii_type == PIIType.PHONE:
            # Check phone number format
            phone = re.sub(r'\D', '', match.group())
            if 10 <= len(phone) <= 15:
                base_confidence = 0.85
            else:
                base_confidence = 0.6
        
        return min(1.0, base_confidence)
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            def luhn_checksum(card_num):
                def digits_of(n):
                    return [int(d) for d in str(n)]
                digits = digits_of(card_num)
                odd_digits = digits[-1::-2]
                even_digits = digits[-2::-2]
                checksum = sum(odd_digits)
                for d in even_digits:
                    checksum += sum(digits_of(d*2))
                return checksum % 10
            
            return luhn_checksum(card_number) == 0
        except (ValueError, TypeError):
            return False
    
    def _apply_security_action(self, original_data: str, action: SecurityAction, pii_type: PIIType) -> str:
        """Apply security action to sensitive data."""
        if action == SecurityAction.REDACT:
            return f"[{pii_type.value.upper()}_REDACTED]"
        
        elif action == SecurityAction.MASK:
            return self._mask_data(original_data, pii_type)
        
        elif action == SecurityAction.HASH:
            return f"[{pii_type.value.upper()}_HASH:{self._hash_sensitive_data(original_data)[:8]}]"
        
        elif action == SecurityAction.ENCRYPT:
            if self.encryption_key:
                encrypted = self._encrypt_data(original_data)
                return f"[{pii_type.value.upper()}_ENCRYPTED:{encrypted[:16]}...]"
            else:
                # Fallback to redaction if no encryption key
                return f"[{pii_type.value.upper()}_REDACTED]"
        
        elif action == SecurityAction.REMOVE:
            return ""
        
        elif action == SecurityAction.ALERT:
            # Allow data through but generate alert
            return original_data
        
        else:
            # Default to redaction
            return f"[{pii_type.value.upper()}_REDACTED]"
    
    def _mask_data(self, data: str, pii_type: PIIType) -> str:
        """Apply masking to sensitive data."""
        if pii_type == PIIType.EMAIL:
            # Mask email: user@domain.com -> u***@domain.com
            if "@" in data:
                local, domain = data.split("@", 1)
                if len(local) > 1:
                    masked_local = local[0] + "*" * (len(local) - 1)
                    return f"{masked_local}@{domain}"
            return "***@***.***"
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Mask credit card: 1234567890123456 -> ****-****-****-3456
            digits = re.sub(r'\D', '', data)
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"
            return "****-****-****-****"
        
        elif pii_type == PIIType.SSN:
            # Mask SSN: 123456789 -> ***-**-6789
            digits = re.sub(r'\D', '', data)
            if len(digits) >= 4:
                return f"***-**-{digits[-4:]}"
            return "***-**-****"
        
        elif pii_type == PIIType.PHONE:
            # Mask phone: (555) 123-4567 -> (***) ***-4567
            digits = re.sub(r'\D', '', data)
            if len(digits) >= 4:
                return f"(***) ***-{digits[-4:]}"
            return "(***) ***-****"
        
        else:
            # Generic masking
            if len(data) <= 4:
                return "*" * len(data)
            else:
                return data[:2] + "*" * (len(data) - 4) + data[-2:]
    
    def _hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for audit trail."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.encryption_key:
            return self._hash_sensitive_data(data)
        
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Use first 32 bytes of key for Fernet
            key = base64.urlsafe_b64encode(self.encryption_key[:32])
            f = Fernet(key)
            encrypted = f.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except ImportError:
            # Fallback to hashing if cryptography not available
            return self._hash_sensitive_data(data)
        except Exception:
            return self._hash_sensitive_data(data)
    
    def _scan_context_data(self, message: LogMessage, pattern: Pattern, config: SecurityPattern) -> List[SecurityEvent]:
        """Scan message context data for sensitive information."""
        events = []
        
        if not message.context:
            return events
        
        def scan_dict(data: Dict[str, Any], path: str = "") -> None:
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, str):
                    matches = pattern.finditer(value)
                    for match in matches:
                        if self._is_false_positive(match.group(), config.false_positive_patterns):
                            continue
                        
                        confidence = self._calculate_confidence(match, config)
                        if confidence < self.config.pii_detection_threshold:
                            continue
                        
                        # Apply security action to context data
                        original_data = match.group()
                        sanitized_data = self._apply_security_action(
                            original_data, 
                            config.action,
                            config.pii_type
                        )
                        
                        # Update context data
                        if sanitized_data != original_data:
                            message.context[key] = value.replace(original_data, sanitized_data)
                        
                        # Create security event
                        event = SecurityEvent(
                            timestamp=time.time(),
                            message_id=f"{message.timestamp}_{hash(message.message)}",
                            correlation_id=message.correlation_id,
                            logger_name=message.logger_name,
                            action=config.action,
                            pii_type=config.pii_type,
                            pattern_name=config.name,
                            classification=config.classification,
                            original_data=self._hash_sensitive_data(original_data),
                            sanitized_data=sanitized_data,
                            confidence=confidence,
                            context={'context_path': current_path}
                        )
                        
                        events.append(event)
                
                elif isinstance(value, dict):
                    scan_dict(value, current_path)
                
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, (str, dict)):
                            if isinstance(item, str):
                                # Scan string in list
                                matches = pattern.finditer(item)
                                for match in matches:
                                    # Similar processing as above
                                    pass
                            elif isinstance(item, dict):
                                scan_dict(item, f"{current_path}[{i}]")
        
        try:
            scan_dict(message.context)
        except Exception as e:
            self._logger.warning(f"Error scanning context data: {e}")
        
        return events
    
    def _record_security_events(self, events: List[SecurityEvent]) -> None:
        """Record security events in audit trail."""
        if not self.enable_audit_trail:
            return
        
        with self.audit_lock:
            for event in events:
                self.security_events.append(event)
                
                # Log security event
                self._logger.warning(f"Security event: {event.action.value} applied to {event.pii_type.value}",
                                   extra={
                                       'security_event': True,
                                       'action': event.action.value,
                                       'pii_type': event.pii_type.value,
                                       'pattern': event.pattern_name,
                                       'classification': event.classification.value,
                                       'confidence': event.confidence,
                                       'correlation_id': event.correlation_id,
                                       'logger_name': event.logger_name
                                   })
    
    def _update_security_stats(self, events: List[SecurityEvent]) -> None:
        """Update security statistics."""
        with self.stats_lock:
            for event in events:
                self.detection_stats[event.pii_type.value] += 1
                self.action_stats[event.action.value] += 1
    
    def _compile_security_patterns(self) -> None:
        """Compile all security patterns for efficient matching."""
        self.compiled_patterns = []
        
        # Built-in PII patterns
        if self.config.enable_pii_detection:
            self._add_builtin_pii_patterns()
        
        # Credential patterns
        if self.config.enable_credential_scrubbing:
            self._add_credential_patterns()
        
        # Custom patterns
        if self.config.enable_custom_patterns:
            self._add_custom_patterns()
        
        # Sort by priority
        self.compiled_patterns.sort(key=lambda x: x[1].priority, reverse=True)
        
        self._logger.info(f"Compiled {len(self.compiled_patterns)} security patterns")
    
    def _add_builtin_pii_patterns(self) -> None:
        """Add built-in PII detection patterns."""
        patterns = [
            # Email addresses
            SecurityPattern(
                name="email_pattern",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                pii_type=PIIType.EMAIL,
                action=self.config.default_pii_action,
                classification=DataClassification.CONFIDENTIAL,
                priority=10
            ),
            
            # Credit card numbers
            SecurityPattern(
                name="credit_card_pattern",
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                pii_type=PIIType.CREDIT_CARD,
                action=self.config.default_pii_action,
                classification=DataClassification.RESTRICTED,
                priority=10
            ),
            
            # Social Security Numbers
            SecurityPattern(
                name="ssn_pattern",
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                pii_type=PIIType.SSN,
                action=self.config.default_pii_action,
                classification=DataClassification.RESTRICTED,
                priority=10
            ),
            
            # Phone numbers
            SecurityPattern(
                name="phone_pattern",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                pii_type=PIIType.PHONE,
                action=self.config.default_pii_action,
                classification=DataClassification.CONFIDENTIAL,
                priority=8
            ),
            
            # IP addresses
            SecurityPattern(
                name="ip_address_pattern",
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                pii_type=PIIType.IP_ADDRESS,
                action=self.config.default_sensitive_action,
                classification=DataClassification.INTERNAL,
                priority=5
            ),
            
            # MAC addresses
            SecurityPattern(
                name="mac_address_pattern",
                pattern=r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
                pii_type=PIIType.MAC_ADDRESS,
                action=self.config.default_sensitive_action,
                classification=DataClassification.INTERNAL,
                priority=5
            )
        ]
        
        for pattern_config in patterns:
            try:
                compiled_pattern = re.compile(pattern_config.pattern)
                self.compiled_patterns.append((compiled_pattern, pattern_config))
            except re.error as e:
                self._logger.warning(f"Invalid built-in pattern '{pattern_config.name}': {e}")
    
    def _add_credential_patterns(self) -> None:
        """Add credential scrubbing patterns."""
        for pattern_str in self.config.api_key_patterns:
            pattern_config = SecurityPattern(
                name=f"credential_pattern_{len(self.compiled_patterns)}",
                pattern=pattern_str,
                pii_type=PIIType.CUSTOM,
                action=self.config.default_credential_action,
                classification=DataClassification.SECRET,
                priority=15
            )
            
            try:
                compiled_pattern = re.compile(pattern_str, re.IGNORECASE)
                self.compiled_patterns.append((compiled_pattern, pattern_config))
            except re.error as e:
                self._logger.warning(f"Invalid credential pattern '{pattern_str}': {e}")
    
    def _add_custom_patterns(self) -> None:
        """Add custom security patterns."""
        for pattern_config in self.config.custom_patterns:
            try:
                if isinstance(pattern_config.pattern, str):
                    compiled_pattern = re.compile(pattern_config.pattern)
                else:
                    compiled_pattern = pattern_config.pattern
                
                self.compiled_patterns.append((compiled_pattern, pattern_config))
            except re.error as e:
                self._logger.warning(f"Invalid custom pattern '{pattern_config.name}': {e}")
    
    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data."""
        try:
            from cryptography.fernet import Fernet
            # Verify encryption key is valid
            if len(self.encryption_key) < 32:
                self._logger.warning("Encryption key too short, padding with zeros")
                self.encryption_key = self.encryption_key.ljust(32, b'\0')
            elif len(self.encryption_key) > 32:
                self._logger.warning("Encryption key too long, truncating")
                self.encryption_key = self.encryption_key[:32]
        except ImportError:
            self._logger.warning("Cryptography library not available, encryption disabled")
            self.encryption_key = None
    
    def add_custom_pattern(self, pattern: SecurityPattern) -> None:
        """Add custom security pattern."""
        self.config.custom_patterns.append(pattern)
        
        # Recompile patterns
        self._compile_security_patterns()
        
        self._logger.info(f"Added custom security pattern",
                         extra={
                             'pattern_name': pattern.name,
                             'pii_type': pattern.pii_type.value,
                             'action': pattern.action.value,
                             'classification': pattern.classification.value
                         })
    
    def remove_custom_pattern(self, pattern_name: str) -> bool:
        """Remove custom security pattern."""
        initial_count = len(self.config.custom_patterns)
        self.config.custom_patterns = [p for p in self.config.custom_patterns if p.name != pattern_name]
        removed = len(self.config.custom_patterns) < initial_count
        
        if removed:
            self._compile_security_patterns()
            self._logger.info(f"Removed custom security pattern: {pattern_name}")
        
        return removed
    
    def get_security_events(self, limit: Optional[int] = None) -> List[SecurityEvent]:
        """Get security events from audit trail."""
        with self.audit_lock:
            events = list(self.security_events)
            if limit:
                return events[-limit:]
            return events
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self.stats_lock:
            return {
                'detection_stats': dict(self.detection_stats),
                'action_stats': dict(self.action_stats),
                'total_events': len(self.security_events),
                'patterns_compiled': len(self.compiled_patterns)
            }
    
    def clear_security_events(self) -> None:
        """Clear security audit trail."""
        with self.audit_lock:
            self.security_events.clear()
        
        self._logger.info("Security audit trail cleared")
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get comprehensive filter summary."""
        base_summary = self.get_performance_summary()
        
        # Security-specific information
        security_summary = {
            **base_summary,
            'pii_detection_enabled': self.config.enable_pii_detection,
            'credential_scrubbing_enabled': self.config.enable_credential_scrubbing,
            'custom_patterns_enabled': self.config.enable_custom_patterns,
            'context_analysis_enabled': self.config.enable_context_analysis,
            'audit_trail_enabled': self.enable_audit_trail,
            'encryption_enabled': bool(self.encryption_key),
            'compiled_patterns': len(self.compiled_patterns),
            'custom_patterns': len(self.config.custom_patterns),
            'api_key_patterns': len(self.config.api_key_patterns),
            'pii_threshold': self.config.pii_detection_threshold,
            'default_pii_action': self.config.default_pii_action.value,
            'default_credential_action': self.config.default_credential_action.value,
            'security_events': len(self.security_events)
        }
        
        # Add statistics
        stats = self.get_security_stats()
        security_summary.update(stats)
        
        return security_summary


# Module exports
__all__ = [
    'SecurityFilter',
    'SecurityAction',
    'DataClassification',
    'PIIType',
    'SecurityPattern',
    'SecurityConfig',
    'SecurityEvent'
]