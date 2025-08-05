"""
Error Pattern Analysis System for Arena Bot Deep Debugging

Advanced error analysis and pattern detection:
- Automatic error classification and categorization
- Pattern recognition for recurring errors and failure modes
- Root cause analysis with correlation across components
- Predictive failure analysis using historical patterns
- Error clustering and similarity detection
- Automated remediation suggestions based on known patterns
- Performance impact analysis of error patterns

Integrates with existing S-tier logging to analyze structured error data.
"""

import time
import re
import threading
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque, Counter
from difflib import SequenceMatcher

from ..logging_system.logger import get_logger, LogLevel


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    DETECTION_FAILURE = "detection_failure"
    AI_PROCESSING_ERROR = "ai_processing_error"
    GUI_ERROR = "gui_error"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    THREAD_ERROR = "thread_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorSignature:
    """
    Unique signature identifying an error pattern.
    
    Used to group similar errors together for pattern analysis.
    """
    
    # Signature identification
    signature_id: str = field(default_factory=lambda: str(uuid4()))
    signature_hash: str = ""
    
    # Error characteristics
    error_type: str = ""
    error_message_pattern: str = ""
    stack_trace_pattern: str = ""
    component_name: str = ""
    
    # Pattern metadata
    category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    # Occurrence tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    occurrence_count: int = 0
    
    # Context patterns
    common_context: Dict[str, Any] = field(default_factory=dict)
    triggers: Set[str] = field(default_factory=set)
    
    def update_occurrence(self, context: Dict[str, Any] = None) -> None:
        """Update occurrence information."""
        self.occurrence_count += 1
        self.last_seen = time.time()
        
        if context:
            # Update common context (intersection of contexts)
            if not self.common_context:
                self.common_context = context.copy()
            else:
                # Keep only common keys with same values
                self.common_context = {
                    k: v for k, v in self.common_context.items()
                    if k in context and context[k] == v
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'signature_id': self.signature_id,
            'signature_hash': self.signature_hash,
            'error_type': self.error_type,
            'error_message_pattern': self.error_message_pattern,
            'component_name': self.component_name,
            'category': self.category.value,
            'severity': self.severity.value,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'occurrence_count': self.occurrence_count,
            'common_context': self.common_context,
            'triggers': list(self.triggers)
        }


@dataclass
class ErrorOccurrence:
    """
    Represents a single error occurrence.
    """
    
    # Occurrence identification
    occurrence_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Error information
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    
    # Context information
    component_name: str = ""
    method_name: str = ""
    correlation_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Signature reference
    signature_id: Optional[str] = None
    
    # Performance impact
    processing_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Recovery information
    recovered: bool = False
    recovery_time_ms: Optional[float] = None
    recovery_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'occurrence_id': self.occurrence_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'component_name': self.component_name,
            'method_name': self.method_name,
            'correlation_id': self.correlation_id,
            'context': self.context,
            'signature_id': self.signature_id,
            'processing_time_ms': self.processing_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'recovered': self.recovered,
            'recovery_time_ms': self.recovery_time_ms,
            'recovery_method': self.recovery_method
        }


@dataclass
class ErrorPattern:
    """
    Represents a pattern of errors with analysis and recommendations.
    """
    
    # Pattern identification
    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    pattern_name: str = ""
    
    # Pattern composition
    signatures: List[ErrorSignature] = field(default_factory=list)
    total_occurrences: int = 0
    
    # Temporal analysis
    frequency_analysis: Dict[str, Any] = field(default_factory=dict)
    trend_direction: str = "stable"  # increasing, decreasing, stable
    
    # Impact analysis
    affected_components: Set[str] = field(default_factory=set)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    availability_impact: float = 0.0
    
    # Predictive analysis
    prediction_confidence: float = 0.0
    next_occurrence_estimate: Optional[float] = None
    
    # Recommendations
    remediation_suggestions: List[str] = field(default_factory=list)
    prevention_strategies: List[str] = field(default_factory=list)
    
    def add_signature(self, signature: ErrorSignature) -> None:
        """Add an error signature to this pattern."""
        self.signatures.append(signature)
        self.total_occurrences += signature.occurrence_count
        self.affected_components.add(signature.component_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'total_occurrences': self.total_occurrences,
            'signature_count': len(self.signatures),
            'frequency_analysis': self.frequency_analysis,
            'trend_direction': self.trend_direction,
            'affected_components': list(self.affected_components),
            'performance_impact': self.performance_impact,
            'availability_impact': self.availability_impact,
            'prediction_confidence': self.prediction_confidence,
            'next_occurrence_estimate': self.next_occurrence_estimate,
            'remediation_suggestions': self.remediation_suggestions,
            'prevention_strategies': self.prevention_strategies
        }


class ErrorClassifier:
    """
    Classifies errors into categories and determines severity.
    """
    
    def __init__(self):
        """Initialize error classifier."""
        self.classification_rules = self._build_classification_rules()
        self.severity_rules = self._build_severity_rules()
    
    def _build_classification_rules(self) -> Dict[ErrorCategory, List[str]]:
        """Build error classification rules."""
        return {
            ErrorCategory.DETECTION_FAILURE: [
                r".*card.*detect.*",
                r".*histogram.*match.*",
                r".*template.*match.*",
                r".*coordinate.*detect.*",
                r".*screen.*detect.*"
            ],
            ErrorCategory.AI_PROCESSING_ERROR: [
                r".*ai.*advisor.*",
                r".*recommendation.*",
                r".*tier.*lookup.*",
                r".*evaluation.*engine.*",
                r".*grandmaster.*"
            ],
            ErrorCategory.GUI_ERROR: [
                r".*tkinter.*",
                r".*gui.*",
                r".*widget.*",
                r".*display.*",
                r".*overlay.*"
            ],
            ErrorCategory.RESOURCE_ERROR: [
                r".*memory.*",
                r".*disk.*space.*",
                r".*cpu.*",
                r".*resource.*unavailable.*"
            ],
            ErrorCategory.NETWORK_ERROR: [
                r".*connection.*",
                r".*network.*",
                r".*timeout.*",
                r".*socket.*",
                r".*http.*"
            ],
            ErrorCategory.DATA_ERROR: [
                r".*json.*parse.*",
                r".*data.*corrupt.*",
                r".*format.*error.*",
                r".*encoding.*"
            ],
            ErrorCategory.CONFIGURATION_ERROR: [
                r".*config.*",
                r".*setting.*",
                r".*parameter.*",
                r".*initialization.*"
            ],
            ErrorCategory.DEPENDENCY_ERROR: [
                r".*import.*error.*",
                r".*module.*not.*found.*",
                r".*dependency.*",
                r".*library.*"
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                r".*timeout.*",
                r".*time.*out.*",
                r".*deadline.*exceeded.*"
            ],
            ErrorCategory.MEMORY_ERROR: [
                r".*memory.*error.*",
                r".*out.*of.*memory.*",
                r".*malloc.*"
            ],
            ErrorCategory.THREAD_ERROR: [
                r".*thread.*",
                r".*lock.*",
                r".*deadlock.*",
                r".*race.*condition.*"
            ]
        }
    
    def _build_severity_rules(self) -> Dict[ErrorSeverity, List[str]]:
        """Build error severity rules."""
        return {
            ErrorSeverity.CRITICAL: [
                r".*critical.*",
                r".*fatal.*",
                r".*crash.*",
                r".*system.*halt.*",
                r".*core.*dump.*"
            ],
            ErrorSeverity.HIGH: [
                r".*error.*",
                r".*exception.*",
                r".*failed.*",
                r".*abort.*"
            ],
            ErrorSeverity.MEDIUM: [
                r".*warning.*",
                r".*warn.*",
                r".*issue.*"
            ],
            ErrorSeverity.LOW: [
                r".*info.*",
                r".*notice.*",
                r".*debug.*"
            ]
        }
    
    def classify_error(self, error_message: str, error_type: str = "", 
                      component_name: str = "") -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error into category and severity.
        
        Returns:
            Tuple of (category, severity)
        """
        # Combine all text for classification
        text_to_classify = f"{error_message} {error_type} {component_name}".lower()
        
        # Classify category
        category = ErrorCategory.UNKNOWN_ERROR
        for cat, patterns in self.classification_rules.items():
            for pattern in patterns:
                if re.search(pattern, text_to_classify, re.IGNORECASE):
                    category = cat
                    break
            if category != ErrorCategory.UNKNOWN_ERROR:
                break
        
        # Classify severity
        severity = ErrorSeverity.MEDIUM  # Default
        for sev, patterns in self.severity_rules.items():
            for pattern in patterns:
                if re.search(pattern, text_to_classify, re.IGNORECASE):
                    severity = sev
                    break
            if severity != ErrorSeverity.MEDIUM:
                break
        
        return category, severity


class PatternDetector:
    """
    Detects patterns in error signatures and occurrences.
    """
    
    def __init__(self):
        """Initialize pattern detector."""
        self.similarity_threshold = 0.8
        self.frequency_threshold = 3  # Minimum occurrences to form pattern
    
    def detect_patterns(self, signatures: List[ErrorSignature]) -> List[ErrorPattern]:
        """
        Detect patterns in error signatures.
        
        Returns:
            List of detected error patterns
        """
        patterns = []
        processed_signatures = set()
        
        for signature in signatures:
            if signature.signature_id in processed_signatures:
                continue
            
            if signature.occurrence_count < self.frequency_threshold:
                continue
            
            # Find similar signatures
            similar_signatures = [signature]
            processed_signatures.add(signature.signature_id)
            
            for other_signature in signatures:
                if (other_signature.signature_id not in processed_signatures and
                    self._are_signatures_similar(signature, other_signature)):
                    similar_signatures.append(other_signature)
                    processed_signatures.add(other_signature.signature_id)
            
            # Create pattern if we have enough signatures
            if len(similar_signatures) >= 1:  # At least one signature
                pattern = self._create_pattern(similar_signatures)
                patterns.append(pattern)
        
        return patterns
    
    def _are_signatures_similar(self, sig1: ErrorSignature, sig2: ErrorSignature) -> bool:
        """Check if two signatures are similar enough to be in the same pattern."""
        # Same category
        if sig1.category != sig2.category:
            return False
        
        # Similar error types
        type_similarity = SequenceMatcher(None, sig1.error_type, sig2.error_type).ratio()
        if type_similarity < self.similarity_threshold:
            return False
        
        # Similar message patterns
        message_similarity = SequenceMatcher(
            None, sig1.error_message_pattern, sig2.error_message_pattern
        ).ratio()
        if message_similarity < self.similarity_threshold:
            return False
        
        # Same component (optional)
        if sig1.component_name and sig2.component_name:
            if sig1.component_name != sig2.component_name:
                return False
        
        return True
    
    def _create_pattern(self, signatures: List[ErrorSignature]) -> ErrorPattern:
        """Create an error pattern from similar signatures."""
        pattern = ErrorPattern()
        
        # Add all signatures
        for signature in signatures:
            pattern.add_signature(signature)
        
        # Generate pattern name
        most_common_type = max(signatures, key=lambda s: s.occurrence_count).error_type
        pattern.pattern_name = f"{most_common_type}_pattern"
        
        # Analyze frequency
        pattern.frequency_analysis = self._analyze_frequency(signatures)
        
        # Analyze trend
        pattern.trend_direction = self._analyze_trend(signatures)
        
        # Calculate performance impact
        pattern.performance_impact = self._calculate_performance_impact(signatures)
        
        # Generate recommendations
        pattern.remediation_suggestions = self._generate_remediation_suggestions(pattern)
        pattern.prevention_strategies = self._generate_prevention_strategies(pattern)
        
        return pattern
    
    def _analyze_frequency(self, signatures: List[ErrorSignature]) -> Dict[str, Any]:
        """Analyze frequency patterns of error signatures."""
        total_occurrences = sum(s.occurrence_count for s in signatures)
        
        # Calculate time-based frequency
        now = time.time()
        time_windows = {
            'last_hour': now - 3600,
            'last_day': now - 86400,
            'last_week': now - 604800
        }
        
        frequency_data = {
            'total_occurrences': total_occurrences,
            'average_occurrences_per_signature': total_occurrences / len(signatures),
            'time_windows': {}
        }
        
        for window_name, cutoff_time in time_windows.items():
            recent_count = sum(
                s.occurrence_count for s in signatures 
                if s.last_seen >= cutoff_time
            )
            frequency_data['time_windows'][window_name] = recent_count
        
        return frequency_data
    
    def _analyze_trend(self, signatures: List[ErrorSignature]) -> str:
        """Analyze trend direction of error pattern."""
        # Simple trend analysis based on recent vs historical occurrences
        now = time.time()
        recent_cutoff = now - 86400  # Last 24 hours
        
        recent_occurrences = sum(
            s.occurrence_count for s in signatures 
            if s.last_seen >= recent_cutoff
        )
        
        total_occurrences = sum(s.occurrence_count for s in signatures)
        historical_occurrences = total_occurrences - recent_occurrences
        
        # Calculate daily rates
        recent_daily_rate = recent_occurrences  # Already 24 hours
        
        # Estimate historical daily rate (rough approximation)
        oldest_signature = min(signatures, key=lambda s: s.first_seen)
        total_days = max(1, (now - oldest_signature.first_seen) / 86400)
        historical_daily_rate = historical_occurrences / total_days if total_days > 1 else 0
        
        if recent_daily_rate > historical_daily_rate * 1.5:
            return "increasing"
        elif recent_daily_rate < historical_daily_rate * 0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_impact(self, signatures: List[ErrorSignature]) -> Dict[str, float]:
        """Calculate performance impact of error pattern."""
        # This would be enhanced with actual performance data
        # For now, provide basic impact estimates
        total_occurrences = sum(s.occurrence_count for s in signatures)
        
        return {
            'error_rate_impact': min(total_occurrences / 1000.0, 1.0),  # Normalized impact
            'estimated_downtime_minutes': total_occurrences * 0.1,  # Rough estimate
            'user_experience_impact': min(total_occurrences / 500.0, 1.0)
        }
    
    def _generate_remediation_suggestions(self, pattern: ErrorPattern) -> List[str]:
        """Generate remediation suggestions for error pattern."""
        suggestions = []
        
        # Category-specific suggestions
        primary_category = max(
            pattern.signatures, 
            key=lambda s: s.occurrence_count
        ).category
        
        if primary_category == ErrorCategory.DETECTION_FAILURE:
            suggestions.extend([
                "Check card detection accuracy and thresholds",
                "Verify screenshot quality and resolution",
                "Update card database and templates",
                "Review coordinate detection parameters"
            ])
        elif primary_category == ErrorCategory.AI_PROCESSING_ERROR:
            suggestions.extend([
                "Check AI model availability and performance",
                "Verify input data format and quality",
                "Review AI processing timeouts",
                "Update AI model parameters"
            ])
        elif primary_category == ErrorCategory.GUI_ERROR:
            suggestions.extend([
                "Check GUI thread safety",
                "Verify widget initialization order",
                "Review event handling logic",
                "Update GUI library versions"
            ])
        elif primary_category == ErrorCategory.RESOURCE_ERROR:
            suggestions.extend([
                "Monitor system resource usage",
                "Implement resource cleanup",
                "Add resource usage limits",
                "Optimize memory allocation"
            ])
        
        # Frequency-based suggestions
        if pattern.total_occurrences > 100:
            suggestions.append("Implement circuit breaker pattern for this component")
        
        if pattern.trend_direction == "increasing":
            suggestions.append("Urgent: Error frequency is increasing - investigate immediately")
        
        return suggestions
    
    def _generate_prevention_strategies(self, pattern: ErrorPattern) -> List[str]:
        """Generate prevention strategies for error pattern."""
        strategies = []
        
        # Common prevention strategies
        strategies.extend([
            "Add more comprehensive error handling",
            "Implement retry mechanisms with exponential backoff",
            "Add input validation and sanitization",
            "Implement health checks and monitoring"
        ])
        
        # Pattern-specific strategies
        if len(pattern.affected_components) > 1:
            strategies.append("Implement error isolation between components")
        
        if pattern.total_occurrences > 50:
            strategies.append("Add automated recovery mechanisms")
        
        return strategies


class ErrorPatternAnalyzer:
    """
    Central error pattern analysis system for Arena Bot.
    
    Analyzes errors from S-tier logging, detects patterns,
    and provides actionable insights for debugging and prevention.
    """
    
    def __init__(self):
        """Initialize error pattern analyzer."""
        self.logger = get_logger("arena_bot.debugging.error_analyzer")
        
        # Components
        self.classifier = ErrorClassifier()
        self.pattern_detector = PatternDetector()
        
        # Data storage
        self.error_signatures: Dict[str, ErrorSignature] = {}
        self.error_occurrences: deque = deque(maxlen=10000)
        self.detected_patterns: List[ErrorPattern] = []
        
        # Configuration
        self.enabled = True
        self.auto_analyze = True
        self.analysis_interval = 300  # 5 minutes
        
        # Background analysis
        self.analysis_thread: Optional[threading.Thread] = None
        self.stop_analysis = threading.Event()
        
        # Statistics
        self.total_errors_analyzed = 0
        self.patterns_detected = 0
        self.last_analysis_time = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def analyze_error(self,
                     error_type: str,
                     error_message: str,
                     component_name: str = "",
                     method_name: str = "",
                     stack_trace: str = "",
                     context: Optional[Dict[str, Any]] = None,
                     correlation_id: Optional[str] = None,
                     processing_time_ms: Optional[float] = None,
                     memory_usage_mb: Optional[float] = None) -> ErrorOccurrence:
        """
        Analyze a single error occurrence.
        
        Args:
            error_type: Type of error (exception class name)
            error_message: Error message
            component_name: Component where error occurred
            method_name: Method where error occurred
            stack_trace: Stack trace (optional)
            context: Additional context information
            correlation_id: Correlation ID for tracing
            processing_time_ms: Processing time when error occurred
            memory_usage_mb: Memory usage when error occurred
            
        Returns:
            ErrorOccurrence object created for this error
        """
        
        if not self.enabled:
            return None
        
        with self.lock:
            # Create error occurrence
            occurrence = ErrorOccurrence(
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                component_name=component_name,
                method_name=method_name,
                correlation_id=correlation_id,
                context=context or {},
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_usage_mb
            )
            
            # Classify error
            category, severity = self.classifier.classify_error(
                error_message, error_type, component_name
            )
            
            # Create or find error signature
            signature = self._get_or_create_signature(
                error_type, error_message, component_name, category, severity
            )
            
            # Update signature
            signature.update_occurrence(context)
            occurrence.signature_id = signature.signature_id
            
            # Store occurrence
            self.error_occurrences.append(occurrence)
            self.total_errors_analyzed += 1
            
            # Log the analysis
            self.logger.warning(
                f"🔍 ERROR_ANALYZED: {error_type} in {component_name} "
                f"(category: {category.value}, severity: {severity.value})",
                extra={
                    'error_analysis': {
                        'occurrence': occurrence.to_dict(),
                        'signature': signature.to_dict(),
                        'category': category.value,
                        'severity': severity.value
                    }
                }
            )
            
            # Trigger pattern analysis if auto-analyze is enabled
            if self.auto_analyze:
                current_time = time.time()
                if current_time - self.last_analysis_time > self.analysis_interval:
                    self._trigger_pattern_analysis()
            
            return occurrence
    
    def _get_or_create_signature(self,
                                error_type: str,
                                error_message: str,
                                component_name: str,
                                category: ErrorCategory,
                                severity: ErrorSeverity) -> ErrorSignature:
        """Get existing signature or create new one."""
        
        # Create signature hash
        signature_text = f"{error_type}:{self._normalize_message(error_message)}:{component_name}"
        signature_hash = hashlib.md5(signature_text.encode()).hexdigest()[:16]
        
        # Check if signature exists
        if signature_hash in self.error_signatures:
            return self.error_signatures[signature_hash]
        
        # Create new signature
        signature = ErrorSignature(
            signature_hash=signature_hash,
            error_type=error_type,
            error_message_pattern=self._extract_message_pattern(error_message),
            component_name=component_name,
            category=category,
            severity=severity
        )
        
        self.error_signatures[signature_hash] = signature
        return signature
    
    def _normalize_message(self, message: str) -> str:
        """Normalize error message for signature creation."""
        # Remove variable parts (numbers, paths, IDs)
        normalized = re.sub(r'\d+', 'N', message)
        normalized = re.sub(r'/[^\s]+', '/PATH', normalized)
        normalized = re.sub(r'[a-f0-9-]{32,}', 'ID', normalized)
        return normalized.lower().strip()
    
    def _extract_message_pattern(self, message: str) -> str:
        """Extract pattern from error message."""
        # This could be enhanced with more sophisticated pattern extraction
        return self._normalize_message(message)
    
    def _trigger_pattern_analysis(self) -> None:
        """Trigger pattern analysis in background."""
        if not self.analysis_thread or not self.analysis_thread.is_alive():
            self.analysis_thread = threading.Thread(
                target=self._perform_pattern_analysis,
                name="ErrorPatternAnalysis",
                daemon=True
            )
            self.analysis_thread.start()
    
    def _perform_pattern_analysis(self) -> None:
        """Perform pattern analysis in background thread."""
        try:
            with self.lock:
                # Get all signatures
                signatures = list(self.error_signatures.values())
                
                # Detect patterns
                patterns = self.pattern_detector.detect_patterns(signatures)
                
                # Update detected patterns
                self.detected_patterns = patterns
                self.patterns_detected = len(patterns)
                self.last_analysis_time = time.time()
                
                # Log pattern analysis results
                self.logger.info(
                    f"🔍 PATTERN_ANALYSIS: Detected {len(patterns)} error patterns "
                    f"from {len(signatures)} signatures",
                    extra={
                        'pattern_analysis': {
                            'patterns_detected': len(patterns),
                            'signatures_analyzed': len(signatures),
                            'analysis_timestamp': self.last_analysis_time
                        }
                    }
                )
                
                # Log significant patterns
                for pattern in patterns:
                    if pattern.total_occurrences > 10:  # Significant threshold
                        self.logger.warning(
                            f"🚨 SIGNIFICANT_PATTERN: {pattern.pattern_name} "
                            f"({pattern.total_occurrences} occurrences, trend: {pattern.trend_direction})",
                            extra={
                                'significant_pattern': pattern.to_dict()
                            }
                        )
                
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
    
    def get_error_patterns(self, 
                          category: Optional[ErrorCategory] = None,
                          min_occurrences: int = 1) -> List[ErrorPattern]:
        """
        Get detected error patterns.
        
        Args:
            category: Filter by error category
            min_occurrences: Minimum occurrences to include
            
        Returns:
            List of error patterns matching criteria
        """
        
        with self.lock:
            patterns = self.detected_patterns.copy()
        
        # Apply filters
        if category:
            patterns = [
                p for p in patterns 
                if any(s.category == category for s in p.signatures)
            ]
        
        if min_occurrences > 1:
            patterns = [
                p for p in patterns 
                if p.total_occurrences >= min_occurrences
            ]
        
        # Sort by occurrence count (descending)
        patterns.sort(key=lambda p: p.total_occurrences, reverse=True)
        
        return patterns
    
    def get_error_signatures(self, 
                           category: Optional[ErrorCategory] = None,
                           component_name: Optional[str] = None) -> List[ErrorSignature]:
        """
        Get error signatures.
        
        Args:
            category: Filter by error category
            component_name: Filter by component name
            
        Returns:
            List of error signatures matching criteria
        """
        
        with self.lock:
            signatures = list(self.error_signatures.values())
        
        # Apply filters
        if category:
            signatures = [s for s in signatures if s.category == category]
        
        if component_name:
            signatures = [s for s in signatures if s.component_name == component_name]
        
        # Sort by occurrence count (descending)
        signatures.sort(key=lambda s: s.occurrence_count, reverse=True)
        
        return signatures
    
    def get_recent_errors(self, hours: int = 24, limit: int = 100) -> List[ErrorOccurrence]:
        """Get recent error occurrences."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            recent_errors = [
                error for error in self.error_occurrences
                if error.timestamp >= cutoff_time
            ]
        
        # Sort by timestamp (most recent first)
        recent_errors.sort(key=lambda e: e.timestamp, reverse=True)
        
        return recent_errors[:limit] if limit > 0 else recent_errors
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get error analysis performance statistics."""
        with self.lock:
            return {
                'total_errors_analyzed': self.total_errors_analyzed,
                'unique_signatures': len(self.error_signatures),
                'patterns_detected': self.patterns_detected,
                'last_analysis_time': self.last_analysis_time,
                'enabled': self.enabled,
                'auto_analyze': self.auto_analyze,
                'analysis_interval': self.analysis_interval
            }
    
    def force_pattern_analysis(self) -> List[ErrorPattern]:
        """Force immediate pattern analysis."""
        self._perform_pattern_analysis()
        return self.detected_patterns.copy()
    
    def enable(self) -> None:
        """Enable error analysis."""
        self.enabled = True
        self.logger.info("🔍 Error pattern analysis enabled")
    
    def disable(self) -> None:
        """Disable error analysis."""
        self.enabled = False
        self.logger.info("🔍 Error pattern analysis disabled")


# Global error pattern analyzer instance
_global_error_analyzer: Optional[ErrorPatternAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_error_analyzer() -> ErrorPatternAnalyzer:
    """Get global error pattern analyzer instance."""
    global _global_error_analyzer
    
    if _global_error_analyzer is None:
        with _analyzer_lock:
            if _global_error_analyzer is None:
                _global_error_analyzer = ErrorPatternAnalyzer()
    
    return _global_error_analyzer


def analyze_error(error_type: str,
                 error_message: str,
                 component_name: str = "",
                 **kwargs) -> ErrorOccurrence:
    """
    Convenience function to analyze errors.
    
    Uses the global error analyzer to analyze errors.
    """
    analyzer = get_error_analyzer()
    return analyzer.analyze_error(
        error_type=error_type,
        error_message=error_message,
        component_name=component_name,
        **kwargs
    )