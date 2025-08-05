"""
Automated Issue Classification System for Arena Bot Deep Debugging

Intelligent issue classification and root cause analysis system that provides:
- Automated categorization of issues using machine learning and rule-based approaches
- Root cause analysis with confidence scoring and evidence correlation
- Pattern-based classification with historical learning and trend analysis
- Multi-dimensional issue analysis (performance, reliability, security, usability)
- Automated severity assessment with impact prediction and escalation rules
- Resolution suggestion engine with success probability scoring
- Integration with all debugging components for comprehensive context analysis
- Learning from resolution outcomes to improve future classifications

This system helps automatically identify, classify, and suggest solutions for
complex issues that are difficult to debug manually, especially those with
subtle patterns or multi-component interactions.
"""

import time
import threading
import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque, Counter
from pathlib import Path
import json
import statistics
from uuid import uuid4

# Import debugging components
from .enhanced_logger import get_enhanced_logger
from .error_analyzer import get_error_analyzer, ErrorPattern
from .exception_handler import get_exception_handler, ExceptionContext
from .health_monitor import get_health_monitor, ComponentHealth
from .performance_analyzer import get_performance_analyzer

from ..logging_system.logger import get_logger, LogLevel


class IssueCategory(Enum):
    """Main issue categories."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"
    DATA_INTEGRITY = "data_integrity"
    UNKNOWN = "unknown"


class IssueSeverity(Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueStatus(Enum):
    """Issue status tracking."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    DUPLICATE = "duplicate"


@dataclass
class IssueSignature:
    """Unique signature for issue identification."""
    
    signature_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Core signature components
    error_message_hash: Optional[str] = None
    exception_type: Optional[str] = None
    stack_trace_hash: Optional[str] = None
    component_name: Optional[str] = None
    function_name: Optional[str] = None
    
    # Contextual signature
    resource_usage_pattern: Optional[str] = None
    timing_pattern: Optional[str] = None
    dependency_pattern: Optional[str] = None
    
    def generate_signature_hash(self) -> str:
        """Generate a unique hash for this signature."""
        signature_data = {
            'error_message_hash': self.error_message_hash,
            'exception_type': self.exception_type,
            'stack_trace_hash': self.stack_trace_hash,
            'component_name': self.component_name,
            'function_name': self.function_name,
            'resource_usage_pattern': self.resource_usage_pattern,
            'timing_pattern': self.timing_pattern,
            'dependency_pattern': self.dependency_pattern
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]


@dataclass
class ClassifiedIssue:
    """Represents a classified issue with analysis and suggestions."""
    
    issue_id: str = field(default_factory=lambda: str(uuid4()))
    signature: IssueSignature = field(default_factory=IssueSignature)
    
    # Classification
    category: IssueCategory = IssueCategory.UNKNOWN
    subcategory: str = ""
    severity: IssueSeverity = IssueSeverity.MEDIUM
    confidence_score: float = 0.0
    
    # Issue details
    title: str = ""
    description: str = ""
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    occurrence_count: int = 1
    
    # Analysis
    root_cause_analysis: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    # Context
    error_context: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution
    suggested_solutions: List[Dict[str, Any]] = field(default_factory=list)
    resolution_notes: str = ""
    status: IssueStatus = IssueStatus.OPEN
    
    # Learning
    feedback_score: Optional[float] = None
    resolution_success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage."""
        return {
            'issue_id': self.issue_id,
            'signature_hash': self.signature.generate_signature_hash(),
            'category': self.category.value,
            'subcategory': self.subcategory,
            'severity': self.severity.value,
            'confidence_score': self.confidence_score,
            'title': self.title,
            'description': self.description,
            'first_occurrence': self.first_occurrence,
            'last_occurrence': self.last_occurrence,
            'occurrence_count': self.occurrence_count,
            'root_cause_analysis': self.root_cause_analysis,
            'contributing_factors': self.contributing_factors,
            'affected_components': self.affected_components,
            'error_context': self.error_context,
            'performance_impact': self.performance_impact,
            'system_state': self.system_state,
            'suggested_solutions': self.suggested_solutions,
            'status': self.status.value,
            'feedback_score': self.feedback_score,
            'resolution_success': self.resolution_success
        }


class RuleBasedClassifier:
    """Rule-based issue classifier using predefined patterns."""
    
    def __init__(self):
        """Initialize rule-based classifier."""
        self.logger = get_enhanced_logger("arena_bot.debugging.issue_classifier.rule_based")
        
        # Classification rules
        self.rules = self._load_classification_rules()
    
    def _load_classification_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load classification rules."""
        return {
            'performance': [
                {
                    'pattern': r'.*slow.*|.*timeout.*|.*performance.*|.*bottleneck.*',
                    'keywords': ['slow', 'timeout', 'performance', 'bottleneck', 'latency', 'response time'],
                    'exception_types': ['TimeoutError', 'TimeoutException'],
                    'subcategory': 'slow_response',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.3
                },
                {
                    'pattern': r'.*memory.*leak.*|.*out of memory.*|.*oom.*',
                    'keywords': ['memory leak', 'out of memory', 'oom', 'memory usage'],
                    'exception_types': ['MemoryError', 'OutOfMemoryError'],
                    'subcategory': 'memory_issue',
                    'severity': IssueSeverity.CRITICAL,
                    'confidence_boost': 0.4
                },
                {
                    'pattern': r'.*cpu.*high.*|.*100%.*cpu.*',
                    'keywords': ['high cpu', 'cpu usage', '100% cpu', 'cpu intensive'],
                    'subcategory': 'cpu_intensive',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.3
                }
            ],
            'reliability': [
                {
                    'pattern': r'.*connection.*failed.*|.*network.*error.*|.*unreachable.*',
                    'keywords': ['connection failed', 'network error', 'unreachable', 'connection timeout'],
                    'exception_types': ['ConnectionError', 'NetworkError', 'ConnectTimeout'],
                    'subcategory': 'network_failure',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.4
                },
                {
                    'pattern': r'.*database.*error.*|.*sql.*error.*|.*query.*failed.*',
                    'keywords': ['database error', 'sql error', 'query failed', 'database connection'],
                    'exception_types': ['DatabaseError', 'SQLError', 'OperationalError'],
                    'subcategory': 'database_failure',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.4
                },
                {
                    'pattern': r'.*deadlock.*|.*race.*condition.*|.*concurrent.*',
                    'keywords': ['deadlock', 'race condition', 'concurrent', 'threading'],
                    'subcategory': 'concurrency_issue',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.3
                }
            ],
            'security': [
                {
                    'pattern': r'.*unauthorized.*|.*permission.*denied.*|.*access.*denied.*',
                    'keywords': ['unauthorized', 'permission denied', 'access denied', 'forbidden'],
                    'exception_types': ['PermissionError', 'UnauthorizedError'],
                    'subcategory': 'access_control',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.4
                },
                {
                    'pattern': r'.*authentication.*failed.*|.*invalid.*credentials.*',
                    'keywords': ['authentication failed', 'invalid credentials', 'login failed'],
                    'subcategory': 'authentication_failure',
                    'severity': IssueSeverity.MEDIUM,
                    'confidence_boost': 0.3
                }
            ],
            'configuration': [
                {
                    'pattern': r'.*configuration.*error.*|.*config.*missing.*|.*setting.*invalid.*',
                    'keywords': ['configuration error', 'config missing', 'setting invalid', 'misconfigured'],
                    'exception_types': ['ConfigurationError', 'ValueError'],
                    'subcategory': 'invalid_configuration',
                    'severity': IssueSeverity.MEDIUM,
                    'confidence_boost': 0.3
                },
                {
                    'pattern': r'.*file.*not.*found.*|.*path.*not.*exist.*',
                    'keywords': ['file not found', 'path not exist', 'missing file'],
                    'exception_types': ['FileNotFoundError', 'IOError'],
                    'subcategory': 'missing_resource',
                    'severity': IssueSeverity.MEDIUM,
                    'confidence_boost': 0.3
                }
            ],
            'compatibility': [
                {
                    'pattern': r'.*version.*mismatch.*|.*incompatible.*|.*deprecated.*',
                    'keywords': ['version mismatch', 'incompatible', 'deprecated', 'version conflict'],
                    'subcategory': 'version_compatibility',
                    'severity': IssueSeverity.MEDIUM,
                    'confidence_boost': 0.3
                },
                {
                    'pattern': r'.*import.*error.*|.*module.*not.*found.*',
                    'keywords': ['import error', 'module not found', 'import failed'],
                    'exception_types': ['ImportError', 'ModuleNotFoundError'],
                    'subcategory': 'dependency_missing',
                    'severity': IssueSeverity.HIGH,
                    'confidence_boost': 0.4
                }
            ]
        }
    
    def classify_issue(self, issue_data: Dict[str, Any]) -> Tuple[IssueCategory, str, IssueSeverity, float]:
        """Classify an issue using rule-based approach."""
        best_category = IssueCategory.UNKNOWN
        best_subcategory = ""
        best_severity = IssueSeverity.MEDIUM
        best_confidence = 0.0
        
        # Extract text for pattern matching
        text_fields = [
            issue_data.get('error_message', ''),
            issue_data.get('description', ''),
            ' '.join(issue_data.get('keywords', [])),
            issue_data.get('exception_type', '')
        ]
        combined_text = ' '.join(text_fields).lower()
        
        # Check each category
        for category_name, category_rules in self.rules.items():
            category = IssueCategory(category_name)
            
            for rule in category_rules:
                confidence = 0.0
                
                # Pattern matching
                if 'pattern' in rule and re.search(rule['pattern'], combined_text, re.IGNORECASE):
                    confidence += 0.4
                
                # Keyword matching
                if 'keywords' in rule:
                    keyword_matches = sum(1 for keyword in rule['keywords'] if keyword.lower() in combined_text)
                    if keyword_matches > 0:
                        confidence += min(0.3, keyword_matches * 0.1)
                
                # Exception type matching
                if 'exception_types' in rule:
                    exception_type = issue_data.get('exception_type', '')
                    if exception_type in rule['exception_types']:
                        confidence += 0.3
                
                # Apply confidence boost
                if confidence > 0 and 'confidence_boost' in rule:
                    confidence += rule['confidence_boost']
                
                # Update best match
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_category = category
                    best_subcategory = rule.get('subcategory', '')
                    best_severity = rule.get('severity', IssueSeverity.MEDIUM)
        
        return best_category, best_subcategory, best_severity, min(best_confidence, 1.0)


class PatternBasedClassifier:
    """Pattern-based classifier using historical data."""
    
    def __init__(self):
        """Initialize pattern-based classifier."""
        self.logger = get_enhanced_logger("arena_bot.debugging.issue_classifier.pattern_based")
        
        # Pattern database
        self.issue_patterns: Dict[str, List[ClassifiedIssue]] = defaultdict(list)
        self.signature_database: Dict[str, ClassifiedIssue] = {}
        
        # Learning parameters
        self.similarity_threshold = 0.7
        self.pattern_confidence_weight = 0.4
    
    def add_classified_issue(self, issue: ClassifiedIssue) -> None:
        """Add a classified issue to the pattern database."""
        signature_hash = issue.signature.generate_signature_hash()
        
        # Store in signature database
        self.signature_database[signature_hash] = issue
        
        # Add to pattern database by category
        self.issue_patterns[issue.category.value].append(issue)
        
        self.logger.debug(f"Added issue pattern: {issue.title} ({issue.category.value})")
    
    def classify_by_similarity(self, issue_data: Dict[str, Any]) -> Tuple[IssueCategory, str, IssueSeverity, float]:
        """Classify issue by similarity to known patterns."""
        # Create signature for the new issue
        new_signature = self._create_signature(issue_data)
        signature_hash = new_signature.generate_signature_hash()
        
        # Check for exact match
        if signature_hash in self.signature_database:
            existing_issue = self.signature_database[signature_hash]
            return existing_issue.category, existing_issue.subcategory, existing_issue.severity, 0.95
        
        # Find similar issues
        best_similarity = 0.0
        best_match = None
        
        for category_issues in self.issue_patterns.values():
            for existing_issue in category_issues:
                similarity = self._calculate_similarity(new_signature, existing_issue.signature, issue_data)
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match = existing_issue
        
        if best_match:
            confidence = best_similarity * self.pattern_confidence_weight
            return best_match.category, best_match.subcategory, best_match.severity, confidence
        
        return IssueCategory.UNKNOWN, "", IssueSeverity.MEDIUM, 0.0
    
    def _create_signature(self, issue_data: Dict[str, Any]) -> IssueSignature:
        """Create issue signature from issue data."""
        signature = IssueSignature()
        
        # Error message hash
        error_message = issue_data.get('error_message', '')
        if error_message:
            # Normalize error message (remove specific values)
            normalized_message = re.sub(r'\d+', '<NUM>', error_message)
            normalized_message = re.sub(r'[a-f0-9]{8,}', '<HASH>', normalized_message)
            signature.error_message_hash = hashlib.md5(normalized_message.encode()).hexdigest()[:8]
        
        # Exception type
        signature.exception_type = issue_data.get('exception_type')
        
        # Stack trace hash
        stack_trace = issue_data.get('stack_trace', '')
        if stack_trace:
            # Normalize stack trace (remove line numbers and specific paths)
            normalized_trace = re.sub(r'line \d+', 'line <NUM>', stack_trace)
            normalized_trace = re.sub(r'/[^\s]+/', '/<PATH>/', normalized_trace)
            signature.stack_trace_hash = hashlib.md5(normalized_trace.encode()).hexdigest()[:8]
        
        # Component and function
        signature.component_name = issue_data.get('component_name')
        signature.function_name = issue_data.get('function_name')
        
        return signature
    
    def _calculate_similarity(self, sig1: IssueSignature, sig2: IssueSignature, 
                             issue_data: Dict[str, Any]) -> float:
        """Calculate similarity between two issue signatures."""
        similarity_score = 0.0
        total_weight = 0.0
        
        # Compare signature components
        comparisons = [
            ('error_message_hash', 0.3),
            ('exception_type', 0.2),
            ('stack_trace_hash', 0.2),
            ('component_name', 0.15),
            ('function_name', 0.15)
        ]
        
        for field, weight in comparisons:
            val1 = getattr(sig1, field)
            val2 = getattr(sig2, field)
            
            if val1 and val2:
                if val1 == val2:
                    similarity_score += weight
                total_weight += weight
            elif not val1 and not val2:
                total_weight += weight
        
        # Normalize by actual comparisons made
        if total_weight > 0:
            return similarity_score / total_weight
        
        return 0.0


class IssueClassifier:
    """
    Main automated issue classification system.
    
    Combines rule-based and pattern-based classification approaches
    to automatically categorize and analyze issues.
    """
    
    def __init__(self):
        """Initialize issue classifier."""
        self.logger = get_enhanced_logger("arena_bot.debugging.issue_classifier")
        
        # Classification components
        self.rule_based_classifier = RuleBasedClassifier()
        self.pattern_based_classifier = PatternBasedClassifier()
        
        # Issue tracking
        self.classified_issues: Dict[str, ClassifiedIssue] = {}
        self.issue_history: deque = deque(maxlen=1000)
        
        # Classification state
        self.classification_enabled = False
        self.learning_enabled = True
        self.auto_resolve_duplicates = True
        
        # Statistics
        self.classification_stats = {
            'total_classified': 0,
            'by_category': defaultdict(int),
            'by_severity': defaultdict(int),
            'accuracy_feedback': deque(maxlen=100)
        }
        
        # Integration with other debugging components
        self.error_analyzer = None
        self.exception_handler = None
        self.performance_analyzer = None
        
    def start_classification(self) -> bool:
        """Start automated issue classification."""
        try:
            self.classification_enabled = True
            
            # Initialize integration with other components
            self._initialize_integrations()
            
            self.logger.critical(
                "🤖 ISSUE_CLASSIFICATION_STARTED: Automated issue analysis enabled",
                extra={
                    'issue_classification_startup': {
                        'learning_enabled': self.learning_enabled,
                        'auto_resolve_duplicates': self.auto_resolve_duplicates,
                        'timestamp': time.time()
                    }
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start issue classification: {e}")
            return False
    
    def stop_classification(self) -> None:
        """Stop automated issue classification."""
        self.classification_enabled = False
        
        self.logger.info(
            "🔄 ISSUE_CLASSIFICATION_STOPPED: Classification system stopped",
            extra={
                'issue_classification_shutdown': {
                    'total_classified': self.classification_stats['total_classified'],
                    'issues_tracked': len(self.classified_issues),
                    'timestamp': time.time()
                }
            }
        )
    
    def classify_issue(self, issue_data: Dict[str, Any]) -> ClassifiedIssue:
        """Classify an issue using all available methods."""
        
        start_time = time.perf_counter()
        
        try:
            # Create initial issue
            issue = ClassifiedIssue()
            issue.signature = self.pattern_based_classifier._create_signature(issue_data)
            
            # Extract basic information
            issue.title = issue_data.get('title', issue_data.get('error_message', 'Unknown Issue')[:100])
            issue.description = issue_data.get('description', issue_data.get('error_message', ''))
            issue.error_context = issue_data.get('context', {})
            
            # Rule-based classification
            rule_category, rule_subcategory, rule_severity, rule_confidence = \
                self.rule_based_classifier.classify_issue(issue_data)
            
            # Pattern-based classification
            pattern_category, pattern_subcategory, pattern_severity, pattern_confidence = \
                self.pattern_based_classifier.classify_by_similarity(issue_data)
            
            # Combine classifications
            if rule_confidence > pattern_confidence:
                issue.category = rule_category
                issue.subcategory = rule_subcategory
                issue.severity = rule_severity
                issue.confidence_score = rule_confidence
                classification_method = "rule_based"
            elif pattern_confidence > 0:
                issue.category = pattern_category
                issue.subcategory = pattern_subcategory
                issue.severity = pattern_severity
                issue.confidence_score = pattern_confidence
                classification_method = "pattern_based"
            else:
                # Default classification
                issue.category = rule_category if rule_confidence > 0 else IssueCategory.UNKNOWN
                issue.subcategory = rule_subcategory
                issue.severity = rule_severity
                issue.confidence_score = max(rule_confidence, 0.1)
                classification_method = "default"
            
            # Enhanced analysis
            self._perform_root_cause_analysis(issue, issue_data)
            self._generate_resolution_suggestions(issue, issue_data)
            self._assess_impact(issue, issue_data)
            
            # Check for duplicates
            duplicate_issue = self._check_for_duplicates(issue)
            if duplicate_issue:
                duplicate_issue.occurrence_count += 1
                duplicate_issue.last_occurrence = time.time()
                
                if self.auto_resolve_duplicates:
                    issue.status = IssueStatus.DUPLICATE
                    
                    self.logger.info(
                        f"🔄 DUPLICATE_ISSUE_DETECTED: {issue.title}",
                        extra={
                            'duplicate_issue': {
                                'original_id': duplicate_issue.issue_id,
                                'duplicate_id': issue.issue_id,
                                'occurrence_count': duplicate_issue.occurrence_count
                            }
                        }
                    )
                    
                    return duplicate_issue
            
            # Store classified issue
            self.classified_issues[issue.issue_id] = issue
            self.issue_history.append(issue)
            
            # Update statistics
            self.classification_stats['total_classified'] += 1
            self.classification_stats['by_category'][issue.category.value] += 1
            self.classification_stats['by_severity'][issue.severity.value] += 1
            
            # Add to pattern database for learning
            if self.learning_enabled:
                self.pattern_based_classifier.add_classified_issue(issue)
            
            classification_time = (time.perf_counter() - start_time) * 1000
            
            self.logger.warning(
                f"🎯 ISSUE_CLASSIFIED: {issue.title} as {issue.category.value}/{issue.subcategory} "
                f"(confidence: {issue.confidence_score:.2f})",
                extra={
                    'issue_classification': {
                        'issue_id': issue.issue_id,
                        'category': issue.category.value,
                        'subcategory': issue.subcategory,
                        'severity': issue.severity.value,
                        'confidence_score': issue.confidence_score,
                        'classification_method': classification_method,
                        'classification_time_ms': classification_time,
                        'root_cause': issue.root_cause_analysis,
                        'suggested_solutions_count': len(issue.suggested_solutions)
                    }
                }
            )
            
            return issue
            
        except Exception as e:
            self.logger.error(f"Issue classification failed: {e}")
            
            # Return basic issue on failure
            fallback_issue = ClassifiedIssue()
            fallback_issue.title = issue_data.get('title', 'Classification Failed')
            fallback_issue.description = f"Classification error: {e}"
            fallback_issue.category = IssueCategory.UNKNOWN
            fallback_issue.severity = IssueSeverity.MEDIUM
            fallback_issue.confidence_score = 0.0
            
            return fallback_issue
    
    def _perform_root_cause_analysis(self, issue: ClassifiedIssue, issue_data: Dict[str, Any]) -> None:
        """Perform root cause analysis for the issue."""
        
        root_causes = []
        contributing_factors = []
        
        # Category-specific analysis
        if issue.category == IssueCategory.PERFORMANCE:
            if 'memory' in issue.subcategory:
                root_causes.append("Memory leak or excessive memory allocation")
                contributing_factors.extend([
                    "Unclosed resources or connections",
                    "Large object retention in memory",
                    "Inefficient data structures"
                ])
            elif 'cpu' in issue.subcategory:
                root_causes.append("CPU-intensive operations or inefficient algorithms")
                contributing_factors.extend([
                    "Inefficient loops or algorithms",
                    "Blocking operations on main thread",
                    "Excessive computational complexity"
                ])
            elif 'slow' in issue.subcategory:
                root_causes.append("Performance bottleneck in critical path")
                contributing_factors.extend([
                    "Slow database queries",
                    "Network latency issues",
                    "Inefficient caching strategy"
                ])
        
        elif issue.category == IssueCategory.RELIABILITY:
            if 'network' in issue.subcategory:
                root_causes.append("Network connectivity or configuration issues")
                contributing_factors.extend([
                    "Network infrastructure problems",
                    "Firewall or proxy configuration",
                    "DNS resolution issues"
                ])
            elif 'database' in issue.subcategory:
                root_causes.append("Database connectivity or query issues")
                contributing_factors.extend([
                    "Database server overload",
                    "Connection pool exhaustion",
                    "Invalid SQL queries or schema"
                ])
            elif 'concurrency' in issue.subcategory:
                root_causes.append("Thread synchronization or race condition")
                contributing_factors.extend([
                    "Improper lock usage",
                    "Shared resource contention",
                    "Deadlock conditions"
                ])
        
        elif issue.category == IssueCategory.CONFIGURATION:
            if 'invalid' in issue.subcategory:
                root_causes.append("Invalid or missing configuration parameters")
                contributing_factors.extend([
                    "Incorrect configuration values",
                    "Missing required settings",
                    "Configuration file corruption"
                ])
            elif 'missing' in issue.subcategory:
                root_causes.append("Missing required files or resources")
                contributing_factors.extend([
                    "File path configuration errors",
                    "Missing dependencies",
                    "Incorrect deployment setup"
                ])
        
        # Contextual analysis
        if 'stack_trace' in issue_data:
            contributing_factors.append("Exception occurred in critical code path")
        
        if 'performance_metrics' in issue_data:
            metrics = issue_data['performance_metrics']
            if metrics.get('memory_usage', 0) > 80:
                contributing_factors.append("High memory usage detected")
            if metrics.get('cpu_usage', 0) > 90:
                contributing_factors.append("High CPU usage detected")
        
        # Set analysis results
        issue.root_cause_analysis = "; ".join(root_causes) if root_causes else "Root cause analysis inconclusive"
        issue.contributing_factors = contributing_factors
    
    def _generate_resolution_suggestions(self, issue: ClassifiedIssue, issue_data: Dict[str, Any]) -> None:
        """Generate resolution suggestions for the issue."""
        
        suggestions = []
        
        # Category-specific suggestions
        if issue.category == IssueCategory.PERFORMANCE:
            if 'memory' in issue.subcategory:
                suggestions.extend([
                    {
                        'title': 'Investigate Memory Usage',
                        'description': 'Use memory profiler to identify large objects and potential leaks',
                        'priority': 'high',
                        'effort': 'medium',
                        'success_probability': 0.8
                    },
                    {
                        'title': 'Implement Resource Cleanup',
                        'description': 'Ensure proper cleanup of resources, connections, and large objects',
                        'priority': 'high',
                        'effort': 'low',
                        'success_probability': 0.7
                    },
                    {
                        'title': 'Optimize Data Structures',
                        'description': 'Review and optimize data structures for memory efficiency',
                        'priority': 'medium',
                        'effort': 'high',
                        'success_probability': 0.6
                    }
                ])
            
            elif 'cpu' in issue.subcategory:
                suggestions.extend([
                    {
                        'title': 'Profile CPU Usage',
                        'description': 'Use CPU profiler to identify performance hotspots',
                        'priority': 'high',
                        'effort': 'low',
                        'success_probability': 0.9
                    },
                    {
                        'title': 'Optimize Algorithms',
                        'description': 'Review and optimize critical algorithms for efficiency',
                        'priority': 'high',
                        'effort': 'high',
                        'success_probability': 0.8
                    },
                    {
                        'title': 'Implement Async Operations',
                        'description': 'Convert blocking operations to asynchronous where possible',
                        'priority': 'medium',
                        'effort': 'medium',
                        'success_probability': 0.7
                    }
                ])
        
        elif issue.category == IssueCategory.RELIABILITY:
            if 'network' in issue.subcategory:
                suggestions.extend([
                    {
                        'title': 'Implement Retry Logic',
                        'description': 'Add exponential backoff retry for network operations',
                        'priority': 'high',
                        'effort': 'low',
                        'success_probability': 0.8
                    },
                    {
                        'title': 'Add Connection Timeout',
                        'description': 'Configure appropriate timeout values for network connections',
                        'priority': 'medium',
                        'effort': 'low',
                        'success_probability': 0.7
                    },
                    {
                        'title': 'Implement Circuit Breaker',
                        'description': 'Add circuit breaker pattern for network resilience',
                        'priority': 'medium',
                        'effort': 'medium',
                        'success_probability': 0.8
                    }
                ])
            
            elif 'database' in issue.subcategory:
                suggestions.extend([
                    {
                        'title': 'Optimize Database Queries',
                        'description': 'Review and optimize slow database queries',
                        'priority': 'high',
                        'effort': 'medium',
                        'success_probability': 0.8
                    },
                    {
                        'title': 'Configure Connection Pool',
                        'description': 'Properly configure database connection pooling',
                        'priority': 'high',
                        'effort': 'low',
                        'success_probability': 0.9
                    },
                    {
                        'title': 'Add Database Monitoring',
                        'description': 'Implement database performance monitoring',
                        'priority': 'medium',
                        'effort': 'medium',
                        'success_probability': 0.7
                    }
                ])
        
        elif issue.category == IssueCategory.CONFIGURATION:
            suggestions.extend([
                {
                    'title': 'Validate Configuration',
                    'description': 'Implement configuration validation on startup',
                    'priority': 'high',
                    'effort': 'low',
                    'success_probability': 0.9
                },
                {
                    'title': 'Add Configuration Documentation',
                    'description': 'Document all configuration parameters and their valid values',
                    'priority': 'medium',
                    'effort': 'low',
                    'success_probability': 0.8
                },
                {
                    'title': 'Implement Configuration Templates',
                    'description': 'Provide configuration templates for different environments',
                    'priority': 'medium',
                    'effort': 'medium',
                    'success_probability': 0.7
                }
            ])
        
        # General suggestions
        suggestions.extend([
            {
                'title': 'Add Monitoring and Alerting',
                'description': 'Implement monitoring to catch similar issues early',
                'priority': 'medium',
                'effort': 'medium',
                'success_probability': 0.8
            },
            {
                'title': 'Improve Error Handling',
                'description': 'Add proper error handling and recovery mechanisms',
                'priority': 'medium',
                'effort': 'low',
                'success_probability': 0.7
            }
        ])
        
        # Sort by priority and success probability
        suggestions.sort(key=lambda x: (
            0 if x['priority'] == 'high' else 1 if x['priority'] == 'medium' else 2,
            -x['success_probability']
        ))
        
        issue.suggested_solutions = suggestions[:5]  # Keep top 5 suggestions
    
    def _assess_impact(self, issue: ClassifiedIssue, issue_data: Dict[str, Any]) -> None:
        """Assess the impact of the issue."""
        
        impact_assessment = {
            'user_experience': 'unknown',
            'system_stability': 'unknown',
            'performance_degradation': 'unknown',
            'business_impact': 'unknown'
        }
        
        # Assess based on category and severity
        if issue.category == IssueCategory.PERFORMANCE:
            if issue.severity in [IssueSeverity.HIGH, IssueSeverity.CRITICAL]:
                impact_assessment['user_experience'] = 'high'
                impact_assessment['performance_degradation'] = 'high'
            else:
                impact_assessment['user_experience'] = 'medium'
                impact_assessment['performance_degradation'] = 'medium'
        
        elif issue.category == IssueCategory.RELIABILITY:
            if issue.severity == IssueSeverity.CRITICAL:
                impact_assessment['system_stability'] = 'high'
                impact_assessment['business_impact'] = 'high'
            else:
                impact_assessment['system_stability'] = 'medium'
                impact_assessment['business_impact'] = 'medium'
        
        elif issue.category == IssueCategory.SECURITY:
            impact_assessment['business_impact'] = 'high'
            impact_assessment['system_stability'] = 'medium'
        
        # Store impact assessment
        issue.performance_impact = impact_assessment
    
    def _check_for_duplicates(self, issue: ClassifiedIssue) -> Optional[ClassifiedIssue]:
        """Check for duplicate issues."""
        
        signature_hash = issue.signature.generate_signature_hash()
        
        # Check for exact signature match
        for existing_issue in self.classified_issues.values():
            if existing_issue.signature.generate_signature_hash() == signature_hash:
                return existing_issue
        
        # Check for similar issues in the same category
        for existing_issue in self.classified_issues.values():
            if (existing_issue.category == issue.category and
                existing_issue.subcategory == issue.subcategory and
                self._calculate_issue_similarity(issue, existing_issue) > 0.8):
                return existing_issue
        
        return None
    
    def _calculate_issue_similarity(self, issue1: ClassifiedIssue, issue2: ClassifiedIssue) -> float:
        """Calculate similarity between two issues."""
        similarity_score = 0.0
        
        # Title similarity
        title1_words = set(issue1.title.lower().split())
        title2_words = set(issue2.title.lower().split())
        
        if title1_words and title2_words:
            title_similarity = len(title1_words.intersection(title2_words)) / len(title1_words.union(title2_words))
            similarity_score += title_similarity * 0.4
        
        # Category and subcategory match
        if issue1.category == issue2.category:
            similarity_score += 0.3
            
            if issue1.subcategory == issue2.subcategory:
                similarity_score += 0.3
        
        return similarity_score
    
    def _initialize_integrations(self) -> None:
        """Initialize integration with other debugging components."""
        try:
            self.error_analyzer = get_error_analyzer()
            self.exception_handler = get_exception_handler()
            self.performance_analyzer = get_performance_analyzer()
            
            self.logger.debug("Issue classifier integrations initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some integrations: {e}")
    
    def provide_feedback(self, issue_id: str, feedback_score: float, 
                        resolution_success: bool, feedback_notes: str = "") -> bool:
        """Provide feedback on issue classification accuracy."""
        
        issue = self.classified_issues.get(issue_id)
        if not issue:
            return False
        
        # Store feedback
        issue.feedback_score = feedback_score
        issue.resolution_success = resolution_success
        issue.resolution_notes = feedback_notes
        
        # Update statistics
        self.classification_stats['accuracy_feedback'].append({
            'issue_id': issue_id,
            'feedback_score': feedback_score,
            'resolution_success': resolution_success,
            'category': issue.category.value,
            'confidence_score': issue.confidence_score
        })
        
        self.logger.info(
            f"📝 CLASSIFICATION_FEEDBACK: {issue.title} scored {feedback_score:.2f}",
            extra={
                'classification_feedback': {
                    'issue_id': issue_id,
                    'feedback_score': feedback_score,
                    'resolution_success': resolution_success,
                    'original_confidence': issue.confidence_score,
                    'category': issue.category.value
                }
            }
        )
        
        return True
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics."""
        
        # Calculate accuracy metrics
        accuracy_data = list(self.classification_stats['accuracy_feedback'])
        accuracy_metrics = {}
        
        if accuracy_data:
            feedback_scores = [item['feedback_score'] for item in accuracy_data]
            accuracy_metrics = {
                'average_feedback_score': statistics.mean(feedback_scores),
                'median_feedback_score': statistics.median(feedback_scores),
                'successful_resolutions': sum(1 for item in accuracy_data if item['resolution_success']),
                'total_feedback_count': len(accuracy_data),
                'success_rate': sum(1 for item in accuracy_data if item['resolution_success']) / len(accuracy_data)
            }
        
        return {
            'classification_enabled': self.classification_enabled,
            'total_classified': self.classification_stats['total_classified'],
            'issues_by_category': dict(self.classification_stats['by_category']),
            'issues_by_severity': dict(self.classification_stats['by_severity']),
            'active_issues': len([issue for issue in self.classified_issues.values() 
                                if issue.status == IssueStatus.OPEN]),
            'accuracy_metrics': accuracy_metrics,
            'learning_enabled': self.learning_enabled,
            'pattern_database_size': len(self.pattern_based_classifier.signature_database)
        }
    
    def get_issue_report(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive issue report."""
        
        issue = self.classified_issues.get(issue_id)
        if not issue:
            return None
        
        return issue.to_dict()
    
    def get_trending_issues(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Get trending issues within a time window."""
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_issues = [
            issue for issue in self.classified_issues.values()
            if issue.last_occurrence >= cutoff_time
        ]
        
        # Sort by occurrence count and recency
        trending_issues = sorted(
            recent_issues,
            key=lambda x: (x.occurrence_count, x.last_occurrence),
            reverse=True
        )
        
        return [issue.to_dict() for issue in trending_issues[:10]]
    
    def shutdown(self) -> None:
        """Gracefully shutdown issue classifier."""
        self.logger.info("🔄 Shutting down issue classifier...")
        self.stop_classification()
        self.logger.info("✅ Issue classifier shutdown complete")


# Global issue classifier instance
_global_issue_classifier: Optional[IssueClassifier] = None
_classifier_lock = threading.Lock()


def get_issue_classifier() -> IssueClassifier:
    """Get global issue classifier instance."""
    global _global_issue_classifier
    
    if _global_issue_classifier is None:
        with _classifier_lock:
            if _global_issue_classifier is None:
                _global_issue_classifier = IssueClassifier()
    
    return _global_issue_classifier


def start_issue_classification() -> bool:
    """
    Start automated issue classification.
    
    Returns:
        True if classification started successfully
    """
    classifier = get_issue_classifier()
    return classifier.start_classification()


def stop_issue_classification() -> None:
    """Stop automated issue classification."""
    classifier = get_issue_classifier()
    classifier.stop_classification()


def classify_issue(issue_data: Dict[str, Any]) -> ClassifiedIssue:
    """Classify an issue automatically."""
    classifier = get_issue_classifier()
    return classifier.classify_issue(issue_data)


def provide_classification_feedback(issue_id: str, feedback_score: float,
                                  resolution_success: bool, notes: str = "") -> bool:
    """Provide feedback on classification accuracy."""
    classifier = get_issue_classifier()
    return classifier.provide_feedback(issue_id, feedback_score, resolution_success, notes)


def get_classification_status() -> Dict[str, Any]:
    """Get issue classifier status."""
    classifier = get_issue_classifier()
    return classifier.get_classification_statistics()


# Convenience decorator for automatic issue classification
def auto_classify_issues(component_name: str = "") -> Callable:
    """
    Decorator to automatically classify exceptions as issues.
    
    Usage:
        @auto_classify_issues("my_component")
        def my_function():
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Auto-classify the exception as an issue
                classifier = get_issue_classifier()
                
                if classifier.classification_enabled:
                    issue_data = {
                        'title': f"Exception in {func.__name__}",
                        'error_message': str(e),
                        'exception_type': type(e).__name__,
                        'component_name': component_name or 'unknown',
                        'function_name': func.__name__,
                        'stack_trace': traceback.format_exc(),
                        'context': {
                            'function_args': len(args),
                            'function_kwargs': list(kwargs.keys()),
                            'timestamp': time.time()
                        }
                    }
                    
                    classified_issue = classifier.classify_issue(issue_data)
                    
                    # Log the classified issue
                    logger = get_enhanced_logger(f"arena_bot.auto_classification.{component_name}")
                    logger.error(
                        f"🤖 AUTO_CLASSIFIED_ISSUE: {classified_issue.title}",
                        extra={
                            'auto_classified_issue': classified_issue.to_dict()
                        }
                    )
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator