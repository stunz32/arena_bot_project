"""
Correlation Filter for S-Tier Logging System.

This module provides correlation ID management and tracking for distributed
tracing, request correlation, and log message routing based on correlation
context and patterns.

Features:
- Automatic correlation ID generation and propagation
- Context-aware correlation tracking
- Distributed tracing integration
- Correlation-based message routing
- Parent-child relationship tracking
- Statistical correlation analysis
- Thread-safe correlation context management
"""

import time
import uuid
import threading
import logging
import contextvars
from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from contextlib import contextmanager

# Import from our components
from .base_filter import BaseFilter, FilterResult
from ..core.hybrid_async_queue import LogMessage


class CorrelationStrategy(Enum):
    """Correlation tracking strategies."""
    GENERATE_IF_MISSING = "generate_if_missing"     # Generate ID if not present
    REQUIRE_EXISTING = "require_existing"           # Only process messages with existing IDs
    PASS_THROUGH = "pass_through"                   # Pass all messages, don't modify
    ENRICH_CONTEXT = "enrich_context"              # Add contextual correlation data


class CorrelationScope(Enum):
    """Scope for correlation tracking."""
    THREAD = "thread"                               # Per-thread correlation
    REQUEST = "request"                             # Per-request correlation
    SESSION = "session"                             # Per-session correlation
    TRANSACTION = "transaction"                     # Per-transaction correlation
    GLOBAL = "global"                              # Global correlation


class CorrelationRouting(Enum):
    """Message routing based on correlation."""
    NONE = "none"                                  # No special routing
    BY_ID = "by_id"                               # Route by correlation ID
    BY_PATTERN = "by_pattern"                     # Route by ID pattern
    BY_CONTEXT = "by_context"                     # Route by correlation context
    BY_HIERARCHY = "by_hierarchy"                 # Route by parent-child relationships


@dataclass
class CorrelationContext:
    """Correlation context information."""
    correlation_id: str
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    
    # Custom context data
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationRule:
    """Rule for correlation-based message routing."""
    name: str
    pattern: str                                    # Regex pattern for correlation ID
    action: CorrelationRouting                     # Routing action
    target_sink: Optional[str] = None              # Target sink name
    priority: int = 1                              # Rule priority (higher wins)
    enabled: bool = True                           # Rule enabled status
    context_filters: Dict[str, Any] = field(default_factory=dict)  # Context-based filters


@dataclass
class CorrelationConfig:
    """Correlation filter configuration."""
    strategy: CorrelationStrategy = CorrelationStrategy.GENERATE_IF_MISSING
    scope: CorrelationScope = CorrelationScope.THREAD
    routing: CorrelationRouting = CorrelationRouting.NONE
    
    # ID generation
    id_prefix: str = ""                            # Prefix for generated IDs
    id_format: str = "uuid4"                       # ID format: uuid4, uuid1, timestamp, custom
    custom_id_generator: Optional[Callable[[], str]] = None  # Custom ID generator
    
    # Context management
    enable_context_vars: bool = True               # Use Python contextvars
    enable_thread_locals: bool = True              # Use thread-local storage
    auto_inherit_context: bool = True              # Auto-inherit context from parent
    context_timeout_seconds: float = 3600.0       # Context timeout (1 hour)
    
    # Routing rules
    routing_rules: List[CorrelationRule] = field(default_factory=list)
    default_sink: Optional[str] = None             # Default sink for unmatched messages
    
    # Parent-child tracking
    enable_hierarchy_tracking: bool = True         # Track parent-child relationships
    max_hierarchy_depth: int = 10                  # Maximum hierarchy depth
    
    # Performance and cleanup
    max_contexts: int = 100000                     # Maximum contexts to track
    cleanup_interval_seconds: float = 300.0       # Cleanup interval (5 minutes)
    enable_statistics: bool = True                 # Enable correlation statistics


@dataclass
class CorrelationStats:
    """Correlation tracking statistics."""
    correlation_id: str
    message_count: int
    first_seen: float
    last_seen: float
    hierarchy_depth: int
    parent_id: Optional[str]
    child_count: int
    context_data_size: int


# Context variables for correlation tracking
_correlation_context: contextvars.ContextVar[Optional[CorrelationContext]] = \
    contextvars.ContextVar('correlation_context', default=None)

# Thread-local storage for correlation context
_thread_local = threading.local()


class CorrelationFilter(BaseFilter):
    """
    Intelligent correlation tracking and routing filter.
    
    Provides sophisticated correlation ID management, distributed tracing
    integration, and correlation-based message routing for comprehensive
    log correlation across distributed systems.
    
    Features:
    - Automatic correlation ID generation and propagation
    - Context-aware correlation tracking
    - Parent-child relationship management
    - Correlation-based message routing
    - Distributed tracing integration
    - Statistical correlation analysis
    """
    
    def __init__(self,
                 name: str = "correlation_filter",
                 config: Optional[CorrelationConfig] = None,
                 enable_distributed_tracing: bool = True,
                 enable_performance_tracking: bool = True,
                 **base_kwargs):
        """
        Initialize correlation filter.
        
        Args:
            name: Filter name for identification
            config: Correlation filter configuration
            enable_distributed_tracing: Enable distributed tracing features
            enable_performance_tracking: Enable performance tracking
            **base_kwargs: Arguments for BaseFilter
        """
        super().__init__(name=name, **base_kwargs)
        
        self.config = config or CorrelationConfig()
        self.enable_distributed_tracing = enable_distributed_tracing
        self.enable_performance_tracking = enable_performance_tracking
        
        # Correlation context tracking
        self.contexts: Dict[str, CorrelationContext] = {}
        self.context_lock = threading.RLock()
        
        # Hierarchy tracking
        self.parent_child_map: Dict[str, Set[str]] = defaultdict(set)  # parent_id -> {child_ids}
        self.child_parent_map: Dict[str, str] = {}                     # child_id -> parent_id
        self.hierarchy_lock = threading.RLock()
        
        # Statistics
        self.correlation_stats: Dict[str, CorrelationStats] = {}
        self.stats_lock = threading.RLock()
        
        # Routing rules compilation
        self._compiled_routing_rules: List[Tuple[Any, CorrelationRule]] = []
        self._compile_routing_rules()
        
        # Performance optimization
        self._context_cache: Dict[str, CorrelationContext] = {}
        self._cache_lock = threading.RLock()
        self._cache_size_limit = 10000
        
        # Cleanup tracking
        self.last_cleanup = time.time()
        
        # ID generation
        self._setup_id_generator()
        
        self._logger.info(f"CorrelationFilter '{name}' initialized",
                         extra={
                             'strategy': self.config.strategy.value,
                             'scope': self.config.scope.value,
                             'routing': self.config.routing.value,
                             'context_vars_enabled': self.config.enable_context_vars,
                             'hierarchy_tracking': self.config.enable_hierarchy_tracking,
                             'routing_rules': len(self.config.routing_rules)
                         })
    
    def _apply_filter(self, message: LogMessage) -> tuple[FilterResult, Optional[LogMessage]]:
        """Apply correlation tracking and routing logic."""
        # Get or create correlation context
        correlation_context = self._get_or_create_correlation_context(message)
        
        # Update message with correlation information
        modified_message = self._enrich_message_with_correlation(message, correlation_context)
        
        # Update statistics
        self._update_correlation_stats(correlation_context, modified_message)
        
        # Apply routing logic if enabled
        routing_result = self._apply_correlation_routing(modified_message, correlation_context)
        
        # Perform periodic cleanup
        self._periodic_cleanup()
        
        return routing_result, modified_message
    
    def _get_or_create_correlation_context(self, message: LogMessage) -> CorrelationContext:
        """Get or create correlation context for message."""
        # Try to get existing correlation ID from message
        correlation_id = message.correlation_id
        
        # Try to get from current context if not in message
        if not correlation_id:
            current_context = self.get_current_context()
            if current_context:
                correlation_id = current_context.correlation_id
        
        # Generate new ID if needed and strategy allows
        if not correlation_id and self.config.strategy == CorrelationStrategy.GENERATE_IF_MISSING:
            correlation_id = self._generate_correlation_id()
        
        # Return existing context or create new one
        if correlation_id:
            return self._get_or_create_context(correlation_id, message)
        else:
            # No correlation ID available and strategy doesn't generate
            # Create minimal context
            return CorrelationContext(
                correlation_id="",
                thread_id=message.thread_id,
                process_id=message.process_id
            )
    
    def _get_or_create_context(self, correlation_id: str, message: LogMessage) -> CorrelationContext:
        """Get or create correlation context."""
        # Check cache first
        with self._cache_lock:
            if correlation_id in self._context_cache:
                return self._context_cache[correlation_id]
        
        with self.context_lock:
            if correlation_id not in self.contexts:
                # Create new context
                context = CorrelationContext(
                    correlation_id=correlation_id,
                    thread_id=message.thread_id,
                    process_id=message.process_id
                )
                
                # Try to inherit from current context
                if self.config.auto_inherit_context:
                    current_context = self.get_current_context()
                    if current_context and current_context.correlation_id != correlation_id:
                        context.parent_id = current_context.correlation_id
                        context.root_id = current_context.root_id or current_context.correlation_id
                        context.session_id = current_context.session_id
                        context.user_id = current_context.user_id
                        context.trace_id = current_context.trace_id
                        
                        # Update hierarchy tracking
                        if self.config.enable_hierarchy_tracking:
                            self._update_hierarchy(current_context.correlation_id, correlation_id)
                
                # Check context limit
                if len(self.contexts) >= self.config.max_contexts:
                    self._cleanup_old_contexts()
                
                self.contexts[correlation_id] = context
            
            context = self.contexts[correlation_id]
            
            # Update cache
            with self._cache_lock:
                if len(self._context_cache) >= self._cache_size_limit:
                    # Remove oldest entries
                    for _ in range(self._cache_size_limit // 4):
                        self._context_cache.pop(next(iter(self._context_cache)))
                
                self._context_cache[correlation_id] = context
            
            return context
    
    def _enrich_message_with_correlation(self, message: LogMessage, context: CorrelationContext) -> LogMessage:
        """Enrich message with correlation information."""
        if self.config.strategy == CorrelationStrategy.PASS_THROUGH:
            return message
        
        # Create enriched message
        enriched = LogMessage(
            timestamp=message.timestamp,
            level=message.level,
            logger_name=message.logger_name,
            message=message.message,
            correlation_id=context.correlation_id or message.correlation_id,
            thread_id=message.thread_id,
            process_id=message.process_id,
            context=message.context.copy() if message.context else {},
            error=message.error
        )
        
        # Add correlation context to message context
        if self.config.strategy == CorrelationStrategy.ENRICH_CONTEXT:
            enriched.context = enriched.context or {}
            enriched.context.update({
                'correlation': {
                    'id': context.correlation_id,
                    'parent_id': context.parent_id,
                    'root_id': context.root_id,
                    'session_id': context.session_id,
                    'user_id': context.user_id,
                    'request_id': context.request_id,
                    'trace_id': context.trace_id,
                    'span_id': context.span_id,
                    'hierarchy_depth': self._get_hierarchy_depth(context.correlation_id)
                }
            })
            
            # Add custom context data
            if context.context_data:
                enriched.context['correlation'].update(context.context_data)
        
        return enriched
    
    def _apply_correlation_routing(self, message: LogMessage, context: CorrelationContext) -> FilterResult:
        """Apply correlation-based routing logic."""
        if self.config.routing == CorrelationRouting.NONE:
            return FilterResult.ACCEPT
        
        # Apply routing rules
        for pattern, rule in self._compiled_routing_rules:
            if not rule.enabled:
                continue
            
            # Check pattern match
            if pattern and pattern.search(context.correlation_id or ""):
                # Check context filters if specified
                if rule.context_filters and not self._match_context_filters(context, rule.context_filters):
                    continue
                
                # Rule matched - apply routing action
                if rule.action == CorrelationRouting.BY_ID:
                    # Add routing information to message context
                    if not message.context:
                        message.context = {}
                    message.context['routing'] = {
                        'rule': rule.name,
                        'target_sink': rule.target_sink,
                        'correlation_id': context.correlation_id
                    }
                
                return FilterResult.ACCEPT
        
        # No rules matched - use default behavior
        return FilterResult.ACCEPT
    
    def _match_context_filters(self, context: CorrelationContext, filters: Dict[str, Any]) -> bool:
        """Check if context matches specified filters."""
        for key, expected_value in filters.items():
            if key == 'user_id':
                if context.user_id != expected_value:
                    return False
            elif key == 'session_id':
                if context.session_id != expected_value:
                    return False
            elif key == 'hierarchy_depth':
                depth = self._get_hierarchy_depth(context.correlation_id)
                if isinstance(expected_value, dict):
                    min_depth = expected_value.get('min', 0)
                    max_depth = expected_value.get('max', float('inf'))
                    if not (min_depth <= depth <= max_depth):
                        return False
                elif depth != expected_value:
                    return False
            elif key in context.context_data:
                if context.context_data[key] != expected_value:
                    return False
            else:
                return False
        
        return True
    
    def _update_hierarchy(self, parent_id: str, child_id: str) -> None:
        """Update parent-child hierarchy tracking."""
        with self.hierarchy_lock:
            # Add to parent-child mapping
            self.parent_child_map[parent_id].add(child_id)
            self.child_parent_map[child_id] = parent_id
            
            # Check hierarchy depth
            depth = self._get_hierarchy_depth(child_id)
            if depth > self.config.max_hierarchy_depth:
                self._logger.warning(f"Correlation hierarchy depth exceeded",
                                   extra={
                                       'child_id': child_id,
                                       'parent_id': parent_id,
                                       'depth': depth,
                                       'max_depth': self.config.max_hierarchy_depth
                                   })
    
    def _get_hierarchy_depth(self, correlation_id: str) -> int:
        """Get hierarchy depth for correlation ID."""
        if not self.config.enable_hierarchy_tracking:
            return 0
        
        with self.hierarchy_lock:
            depth = 0
            current_id = correlation_id
            
            while current_id in self.child_parent_map and depth < self.config.max_hierarchy_depth:
                current_id = self.child_parent_map[current_id]
                depth += 1
                
                # Prevent infinite loops
                if depth > self.config.max_hierarchy_depth:
                    break
            
            return depth
    
    def _update_correlation_stats(self, context: CorrelationContext, message: LogMessage) -> None:
        """Update correlation statistics."""
        if not self.config.enable_statistics:
            return
        
        correlation_id = context.correlation_id
        if not correlation_id:
            return
        
        current_time = time.time()
        
        with self.stats_lock:
            if correlation_id not in self.correlation_stats:
                self.correlation_stats[correlation_id] = CorrelationStats(
                    correlation_id=correlation_id,
                    message_count=0,
                    first_seen=current_time,
                    last_seen=current_time,
                    hierarchy_depth=self._get_hierarchy_depth(correlation_id),
                    parent_id=context.parent_id,
                    child_count=len(self.parent_child_map.get(correlation_id, set())),
                    context_data_size=len(context.context_data)
                )
            
            stats = self.correlation_stats[correlation_id]
            stats.message_count += 1
            stats.last_seen = current_time
            stats.child_count = len(self.parent_child_map.get(correlation_id, set()))
            stats.context_data_size = len(context.context_data)
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of old contexts."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.config.cleanup_interval_seconds:
            return
        
        self._cleanup_old_contexts()
        self.last_cleanup = current_time
    
    def _cleanup_old_contexts(self) -> None:
        """Clean up old correlation contexts."""
        current_time = time.time()
        cutoff_time = current_time - self.config.context_timeout_seconds
        
        contexts_to_remove = []
        
        with self.context_lock:
            for correlation_id, context in self.contexts.items():
                if context.created_at < cutoff_time:
                    contexts_to_remove.append(correlation_id)
            
            # Remove old contexts
            for correlation_id in contexts_to_remove:
                del self.contexts[correlation_id]
                
                # Remove from cache
                with self._cache_lock:
                    self._context_cache.pop(correlation_id, None)
                
                # Remove from hierarchy
                if self.config.enable_hierarchy_tracking:
                    with self.hierarchy_lock:
                        # Remove as parent
                        children = self.parent_child_map.pop(correlation_id, set())
                        
                        # Update children to remove parent reference
                        for child_id in children:
                            self.child_parent_map.pop(child_id, None)
                        
                        # Remove as child
                        parent_id = self.child_parent_map.pop(correlation_id, None)
                        if parent_id:
                            self.parent_child_map[parent_id].discard(correlation_id)
                
                # Remove stats
                if self.config.enable_statistics:
                    with self.stats_lock:
                        self.correlation_stats.pop(correlation_id, None)
        
        if contexts_to_remove:
            self._logger.debug(f"Cleaned up {len(contexts_to_remove)} old correlation contexts")
    
    def _setup_id_generator(self) -> None:
        """Setup correlation ID generator."""
        if self.config.custom_id_generator:
            self._id_generator = self.config.custom_id_generator
        elif self.config.id_format == "uuid4":
            self._id_generator = lambda: f"{self.config.id_prefix}{uuid.uuid4().hex}"
        elif self.config.id_format == "uuid1":
            self._id_generator = lambda: f"{self.config.id_prefix}{uuid.uuid1().hex}"
        elif self.config.id_format == "timestamp":
            self._id_generator = lambda: f"{self.config.id_prefix}{int(time.time() * 1000000)}"
        else:
            # Default to uuid4
            self._id_generator = lambda: f"{self.config.id_prefix}{uuid.uuid4().hex}"
    
    def _generate_correlation_id(self) -> str:
        """Generate new correlation ID."""
        return self._id_generator()
    
    def _compile_routing_rules(self) -> None:
        """Compile routing rules for efficient pattern matching."""
        import re
        
        self._compiled_routing_rules = []
        
        for rule in sorted(self.config.routing_rules, key=lambda r: r.priority, reverse=True):
            try:
                if rule.pattern:
                    compiled_pattern = re.compile(rule.pattern)
                    self._compiled_routing_rules.append((compiled_pattern, rule))
                else:
                    self._compiled_routing_rules.append((None, rule))
            except re.error as e:
                self._logger.warning(f"Invalid routing rule pattern '{rule.pattern}': {e}")
    
    def get_current_context(self) -> Optional[CorrelationContext]:
        """Get current correlation context."""
        # Try context variables first
        if self.config.enable_context_vars:
            context = _correlation_context.get()
            if context:
                return context
        
        # Try thread-local storage
        if self.config.enable_thread_locals:
            return getattr(_thread_local, 'correlation_context', None)
        
        return None
    
    def set_current_context(self, context: CorrelationContext) -> None:
        """Set current correlation context."""
        # Set in context variables
        if self.config.enable_context_vars:
            _correlation_context.set(context)
        
        # Set in thread-local storage
        if self.config.enable_thread_locals:
            _thread_local.correlation_context = context
        
        # Store in filter's context tracking
        with self.context_lock:
            self.contexts[context.correlation_id] = context
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None, **context_data):
        """Context manager for correlation tracking."""
        # Generate ID if not provided
        if not correlation_id:
            correlation_id = self._generate_correlation_id()
        
        # Create context
        context = CorrelationContext(
            correlation_id=correlation_id,
            **context_data
        )
        
        # Set context
        old_context = self.get_current_context()
        self.set_current_context(context)
        
        try:
            yield context
        finally:
            # Restore old context
            if old_context:
                self.set_current_context(old_context)
            else:
                self.clear_current_context()
    
    def clear_current_context(self) -> None:
        """Clear current correlation context."""
        if self.config.enable_context_vars:
            _correlation_context.set(None)
        
        if self.config.enable_thread_locals:
            if hasattr(_thread_local, 'correlation_context'):
                delattr(_thread_local, 'correlation_context')
    
    def add_routing_rule(self, rule: CorrelationRule) -> None:
        """Add correlation routing rule."""
        self.config.routing_rules.append(rule)
        self._compile_routing_rules()
        
        self._logger.info(f"Added correlation routing rule",
                         extra={
                             'rule_name': rule.name,
                             'pattern': rule.pattern,
                             'action': rule.action.value,
                             'priority': rule.priority
                         })
    
    def remove_routing_rule(self, rule_name: str) -> bool:
        """Remove correlation routing rule."""
        initial_count = len(self.config.routing_rules)
        self.config.routing_rules = [r for r in self.config.routing_rules if r.name != rule_name]
        removed = len(self.config.routing_rules) < initial_count
        
        if removed:
            self._compile_routing_rules()
            self._logger.info(f"Removed correlation routing rule: {rule_name}")
        
        return removed
    
    def get_correlation_stats(self) -> List[CorrelationStats]:
        """Get correlation statistics."""
        if not self.config.enable_statistics:
            return []
        
        with self.stats_lock:
            return list(self.correlation_stats.values())
    
    def get_hierarchy_info(self, correlation_id: str) -> Dict[str, Any]:
        """Get hierarchy information for correlation ID."""
        if not self.config.enable_hierarchy_tracking:
            return {}
        
        with self.hierarchy_lock:
            return {
                'correlation_id': correlation_id,
                'parent_id': self.child_parent_map.get(correlation_id),
                'children': list(self.parent_child_map.get(correlation_id, set())),
                'depth': self._get_hierarchy_depth(correlation_id),
                'is_root': correlation_id not in self.child_parent_map,
                'is_leaf': len(self.parent_child_map.get(correlation_id, set())) == 0
            }
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get comprehensive filter summary."""
        base_summary = self.get_performance_summary()
        
        # Correlation-specific information
        correlation_summary = {
            **base_summary,
            'strategy': self.config.strategy.value,
            'scope': self.config.scope.value,
            'routing': self.config.routing.value,
            'id_format': self.config.id_format,
            'id_prefix': self.config.id_prefix,
            'context_vars_enabled': self.config.enable_context_vars,
            'thread_locals_enabled': self.config.enable_thread_locals,
            'hierarchy_tracking': self.config.enable_hierarchy_tracking,
            'active_contexts': len(self.contexts),
            'routing_rules': len(self.config.routing_rules),
            'max_contexts': self.config.max_contexts,
            'context_timeout_seconds': self.config.context_timeout_seconds
        }
        
        # Add statistics if enabled
        if self.config.enable_statistics:
            stats = self.get_correlation_stats()
            correlation_summary.update({
                'tracked_correlations': len(stats),
                'total_messages': sum(s.message_count for s in stats),
                'active_hierarchies': len(self.parent_child_map)
            })
        
        return correlation_summary


# Module exports
__all__ = [
    'CorrelationFilter',
    'CorrelationStrategy',
    'CorrelationScope',
    'CorrelationRouting',
    'CorrelationContext',
    'CorrelationRule',
    'CorrelationConfig',
    'CorrelationStats'
]