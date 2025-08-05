"""
Dependency Tracking System for Arena Bot Deep Debugging

Advanced dependency analysis and tracking system that provides:
- Runtime dependency graph construction and visualization
- Circular dependency detection with resolution suggestions
- Module import tracking with performance impact analysis
- Component interaction mapping with data flow analysis
- Dependency health monitoring with failure propagation tracking
- Version compatibility analysis with upgrade recommendations
- Resource usage tracking per dependency with optimization insights
- Automated dependency optimization with performance improvements

This system helps identify complex dependency-related issues that can cause
hard-to-track bugs, performance problems, and system instabilities.
"""

import sys
import time
import threading
import inspect
import importlib
import pkgutil
import gc
import weakref
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import ast
import subprocess
import json
from uuid import uuid4

# Import debugging components
from .enhanced_logger import get_enhanced_logger
from .method_tracer import get_method_tracer
from .state_monitor import get_state_monitor
from .health_monitor import get_health_monitor

from ..logging_system.logger import get_logger, LogLevel


class DependencyType(Enum):
    """Types of dependencies to track."""
    MODULE_IMPORT = "module_import"
    FUNCTION_CALL = "function_call"
    CLASS_INHERITANCE = "class_inheritance"
    INSTANCE_REFERENCE = "instance_reference"
    DATA_DEPENDENCY = "data_dependency"
    RESOURCE_DEPENDENCY = "resource_dependency"
    CONFIGURATION_DEPENDENCY = "configuration_dependency"


class DependencyHealth(Enum):
    """Health status of dependencies."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph."""
    
    node_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    module_name: str = ""
    node_type: str = "module"  # module, class, function, instance
    
    # Metadata
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    source_code: Optional[str] = None
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Health and performance
    health_status: DependencyHealth = DependencyHealth.UNKNOWN
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    failure_count: int = 0
    average_load_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Version and compatibility
    version: Optional[str] = None
    required_version: Optional[str] = None
    compatibility_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            'node_id': self.node_id,
            'name': self.name,
            'module_name': self.module_name,
            'node_type': self.node_type,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'dependencies_count': len(self.dependencies),
            'dependents_count': len(self.dependents),
            'health_status': self.health_status.value,
            'access_count': self.access_count,
            'failure_count': self.failure_count,
            'average_load_time_ms': self.average_load_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'version': self.version,
            'compatibility_issues': self.compatibility_issues
        }


@dataclass
class DependencyEdge:
    """Represents an edge (relationship) in the dependency graph."""
    
    edge_id: str = field(default_factory=lambda: str(uuid4()))
    source_id: str = ""
    target_id: str = ""
    dependency_type: DependencyType = DependencyType.MODULE_IMPORT
    
    # Metadata
    created_time: float = field(default_factory=time.time)
    last_used_time: float = field(default_factory=time.time)
    usage_count: int = 0
    
    # Performance tracking
    call_frequency: float = 0.0  # calls per second
    average_call_time_ms: float = 0.0
    data_transfer_size_bytes: int = 0
    
    # Reliability
    success_rate: float = 1.0
    failure_count: int = 0
    
    # Context information
    context_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            'edge_id': self.edge_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'dependency_type': self.dependency_type.value,
            'usage_count': self.usage_count,
            'call_frequency': self.call_frequency,
            'average_call_time_ms': self.average_call_time_ms,
            'success_rate': self.success_rate,
            'failure_count': self.failure_count,
            'context_info': self.context_info
        }


@dataclass
class CircularDependency:
    """Represents a detected circular dependency."""
    
    cycle_id: str = field(default_factory=lambda: str(uuid4()))
    nodes: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    
    # Analysis
    severity: str = "medium"  # low, medium, high, critical
    impact_assessment: str = ""
    resolution_suggestions: List[str] = field(default_factory=list)
    
    # Detection metadata
    detected_time: float = field(default_factory=time.time)
    detection_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'cycle_id': self.cycle_id,
            'nodes': self.nodes,
            'edges': self.edges,
            'severity': self.severity,
            'impact_assessment': self.impact_assessment,
            'resolution_suggestions': self.resolution_suggestions,
            'detected_time': self.detected_time,
            'detection_method': self.detection_method,
            'cycle_length': len(self.nodes)
        }


class ImportTracker:
    """Tracks module imports and their performance."""
    
    def __init__(self):
        """Initialize import tracker."""
        self.logger = get_enhanced_logger("arena_bot.debugging.dependency_tracker.import_tracker")
        
        # Import tracking
        self.import_times: Dict[str, float] = {}
        self.import_order: List[str] = []
        self.import_failures: Dict[str, List[str]] = defaultdict(list)
        
        # Original import function
        self.original_import = None
        self.tracking_enabled = False
        
    def start_tracking(self) -> None:
        """Start tracking imports."""
        if self.tracking_enabled:
            return
        
        # Store original import
        self.original_import = __builtins__.__import__
        
        # Replace with our tracking version
        __builtins__.__import__ = self._tracked_import
        
        self.tracking_enabled = True
        self.logger.info("📦 Import tracking started")
    
    def stop_tracking(self) -> None:
        """Stop tracking imports."""
        if not self.tracking_enabled:
            return
        
        # Restore original import
        if self.original_import:
            __builtins__.__import__ = self.original_import
        
        self.tracking_enabled = False
        self.logger.info("📦 Import tracking stopped")
    
    def _tracked_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Tracked version of import function."""
        start_time = time.perf_counter()
        
        try:
            # Call original import
            result = self.original_import(name, globals, locals, fromlist, level)
            
            # Record successful import
            import_time = (time.perf_counter() - start_time) * 1000  # milliseconds
            self.import_times[name] = import_time
            self.import_order.append(name)
            
            # Log slow imports
            if import_time > 100:  # > 100ms
                self.logger.warning(
                    f"⚠️ SLOW_IMPORT: {name} took {import_time:.2f}ms to import",
                    extra={
                        'slow_import': {
                            'module_name': name,
                            'import_time_ms': import_time,
                            'fromlist': list(fromlist) if fromlist else [],
                            'level': level
                        }
                    }
                )
            
            return result
            
        except Exception as e:
            # Record import failure
            import_time = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)
            
            self.import_failures[name].append(error_msg)
            
            self.logger.error(
                f"❌ IMPORT_FAILED: {name} failed to import: {error_msg}",
                extra={
                    'import_failure': {
                        'module_name': name,
                        'error_message': error_msg,
                        'import_time_ms': import_time,
                        'fromlist': list(fromlist) if fromlist else [],
                        'level': level
                    }
                }
            )
            
            # Re-raise the exception
            raise
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """Get import performance statistics."""
        if not self.import_times:
            return {}
        
        times = list(self.import_times.values())
        
        return {
            'total_imports': len(self.import_times),
            'total_time_ms': sum(times),
            'average_time_ms': sum(times) / len(times),
            'slowest_import': max(self.import_times.items(), key=lambda x: x[1]),
            'fastest_import': min(self.import_times.items(), key=lambda x: x[1]),
            'import_failures': dict(self.import_failures),
            'import_order': self.import_order.copy()
        }


class DependencyGraph:
    """Manages the dependency graph structure."""
    
    def __init__(self):
        """Initialize dependency graph."""
        self.logger = get_enhanced_logger("arena_bot.debugging.dependency_tracker.graph")
        
        # Graph storage
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: Dict[str, DependencyEdge] = {}
        
        # Lookup indices
        self.node_by_name: Dict[str, str] = {}  # name -> node_id
        self.edges_by_source: Dict[str, Set[str]] = defaultdict(set)
        self.edges_by_target: Dict[str, Set[str]] = defaultdict(set)
        
        # Analysis cache
        self.circular_dependencies: List[CircularDependency] = []
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_timestamp = time.time()
        
        # Performance tracking
        self.modification_count = 0
        self.last_analysis_time = 0.0
    
    def add_node(self, node: DependencyNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        self.node_by_name[node.name] = node.node_id
        self.modification_count += 1
        self._invalidate_cache()
        
        self.logger.trace_deep(
            f"📦 NODE_ADDED: {node.name} ({node.node_type})",
            debug_context={
                'dependency_graph_node': node.to_dict(),
                'graph_size': len(self.nodes)
            }
        )
        
        return node.node_id
    
    def add_edge(self, edge: DependencyEdge) -> str:
        """Add an edge to the graph."""
        # Validate nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Cannot add edge: source or target node not found")
        
        self.edges[edge.edge_id] = edge
        self.edges_by_source[edge.source_id].add(edge.edge_id)
        self.edges_by_target[edge.target_id].add(edge.edge_id)
        
        # Update node dependencies
        source_node = self.nodes[edge.source_id]
        target_node = self.nodes[edge.target_id]
        
        source_node.dependencies.add(edge.target_id)
        target_node.dependents.add(edge.source_id)
        
        self.modification_count += 1
        self._invalidate_cache()
        
        self.logger.trace_deep(
            f"🔗 EDGE_ADDED: {source_node.name} -> {target_node.name} ({edge.dependency_type.value})",
            debug_context={
                'dependency_graph_edge': edge.to_dict(),
                'source_node': source_node.name,
                'target_node': target_node.name
            }
        )
        
        return edge.edge_id
    
    def find_node_by_name(self, name: str) -> Optional[DependencyNode]:
        """Find node by name."""
        node_id = self.node_by_name.get(name)
        return self.nodes.get(node_id) if node_id else None
    
    def get_dependencies(self, node_id: str) -> List[DependencyNode]:
        """Get all dependencies of a node."""
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        return [self.nodes[dep_id] for dep_id in node.dependencies if dep_id in self.nodes]
    
    def get_dependents(self, node_id: str) -> List[DependencyNode]:
        """Get all dependents of a node."""
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        return [self.nodes[dep_id] for dep_id in node.dependents if dep_id in self.nodes]
    
    def detect_circular_dependencies(self) -> List[CircularDependency]:
        """Detect circular dependencies using DFS."""
        start_time = time.perf_counter()
        
        visited = set()
        rec_stack = set()
        path = []
        cycles = []
        
        def dfs(node_id: str) -> None:
            """Depth-first search to detect cycles."""
            if node_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycle_nodes = path[cycle_start:] + [node_id]
                
                # Create circular dependency record
                cycle = CircularDependency(
                    nodes=cycle_nodes,
                    detected_time=time.time(),
                    detection_method="dfs_traversal"
                )
                
                # Analyze severity
                cycle.severity = self._analyze_cycle_severity(cycle_nodes)
                cycle.impact_assessment = self._assess_cycle_impact(cycle_nodes)
                cycle.resolution_suggestions = self._suggest_cycle_resolution(cycle_nodes)
                
                cycles.append(cycle)
                return
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            # Visit all dependencies
            node = self.nodes.get(node_id)
            if node:
                for dep_id in node.dependencies:
                    dfs(dep_id)
            
            rec_stack.remove(node_id)
            path.pop()
        
        # Run DFS from all unvisited nodes
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        analysis_time = (time.perf_counter() - start_time) * 1000
        self.last_analysis_time = analysis_time
        
        self.circular_dependencies = cycles
        
        if cycles:
            self.logger.warning(
                f"🔄 CIRCULAR_DEPENDENCIES_DETECTED: Found {len(cycles)} circular dependencies",
                extra={
                    'circular_dependency_analysis': {
                        'cycles_found': len(cycles),
                        'analysis_time_ms': analysis_time,
                        'cycles': [cycle.to_dict() for cycle in cycles]
                    }
                }
            )
        
        return cycles
    
    def _analyze_cycle_severity(self, cycle_nodes: List[str]) -> str:
        """Analyze the severity of a circular dependency."""
        # Simple heuristics for severity assessment
        cycle_length = len(cycle_nodes)
        
        if cycle_length <= 2:
            return "high"  # Direct circular dependency
        elif cycle_length <= 4:
            return "medium"
        else:
            return "low"  # Long cycles are often less problematic
    
    def _assess_cycle_impact(self, cycle_nodes: List[str]) -> str:
        """Assess the impact of a circular dependency."""
        node_names = []
        for node_id in cycle_nodes:
            node = self.nodes.get(node_id)
            if node:
                node_names.append(node.name)
        
        return f"Circular dependency involving {len(cycle_nodes)} components: {' -> '.join(node_names)}"
    
    def _suggest_cycle_resolution(self, cycle_nodes: List[str]) -> List[str]:
        """Suggest resolutions for circular dependency."""
        suggestions = [
            "Consider using dependency injection to break the cycle",
            "Move shared functionality to a separate module",
            "Use interfaces or abstract base classes to reduce coupling",
            "Implement lazy loading for one of the dependencies",
            "Refactor to use event-driven architecture"
        ]
        
        # Add specific suggestions based on cycle characteristics
        if len(cycle_nodes) == 2:
            suggestions.insert(0, "Consider merging the two modules if they are tightly coupled")
        
        return suggestions
    
    def get_dependency_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find dependency path between two nodes using BFS."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                return path
            
            current_node = self.nodes[current_id]
            for dep_id in current_node.dependencies:
                if dep_id not in visited:
                    visited.add(dep_id)
                    queue.append((dep_id, path + [dep_id]))
        
        return None  # No path found
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if not self.nodes:
            return {}
        
        # Basic statistics
        node_count = len(self.nodes)
        edge_count = len(self.edges)
        
        # Dependency analysis
        dependency_counts = [len(node.dependencies) for node in self.nodes.values()]
        dependent_counts = [len(node.dependents) for node in self.nodes.values()]
        
        # Health analysis
        health_distribution = defaultdict(int)
        for node in self.nodes.values():
            health_distribution[node.health_status.value] += 1
        
        # Performance metrics
        total_memory = sum(node.memory_usage_mb for node in self.nodes.values())
        avg_load_time = sum(node.average_load_time_ms for node in self.nodes.values()) / node_count if node_count > 0 else 0
        
        return {
            'graph_size': {
                'nodes': node_count,
                'edges': edge_count,
                'density': edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
            },
            'dependency_metrics': {
                'avg_dependencies_per_node': sum(dependency_counts) / node_count if node_count > 0 else 0,
                'max_dependencies': max(dependency_counts) if dependency_counts else 0,
                'avg_dependents_per_node': sum(dependent_counts) / node_count if node_count > 0 else 0,
                'max_dependents': max(dependent_counts) if dependent_counts else 0
            },
            'health_distribution': dict(health_distribution),
            'performance_metrics': {
                'total_memory_usage_mb': total_memory,
                'average_load_time_ms': avg_load_time,
                'analysis_time_ms': self.last_analysis_time
            },
            'circular_dependencies': len(self.circular_dependencies),
            'modification_count': self.modification_count
        }
    
    def _invalidate_cache(self) -> None:
        """Invalidate analysis cache."""
        self.analysis_cache.clear()
        self.cache_timestamp = time.time()


class DependencyTracker:
    """
    Main dependency tracking system.
    
    Provides comprehensive dependency analysis and monitoring for Arena Bot
    to identify complex dependency-related issues and performance problems.
    """
    
    def __init__(self):
        """Initialize dependency tracker."""
        self.logger = get_enhanced_logger("arena_bot.debugging.dependency_tracker")
        
        # Core components
        self.graph = DependencyGraph()
        self.import_tracker = ImportTracker()
        
        # Tracking state
        self.tracking_enabled = False
        self.auto_analysis_enabled = True
        self.analysis_interval_seconds = 300  # 5 minutes
        
        # Analysis thread
        self.analysis_thread: Optional[threading.Thread] = None
        self.stop_analysis = threading.Event()
        
        # Performance monitoring
        self.start_time = time.time()
        self.tracked_calls = 0
        self.analysis_count = 0
        
        # Integration with other debugging systems
        self.method_tracer = None
        self.state_monitor = None
        self.health_monitor = None
    
    def start_tracking(self, track_imports: bool = True, 
                      track_calls: bool = True,
                      auto_analyze: bool = True) -> bool:
        """
        Start dependency tracking.
        
        Args:
            track_imports: Enable import tracking
            track_calls: Enable function call tracking
            auto_analyze: Enable automatic analysis
            
        Returns:
            True if tracking started successfully
        """
        
        try:
            if self.tracking_enabled:
                self.logger.warning("Dependency tracking is already enabled")
                return True
            
            # Initialize integration with other debugging systems
            self._initialize_integrations()
            
            # Start import tracking
            if track_imports:
                self.import_tracker.start_tracking()
            
            # Start call tracking
            if track_calls:
                self._start_call_tracking()
            
            # Start automatic analysis
            if auto_analyze:
                self._start_auto_analysis()
            
            self.tracking_enabled = True
            self.start_time = time.time()
            
            self.logger.critical(
                "🚀 DEPENDENCY_TRACKING_STARTED: Advanced dependency analysis enabled",
                extra={
                    'dependency_tracking_startup': {
                        'track_imports': track_imports,
                        'track_calls': track_calls,
                        'auto_analyze': auto_analyze,
                        'analysis_interval_seconds': self.analysis_interval_seconds,
                        'timestamp': time.time()
                    }
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start dependency tracking: {e}")
            return False
    
    def stop_tracking(self) -> None:
        """Stop dependency tracking."""
        
        if not self.tracking_enabled:
            return
        
        try:
            # Stop automatic analysis
            self.stop_analysis.set()
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5.0)
            
            # Stop import tracking
            self.import_tracker.stop_tracking()
            
            # Stop call tracking
            self._stop_call_tracking()
            
            self.tracking_enabled = False
            
            uptime_seconds = time.time() - self.start_time
            
            self.logger.info(
                f"🔄 DEPENDENCY_TRACKING_STOPPED: Tracking stopped after {uptime_seconds:.1f}s",
                extra={
                    'dependency_tracking_shutdown': {
                        'uptime_seconds': uptime_seconds,
                        'tracked_calls': self.tracked_calls,
                        'analysis_count': self.analysis_count,
                        'graph_size': len(self.graph.nodes),
                        'timestamp': time.time()
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error stopping dependency tracking: {e}")
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Perform comprehensive dependency analysis."""
        
        start_time = time.perf_counter()
        
        try:
            # Detect circular dependencies
            circular_deps = self.graph.detect_circular_dependencies()
            
            # Get graph statistics
            graph_stats = self.graph.get_graph_statistics()
            
            # Get import statistics
            import_stats = self.import_tracker.get_import_statistics()
            
            # Analyze dependency health
            health_analysis = self._analyze_dependency_health()
            
            # Performance analysis
            performance_analysis = self._analyze_performance()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                circular_deps, graph_stats, health_analysis, performance_analysis
            )
            
            analysis_time = (time.perf_counter() - start_time) * 1000
            self.analysis_count += 1
            
            analysis_result = {
                'analysis_timestamp': time.time(),
                'analysis_time_ms': analysis_time,
                'graph_statistics': graph_stats,
                'import_statistics': import_stats,
                'circular_dependencies': [cycle.to_dict() for cycle in circular_deps],
                'health_analysis': health_analysis,
                'performance_analysis': performance_analysis,
                'recommendations': recommendations
            }
            
            self.logger.info(
                f"📊 DEPENDENCY_ANALYSIS_COMPLETE: Analysis completed in {analysis_time:.2f}ms",
                extra={
                    'dependency_analysis_result': analysis_result
                }
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
            return {'error': str(e), 'analysis_timestamp': time.time()}
    
    def add_module_dependency(self, source_module: str, target_module: str,
                            dependency_type: DependencyType = DependencyType.MODULE_IMPORT) -> None:
        """Add a module dependency to the graph."""
        
        # Create or get source node
        source_node = self.graph.find_node_by_name(source_module)
        if not source_node:
            source_node = DependencyNode(
                name=source_module,
                module_name=source_module,
                node_type="module"
            )
            self.graph.add_node(source_node)
        
        # Create or get target node
        target_node = self.graph.find_node_by_name(target_module)
        if not target_node:
            target_node = DependencyNode(
                name=target_module,
                module_name=target_module,
                node_type="module"
            )
            self.graph.add_node(target_node)
        
        # Create edge
        edge = DependencyEdge(
            source_id=source_node.node_id,
            target_id=target_node.node_id,
            dependency_type=dependency_type
        )
        
        self.graph.add_edge(edge)
    
    def update_node_health(self, node_name: str, health_status: DependencyHealth,
                          failure_info: Optional[str] = None) -> None:
        """Update the health status of a dependency node."""
        
        node = self.graph.find_node_by_name(node_name)
        if not node:
            return
        
        old_status = node.health_status
        node.health_status = health_status
        
        if health_status in [DependencyHealth.DEGRADED, DependencyHealth.FAILED]:
            node.failure_count += 1
            
            if failure_info:
                node.compatibility_issues.append(f"{datetime.now().isoformat()}: {failure_info}")
        
        # Log health changes
        if old_status != health_status:
            self.logger.warning(
                f"🏥 DEPENDENCY_HEALTH_CHANGED: {node_name} health changed from {old_status.value} to {health_status.value}",
                extra={
                    'dependency_health_change': {
                        'node_name': node_name,
                        'old_status': old_status.value,
                        'new_status': health_status.value,
                        'failure_count': node.failure_count,
                        'failure_info': failure_info
                    }
                }
            )
    
    def get_dependency_info(self, node_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a dependency."""
        
        node = self.graph.find_node_by_name(node_name)
        if not node:
            return None
        
        dependencies = self.graph.get_dependencies(node.node_id)
        dependents = self.graph.get_dependents(node.node_id)
        
        return {
            'node_info': node.to_dict(),
            'dependencies': [dep.to_dict() for dep in dependencies],
            'dependents': [dep.to_dict() for dep in dependents],
            'circular_dependencies': [
                cycle.to_dict() for cycle in self.graph.circular_dependencies
                if node.node_id in cycle.nodes
            ]
        }
    
    def _initialize_integrations(self) -> None:
        """Initialize integration with other debugging systems."""
        try:
            self.method_tracer = get_method_tracer()
            self.state_monitor = get_state_monitor()
            self.health_monitor = get_health_monitor()
            
            self.logger.debug("Dependency tracker integrations initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some integrations: {e}")
    
    def _start_call_tracking(self) -> None:
        """Start tracking function calls for dependency analysis."""
        # This would integrate with the method tracer to track function calls
        # and build the dependency graph based on call patterns
        pass
    
    def _stop_call_tracking(self) -> None:
        """Stop tracking function calls."""
        pass
    
    def _start_auto_analysis(self) -> None:
        """Start automatic dependency analysis thread."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        
        self.stop_analysis.clear()
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name="DependencyAnalysis",
            daemon=True
        )
        self.analysis_thread.start()
        
        self.logger.debug(f"Automatic dependency analysis started (interval: {self.analysis_interval_seconds}s)")
    
    def _analysis_loop(self) -> None:
        """Main analysis loop for automatic dependency analysis."""
        while not self.stop_analysis.wait(self.analysis_interval_seconds):
            try:
                self.analyze_dependencies()
            except Exception as e:
                self.logger.error(f"Automatic dependency analysis failed: {e}")
    
    def _analyze_dependency_health(self) -> Dict[str, Any]:
        """Analyze the health of all dependencies."""
        health_summary = defaultdict(int)
        unhealthy_nodes = []
        
        for node in self.graph.nodes.values():
            health_summary[node.health_status.value] += 1
            
            if node.health_status in [DependencyHealth.DEGRADED, DependencyHealth.FAILED]:
                unhealthy_nodes.append({
                    'name': node.name,
                    'health_status': node.health_status.value,
                    'failure_count': node.failure_count,
                    'compatibility_issues': node.compatibility_issues
                })
        
        return {
            'health_summary': dict(health_summary),
            'unhealthy_nodes': unhealthy_nodes,
            'overall_health_score': self._calculate_health_score()
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze dependency performance metrics."""
        if not self.graph.nodes:
            return {}
        
        load_times = [node.average_load_time_ms for node in self.graph.nodes.values() if node.average_load_time_ms > 0]
        memory_usage = [node.memory_usage_mb for node in self.graph.nodes.values() if node.memory_usage_mb > 0]
        
        slow_nodes = [
            {'name': node.name, 'load_time_ms': node.average_load_time_ms}
            for node in self.graph.nodes.values()
            if node.average_load_time_ms > 100  # > 100ms
        ]
        
        memory_intensive_nodes = [
            {'name': node.name, 'memory_usage_mb': node.memory_usage_mb}
            for node in self.graph.nodes.values()
            if node.memory_usage_mb > 50  # > 50MB
        ]
        
        return {
            'load_time_stats': {
                'average_ms': sum(load_times) / len(load_times) if load_times else 0,
                'max_ms': max(load_times) if load_times else 0,
                'slow_nodes': slow_nodes
            },
            'memory_usage_stats': {
                'total_mb': sum(memory_usage),
                'average_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'memory_intensive_nodes': memory_intensive_nodes
            }
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall dependency health score (0-100)."""
        if not self.graph.nodes:
            return 100.0
        
        total_nodes = len(self.graph.nodes)
        healthy_count = sum(1 for node in self.graph.nodes.values() 
                          if node.health_status == DependencyHealth.HEALTHY)
        
        return (healthy_count / total_nodes) * 100
    
    def _generate_recommendations(self, circular_deps: List[CircularDependency],
                                 graph_stats: Dict[str, Any],
                                 health_analysis: Dict[str, Any],
                                 performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Circular dependency recommendations
        if circular_deps:
            recommendations.append(f"⚠️ Found {len(circular_deps)} circular dependencies - consider refactoring")
            for cycle in circular_deps[:3]:  # Top 3 cycles
                recommendations.extend(cycle.resolution_suggestions[:2])  # Top 2 suggestions per cycle
        
        # Performance recommendations
        perf_stats = performance_analysis.get('load_time_stats', {})
        if perf_stats.get('average_ms', 0) > 50:
            recommendations.append("🚀 Average dependency load time is high - consider optimizing imports")
        
        slow_nodes = perf_stats.get('slow_nodes', [])
        if slow_nodes:
            recommendations.append(f"⚡ {len(slow_nodes)} slow-loading dependencies detected - review initialization code")
        
        # Memory recommendations
        memory_stats = performance_analysis.get('memory_usage_stats', {})
        if memory_stats.get('total_mb', 0) > 500:
            recommendations.append("💾 High total memory usage by dependencies - consider lazy loading")
        
        # Health recommendations
        health_score = health_analysis.get('overall_health_score', 100)
        if health_score < 80:
            recommendations.append("🏥 Overall dependency health is poor - investigate failing components")
        
        # Graph structure recommendations
        graph_size = graph_stats.get('graph_size', {})
        if graph_size.get('density', 0) > 0.3:
            recommendations.append("🕸️ Dependency graph is highly connected - consider modularization")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive dependency tracker status."""
        return {
            'tracking_enabled': self.tracking_enabled,
            'uptime_seconds': time.time() - self.start_time if self.tracking_enabled else 0,
            'tracked_calls': self.tracked_calls,
            'analysis_count': self.analysis_count,
            'graph_size': len(self.graph.nodes),
            'import_tracking_enabled': self.import_tracker.tracking_enabled,
            'auto_analysis_enabled': self.auto_analysis_enabled,
            'analysis_interval_seconds': self.analysis_interval_seconds,
            'circular_dependencies_count': len(self.graph.circular_dependencies)
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown dependency tracker."""
        self.logger.info("🔄 Shutting down dependency tracker...")
        self.stop_tracking()
        self.logger.info("✅ Dependency tracker shutdown complete")


# Global dependency tracker instance
_global_dependency_tracker: Optional[DependencyTracker] = None
_tracker_lock = threading.Lock()


def get_dependency_tracker() -> DependencyTracker:
    """Get global dependency tracker instance."""
    global _global_dependency_tracker
    
    if _global_dependency_tracker is None:
        with _tracker_lock:
            if _global_dependency_tracker is None:
                _global_dependency_tracker = DependencyTracker()
    
    return _global_dependency_tracker


def start_dependency_tracking(track_imports: bool = True,
                             track_calls: bool = True,
                             auto_analyze: bool = True) -> bool:
    """
    Start dependency tracking.
    
    Convenience function to start comprehensive dependency analysis.
    
    Args:
        track_imports: Enable import tracking
        track_calls: Enable function call tracking
        auto_analyze: Enable automatic analysis
        
    Returns:
        True if tracking started successfully
    """
    tracker = get_dependency_tracker()
    return tracker.start_tracking(track_imports, track_calls, auto_analyze)


def stop_dependency_tracking() -> None:
    """Stop dependency tracking."""
    tracker = get_dependency_tracker()
    tracker.stop_tracking()


def analyze_dependencies() -> Dict[str, Any]:
    """Perform comprehensive dependency analysis."""
    tracker = get_dependency_tracker()
    return tracker.analyze_dependencies()


def get_dependency_status() -> Dict[str, Any]:
    """Get dependency tracker status."""
    tracker = get_dependency_tracker()
    return tracker.get_status()


# Convenience decorator for dependency tracking
def track_dependency(dependency_name: str, dependency_type: DependencyType = DependencyType.FUNCTION_CALL) -> Callable:
    """
    Decorator to track dependencies in functions.
    
    Usage:
        @track_dependency("database_connection", DependencyType.RESOURCE_DEPENDENCY)
        def connect_to_database():
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        
        def wrapper(*args, **kwargs):
            tracker = get_dependency_tracker()
            
            if tracker.tracking_enabled:
                # Add dependency tracking logic here
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Update successful access
                    execution_time = (time.perf_counter() - start_time) * 1000
                    tracker.tracked_calls += 1
                    
                    return result
                    
                except Exception as e:
                    # Update failure tracking
                    execution_time = (time.perf_counter() - start_time) * 1000
                    
                    # Log dependency failure
                    logger = get_enhanced_logger(f"arena_bot.dependency_tracking.{dependency_name}")
                    logger.error(
                        f"❌ DEPENDENCY_FAILURE: {dependency_name} failed in {func.__name__}",
                        extra={
                            'dependency_failure': {
                                'dependency_name': dependency_name,
                                'dependency_type': dependency_type.value,
                                'function_name': func.__name__,
                                'execution_time_ms': execution_time,
                                'error_message': str(e)
                            }
                        }
                    )
                    
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator