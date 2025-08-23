"""
Live diagnostics and observability for Phase 4.

Provides comprehensive diagnostic information for live capture operations:
- Capture backend status and performance
- Window tracking and bounds information
- DPI scale detection and multi-monitor setup
- Click-through and overlay status
- Performance metrics and timing
- Error tracking and troubleshooting guidance
"""

import time
import platform
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import threading

from arena_bot.capture.capture_backend import AdaptiveCaptureManager, CaptureFrame
from arena_bot.utils.window_tracker import WindowTracker, NormalizedBounds
from arena_bot.utils.timing_utils import PerformanceTracker
from arena_bot.utils.debug_dump import DebugDumpManager


logger = logging.getLogger(__name__)


@dataclass
class LiveDiagnosticSnapshot:
    """Snapshot of live system state for diagnostics."""
    timestamp: float
    capture_backend: str
    window_bounds: Optional[Tuple[int, int, int, int]]
    dpi_scale: float
    capture_timing_ms: float
    overlay_timing_ms: float
    click_through_active: bool
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiveDiagnosticsCollector:
    """
    Collects and manages live diagnostic information.
    
    Provides real-time insights into:
    - Capture system performance
    - Window tracking status
    - Overlay functionality
    - System configuration
    - Error patterns
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize diagnostics collector.
        
        Args:
            max_history: Maximum number of diagnostic snapshots to keep
        """
        self.max_history = max_history
        self.snapshots = deque(maxlen=max_history)
        self.performance_tracker = PerformanceTracker()
        self.start_time = time.time()
        self.error_history = deque(maxlen=50)
        
        # System info
        self.system_info = self._collect_system_info()
        
        # Component references (optional)
        self.capture_manager: Optional[AdaptiveCaptureManager] = None
        self.window_tracker: Optional[WindowTracker] = None
        
        self._lock = threading.RLock()
        
        logger.info("Live diagnostics collector initialized")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect static system information."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
        except ImportError:
            memory_info = None
            cpu_count = None
        
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu_count': cpu_count,
            'memory_total_gb': memory_info.total / (1024**3) if memory_info else None,
            'process_start_time': self.start_time
        }
    
    def set_capture_manager(self, capture_manager: AdaptiveCaptureManager):
        """Set capture manager reference for diagnostics."""
        self.capture_manager = capture_manager
    
    def set_window_tracker(self, window_tracker: WindowTracker):
        """Set window tracker reference for diagnostics."""
        self.window_tracker = window_tracker
    
    def record_capture_operation(self, frame: CaptureFrame, operation_duration_ms: float):
        """
        Record a capture operation for diagnostics.
        
        Args:
            frame: Captured frame with metadata
            operation_duration_ms: Total operation duration
        """
        with self._lock:
            # Record performance metrics
            self.performance_tracker.record_stage('frame_capture', frame.capture_duration_ms)
            self.performance_tracker.record_stage('total_operation', operation_duration_ms)
            
            # Get current window bounds
            window_bounds = None
            if self.window_tracker:
                current_bounds = self.window_tracker.get_current_bounds()
                if current_bounds:
                    window_bounds = current_bounds.get_rect()
            
            # Create diagnostic snapshot
            snapshot = LiveDiagnosticSnapshot(
                timestamp=time.time(),
                capture_backend=frame.backend_name,
                window_bounds=window_bounds,
                dpi_scale=frame.dpi_scale,
                capture_timing_ms=frame.capture_duration_ms,
                overlay_timing_ms=0,  # TODO: Track overlay timing separately
                click_through_active=False,  # TODO: Get from overlay
                metadata={
                    'frame_size': frame.image.shape[:2] if frame.image is not None else None,
                    'source_rect': frame.source_rect,
                    'total_operation_ms': operation_duration_ms
                }
            )
            
            self.snapshots.append(snapshot)
            
            logger.debug(f"Recorded capture operation: {frame.backend_name} "
                        f"in {frame.capture_duration_ms:.1f}ms")
    
    def record_error(self, error: str, category: str = "general"):
        """
        Record an error for diagnostic tracking.
        
        Args:
            error: Error message
            category: Error category for classification
        """
        with self._lock:
            error_entry = {
                'timestamp': time.time(),
                'error': error,
                'category': category
            }
            self.error_history.append(error_entry)
            
            logger.warning(f"Diagnostic error recorded: [{category}] {error}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current live system status.
        
        Returns:
            Comprehensive status dictionary
        """
        with self._lock:
            current_time = time.time()
            uptime_seconds = current_time - self.start_time
            
            # Capture system status
            capture_status = {}
            if self.capture_manager:
                try:
                    backend = self.capture_manager.get_active_backend()
                    capture_status = {
                        'backend_name': backend.get_name(),
                        'backend_available': backend.is_available(),
                        'hearthstone_window_found': self.capture_manager.find_hearthstone_window() is not None
                    }
                except Exception as e:
                    capture_status = {'error': str(e)}
            
            # Window tracking status
            window_status = {}
            if self.window_tracker:
                window_status = self.window_tracker.get_status()
            
            # Performance summary
            perf_stats = self.performance_tracker.get_performance_stats()
            
            # Recent errors
            recent_errors = [
                e for e in self.error_history 
                if current_time - e['timestamp'] < 300  # Last 5 minutes
            ]
            
            # Snapshot summary
            snapshot_summary = {}
            if self.snapshots:
                recent_snapshots = [s for s in self.snapshots if current_time - s.timestamp < 60]
                if recent_snapshots:
                    avg_capture_time = sum(s.capture_timing_ms for s in recent_snapshots) / len(recent_snapshots)
                    backends_used = set(s.capture_backend for s in recent_snapshots)
                    
                    snapshot_summary = {
                        'recent_capture_count': len(recent_snapshots),
                        'avg_capture_time_ms': avg_capture_time,
                        'backends_used': list(backends_used),
                        'latest_snapshot': {
                            'backend': recent_snapshots[-1].capture_backend,
                            'window_bounds': recent_snapshots[-1].window_bounds,
                            'dpi_scale': recent_snapshots[-1].dpi_scale,
                            'capture_time_ms': recent_snapshots[-1].capture_timing_ms
                        }
                    }
            
            return {
                'timestamp': current_time,
                'uptime_seconds': uptime_seconds,
                'system_info': self.system_info,
                'capture_status': capture_status,
                'window_tracking_status': window_status,
                'performance_stats': perf_stats,
                'error_summary': {
                    'total_errors': len(self.error_history),
                    'recent_errors': len(recent_errors),
                    'error_categories': list(set(e['category'] for e in recent_errors))
                },
                'snapshot_summary': snapshot_summary,
                'diagnostics_health': 'good' if len(recent_errors) == 0 else 'warning' if len(recent_errors) < 5 else 'critical'
            }
    
    def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """
        Get detailed diagnostic information.
        
        Returns:
            Comprehensive diagnostic report
        """
        with self._lock:
            status = self.get_current_status()
            
            # Add detailed information
            detailed = {
                'status': status,
                'detailed_performance': self._analyze_performance_patterns(),
                'error_analysis': self._analyze_error_patterns(),
                'recommendations': self._generate_recommendations(),
                'troubleshooting_info': self._generate_troubleshooting_info()
            }
            
            return detailed
    
    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns from snapshots."""
        if not self.snapshots:
            return {'no_data': True}
        
        # Analyze capture timing trends
        capture_times = [s.capture_timing_ms for s in self.snapshots]
        backend_performance = {}
        
        for snapshot in self.snapshots:
            backend = snapshot.capture_backend
            if backend not in backend_performance:
                backend_performance[backend] = []
            backend_performance[backend].append(snapshot.capture_timing_ms)
        
        # Calculate statistics
        avg_capture_time = sum(capture_times) / len(capture_times)
        min_capture_time = min(capture_times)
        max_capture_time = max(capture_times)
        
        # Backend comparison
        backend_stats = {}
        for backend, times in backend_performance.items():
            backend_stats[backend] = {
                'count': len(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times)
            }
        
        return {
            'capture_timing': {
                'avg_ms': avg_capture_time,
                'min_ms': min_capture_time,
                'max_ms': max_capture_time,
                'sample_count': len(capture_times)
            },
            'backend_performance': backend_stats,
            'performance_trend': 'stable' if max_capture_time - min_capture_time < 100 else 'variable'
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns from history."""
        if not self.error_history:
            return {'no_errors': True}
        
        # Group errors by category
        error_categories = {}
        for error_entry in self.error_history:
            category = error_entry['category']
            if category not in error_categories:
                error_categories[category] = []
            error_categories[category].append(error_entry)
        
        # Find most common errors
        error_frequency = {}
        for error_entry in self.error_history:
            error_text = error_entry['error']
            error_frequency[error_text] = error_frequency.get(error_text, 0) + 1
        
        most_common = sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_errors': len(self.error_history),
            'error_categories': {cat: len(errors) for cat, errors in error_categories.items()},
            'most_common_errors': most_common,
            'recent_error_rate': len([e for e in self.error_history if time.time() - e['timestamp'] < 300]) / 5  # errors per minute
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic data."""
        recommendations = []
        
        # Performance recommendations
        if self.snapshots:
            avg_capture_time = sum(s.capture_timing_ms for s in self.snapshots) / len(self.snapshots)
            if avg_capture_time > 200:
                recommendations.append("Consider using DXGI capture backend for better performance")
            
            # Check for DPI issues
            dpi_scales = [s.dpi_scale for s in self.snapshots if s.dpi_scale != 1.0]
            if dpi_scales:
                recommendations.append("High DPI detected - ensure overlay positioning accounts for DPI scaling")
        
        # Error-based recommendations
        recent_errors = [e for e in self.error_history if time.time() - e['timestamp'] < 300]
        if len(recent_errors) > 10:
            recommendations.append("High error rate detected - check system resources and Hearthstone window status")
        
        # Window tracking recommendations
        if self.window_tracker:
            status = self.window_tracker.get_status()
            if not status.get('is_tracking', False):
                recommendations.append("Window tracking not active - ensure Hearthstone is running and visible")
        
        return recommendations
    
    def _generate_troubleshooting_info(self) -> Dict[str, Any]:
        """Generate troubleshooting information."""
        info = {
            'common_issues': [
                {
                    'issue': 'Capture fails or returns black screen',
                    'solutions': [
                        'Ensure Hearthstone is not minimized',
                        'Try running as administrator',
                        'Check Windows GPU scheduling settings',
                        'Verify Hearthstone is in windowed or borderless mode'
                    ]
                },
                {
                    'issue': 'Window not found or tracking fails',
                    'solutions': [
                        'Launch Hearthstone before starting arena bot',
                        'Ensure Hearthstone window title contains "Hearthstone"',
                        'Check for multiple Hearthstone instances',
                        'Verify window is not minimized or hidden'
                    ]
                },
                {
                    'issue': 'Poor capture performance',
                    'solutions': [
                        'Close unnecessary applications',
                        'Update graphics drivers',
                        'Use DXGI backend if available',
                        'Reduce Hearthstone graphics settings'
                    ]
                }
            ],
            'environment_check': {
                'windows_platform': platform.system() == 'Windows',
                'python_version_ok': platform.python_version() >= '3.8',
                'recommended_python': '3.9 or higher'
            }
        }
        
        return info
    
    def print_diagnostics(self, detailed: bool = False):
        """
        Print diagnostic information to console.
        
        Args:
            detailed: Whether to print detailed diagnostics
        """
        if detailed:
            diagnostics = self.get_detailed_diagnostics()
            self._print_detailed_diagnostics(diagnostics)
        else:
            status = self.get_current_status()
            self._print_status_summary(status)
    
    def _print_status_summary(self, status: Dict[str, Any]):
        """Print concise status summary."""
        print("\n📊 Live Diagnostics Summary")
        print("=" * 40)
        
        # System info
        sys_info = status['system_info']
        print(f"🖥️  Platform: {sys_info['platform']} {sys_info['platform_version']}")
        print(f"⏱️  Uptime: {status['uptime_seconds']:.1f}s")
        
        # Capture status
        capture_status = status['capture_status']
        if 'backend_name' in capture_status:
            backend_icon = "✅" if capture_status.get('backend_available', False) else "❌"
            print(f"📱 Capture Backend: {backend_icon} {capture_status['backend_name']}")
            
            window_icon = "✅" if capture_status.get('hearthstone_window_found', False) else "❌"
            print(f"🎯 Hearthstone Window: {window_icon}")
        
        # Performance
        perf_stats = status['performance_stats']
        if perf_stats:
            # Find capture timing
            for stage, stats in perf_stats.items():
                if 'capture' in stage and stats['count'] > 0:
                    avg_ms = stats['total_ms'] / stats['count']
                    print(f"⚡ Avg Capture Time: {avg_ms:.1f}ms")
                    break
        
        # Health status
        health = status['diagnostics_health']
        health_icons = {'good': '✅', 'warning': '⚠️', 'critical': '🚨'}
        print(f"🏥 Health: {health_icons.get(health, '❓')} {health.upper()}")
        
        # Recent errors
        error_summary = status['error_summary']
        if error_summary['recent_errors'] > 0:
            print(f"🚨 Recent Errors: {error_summary['recent_errors']}")
    
    def _print_detailed_diagnostics(self, diagnostics: Dict[str, Any]):
        """Print detailed diagnostic information."""
        print("\n📊 Detailed Live Diagnostics")
        print("=" * 50)
        
        # Status summary
        self._print_status_summary(diagnostics['status'])
        
        # Performance analysis
        perf_analysis = diagnostics['detailed_performance']
        if not perf_analysis.get('no_data', False):
            print("\n📈 Performance Analysis:")
            
            capture_timing = perf_analysis['capture_timing']
            print(f"   📸 Capture Timing: {capture_timing['avg_ms']:.1f}ms avg "
                  f"({capture_timing['min_ms']:.1f}-{capture_timing['max_ms']:.1f}ms range)")
            
            backend_perf = perf_analysis['backend_performance']
            for backend, stats in backend_perf.items():
                print(f"   🔧 {backend}: {stats['avg_ms']:.1f}ms avg ({stats['count']} samples)")
        
        # Error analysis
        error_analysis = diagnostics['error_analysis']
        if not error_analysis.get('no_errors', False):
            print("\n🚨 Error Analysis:")
            print(f"   Total Errors: {error_analysis['total_errors']}")
            
            if error_analysis['most_common_errors']:
                print("   Most Common:")
                for error, count in error_analysis['most_common_errors'][:3]:
                    print(f"     • {error[:50]}{'...' if len(error) > 50 else ''} ({count}x)")
        
        # Recommendations
        recommendations = diagnostics['recommendations']
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
    
    def save_diagnostic_report(self, output_path: str):
        """Save detailed diagnostic report to file."""
        diagnostics = self.get_detailed_diagnostics()
        
        import json
        with open(output_path, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        logger.info(f"Diagnostic report saved to: {output_path}")


# Global diagnostics instance for easy access
_global_diagnostics: Optional[LiveDiagnosticsCollector] = None


def get_live_diagnostics() -> LiveDiagnosticsCollector:
    """Get global live diagnostics collector instance."""
    global _global_diagnostics
    if _global_diagnostics is None:
        _global_diagnostics = LiveDiagnosticsCollector()
    return _global_diagnostics


def print_live_diagnostics(detailed: bool = False):
    """Print live diagnostics to console."""
    diagnostics = get_live_diagnostics()
    diagnostics.print_diagnostics(detailed=detailed)