"""
Performance timing utilities and budget enforcement.

This module provides lightweight timing decorators and utilities for measuring
and enforcing per-stage performance budgets across the detection pipeline.
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from threading import local

logger = logging.getLogger(__name__)

# Thread-local storage for timing data
_timing_data = local()


@dataclass
class TimingBudget:
    """Performance budget configuration for a stage."""
    stage_name: str
    budget_ms: float
    warning_threshold: float = 0.8  # Warn at 80% of budget
    critical_threshold: float = 1.0  # Critical at 100% of budget


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    stage_name: str
    duration_ms: float
    budget_ms: Optional[float] = None
    exceeded_budget: bool = False
    percentage_of_budget: Optional[float] = None
    status: str = "ok"  # ok, warning, critical, skipped


class PerformanceTracker:
    """
    Performance tracking and budget enforcement system.
    
    Tracks per-stage timing and validates against performance budgets.
    """
    
    # Default performance budgets (ms)
    DEFAULT_BUDGETS = {
        'coordinates': 60,
        'eligibility_filter': 20,
        'histogram_match': 150,
        'template_validation': 80,
        'ai_advisor': 150,
        'ui_render': 40,
        'total': 500
    }
    
    def __init__(self, budgets: Optional[Dict[str, float]] = None):
        """
        Initialize performance tracker.
        
        Args:
            budgets: Custom budget configuration, or None for defaults
        """
        self.budgets = budgets or self.DEFAULT_BUDGETS.copy()
        self.results: Dict[str, TimingResult] = {}
        self.start_time: Optional[float] = None
        self.logger = logger
        
    def get_timing_data(self):
        """Get or create timing data for current thread."""
        if not hasattr(_timing_data, 'tracker'):
            _timing_data.tracker = {}
        return _timing_data.tracker
    
    def set_budget(self, stage_name: str, budget_ms: float):
        """Set budget for a specific stage."""
        self.budgets[stage_name] = budget_ms
        self.logger.debug(f"Set budget for {stage_name}: {budget_ms}ms")
    
    def start_session(self):
        """Start a new timing session."""
        self.start_time = time.time()
        self.results.clear()
        timing_data = self.get_timing_data()
        timing_data.clear()
        self.logger.debug("Started performance tracking session")
    
    def record_stage(self, stage_name: str, duration_ms: float, skipped: bool = False):
        """
        Record timing for a stage.
        
        Args:
            stage_name: Name of the stage
            duration_ms: Duration in milliseconds
            skipped: Whether stage was skipped (e.g., sidecar mode)
        """
        budget_ms = self.budgets.get(stage_name)
        
        if skipped:
            result = TimingResult(
                stage_name=stage_name,
                duration_ms=0.0,
                budget_ms=budget_ms,
                exceeded_budget=False,
                percentage_of_budget=0.0,
                status="skipped"
            )
        else:
            exceeded_budget = budget_ms is not None and duration_ms > budget_ms
            percentage = (duration_ms / budget_ms * 100) if budget_ms else None
            
            # Determine status
            if percentage is None:
                status = "ok"
            elif percentage >= 100:
                status = "critical"
            elif percentage >= 80:
                status = "warning"
            else:
                status = "ok"
            
            result = TimingResult(
                stage_name=stage_name,
                duration_ms=duration_ms,
                budget_ms=budget_ms,
                exceeded_budget=exceeded_budget,
                percentage_of_budget=percentage,
                status=status
            )
        
        self.results[stage_name] = result
        
        # Log result
        if skipped:
            self.logger.debug(f"⏭️  {stage_name}: skipped")
        elif exceeded_budget:
            self.logger.warning(f"🚨 {stage_name}: {duration_ms:.1f}ms "
                               f"(>{budget_ms:.1f}ms budget, {percentage:.1f}%)")
        elif result.status == "warning":
            self.logger.info(f"⚠️  {stage_name}: {duration_ms:.1f}ms "
                            f"({percentage:.1f}% of budget)")
        else:
            self.logger.debug(f"✅ {stage_name}: {duration_ms:.1f}ms")
    
    def get_total_duration(self) -> float:
        """Get total session duration in milliseconds."""
        if self.start_time is None:
            return 0.0
        
        # Include all non-skipped stages
        total = sum(r.duration_ms for r in self.results.values() 
                   if r.status != "skipped")
        return total
    
    def check_total_budget(self) -> TimingResult:
        """Check total session against total budget."""
        total_duration = self.get_total_duration()
        total_budget = self.budgets.get('total')
        
        return TimingResult(
            stage_name='total',
            duration_ms=total_duration,
            budget_ms=total_budget,
            exceeded_budget=(total_budget is not None and total_duration > total_budget),
            percentage_of_budget=((total_duration / total_budget * 100) 
                                 if total_budget else None),
            status=("critical" if total_budget and total_duration > total_budget else "ok")
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary with all results."""
        total_result = self.check_total_budget()
        
        return {
            'stage_timings': {name: result.duration_ms for name, result in self.results.items()},
            'stage_statuses': {name: result.status for name, result in self.results.items()},
            'budget_compliance': {
                name: not result.exceeded_budget for name, result in self.results.items()
            },
            'total_ms': total_result.duration_ms,
            'total_budget_ms': total_result.budget_ms,
            'total_exceeded': total_result.exceeded_budget,
            'summary': self._generate_summary_text()
        }
    
    def _generate_summary_text(self) -> str:
        """Generate human-readable summary text."""
        total_result = self.check_total_budget()
        stage_count = len([r for r in self.results.values() if r.status != "skipped"])
        skipped_count = len([r for r in self.results.values() if r.status == "skipped"])
        failed_count = len([r for r in self.results.values() if r.exceeded_budget])
        
        summary = f"{stage_count} stages, {total_result.duration_ms:.1f}ms total"
        
        if total_result.budget_ms:
            summary += f" ({total_result.percentage_of_budget:.1f}% of {total_result.budget_ms:.1f}ms budget)"
            
        if skipped_count > 0:
            summary += f", {skipped_count} skipped"
            
        if failed_count > 0:
            summary += f", {failed_count} over budget"
            
        return summary


def timing_decorator(stage_name: str, budget_ms: Optional[float] = None, 
                    tracker: Optional[PerformanceTracker] = None):
    """
    Decorator for automatic stage timing.
    
    Args:
        stage_name: Name of the stage being timed
        budget_ms: Optional budget override
        tracker: Optional tracker instance (uses thread-local if None)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create tracker
            if tracker is None:
                timing_data = _timing_data.__dict__.setdefault('default_tracker', 
                                                              PerformanceTracker())
                active_tracker = timing_data
            else:
                active_tracker = tracker
            
            # Override budget if specified
            if budget_ms is not None:
                active_tracker.set_budget(stage_name, budget_ms)
            
            # Time the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                active_tracker.record_stage(stage_name, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                active_tracker.record_stage(stage_name, duration_ms)
                logger.error(f"Stage {stage_name} failed after {duration_ms:.1f}ms: {e}")
                raise
                
        return wrapper
    return decorator


@contextmanager
def timed_stage(stage_name: str, tracker: Optional[PerformanceTracker] = None, 
               skipped: bool = False):
    """
    Context manager for timing a code block.
    
    Args:
        stage_name: Name of the stage
        tracker: Performance tracker instance
        skipped: Whether to mark as skipped
    """
    if tracker is None:
        timing_data = _timing_data.__dict__.setdefault('default_tracker', 
                                                      PerformanceTracker())
        active_tracker = timing_data
    else:
        active_tracker = tracker
    
    if skipped:
        active_tracker.record_stage(stage_name, 0.0, skipped=True)
        yield
    else:
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            active_tracker.record_stage(stage_name, duration_ms)


def get_default_tracker() -> PerformanceTracker:
    """Get the default thread-local performance tracker."""
    return _timing_data.__dict__.setdefault('default_tracker', PerformanceTracker())


def reset_default_tracker():
    """Reset the default thread-local performance tracker."""
    if hasattr(_timing_data, 'default_tracker'):
        delattr(_timing_data, 'default_tracker')


def format_timing_output(tracker: PerformanceTracker, detailed: bool = False) -> str:
    """
    Format timing results for display.
    
    Args:
        tracker: Performance tracker instance
        detailed: Whether to include detailed breakdown
        
    Returns:
        Formatted timing output string
    """
    summary = tracker.get_summary()
    
    output = []
    output.append(f"Performance Summary: {summary['summary']}")
    
    if detailed:
        output.append("\nStage Breakdown:")
        for stage_name, duration in summary['stage_timings'].items():
            if stage_name in tracker.results:
                result = tracker.results[stage_name]
                budget_str = f"/{result.budget_ms:.0f}ms" if result.budget_ms else ""
                status_icon = {
                    'ok': '✅',
                    'warning': '⚠️',
                    'critical': '🚨',
                    'skipped': '⏭️'
                }.get(result.status, '❓')
                
                output.append(f"  {status_icon} {stage_name}: {duration:.1f}ms{budget_str}")
    
    return '\n'.join(output)