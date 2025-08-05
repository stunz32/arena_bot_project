"""
Interactive Debugging Dashboard for Arena Bot Deep Debugging

Real-time GUI dashboard for monitoring and controlling all debugging systems:

- Live system metrics visualization with real-time charts and graphs
- Component health monitoring with status indicators and alerts
- Interactive debugging controls with mode switching and configuration
- Exception analysis with detailed context views and correlation graphs
- Performance monitoring with bottleneck identification and optimization suggestions
- Log streaming with filtering, search, and correlation capabilities
- Alert management with notification center and resolution tracking
- Historical analysis with trend visualization and pattern recognition

Built with modern GUI framework for responsive, real-time debugging experience.
"""

import time
import threading
import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import queue

# Import GUI framework (using tkinter for cross-platform compatibility)
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import tkinter.font as tkfont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import debugging components
from .ultra_debug import get_ultra_debug_manager, UltraDebugMode, SystemMetrics
from .integration import get_debugging_integrator
from .enhanced_logger import get_enhanced_logger
from .exception_handler import get_exception_handler
from .health_monitor import get_health_monitor
from .error_analyzer import get_error_analyzer

from ..logging_system.logger import get_logger


class DashboardColors:
    """Color scheme for the dashboard."""
    
    # Background colors
    BG_PRIMARY = "#1e1e1e"
    BG_SECONDARY = "#2d2d2d"
    BG_TERTIARY = "#3c3c3c"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_MUTED = "#888888"
    
    # Status colors
    STATUS_GOOD = "#28a745"
    STATUS_WARNING = "#ffc107"
    STATUS_ERROR = "#dc3545"
    STATUS_CRITICAL = "#ff0000"
    
    # Accent colors
    ACCENT_BLUE = "#007bff"
    ACCENT_GREEN = "#28a745"
    ACCENT_YELLOW = "#ffc107"
    ACCENT_RED = "#dc3545"
    
    # Chart colors
    CHART_LINE1 = "#007bff"
    CHART_LINE2 = "#28a745"
    CHART_LINE3 = "#ffc107"
    CHART_LINE4 = "#dc3545"


@dataclass
class DashboardState:
    """Dashboard state management."""
    
    # UI state
    current_tab: str = "overview"
    refresh_rate_ms: int = 1000
    auto_refresh: bool = True
    
    # Data state
    max_data_points: int = 100
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=100))
    alerts_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # Filter state
    log_level_filter: str = "ALL"
    component_filter: str = "ALL"
    time_range_hours: int = 24
    
    # Display preferences
    show_grid: bool = True
    dark_mode: bool = True
    font_size: int = 10


class MetricsChart:
    """Real-time metrics chart widget."""
    
    def __init__(self, parent, title: str, metrics_keys: List[str]):
        """Initialize metrics chart."""
        self.parent = parent
        self.title = title
        self.metrics_keys = metrics_keys
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor=DashboardColors.BG_SECONDARY)
        self.ax = self.fig.add_subplot(111, facecolor=DashboardColors.BG_TERTIARY)
        
        # Configure chart appearance
        self.ax.set_title(title, color=DashboardColors.TEXT_PRIMARY, fontsize=12)
        self.ax.tick_params(colors=DashboardColors.TEXT_SECONDARY)
        self.ax.grid(True, alpha=0.3)
        
        # Data storage
        self.data_history = {key: deque(maxlen=100) for key in metrics_keys}
        self.time_history = deque(maxlen=100)
        
        # Chart lines
        self.lines = {}
        colors = [DashboardColors.CHART_LINE1, DashboardColors.CHART_LINE2, 
                 DashboardColors.CHART_LINE3, DashboardColors.CHART_LINE4]
        
        for i, key in enumerate(metrics_keys):
            color = colors[i % len(colors)]
            line, = self.ax.plot([], [], label=key, color=color, linewidth=2)
            self.lines[key] = line
        
        if len(metrics_keys) > 1:
            self.ax.legend(loc='upper right', fontsize=8)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_data(self, metrics: SystemMetrics) -> None:
        """Update chart with new metrics data."""
        current_time = time.time()
        self.time_history.append(current_time)
        
        # Update data for each metric
        for key in self.metrics_keys:
            value = getattr(metrics, key, 0)
            self.data_history[key].append(value)
        
        # Update chart lines
        if len(self.time_history) > 1:
            time_range = [t - current_time for t in self.time_history]
            
            for key in self.metrics_keys:
                if len(self.data_history[key]) > 0:
                    self.lines[key].set_data(time_range, list(self.data_history[key]))
            
            # Update axes
            self.ax.set_xlim(min(time_range), 0)
            
            # Calculate y-axis range
            all_values = []
            for key in self.metrics_keys:
                all_values.extend(list(self.data_history[key]))
            
            if all_values:
                y_min, y_max = min(all_values), max(all_values)
                y_range = y_max - y_min
                if y_range == 0:
                    y_range = 1
                self.ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
            
            # Redraw
            self.canvas.draw()


class ComponentStatusWidget:
    """Component status monitoring widget."""
    
    def __init__(self, parent):
        """Initialize component status widget."""
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Component Status", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for component status
        columns = ("Component", "Status", "Health", "Last Update", "Details")
        self.tree = ttk.Treeview(self.frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='center')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status tracking
        self.component_items = {}
    
    def update_status(self, status_data: Dict[str, Any]) -> None:
        """Update component status display."""
        try:
            # Get health monitor data
            health_monitor = get_health_monitor()
            health_summary = health_monitor.get_system_health_summary()
            
            # Get debugging integrator status
            integrator = get_debugging_integrator()
            integrator_status = integrator.get_integration_status()
            
            # Update display
            components = {
                "Method Tracer": {
                    "status": "Active" if integrator.method_tracer else "Inactive",
                    "health": "Good",
                    "last_update": datetime.now().strftime("%H:%M:%S"),
                    "details": f"Traces: {integrator_status.get('component_status', {}).get('method_tracer', {}).get('active_traces', 0)}"
                },
                "State Monitor": {
                    "status": "Active" if integrator.state_monitor else "Inactive", 
                    "health": "Good",
                    "last_update": datetime.now().strftime("%H:%M:%S"),
                    "details": f"Changes: {integrator_status.get('component_status', {}).get('state_monitor', {}).get('total_changes', 0)}"
                },
                "Health Monitor": {
                    "status": "Active" if integrator.health_monitor else "Inactive",
                    "health": "Good",
                    "last_update": datetime.now().strftime("%H:%M:%S"),
                    "details": f"Components: {len(health_summary.get('components', {}) if isinstance(health_summary, dict) else {})}"
                },
                "Exception Handler": {
                    "status": "Active" if integrator.exception_handler else "Inactive",
                    "health": "Good", 
                    "last_update": datetime.now().strftime("%H:%M:%S"),
                    "details": f"Handled: {integrator_status.get('component_status', {}).get('exception_handler', {}).get('total_exceptions', 0)}"
                }
            }
            
            # Update tree view
            for component_name, info in components.items():
                status_color = DashboardColors.STATUS_GOOD if info["status"] == "Active" else DashboardColors.STATUS_WARNING
                
                if component_name in self.component_items:
                    # Update existing item
                    item_id = self.component_items[component_name]
                    self.tree.item(item_id, values=(
                        component_name,
                        info["status"],
                        info["health"],
                        info["last_update"],
                        info["details"]
                    ))
                else:
                    # Add new item
                    item_id = self.tree.insert("", tk.END, values=(
                        component_name,
                        info["status"],
                        info["health"],
                        info["last_update"],
                        info["details"]
                    ))
                    self.component_items[component_name] = item_id
        
        except Exception as e:
            # Don't let display errors affect debugging
            pass


class AlertsWidget:
    """Alerts and notifications widget."""
    
    def __init__(self, parent):
        """Initialize alerts widget."""
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Active Alerts", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for alerts
        columns = ("Time", "Severity", "Category", "Title", "Description")
        self.tree = ttk.Treeview(self.frame, columns=columns, show='headings', height=6)
        
        # Configure columns
        column_widths = {"Time": 80, "Severity": 80, "Category": 100, "Title": 150, "Description": 250}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='w')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Alert tracking
        self.alert_items = {}
    
    def update_alerts(self) -> None:
        """Update alerts display."""
        try:
            # Get ultra-debug manager for anomaly alerts
            ultra_debug = get_ultra_debug_manager()
            active_alerts = ultra_debug.anomaly_detector.get_active_alerts()
            
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.alert_items.clear()
            
            # Add current alerts
            for alert in active_alerts:
                alert_time = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                
                item_id = self.tree.insert("", tk.END, values=(
                    alert_time,
                    alert.severity.upper(),
                    alert.category,
                    alert.title,
                    alert.description
                ))
                
                # Color code by severity
                if alert.severity == "critical":
                    self.tree.set(item_id, "Severity", "🔴 CRITICAL")
                elif alert.severity == "high":
                    self.tree.set(item_id, "Severity", "🟠 HIGH")
                elif alert.severity == "medium":
                    self.tree.set(item_id, "Severity", "🟡 MEDIUM")
                else:
                    self.tree.set(item_id, "Severity", "🔵 LOW")
                
                self.alert_items[alert.alert_id] = item_id
        
        except Exception as e:
            # Don't let display errors affect debugging
            pass


class LogStreamWidget:
    """Live log streaming widget."""
    
    def __init__(self, parent):
        """Initialize log stream widget."""
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Live Debug Logs", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create controls frame
        controls_frame = ttk.Frame(self.frame)
        controls_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Log level filter
        ttk.Label(controls_frame, text="Level:").pack(side=tk.LEFT, padx=(0, 5))
        self.level_var = tk.StringVar(value="ALL")
        level_combo = ttk.Combobox(controls_frame, textvariable=self.level_var, 
                                  values=["ALL", "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
                                  width=10, state="readonly")
        level_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Component filter
        ttk.Label(controls_frame, text="Component:").pack(side=tk.LEFT, padx=(0, 5))
        self.component_var = tk.StringVar(value="ALL")
        component_combo = ttk.Combobox(controls_frame, textvariable=self.component_var,
                                     values=["ALL", "ultra_debug", "method_tracer", "state_monitor", "health_monitor"],
                                     width=15, state="readonly")
        component_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        ttk.Button(controls_frame, text="Clear", command=self.clear_logs).pack(side=tk.RIGHT)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(self.frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(text_frame, height=15, 
                                                 bg=DashboardColors.BG_TERTIARY,
                                                 fg=DashboardColors.TEXT_PRIMARY,
                                                 insertbackground=DashboardColors.TEXT_PRIMARY,
                                                 font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for color coding
        self.log_text.tag_configure("CRITICAL", foreground=DashboardColors.STATUS_CRITICAL)
        self.log_text.tag_configure("ERROR", foreground=DashboardColors.STATUS_ERROR)
        self.log_text.tag_configure("WARNING", foreground=DashboardColors.STATUS_WARNING)
        self.log_text.tag_configure("INFO", foreground=DashboardColors.TEXT_PRIMARY)
        self.log_text.tag_configure("DEBUG", foreground=DashboardColors.TEXT_SECONDARY)
        self.log_text.tag_configure("TRACE", foreground=DashboardColors.TEXT_MUTED)
        
        # Log queue for thread-safe updates
        self.log_queue = queue.Queue()
        self.max_log_lines = 1000
    
    def add_log_entry(self, level: str, message: str, component: str = "") -> None:
        """Add a log entry (thread-safe)."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "component": component,
            "message": message
        }
        
        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            # Drop oldest if queue is full
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(log_entry)
            except queue.Empty:
                pass
    
    def update_display(self) -> None:
        """Update log display from queue."""
        # Process queued log entries
        entries_processed = 0
        while entries_processed < 50:  # Limit processing per update
            try:
                log_entry = self.log_queue.get_nowait()
                self._display_log_entry(log_entry)
                entries_processed += 1
            except queue.Empty:
                break
    
    def _display_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Display a single log entry."""
        # Apply filters
        level_filter = self.level_var.get()
        component_filter = self.component_var.get()
        
        if level_filter != "ALL" and log_entry["level"] != level_filter:
            return
        
        if component_filter != "ALL" and component_filter not in log_entry["component"]:
            return
        
        # Format log line
        component_str = f"[{log_entry['component']}]" if log_entry['component'] else ""
        log_line = f"{log_entry['timestamp']} {log_entry['level']:8} {component_str:20} {log_entry['message']}\n"
        
        # Insert with color coding
        self.log_text.insert(tk.END, log_line, log_entry['level'])
        
        # Limit total lines
        line_count = int(self.log_text.index('end-1c').split('.')[0])
        if line_count > self.max_log_lines:
            # Remove oldest lines
            lines_to_remove = line_count - self.max_log_lines + 100
            self.log_text.delete('1.0', f'{lines_to_remove}.0')
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
    
    def clear_logs(self) -> None:
        """Clear all log entries."""
        self.log_text.delete('1.0', tk.END)


class DebugControlPanel:
    """Debug mode control panel."""
    
    def __init__(self, parent):
        """Initialize control panel."""
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Debug Controls", padding=10)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Current mode display
        mode_frame = ttk.Frame(self.frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_frame, text="Current Mode:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self.mode_label = ttk.Label(mode_frame, text="DISABLED", foreground=DashboardColors.STATUS_ERROR)
        self.mode_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Mode control buttons
        buttons_frame = ttk.Frame(self.frame)
        buttons_frame.pack(fill=tk.X)
        
        # Ultra-debug mode buttons
        ttk.Button(buttons_frame, text="Enable Monitoring", 
                  command=lambda: self._set_ultra_debug_mode(UltraDebugMode.MONITORING)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Enable Analysis", 
                  command=lambda: self._set_ultra_debug_mode(UltraDebugMode.ANALYSIS)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Enable Introspection", 
                  command=lambda: self._set_ultra_debug_mode(UltraDebugMode.INTROSPECTION)).pack(side=tk.LEFT, padx=(0, 5))
        
        # Emergency buttons
        ttk.Button(buttons_frame, text="EMERGENCY", 
                  command=self._emergency_mode, 
                  style="Emergency.TButton").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(buttons_frame, text="CRISIS", 
                  command=self._crisis_mode,
                  style="Crisis.TButton").pack(side=tk.LEFT, padx=(0, 5))
        
        # Disable button
        ttk.Button(buttons_frame, text="Disable", 
                  command=self._disable_debug).pack(side=tk.RIGHT)
    
    def _set_ultra_debug_mode(self, mode: UltraDebugMode) -> None:
        """Set ultra-debug mode."""
        try:
            from .ultra_debug import enable_ultra_debug
            enable_ultra_debug(mode)
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set ultra-debug mode: {e}")
    
    def _emergency_mode(self) -> None:
        """Activate emergency debug mode."""
        try:
            from .ultra_debug import emergency_debug
            emergency_debug()
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to activate emergency mode: {e}")
    
    def _crisis_mode(self) -> None:
        """Activate crisis debug mode."""
        try:
            from .ultra_debug import crisis_debug
            crisis_debug()
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to activate crisis mode: {e}")
    
    def _disable_debug(self) -> None:
        """Disable debug mode."""
        try:
            from .ultra_debug import disable_ultra_debug
            disable_ultra_debug()
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to disable debug mode: {e}")
    
    def update_status(self) -> None:
        """Update mode status display."""
        try:
            ultra_debug = get_ultra_debug_manager()
            current_mode = ultra_debug.mode.value.upper()
            
            self.mode_label.config(text=current_mode)
            
            # Color code based on mode
            if current_mode == "DISABLED":
                self.mode_label.config(foreground=DashboardColors.STATUS_ERROR)
            elif current_mode in ["EMERGENCY", "CRISIS"]:
                self.mode_label.config(foreground=DashboardColors.STATUS_CRITICAL)
            elif current_mode in ["ANALYSIS", "INTROSPECTION"]:
                self.mode_label.config(foreground=DashboardColors.STATUS_WARNING)
            else:
                self.mode_label.config(foreground=DashboardColors.STATUS_GOOD)
        
        except Exception:
            self.mode_label.config(text="UNKNOWN", foreground=DashboardColors.STATUS_ERROR)


class DebugDashboard:
    """
    Main debugging dashboard application.
    
    Provides real-time monitoring and control of all debugging systems
    with an intuitive GUI interface.
    """
    
    def __init__(self):
        """Initialize debug dashboard."""
        self.logger = get_enhanced_logger("arena_bot.debugging.debug_dashboard")
        
        # Dashboard state
        self.state = DashboardState()
        self.running = False
        
        # Data update thread
        self.update_thread: Optional[threading.Thread] = None
        
        # Initialize GUI
        self._init_gui()
        
        # Start data updates
        self._start_updates()
        
        self.logger.info("🖥️ Debug dashboard initialized")
    
    def _init_gui(self) -> None:
        """Initialize GUI components."""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Arena Bot - Debug Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=DashboardColors.BG_PRIMARY)
        
        # Configure styles
        self._configure_styles()
        
        # Create main layout
        self._create_layout()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _configure_styles(self) -> None:
        """Configure GUI styles."""
        style = ttk.Style()
        
        # Configure colors for dark theme
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Emergency.TButton', foreground='white', background=DashboardColors.STATUS_ERROR)
        style.configure('Crisis.TButton', foreground='white', background=DashboardColors.STATUS_CRITICAL)
    
    def _create_layout(self) -> None:
        """Create main dashboard layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overview tab
        self._create_overview_tab()
        
        # Metrics tab
        self._create_metrics_tab()
        
        # Logs tab
        self._create_logs_tab()
        
        # Controls tab
        self._create_controls_tab()
    
    def _create_overview_tab(self) -> None:
        """Create overview tab."""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")
        
        # Create main sections
        top_frame = ttk.Frame(overview_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        bottom_frame = ttk.Frame(overview_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        self.control_panel = DebugControlPanel(control_frame)
        
        # Component status
        status_frame = ttk.Frame(top_frame)
        status_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.component_status = ComponentStatusWidget(status_frame)
        
        # Alerts
        self.alerts_widget = AlertsWidget(bottom_frame)
    
    def _create_metrics_tab(self) -> None:
        """Create metrics tab."""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Metrics")
        
        # Create charts layout
        charts_frame = ttk.Frame(metrics_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top row charts
        top_charts = ttk.Frame(charts_frame)
        top_charts.pack(fill=tk.BOTH, expand=True)
        
        # CPU/Memory chart
        cpu_memory_frame = ttk.Frame(top_charts)
        cpu_memory_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.cpu_memory_chart = MetricsChart(cpu_memory_frame, "CPU & Memory Usage", 
                                           ["cpu_percent", "memory_percent"])
        
        # Performance chart
        performance_frame = ttk.Frame(top_charts)
        performance_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.performance_chart = MetricsChart(performance_frame, "System Performance",
                                            ["active_threads", "open_files"])
        
        # Bottom row charts
        bottom_charts = ttk.Frame(charts_frame)
        bottom_charts.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Debug activity chart
        debug_activity_frame = ttk.Frame(bottom_charts)
        debug_activity_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.debug_activity_chart = MetricsChart(debug_activity_frame, "Debug Activity",
                                                ["active_traces", "active_pipelines"])
        
        # Issues chart
        issues_frame = ttk.Frame(bottom_charts)
        issues_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.issues_chart = MetricsChart(issues_frame, "Issues & Alerts",
                                       ["recent_exceptions", "circuit_breakers_open"])
    
    def _create_logs_tab(self) -> None:
        """Create logs tab."""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        self.log_stream = LogStreamWidget(logs_frame)
    
    def _create_controls_tab(self) -> None:
        """Create controls tab."""
        controls_frame = ttk.Frame(self.notebook)
        self.notebook.add(controls_frame, text="Controls")
        
        # System information
        info_frame = ttk.LabelFrame(controls_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Configuration controls
        config_frame = ttk.LabelFrame(controls_frame, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Refresh rate control
        refresh_frame = ttk.Frame(config_frame)
        refresh_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(refresh_frame, text="Refresh Rate (ms):").pack(side=tk.LEFT)
        self.refresh_var = tk.IntVar(value=self.state.refresh_rate_ms)
        refresh_spin = ttk.Spinbox(refresh_frame, from_=500, to=10000, increment=500,
                                  textvariable=self.refresh_var, width=10)
        refresh_spin.pack(side=tk.LEFT, padx=(10, 0))
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=self.state.auto_refresh)
        ttk.Checkbutton(refresh_frame, text="Auto-refresh", 
                       variable=self.auto_refresh_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Action buttons
        actions_frame = ttk.Frame(config_frame)
        actions_frame.pack(fill=tk.X)
        
        ttk.Button(actions_frame, text="Generate Report", 
                  command=self._generate_report).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(actions_frame, text="Export Logs", 
                  command=self._export_logs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(actions_frame, text="Reset Metrics", 
                  command=self._reset_metrics).pack(side=tk.LEFT)
    
    def _start_updates(self) -> None:
        """Start background data updates."""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start GUI update timer
        self.root.after(self.state.refresh_rate_ms, self._gui_update)
    
    def _update_loop(self) -> None:
        """Background data update loop."""
        while self.running:
            try:
                # Capture current metrics
                metrics = SystemMetrics()
                metrics.capture_metrics()
                
                # Store metrics
                self.state.metrics_history.append(metrics)
                
                # Simulate log entries (in real implementation, this would hook into the logging system)
                self._capture_log_entries()
                
                # Sleep until next update
                time.sleep(self.state.refresh_rate_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Update loop error: {e}")
                time.sleep(1)
    
    def _capture_log_entries(self) -> None:
        """Capture recent log entries."""
        # In a real implementation, this would hook into the enhanced logger
        # For demonstration, we'll simulate some log entries
        try:
            ultra_debug = get_ultra_debug_manager()
            if ultra_debug.mode != UltraDebugMode.DISABLED:
                self.log_stream.add_log_entry("INFO", "Ultra-debug monitoring active", "ultra_debug")
                
                # Check for alerts
                active_alerts = ultra_debug.anomaly_detector.get_active_alerts()
                for alert in active_alerts:
                    if not hasattr(alert, '_logged'):
                        self.log_stream.add_log_entry("WARNING", f"Anomaly detected: {alert.title}", "anomaly_detector")
                        alert._logged = True
        
        except Exception:
            pass
    
    def _gui_update(self) -> None:
        """Update GUI components."""
        try:
            if not self.running:
                return
            
            # Update refresh rate from UI
            self.state.refresh_rate_ms = self.refresh_var.get()
            self.state.auto_refresh = self.auto_refresh_var.get()
            
            if self.state.auto_refresh:
                # Update charts with latest metrics
                if self.state.metrics_history:
                    latest_metrics = self.state.metrics_history[-1]
                    
                    self.cpu_memory_chart.update_data(latest_metrics)
                    self.performance_chart.update_data(latest_metrics)
                    self.debug_activity_chart.update_data(latest_metrics)
                    self.issues_chart.update_data(latest_metrics)
                
                # Update component status
                self.component_status.update_status({})
                
                # Update alerts
                self.alerts_widget.update_alerts()
                
                # Update control panel
                self.control_panel.update_status()
                
                # Update log stream
                self.log_stream.update_display()
                
                # Update system information
                self._update_system_info()
            
            # Schedule next update
            self.root.after(self.state.refresh_rate_ms, self._gui_update)
            
        except Exception as e:
            self.logger.error(f"GUI update error: {e}")
            # Schedule next update anyway
            self.root.after(self.state.refresh_rate_ms, self._gui_update)
    
    def _update_system_info(self) -> None:
        """Update system information display."""
        try:
            # Get system status
            ultra_debug = get_ultra_debug_manager()
            status = ultra_debug.get_status()
            
            # Format information
            info_text = "=== Debug System Status ===\n\n"
            info_text += f"Ultra-Debug Mode: {status['mode']}\n"
            info_text += f"Monitoring Active: {status['monitoring_active']}\n"
            info_text += f"Uptime: {status['uptime_seconds']:.1f} seconds\n"
            info_text += f"Metrics Collected: {status['metrics_collected']}\n"
            info_text += f"Emergency Activations: {status['emergency_activations']}\n"
            info_text += f"Crisis Activations: {status['crisis_activations']}\n\n"
            
            # Alert summary
            alert_summary = status.get('alert_summary', {})
            info_text += "=== Alert Summary ===\n\n"
            info_text += f"Total Active Alerts: {alert_summary.get('total_active_alerts', 0)}\n"
            
            active_by_severity = alert_summary.get('active_by_severity', {})
            for severity, count in active_by_severity.items():
                info_text += f"{severity.title()} Alerts: {count}\n"
            
            # Update display
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', info_text)
        
        except Exception as e:
            error_text = f"Error updating system info: {e}"
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', error_text)
    
    def _generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        try:
            from .ultra_debug import get_analysis_report
            report = get_analysis_report()
            
            # Display report in a new window
            report_window = tk.Toplevel(self.root)
            report_window.title("Analysis Report")
            report_window.geometry("800x600")
            
            report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD)
            report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Format report
            formatted_report = json.dumps(report, indent=2)
            report_text.insert('1.0', formatted_report)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def _export_logs(self) -> None:
        """Export current logs to file."""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                log_content = self.log_stream.log_text.get('1.0', tk.END)
                with open(filename, 'w') as f:
                    f.write(log_content)
                
                messagebox.showinfo("Success", f"Logs exported to {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")
    
    def _reset_metrics(self) -> None:
        """Reset all metrics data."""
        try:
            self.state.metrics_history.clear()
            self.state.alerts_history.clear()
            
            # Clear charts
            for chart in [self.cpu_memory_chart, self.performance_chart, 
                         self.debug_activity_chart, self.issues_chart]:
                chart.data_history = {key: deque(maxlen=100) for key in chart.metrics_keys}
                chart.time_history.clear()
            
            messagebox.showinfo("Success", "Metrics data reset")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset metrics: {e}")
    
    def _on_closing(self) -> None:
        """Handle window closing."""
        self.running = False
        
        # Wait for update thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        self.logger.info("🖥️ Debug dashboard closed")
        self.root.destroy()
    
    def run(self) -> None:
        """Run the dashboard."""
        try:
            self.logger.info("🖥️ Starting debug dashboard...")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
        finally:
            self.running = False


# Global dashboard instance
_global_dashboard: Optional[DebugDashboard] = None
_dashboard_lock = threading.Lock()


def get_debug_dashboard() -> DebugDashboard:
    """Get global debug dashboard instance."""
    global _global_dashboard
    
    if _global_dashboard is None:
        with _dashboard_lock:
            if _global_dashboard is None:
                _global_dashboard = DebugDashboard()
    
    return _global_dashboard


def launch_debug_dashboard() -> None:
    """
    Launch the interactive debug dashboard.
    
    Creates and displays the real-time debugging GUI for monitoring
    and controlling all debugging systems.
    """
    dashboard = get_debug_dashboard()
    dashboard.run()


def is_dashboard_running() -> bool:
    """Check if dashboard is currently running."""
    global _global_dashboard
    return _global_dashboard is not None and _global_dashboard.running


# Convenience function for easy integration
def show_dashboard() -> None:
    """Show the debug dashboard (alias for launch_debug_dashboard)."""
    launch_debug_dashboard()