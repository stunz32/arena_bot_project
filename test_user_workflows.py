#!/usr/bin/env python3
"""
🎮 User Workflow Testing System

Tests actual user interactions and workflows that users will encounter.
Simulates real usage scenarios like clicking buttons, analyzing screenshots,
using overlays, and testing error handling.

Usage:
    python3 test_user_workflows.py --all
    python3 test_user_workflows.py --scenario draft_analysis
    python3 test_user_workflows.py --interactive
"""

import sys
import os
import json
import time
import traceback
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    from PIL import Image, ImageTk
    from app.debug_utils import create_debug_snapshot, analyze_layout_issues
    TKINTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GUI imports not available: {e}")
    TKINTER_AVAILABLE = False

try:
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Main GUI not available: {e}")
    GUI_AVAILABLE = False

@dataclass
class UserAction:
    """Represents a user action in a workflow"""
    action_type: str  # "click", "input", "wait", "verify"
    target: str       # Button name, element ID, etc.
    parameters: Dict[str, Any] = None
    expected_result: str = None
    timeout: float = 5.0
    
@dataclass
class WorkflowResult:
    """Result of running a user workflow"""
    workflow_name: str
    success: bool
    actions_completed: int
    total_actions: int
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    screenshots: List[str] = None
    duration: float = 0.0

class UserWorkflowTester:
    """Tests real user workflows and interactions"""
    
    def __init__(self, headless: bool = True, capture_screenshots: bool = True):
        self.headless = headless
        self.capture_screenshots = capture_screenshots
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        self.test_results: List[WorkflowResult] = []
        self.current_gui = None
        self.screenshot_counter = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Centralized logging"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def capture_screenshot(self, name: str = None) -> str:
        """Capture screenshot of current GUI state"""
        if not self.capture_screenshots or not self.current_gui:
            return None
        
        try:
            if name is None:
                name = f"workflow_step_{self.screenshot_counter}"
                self.screenshot_counter += 1
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.artifacts_dir / f"{name}_{timestamp}.png"
            
            # Capture screenshot using debug utils
            if hasattr(self.current_gui, 'root') and self.current_gui.root:
                create_debug_snapshot(self.current_gui.root, name)
                
            return str(screenshot_path)
            
        except Exception as e:
            self.log(f"Screenshot capture failed: {e}", "WARNING")
            return None
    
    def find_widget_by_name(self, root, widget_name: str):
        """Find a widget by name or text"""
        def search_widget(widget, name):
            # Check widget text/name
            try:
                if hasattr(widget, 'cget'):
                    text = widget.cget('text') if 'text' in widget.keys() else ''
                    if name.lower() in text.lower():
                        return widget
            except:
                pass
            
            # Check children
            try:
                for child in widget.winfo_children():
                    result = search_widget(child, name)
                    if result:
                        return result
            except:
                pass
            
            return None
        
        return search_widget(root, widget_name)
    
    def perform_action(self, action: UserAction) -> bool:
        """Perform a single user action"""
        try:
            self.log(f"Performing action: {action.action_type} on {action.target}")
            
            if action.action_type == "click":
                return self.perform_click(action)
            elif action.action_type == "input":
                return self.perform_input(action)
            elif action.action_type == "wait":
                return self.perform_wait(action)
            elif action.action_type == "verify":
                return self.perform_verify(action)
            elif action.action_type == "screenshot":
                self.capture_screenshot(action.target)
                return True
            else:
                self.log(f"Unknown action type: {action.action_type}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Action failed: {e}", "ERROR")
            return False
    
    def perform_click(self, action: UserAction) -> bool:
        """Simulate clicking a button or widget"""
        if not self.current_gui or not hasattr(self.current_gui, 'root'):
            return False
        
        # Find the widget
        target_widget = None
        
        # Try to find by attribute name first
        if hasattr(self.current_gui, action.target):
            target_widget = getattr(self.current_gui, action.target)
        else:
            # Search by text/name
            target_widget = self.find_widget_by_name(self.current_gui.root, action.target)
        
        if not target_widget:
            self.log(f"Widget not found: {action.target}", "ERROR")
            return False
        
        try:
            # Simulate click by invoking the widget
            if hasattr(target_widget, 'invoke'):
                target_widget.invoke()
            elif hasattr(target_widget, 'command'):
                command = target_widget.cget('command')
                if command:
                    command()
            else:
                # Generate click event
                target_widget.event_generate('<Button-1>')
            
            self.log(f"Successfully clicked: {action.target}")
            return True
            
        except Exception as e:
            self.log(f"Click failed on {action.target}: {e}", "ERROR")
            return False
    
    def perform_input(self, action: UserAction) -> bool:
        """Simulate text input"""
        # This would be implemented for text fields
        self.log(f"Text input simulation: {action.target}")
        return True
    
    def perform_wait(self, action: UserAction) -> bool:
        """Wait for specified time or condition"""
        wait_time = action.parameters.get('time', 1.0) if action.parameters else 1.0
        self.log(f"Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        return True
    
    def perform_verify(self, action: UserAction) -> bool:
        """Verify expected state or result"""
        self.log(f"Verifying: {action.expected_result}")
        
        if not self.current_gui:
            return False
        
        # Check for expected text in status or log
        expected = action.expected_result.lower() if action.expected_result else ""
        
        # Check status label
        if hasattr(self.current_gui, 'status_label'):
            try:
                status_text = self.current_gui.status_label.cget('text').lower()
                if expected in status_text:
                    self.log(f"Verification successful: found '{expected}' in status")
                    return True
            except:
                pass
        
        # Check log widget
        if hasattr(self.current_gui, 'log_text_widget'):
            try:
                log_content = self.current_gui.log_text_widget.get('1.0', tk.END).lower()
                if expected in log_content:
                    self.log(f"Verification successful: found '{expected}' in log")
                    return True
            except:
                pass
        
        self.log(f"Verification failed: '{expected}' not found", "WARNING")
        return False
    
    def run_workflow(self, workflow_name: str, actions: List[UserAction]) -> WorkflowResult:
        """Run a complete user workflow"""
        start_time = time.time()
        self.log(f"🎮 Starting workflow: {workflow_name}")
        
        screenshots = []
        actions_completed = 0
        error_message = None
        
        try:
            # Initialize GUI if needed
            if not self.current_gui and GUI_AVAILABLE:
                self.current_gui = IntegratedArenaBotGUI()
                if self.headless:
                    # Set up for headless operation
                    if hasattr(self.current_gui, 'root'):
                        self.current_gui.root.withdraw()
            
            # Take initial screenshot
            initial_screenshot = self.capture_screenshot(f"{workflow_name}_start")
            if initial_screenshot:
                screenshots.append(initial_screenshot)
            
            # Execute each action
            for i, action in enumerate(actions):
                if not self.perform_action(action):
                    error_message = f"Action {i+1} failed: {action.action_type} on {action.target}"
                    break
                
                actions_completed += 1
                
                # Take screenshot after action if requested
                if action.action_type in ["click", "input"]:
                    screenshot = self.capture_screenshot(f"{workflow_name}_step_{i+1}")
                    if screenshot:
                        screenshots.append(screenshot)
                
                # Small delay between actions
                time.sleep(0.1)
            
            # Take final screenshot
            final_screenshot = self.capture_screenshot(f"{workflow_name}_end")
            if final_screenshot:
                screenshots.append(final_screenshot)
            
            success = actions_completed == len(actions)
            duration = time.time() - start_time
            
            result = WorkflowResult(
                workflow_name=workflow_name,
                success=success,
                actions_completed=actions_completed,
                total_actions=len(actions),
                error_message=error_message,
                screenshots=screenshots,
                duration=duration,
                details={
                    "gui_available": GUI_AVAILABLE,
                    "tkinter_available": TKINTER_AVAILABLE,
                    "headless_mode": self.headless
                }
            )
            
            if success:
                self.log(f"✅ Workflow '{workflow_name}' completed successfully ({duration:.2f}s)")
            else:
                self.log(f"❌ Workflow '{workflow_name}' failed: {error_message}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_message = f"Workflow crashed: {str(e)}"
            self.log(f"💥 Workflow '{workflow_name}' crashed: {e}", "ERROR")
            
            return WorkflowResult(
                workflow_name=workflow_name,
                success=False,
                actions_completed=actions_completed,
                total_actions=len(actions),
                error_message=error_message,
                duration=duration,
                details={"traceback": traceback.format_exc()}
            )
    
    # ========================================
    # USER WORKFLOW SCENARIOS
    # ========================================
    
    def scenario_basic_startup(self) -> WorkflowResult:
        """Test basic GUI startup and initialization"""
        actions = [
            UserAction("wait", "startup", {"time": 2.0}),
            UserAction("screenshot", "basic_startup"),
            UserAction("verify", "status", expected_result="ready"),
        ]
        return self.run_workflow("Basic Startup", actions)
    
    def scenario_screenshot_analysis(self) -> WorkflowResult:
        """Test the 'Analyze Screenshot' functionality that users commonly use"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("screenshot", "before_analysis"),
            UserAction("click", "screenshot_btn"),  # Main screenshot button
            UserAction("wait", "analysis", {"time": 3.0}),
            UserAction("screenshot", "during_analysis"),
            UserAction("verify", "analysis", expected_result="analysis"),
            UserAction("wait", "completion", {"time": 2.0}),
            UserAction("screenshot", "after_analysis"),
        ]
        return self.run_workflow("Screenshot Analysis", actions)
    
    def scenario_detection_method_toggle(self) -> WorkflowResult:
        """Test toggling between detection methods"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("click", "ultimate_detection_btn"),
            UserAction("wait", "toggle", {"time": 0.5}),
            UserAction("screenshot", "ultimate_detection_on"),
            UserAction("click", "phash_detection_btn"),
            UserAction("wait", "toggle", {"time": 0.5}),
            UserAction("screenshot", "phash_detection_on"),
            UserAction("click", "arena_priority_btn"),
            UserAction("wait", "toggle", {"time": 0.5}),
            UserAction("screenshot", "arena_priority_on"),
        ]
        return self.run_workflow("Detection Method Toggle", actions)
    
    def scenario_draft_overlay_toggle(self) -> WorkflowResult:
        """Test toggling draft overlay functionality"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("click", "start_btn"),  # Start monitoring
            UserAction("wait", "monitoring", {"time": 2.0}),
            UserAction("screenshot", "monitoring_started"),
            UserAction("verify", "monitoring", expected_result="monitoring"),
            UserAction("click", "start_btn"),  # Stop monitoring
            UserAction("wait", "stopped", {"time": 1.0}),
            UserAction("screenshot", "monitoring_stopped"),
        ]
        return self.run_workflow("Draft Overlay Toggle", actions)
    
    def scenario_coordinate_selection(self) -> WorkflowResult:
        """Test coordinate selection functionality"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("click", "coord_select_btn"),
            UserAction("wait", "coordinate_dialog", {"time": 2.0}),
            UserAction("screenshot", "coordinate_selector"),
            UserAction("click", "coord_mode_btn"),
            UserAction("wait", "toggle", {"time": 0.5}),
            UserAction("screenshot", "custom_coords_mode"),
        ]
        return self.run_workflow("Coordinate Selection", actions)
    
    def scenario_settings_dialog(self) -> WorkflowResult:
        """Test opening and using settings dialog"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("click", "settings_btn"),
            UserAction("wait", "settings_dialog", {"time": 2.0}),
            UserAction("screenshot", "settings_opened"),
            # Settings dialog should open here
        ]
        return self.run_workflow("Settings Dialog", actions)
    
    def scenario_debug_mode_toggle(self) -> WorkflowResult:
        """Test debug mode functionality"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("click", "debug_mode_btn"),
            UserAction("wait", "debug_toggle", {"time": 0.5}),
            UserAction("screenshot", "debug_mode_on"),
            UserAction("click", "verbose_logging_btn"),
            UserAction("wait", "verbose_toggle", {"time": 0.5}),
            UserAction("screenshot", "verbose_logging_on"),
        ]
        return self.run_workflow("Debug Mode Toggle", actions)
    
    def scenario_performance_report(self) -> WorkflowResult:
        """Test performance report functionality"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            UserAction("click", "perf_report_btn"),
            UserAction("wait", "report", {"time": 2.0}),
            UserAction("screenshot", "performance_report"),
        ]
        return self.run_workflow("Performance Report", actions)
    
    def scenario_correction_workflow(self) -> WorkflowResult:
        """Test card correction workflow"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            # First need to have some analysis results
            UserAction("click", "screenshot_btn"),
            UserAction("wait", "analysis", {"time": 5.0}),
            # Try to access correction features
            UserAction("click", "undo_btn"),
            UserAction("wait", "undo", {"time": 0.5}),
            UserAction("screenshot", "after_undo"),
            UserAction("click", "redo_btn"),
            UserAction("wait", "redo", {"time": 0.5}),
            UserAction("screenshot", "after_redo"),
            UserAction("click", "history_btn"),
            UserAction("wait", "history", {"time": 1.0}),
            UserAction("screenshot", "correction_history"),
        ]
        return self.run_workflow("Correction Workflow", actions)
    
    def scenario_ai_helper_controls(self) -> WorkflowResult:
        """Test AI helper controls"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            # Test archetype selection
            UserAction("click", "archetype_menu"),
            UserAction("wait", "archetype", {"time": 1.0}),
            UserAction("screenshot", "archetype_menu"),
        ]
        return self.run_workflow("AI Helper Controls", actions)
    
    def scenario_error_handling(self) -> WorkflowResult:
        """Test error handling with invalid operations"""
        actions = [
            UserAction("wait", "startup", {"time": 1.0}),
            # Try to perform operations that might cause errors
            UserAction("click", "screenshot_btn"),
            UserAction("click", "screenshot_btn"),  # Double click quickly
            UserAction("wait", "potential_error", {"time": 2.0}),
            UserAction("screenshot", "error_handling"),
            # Try clicking multiple things rapidly
            UserAction("click", "ultimate_detection_btn"),
            UserAction("click", "phash_detection_btn"),
            UserAction("click", "arena_priority_btn"),
            UserAction("wait", "rapid_clicks", {"time": 1.0}),
            UserAction("screenshot", "after_rapid_clicks"),
        ]
        return self.run_workflow("Error Handling", actions)
    
    def run_all_scenarios(self) -> List[WorkflowResult]:
        """Run all user workflow scenarios"""
        scenarios = [
            self.scenario_basic_startup,
            self.scenario_screenshot_analysis,
            self.scenario_detection_method_toggle,
            self.scenario_draft_overlay_toggle,
            self.scenario_coordinate_selection,
            self.scenario_settings_dialog,
            self.scenario_debug_mode_toggle,
            self.scenario_performance_report,
            self.scenario_correction_workflow,
            self.scenario_ai_helper_controls,
            self.scenario_error_handling,
        ]
        
        results = []
        for scenario in scenarios:
            try:
                result = scenario()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                self.log(f"Scenario failed to run: {e}", "ERROR")
                failed_result = WorkflowResult(
                    workflow_name=f"Failed_{scenario.__name__}",
                    success=False,
                    actions_completed=0,
                    total_actions=0,
                    error_message=str(e),
                    duration=0.0
                )
                results.append(failed_result)
                self.test_results.append(failed_result)
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        successful_workflows = [r for r in self.test_results if r.success]
        failed_workflows = [r for r in self.test_results if not r.success]
        
        total_actions = sum(r.total_actions for r in self.test_results)
        completed_actions = sum(r.actions_completed for r in self.test_results)
        
        report = {
            "summary": {
                "total_workflows": len(self.test_results),
                "successful_workflows": len(successful_workflows),
                "failed_workflows": len(failed_workflows),
                "success_rate": len(successful_workflows) / len(self.test_results) * 100,
                "total_actions": total_actions,
                "completed_actions": completed_actions,
                "action_completion_rate": completed_actions / total_actions * 100 if total_actions > 0 else 0,
                "total_duration": sum(r.duration for r in self.test_results),
                "average_duration": sum(r.duration for r in self.test_results) / len(self.test_results)
            },
            "successful_workflows": [
                {
                    "name": r.workflow_name,
                    "duration": r.duration,
                    "actions_completed": r.actions_completed,
                    "screenshots": len(r.screenshots) if r.screenshots else 0
                }
                for r in successful_workflows
            ],
            "failed_workflows": [
                {
                    "name": r.workflow_name,
                    "error": r.error_message,
                    "actions_completed": r.actions_completed,
                    "total_actions": r.total_actions,
                    "duration": r.duration
                }
                for r in failed_workflows
            ],
            "detailed_results": [asdict(r) for r in self.test_results],
            "environment": {
                "gui_available": GUI_AVAILABLE,
                "tkinter_available": TKINTER_AVAILABLE,
                "headless_mode": self.headless,
                "screenshots_enabled": self.capture_screenshots
            }
        }
        
        return report
    
    def save_report(self, filename: str = None) -> Path:
        """Save test report to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"user_workflow_report_{timestamp}.json"
        
        report_path = self.artifacts_dir / filename
        report = self.generate_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"📊 User workflow report saved: {report_path}")
        return report_path
    
    def cleanup(self):
        """Clean up resources"""
        if self.current_gui:
            try:
                if hasattr(self.current_gui, 'stop'):
                    self.current_gui.stop()
                if hasattr(self.current_gui, 'root') and self.current_gui.root:
                    self.current_gui.root.quit()
                    self.current_gui.root.destroy()
            except:
                pass
            self.current_gui = None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test real user workflows and interactions")
    parser.add_argument("--all", action="store_true", help="Run all workflow scenarios")
    parser.add_argument("--scenario", type=str, help="Run specific scenario")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
    parser.add_argument("--no-screenshots", action="store_true", help="Disable screenshot capture")
    
    args = parser.parse_args()
    
    # Create tester
    tester = UserWorkflowTester(
        headless=args.headless,
        capture_screenshots=not args.no_screenshots
    )
    
    try:
        print("🎮 User Workflow Testing System")
        print("=" * 50)
        
        if args.all:
            print("Running all workflow scenarios...")
            results = tester.run_all_scenarios()
            
        elif args.scenario:
            print(f"Running scenario: {args.scenario}")
            scenario_method = getattr(tester, f"scenario_{args.scenario}", None)
            if scenario_method:
                result = scenario_method()
                results = [result]
            else:
                print(f"❌ Unknown scenario: {args.scenario}")
                return 1
                
        elif args.interactive:
            print("Interactive mode not yet implemented")
            return 1
            
        else:
            print("Please specify --all, --scenario <name>, or --interactive")
            return 1
        
        # Generate and save report
        report_path = tester.save_report()
        report = tester.generate_report()
        
        # Print summary
        print(f"\n🎯 USER WORKFLOW TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Workflows: {report['summary']['total_workflows']}")
        print(f"✅ Successful: {report['summary']['successful_workflows']}")
        print(f"❌ Failed: {report['summary']['failed_workflows']}")
        print(f"📊 Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"⚡ Total Actions: {report['summary']['completed_actions']}/{report['summary']['total_actions']}")
        print(f"📁 Report: {report_path}")
        
        if report['failed_workflows']:
            print(f"\n❌ FAILED WORKFLOWS:")
            for failure in report['failed_workflows']:
                print(f"  • {failure['name']}: {failure['error']}")
        
        if report['successful_workflows']:
            print(f"\n✅ SUCCESSFUL WORKFLOWS:")
            for success in report['successful_workflows']:
                print(f"  • {success['name']} ({success['duration']:.2f}s, {success['actions_completed']} actions)")
        
        # Exit code based on results
        if report['summary']['failed_workflows'] > 0:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"💥 Testing system crashed: {e}")
        traceback.print_exc()
        return 2
    
    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())