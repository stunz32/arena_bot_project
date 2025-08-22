#!/usr/bin/env python3
"""
Tkinter GUI debugging utilities for Arena Bot
Provides screenshot capture and widget tree inspection for tkinter applications.
Adapted from PyQt6 solution for tkinter-based GUI debugging.
"""

import tkinter as tk
from tkinter import ttk
import json
import os
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _ensure_dir(path: str) -> None:
    """Ensure directory exists for artifact storage."""
    os.makedirs(path, exist_ok=True)

def snap_fullscreen(path: str = "artifacts/fullscreen.png") -> str:
    """
    Capture fullscreen screenshot using tkinter.
    
    @param {str} path - Output path for screenshot
    @returns {str} - Path where screenshot was saved
    """
    _ensure_dir("artifacts")
    
    # Create a temporary root if none exists
    temp_root = None
    root = tk._default_root
    if root is None:
        temp_root = tk.Tk()
        temp_root.withdraw()  # Hide the temporary window
        root = temp_root
    
    try:
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # For fullscreen capture on tkinter, we need to use system tools
        # This is a limitation of tkinter - we'll use PIL/ImageGrab
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            screenshot.save(path)
            logger.info(f"Fullscreen screenshot saved to {path}")
            return path
        except ImportError:
            logger.warning("PIL not available for fullscreen capture")
            # Fallback: create a placeholder image
            with open(path.replace('.png', '_placeholder.txt'), 'w') as f:
                f.write(f"Fullscreen capture placeholder: {screen_width}x{screen_height}")
            return path.replace('.png', '_placeholder.txt')
    
    finally:
        if temp_root:
            temp_root.destroy()

def snap_widget(widget: tk.Widget, name: str) -> str:
    """
    Capture screenshot of a specific tkinter widget.
    
    @param {tk.Widget} widget - Widget to capture
    @param {str} name - Name for the output file
    @returns {str} - Path where screenshot was saved
    """
    _ensure_dir("artifacts")
    path = f"artifacts/{name}.png"
    
    try:
        # Update widget to ensure it's rendered
        widget.update_idletasks()
        
        # Get widget geometry
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        width = widget.winfo_width()
        height = widget.winfo_height()
        
        # Capture widget area using PIL if available
        try:
            from PIL import ImageGrab
            bbox = (x, y, x + width, y + height)
            screenshot = ImageGrab.grab(bbox)
            screenshot.save(path)
            logger.info(f"Widget screenshot saved to {path}")
            return path
        except ImportError:
            logger.warning("PIL not available for widget capture")
            # Fallback: save widget info as text
            info_path = path.replace('.png', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Widget: {widget.__class__.__name__}\n")
                f.write(f"Geometry: {width}x{height} at ({x}, {y})\n")
                f.write(f"State: {widget.cget('state') if hasattr(widget, 'cget') else 'N/A'}\n")
            return info_path
    
    except Exception as e:
        logger.error(f"Error capturing widget {name}: {e}")
        return ""

def get_widget_info(widget: tk.Widget) -> Dict[str, Any]:
    """
    Extract comprehensive information from a tkinter widget.
    
    @param {tk.Widget} widget - Widget to analyze
    @returns {Dict[str, Any]} - Widget information dictionary
    """
    try:
        widget.update_idletasks()
        
        info = {
            "class": widget.__class__.__name__,
            "widget_name": str(widget),
            "geometry": {
                "x": widget.winfo_x(),
                "y": widget.winfo_y(), 
                "width": widget.winfo_width(),
                "height": widget.winfo_height(),
                "root_x": widget.winfo_rootx(),
                "root_y": widget.winfo_rooty()
            },
            "state": {},
            "children": []
        }
        
        # Extract widget-specific properties
        try:
            # Common properties
            if hasattr(widget, 'cget'):
                common_props = ['state', 'text', 'bg', 'fg', 'font', 'relief', 'borderwidth']
                for prop in common_props:
                    try:
                        value = widget.cget(prop)
                        # Convert Tcl objects to strings for JSON serialization
                        info["state"][prop] = str(value) if value is not None else ""
                    except tk.TclError:
                        pass  # Property doesn't exist for this widget type
            
            # Special handling for different widget types
            if isinstance(widget, (tk.Entry, tk.Text)):
                try:
                    info["state"]["value"] = widget.get()
                except:
                    pass
            elif isinstance(widget, ttk.Combobox):
                try:
                    info["state"]["value"] = widget.get()
                    info["state"]["values"] = widget['values']
                except:
                    pass
            elif isinstance(widget, (tk.Scale, ttk.Scale)):
                try:
                    info["state"]["value"] = widget.get()
                except:
                    pass
        
        except Exception as e:
            logger.warning(f"Error extracting properties from {widget}: {e}")
        
        # Get children
        try:
            for child in widget.winfo_children():
                info["children"].append(get_widget_info(child))
        except Exception as e:
            logger.warning(f"Error getting children of {widget}: {e}")
        
        return info
    
    except Exception as e:
        logger.error(f"Error analyzing widget {widget}: {e}")
        return {
            "class": "ERROR",
            "error": str(e),
            "widget_name": str(widget) if widget else "None"
        }

def dump_widget_tree(root: tk.Widget, out_path: str = "artifacts/widget_tree.json") -> str:
    """
    Dump complete widget tree to JSON for analysis.
    
    @param {tk.Widget} root - Root widget to analyze
    @param {str} out_path - Output path for JSON file
    @returns {str} - Path where tree was saved
    """
    _ensure_dir("artifacts")
    
    try:
        tree_info = get_widget_info(root)
        
        # Add metadata
        metadata = {
            "timestamp": str(Path().cwd()),
            "root_class": root.__class__.__name__,
            "screen_info": {
                "width": root.winfo_screenwidth() if hasattr(root, 'winfo_screenwidth') else 0,
                "height": root.winfo_screenheight() if hasattr(root, 'winfo_screenheight') else 0
            }
        }
        
        output = {
            "metadata": metadata,
            "widget_tree": tree_info
        }
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Widget tree dumped to {out_path}")
        return out_path
    
    except Exception as e:
        logger.error(f"Error dumping widget tree: {e}")
        return ""

def analyze_layout_issues(root: tk.Widget) -> Dict[str, Any]:
    """
    Analyze common layout issues in tkinter applications.
    
    @param {tk.Widget} root - Root widget to analyze
    @returns {Dict[str, Any]} - Analysis results
    """
    issues = {
        "overlapping_widgets": [],
        "zero_size_widgets": [],
        "missing_pack_grid": [],
        "potential_problems": []
    }
    
    def check_widget_recursive(widget: tk.Widget, path: str = ""):
        try:
            current_path = f"{path}/{widget.__class__.__name__}"
            
            # Check for zero-size widgets
            if widget.winfo_width() == 0 or widget.winfo_height() == 0:
                issues["zero_size_widgets"].append({
                    "path": current_path,
                    "size": f"{widget.winfo_width()}x{widget.winfo_height()}"
                })
            
            # Check for layout manager usage
            manager = widget.winfo_manager()
            if not manager and widget.winfo_children():
                issues["missing_pack_grid"].append({
                    "path": current_path,
                    "has_children": len(widget.winfo_children())
                })
            
            # Recursively check children
            for child in widget.winfo_children():
                check_widget_recursive(child, current_path)
        
        except Exception as e:
            issues["potential_problems"].append({
                "path": current_path,
                "error": str(e)
            })
    
    try:
        check_widget_recursive(root)
    except Exception as e:
        logger.error(f"Error during layout analysis: {e}")
    
    return issues

def create_debug_snapshot(root: tk.Widget, snapshot_name: str = "debug_snapshot") -> Dict[str, str]:
    """
    Create a complete debug snapshot including screenshots and analysis.
    
    @param {tk.Widget} root - Root widget to snapshot
    @param {str} snapshot_name - Base name for snapshot files
    @returns {Dict[str, str]} - Paths to generated files
    """
    _ensure_dir("artifacts")
    
    results = {}
    
    try:
        # Capture widget screenshot
        widget_path = snap_widget(root, snapshot_name)
        results["widget_screenshot"] = widget_path
        
        # Capture fullscreen
        fullscreen_path = snap_fullscreen(f"artifacts/{snapshot_name}_fullscreen.png")
        results["fullscreen_screenshot"] = fullscreen_path
        
        # Dump widget tree
        tree_path = dump_widget_tree(root, f"artifacts/{snapshot_name}_widget_tree.json")
        results["widget_tree"] = tree_path
        
        # Analyze layout issues
        issues = analyze_layout_issues(root)
        issues_path = f"artifacts/{snapshot_name}_layout_analysis.json"
        with open(issues_path, 'w') as f:
            json.dump(issues, f, indent=2)
        results["layout_analysis"] = issues_path
        
        logger.info(f"Debug snapshot created: {snapshot_name}")
        
    except Exception as e:
        logger.error(f"Error creating debug snapshot: {e}")
        results["error"] = str(e)
    
    return results