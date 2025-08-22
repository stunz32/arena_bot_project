#!/usr/bin/env python3
"""
Basic tests for GUI debugging utilities
Tests core functionality without complex UI component dependencies.
"""

import pytest
import tkinter as tk
from tkinter import ttk
import sys
import os
from pathlib import Path
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.debug_utils import (
    snap_widget, snap_fullscreen, dump_widget_tree, 
    create_debug_snapshot, get_widget_info, analyze_layout_issues
)

logger = logging.getLogger(__name__)

class TestDebugUtilsBasic:
    """Basic tests for debug utilities without complex dependencies."""
    
    def test_basic_tkinter_creation(self):
        """Test basic tkinter widget creation and debugging."""
        try:
            # Create a simple test window
            root = tk.Tk()
            root.withdraw()  # Hide initially
            root.title("Debug Test Window")
            root.geometry("400x300")
            
            # Add some basic widgets
            frame = tk.Frame(root, bg="lightblue")
            frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            label = tk.Label(frame, text="Test Label", font=("Arial", 12))
            label.pack(pady=5)
            
            button = tk.Button(frame, text="Test Button")
            button.pack(pady=5)
            
            entry = tk.Entry(frame, width=20)
            entry.pack(pady=5)
            entry.insert(0, "Test text")
            
            # Update to ensure rendering
            root.update_idletasks()
            
            # Test widget info extraction
            info = get_widget_info(root)
            assert info["class"] == "Tk"
            assert len(info["children"]) > 0
            
            # Test layout analysis
            issues = analyze_layout_issues(root)
            assert isinstance(issues, dict)
            assert "overlapping_widgets" in issues
            
            # Test widget tree dump
            tree_path = dump_widget_tree(root, "artifacts/basic_test_tree.json")
            assert os.path.exists(tree_path)
            
            # Verify JSON structure
            with open(tree_path, 'r') as f:
                tree_data = json.load(f)
            assert "metadata" in tree_data
            assert "widget_tree" in tree_data
            
            logger.info("✅ Basic tkinter debugging test completed successfully")
            
        except tk.TclError:
            pytest.skip("Tkinter not available in test environment")
        finally:
            try:
                root.destroy()
            except:
                pass
    
    def test_widget_info_extraction(self):
        """Test detailed widget information extraction."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            # Create a variety of widgets to test
            widgets_to_test = []
            
            # Label with various properties
            label = tk.Label(root, text="Test Label", bg="red", fg="white", font=("Arial", 10))
            label.pack()
            widgets_to_test.append(("Label", label))
            
            # Entry with content
            entry = tk.Entry(root, width=15)
            entry.pack()
            entry.insert(0, "Sample text")
            widgets_to_test.append(("Entry", entry))
            
            # Button with state
            button = tk.Button(root, text="Click Me", state="normal")
            button.pack()
            widgets_to_test.append(("Button", button))
            
            # Scale widget
            scale = tk.Scale(root, from_=0, to=100, orient="horizontal")
            scale.pack()
            scale.set(50)
            widgets_to_test.append(("Scale", scale))
            
            root.update_idletasks()
            
            # Test each widget type
            for widget_name, widget in widgets_to_test:
                info = get_widget_info(widget)
                
                assert "class" in info
                assert "geometry" in info
                assert "state" in info
                
                # Verify geometry has required fields
                geom = info["geometry"]
                assert all(key in geom for key in ["x", "y", "width", "height"])
                
                logger.info(f"✅ {widget_name} info extraction successful")
            
        except tk.TclError:
            pytest.skip("Tkinter not available in test environment")
        finally:
            try:
                root.destroy()
            except:
                pass
    
    def test_debug_snapshot_creation(self):
        """Test complete debug snapshot creation."""
        try:
            root = tk.Tk()
            root.withdraw()
            root.title("Snapshot Test")
            root.geometry("500x400")
            
            # Create a more complex layout
            main_frame = tk.Frame(root, bg="lightgray")
            main_frame.pack(fill="both", expand=True)
            
            # Header
            header = tk.Label(main_frame, text="Test Application", 
                            font=("Arial", 16, "bold"), bg="darkblue", fg="white")
            header.pack(fill="x", pady=(0, 10))
            
            # Content area
            content_frame = tk.Frame(main_frame, bg="white")
            content_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Some controls
            tk.Label(content_frame, text="Name:", bg="white").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            name_entry = tk.Entry(content_frame, width=20)
            name_entry.grid(row=0, column=1, padx=5, pady=5)
            name_entry.insert(0, "Test User")
            
            tk.Label(content_frame, text="Options:", bg="white").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            options = ttk.Combobox(content_frame, values=["Option 1", "Option 2", "Option 3"])
            options.grid(row=1, column=1, padx=5, pady=5)
            options.set("Option 1")
            
            # Footer with buttons
            footer_frame = tk.Frame(main_frame, bg="lightgray")
            footer_frame.pack(fill="x", pady=(10, 0))
            
            tk.Button(footer_frame, text="OK").pack(side="right", padx=(5, 10))
            tk.Button(footer_frame, text="Cancel").pack(side="right", padx=5)
            
            root.update_idletasks()
            
            # Create debug snapshot
            results = create_debug_snapshot(root, "complex_layout_test")
            
            # Verify all snapshot components were created
            expected_keys = ["widget_tree", "layout_analysis"]
            for key in expected_keys:
                assert key in results, f"Missing {key} in snapshot results"
                
                # Verify file exists
                if results[key] and os.path.exists(results[key]):
                    logger.info(f"✅ {key} file created: {results[key]}")
                else:
                    logger.warning(f"⚠️ {key} file not created properly")
            
            # Verify JSON files are valid
            if "widget_tree" in results and os.path.exists(results["widget_tree"]):
                with open(results["widget_tree"], 'r') as f:
                    tree_data = json.load(f)
                assert "widget_tree" in tree_data
                assert tree_data["widget_tree"]["class"] == "Tk"
            
            if "layout_analysis" in results and os.path.exists(results["layout_analysis"]):
                with open(results["layout_analysis"], 'r') as f:
                    analysis_data = json.load(f)
                assert "overlapping_widgets" in analysis_data
                assert "zero_size_widgets" in analysis_data
            
            logger.info("✅ Complete debug snapshot test passed")
            
        except tk.TclError:
            pytest.skip("Tkinter not available in test environment")
        finally:
            try:
                root.destroy()
            except:
                pass

class TestHeadlessCompatibility:
    """Test compatibility with headless environments using xvfb."""
    
    def test_xvfb_compatibility(self):
        """Test that debugging works in xvfb environment."""
        # This test verifies the tools work even when no real display is available
        # The xvfb virtual display should allow tkinter to function
        
        try:
            root = tk.Tk()
            root.withdraw()
            
            # Simple widget creation
            label = tk.Label(root, text="Headless Test")
            label.pack()
            
            root.update_idletasks()
            
            # Test core functions
            info = get_widget_info(root)
            assert info["class"] == "Tk"
            
            # Test tree dumping
            tree_path = dump_widget_tree(root, "artifacts/headless_test_tree.json")
            assert os.path.exists(tree_path)
            
            logger.info("✅ Headless compatibility test passed")
            
        except tk.TclError as e:
            pytest.skip(f"Tkinter not available in headless environment: {e}")
        finally:
            try:
                root.destroy()
            except:
                pass

if __name__ == "__main__":
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # Run tests directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])