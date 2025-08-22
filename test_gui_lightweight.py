#!/usr/bin/env python3
"""
Lightweight GUI debugging - tests GUI layout without heavy systems.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.debug_utils import create_debug_snapshot, analyze_layout_issues

def create_lightweight_gui_test():
    """Create a lightweight version of the main GUI for testing."""
    print("🚀 Creating lightweight GUI test...")
    
    root = tk.Tk()
    root.title("Arena Bot GUI - Debug Test")
    root.geometry("1200x800")
    root.configure(bg="#2c3e50")
    
    # Create main layout similar to the real GUI
    main_frame = tk.Frame(root, bg="#2c3e50")
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Header section
    header_frame = tk.Frame(main_frame, bg="#34495e", height=60)
    header_frame.pack(fill="x", pady=(0, 10))
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="🎮 Arena Bot - Detection Status", 
             font=("Arial", 16, "bold"), bg="#34495e", fg="#ecf0f1").pack(expand=True)
    
    # Status bar
    status_frame = tk.Frame(main_frame, bg="#34495e", height=30)
    status_frame.pack(fill="x", pady=(0, 10))
    status_frame.pack_propagate(False)
    
    tk.Label(status_frame, text="Status: Ready", bg="#34495e", fg="#27ae60", 
             font=("Arial", 10)).pack(side="left", padx=10, pady=5)
    tk.Label(status_frame, text="Accuracy: 100%", bg="#34495e", fg="#3498db", 
             font=("Arial", 10)).pack(side="right", padx=10, pady=5)
    
    # Main content area
    content_frame = tk.Frame(main_frame, bg="#2c3e50")
    content_frame.pack(fill="both", expand=True)
    
    # Left panel - Cards
    left_panel = tk.LabelFrame(content_frame, text="Detected Cards", 
                              font=("Arial", 12, "bold"), bg="#2c3e50", fg="#ecf0f1")
    left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Simulate 3 card slots
    card_colors = ["#e74c3c", "#f39c12", "#27ae60"]
    for i, color in enumerate(card_colors):
        card_frame = tk.Frame(left_panel, bg=color, relief="raised", bd=2, height=120)
        card_frame.pack(fill="x", padx=10, pady=5)
        card_frame.pack_propagate(False)
        
        tk.Label(card_frame, text=f"Card {i+1}", font=("Arial", 14, "bold"), 
                bg=color, fg="white").pack(pady=10)
        
        # Create a nested frame that might cause layout issues
        inner_frame = tk.Frame(card_frame, bg=color)
        inner_frame.pack(fill="x", padx=10)
        
        tk.Label(inner_frame, text=f"Tier: {90-i*10}", bg=color, fg="white").pack(side="left")
        tk.Label(inner_frame, text=f"Score: {85-i*5}%", bg=color, fg="white").pack(side="right")
    
    # Right panel - Controls
    right_panel = tk.LabelFrame(content_frame, text="Controls", 
                               font=("Arial", 12, "bold"), bg="#2c3e50", fg="#ecf0f1")
    right_panel.pack(side="right", fill="y", padx=(10, 0))
    
    # Detection controls
    detection_frame = tk.LabelFrame(right_panel, text="Detection", bg="#2c3e50", fg="#ecf0f1")
    detection_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Button(detection_frame, text="Start Detection", bg="#27ae60", fg="white").pack(fill="x", pady=2)
    tk.Button(detection_frame, text="Stop Detection", bg="#e74c3c", fg="white").pack(fill="x", pady=2)
    
    # Settings with potential issues
    settings_frame = tk.LabelFrame(right_panel, text="Settings", bg="#2c3e50", fg="#ecf0f1")
    settings_frame.pack(fill="x", padx=5, pady=5)
    
    # Create problematic widgets
    # 1. Widget with no layout manager (will show in analysis)
    orphan_label = tk.Label(settings_frame, text="Orphaned Widget", bg="#e74c3c", fg="white")
    # Note: Not packed - this will be detected as missing layout manager
    
    # 2. Widget that might have zero size
    zero_frame = tk.Frame(settings_frame, bg="#f39c12")
    zero_frame.pack(fill="x")
    # Empty frame with no content - might show as zero size
    
    # 3. Overlapping widgets (intentional issue)
    overlap_frame = tk.Frame(settings_frame, bg="#9b59b6")
    overlap_frame.pack(fill="x", pady=2)
    
    label1 = tk.Label(overlap_frame, text="Overlapping 1", bg="#9b59b6", fg="white")
    label1.place(x=10, y=5)
    
    label2 = tk.Label(overlap_frame, text="Overlapping 2", bg="#8e44ad", fg="white")
    label2.place(x=15, y=10)  # Slightly offset - might overlap
    
    # Normal controls
    tk.Label(settings_frame, text="Mode:", bg="#2c3e50", fg="#ecf0f1").pack(anchor="w", padx=5)
    mode_combo = ttk.Combobox(settings_frame, values=["Auto", "Manual", "Debug"], state="readonly")
    mode_combo.pack(fill="x", padx=5, pady=2)
    mode_combo.set("Auto")
    
    tk.Label(settings_frame, text="Confidence:", bg="#2c3e50", fg="#ecf0f1").pack(anchor="w", padx=5)
    confidence_scale = tk.Scale(settings_frame, from_=0, to=100, orient="horizontal", 
                               bg="#2c3e50", fg="#ecf0f1", highlightbackground="#2c3e50")
    confidence_scale.pack(fill="x", padx=5)
    confidence_scale.set(75)
    
    # Debug panel
    debug_frame = tk.LabelFrame(right_panel, text="Debug", bg="#2c3e50", fg="#ecf0f1")
    debug_frame.pack(fill="x", padx=5, pady=5)
    
    tk.Button(debug_frame, text="Capture GUI State", bg="#9b59b6", fg="white",
             command=lambda: capture_gui_debug(root)).pack(fill="x", pady=2)
    tk.Button(debug_frame, text="Layout Analysis", bg="#3498db", fg="white",
             command=lambda: analyze_gui_layout(root)).pack(fill="x", pady=2)
    
    # Bottom status
    bottom_frame = tk.Frame(main_frame, bg="#34495e", height=25)
    bottom_frame.pack(fill="x", side="bottom", pady=(10, 0))
    bottom_frame.pack_propagate(False)
    
    tk.Label(bottom_frame, text="Ready for testing | Use Debug buttons to capture GUI state", 
             bg="#34495e", fg="#95a5a6", font=("Arial", 9)).pack(expand=True)
    
    return root

def capture_gui_debug(root):
    """Capture debug information for the GUI."""
    print("\n🔍 Capturing GUI debug information...")
    results = create_debug_snapshot(root, "lightweight_gui_test")
    
    print("📊 Debug capture results:")
    for key, path in results.items():
        if path and os.path.exists(path):
            print(f"  ✅ {key}: {path}")

def analyze_gui_layout(root):
    """Analyze GUI layout for issues."""
    print("\n🔍 Analyzing GUI layout...")
    issues = analyze_layout_issues(root)
    
    print(f"📋 Layout Analysis Results:")
    print(f"  - Zero-size widgets: {len(issues['zero_size_widgets'])}")
    print(f"  - Missing layout managers: {len(issues['missing_pack_grid'])}")
    print(f"  - Potential problems: {len(issues['potential_problems'])}")
    
    if issues['zero_size_widgets']:
        print(f"\n⚠️ Zero-size widgets found:")
        for widget in issues['zero_size_widgets']:
            print(f"     {widget['path']}: {widget['size']}")
    
    if issues['missing_pack_grid']:
        print(f"\n⚠️ Missing layout managers:")
        for widget in issues['missing_pack_grid']:
            print(f"     {widget['path']}")

def test_arena_bot_ui_components():
    """Test the actual UI components without heavy systems."""
    print("\n🧪 Testing Arena Bot UI Components (Lightweight)")
    print("=" * 60)
    
    # Test draft overlay
    try:
        print("📝 Testing DraftOverlay...")
        from arena_bot.ui.draft_overlay import DraftOverlay, OverlayConfig
        
        config = OverlayConfig()
        overlay = DraftOverlay(config)
        
        print("  ✅ DraftOverlay created successfully")
        
        # Mock the monitoring to avoid system dependencies
        with patch.object(overlay, '_start_monitoring'):
            overlay.initialize()
            
            if overlay.root:
                overlay.root.update_idletasks()
                
                # Quick analysis
                results = create_debug_snapshot(overlay.root, "draft_overlay_test")
                issues = analyze_layout_issues(overlay.root)
                
                print(f"  📊 DraftOverlay analysis:")
                print(f"     - Layout issues: {len(issues['zero_size_widgets']) + len(issues['missing_pack_grid'])}")
                
                overlay.root.destroy()
            
            overlay.cleanup()
        
    except Exception as e:
        print(f"  ❌ DraftOverlay test failed: {e}")
    
    # Test visual overlay  
    try:
        print("📝 Testing VisualOverlay...")
        
        # Mock the dependencies to avoid heavy imports
        with patch.dict('sys.modules', {
            'arena_bot.ui.visual_overlay.get_s_tier_logger': Mock()
        }):
            from arena_bot.ui.visual_overlay import VisualOverlay
            
            overlay = VisualOverlay()
            print("  ✅ VisualOverlay created successfully")
            
            # Note: VisualOverlay has complex dependencies, so we'll just test creation
            
    except Exception as e:
        print(f"  ❌ VisualOverlay test failed: {e}")

def main():
    """Run lightweight GUI debugging."""
    print("🎮 Arena Bot - Lightweight GUI Debugging")
    print("=" * 60)
    print("This version tests GUI layout without loading heavy detection systems.")
    print()
    
    # Ensure artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    
    try:
        # Create and test lightweight GUI
        print("🚀 Creating lightweight GUI...")
        root = create_lightweight_gui_test()
        
        print("✅ GUI created successfully")
        print("📸 Capturing initial state...")
        
        # Let it render
        root.update_idletasks()
        
        # Immediate analysis
        results = create_debug_snapshot(root, "arena_bot_gui_lightweight")
        issues = analyze_layout_issues(root)
        
        print(f"\n📊 QUICK GUI ANALYSIS RESULTS:")
        print(f"  🔸 Zero-size widgets: {len(issues['zero_size_widgets'])}")
        print(f"  🔸 Missing layout managers: {len(issues['missing_pack_grid'])}")
        print(f"  🔸 Potential problems: {len(issues['potential_problems'])}")
        
        # Show specific issues
        if issues['zero_size_widgets']:
            print(f"\n⚠️ ZERO-SIZE WIDGETS:")
            for widget in issues['zero_size_widgets']:
                print(f"     - {widget['path']}: {widget['size']}")
        
        if issues['missing_pack_grid']:
            print(f"\n⚠️ MISSING LAYOUT MANAGERS:")
            for widget in issues['missing_pack_grid']:
                print(f"     - {widget['path']}")
        
        if issues['potential_problems']:
            print(f"\n⚠️ POTENTIAL PROBLEMS:")
            for problem in issues['potential_problems']:
                print(f"     - {problem['path']}: {problem['error']}")
        
        # Calculate widget count
        widget_tree_path = results.get("widget_tree")
        if widget_tree_path and os.path.exists(widget_tree_path):
            with open(widget_tree_path, 'r') as f:
                tree_data = json.load(f)
            
            def count_widgets(node):
                count = 1
                for child in node.get("children", []):
                    count += count_widgets(child)
                return count
            
            total_widgets = count_widgets(tree_data["widget_tree"])
            total_issues = len(issues['zero_size_widgets']) + len(issues['missing_pack_grid']) + len(issues['potential_problems'])
            
            issue_percentage = (total_issues / total_widgets * 100) if total_widgets > 0 else 0
            
            print(f"\n📈 GUI HEALTH METRICS:")
            print(f"  🔸 Total widgets: {total_widgets}")
            print(f"  🔸 Issues found: {total_issues}")
            print(f"  🔸 Issue rate: {issue_percentage:.1f}%")
            
            if issue_percentage == 0:
                health = "🟢 EXCELLENT"
            elif issue_percentage < 5:
                health = "🟡 GOOD"
            elif issue_percentage < 15:
                health = "🟠 NEEDS ATTENTION"
            else:
                health = "🔴 POOR"
            
            print(f"  🔸 Health status: {health}")
        
        print(f"\n✅ Lightweight GUI analysis complete!")
        
        # Test UI components
        test_arena_bot_ui_components()
        
        # Cleanup
        root.destroy()
        
        print(f"\n🎉 ALL GUI DEBUGGING COMPLETE!")
        print(f"\n📁 Check artifacts/ directory for detailed analysis files")
        
    except Exception as e:
        print(f"❌ GUI debugging failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()