#!/usr/bin/env python3
"""
GUI Debugging Solution Demonstration
Shows how to use the tkinter-adapted debugging tools for Arena Bot GUI development.
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.debug_utils import create_debug_snapshot, snap_widget, dump_widget_tree, analyze_layout_issues

def create_demo_gui():
    """Create a demo GUI that simulates Arena Bot interface elements."""
    root = tk.Tk()
    root.title("Arena Bot - GUI Debugging Demo")
    root.geometry("800x600")
    root.configure(bg="#2c3e50")
    
    # Header section (simulating draft overlay header)
    header_frame = tk.Frame(root, bg="#34495e", height=80)
    header_frame.pack(fill="x", pady=(0, 10))
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(header_frame, text="🎮 Arena Draft Assistant", 
                          font=("Arial", 18, "bold"), bg="#34495e", fg="#ecf0f1")
    title_label.pack(expand=True)
    
    # Main content area
    main_frame = tk.Frame(root, bg="#2c3e50")
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Left panel - Card recommendations (simulating draft overlay)
    left_panel = tk.LabelFrame(main_frame, text="Card Recommendations", 
                              font=("Arial", 12, "bold"), bg="#2c3e50", fg="#ecf0f1")
    left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Simulate 3 card choices
    card_colors = ["#e74c3c", "#f39c12", "#27ae60"]  # Red, Orange, Green
    card_names = ["Fireball", "Lightning Bolt", "Healing Potion"]
    
    for i, (color, name) in enumerate(zip(card_colors, card_names)):
        card_frame = tk.Frame(left_panel, bg=color, relief="raised", bd=2)
        card_frame.pack(fill="x", padx=10, pady=5)
        
        # Card name
        name_label = tk.Label(card_frame, text=name, font=("Arial", 14, "bold"), 
                             bg=color, fg="white")
        name_label.pack(pady=5)
        
        # Stats
        stats_frame = tk.Frame(card_frame, bg=color)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(stats_frame, text=f"Tier Score: {85-i*10}", bg=color, fg="white").pack(side="left")
        tk.Label(stats_frame, text=f"Win Rate: {65-i*5}%", bg=color, fg="white").pack(side="right")
    
    # Right panel - Controls (simulating settings)
    right_panel = tk.LabelFrame(main_frame, text="Controls & Settings", 
                               font=("Arial", 12, "bold"), bg="#2c3e50", fg="#ecf0f1")
    right_panel.pack(side="right", fill="y", padx=(10, 0))
    
    # Detection controls
    tk.Label(right_panel, text="Detection Status:", bg="#2c3e50", fg="#ecf0f1").pack(pady=5)
    
    status_var = tk.StringVar(value="Active")
    status_combo = ttk.Combobox(right_panel, textvariable=status_var, 
                               values=["Active", "Paused", "Stopped"], state="readonly")
    status_combo.pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(right_panel, bg="#2c3e50")
    button_frame.pack(fill="x", pady=10)
    
    tk.Button(button_frame, text="Start Detection", bg="#27ae60", fg="white").pack(fill="x", pady=2)
    tk.Button(button_frame, text="Pause", bg="#f39c12", fg="white").pack(fill="x", pady=2)
    tk.Button(button_frame, text="Settings", bg="#3498db", fg="white").pack(fill="x", pady=2)
    tk.Button(button_frame, text="Debug Capture", bg="#9b59b6", fg="white", 
             command=lambda: capture_debug_info(root)).pack(fill="x", pady=2)
    
    # Footer status bar
    footer_frame = tk.Frame(root, bg="#34495e", height=30)
    footer_frame.pack(fill="x", side="bottom")
    footer_frame.pack_propagate(False)
    
    status_label = tk.Label(footer_frame, text="Ready - Monitoring for draft screens...", 
                           bg="#34495e", fg="#ecf0f1", font=("Arial", 10))
    status_label.pack(side="left", padx=10, pady=5)
    
    accuracy_label = tk.Label(footer_frame, text="Detection Accuracy: 100%", 
                             bg="#34495e", fg="#27ae60", font=("Arial", 10, "bold"))
    accuracy_label.pack(side="right", padx=10, pady=5)
    
    return root

def capture_debug_info(root):
    """Capture debug information for the current GUI state."""
    print("\n🔍 Capturing debug information...")
    
    # Create comprehensive debug snapshot
    results = create_debug_snapshot(root, "arena_bot_demo")
    
    print("📊 Debug capture results:")
    for key, path in results.items():
        if path and Path(path).exists():
            print(f"  ✅ {key}: {path}")
        else:
            print(f"  ❌ {key}: Failed to create")
    
    # Show layout analysis
    issues = analyze_layout_issues(root)
    print(f"\n📋 Layout Analysis:")
    print(f"  - Zero-size widgets: {len(issues['zero_size_widgets'])}")
    print(f"  - Missing layout managers: {len(issues['missing_pack_grid'])}")
    print(f"  - Potential problems: {len(issues['potential_problems'])}")
    
    # Show widget tree summary
    tree_path = results.get("widget_tree")
    if tree_path and Path(tree_path).exists():
        with open(tree_path, 'r') as f:
            tree_data = json.load(f)
        
        def count_widgets(node):
            count = 1
            for child in node.get("children", []):
                count += count_widgets(child)
            return count
        
        widget_count = count_widgets(tree_data["widget_tree"])
        print(f"  - Total widgets: {widget_count}")
        print(f"  - Root class: {tree_data['widget_tree']['class']}")
        print(f"  - Screen resolution: {tree_data['metadata']['screen_info']['width']}x{tree_data['metadata']['screen_info']['height']}")

def demonstrate_usage():
    """Demonstrate the GUI debugging solution."""
    print("🎮 Arena Bot GUI Debugging Solution Demo")
    print("=" * 50)
    print()
    print("This demonstrates your friend's solution adapted for tkinter:")
    print("1. 📸 Screenshot capture (fullscreen & widget-specific)")
    print("2. 🌳 Widget tree extraction with properties")
    print("3. 🔍 Layout analysis and issue detection")
    print("4. 📊 Complete debug snapshots")
    print()
    print("Usage patterns:")
    print("• Run pytest with xvfb for headless testing")
    print("• Capture GUI state at specific points")
    print("• Analyze layout issues automatically")
    print("• Generate evidence-based debugging data")
    print()
    print("Starting demo GUI... (Click 'Debug Capture' to see it in action)")
    print()

if __name__ == "__main__":
    # Ensure artifacts directory exists
    Path("artifacts").mkdir(exist_ok=True)
    
    demonstrate_usage()
    
    try:
        # Create and show demo GUI
        root = create_demo_gui()
        
        # Initial debug capture
        print("📸 Taking initial snapshot...")
        initial_results = create_debug_snapshot(root, "initial_state")
        print(f"✅ Initial snapshot saved to artifacts/")
        
        # Run the GUI
        print("🚀 Demo GUI is running. Close window to complete demo.")
        root.mainloop()
        
    except tk.TclError as e:
        print(f"❌ Cannot run GUI demo: {e}")
        print("💡 Try running with xvfb: xvfb-run python3 demo_gui_debugging.py")
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    
    print("\n✅ Demo completed!")
    print("\n📁 Check artifacts/ directory for generated debug files:")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        for file in sorted(artifacts_dir.glob("*")):
            print(f"  - {file.name}")
    else:
        print("  (No artifacts generated)")