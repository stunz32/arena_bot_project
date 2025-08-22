#!/usr/bin/env python3
"""
Comprehensive GUI debugging test for the main Arena Bot GUI.
This will create, analyze, and identify all GUI issues in integrated_arena_bot_gui.py
"""

import sys
import os
from pathlib import Path
import traceback
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.debug_utils import create_debug_snapshot, analyze_layout_issues

def test_main_gui_comprehensive():
    """Test the main GUI and capture all issues."""
    print("🔍 COMPREHENSIVE GUI DEBUGGING - Arena Bot Main Interface")
    print("=" * 80)
    
    try:
        # Import the main GUI class
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        print("✅ Successfully imported IntegratedArenaBotGUI")
        
        # Create GUI instance
        print("🚀 Creating GUI instance...")
        gui = IntegratedArenaBotGUI()
        
        print("✅ GUI instance created successfully")
        
        # Setup GUI
        print("🎨 Setting up GUI layout...")
        gui.setup_gui()
        
        print("✅ GUI setup completed")
        
        # Let GUI render
        if hasattr(gui, 'root') and gui.root:
            gui.root.update_idletasks()
            
            print("📸 Capturing comprehensive debug snapshot...")
            
            # Create complete debug snapshot
            snapshot_results = create_debug_snapshot(gui.root, "main_arena_bot_gui")
            
            print("\n📊 GUI DEBUG SNAPSHOT RESULTS:")
            for key, path in snapshot_results.items():
                if path and os.path.exists(path):
                    print(f"  ✅ {key}: {path}")
                else:
                    print(f"  ❌ {key}: Failed to create")
            
            # Perform layout analysis
            print("\n🔍 ANALYZING LAYOUT ISSUES...")
            layout_issues = analyze_layout_issues(gui.root)
            
            print(f"\n📋 LAYOUT ANALYSIS RESULTS:")
            print(f"  🔸 Zero-size widgets: {len(layout_issues['zero_size_widgets'])}")
            print(f"  🔸 Missing layout managers: {len(layout_issues['missing_pack_grid'])}")
            print(f"  🔸 Potential problems: {len(layout_issues['potential_problems'])}")
            print(f"  🔸 Overlapping widgets: {len(layout_issues['overlapping_widgets'])}")
            
            # Detailed issue reporting
            if layout_issues['zero_size_widgets']:
                print(f"\n⚠️ ZERO-SIZE WIDGETS DETECTED:")
                for widget in layout_issues['zero_size_widgets']:
                    print(f"     - {widget['path']}: {widget['size']}")
            
            if layout_issues['missing_pack_grid']:
                print(f"\n⚠️ MISSING LAYOUT MANAGERS:")
                for widget in layout_issues['missing_pack_grid']:
                    print(f"     - {widget['path']}: {widget['has_children']} children")
            
            if layout_issues['potential_problems']:
                print(f"\n⚠️ POTENTIAL PROBLEMS:")
                for problem in layout_issues['potential_problems']:
                    print(f"     - {problem['path']}: {problem['error']}")
            
            # Count total widgets
            widget_tree_path = snapshot_results.get("widget_tree")
            if widget_tree_path and os.path.exists(widget_tree_path):
                with open(widget_tree_path, 'r') as f:
                    tree_data = json.load(f)
                
                def count_widgets(node):
                    count = 1
                    for child in node.get("children", []):
                        count += count_widgets(child)
                    return count
                
                total_widgets = count_widgets(tree_data["widget_tree"])
                
                print(f"\n📊 GUI COMPLEXITY METRICS:")
                print(f"  🔸 Total widgets: {total_widgets}")
                print(f"  🔸 Root class: {tree_data['widget_tree']['class']}")
                print(f"  🔸 Screen resolution: {tree_data['metadata']['screen_info']['width']}x{tree_data['metadata']['screen_info']['height']}")
                
                # Calculate issue percentage
                total_issues = (len(layout_issues['zero_size_widgets']) + 
                              len(layout_issues['missing_pack_grid']) + 
                              len(layout_issues['potential_problems']))
                
                issue_percentage = (total_issues / total_widgets * 100) if total_widgets > 0 else 0
                
                print(f"  🔸 Issue rate: {issue_percentage:.1f}% ({total_issues}/{total_widgets})")
                
                # Overall health assessment
                if issue_percentage == 0:
                    health_status = "🟢 EXCELLENT"
                elif issue_percentage < 5:
                    health_status = "🟡 GOOD"
                elif issue_percentage < 15:
                    health_status = "🟠 NEEDS ATTENTION"
                else:
                    health_status = "🔴 POOR"
                
                print(f"  🔸 GUI Health: {health_status}")
            
            # Component-specific analysis
            print(f"\n🔍 COMPONENT-SPECIFIC ANALYSIS:")
            
            # Check for common GUI anti-patterns
            component_issues = []
            
            # Check for hardcoded sizes
            if hasattr(gui, 'root'):
                geometry = gui.root.geometry()
                if 'x' in geometry and '+' not in geometry:
                    component_issues.append("Fixed window size detected - may not scale properly")
            
            # Check for missing error handling in GUI setup
            gui_setup_methods = ['setup_gui', 'init_card_detection', 'init_ai_advisor']
            for method_name in gui_setup_methods:
                if hasattr(gui, method_name):
                    # This is a basic check - in a real implementation, we'd analyze the method code
                    component_issues.append(f"Method {method_name} exists - ensure proper error handling")
            
            if component_issues:
                print(f"  🔸 Component recommendations:")
                for issue in component_issues[:5]:  # Limit to first 5
                    print(f"     - {issue}")
            
            print(f"\n✅ COMPREHENSIVE GUI ANALYSIS COMPLETE!")
            print(f"\n📁 All debug files saved to artifacts/ directory")
            
            # Cleanup
            gui.root.destroy()
            
        else:
            print("❌ GUI root window not created properly")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        print("💡 This might be due to missing dependencies or complex AI imports")
        return False
    
    except Exception as e:
        print(f"❌ Error during GUI analysis: {e}")
        print(f"📋 Traceback:")
        traceback.print_exc()
        return False

def test_ui_components():
    """Test individual UI components from arena_bot/ui/"""
    print("\n🔍 TESTING INDIVIDUAL UI COMPONENTS")
    print("=" * 80)
    
    components = [
        ('arena_bot.ui.draft_overlay', 'DraftOverlay'),
        ('arena_bot.ui.visual_overlay', 'VisualOverlay'),
    ]
    
    for module_name, class_name in components:
        print(f"\n🧪 Testing {class_name}...")
        
        try:
            # Import component
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            print(f"  ✅ {class_name} imported successfully")
            
            # Try to create instance
            if class_name == 'DraftOverlay':
                from arena_bot.ui.draft_overlay import OverlayConfig
                config = OverlayConfig()
                component = component_class(config)
            else:
                component = component_class()
            
            print(f"  ✅ {class_name} instance created")
            
            # Test initialization
            if hasattr(component, 'initialize'):
                component.initialize()
                print(f"  ✅ {class_name} initialized")
                
                # Capture debug info if it has a root window
                if hasattr(component, 'root') and component.root:
                    component.root.update_idletasks()
                    snapshot = create_debug_snapshot(component.root, f"{class_name.lower()}_component")
                    print(f"  📸 Debug snapshot captured for {class_name}")
                    component.root.destroy()
            
            # Cleanup
            if hasattr(component, 'cleanup'):
                component.cleanup()
            
        except Exception as e:
            print(f"  ❌ Error testing {class_name}: {e}")
            continue

if __name__ == "__main__":
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    print("🎮 Arena Bot GUI Debugging Suite")
    print("=" * 80)
    
    success = test_main_gui_comprehensive()
    
    if success:
        test_ui_components()
        
        print(f"\n🎉 GUI DEBUGGING COMPLETE!")
        print(f"\n📂 Check artifacts/ directory for:")
        print(f"   📄 Widget trees (JSON)")
        print(f"   📸 Screenshots (PNG)")
        print(f"   📊 Layout analysis (JSON)")
        
        # List generated files
        artifacts_dir = Path("artifacts")
        if artifacts_dir.exists():
            print(f"\n📁 Generated files:")
            for file in sorted(artifacts_dir.glob("main_arena_bot_gui*")):
                print(f"   - {file.name}")
    else:
        print(f"\n❌ GUI debugging failed - check error messages above")