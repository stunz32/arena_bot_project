#!/usr/bin/env python3
"""
Debug version of Arena Bot to identify crash issues.
"""

import sys
import traceback
from pathlib import Path

def debug_imports():
    """Test imports one by one to find the issue."""
    print("🔍 Debugging Arena Bot...")
    print("=" * 40)
    
    try:
        print("1. Testing basic imports...")
        import cv2
        print("   ✅ OpenCV imported successfully")
        
        import numpy as np
        print("   ✅ NumPy imported successfully")
        
        import logging
        print("   ✅ Logging imported successfully")
        
    except Exception as e:
        print(f"   ❌ Basic import failed: {e}")
        return False
    
    try:
        print("2. Testing path setup...")
        sys.path.insert(0, str(Path(__file__).parent))
        print("   ✅ Path setup successful")
        
    except Exception as e:
        print(f"   ❌ Path setup failed: {e}")
        return False
    
    try:
        print("3. Testing arena_bot imports...")
        from arena_bot.ai.draft_advisor import get_draft_advisor
        print("   ✅ Draft advisor imported successfully")
        
        from arena_bot.core.surf_detector import get_surf_detector
        print("   ✅ SURF detector imported successfully")
        
    except Exception as e:
        print(f"   ❌ Arena bot import failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False
    
    return True

def test_screenshot():
    """Test if screenshot exists."""
    try:
        print("4. Testing screenshot...")
        screenshot_path = "screenshot.png"
        
        if not Path(screenshot_path).exists():
            print(f"   ❌ Screenshot not found: {screenshot_path}")
            return False
        
        import cv2
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"   ❌ Could not load screenshot: {screenshot_path}")
            return False
        
        print(f"   ✅ Screenshot loaded: {screenshot.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Screenshot test failed: {e}")
        return False

def run_simple_test():
    """Run a simple version without full integration."""
    try:
        print("5. Running simple arena bot test...")
        
        # Import the working components
        from arena_bot.ai.draft_advisor import get_draft_advisor
        from arena_bot.core.surf_detector import get_surf_detector
        
        print("   ✅ Components imported")
        
        # Test draft advisor
        advisor = get_draft_advisor()
        test_choice = advisor.analyze_draft_choice(['TOY_380', 'ULD_309', 'TTN_042'], 'warrior')
        
        print(f"   ✅ Draft analysis working")
        print(f"   👑 Recommendation: Card {test_choice.recommended_pick + 1} ({test_choice.cards[test_choice.recommended_pick].card_code})")
        
        # Test interface detection
        surf_detector = get_surf_detector()
        import cv2
        screenshot = cv2.imread("screenshot.png")
        interface_rect = surf_detector.detect_arena_interface(screenshot)
        
        if interface_rect:
            print(f"   ✅ Interface detection working: {interface_rect}")
        else:
            print(f"   ⚠️  Interface detection returned None")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simple test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def main():
    """Main debug function."""
    print("🎯 Arena Bot Debug Tool")
    print("This will help identify why the Arena Bot crashes")
    print()
    
    # Test each component
    if not debug_imports():
        print("\n❌ Import issues found. Cannot continue.")
        return
    
    if not test_screenshot():
        print("\n⚠️  Screenshot issues found, but continuing...")
    
    if run_simple_test():
        print("\n🎉 Arena Bot components are working!")
        print("The crash might be in the main execution flow.")
        
        # Try running the actual complete bot
        try:
            print("\n6. Testing complete arena bot...")
            exec(open('complete_arena_bot.py').read())
            
        except Exception as e:
            print(f"\n❌ Complete arena bot crashed: {e}")
            print(f"Error details: {traceback.format_exc()}")
    else:
        print("\n❌ Arena Bot components have issues.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        input("\nPress Enter to exit...")