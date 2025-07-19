#!/usr/bin/env python3
"""
Screenshot-based testing for Arena Bot.
Supports multiple methods for getting Hearthstone screenshots in WSL.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_with_file(screenshot_path: str):
    """Test card detection with a provided screenshot file."""
    print(f"🖼️  Testing with screenshot: {screenshot_path}")
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot file not found: {screenshot_path}")
        return False
    
    try:
        # Load the screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print("❌ Failed to load screenshot")
            return False
        
        print(f"✅ Screenshot loaded: {screenshot.shape}")
        
        # Initialize card recognition system
        from arena_bot.core.card_recognizer import get_card_recognizer
        
        card_recognizer = get_card_recognizer()
        
        if not card_recognizer.initialize():
            print("❌ Failed to initialize card recognizer")
            return False
        
        print("✅ Card recognizer initialized")
        
        # Test card detection on the screenshot
        print("🔍 Detecting cards...")
        
        # For now, let's test with manual regions (you'll need to adjust these)
        # These are typical arena card choice regions (adjust based on resolution)
        height, width = screenshot.shape[:2]
        
        # Assume 1920x1080 or scale accordingly
        card_regions = [
            (width//4 - 100, height//2 - 150, 200, 300),    # Left card
            (width//2 - 100, height//2 - 150, 200, 300),    # Middle card  
            (3*width//4 - 100, height//2 - 150, 200, 300)   # Right card
        ]
        
        results = []
        for i, (x, y, w, h) in enumerate(card_regions):
            print(f"  Testing card region {i+1}: ({x}, {y}, {w}, {h})")
            
            # Extract card region
            if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                card_image = screenshot[y:y+h, x:x+w]
                
                # Test detection
                from arena_bot.detection.histogram_matcher import get_histogram_matcher
                matcher = get_histogram_matcher()
                
                match = matcher.match_card(card_image)
                
                if match:
                    print(f"    ✅ Card {i+1}: {match.card_code} (confidence: {match.confidence:.3f})")
                    results.append(match)
                else:
                    print(f"    ❌ Card {i+1}: No match found")
            else:
                print(f"    ⚠️  Card {i+1}: Region out of bounds")
        
        print(f"\n📊 Detection Results: {len(results)}/3 cards detected")
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def capture_with_wsl_screenshot():
    """Try to capture screenshot using WSL methods."""
    print("📸 Attempting to capture screenshot in WSL...")
    
    methods = [
        # Method 1: If you have Windows screenshot in clipboard
        ("powershell.exe", "Get-Clipboard -Format Image | Out-File -FilePath screenshot.png"),
        
        # Method 2: Use PowerShell to take screenshot
        ("powershell.exe", "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('%{PRTSC}')"),
    ]
    
    print("⚠️  WSL screenshot capture requires manual setup.")
    print("📋 Options:")
    print("1. Take a screenshot in Windows (Win+Shift+S or Print Screen)")
    print("2. Save it as 'screenshot.png' in this directory")
    print("3. Or provide the path to an existing Hearthstone screenshot")
    
    return False

def main():
    """Main testing function."""
    print("🎮 Arena Bot - Screenshot Testing")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    print("📋 Testing Options:")
    print("1. Test with existing screenshot file")
    print("2. Manual screenshot capture instructions")
    print()
    
    # Look for common screenshot locations
    screenshot_paths = [
        "screenshot.png",
        "test_screenshot.png", 
        "hearthstone_screenshot.png",
        str(Path.home() / "Pictures" / "screenshot.png"),
        str(Path.home() / "Desktop" / "screenshot.png")
    ]
    
    found_screenshot = None
    for path in screenshot_paths:
        if os.path.exists(path):
            found_screenshot = path
            break
    
    if found_screenshot:
        print(f"🎯 Found screenshot: {found_screenshot}")
        success = test_with_file(found_screenshot)
        
        if success:
            print("🎉 Screenshot testing completed successfully!")
        else:
            print("⚠️  Testing completed with issues. Check output above.")
    else:
        print("📁 No screenshot found in common locations.")
        print()
        print("📝 To test with real screenshots:")
        print("1. Take a Hearthstone arena draft screenshot (showing 3 card choices)")
        print("2. Save it as 'screenshot.png' in this directory")
        print("3. Run this script again")
        print()
        print("💡 Alternatively, provide a path:")
        print("   python3 test_with_screenshots.py /path/to/screenshot.png")
        
        capture_with_wsl_screenshot()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Screenshot path provided as argument
        screenshot_path = sys.argv[1]
        success = test_with_file(screenshot_path)
        sys.exit(0 if success else 1)
    else:
        main()