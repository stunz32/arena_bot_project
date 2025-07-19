#!/usr/bin/env python3
"""
Live Hearthstone Arena Card Detection Test
Takes a screenshot and runs the ultimate detector
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

# Add the project directory to the path
sys.path.append('/home/marcco/arena_bot_project')

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    
    print("🎯 LIVE HEARTHSTONE TEST")
    print("=" * 50)
    
    # Take screenshot
    print("📸 Taking screenshot...")
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    # Save screenshot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = f"/home/marcco/arena_bot_project/live_test_{timestamp}.png"
    cv2.imwrite(screenshot_path, screenshot_cv)
    print(f"✅ Screenshot saved: {screenshot_path}")
    
    # Import and run the ultimate detector
    from ultimate_card_detector_clean import UltimateCardDetector
    
    print("\n🎯 Running Ultimate Card Detector...")
    detector = UltimateCardDetector()
    
    # Run detection without target injection for live testing
    results = detector.detect_cards_live(screenshot_path)
    
    if results:
        print(f"\n✅ Detection successful!")
        print(f"📊 Detected {len(results)} cards:")
        for i, card in enumerate(results, 1):
            print(f"   {i}: {card['name']} ({card['card_id']}) - confidence: {card['confidence']:.3f}")
    else:
        print("\n❌ No cards detected")
        print("💡 Make sure you're in an Arena draft screen")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("💡 Try: source venv/bin/activate")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()