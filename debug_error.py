#!/usr/bin/env python3
"""
Debug script to replicate the 'list index out of range' error
"""

import sys
import traceback
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

def debug_screenshot_analysis():
    """Debug the screenshot analysis to find the error."""
    try:
        from integrated_arena_bot_headless import IntegratedArenaBotHeadless
        
        print("🔍 Initializing bot...")
        bot = IntegratedArenaBotHeadless()
        
        print("🔍 Starting screenshot analysis...")
        screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
        
        print(f"🔍 Analyzing: {screenshot_path}")
        detected_cards = bot.analyze_screenshot(screenshot_path)
        
        if detected_cards:
            print(f"✅ Cards detected: {len(detected_cards)}")
            for i, card in enumerate(detected_cards):
                print(f"   Card {i+1}: {card}")
            
            print("🔍 Getting recommendation...")
            recommendation = bot.get_recommendation(detected_cards)
            
            if recommendation:
                print(f"✅ Recommendation received")
                bot.display_analysis(detected_cards, recommendation)
            else:
                print("❌ No recommendation generated")
        else:
            print("❌ No cards detected")
            
    except Exception as e:
        print(f"❌ ERROR CAUGHT: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_screenshot_analysis()