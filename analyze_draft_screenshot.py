#!/usr/bin/env python3
"""
Direct draft screenshot analysis - shows card rankings and recommendations
"""

import sys
from pathlib import Path

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

from integrated_arena_bot_headless import IntegratedArenaBotHeadless

def main():
    print("🎯 ARENA DRAFT CARD ANALYSIS")
    print("=" * 60)
    
    # Initialize bot
    bot = IntegratedArenaBotHeadless()
    
    # Your screenshot path
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    
    print(f"\n📸 Analyzing your Arena draft screenshot...")
    print(f"📁 Path: {screenshot_path}")
    
    # Analyze screenshot for cards
    detected_cards = bot.analyze_screenshot(screenshot_path)
    
    if detected_cards:
        print(f"\n✅ Found {len(detected_cards)} cards!")
        
        # Get AI recommendation 
        recommendation = bot.get_recommendation(detected_cards)
        
        # Display comprehensive analysis
        bot.display_analysis(detected_cards, recommendation)
        
    else:
        print("❌ No cards detected")
        
        # Show a demo with known cards anyway
        print("\n📋 DEMO: Showing how recommendations look with example cards...")
        demo_cards = [
            {'position': 1, 'card_code': 'AV_326', 'confidence': 0.9},  # Bloodsail Deckhand
            {'position': 2, 'card_code': 'BAR_081', 'confidence': 0.8}, # Conviction 
            {'position': 3, 'card_code': 'AT_073', 'confidence': 0.7}   # Competitive Spirit
        ]
        
        demo_recommendation = bot.get_recommendation(demo_cards)
        bot.display_analysis(demo_cards, demo_recommendation)

if __name__ == "__main__":
    main()