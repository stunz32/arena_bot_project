#!/usr/bin/env python3
"""
Enhanced Arena Bot - Command Line Demo
Shows all the new user-friendly features without GUI requirements.
"""

import sys
import time
from pathlib import Path
from enum import Enum

class HearthstoneScreen(Enum):
    """Different Hearthstone screen types."""
    MAIN_MENU = "Main Menu"
    ARENA_DRAFT = "Arena Draft"
    COLLECTION = "Collection"
    PLAY_MODE = "Play Mode"
    IN_GAME = "In Game"
    UNKNOWN = "Unknown Screen"

# Enhanced card name database
CARD_NAMES = {
    'TOY_380': 'Toy Captain Tarim',
    'ULD_309': 'Dragonqueen Alexstrasza', 
    'TTN_042': 'Thassarian',
    'AT_001': 'Flame Lance',
    'EX1_046': 'Dark Iron Dwarf',
    'CS2_029': 'Fireball',
    'CS2_032': 'Flamestrike',
    'CS2_234': 'Shadow Word: Pain',
    'CS2_235': 'Northshire Cleric',
}

def get_card_name(card_code: str) -> str:
    """Get user-friendly card name."""
    clean_code = card_code.replace('_premium', '')
    if clean_code in CARD_NAMES:
        name = CARD_NAMES[clean_code]
        if '_premium' in card_code:
            return f"{name} ✨"  # Golden star for premium
        return name
    return f"Unknown Card ({clean_code})"

def detect_screen_type():
    """Simulate screen detection."""
    # For demo, we'll simulate detecting an arena draft
    return HearthstoneScreen.ARENA_DRAFT

def enhance_reasoning(original_reasoning: str, cards: list, recommended_index: int) -> str:
    """Create detailed explanation for the recommendation."""
    recommended_card = cards[recommended_index]
    card_name = get_card_name(recommended_card['card_code'])
    
    enhanced = f"\n💭 DETAILED EXPLANATION:\n"
    enhanced += f"{'='*50}\n"
    enhanced += f"{card_name} is the best choice here because:\n\n"
    
    # Tier explanation
    tier = recommended_card['tier']
    if tier in ['S', 'A']:
        enhanced += f"🏆 HIGH TIER CARD: It's a {tier}-tier card, which means it's among the\n"
        enhanced += f"   strongest cards in Arena drafts. These cards consistently\n"
        enhanced += f"   perform well and have high impact on games.\n\n"
    elif tier == 'B':
        enhanced += f"⭐ SOLID CARD: It's a reliable B-tier card with good overall value.\n"
        enhanced += f"   These cards form the backbone of strong Arena decks.\n\n"
    
    # Win rate explanation
    win_rate = recommended_card['win_rate']
    if win_rate >= 0.60:
        enhanced += f"📈 EXCELLENT WIN RATE: This card has a {win_rate:.0%} win rate when drafted,\n"
        enhanced += f"   meaning decks with this card win significantly more often.\n\n"
    elif win_rate >= 0.55:
        enhanced += f"📊 GOOD WIN RATE: With a {win_rate:.0%} win rate, this card contributes\n"
        enhanced += f"   positively to your deck's overall performance.\n\n"
    
    # Score explanation
    score = recommended_card['tier_score']
    if score >= 80:
        enhanced += f"🎯 HIGH SCORE: {score:.0f}/100 score indicates this card is\n"
        enhanced += f"   consistently powerful and versatile.\n\n"
    elif score >= 60:
        enhanced += f"📊 GOOD SCORE: {score:.0f}/100 score shows this card is\n"
        enhanced += f"   above average and reliable.\n\n"
    
    # Card-specific notes
    if recommended_card['notes']:
        enhanced += f"🔍 CARD ANALYSIS: {recommended_card['notes']}\n\n"
    
    # Comparison with alternatives
    other_cards = [card for i, card in enumerate(cards) if i != recommended_index]
    if other_cards:
        enhanced += f"⚖️  COMPARISON WITH ALTERNATIVES:\n"
        for i, card in enumerate(other_cards):
            other_name = get_card_name(card['card_code'])
            enhanced += f"   • {other_name} ({card['tier']}-tier, {card['win_rate']:.0%} win rate)\n"
            
            if card['tier_score'] < recommended_card['tier_score']:
                diff = recommended_card['tier_score'] - card['tier_score']
                enhanced += f"     Lower score by {diff:.0f} points - less impactful\n"
            
            if card['win_rate'] < recommended_card['win_rate']:
                enhanced += f"     Lower win rate - less reliable for victories\n"
        enhanced += f"\n"
    
    enhanced += f"🎯 CONCLUSION: {card_name} offers the best combination of\n"
    enhanced += f"   power level, reliability, and win rate for this pick.\n"
    
    return enhanced

def main():
    """Demonstrate the enhanced Arena Bot features."""
    print("🎯 ENHANCED ARENA BOT - FEATURE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Setup
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from arena_bot.ai.draft_advisor import get_draft_advisor
        from arena_bot.core.surf_detector import get_surf_detector
        
        print("✅ Arena Bot components loaded successfully")
        print()
        
        # Initialize components
        advisor = get_draft_advisor()
        surf_detector = get_surf_detector()
        
        # Simulate real-time monitoring
        print("🔍 SIMULATING REAL-TIME MONITORING...")
        print("=" * 50)
        
        # Step 1: Screen Detection Demo
        print("📺 SCREEN DETECTION:")
        current_screen = detect_screen_type()
        print(f"   Current Screen: {current_screen.value} 🎯")
        print("   ✅ Bot recognizes you're in Arena Draft mode!")
        print()
        
        time.sleep(1)
        
        # Step 2: Card Detection Demo
        print("🎮 CARD DETECTION:")
        detected_cards = ['TOY_380', 'ULD_309', 'TTN_042']
        print("   Raw Detection:", detected_cards)
        print("   User-Friendly Names:")
        for i, card_code in enumerate(detected_cards):
            card_name = get_card_name(card_code)
            print(f"   {i+1}. {card_name}")
        print("   ✅ Card codes converted to readable names!")
        print()
        
        time.sleep(1)
        
        # Step 3: Enhanced Analysis Demo
        print("🧠 ENHANCED DRAFT ANALYSIS:")
        choice = advisor.analyze_draft_choice(detected_cards, 'warrior')
        
        recommended_card = choice.cards[choice.recommended_pick]
        recommended_name = get_card_name(recommended_card.card_code)
        
        print(f"   👑 RECOMMENDED PICK: {recommended_name}")
        print(f"   🎯 Confidence Level: {choice.recommendation_level.value.upper()}")
        print()
        
        time.sleep(1)
        
        # Step 4: Detailed Explanation Demo
        enhanced_explanation = enhance_reasoning(
            choice.reasoning, 
            [
                {
                    'card_code': card.card_code,
                    'tier': card.tier_letter,
                    'tier_score': card.tier_score,
                    'win_rate': card.win_rate,
                    'notes': card.notes
                }
                for card in choice.cards
            ], 
            choice.recommended_pick
        )
        
        print(enhanced_explanation)
        
        # Step 5: Complete Card Comparison
        print("\n📊 COMPLETE CARD COMPARISON:")
        print("=" * 50)
        
        for i, card in enumerate(choice.cards):
            is_recommended = (i == choice.recommended_pick)
            marker = "👑 BEST PICK" if is_recommended else "     OPTION"
            card_name = get_card_name(card.card_code)
            
            print(f"{marker}: {card_name}")
            print(f"         Tier: {card.tier_letter} | Score: {card.tier_score:.0f}/100 | Win Rate: {card.win_rate:.0%}")
            if card.notes:
                print(f"         Notes: {card.notes}")
            print()
        
        # Step 6: Real-time Features Summary
        print("🚀 REAL-TIME FEATURES SUMMARY:")
        print("=" * 50)
        print("✅ Screen Detection - Knows what Hearthstone screen you're on")
        print("✅ User-Friendly Names - No more confusing card codes!")
        print("✅ Detailed Explanations - Understand WHY each pick is recommended")
        print("✅ Card Comparisons - See exactly how options compare")
        print("✅ Tier Analysis - Understand card power levels")
        print("✅ Win Rate Data - See which cards actually win games")
        print("✅ Real-Time Updates - Instant recommendations as you draft")
        print()
        
        print("🎉 ENHANCED ARENA BOT READY!")
        print("   This is what you'll see in the real-time overlay version!")
        print("   Much more user-friendly than the original Arena Tracker!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()