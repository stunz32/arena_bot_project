#!/usr/bin/env python3
"""
INTEGRATED ARENA BOT - Complete System
Combines all built systems: visual detection, log monitoring, AI recommendations, window detection
"""

import sys
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

class IntegratedArenaBot:
    """
    Complete Arena Bot integrating all systems:
    - Visual card detection (histogram + template matching)
    - Log monitoring (Arena Tracker style)
    - AI recommendations (draft advisor)
    - Automatic window detection
    """
    
    def __init__(self):
        """Initialize the integrated arena bot."""
        print("🚀 INTEGRATED ARENA BOT - COMPLETE SYSTEM")
        print("=" * 80)
        print("✅ Combining all built systems:")
        print("   • Visual card detection (histogram + template matching)")
        print("   • Log monitoring (Arena Tracker methodology)")
        print("   • AI recommendations (draft advisor)")
        print("   • Automatic Hearthstone window detection")
        print("=" * 80)
        
        # Initialize all subsystems
        self.init_visual_detection()
        self.init_log_monitoring()
        self.init_ai_advisor()
        self.init_window_detection()
        
        # State management
        self.running = False
        self.current_screen = "Unknown"
        self.in_draft = False
        self.last_visual_check = 0
        self.visual_check_interval = 2.0  # Check screen every 2 seconds
        
        print("🎯 Integrated Arena Bot ready!")
    
    def init_visual_detection(self):
        """Initialize the visual card detection system."""
        try:
            from arena_bot.core.card_recognizer import get_card_recognizer
            from arena_bot.core.screen_detector import get_screen_detector
            
            self.card_recognizer = get_card_recognizer()
            self.screen_detector = get_screen_detector()
            
            print("✅ Visual detection system loaded")
        except Exception as e:
            print(f"⚠️ Visual detection not available: {e}")
            self.card_recognizer = None
            self.screen_detector = None
    
    def init_log_monitoring(self):
        """Initialize the log monitoring system."""
        try:
            from hearthstone_log_monitor import HearthstoneLogMonitor
            
            self.log_monitor = HearthstoneLogMonitor()
            self.setup_log_callbacks()
            
            print("✅ Log monitoring system loaded")
        except Exception as e:
            print(f"⚠️ Log monitoring not available: {e}")
            self.log_monitor = None
    
    def init_ai_advisor(self):
        """Initialize the AI draft advisor."""
        try:
            from arena_bot.ai.draft_advisor import get_draft_advisor
            
            self.advisor = get_draft_advisor()
            print("✅ AI draft advisor loaded")
        except Exception as e:
            print(f"⚠️ AI advisor not available: {e}")
            self.advisor = None
    
    def init_window_detection(self):
        """Initialize window detection system."""
        try:
            from arena_bot.core.window_detector import get_window_detector
            
            self.window_detector = get_window_detector()
            print("✅ Window detection system loaded")
        except Exception as e:
            print(f"⚠️ Window detection not available: {e}")
            self.window_detector = None
    
    def setup_log_callbacks(self):
        """Setup callbacks for log monitoring."""
        if not self.log_monitor:
            return
        
        def on_draft_start():
            print("\n🎯 LOG: Arena draft started - enabling visual monitoring")
            self.in_draft = True
        
        def on_draft_complete(picks):
            print("\n🏆 LOG: Arena draft completed")
            self.in_draft = False
        
        def on_game_state_change(old_state, new_state):
            print(f"🎮 LOG: Game state: {old_state.value} → {new_state.value}")
            self.in_draft = (new_state.value == "Arena Draft")
        
        def on_draft_pick(pick):
            print(f"📋 LOG: Pick detected - {pick.card_code}")
        
        self.log_monitor.on_draft_start = on_draft_start
        self.log_monitor.on_draft_complete = on_draft_complete
        self.log_monitor.on_game_state_change = on_game_state_change
        self.log_monitor.on_draft_pick = on_draft_pick
    
    def find_hearthstone_window(self):
        """Find Hearthstone window on screen."""
        if not self.window_detector:
            return None
        
        try:
            windows = self.window_detector.find_hearthstone_windows()
            if windows:
                return windows[0]  # Use first window found
            return None
        except Exception as e:
            print(f"❌ Window detection error: {e}")
            return None
    
    def capture_hearthstone_screen(self):
        """Capture Hearthstone screen if window is found."""
        if not self.screen_detector:
            return None
        
        try:
            # Try to find Hearthstone window
            window_info = self.find_hearthstone_window()
            
            if window_info:
                # Capture specific window region
                screenshot = self.screen_detector.capture_region(
                    window_info['x'], window_info['y'],
                    window_info['width'], window_info['height']
                )
                return screenshot
            else:
                # Fallback: capture full screen
                return self.screen_detector.capture_screen()
                
        except Exception as e:
            print(f"❌ Screen capture error: {e}")
            return None
    
    def detect_arena_interface(self, screenshot):
        """Detect if arena draft interface is visible."""
        if not screenshot:
            return False
        
        try:
            # Use existing SURF detection or simple color analysis
            if hasattr(self.card_recognizer, 'detect_arena_interface'):
                return self.card_recognizer.detect_arena_interface(screenshot)
            
            # Fallback: simple color detection for arena interface
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # Look for arena's characteristic red/brown colors
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            red_pixels = cv2.countNonZero(red_mask)
            total_pixels = screenshot.shape[0] * screenshot.shape[1]
            red_percentage = red_pixels / total_pixels
            
            return red_percentage > 0.03  # At least 3% red pixels indicates arena
            
        except Exception as e:
            print(f"❌ Arena interface detection error: {e}")
            return False
    
    def analyze_draft_cards(self, screenshot):
        """Analyze and provide recommendations for draft cards."""
        if not self.card_recognizer or not screenshot:
            return None
        
        try:
            print("🔍 Analyzing draft cards...")
            
            # Use the card recognition system to detect cards
            detection_result = self.card_recognizer.detect_cards(screenshot)
            
            if detection_result and detection_result.get('success'):
                detected_cards = detection_result.get('cards', [])
                
                if len(detected_cards) == 3:
                    # Get AI recommendation
                    if self.advisor:
                        card_codes = [card.get('card_code', '') for card in detected_cards]
                        choice = self.advisor.analyze_draft_choice(card_codes, 'unknown')
                        
                        return {
                            'cards': detected_cards,
                            'recommendation': choice,
                            'success': True
                        }
                    else:
                        return {
                            'cards': detected_cards,
                            'recommendation': None,
                            'success': True
                        }
            
            return {'success': False, 'error': 'Could not detect 3 cards'}
            
        except Exception as e:
            print(f"❌ Card analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def display_recommendation(self, analysis):
        """Display draft recommendation."""
        if not analysis or not analysis.get('success'):
            print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
            return
        
        cards = analysis['cards']
        recommendation = analysis.get('recommendation')
        
        print("\n" + "🎯" * 60)
        print("🎯 ARENA DRAFT ANALYSIS")
        print("🎯" * 60)
        
        print(f"\n📋 DETECTED CARDS:")
        for i, card in enumerate(cards):
            card_code = card.get('card_code', 'Unknown')
            confidence = card.get('confidence', 0)
            print(f"   {i+1}. {card_code} (confidence: {confidence:.1%})")
        
        if recommendation:
            rec_pick = recommendation.recommended_pick + 1
            rec_card = recommendation.cards[recommendation.recommended_pick]
            
            print(f"\n👑 RECOMMENDED PICK: Card {rec_pick}")
            print(f"🎯 CARD: {rec_card.card_code}")
            print(f"📊 TIER: {rec_card.tier_letter}")
            print(f"💯 SCORE: {rec_card.tier_score:.0f}/100")
            print(f"📈 WIN RATE: {rec_card.win_rate:.0%}")
            
            print(f"\n💭 REASONING:")
            print(f"   {recommendation.reasoning}")
        
        print("🎯" * 60)
    
    def visual_monitoring_loop(self):
        """Visual monitoring loop - checks screen periodically."""
        print("👁️ Starting visual monitoring...")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check screen every N seconds
                if current_time - self.last_visual_check >= self.visual_check_interval:
                    self.last_visual_check = current_time
                    
                    # Capture screen
                    screenshot = self.capture_hearthstone_screen()
                    
                    if screenshot is not None:
                        # Check if in arena draft
                        in_arena = self.detect_arena_interface(screenshot)
                        
                        if in_arena and self.in_draft:
                            # Analyze cards and provide recommendation
                            analysis = self.analyze_draft_cards(screenshot)
                            if analysis and analysis.get('success'):
                                self.display_recommendation(analysis)
                        
                        # Update current screen status
                        new_screen = "Arena Draft" if in_arena else "Other"
                        if new_screen != self.current_screen:
                            print(f"📺 Visual: Screen changed to {new_screen}")
                            self.current_screen = new_screen
                
                time.sleep(0.5)  # Short sleep between checks
                
            except Exception as e:
                print(f"❌ Visual monitoring error: {e}")
                time.sleep(2)
    
    def start(self):
        """Start the integrated arena bot."""
        print(f"\n🚀 STARTING INTEGRATED ARENA BOT")
        print("=" * 80)
        print("🎯 This bot combines ALL your built systems:")
        print("   📖 Log monitoring for authoritative draft state")
        print("   📺 Visual detection for real-time card analysis")
        print("   🤖 AI recommendations for optimal picks")
        print("   🔍 Automatic window detection")
        print()
        print("🎮 How to use:")
        print("   1. Open Hearthstone")
        print("   2. Start an Arena draft")
        print("   3. Get instant visual + log-based recommendations")
        print("=" * 80)
        
        self.running = True
        
        # Start log monitoring
        if self.log_monitor:
            self.log_monitor.start_monitoring()
            print("✅ Log monitoring started")
        
        # Start visual monitoring in separate thread
        if self.card_recognizer and self.screen_detector:
            self.visual_thread = threading.Thread(target=self.visual_monitoring_loop, daemon=True)
            self.visual_thread.start()
            print("✅ Visual monitoring started")
        
        try:
            print(f"\n✅ INTEGRATED ARENA BOT IS RUNNING!")
            print("👀 Monitoring both logs and screen...")
            print("⏸️  Press Ctrl+C to stop")
            
            # Main loop
            heartbeat_counter = 0
            while True:
                time.sleep(10)
                heartbeat_counter += 1
                
                if heartbeat_counter % 6 == 0:  # Every minute
                    status = "ACTIVE" if self.in_draft else "STANDBY"
                    print(f"\n💓 Bot Status: {status} | Screen: {self.current_screen}")
                    
                    if self.log_monitor:
                        state = self.log_monitor.get_current_state()
                        if state['draft_picks_count'] > 0:
                            print(f"   🎯 Draft Progress: {state['draft_picks_count']}/30")
        
        except KeyboardInterrupt:
            print(f"\n\n🛑 STOPPING INTEGRATED ARENA BOT...")
            self.stop()
    
    def stop(self):
        """Stop the integrated arena bot."""
        self.running = False
        
        if self.log_monitor:
            self.log_monitor.stop_monitoring()
        
        print("✅ Integrated Arena Bot stopped successfully")
        print("🎯 Thanks for using the complete Arena Bot system!")

def main():
    """Run the integrated arena bot."""
    bot = IntegratedArenaBot()
    bot.start()

if __name__ == "__main__":
    main()