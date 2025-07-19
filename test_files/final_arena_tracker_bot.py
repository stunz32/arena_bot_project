#!/usr/bin/env python3
"""
FINAL ARENA TRACKER BOT - Complete Implementation
✅ Log file monitoring (Arena Tracker methodology)
✅ Prominent screen detection (VERY OBVIOUS displays)
✅ Real card names and detailed explanations
✅ Smart log directory discovery
✅ Real-time draft pick detection
✅ No external dependencies
"""

import sys
import time
import threading
from pathlib import Path
from hearthstone_log_monitor import HearthstoneLogMonitor, GameState, DraftPick

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

class FinalArenaTrackerBot:
    """
    Complete Arena Bot using Arena Tracker's proven methodology.
    Features PROMINENT screen detection and comprehensive draft assistance.
    """
    
    def __init__(self):
        """Initialize the final Arena Tracker bot."""
        print("🎯 FINAL ARENA TRACKER BOT")
        print("=" * 80)
        print("✅ Features enabled:")
        print("   • Log file monitoring (Arena Tracker methodology)")
        print("   • PROMINENT screen detection (impossible to miss!)")
        print("   • Real card names instead of codes")
        print("   • Detailed draft recommendations")
        print("   • Smart log directory discovery") 
        print("   • Real-time draft pick detection")
        print("=" * 80)
        
        # Initialize log monitor
        self.log_monitor = HearthstoneLogMonitor()
        self.setup_enhanced_callbacks()
        
        # Initialize draft advisor
        try:
            from arena_bot.ai.draft_advisor import get_draft_advisor
            self.advisor = get_draft_advisor()
            print("✅ Draft advisor loaded successfully")
        except Exception as e:
            print(f"⚠️ Draft advisor not available: {e}")
            self.advisor = None
        
        # Enhanced card database
        self.card_names = self.load_enhanced_card_database()
        
        # State tracking
        self.draft_recommendations_enabled = True
        self.session_stats = {
            'screen_changes': 0,
            'draft_picks_detected': 0,
            'recommendations_given': 0,
            'session_start': time.time()
        }
        
        print("🚀 Final Arena Tracker Bot ready!")
    
    def load_enhanced_card_database(self) -> dict:
        """Load comprehensive card name database."""
        try:
            from arena_bot.data.card_names import CARD_NAMES
            print(f"✅ Loaded {len(CARD_NAMES)} card names from database")
            return CARD_NAMES
        except ImportError:
            # Enhanced fallback database with your actual cards
            return {
                'TOY_380': 'Toy Captain Tarim',
                'ULD_309': 'Dragonqueen Alexstrasza',
                'TTN_042': 'Thassarian',
                'EDR_464': 'Sparky Apprentice',
                'EDR_476': 'Thunderous Heart', 
                'EDR_461': 'Crystal Cluster',
                'UNG_854': 'Obsidian Shard',
                'TTN_715': 'Sanguine Depths',
                'CORE_CS1_112': 'Holy Nova',
                'GDB_311': 'Moonlit Guidance',
                'VAC_404': 'Mystery Winner',
                'BAR_310': 'Battleground Battlemaster',
                'LOOT_410': 'Plated Beetle',
                'GDB_439': 'Crimson Sigil Runner',
                'GDB_132': 'Felfire Deadeye',
                'SCH_233': 'Goody Two-Shields',
                'LOOT_008': 'Mithril Spellstone',
                'BT_252': 'Inner Demon',
                'AV_328': 'Bloodsail Deckhand',
                'CORE_ULD_723': 'Mogu Cultist',
                'EDR_462': 'Spark Engine',
                'TTN_812': 'Grave Defiler',
                'DMF_187': 'Deck of Lunacy',
                'TTN_700': 'Catacomb Guard',
                'CFM_605': 'Lotus Illusionist',
                'WW_393': 'Tentacle Swarm',
                'TLC_820': 'Anub\'Rekhan',
                'TOY_386': 'Toy Gyrocopter',
                'HERO_09y': 'Rexxar (Hunter)',
                'HERO_01': 'Garrosh (Warrior)',
                'HERO_02': 'Jaina (Mage)',
                'HERO_03': 'Uther (Paladin)',
                'HERO_04': 'Rexxar (Hunter)',
                'HERO_05': 'Valeera (Rogue)',
                'HERO_06': 'Thrall (Shaman)',
                'HERO_07': 'Guldan (Warlock)',
                'HERO_08': 'Anduin (Priest)',
                'HERO_09': 'Malfurion (Druid)',
            }
    
    def get_card_display_name(self, card_code: str) -> str:
        """Get enhanced card display name with premium indicator."""
        clean_code = card_code.replace('_premium', '')
        
        if clean_code in self.card_names:
            name = self.card_names[clean_code]
            if '_premium' in card_code:
                return f"{name} ✨ (Golden)"
            return name
        return f"Unknown Card ({clean_code})"
    
    def setup_enhanced_callbacks(self):
        """Setup enhanced callbacks with prominent displays."""
        
        def on_screen_change(old_state, new_state):
            self.session_stats['screen_changes'] += 1
            
            # Show current bot mode based on screen
            print(f"\n🤖 BOT MODE UPDATE:")
            if new_state == GameState.HUB:
                print("   Status: STANDBY - Ready for Arena")
                print("   Action: Waiting for you to start an Arena draft")
            elif new_state == GameState.ARENA_DRAFT:
                print("   Status: ACTIVE - Monitoring draft picks")
                print("   Action: Will provide recommendations for each pick")
            elif new_state == GameState.GAMEPLAY:
                print("   Status: MATCH MODE - Observing gameplay")
                print("   Action: Arena Bot on standby during match")
            elif new_state == GameState.COLLECTION:
                print("   Status: BROWSING - Collection mode")
                print("   Action: Arena Bot waiting")
            else:
                print(f"   Status: MONITORING - In {new_state.value}")
                print("   Action: Watching for Arena activity")
        
        def on_draft_start():
            print("\n" + "🚀" * 25)
            print("🚀 ARENA DRAFT SESSION STARTED! 🚀")
            print("🚀" * 25)
            print("🎯 Bot will now provide detailed recommendations for each pick!")
            print("📊 Recommendations include card names, tier scores, and explanations")
            print()
        
        def on_draft_pick(pick: DraftPick):
            self.session_stats['draft_picks_detected'] += 1
            
            # Display prominent pick notification
            pick_num = len(self.log_monitor.current_draft_picks)
            card_name = self.get_card_display_name(pick.card_code)
            
            print("\n" + "🎯" * 70)
            print("🎯" + " " * 66 + "🎯")
            print(f"🎯{' ' * 15}DRAFT PICK #{pick_num:2d} DETECTED!{' ' * 15}🎯")
            print("🎯" + " " * 66 + "🎯")
            print("🎯" * 70)
            
            print(f"\n📋 PICK DETAILS:")
            print(f"   🎯 Pick: #{pick_num}/30")
            print(f"   🃏 Card: {card_name}")
            print(f"   🔢 Code: {pick.card_code}")
            print(f"   🎰 Slot: {pick.slot}")
            print(f"   ✨ Premium: {'YES (Golden!)' if pick.is_premium else 'No'}")
            print(f"   🕒 Time: {pick.timestamp.strftime('%H:%M:%S')}")
            
            # Provide recommendation if advisor available
            if self.advisor and self.draft_recommendations_enabled:
                self.provide_enhanced_recommendation(pick)
        
        def on_draft_complete(picks):
            print("\n" + "🏆" * 50)
            print("🏆" + " " * 46 + "🏆")
            print("🏆" + " " * 15 + "DRAFT COMPLETED!" + " " * 15 + "🏆")
            print("🏆" + " " * 46 + "🏆")
            print("🏆" * 50)
            
            print(f"\n📊 DRAFT SUMMARY:")
            print(f"   🎯 Total Picks: {len(picks)}")
            print(f"   ⏱️ Duration: {self.calculate_draft_duration()}")
            print(f"   👑 Hero: {self.get_card_display_name(self.log_monitor.current_hero) if self.log_monitor.current_hero else 'Unknown'}")
            
            # Show final deck preview
            print(f"\n🎮 FINAL DECK PREVIEW (Last 5 picks):")
            for i, pick in enumerate(picks[-5:], 1):
                card_name = self.get_card_display_name(pick.card_code)
                premium = " ✨" if pick.is_premium else ""
                print(f"   {len(picks)-5+i:2d}. {card_name}{premium}")
        
        # Connect all callbacks
        self.log_monitor.on_game_state_change = on_screen_change
        self.log_monitor.on_draft_start = on_draft_start
        self.log_monitor.on_draft_pick = on_draft_pick
        self.log_monitor.on_draft_complete = on_draft_complete
    
    def provide_enhanced_recommendation(self, pick: DraftPick):
        """Provide detailed recommendation for the draft pick."""
        try:
            self.session_stats['recommendations_given'] += 1
            
            print(f"\n💭 ANALYZING PICK #{len(self.log_monitor.current_draft_picks)}...")
            print("=" * 60)
            
            # Get recent picks for context
            recent_picks = [p.card_code for p in self.log_monitor.current_draft_picks[-3:]]
            
            # Get recommendation
            choice = self.advisor.analyze_draft_choice(recent_picks, 'hunter')
            
            # Display recommendation
            recommended_card = choice.cards[choice.recommended_pick]
            recommended_name = self.get_card_display_name(recommended_card.card_code)
            
            print(f"👑 RECOMMENDED PICK: {recommended_name}")
            print(f"📊 Analysis:")
            print(f"   🏆 Tier: {recommended_card.tier_letter}")
            print(f"   📈 Score: {recommended_card.tier_score:.0f}/100")
            print(f"   🎯 Win Rate: {recommended_card.win_rate:.0%}")
            print(f"   🔍 Confidence: {choice.recommendation_level.value.upper()}")
            
            if recommended_card.notes:
                print(f"   💡 Notes: {recommended_card.notes}")
            
            print(f"\n💭 REASONING:")
            print(f"   {choice.reasoning}")
            
            # Show all options comparison
            print(f"\n📋 ALL OPTIONS:")
            for i, card in enumerate(choice.cards):
                is_recommended = (i == choice.recommended_pick)
                marker = "👑 BEST" if is_recommended else "📋 ALT "
                card_name = self.get_card_display_name(card.card_code)
                
                print(f"   {marker}: {card_name}")
                print(f"      Tier {card.tier_letter} • {card.win_rate:.0%} win rate • {card.tier_score:.0f}/100")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Recommendation error: {e}")
    
    def calculate_draft_duration(self) -> str:
        """Calculate draft duration."""
        duration = time.time() - self.session_stats['session_start']
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s"
    
    def display_session_stats(self):
        """Display session statistics."""
        print(f"\n📊 SESSION STATISTICS:")
        print(f"   🔄 Screen Changes: {self.session_stats['screen_changes']}")
        print(f"   🎯 Draft Picks Detected: {self.session_stats['draft_picks_detected']}")
        print(f"   💭 Recommendations Given: {self.session_stats['recommendations_given']}")
        print(f"   ⏱️ Session Duration: {self.calculate_draft_duration()}")
    
    def run(self):
        """Run the complete Arena Tracker bot."""
        print(f"\n🚀 STARTING FINAL ARENA TRACKER BOT")
        print("=" * 80)
        print("🎯 This bot uses Arena Tracker's proven methodology:")
        print("   📖 Monitors Hearthstone log files for authoritative data")
        print("   📺 Detects screen changes with PROMINENT displays")
        print("   🎯 Provides detailed draft recommendations")
        print("   💎 Shows real card names instead of cryptic codes")
        print("   🤖 Updates bot status based on current screen")
        print()
        print("🎮 How to use:")
        print("   1. Open Hearthstone")
        print("   2. Navigate to different screens (you'll see big notifications!)")
        print("   3. Start an Arena draft")
        print("   4. Get instant recommendations with detailed explanations")
        print("=" * 80)
        
        # Start log monitoring
        self.log_monitor.start_monitoring()
        
        # Show initial state
        print(f"\n📊 INITIAL STATE:")
        state = self.log_monitor.get_current_state()
        print(f"   📺 Current Screen: {state['game_state']}")
        print(f"   📁 Log Directory: {state['log_directory']}")
        print(f"   📖 Available Logs: {', '.join(state['available_logs'])}")
        
        try:
            print(f"\n✅ FINAL ARENA TRACKER BOT IS RUNNING!")
            print("👀 Watching for Hearthstone activity...")
            print("⏸️  Press Ctrl+C to stop")
            
            # Main monitoring loop
            heartbeat_counter = 0
            while True:
                time.sleep(10)  # Heartbeat every 10 seconds
                heartbeat_counter += 1
                
                if heartbeat_counter % 6 == 0:  # Every minute
                    current_state = self.log_monitor.get_current_state()
                    print(f"\n💓 Bot Status: Active in {current_state['game_state']}")
                    
                    if current_state['draft_picks_count'] > 0:
                        progress = f"{current_state['draft_picks_count']}/30"
                        print(f"   🎯 Draft Progress: {progress}")
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 STOPPING FINAL ARENA TRACKER BOT...")
            print("=" * 80)
            
            # Show session summary
            final_state = self.log_monitor.get_current_state()
            print(f"📊 FINAL SESSION SUMMARY:")
            print(f"   📺 Final Screen: {final_state['game_state']}")
            print(f"   🎯 Total Draft Picks: {final_state['draft_picks_count']}")
            print(f"   👑 Hero: {final_state['current_hero'] or 'None'}")
            
            if final_state['recent_picks']:
                print(f"   🎮 Recent Picks:")
                for pick in final_state['recent_picks']:
                    card_name = self.get_card_display_name(pick['card_code'])
                    premium = " ✨" if pick['is_premium'] else ""
                    print(f"      • {card_name}{premium}")
            
            self.display_session_stats()
            self.log_monitor.stop_monitoring()
            
            print("=" * 80)
            print("✅ Final Arena Tracker Bot stopped successfully")
            print("🎯 Thanks for using the Arena Tracker style bot!")
            print("📊 This bot used Arena Tracker's complete methodology")

def main():
    """Run the final Arena Tracker bot."""
    bot = FinalArenaTrackerBot()
    bot.run()

if __name__ == "__main__":
    main()