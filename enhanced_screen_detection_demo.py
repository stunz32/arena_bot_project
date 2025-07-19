#!/usr/bin/env python3
"""
Enhanced Screen Detection Demo
Shows prominent, obvious screen detection from Hearthstone logs.
"""

import time
from hearthstone_log_monitor import HearthstoneLogMonitor

def main():
    print("🎯 ENHANCED HEARTHSTONE SCREEN DETECTION")
    print("=" * 80)
    print("This demo shows PROMINENT, OBVIOUS screen detection from logs")
    print("You'll see big, clear notifications when you change screens!")
    print("=" * 80)
    
    monitor = HearthstoneLogMonitor()
    
    # Enhanced callbacks with prominent displays
    def on_state_change(old_state, new_state):
        print(f"\n🔄 STATE TRANSITION: {old_state.value} ➜ {new_state.value}")
        
        # Show what the bot is doing in each state
        if new_state.value == "Main Menu":
            print("🤖 Bot Status: STANDBY - Waiting for Arena")
        elif new_state.value == "Arena Draft":
            print("🤖 Bot Status: ACTIVE - Monitoring draft picks")
        elif new_state.value == "Playing Match":
            print("🤖 Bot Status: WATCHING - Match in progress")
        elif new_state.value == "Collection":
            print("🤖 Bot Status: IDLE - Collection browsing")
        else:
            print(f"🤖 Bot Status: MONITORING - In {new_state.value}")
    
    def on_draft_pick(pick):
        # Super prominent draft pick display
        print("\n" + "🎯" * 60)
        print("🎯" + " " * 56 + "🎯")
        print("🎯" + " " * 20 + "DRAFT PICK DETECTED!" + " " * 20 + "🎯")
        print("🎯" + " " * 56 + "🎯")
        print("🎯" * 60)
        
        print(f"\n📋 PICK DETAILS:")
        print(f"   🎯 Pick Number: #{len(monitor.current_draft_picks)}")
        print(f"   🃏 Card Code: {pick.card_code}")
        print(f"   🎰 Slot: {pick.slot}")
        print(f"   ✨ Premium: {'Yes (Golden!)' if pick.is_premium else 'No'}")
        print(f"   🕒 Time: {pick.timestamp.strftime('%H:%M:%S')}")
        print()
    
    def on_draft_start():
        print("🚨 DRAFT DETECTION: Arena draft process initiated!")
    
    # Connect callbacks
    monitor.on_game_state_change = on_state_change
    monitor.on_draft_pick = on_draft_pick
    monitor.on_draft_start = on_draft_start
    
    # Show initial state
    print("\n📊 INITIAL DETECTION:")
    state = monitor.get_current_state()
    print(f"   📺 Current Screen: {state['game_state']}")
    print(f"   📁 Log Directory: {state['log_directory']}")
    print(f"   📖 Available Logs: {state['available_logs']}")
    
    # Start monitoring
    monitor.start_monitoring()
    
    print(f"\n🚀 STARTING ENHANCED MONITORING...")
    print("=" * 80)
    print("🎮 INSTRUCTIONS:")
    print("   1. Open Hearthstone")
    print("   2. Navigate between screens (Main Menu ↔ Arena ↔ Collection)")
    print("   3. Watch for BIG, OBVIOUS screen detection notifications!")
    print("   4. Start an Arena draft to see pick detection")
    print("=" * 80)
    print()
    print("👀 Watching for screen changes...")
    print("⏸️  Press Ctrl+C to stop")
    
    try:
        # Show live status updates
        counter = 0
        while True:
            time.sleep(5)  # Update every 5 seconds
            counter += 1
            
            # Show heartbeat every 30 seconds
            if counter % 6 == 0:
                current_state = monitor.get_current_state()
                print(f"\n💓 HEARTBEAT #{counter//6}: Currently in {current_state['game_state']}")
                
                if current_state['draft_picks_count'] > 0:
                    print(f"   🎯 Draft Progress: {current_state['draft_picks_count']}/30 picks")
                    
                if current_state['recent_picks']:
                    print(f"   🎮 Recent Pick: {current_state['recent_picks'][-1]['card_code']}")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 STOPPING ENHANCED MONITOR...")
        print("=" * 80)
        
        # Show final summary
        final_state = monitor.get_current_state()
        print(f"📊 FINAL SESSION SUMMARY:")
        print(f"   📺 Final Screen: {final_state['game_state']}")
        print(f"   🎯 Total Draft Picks: {final_state['draft_picks_count']}")
        print(f"   👑 Hero: {final_state['current_hero'] or 'None selected'}")
        
        if final_state['recent_picks']:
            print(f"   🎮 Last 3 Picks:")
            for pick in final_state['recent_picks'][-3:]:
                premium = " ✨" if pick['is_premium'] else ""
                print(f"      • {pick['card_code']}{premium}")
        
        monitor.stop_monitoring()
        print("✅ Enhanced monitor stopped")
        print("👋 Thanks for testing the screen detection system!")

if __name__ == "__main__":
    main()