#!/usr/bin/env python3
"""
Test the Hearthstone log monitoring system.
This tests the core Arena Tracker methodology - log file monitoring.
"""

import time
from hearthstone_log_monitor import HearthstoneLogMonitor

def main():
    print("🎯 TESTING HEARTHSTONE LOG MONITOR")
    print("=" * 50)
    
    monitor = HearthstoneLogMonitor()
    
    # Set up callbacks
    def on_state_change(old_state, new_state):
        print(f"\n📺 SCREEN CHANGE: {old_state.value} -> {new_state.value}")
    
    def on_draft_pick(pick):
        print(f"\n🎯 DRAFT PICK DETECTED!")
        print(f"   Slot: {pick.slot}")
        print(f"   Card: {pick.card_code}")
        print(f"   Premium: {'Yes ✨' if pick.is_premium else 'No'}")
        print(f"   Time: {pick.timestamp}")
    
    def on_draft_start():
        print(f"\n🚀 ARENA DRAFT STARTED!")
    
    monitor.on_game_state_change = on_state_change
    monitor.on_draft_pick = on_draft_pick
    monitor.on_draft_start = on_draft_start
    
    # Check current state first
    print("\n📊 INITIAL STATE CHECK:")
    state = monitor.get_current_state()
    for key, value in state.items():
        print(f"   {key}: {value}")
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        print(f"\n🔍 Monitoring logs for changes...")
        print("   • Start a Hearthstone Arena draft to see real-time detection")
        print("   • The bot will detect picks from Arena.log automatically")
        print("   • Press Ctrl+C to stop")
        
        for i in range(60):  # Monitor for 60 seconds
            print(f"\rMonitoring... {60-i}s remaining", end="", flush=True)
            time.sleep(1)
            
        print(f"\n\n📊 FINAL STATE:")
        final_state = monitor.get_current_state()
        for key, value in final_state.items():
            print(f"   {key}: {value}")
            
    except KeyboardInterrupt:
        print(f"\n⏸️ Interrupted by user")
    
    finally:
        monitor.stop_monitoring()
        print("✅ Test completed")

if __name__ == "__main__":
    main()