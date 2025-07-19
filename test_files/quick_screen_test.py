#!/usr/bin/env python3
"""
Quick test of the screen detection system.
"""

from hearthstone_log_monitor import HearthstoneLogMonitor
import time

def main():
    print("🎯 QUICK SCREEN DETECTION TEST")
    print("=" * 40)
    
    monitor = HearthstoneLogMonitor()
    
    # Simple callback to show screen changes
    def on_screen_change(old_state, new_state):
        print(f"\n🚨 SCREEN CHANGE DETECTED!")
        print(f"   From: {old_state.value}")
        print(f"   To: {new_state.value}")
        print("   (This would show a big banner in the full bot)")
    
    monitor.on_game_state_change = on_screen_change
    
    # Start monitoring
    print("🚀 Starting quick test...")
    monitor.start_monitoring()
    
    # Run for 30 seconds
    for i in range(30):
        print(f"\rTesting... {30-i}s remaining", end="", flush=True)
        time.sleep(1)
    
    # Show final state
    print(f"\n\n📊 FINAL STATE:")
    state = monitor.get_current_state()
    print(f"   Screen: {state['game_state']}")
    print(f"   Logs: {state['available_logs']}")
    print(f"   Picks: {state['draft_picks_count']}")
    
    monitor.stop_monitoring()
    print("✅ Quick test completed!")

if __name__ == "__main__":
    main()