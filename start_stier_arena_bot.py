#!/usr/bin/env python3
"""
S-TIER ARENA BOT LAUNCHER
Easy launcher for the S-Tier Arena Bot Production Edition

Simply run this script to start the Arena Bot with full S-Tier logging integration.
All the complex async/GUI coordination is handled automatically.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch the S-Tier Arena Bot."""
    print("🎮 ARENA BOT - S-TIER LAUNCHER")
    print("=" * 50)
    
    try:
        # Import and run the S-Tier Arena Bot
        from arena_bot_s_tier_final import run_stier_arena_bot
        
        print("🚀 Launching S-Tier Arena Bot...")
        print("   • Enterprise-grade logging active")
        print("   • High-performance async architecture")
        print("   • Real-time monitoring enabled")
        print("   • Rich contextual observability")
        print("-" * 50)
        
        # Launch the bot
        run_stier_arena_bot()
        
    except ImportError as e:
        print(f"❌ Failed to import S-Tier Arena Bot: {e}")
        print("\n🔧 Make sure all dependencies are installed:")
        print("   pip install asyncio tkinter pillow numpy")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 S-Tier Arena Bot failed to start: {e}")
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()