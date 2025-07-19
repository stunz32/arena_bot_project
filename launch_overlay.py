#!/usr/bin/env python3
"""
Launch the Arena Bot overlay interface.
"""

import sys
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

from arena_bot.ui.draft_overlay import create_draft_overlay, OverlayConfig

def main():
    """Launch the draft overlay."""
    print("🎯 Arena Bot - Real-Time Draft Assistant")
    print("=" * 50)
    print()
    print("Starting overlay interface...")
    print("• The overlay will appear in the top-right corner")
    print("• Click 'Update' to analyze the current draft")
    print("• Use 'Settings' to adjust opacity and update frequency")
    print("• The overlay stays on top of all windows")
    print()
    print("Press Ctrl+C or close the overlay window to exit.")
    print()
    
    # Configure overlay
    config = OverlayConfig(
        opacity=0.9,
        update_interval=3.0,
        show_tier_scores=True,
        show_win_rates=True,
        font_size=10
    )
    
    # Create and start overlay
    overlay = create_draft_overlay(config)
    
    try:
        overlay.start()
    except KeyboardInterrupt:
        print("\n✨ Arena Bot overlay stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Overlay error: {e}")

if __name__ == "__main__":
    # Set up minimal logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise for the overlay
        format='%(levelname)s - %(message)s'
    )
    
    main()