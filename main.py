#!/usr/bin/env python3
"""
Hearthstone Arena Draft Bot - Main Entry Point

A simple, clean entry point for the Arena Bot application.
Follows the CLAUDE.md principle of keeping changes minimal and focused.
"""

import sys
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

from arena_bot.utils.logging_config import setup_logging
from arena_bot.utils.config import load_config


def main():
    """Main entry point for the Arena Bot application."""
    print("🎮 Hearthstone Arena Draft Bot")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize card recognition system
        print("🔄 Initializing card recognition system...")
        from arena_bot.core.card_recognizer import get_card_recognizer
        
        card_recognizer = get_card_recognizer()
        
        if card_recognizer.initialize():
            print("✅ Card recognition system initialized!")
            
            # Display system statistics
            stats = card_recognizer.get_detection_stats()
            print(f"📊 System Stats:")
            print(f"   - Histogram database: {stats['histogram_database_size']} cards")
            print(f"   - Mana templates: {stats['template_counts'][0]}")
            print(f"   - Rarity templates: {stats['template_counts'][1]}")
            print(f"   - Available screens: {stats['screen_count']}")
            
            # Test detection (if user wants to)
            print("\n🔍 Arena Bot is ready for card detection!")
            print("📝 The system includes:")
            print("   - Arena Tracker's proven histogram matching")
            print("   - Template matching for mana cost and rarity")
            print("   - Validation engine for accuracy")
            print("   - Support for Underground mode (ready for implementation)")
            
            logger.info("Arena Bot fully operational")
            
        else:
            print("❌ Failed to initialize card recognition system")
            logger.error("Card recognition system initialization failed")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to initialize Arena Bot: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()