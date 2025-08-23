#!/usr/bin/env python3
"""
Hearthstone Arena Draft Bot - Main Entry Point

A simple, clean entry point for the Arena Bot application.
Follows the CLAUDE.md principle of keeping changes minimal and focused.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

from arena_bot.utils.logging_config import setup_logging
from arena_bot.utils.config import load_config
from arena_bot.utils.debug_dump import begin_run, end_run, debug_run
from arena_bot.cli import run_replay_from_cli


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hearthstone Arena Draft Bot - Real-time card detection and AI recommendations"
    )
    
    parser.add_argument(
        "--debug-tag",
        type=str,
        help="Label debug run with specified tag for easy identification"
    )
    
    parser.add_argument(
        "--replay",
        type=str,
        help="Replay mode: process screenshots from directory or glob pattern"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        choices=["live", "replay"],
        default="live",
        help="Source mode: 'live' for real capture, 'replay' for offline processing (default: live)"
    )
    
    parser.add_argument(
        "--live-smoke",
        action="store_true",
        help="Live smoke mode: perform single capture-and-render pass, print diag, and exit"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: prevent network calls, use cached/static data"
    )
    
    parser.add_argument(
        "--diag",
        action="store_true", 
        help="Diagnostics mode: print per-stage timing and performance info"
    )
    
    parser.add_argument(
        "--ui-safe-demo",
        action="store_true",
        help="UI Safe Demo mode: render diagnostic UI elements independent of CV/AI data"
    )
    
    parser.add_argument(
        "--ui-doctor",
        action="store_true",
        help="UI Doctor mode: diagnose UI health and exit with status code"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the Arena Bot application."""
    # Parse command line arguments
    args = parse_args()
    
    print("🎮 Hearthstone Arena Draft Bot")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Start debug run if requested
    debug_run_context = None
    if args.debug_tag:
        print(f"🐛 Debug mode: {args.debug_tag}")
        debug_run_context = debug_run(args.debug_tag)
        debug_run_context.__enter__()
        logger.info(f"Started debug run: {args.debug_tag}")
    
    # Print mode flags
    if args.replay:
        print(f"📼 Replay mode: {args.replay}")
    if args.source == "live":
        print("📱 Source: Live capture")
    elif args.source == "replay":
        print("📼 Source: Replay mode")
    if args.live_smoke:
        print("🧪 Live smoke mode: single capture and exit")
    if args.offline:
        print("🌐 Offline mode: no network calls")
    if args.diag:
        print("📊 Diagnostics mode: timing enabled")
    if args.ui_safe_demo:
        print("🎨 UI Safe Demo mode: diagnostic rendering enabled")
    if args.ui_doctor:
        print("🩺 UI Doctor mode: diagnostic UI health check")
    
    # Handle replay mode (legacy support)
    if args.replay:
        run_replay_from_cli(args)
        return
    
    # Handle source modes
    if args.source == "replay" and not args.replay:
        print("❌ Error: --source=replay requires --replay argument")
        sys.exit(1)
    
    # Handle UI doctor mode
    if args.ui_doctor:
        from arena_bot.cli import run_ui_doctor_from_cli
        run_ui_doctor_from_cli(args)
        return
    
    # Handle live smoke mode
    if args.live_smoke:
        from arena_bot.cli import run_live_smoke_from_cli
        run_live_smoke_from_cli(args)
        return
    
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
            
            # If GUI flags are present, launch GUI instead of staying in CLI mode
            if args.ui_safe_demo or args.ui_doctor:
                print("\n🖥️ Launching GUI interface...")
                from integrated_arena_bot_gui import IntegratedArenaBotGUI
                bot = IntegratedArenaBotGUI(ui_safe_demo=args.ui_safe_demo)
                bot.run()
                
        else:
            print("❌ Failed to initialize card recognition system")
            logger.error("Card recognition system initialization failed")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to initialize Arena Bot: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    finally:
        # Clean up debug run if active
        if debug_run_context:
            try:
                debug_run_context.__exit__(None, None, None)
                logger.info("Debug run completed")
            except Exception as e:
                logger.warning(f"Failed to clean up debug run: {e}")


if __name__ == "__main__":
    main()