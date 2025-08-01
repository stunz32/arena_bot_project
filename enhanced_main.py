#!/usr/bin/env python3
"""
Enhanced Arena Bot with S-Tier Logging Integration

This demonstrates how to integrate the S-tier logging system with your
existing Arena Bot for maximum performance and observability.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import S-tier logging
from arena_bot.logging_system import setup_s_tier_logging, get_s_tier_logger, get_system_health

# Import your existing Arena Bot components
from arena_bot.utils.config import load_config
from arena_bot.core.card_recognizer import get_card_recognizer


async def setup_enhanced_logging():
    """Setup S-tier logging with optimal configuration for Arena Bot."""
    
    print("🔧 Setting up S-tier logging system...")
    
    try:
        # Setup with Arena Bot optimized settings
        manager = await setup_s_tier_logging(
            environment="development",  # Change to "production" for live use
            system_name="arena-bot",
            
            # Performance settings
            enable_performance_monitoring=True,
            async_enabled=True,
            async_queue_size=5000,  # High capacity for game events
            worker_threads=3,       # Optimal for real-time processing
            
            # Arena Bot specific settings
            enable_structured_logging=True,  # For game data analysis
            enable_context_enrichment=True,  # Track user sessions
            
            # Optional: Integration with existing monitoring
            enable_metrics_integration=False,  # Set to True if you want metrics
        )
        
        print("✅ S-tier logging system initialized")
        return manager
        
    except Exception as e:
        print(f"❌ Failed to setup S-tier logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        return None


async def main():
    """Enhanced main function with S-tier logging."""
    
    print("🎮 Enhanced Arena Draft Bot with S-Tier Logging")
    print("=" * 55)
    
    # Setup S-tier logging first
    logging_manager = await setup_enhanced_logging()
    
    # Get high-performance loggers for different components
    main_logger = await get_s_tier_logger("arena_bot.main")
    card_logger = await get_s_tier_logger("arena_bot.cards")
    perf_logger = await get_s_tier_logger("arena_bot.performance")
    
    try:
        # Log startup with rich context
        await main_logger.info("🚀 Arena Bot starting up", extra={
            'version': '2.0.0',
            'logging_system': 's_tier',
            'performance_mode': 'enhanced'
        })
        
        # Load configuration (with logging)
        await main_logger.info("📁 Loading configuration...")
        config = load_config()
        await main_logger.info("✅ Configuration loaded", extra={
            'config_items': len(config) if hasattr(config, '__len__') else 'unknown'
        })
        
        # Initialize card recognition system (with performance logging)
        await card_logger.info("🔄 Initializing card recognition system...")
        start_time = asyncio.get_event_loop().time()
        
        card_recognizer = get_card_recognizer()
        
        if card_recognizer.initialize():
            init_time = asyncio.get_event_loop().time() - start_time
            
            # Log detailed initialization results
            stats = card_recognizer.get_detection_stats()
            await card_logger.info("✅ Card recognition system initialized", extra={
                'performance': {
                    'initialization_time_ms': init_time * 1000,
                    'histogram_database_size': stats['histogram_database_size'],
                    'template_counts': stats['template_counts'],
                    'screen_count': stats['screen_count']
                },
                'system_stats': stats
            })
            
            # Log performance metrics
            await perf_logger.info("📊 System initialization metrics", extra={
                'performance': {
                    'init_duration_ms': init_time * 1000,
                    'database_size': stats['histogram_database_size'],
                    'templates_loaded': sum(stats['template_counts']),
                    'memory_usage_estimate_mb': stats['histogram_database_size'] * 0.001  # Rough estimate
                }
            })
            
            # Display enhanced system information
            print(f"✅ Enhanced Arena Bot ready!")
            print(f"📊 System Stats:")
            print(f"   - Logging: S-tier high-performance ({await get_logging_performance()})")
            print(f"   - Cards: {stats['histogram_database_size']} in database")
            print(f"   - Templates: {sum(stats['template_counts'])} loaded") 
            print(f"   - Screens: {stats['screen_count']} supported")
            print(f"   - Init time: {init_time*1000:.1f}ms")
            
            # Check system health
            health_status = await get_system_health()
            await main_logger.info("🏥 System health check", extra={
                'health_status': health_status
            })
            
            # Log that we're ready for game detection
            await main_logger.info("🎯 Arena Bot fully operational and ready for card detection", extra={
                'features': [
                    'S-tier high-performance logging',
                    'Advanced card recognition',
                    'Real-time performance monitoring',
                    'Comprehensive error tracking',
                    'Rich context logging'
                ]
            })
            
            # You can now run your normal Arena Bot loop here
            # The logging system will automatically handle all performance monitoring
            print("\n🎮 Ready to detect cards and provide draft advice!")
            print("📝 All game events are now logged with high-performance S-tier system")
            
        else:
            await card_logger.error("❌ Failed to initialize card recognition system")
            await main_logger.critical("🚨 Arena Bot startup failed - card recognition unavailable")
            sys.exit(1)
        
    except Exception as e:
        await main_logger.error("💥 Arena Bot crashed during startup", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        print(f"❌ Error: {e}")
        sys.exit(1)


async def get_logging_performance():
    """Get current logging system performance stats."""
    try:
        health = await get_system_health()
        if 'performance' in health:
            perf = health['performance']
            return f"{perf.get('message_rate', 0):.0f} msg/s"
        return "Active"
    except:
        return "Unknown"


def run_enhanced_arena_bot():
    """Run the enhanced Arena Bot with S-tier logging."""
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Arena Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Arena Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_enhanced_arena_bot()