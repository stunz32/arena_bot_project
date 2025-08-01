#!/usr/bin/env python3
"""
Quick Start Guide for S-Tier Logging System

This script demonstrates how to initialize and use the S-tier logging system
with various configuration options and logging patterns.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def basic_usage_example():
    """Example of basic logging system usage."""
    print("🚀 S-Tier Logging System - Quick Start Guide")
    print("=" * 50)
    
    # Method 1: Simple initialization with default config
    print("\n📝 Method 1: Basic Usage")
    try:
        from arena_bot.logging_system import initialize_logging, get_logger, shutdown_logging
        from arena_bot.logging_system.config import get_development_config
        
        # Initialize the logging system
        config = get_development_config()
        await initialize_logging(config)
        print("✅ Logging system initialized")
        
        # Get a logger and use it
        logger = get_logger("my_app")
        
        # Basic logging
        logger.info("Hello from S-Tier logging system!")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        # Structured logging with context
        logger.info("User action completed", extra={
            'user_id': 'user123',
            'action': 'login',
            'duration_ms': 150,
            'success': True
        })
        
        print("✅ Basic logging completed")
        
        # Shutdown gracefully
        await shutdown_logging()
        print("✅ Logging system shutdown")
        
    except Exception as e:
        print(f"❌ Basic usage failed: {e}")
        import traceback
        traceback.print_exc()


async def advanced_usage_example():
    """Example of advanced logging system usage with custom configuration."""
    print("\n📝 Method 2: Advanced Usage with Custom Config")
    
    try:
        from arena_bot.logging_system.config import LoggingSystemConfig
        from arena_bot.logging_system import initialize_logging, get_logger
        
        # Create custom configuration
        custom_config = {
            "version": "1.0",
            "system_name": "my-custom-app",
            "environment": "development",
            "performance": {
                "enable_async_processing": True,
                "buffer_size": 5000,
                "worker_threads": 2
            },
            "security": {
                "enable_pii_detection": True,
                "enable_audit_trail": False
            },
            "loggers": {
                "root": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "console",
                    "level": "INFO"
                }
            },
            "sinks": {
                "console": {
                    "type": "console",
                    "enabled": True,
                    "level": "INFO"
                }
            }
        }
        
        # Parse and initialize
        config = LoggingSystemConfig.parse_obj(custom_config)
        await initialize_logging(config)
        print("✅ Advanced logging system initialized")
        
        # Use multiple loggers
        app_logger = get_logger("app")
        db_logger = get_logger("database")
        api_logger = get_logger("api")
        
        # Log from different components
        app_logger.info("Application started")
        db_logger.debug("Database connection established")
        api_logger.warning("Rate limit approaching", extra={'requests_remaining': 10})
        
        print("✅ Advanced logging completed")
        
    except Exception as e:
        print(f"❌ Advanced usage failed: {e}")
        import traceback
        traceback.print_exc()


def simple_synchronous_example():
    """Example of simple synchronous logging (fallback mode)."""
    print("\n📝 Method 3: Simple Synchronous Logging")
    
    try:
        # For simple cases, you can use standard Python logging enhanced by S-tier
        import logging
        
        # Configure basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Get loggers
        logger = logging.getLogger("simple_app")
        
        # Basic logging
        logger.info("Simple logging example")
        logger.warning("This works without S-tier initialization")
        logger.error("But you miss the advanced features")
        
        print("✅ Simple synchronous logging works")
        
    except Exception as e:
        print(f"❌ Simple logging failed: {e}")


async def main():
    """Run all examples."""
    print("🔥 S-Tier Logging System Examples")
    print("This will demonstrate different ways to use the logging system.\n")
    
    # Run examples
    await basic_usage_example()
    await advanced_usage_example()
    simple_synchronous_example()
    
    print("\n" + "=" * 50)
    print("📚 Usage Summary:")
    print("1. For most apps: Use get_development_config() → initialize_logging() → get_logger()")
    print("2. For custom needs: Create LoggingSystemConfig → initialize_logging() → get_logger()")
    print("3. For simple cases: Standard Python logging works as fallback")
    print("\n🎉 S-Tier Logging System is ready for production use!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Examples failed: {e}")
        import traceback
        traceback.print_exc()