#!/usr/bin/env python3
"""
Basic Integration Test for S-Tier Logging System.

This script tests core functionality of the logging system including:
- System initialization and configuration
- Basic logging operations (sync and async)
- Health checks and performance monitoring
- Manager orchestration and coordination
"""

import asyncio
import time
import sys
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that all core components can be imported."""
    print("🔍 Testing basic imports...")
    
    try:
        from arena_bot.logging_system import get_logger, initialize_logging, shutdown_logging
        from arena_bot.logging_system.logger import LoggerManager, STierLogger
        from arena_bot.logging_system.config import create_development_config
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_creation():
    """Test configuration creation and validation."""
    print("\n🔧 Testing configuration creation...")
    
    try:
        from arena_bot.logging_system.config import create_development_config, LoggingSystemConfig
        
        # Test creating config from function
        config = create_development_config()
        print(f"✅ Development config created: {config.system_name}")
        
        # Test basic validation
        if config.performance.enable_async_processing:
            print("✅ Configuration validation passed")
            return True, config
        else:
            print("❌ Configuration validation failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False, None


async def test_logger_initialization():
    """Test logger manager initialization."""
    print("\n🚀 Testing logger initialization...")
    
    try:
        from arena_bot.logging_system.logger import LoggerManager
        from arena_bot.logging_system.config import create_development_config
        
        # Create configuration
        config = create_development_config()
        
        # Initialize logger manager
        logger_manager = LoggerManager()
        await logger_manager.initialize(config)
        
        print("✅ Logger manager initialized successfully")
        
        # Test health check
        if logger_manager.is_healthy():
            print("✅ Logger manager health check passed")
            return True, logger_manager
        else:
            print("❌ Logger manager health check failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Logger initialization test failed: {e}")
        traceback.print_exc()
        return False, None


async def test_basic_logging():
    """Test basic logging operations."""
    print("\n📝 Testing basic logging operations...")
    
    try:
        from arena_bot.logging_system import get_logger, initialize_logging
        from arena_bot.logging_system.config import create_development_config
        
        # Initialize logging system
        config = create_development_config()
        await initialize_logging(config)
        
        # Get logger instance
        logger = get_logger("integration_test")
        
        # Test different log levels
        logger.debug("Debug message - testing debug level")
        logger.info("Info message - testing info level") 
        logger.warning("Warning message - testing warning level")
        logger.error("Error message - testing error level")
        
        # Test structured logging with context
        logger.info("Structured message", extra={
            'user_id': 'test_user_123',
            'operation': 'integration_test',
            'duration_ms': 42.5
        })
        
        print("✅ Basic logging operations completed")
        
        # Give async processing time to complete
        await asyncio.sleep(0.1)
        
        return True
        
    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        traceback.print_exc()
        return False


async def test_performance_monitoring():
    """Test performance monitoring and health checks."""
    print("\n📊 Testing performance monitoring...")
    
    try:
        from arena_bot.logging_system.logger import LoggerManager
        from arena_bot.logging_system.config import create_development_config
        
        # Get initialized logger manager
        config = create_development_config()
        logger_manager = LoggerManager()
        await logger_manager.initialize(config)
        
        # Test performance stats
        stats = logger_manager.get_performance_stats()
        if isinstance(stats, dict):
            print(f"✅ Performance stats retrieved: {len(stats)} metrics")
            
            # Print key metrics
            if 'queue_stats' in stats:
                queue_stats = stats['queue_stats']
                print(f"   Queue size: {queue_stats.get('current_size', 0)}")
                print(f"   Messages processed: {queue_stats.get('total_processed', 0)}")
            
            return True
        else:
            print("❌ Failed to retrieve performance stats")
            return False
            
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False


async def test_manager_coordination():
    """Test coordination between different managers."""
    print("\n🎭 Testing manager coordination...")
    
    try:
        from arena_bot.logging_system.formatters.formatter_manager import FormatterManager
        from arena_bot.logging_system.sinks.sink_manager import SinkManager  
        from arena_bot.logging_system.filters.filter_manager import FilterManager
        from arena_bot.logging_system.config import create_development_config
        
        config = create_development_config()
        
        # Initialize all managers
        formatter_manager = FormatterManager()
        await formatter_manager.initialize(config)
        
        sink_manager = SinkManager()
        await sink_manager.initialize(config)
        
        filter_manager = FilterManager()
        await filter_manager.initialize(config)
        
        # Test health checks
        managers_healthy = [
            formatter_manager.is_healthy(),
            sink_manager.is_healthy(),
            filter_manager.is_healthy()
        ]
        
        if all(managers_healthy):
            print("✅ All managers initialized and healthy")
            
            # Test performance stats from each manager
            formatter_stats = formatter_manager.get_performance_stats()
            sink_stats = sink_manager.get_performance_stats()
            filter_stats = filter_manager.get_performance_stats()
            
            print(f"   Formatter Manager: {formatter_stats['active_formatters']} formatters")
            print(f"   Sink Manager: {sink_stats['active_sinks']} sinks")  
            print(f"   Filter Manager: {filter_stats['active_filters']} filters")
            
            return True
        else:
            print(f"❌ Manager health checks failed: {managers_healthy}")
            return False
            
    except Exception as e:
        print(f"❌ Manager coordination test failed: {e}")
        traceback.print_exc()
        return False


async def test_system_shutdown():
    """Test graceful system shutdown."""
    print("\n🛑 Testing system shutdown...")
    
    try:
        from arena_bot.logging_system import shutdown_logging
        
        # Initiate shutdown
        await shutdown_logging()
        
        print("✅ System shutdown completed gracefully")
        return True
        
    except Exception as e:
        print(f"❌ System shutdown test failed: {e}")
        traceback.print_exc()
        return False


async def run_integration_tests():
    """Run all integration tests."""
    print("🔥 S-Tier Logging System - Integration Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Basic imports
    if test_basic_imports():
        tests_passed += 1
    
    # Test 2: Configuration creation
    config_success, config = test_configuration_creation()
    if config_success:
        tests_passed += 1
    
    # Test 3: Logger initialization
    logger_success, logger_manager = await test_logger_initialization()
    if logger_success:
        tests_passed += 1
    
    # Test 4: Basic logging
    if await test_basic_logging():
        tests_passed += 1
    
    # Test 5: Performance monitoring
    if await test_performance_monitoring():
        tests_passed += 1
    
    # Test 6: Manager coordination
    if await test_manager_coordination():
        tests_passed += 1
    
    # Test 7: System shutdown
    if await test_system_shutdown():
        tests_passed += 1
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Results")
    print(f"   Tests Passed: {tests_passed}/{total_tests}")
    print(f"   Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    print(f"   Duration: {duration:.2f} seconds")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! S-Tier Logging System is operational.")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    print("Starting S-Tier Logging System Integration Test...")
    
    try:
        # Run the integration tests
        success = asyncio.run(run_integration_tests())
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Integration test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Integration test crashed: {e}")
        traceback.print_exc()
        sys.exit(1)