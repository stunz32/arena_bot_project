#!/usr/bin/env python3
"""
Core Components Test for S-Tier Logging System

This script tests the core components directly without the full configuration system
to isolate any issues and validate the fundamental functionality.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_core_queue_functionality():
    """Test the core queue functionality."""
    print("🔧 Test 1: Core Queue Components")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.core import HybridAsyncQueue, LogMessage
        
        # Create queue
        queue = HybridAsyncQueue(ring_buffer_capacity=1000)
        print("✅ HybridAsyncQueue created")
        
        # Test basic operations
        message = LogMessage(
            level=20,  # INFO
            message="Test core component message",
            logger_name="test_core"
        )
        
        success = queue.put(message)
        print(f"✅ Message queued: {success}")
        
        retrieved = queue.get()
        print(f"✅ Message retrieved: {retrieved.message if retrieved else 'None'}")
        
        # Performance test
        start_time = time.perf_counter()
        for i in range(100):
            test_msg = LogMessage(
                level=20,
                message=f"Performance test {i}",
                logger_name="perf_test"
            )
            queue.put(test_msg)
        
        elapsed = time.perf_counter() - start_time
        rate = 100 / elapsed
        print(f"✅ Performance: {rate:.0f} messages/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Core queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_logger_manager_direct():
    """Test LoggerManager directly with minimal configuration."""
    print("\n📝 Test 2: Logger Manager Direct")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.logger import LoggerManager 
        from arena_bot.logging_system.config.models import (
            LoggingSystemConfig, HandlerConfig, 
            SinkConfig, FilterConfig, LoggerConfig, PerformanceConfig
        )
        
        # Create minimal configuration directly
        config = LoggingSystemConfig(
            environment="development",
            handlers={
                "test_console": HandlerConfig(
                    name="test_console",
                    type="console",
                    level="INFO"
                )
            },
            sinks={
                "test_sink": SinkConfig(
                    name="test_sink",
                    type="console",
                    level="INFO"
                )
            },
            filters={},
            loggers={
                "test": LoggerConfig(
                    name="test",
                    level="INFO",
                    handlers=["test_console"],
                    sinks=["test_sink"],
                    filters=[]
                )
            },
            performance=PerformanceConfig(
                enable_async_processing=True,
                async_queue_size=1000,
                worker_threads=2
            )
        )
        
        print("✅ Minimal configuration created")
        
        # Create LoggerManager directly
        logger_manager = LoggerManager()
        await logger_manager.initialize(config)
        print("✅ LoggerManager initialized")
        
        # Get a logger
        logger = await logger_manager.get_logger("test")
        print(f"✅ Logger created: {logger.name}")
        
        # Test logging
        await logger.info("Direct logger manager test")
        print("✅ Async logging successful")
        
        # Test sync logging
        logger.info("Sync logging test")
        print("✅ Sync logging successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Logger manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_formatter_inheritance():
    """Test formatter inheritance chain."""
    print("\n🎨 Test 3: Formatter Inheritance")  
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.formatters import (
            StructuredFormatter, ConsoleFormatter
        )
        from arena_bot.logging_system.core import LogMessage
        
        # Test structured formatter
        structured_formatter = StructuredFormatter("test_structured")
        print("✅ StructuredFormatter created")
        
        # Test console formatter  
        console_formatter = ConsoleFormatter("test_console")
        print("✅ ConsoleFormatter created")
        
        # Test formatting
        test_message = LogMessage(
            level=20,
            message="Test formatting",
            logger_name="formatter_test"
        )
        
        # Test console formatting
        console_formatted = console_formatter.format(test_message)
        print(f"✅ Console message formatted: {len(console_formatted) > 0}")
        
        # Test structured formatting
        structured_formatted = structured_formatter.format(test_message)
        print(f"✅ Structured message formatted: {len(structured_formatted) > 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Formatter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_health_diagnostics():
    """Test health and diagnostics systems."""
    print("\n🏥 Test 4: Health & Diagnostics")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.diagnostics import HealthChecker, PerformanceProfiler
        from arena_bot.logging_system.logger import LoggerManager
        
        # Create a minimal logger manager for health checker
        logger_manager = LoggerManager()
        
        # Test health checker with logger manager
        health_checker = HealthChecker(logger_manager)
        await health_checker.initialize()
        print("✅ HealthChecker created and initialized")
        
        health_status = await health_checker.get_health_status()
        print(f"✅ Health status retrieved: {health_status.get('overall_status', 'unknown')}")
        
        # Test performance profiler with logger manager
        profiler = PerformanceProfiler(logger_manager)
        await profiler.initialize()
        print("✅ PerformanceProfiler created and initialized")
        
        stats = await profiler.get_performance_stats()
        print(f"✅ Performance stats retrieved: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Health diagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all core component tests."""
    print("🧪 S-Tier Logging System - Core Components Test")
    print("=" * 55)
    
    # Run all tests
    test_results = []
    
    test_results.append(await test_core_queue_functionality())
    test_results.append(await test_logger_manager_direct())
    test_results.append(await test_formatter_inheritance())
    test_results.append(await test_health_diagnostics())
    
    # Summary
    print("\n" + "=" * 55)
    print("📋 Test Results Summary:")
    
    test_names = [
        "Core Queue Components",
        "Logger Manager Direct", 
        "Formatter Inheritance",
        "Health & Diagnostics"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {name}: {status}")
    
    # Overall result
    passed = sum(test_results)
    total = len(test_results)
    success_rate = (passed / total) * 100
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({success_rate:.0f}%)")
    
    if passed == total:
        print("🎉 All core component tests PASSED!")
        print("✅ S-Tier logging system core functionality is working.")
        return True
    else:
        print("⚠️  Some core tests failed. System partially functional.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Tests crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)