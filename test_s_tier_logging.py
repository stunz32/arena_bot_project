#!/usr/bin/env python3
"""
Quick test script to validate S-tier logging system integration.

This script performs basic import and initialization tests to ensure
the S-tier logging system is properly integrated and functional.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_s_tier_logging():
    """Test S-tier logging system initialization."""
    print("🧪 Testing S-tier logging system...")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from arena_bot.logging_system import (
            setup_s_tier_logging,
            get_logger,
            LoggerManager,
            STierLogger,
            LogLevel,
            get_system_health
        )
        print("✅ All imports successful")
        
        # Test basic setup
        print("🔄 Testing system setup...")
        manager = setup_s_tier_logging(
            environment="development",
            enable_performance_monitoring=False,  # Disable for quick test
            enable_metrics_integration=False      # Disable existing monitoring integration
        )
        print("✅ System setup successful")
        
        # Test logger creation
        print("📝 Testing logger creation...")
        logger = get_logger("test.logger")
        print(f"✅ Logger created: {logger.name}")
        
        # Test basic logging
        print("🔍 Testing basic logging...")
        logger.info("Test log message", extra={"test": True})
        await logger.ainfo("Test async log message", extra={"async": True})
        print("✅ Basic logging successful")
        
        # Test health check
        print("🏥 Testing health check...")
        health = get_system_health()
        print(f"✅ Health check complete: {health.get('unified_status', {}).get('status', 'unknown')}")
        
        # Test performance stats
        print("📊 Testing performance stats...")
        stats = manager.get_performance_stats()
        print(f"✅ Performance stats: {stats.get('total_logs_processed', 0)} logs processed")
        
        print("\n🎉 All tests passed! S-tier logging system is functional.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            # Cleanup
            print("🧹 Cleaning up...")
            from arena_bot.logging_system import shutdown
            shutdown()
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function."""
    print("🚀 S-tier Logging System Integration Test")
    print("=" * 50)
    
    # Run async test
    success = asyncio.run(test_s_tier_logging())
    
    if success:
        print("\n✅ Integration test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Integration test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()