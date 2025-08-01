#!/usr/bin/env python3
"""
Test Script for S-Tier Logging System Async Queue

This script tests the core async queue functionality to ensure it works properly.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_basic_queue_operations():
    """Test basic queue put/get operations."""
    print("🧪 Test 1: Basic Queue Operations")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.core import HybridAsyncQueue, LogMessage
        
        # Create queue
        queue = HybridAsyncQueue(ring_buffer_capacity=1000)
        print("✅ Queue created successfully")
        
        # Create test messages
        messages = []
        for i in range(5):
            message = LogMessage(
                level=20,  # INFO
                message=f"Test message {i}",
                logger_name="test_logger"
            )
            messages.append(message)
        
        print(f"✅ Created {len(messages)} test messages")
        
        # Test putting messages
        for i, message in enumerate(messages):
            success = queue.put(message)
            if success:
                print(f"   📝 Put message {i}: {message.message}")
            else:
                print(f"   ❌ Failed to put message {i}")
        
        print(f"✅ All messages queued. Queue size: {queue.size()}")
        
        # Test getting messages
        retrieved_messages = []
        for i in range(len(messages)):
            message = queue.get()
            if message:
                retrieved_messages.append(message)
                print(f"   📄 Got message {i}: {message.message}")
            else:
                print(f"   ❌ Failed to get message {i}")
        
        print(f"✅ All messages retrieved. Queue size: {queue.size()}")
        
        # Verify message integrity
        for orig, retr in zip(messages, retrieved_messages):
            assert orig.message == retr.message
            assert orig.level == retr.level
            assert orig.logger_name == retr.logger_name
        
        print("✅ Message integrity verified")
        return True
        
    except Exception as e:
        print(f"❌ Basic queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_queue_performance():
    """Test queue performance with many messages."""
    print("\n🚀 Test 2: Queue Performance")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.core import HybridAsyncQueue, LogMessage
        
        # Create queue
        queue = HybridAsyncQueue(ring_buffer_capacity=10000)
        
        # Performance test parameters
        num_messages = 1000
        
        # Create messages
        messages = [
            LogMessage(
                level=20,
                message=f"Performance test message {i}",
                logger_name="perf_logger"
            )
            for i in range(num_messages)
        ]
        
        # Test put performance
        start_time = time.perf_counter()
        for message in messages:
            queue.put(message)
        put_time = time.perf_counter() - start_time
        
        print(f"✅ Put {num_messages} messages in {put_time:.3f}s")
        print(f"   Rate: {num_messages/put_time:.0f} messages/second")
        
        # Test get performance
        start_time = time.perf_counter()
        retrieved = []
        for _ in range(num_messages):
            message = queue.get()
            if message:
                retrieved.append(message)
        get_time = time.perf_counter() - start_time
        
        print(f"✅ Got {num_messages} messages in {get_time:.3f}s")
        print(f"   Rate: {num_messages/get_time:.0f} messages/second")
        
        # Overall performance
        total_time = put_time + get_time
        overall_rate = (num_messages * 2) / total_time  # put + get operations
        
        print(f"✅ Overall rate: {overall_rate:.0f} operations/second")
        
        # Verify no messages lost
        assert len(retrieved) == num_messages
        print("✅ No messages lost")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_queue_stats():
    """Test queue statistics and monitoring."""
    print("\n📊 Test 3: Queue Statistics")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.core import HybridAsyncQueue, LogMessage
        
        # Create queue
        queue = HybridAsyncQueue(ring_buffer_capacity=100)
        
        # Add some messages
        for i in range(10):
            message = LogMessage(
                level=30,  # WARNING
                message=f"Stats test message {i}",
                logger_name="stats_logger"
            )
            queue.put(message)
        
        # Check basic stats
        print(f"✅ Queue size: {queue.size()}")
        print(f"✅ Queue capacity: {queue.ring_buffer_capacity}")
        print(f"✅ Queue empty: {queue.size() == 0}")
        print(f"✅ Queue utilization: {(queue.size() / queue.ring_buffer_capacity) * 100:.1f}%")
        
        # Test queue management
        if hasattr(queue, 'get_stats'):
            stats = queue.get_stats()
            print(f"✅ Queue stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Queue stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_message_serialization():
    """Test LogMessage serialization capabilities."""
    print("\n🔄 Test 4: Message Serialization")
    print("-" * 40)
    
    try:
        from arena_bot.logging_system.core import LogMessage
        
        # Create a complex message
        message = LogMessage(
            level=40,  # ERROR
            message="Test error message",
            logger_name="serialization_test",
            context={'user_id': 'user123', 'action': 'login'},
            performance={'duration_ms': 250, 'memory_mb': 45},
            system={'hostname': 'test-server', 'pid': 12345}
        )
        
        print("✅ Complex message created")
        
        # Test dict conversion
        message_dict = message.to_dict()
        print(f"✅ Dict conversion: {len(message_dict)} fields")
        
        # Test JSON conversion
        message_json = message.to_json()
        print(f"✅ JSON conversion: {len(message_json)} characters")
        
        # Verify key fields
        assert message_dict['level'] == 40
        assert message_dict['message'] == "Test error message"
        assert message_dict['context']['user_id'] == 'user123'
        
        print("✅ Serialization data integrity verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Message serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all async queue tests."""
    print("🧪 S-Tier Logging System - Async Queue Tests")
    print("=" * 50)
    
    # Run all tests
    test_results = []
    
    test_results.append(await test_basic_queue_operations())
    test_results.append(await test_queue_performance())
    test_results.append(await test_queue_stats())
    test_results.append(await test_message_serialization())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    test_names = [
        "Basic Queue Operations",
        "Queue Performance", 
        "Queue Statistics",
        "Message Serialization"
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
        print("🎉 All async queue tests PASSED! Core functionality is working.")
        return True
    else:
        print("⚠️  Some tests failed. Review errors above.")
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