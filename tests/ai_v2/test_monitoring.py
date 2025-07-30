#!/usr/bin/env python3
"""
Test suite for AI Helper v2 monitoring system
Tests performance monitoring, resource management, and hardening features
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from arena_bot.ai_v2.monitoring import (
    # Core classes
    LockFreeRingBuffer, PerformanceMonitor, GlobalResourceManager,
    ResourceThresholds, MemoryTracker,
    # Enums
    MetricType, MonitoringState, ResourceState, ResourceType,
    # Global functions
    get_performance_monitor, get_resource_manager
)


class TestLockFreeRingBuffer:
    """Test lock-free ring buffer implementation"""
    
    def test_ring_buffer_creation(self):
        """Test ring buffer creation and initial state"""
        buffer = LockFreeRingBuffer(capacity=10)
        
        assert buffer.capacity == 10
        assert buffer._size == 0
        assert buffer._write_index == 0
        assert buffer._read_index == 0
        
    def test_ring_buffer_put_get(self):
        """Test basic put and get operations"""
        buffer = LockFreeRingBuffer(capacity=5)
        
        # Add items
        assert buffer.put("item1") is True
        assert buffer.put("item2") is True
        assert buffer.put("item3") is True
        
        # Get recent items
        recent = buffer.get_recent(3)
        assert len(recent) == 3
        assert recent[0] == "item3"  # Most recent first
        assert recent[1] == "item2"
        assert recent[2] == "item1"
        
    def test_ring_buffer_overflow(self):
        """Test ring buffer overflow behavior"""
        buffer = LockFreeRingBuffer(capacity=3)
        
        # Fill buffer
        buffer.put("item1")
        buffer.put("item2")
        buffer.put("item3")
        
        # Overflow - should overwrite oldest
        buffer.put("item4")
        buffer.put("item5")
        
        recent = buffer.get_recent()
        assert len(recent) == 3
        assert recent[0] == "item5"  # Most recent
        assert recent[1] == "item4"
        assert recent[2] == "item3"
        # item1 and item2 should be overwritten
        
    def test_ring_buffer_clear(self):
        """Test ring buffer clear operation"""
        buffer = LockFreeRingBuffer(capacity=5)
        
        buffer.put("item1")
        buffer.put("item2")
        assert len(buffer.get_recent()) == 2
        
        buffer.clear()
        assert len(buffer.get_recent()) == 0
        assert buffer._size == 0
        
    def test_ring_buffer_thread_safety(self):
        """Test ring buffer thread safety"""
        buffer = LockFreeRingBuffer(capacity=100)
        results = []
        errors = []
        
        def writer_thread(thread_id):
            try:
                for i in range(20):
                    buffer.put(f"thread_{thread_id}_item_{i}")
                    time.sleep(0.001)
                results.append(f"writer_{thread_id}_done")
            except Exception as e:
                errors.append(f"Writer {thread_id}: {e}")
                
        def reader_thread(thread_id):
            try:
                for i in range(10):
                    items = buffer.get_recent(5)
                    time.sleep(0.002)
                results.append(f"reader_{thread_id}_done")
            except Exception as e:
                errors.append(f"Reader {thread_id}: {e}")
                
        # Start multiple threads
        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
            threads.append(threading.Thread(target=reader_thread, args=(i,)))
            
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All operations should complete without errors
        assert len(errors) == 0
        assert len(results) == 6  # 3 writers + 3 readers


class TestPerformanceMonitor:
    """Test performance monitoring with hardening features"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create fresh monitor instance
        self.monitor = PerformanceMonitor()
        
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.state == MonitoringState.INACTIVE
        assert not self.monitor._circuit_breaker_open
        assert not self.monitor._suicide_triggered
        assert self.monitor._suicide_threshold_violations == 0
        
    def test_lazy_activation(self):
        """Test P0.5.1: Lazy activation when performance degradation detected"""
        # Mock high CPU usage to trigger activation
        with patch('arena_bot.ai_v2.monitoring.psutil') as mock_psutil:
            mock_process = MagicMock()
            mock_process.cpu_percent.return_value = 75.0  # Above threshold
            mock_psutil.Process.return_value = mock_process
            mock_psutil.cpu_percent.return_value = 75.0
            
            # Simulate sustained high CPU
            for _ in range(4):  # More than ACTIVATION_DURATION
                should_activate = self.monitor._check_activation_conditions()
                time.sleep(0.1)
                
            # Should activate monitoring
            assert self.monitor.state == MonitoringState.ACTIVE
            
    def test_metric_recording(self):
        """Test metric recording functionality"""
        # Activate monitoring first
        self.monitor.state = MonitoringState.ACTIVE
        
        # Record various metric types
        self.monitor.record_metric("test_counter", 5, MetricType.COUNTER)
        self.monitor.record_metric("test_gauge", 42.5, MetricType.GAUGE)
        self.monitor.record_metric("test_timer", 0.125, MetricType.TIMER)
        
        # Verify metrics were recorded
        counter_summary = self.monitor.get_metrics_summary("test_counter")
        assert counter_summary["latest"] == 5
        assert counter_summary["type"] == "counter"
        
        gauge_summary = self.monitor.get_metrics_summary("test_gauge")
        assert gauge_summary["latest"] == 42.5
        
    def test_circuit_breaker_activation(self):
        """Test P0.5.3: Circuit breaker activation"""
        # Mock high resource usage to trigger circuit breaker
        with patch('arena_bot.ai_v2.monitoring.psutil') as mock_psutil:
            mock_process = MagicMock()
            # Memory exceeding MAX_MEMORY_MB (5MB)
            mock_process.memory_info.return_value.rss = 10 * 1024 * 1024  # 10MB
            mock_process.cpu_percent.return_value = 5.0  # Above MAX_CPU_PERCENT (2%)
            mock_psutil.Process.return_value = mock_process
            
            # Force multiple failures to trigger circuit breaker
            self.monitor._circuit_breaker_failures = 2
            self.monitor._check_self_resource_usage()
            
            # Circuit breaker should be active
            assert self.monitor._circuit_breaker_open
            assert self.monitor.state == MonitoringState.DISABLED
            
    def test_suicide_protocol(self):
        """Test P0.6.7: Resource monitor suicide protocol"""
        # Mock sustained high resource usage
        with patch('arena_bot.ai_v2.monitoring.psutil') as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 10 * 1024 * 1024  # 10MB
            mock_process.cpu_percent.return_value = 5.0  # Above limit
            mock_psutil.Process.return_value = mock_process
            
            # Simulate multiple violations over time
            for _ in range(4):
                # Reset time check to force checking
                self.monitor._last_suicide_check = time.time() - 6
                self.monitor._check_self_resource_usage()
                time.sleep(0.1)
                
            # Suicide protocol should be triggered
            assert self.monitor._suicide_triggered
            assert self.monitor.state == MonitoringState.DEGRADED
            
    def test_timer_context_manager(self):
        """Test timer context manager"""
        self.monitor.state = MonitoringState.ACTIVE
        
        with self.monitor.start_timer("test_operation") as timer:
            time.sleep(0.01)  # Brief operation
            
        # Timer should have recorded the operation
        summary = self.monitor.get_metrics_summary("test_operation_duration_seconds")
        assert summary["latest"] > 0.005  # Should be > 5ms
        assert summary["type"] == "timer"
        
    def test_performance_report_generation(self):
        """Test comprehensive performance report generation"""
        self.monitor.state = MonitoringState.ACTIVE
        
        # Record some metrics
        self.monitor.record_metric("test_metric", 100, MetricType.GAUGE)
        
        report = self.monitor.get_performance_report()
        
        assert "timestamp" in report
        assert report["monitoring_state"] == "active"
        assert "summary" in report
        assert "system_health" in report
        
    def test_monitor_reset_circuit_breaker(self):
        """Test circuit breaker reset functionality"""
        # Trigger circuit breaker
        self.monitor._circuit_breaker_open = True
        self.monitor.state = MonitoringState.DISABLED
        
        # Reset should restore functionality
        self.monitor.reset_circuit_breaker()
        
        assert not self.monitor._circuit_breaker_open
        assert self.monitor.state == MonitoringState.INACTIVE
        assert self.monitor._circuit_breaker_failures == 0


class TestGlobalResourceManager:
    """Test global resource management system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_manager = GlobalResourceManager()
        
    def test_resource_manager_initialization(self):
        """Test resource manager initialization"""
        assert len(self.resource_manager._resource_states) == len(ResourceType)
        
        # All resources should start in NORMAL state
        for resource_type, state in self.resource_manager._resource_states.items():
            assert state == ResourceState.NORMAL
            
        # Should have buffers for all resource types
        assert len(self.resource_manager._resource_buffers) == len(ResourceType)
        
        # Should have recovery protocols
        assert ResourceType.MEMORY in self.resource_manager._recovery_protocols
        assert ResourceType.CPU in self.resource_manager._recovery_protocols
        
    def test_resource_thresholds(self):
        """Test resource threshold definitions"""
        thresholds = ResourceThresholds()
        
        # Memory thresholds should be progressive
        assert thresholds.MEMORY_WARNING < thresholds.MEMORY_CRITICAL
        assert thresholds.MEMORY_CRITICAL < thresholds.MEMORY_EMERGENCY
        assert thresholds.MEMORY_EMERGENCY < thresholds.MEMORY_MAX
        
        # CPU thresholds should be progressive
        assert thresholds.CPU_WARNING < thresholds.CPU_CRITICAL
        assert thresholds.CPU_CRITICAL < thresholds.CPU_EMERGENCY
        assert thresholds.CPU_EMERGENCY < thresholds.CPU_MAX
        
    @patch('arena_bot.ai_v2.monitoring.psutil')
    def test_resource_collection(self, mock_psutil):
        """Test P0.6.1: Centralized resource monitoring"""
        # Mock system resources
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process.cpu_percent.return_value = 15.0
        mock_process.num_threads.return_value = 8
        mock_psutil.Process.return_value = mock_process
        
        # Collect resources
        self.resource_manager._collect_all_resources()
        
        # Check that data was collected
        memory_data = self.resource_manager._resource_buffers[ResourceType.MEMORY].get_recent(1)
        assert len(memory_data) > 0
        assert memory_data[0]['value'] == 100.0  # 100MB
        
    def test_resource_state_transitions(self):
        """Test resource state management"""
        # Simulate high memory usage
        self.resource_manager._resource_buffers[ResourceType.MEMORY].put({
            'timestamp': time.time(),
            'value': 450.0,  # Above MEMORY_CRITICAL (450MB)
            'unit': 'MB'
        })
        
        # Check thresholds
        self.resource_manager._check_resource_thresholds()
        
        # Memory should be in CRITICAL state
        assert self.resource_manager._resource_states[ResourceType.MEMORY] == ResourceState.CRITICAL
        
    def test_emergency_recovery_protocols(self):
        """Test P0.6.3: Emergency resource recovery protocol"""
        # Set memory to EMERGENCY state
        self.resource_manager._resource_states[ResourceType.MEMORY] = ResourceState.EMERGENCY
        
        # Track recovery executions
        recovery_executed = []
        
        def mock_recovery():
            recovery_executed.append("memory_recovery")
            
        # Replace recovery protocol with mock
        self.resource_manager._recovery_protocols[ResourceType.MEMORY] = [mock_recovery]
        
        # Trigger emergency check
        self.resource_manager._check_emergency_conditions()
        
        # Recovery should have been executed
        assert len(recovery_executed) > 0
        assert self.resource_manager._emergency_activations > 0
        
    def test_component_registration(self):
        """Test component registration for cleanup"""
        test_component = {"name": "test_component"}
        cleanup_called = []
        
        def cleanup_handler():
            cleanup_called.append("cleanup")
            
        # Register component
        self.resource_manager.register_component(test_component, cleanup_handler)
        
        # Verify registration
        assert len(self.resource_manager._cleanup_handlers) > 0
        
        # Test cleanup execution
        self.resource_manager._clear_caches()
        assert len(cleanup_called) > 0
        
    def test_resource_dashboard(self):
        """Test P0.6.2: Resource usage dashboard"""
        dashboard = self.resource_manager.get_resource_dashboard()
        
        assert "timestamp" in dashboard
        assert "session_uptime_hours" in dashboard
        assert "resources" in dashboard
        assert "states" in dashboard
        assert "emergency_activations" in dashboard
        
        # States should contain all resource types
        for resource_type in ResourceType:
            assert resource_type.value in dashboard["states"]
            
    def test_health_status_assessment(self):
        """Test comprehensive health status"""
        health = self.resource_manager.get_health_status()
        
        assert "overall_health" in health
        assert "critical_resources" in health
        assert "emergency_resources" in health
        assert "session_uptime_hours" in health
        assert "recommend_restart" in health
        
        # Should start as healthy
        assert health["overall_health"] == "healthy"
        assert health["critical_resources"] == 0
        assert health["emergency_resources"] == 0
        
    def test_session_health_monitoring(self):
        """Test P0.6.4: Session health monitoring"""
        # Simulate degradation events
        self.resource_manager._degradation_events.append({
            'timestamp': time.time(),
            'resource': ResourceType.MEMORY.value,
            'severity': 'high'
        })
        
        # Update session health
        self.resource_manager._update_session_health()
        
        # Should detect degradation pattern
        health = self.resource_manager.get_health_status()
        assert "recent_degradations" in health
        
    def test_monitoring_thread_lifecycle(self):
        """Test resource monitoring thread management"""
        assert not self.resource_manager._monitoring_active
        
        # Start monitoring
        self.resource_manager.start_monitoring()
        assert self.resource_manager._monitoring_active
        assert self.resource_manager._monitoring_thread is not None
        
        # Brief wait for thread to start
        time.sleep(0.1)
        
        # Stop monitoring
        self.resource_manager.stop_monitoring()
        assert not self.resource_manager._monitoring_active


class TestMemoryTracker:
    """Test memory usage tracking"""
    
    def test_memory_tracker_initialization(self):
        """Test memory tracker initialization"""
        tracker = MemoryTracker(alert_threshold_mb=50.0)
        
        assert tracker.alert_threshold_mb == 50.0
        assert tracker.peak_usage_mb == 0
        assert tracker.monitor is not None
        
    def test_memory_tracking_context(self):
        """Test memory tracking context manager"""
        tracker = MemoryTracker(alert_threshold_mb=1.0)  # Low threshold for testing
        
        with patch('arena_bot.ai_v2.monitoring.psutil') as mock_psutil:
            mock_process = MagicMock()
            # Simulate memory increase during operation
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB initially
            mock_psutil.Process.return_value = mock_process
            
            with tracker.track_memory("test_operation"):
                # Simulate memory increase
                mock_process.memory_info.return_value.rss = 105 * 1024 * 1024  # 105MB after
                
            # Peak usage should be updated
            assert tracker.peak_usage_mb >= 100.0


class TestGlobalInstances:
    """Test global instance management"""
    
    def test_performance_monitor_singleton(self):
        """Test global performance monitor singleton"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2  # Same instance
        assert isinstance(monitor1, PerformanceMonitor)
        
    def test_resource_manager_singleton(self):
        """Test global resource manager singleton"""
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        assert manager1 is manager2  # Same instance
        assert isinstance(manager1, GlobalResourceManager)


class TestMonitoringIntegration:
    """Test integration between monitoring components"""
    
    def test_performance_monitor_resource_manager_integration(self):
        """Test integration between monitor and resource manager"""
        monitor = get_performance_monitor()
        resource_manager = get_resource_manager()
        
        # Both should be available
        assert monitor is not None
        assert resource_manager is not None
        
        # Resource manager should be able to track monitoring overhead
        monitor.state = MonitoringState.ACTIVE
        monitor.record_metric("test_integration", 42, MetricType.GAUGE)
        
        dashboard = resource_manager.get_resource_dashboard()
        assert dashboard is not None
        
    def test_monitoring_system_stress_test(self):
        """Stress test the monitoring system"""
        monitor = get_performance_monitor()
        monitor.state = MonitoringState.ACTIVE
        
        errors = []
        
        def stress_worker(worker_id):
            try:
                for i in range(100):
                    # Mix of operations
                    monitor.record_metric(f"stress_{worker_id}_{i}", i, MetricType.COUNTER)
                    
                    if i % 10 == 0:
                        monitor.get_metrics_summary()
                        
                    if i % 20 == 0:
                        with monitor.start_timer(f"timer_{worker_id}_{i}"):
                            time.sleep(0.001)
                            
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
                
        # Start multiple stress workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=stress_worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Should complete without errors
        assert len(errors) == 0
        
        # System should still be responsive
        summary = monitor.get_metrics_summary()
        assert summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])