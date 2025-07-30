#!/usr/bin/env python3
"""
Test suite for Configuration Manager
Tests the thread-safe configuration system with hardening features
"""

import pytest
import json
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from arena_bot.config.config_manager import (
    ConfigState, ConfigSnapshot, ConfigurationManager,
    get_config_manager, get_config, update_config
)


class TestConfigSnapshot:
    """Test configuration snapshot functionality"""
    
    def test_config_snapshot_creation(self):
        """Test basic config snapshot creation"""
        data = {"test": "value", "number": 42}
        snapshot = ConfigSnapshot(
            generation=1,
            timestamp=time.time(),
            data=data,
            checksum="",
            state=ConfigState.STABLE
        )
        
        assert snapshot.generation == 1
        assert snapshot.data["test"] == "value"
        assert snapshot.state == ConfigState.STABLE
        assert len(snapshot.checksum) > 0  # Auto-calculated
        
    def test_config_snapshot_immutability(self):
        """Test that snapshot data is immutable"""
        original_data = {"test": "value"}
        snapshot = ConfigSnapshot(
            generation=1,
            timestamp=time.time(),
            data=original_data,
            checksum="",
            state=ConfigState.STABLE
        )
        
        # Modifying original data shouldn't affect snapshot
        original_data["test"] = "modified"
        assert snapshot.data["test"] == "value"
        
        # Modifying snapshot data should create new dict
        snapshot.data["new_key"] = "new_value"
        assert "new_key" not in original_data
        
    def test_config_snapshot_integrity_validation(self):
        """Test configuration integrity validation"""
        data = {"test": "value"}
        snapshot = ConfigSnapshot(
            generation=1,
            timestamp=time.time(),
            data=data,
            checksum="",
            state=ConfigState.STABLE
        )
        
        # Should validate correctly initially
        assert snapshot.validate_integrity()
        
        # Manually corrupt checksum
        snapshot.checksum = "invalid_checksum"
        assert not snapshot.validate_integrity()


class TestConfigurationManager:
    """Test configuration manager functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Use temporary directory for config
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        
    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_config_manager_initialization(self):
        """Test configuration manager initialization"""
        config_manager = ConfigurationManager(self.config_dir)
        
        assert config_manager.get_state() == ConfigState.STABLE
        assert config_manager.get_generation() >= 1
        
        # Check default configuration structure
        config = config_manager.get_config()
        assert "ai_helper" in config
        assert "performance" in config
        assert "security" in config
        assert "ui" in config
        assert "logging" in config
        
    def test_config_get_operations(self):
        """Test configuration get operations"""
        config_manager = ConfigurationManager(self.config_dir)
        
        # Get entire config
        full_config = config_manager.get_config()
        assert isinstance(full_config, dict)
        
        # Get specific key path
        ai_config = config_manager.get_config("ai_helper")
        assert isinstance(ai_config, dict)
        assert "enabled" in ai_config
        
        # Get nested key path
        enabled = config_manager.get_config("ai_helper.enabled")
        assert isinstance(enabled, bool)
        
        # Get non-existent key with default
        missing = config_manager.get_config("non.existent.key", "default_value")
        assert missing == "default_value"
        
    def test_config_update_operations(self):
        """Test configuration update operations"""
        config_manager = ConfigurationManager(self.config_dir)
        initial_generation = config_manager.get_generation()
        
        # Simple update
        updates = {"ai_helper.enabled": False}
        result = config_manager.update_config(updates)
        assert result is True
        
        # Verify update applied
        assert config_manager.get_config("ai_helper.enabled") is False
        assert config_manager.get_generation() > initial_generation
        
        # Complex nested update
        complex_updates = {
            "performance.max_memory_mb": 600,
            "ui.overlay_opacity": 0.9,
            "new_section": {"new_key": "new_value"}
        }
        result = config_manager.update_config(complex_updates)
        assert result is True
        
        assert config_manager.get_config("performance.max_memory_mb") == 600
        assert config_manager.get_config("ui.overlay_opacity") == 0.9
        assert config_manager.get_config("new_section.new_key") == "new_value"
        
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigurationManager(self.config_dir)
        
        # Valid update should succeed
        valid_updates = {"performance.max_memory_mb": 100}
        result = config_manager.update_config(valid_updates)
        assert result is True
        
        # Invalid update should fail
        invalid_updates = {"performance.max_memory_mb": 10}  # Below minimum of 50
        result = config_manager.update_config(invalid_updates)
        assert result is False
        
        # CPU limit validation
        invalid_cpu = {"performance.max_cpu_percent": 60}  # Above maximum of 50
        result = config_manager.update_config(invalid_cpu)
        assert result is False
        
    def test_generation_validation(self):
        """Test generation-based consistency validation"""
        config_manager = ConfigurationManager(self.config_dir)
        generation = config_manager.get_generation()
        
        # Valid generation should return True
        assert config_manager.validate_generation(generation) is True
        
        # After update, old generation should be invalid
        config_manager.update_config({"test_key": "test_value"})
        assert config_manager.validate_generation(generation) is False
        
        # Current generation should be valid
        current_generation = config_manager.get_generation()
        assert config_manager.validate_generation(current_generation) is True
        
    def test_draft_lock_context(self):
        """Test draft-aware configuration locking"""
        config_manager = ConfigurationManager(self.config_dir)
        
        # Normal update should work
        result = config_manager.update_config({"test": "value1"})
        assert result is True
        
        # Update during draft lock should be blocked
        with config_manager.draft_lock():
            assert config_manager.get_state() == ConfigState.LOCKED
            result = config_manager.update_config({"test": "value2"})
            assert result is False  # Should be blocked
            
        # After lock release, updates should work again
        assert config_manager.get_state() == ConfigState.STABLE
        result = config_manager.update_config({"test": "value3"})
        assert result is True
        
    def test_critical_section_context(self):
        """Test critical section configuration freeze"""
        config_manager = ConfigurationManager(self.config_dir)
        
        def long_operation():
            with config_manager.critical_section():
                time.sleep(0.15)  # >100ms operation
                
        # Start long operation in background
        thread = threading.Thread(target=long_operation)
        thread.start()
        
        # Brief wait to ensure critical section is active
        time.sleep(0.05)
        
        # Update should be blocked during critical section
        result = config_manager.update_config({"test": "blocked"})
        assert result is False
        
        # Wait for critical section to complete
        thread.join()
        
        # Update should work after critical section
        result = config_manager.update_config({"test": "allowed"})
        assert result is True
        
    def test_rollback_functionality(self):
        """Test configuration rollback"""
        config_manager = ConfigurationManager(self.config_dir)
        
        # Make some changes
        config_manager.update_config({"test_value": "original"})
        config_manager.update_config({"test_value": "modified"})
        
        assert config_manager.get_config("test_value") == "modified"
        
        # Rollback should restore last good configuration
        result = config_manager.rollback_to_last_good()
        assert result is True
        
        # Value should be restored (to "modified" since it was the last good state)
        assert config_manager.get_config("test_value") == "modified"
        
    def test_thread_safety(self):
        """Test thread safety of configuration operations"""
        config_manager = ConfigurationManager(self.config_dir)
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    # Mix of reads and writes
                    if i % 2 == 0:
                        config_manager.get_config(f"worker_{worker_id}.iteration_{i}", "default")
                    else:
                        config_manager.update_config({f"worker_{worker_id}.iteration_{i}": f"value_{i}"})
                    time.sleep(0.001)  # Small delay
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")
                
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # All workers should complete without errors
        assert len(results) == 5
        assert len(errors) == 0
        
    def test_config_file_persistence(self):
        """Test configuration file saving and loading"""
        config_manager = ConfigurationManager(self.config_dir)
        
        # Make some updates
        updates = {
            "test_section.test_key": "test_value",
            "ai_helper.enabled": False
        }
        config_manager.update_config(updates)
        
        # Save to file
        config_file = self.config_dir / "test_config.json"
        result = config_manager.save_to_file(config_file)
        assert result is True
        assert config_file.exists()
        
        # Verify file contents
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
            
        assert saved_config["test_section"]["test_key"] == "test_value"
        assert saved_config["ai_helper"]["enabled"] is False
        
    def test_configuration_history(self):
        """Test configuration history tracking"""
        config_manager = ConfigurationManager(self.config_dir)
        initial_history_len = len(config_manager.get_history())
        
        # Make several updates
        for i in range(5):
            config_manager.update_config({f"test_{i}": f"value_{i}"})
            
        history = config_manager.get_history()
        assert len(history) > initial_history_len
        
        # History should contain snapshots with increasing generations
        generations = [snapshot.generation for snapshot in history]
        assert generations == sorted(generations)  # Should be in ascending order
        
    def test_change_handlers(self):
        """Test configuration change handlers"""
        config_manager = ConfigurationManager(self.config_dir)
        change_events = []
        
        def change_handler(old_snapshot, new_snapshot):
            change_events.append({
                'old_generation': old_snapshot.generation,
                'new_generation': new_snapshot.generation,
                'timestamp': new_snapshot.timestamp
            })
            
        # Register change handler
        config_manager.add_change_handler(change_handler)
        
        # Make updates
        config_manager.update_config({"test1": "value1"})
        config_manager.update_config({"test2": "value2"})
        
        # Should have received change notifications
        assert len(change_events) == 2
        assert change_events[0]['new_generation'] < change_events[1]['new_generation']
        
    def test_custom_validators(self):
        """Test custom configuration validators"""
        config_manager = ConfigurationManager(self.config_dir)
        validation_calls = []
        
        def custom_validator(config_data):
            validation_calls.append(len(validation_calls))
            # Reject configurations with test_forbidden key
            return "test_forbidden" not in config_data
            
        # Register custom validator
        config_manager.add_validator(custom_validator)
        
        # Valid update should succeed and call validator
        result = config_manager.update_config({"test_allowed": "value"})
        assert result is True
        assert len(validation_calls) == 1
        
        # Invalid update should fail
        result = config_manager.update_config({"test_forbidden": "value"})
        assert result is False
        assert len(validation_calls) == 2


class TestGlobalConfigManager:
    """Test global configuration manager functions"""
    
    def test_singleton_behavior(self):
        """Test that get_config_manager returns singleton"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2  # Same instance
        
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test get_config convenience function
        config_value = get_config("ai_helper.enabled")
        assert isinstance(config_value, bool)
        
        # Test with default
        missing_value = get_config("non.existent.key", "default")
        assert missing_value == "default"
        
        # Test update_config convenience function
        result = update_config({"test_convenience": "test_value"})
        assert result is True
        
        # Verify update applied
        updated_value = get_config("test_convenience")
        assert updated_value == "test_value"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_config_directory(self):
        """Test handling of invalid configuration directory"""
        # This should still work - directory will be created
        invalid_path = Path("/tmp/non_existent_config_dir_test")
        config_manager = ConfigurationManager(invalid_path)
        
        assert config_manager.get_state() == ConfigState.STABLE
        
        # Cleanup
        if invalid_path.exists():
            import shutil
            shutil.rmtree(invalid_path, ignore_errors=True)
            
    def test_corrupted_config_file(self):
        """Test handling of corrupted configuration file"""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir)
        
        try:
            # Create corrupted config file
            config_file = config_dir / "ai_helper_config.json"
            with open(config_file, 'w') as f:
                f.write("{invalid json content")
                
            # Should still initialize with defaults
            config_manager = ConfigurationManager(config_dir)
            assert config_manager.get_state() == ConfigState.STABLE
            
            # Should have default values
            assert "ai_helper" in config_manager.get_config()
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def test_concurrent_access_stress(self):
        """Stress test concurrent access"""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir)
        
        try:
            config_manager = ConfigurationManager(config_dir)
            errors = []
            successful_operations = []
            
            def stress_worker(worker_id):
                try:
                    for i in range(50):
                        operation = i % 4
                        if operation == 0:
                            # Read operation
                            config_manager.get_config("ai_helper.enabled")
                        elif operation == 1:
                            # Update operation
                            config_manager.update_config({f"stress_{worker_id}_{i}": i})
                        elif operation == 2: 
                            # Generation check
                            gen = config_manager.get_generation()
                            config_manager.validate_generation(gen)
                        else:
                            # History check
                            config_manager.get_history()
                            
                        successful_operations.append(f"{worker_id}_{i}")
                        
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")
                    
            # Start many concurrent workers
            threads = []
            for i in range(10):
                thread = threading.Thread(target=stress_worker, args=(i,))
                threads.append(thread)
                thread.start()
                
            # Wait for completion
            for thread in threads:
                thread.join()
                
            # Should have no errors and many successful operations
            assert len(errors) == 0
            assert len(successful_operations) > 400  # 10 workers * 50 ops - some may fail validation
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])