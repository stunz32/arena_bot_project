"""
Test suite for debug dump utility - filesystem behavior and JSON validity.

Tests the black-box debug dump system to ensure reliable artifact capture,
proper filesystem operations, and JSON serialization correctness.
"""

import json
import tempfile
import shutil
import threading
import time
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.utils.debug_dump import (
    begin_run, dump_image, dump_json, end_run, debug_run,
    get_current_run_dir, is_debug_active, DebugDumpError,
    dump_detection_failure, dump_stage_timing,
    _get_current_run, _set_current_run
)


class TestDebugDumpCore(TestCase):
    """Test core debug dump functionality"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.original_debug_base = None
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch the DEBUG_BASE_DIR to use our temp directory
        import arena_bot.utils.debug_dump as dump_module
        self.original_debug_base = dump_module.DEBUG_BASE_DIR
        dump_module.DEBUG_BASE_DIR = self.temp_dir / ".debug_runs"
        
        # Ensure clean state
        _set_current_run(None)
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original debug base
        if self.original_debug_base:
            import arena_bot.utils.debug_dump as dump_module
            dump_module.DEBUG_BASE_DIR = self.original_debug_base
        
        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Ensure clean state
        _set_current_run(None)
    
    def test_begin_run_creates_directory(self):
        """Test that begin_run creates timestamped directory with metadata"""
        tag = "test_pipeline"
        run_dir = begin_run(tag)
        
        # Check directory exists
        self.assertTrue(run_dir.exists())
        self.assertTrue(run_dir.is_dir())
        
        # Check directory name format
        dir_name = run_dir.name
        parts = dir_name.split('_')
        self.assertGreaterEqual(len(parts), 3)  # timestamp_tag_uuid
        self.assertIn(tag, dir_name)
        
        # Check metadata file
        metadata_file = run_dir / "run_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["tag"], tag)
        self.assertIn("start_time", metadata)
        self.assertIn("run_id", metadata)
        self.assertIn("thread_id", metadata)
        self.assertEqual(metadata["artifacts"], [])
    
    def test_begin_run_invalid_tag(self):
        """Test begin_run with invalid tag raises error"""
        with self.assertRaises(DebugDumpError):
            begin_run("")
        
        with self.assertRaises(DebugDumpError):
            begin_run(None)
        
        with self.assertRaises(DebugDumpError):
            begin_run(123)
    
    def test_get_current_run_tracking(self):
        """Test current run tracking works correctly"""
        # Initially no run
        self.assertIsNone(get_current_run_dir())
        self.assertFalse(is_debug_active())
        
        # Start run
        run_dir = begin_run("test")
        self.assertEqual(get_current_run_dir(), run_dir)
        self.assertTrue(is_debug_active())
        
        # End run
        end_run()
        self.assertIsNone(get_current_run_dir())
        self.assertFalse(is_debug_active())
    
    def test_dump_json_valid_object(self):
        """Test JSON dumping with valid objects"""
        begin_run("json_test")
        
        test_obj = {
            "stage": "detection",
            "confidence": 0.85,
            "candidates": ["card1", "card2"],
            "metadata": {"timestamp": "2025-01-01T00:00:00"}
        }
        
        json_path = dump_json(test_obj, "test_results")
        
        # Check file exists
        self.assertIsNotNone(json_path)
        self.assertTrue(json_path.exists())
        self.assertEqual(json_path.suffix, ".json")
        
        # Check content is valid JSON
        with open(json_path) as f:
            loaded_data = json.load(f)
        
        # Original data should be present
        for key, value in test_obj.items():
            self.assertEqual(loaded_data[key], value)
        
        # Debug metadata should be added
        self.assertIn("_debug_timestamp", loaded_data)
        self.assertIn("_debug_thread_id", loaded_data)
        
        end_run()
    
    def test_dump_json_invalid_inputs(self):
        """Test JSON dumping with invalid inputs"""
        begin_run("json_error_test")
        
        # Invalid name
        with self.assertRaises(DebugDumpError):
            dump_json({"test": "data"}, "")
        
        with self.assertRaises(DebugDumpError):
            dump_json({"test": "data"}, None)
        
        # Invalid object
        with self.assertRaises(DebugDumpError):
            dump_json("not a dict", "test")
        
        with self.assertRaises(DebugDumpError):
            dump_json(None, "test")
        
        end_run()
    
    def test_dump_json_no_active_run(self):
        """Test JSON dumping when no debug run is active"""
        # Should return None and not crash
        result = dump_json({"test": "data"}, "test")
        self.assertIsNone(result)
    
    def test_end_run_finalizes_metadata(self):
        """Test that end_run properly finalizes metadata"""
        run_dir = begin_run("finalize_test")
        
        # Add some artifacts
        dump_json({"test": "data"}, "test_artifact")
        
        end_run()
        
        # Check metadata was finalized
        metadata_file = run_dir / "run_metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        self.assertIn("end_time", metadata)
        self.assertIn("duration_ms", metadata)
        self.assertGreater(metadata["duration_ms"], 0)
        self.assertEqual(len(metadata["artifacts"]), 1)
    
    def test_debug_run_context_manager(self):
        """Test debug_run context manager"""
        with debug_run("context_test") as run_dir:
            self.assertTrue(is_debug_active())
            self.assertEqual(get_current_run_dir(), run_dir)
            
            # Can dump artifacts
            dump_json({"test": "context"}, "context_artifact")
        
        # Should be cleaned up after context
        self.assertFalse(is_debug_active())
        self.assertIsNone(get_current_run_dir())
    
    def test_thread_local_isolation(self):
        """Test that debug runs are isolated per thread"""
        results = {}
        
        def thread_worker(thread_id):
            tag = f"thread_{thread_id}"
            run_dir = begin_run(tag)
            results[thread_id] = {
                'run_dir': run_dir,
                'is_active': is_debug_active(),
                'current_dir': get_current_run_dir()
            }
            time.sleep(0.1)  # Let other threads start
            dump_json({"thread": thread_id}, f"data_{thread_id}")
            end_run()
        
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check each thread had its own isolated run
        self.assertEqual(len(results), 3)
        run_dirs = [r['run_dir'] for r in results.values()]
        self.assertEqual(len(set(run_dirs)), 3)  # All different directories
        
        # Check artifacts were created in each thread's directory
        for thread_id, result in results.items():
            artifact_file = result['run_dir'] / f"data_{thread_id}.json"
            self.assertTrue(artifact_file.exists())


class TestDebugDumpImageHandling(TestCase):
    """Test image handling in debug dump"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch the DEBUG_BASE_DIR
        import arena_bot.utils.debug_dump as dump_module
        self.original_debug_base = dump_module.DEBUG_BASE_DIR
        dump_module.DEBUG_BASE_DIR = self.temp_dir / ".debug_runs"
        
        _set_current_run(None)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.original_debug_base:
            import arena_bot.utils.debug_dump as dump_module
            dump_module.DEBUG_BASE_DIR = self.original_debug_base
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        _set_current_run(None)
    
    @mock.patch('arena_bot.utils.debug_dump.HAS_CV2', False)
    def test_dump_image_file_path(self):
        """Test dumping image from file path (without OpenCV)"""
        begin_run("image_test")
        
        # Create a dummy image file
        test_image = self.temp_dir / "test_image.png"
        test_image.write_text("fake image data")
        
        result_path = dump_image(str(test_image), "copied_image")
        
        # Check file was copied
        self.assertIsNotNone(result_path)
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.suffix, ".png")
        
        # Check content was copied
        self.assertEqual(result_path.read_text(), "fake image data")
        
        end_run()
    
    def test_dump_image_invalid_inputs(self):
        """Test image dumping with invalid inputs"""
        begin_run("image_error_test")
        
        # Invalid name
        with self.assertRaises(DebugDumpError):
            dump_image("dummy_path", "")
        
        # Non-existent file
        with self.assertRaises(DebugDumpError):
            dump_image("/nonexistent/file.png", "test")
        
        end_run()
    
    def test_dump_image_with_metadata(self):
        """Test image dumping with metadata"""
        begin_run("image_metadata_test")
        
        # Create dummy image
        test_image = self.temp_dir / "test.jpg"
        test_image.write_text("dummy")
        
        metadata = {
            "resolution": "1920x1080",
            "detection_confidence": 0.85,
            "stage": "coordinate_detection"
        }
        
        image_path = dump_image(str(test_image), "test_with_meta", metadata)
        
        # Check metadata file was created
        metadata_path = image_path.parent / "test_with_meta_metadata.json"
        self.assertTrue(metadata_path.exists())
        
        with open(metadata_path) as f:
            saved_metadata = json.load(f)
        
        for key, value in metadata.items():
            self.assertEqual(saved_metadata[key], value)
        
        end_run()


class TestDebugDumpConvenience(TestCase):
    """Test convenience functions for pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        import arena_bot.utils.debug_dump as dump_module
        self.original_debug_base = dump_module.DEBUG_BASE_DIR
        dump_module.DEBUG_BASE_DIR = self.temp_dir / ".debug_runs"
        
        _set_current_run(None)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.original_debug_base:
            import arena_bot.utils.debug_dump as dump_module
            dump_module.DEBUG_BASE_DIR = self.original_debug_base
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        _set_current_run(None)
    
    @mock.patch('arena_bot.utils.debug_dump.HAS_CV2', False)
    def test_dump_detection_failure(self):
        """Test detection failure convenience function"""
        # Create fake numpy array (mock it since we don't have opencv)
        fake_frame = MagicMock()
        fake_frame.__class__.__name__ = 'ndarray'
        
        begin_run("detection_failure_test")
        
        candidates = [
            {"name": "card1", "confidence": 0.3},
            {"name": "card2", "confidence": 0.2}
        ]
        
        with mock.patch('arena_bot.utils.debug_dump.dump_image') as mock_dump_image:
            dump_detection_failure(
                fake_frame, candidates, 0.5, 
                "histogram_match", "Low confidence matches"
            )
        
        # Check image dump was called
        mock_dump_image.assert_called_once()
        
        # Check context JSON was created
        run_dir = get_current_run_dir()
        context_files = list(run_dir.glob("failure_histogram_match_context.json"))
        self.assertEqual(len(context_files), 1)
        
        with open(context_files[0]) as f:
            context = json.load(f)
        
        self.assertEqual(context["stage"], "histogram_match")
        self.assertEqual(context["error"], "Low confidence matches")
        self.assertEqual(context["confidence_threshold"], 0.5)
        self.assertEqual(context["num_candidates"], 2)
        
        end_run()
    
    def test_dump_stage_timing(self):
        """Test stage timing convenience function"""
        begin_run("timing_test")
        
        dump_stage_timing("coordinate_detection", 45.7, {"resolution": "1920x1080"})
        
        # Check timing file was created
        run_dir = get_current_run_dir()
        timing_files = list(run_dir.glob("timing_coordinate_detection.json"))
        self.assertEqual(len(timing_files), 1)
        
        with open(timing_files[0]) as f:
            timing_data = json.load(f)
        
        self.assertEqual(timing_data["stage"], "coordinate_detection")
        self.assertEqual(timing_data["duration_ms"], 45.7)
        self.assertEqual(timing_data["resolution"], "1920x1080")
        self.assertIn("timestamp", timing_data)
        
        end_run()
    
    def test_convenience_functions_no_active_run(self):
        """Test convenience functions when no debug run is active"""
        # Should not crash
        fake_frame = MagicMock()
        dump_detection_failure(fake_frame, [], 0.5, "test", "error")
        dump_stage_timing("test", 100.0)


class TestDebugDumpEdgeCases(TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        import arena_bot.utils.debug_dump as dump_module
        self.original_debug_base = dump_module.DEBUG_BASE_DIR
        dump_module.DEBUG_BASE_DIR = self.temp_dir / ".debug_runs"
        
        _set_current_run(None)
    
    def tearDown(self):
        """Clean up test environment"""  
        if self.original_debug_base:
            import arena_bot.utils.debug_dump as dump_module
            dump_module.DEBUG_BASE_DIR = self.original_debug_base
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        _set_current_run(None)
    
    def test_name_sanitization(self):
        """Test that names are properly sanitized for filesystem"""
        begin_run("sanitization_test")
        
        # Names with special characters
        weird_name = "test/name\\with:special*chars?"
        
        json_path = dump_json({"test": "data"}, weird_name)
        
        # Check sanitized filename
        expected_sanitized = "test_name_with_special_chars_"
        self.assertTrue(json_path.name.startswith(expected_sanitized))
        
        end_run()
    
    def test_multiple_runs_same_tag(self):
        """Test that multiple runs with same tag get unique directories"""
        tag = "same_tag"
        
        run_dir1 = begin_run(tag)
        end_run()
        
        run_dir2 = begin_run(tag) 
        end_run()
        
        # Should be different directories
        self.assertNotEqual(run_dir1, run_dir2)
        self.assertTrue(run_dir1.exists())
        self.assertTrue(run_dir2.exists())
    
    def test_json_serialization_complex_objects(self):
        """Test JSON serialization with complex objects"""
        begin_run("complex_json_test")
        
        from datetime import datetime
        
        complex_obj = {
            "timestamp": datetime.now(),  # Will be converted to string
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"}
            },
            "none_value": None,
            "boolean": True
        }
        
        json_path = dump_json(complex_obj, "complex_data")
        
        # Should successfully serialize
        with open(json_path) as f:
            loaded = json.load(f)
        
        # Check structure is preserved
        self.assertEqual(loaded["nested"]["list"], [1, 2, 3])
        self.assertEqual(loaded["boolean"], True)
        self.assertIsNone(loaded["none_value"])
        
        end_run()