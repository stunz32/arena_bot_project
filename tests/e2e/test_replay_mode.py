"""
End-to-End tests for Replay Mode

Tests the complete replay pipeline from image fixtures to final results,
ensuring deterministic processing and proper debug artifact generation.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.cli import run_replay


class TestReplayMode(TestCase):
    """Test replay mode end-to-end functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.fixtures_dir = Path(__file__).parent.parent / "fixtures" / "end_to_end" / "drafts"
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch debug base directory to use temp
        import arena_bot.utils.debug_dump as dump_module
        self.original_debug_base = dump_module.DEBUG_BASE_DIR
        dump_module.DEBUG_BASE_DIR = self.temp_dir / ".debug_runs"
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore debug base
        if self.original_debug_base:
            import arena_bot.utils.debug_dump as dump_module
            dump_module.DEBUG_BASE_DIR = self.original_debug_base
        
        # Clean up temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_replay_fixtures_offline(self):
        """Test replay mode with golden fixtures in offline mode"""
        # Ensure fixtures exist
        self.assertTrue(self.fixtures_dir.exists(), f"Fixtures directory not found: {self.fixtures_dir}")
        
        pack_001 = self.fixtures_dir / "pack_001.png"
        pack_002 = self.fixtures_dir / "pack_002.png"
        self.assertTrue(pack_001.exists(), "pack_001.png fixture missing")
        self.assertTrue(pack_002.exists(), "pack_002.png fixture missing")
        
        # Run replay on fixtures
        results = run_replay(
            paths=str(self.fixtures_dir),
            offline=True,
            debug_tag="phase2_test"
        )
        
        # Verify results structure
        self.assertEqual(len(results), 2, "Should process exactly 2 frames")
        
        # Check each frame result
        for i, result in enumerate(results):
            with self.subTest(frame=i):
                # Verify required fields
                self.assertIn("cards", result)
                self.assertIn("processing_time_ms", result)
                self.assertIn("stage_timings", result)
                self.assertIn("offline_mode", result)
                self.assertIn("sidecar_used", result)
                
                # Verify offline mode
                self.assertTrue(result["offline_mode"])
                self.assertTrue(result["sidecar_used"])
                
                # Verify card count
                cards = result["cards"]
                self.assertEqual(len(cards), 3, "Each frame should have exactly 3 cards")
                
                # Verify card structure
                for card in cards:
                    self.assertIn("id", card)
                    self.assertIn("name", card)
                    self.assertIn("mana_cost", card)
                    self.assertIn("tier_score", card)
                    self.assertIsInstance(card["tier_score"], (int, float))
                
                # Verify cards are sorted by tier_score descending
                tier_scores = [card["tier_score"] for card in cards]
                self.assertEqual(tier_scores, sorted(tier_scores, reverse=True),
                               "Cards should be sorted by tier_score descending")
    
    def test_replay_specific_ordering(self):
        """Test that specific fixture ordering matches expectations"""
        results = run_replay(
            paths=str(self.fixtures_dir),
            offline=True,
            debug_tag="phase2_ordering_test"
        )
        
        self.assertEqual(len(results), 2)
        
        # Find pack_001 result (Fireball should be top)
        pack_001_result = None
        pack_002_result = None
        
        for result in results:
            if "pack_001" in result["image_path"]:
                pack_001_result = result
            elif "pack_002" in result["image_path"]:
                pack_002_result = result
        
        self.assertIsNotNone(pack_001_result, "pack_001 result not found")
        self.assertIsNotNone(pack_002_result, "pack_002 result not found")
        
        # Verify pack_001 ordering: Fireball (85.0) > Frostbolt (72.0) > Arcane Intellect (68.0)
        pack_001_cards = pack_001_result["cards"]
        self.assertEqual(pack_001_cards[0]["name"], "Fireball")
        self.assertEqual(pack_001_cards[0]["tier_score"], 85.0)
        self.assertEqual(pack_001_cards[1]["name"], "Frostbolt")
        self.assertEqual(pack_001_cards[2]["name"], "Arcane Intellect")
        
        # Verify pack_002 ordering: Polymorph (82.0) > Flamestrike (78.0) > Mirror Image (45.0)
        pack_002_cards = pack_002_result["cards"]
        self.assertEqual(pack_002_cards[0]["name"], "Polymorph")
        self.assertEqual(pack_002_cards[0]["tier_score"], 82.0)
        self.assertEqual(pack_002_cards[1]["name"], "Flamestrike")
        self.assertEqual(pack_002_cards[2]["name"], "Mirror Image")
    
    def test_replay_debug_artifacts(self):
        """Test that debug artifacts are properly generated"""
        debug_tag = "phase2_debug_test"
        
        results = run_replay(
            paths=str(self.fixtures_dir),
            offline=True,
            debug_tag=debug_tag
        )
        
        self.assertEqual(len(results), 2)
        
        # Check debug directory was created
        debug_base = self.temp_dir / ".debug_runs"
        self.assertTrue(debug_base.exists())
        
        # Find the debug run directory
        debug_runs = list(debug_base.glob(f"*_{debug_tag}_*"))
        self.assertEqual(len(debug_runs), 1, "Should create exactly one debug run")
        
        debug_run_dir = debug_runs[0]
        
        # Check metadata file
        metadata_file = debug_run_dir / "run_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["tag"], debug_tag)
        self.assertIn("artifacts", metadata)
        
        # Check per-frame artifacts were created
        result_files = list(debug_run_dir.glob("result_pack_*.json"))
        self.assertEqual(len(result_files), 2, "Should have result JSON for each frame")
        
        image_files = list(debug_run_dir.glob("frame_pack_*.png"))
        self.assertEqual(len(image_files), 2, "Should have image dump for each frame")
        
        timing_files = list(debug_run_dir.glob("timings_pack_*.json"))
        self.assertEqual(len(timing_files), 2, "Should have timing data for each frame")
        
        # Verify one result file content
        with open(result_files[0]) as f:
            result_data = json.load(f)
        
        self.assertIn("cards", result_data)
        self.assertIn("stage_timings", result_data)
        self.assertEqual(len(result_data["cards"]), 3)
    
    def test_replay_no_sidecar_fallback(self):
        """Test replay mode fallback when no sidecar labels exist"""
        # Create temporary image without sidecar
        temp_image_dir = self.temp_dir / "no_sidecar"
        temp_image_dir.mkdir()
        
        # Copy one fixture without its sidecar
        pack_001_original = self.fixtures_dir / "pack_001.png"
        pack_001_copy = temp_image_dir / "test_no_sidecar.png"
        shutil.copy2(pack_001_original, pack_001_copy)
        
        # Run replay (should use mock detection)
        results = run_replay(
            paths=str(temp_image_dir),
            offline=True,
            debug_tag="no_sidecar_test"
        )
        
        self.assertEqual(len(results), 1)
        
        result = results[0]
        self.assertFalse(result["sidecar_used"], "Should not use sidecar when none exists")
        self.assertTrue(result["offline_mode"])
        
        # Should still have 3 cards (mock data)
        self.assertEqual(len(result["cards"]), 3)
        
        # Verify mock card names
        card_names = [card["name"] for card in result["cards"]]
        expected_mock_names = ["Mock Frostbolt", "Mock Fireball", "Mock Flamestrike"]  # Sorted by tier_score
        self.assertEqual(card_names, expected_mock_names)
    
    def test_replay_empty_directory(self):
        """Test replay mode with empty directory"""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        results = run_replay(
            paths=str(empty_dir),
            offline=True,
            debug_tag="empty_test"
        )
        
        self.assertEqual(len(results), 0, "Should return empty results for empty directory")
    
    def test_replay_timing_structure(self):
        """Test that timing data has expected structure"""
        results = run_replay(
            paths=str(self.fixtures_dir),
            offline=True,
            debug_tag="timing_test"
        )
        
        self.assertGreater(len(results), 0)
        
        result = results[0]
        stage_timings = result["stage_timings"]
        
        # Verify expected stages are present
        expected_stages = [
            "coordinates", "eligibility_filter", "histogram_match",
            "template_validation", "ai_advisor", "ui_render"
        ]
        
        for stage in expected_stages:
            self.assertIn(stage, stage_timings, f"Stage '{stage}' missing from timings")
            self.assertIsInstance(stage_timings[stage], (int, float))
            self.assertGreater(stage_timings[stage], 0, f"Stage '{stage}' should have positive timing")
        
        # Verify total timing
        total_time = result["processing_time_ms"]
        expected_total = sum(stage_timings.values())
        self.assertEqual(total_time, expected_total, "Total time should match sum of stage timings")