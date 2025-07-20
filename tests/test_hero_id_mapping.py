"""
Unit Tests for Hero ID Mapping and Class Translation

Comprehensive tests for hero ID mapping functionality, class translation,
and cards database integration with known samples and edge cases.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.data.cards_json_loader import CardsJsonLoader


class TestHeroIdMapping(unittest.TestCase):
    """Test suite for hero ID mapping and class translation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock cards data for testing
        self.mock_cards_data = {
            "HERO_01": {
                "id": "HERO_01",
                "name": "Garrosh Hellscream",
                "playerClass": "WARRIOR",
                "type": "HERO",
                "dbfId": 813
            },
            "HERO_02": {
                "id": "HERO_02", 
                "name": "Jaina Proudmoore",
                "playerClass": "MAGE",
                "type": "HERO",
                "dbfId": 637
            },
            "HERO_03": {
                "id": "HERO_03",
                "name": "Rexxar", 
                "playerClass": "HUNTER",
                "type": "HERO",
                "dbfId": 31
            },
            "HERO_04": {
                "id": "HERO_04",
                "name": "Uther Lightbringer",
                "playerClass": "PALADIN", 
                "type": "HERO",
                "dbfId": 671
            },
            "HERO_05": {
                "id": "HERO_05",
                "name": "Anduin Wrynn",
                "playerClass": "PRIEST",
                "type": "HERO", 
                "dbfId": 813
            },
            "HERO_06": {
                "id": "HERO_06",
                "name": "Valeera Sanguinar",
                "playerClass": "ROGUE",
                "type": "HERO",
                "dbfId": 930
            },
            "HERO_07": {
                "id": "HERO_07",
                "name": "Thrall",
                "playerClass": "SHAMAN", 
                "type": "HERO",
                "dbfId": 1066
            },
            "HERO_08": {
                "id": "HERO_08",
                "name": "Gul'dan",
                "playerClass": "WARLOCK",
                "type": "HERO",
                "dbfId": 893
            },
            "HERO_09": {
                "id": "HERO_09",
                "name": "Malfurion Stormrage", 
                "playerClass": "DRUID",
                "type": "HERO",
                "dbfId": 274
            },
            "HERO_10": {
                "id": "HERO_10",
                "name": "Illidan Stormrage",
                "playerClass": "DEMONHUNTER",
                "type": "HERO",
                "dbfId": 56550
            },
            # Add some regular cards for testing
            "CS2_023": {
                "id": "CS2_023",
                "name": "Arcane Intellect",
                "playerClass": "MAGE",
                "type": "SPELL",
                "dbfId": 489,
                "cost": 3
            },
            "CS2_102": {
                "id": "CS2_102", 
                "name": "Heroic Strike",
                "playerClass": "WARRIOR",
                "type": "SPELL",
                "dbfId": 1003,
                "cost": 2
            }
        }
        
        # Create mock loader instance
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load:
            mock_load.return_value = self.mock_cards_data
            self.loader = CardsJsonLoader()
    
    def test_dbf_id_to_card_id_mapping_creation(self):
        """Test that DBF ID to card ID mapping is created correctly."""
        # Test known mappings
        self.assertEqual(self.loader.get_card_id_from_dbf_id(813), "HERO_01")
        self.assertEqual(self.loader.get_card_id_from_dbf_id(637), "HERO_02")
        self.assertEqual(self.loader.get_card_id_from_dbf_id(31), "HERO_03")
        self.assertEqual(self.loader.get_card_id_from_dbf_id(489), "CS2_023")
        
        # Test non-existent DBF ID
        self.assertIsNone(self.loader.get_card_id_from_dbf_id(99999))
    
    def test_card_id_to_dbf_id_mapping_creation(self):
        """Test that card ID to DBF ID mapping is created correctly.""" 
        # Test known mappings
        self.assertEqual(self.loader.get_dbf_id_from_card_id("HERO_01"), 813)
        self.assertEqual(self.loader.get_dbf_id_from_card_id("HERO_02"), 637)
        self.assertEqual(self.loader.get_dbf_id_from_card_id("CS2_023"), 489)
        
        # Test non-existent card ID
        self.assertIsNone(self.loader.get_dbf_id_from_card_id("INVALID_CARD"))
    
    def test_hero_class_translation(self):
        """Test hero card ID to class name translation."""
        # Test all hero classes
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_01"), "WARRIOR")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_02"), "MAGE")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_03"), "HUNTER")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_04"), "PALADIN")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_05"), "PRIEST")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_06"), "ROGUE")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_07"), "SHAMAN")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_08"), "WARLOCK")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_09"), "DRUID")
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_10"), "DEMONHUNTER")
        
        # Test non-hero card
        self.assertIsNone(self.loader.get_class_from_hero_card_id("CS2_023"))
        
        # Test non-existent card
        self.assertIsNone(self.loader.get_class_from_hero_card_id("INVALID_HERO"))
    
    def test_mapping_statistics_logging(self):
        """Test that mapping statistics are logged correctly."""
        stats = self.loader.get_id_mapping_statistics()
        
        # Check basic statistics
        self.assertIn('total_cards_loaded', stats)
        self.assertIn('dbf_id_mappings', stats)
        self.assertIn('hero_cards_found', stats)
        self.assertIn('mapping_status', stats)
        
        # Verify counts
        self.assertEqual(stats['total_cards_loaded'], len(self.mock_cards_data))
        self.assertEqual(stats['dbf_id_mappings'], len(self.mock_cards_data))
        self.assertEqual(stats['hero_cards_found'], 10)  # 10 hero cards in mock data
        self.assertEqual(stats['mapping_status'], 'complete')
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with None input
        self.assertIsNone(self.loader.get_card_id_from_dbf_id(None))
        self.assertIsNone(self.loader.get_dbf_id_from_card_id(None))
        self.assertIsNone(self.loader.get_class_from_hero_card_id(None))
        
        # Test with invalid types
        self.assertIsNone(self.loader.get_card_id_from_dbf_id("not_a_number"))
        self.assertIsNone(self.loader.get_dbf_id_from_card_id(12345))
        self.assertIsNone(self.loader.get_class_from_hero_card_id(67890))
        
        # Test with empty string
        self.assertIsNone(self.loader.get_dbf_id_from_card_id(""))
        self.assertIsNone(self.loader.get_class_from_hero_card_id(""))
    
    def test_case_sensitivity(self):
        """Test case sensitivity handling."""
        # Test that card IDs are case sensitive (as expected)
        self.assertEqual(self.loader.get_class_from_hero_card_id("HERO_01"), "WARRIOR")
        self.assertIsNone(self.loader.get_class_from_hero_card_id("hero_01"))
        self.assertIsNone(self.loader.get_class_from_hero_card_id("Hero_01"))
    
    def test_duplicate_dbf_ids(self):
        """Test handling of duplicate DBF IDs."""
        # In our mock data, HERO_01 and HERO_05 both have dbfId 813
        # The loader should handle this gracefully, typically keeping the first one
        result = self.loader.get_card_id_from_dbf_id(813)
        self.assertIsNotNone(result)
        self.assertIn(result, ["HERO_01", "HERO_05"])
    
    def test_performance_with_large_dataset(self):
        """Test performance characteristics with larger dataset."""
        import time
        
        # Test lookup performance (should be O(1) due to dictionary)
        start_time = time.time()
        for _ in range(1000):
            self.loader.get_card_id_from_dbf_id(637)
            self.loader.get_class_from_hero_card_id("HERO_02")
        end_time = time.time()
        
        # Should complete 1000 lookups in well under 1 second
        self.assertLess(end_time - start_time, 1.0)
    
    def test_integration_with_hsreplay_data(self):
        """Test integration scenarios with HSReplay data formats."""
        # Test common HSReplay card ID patterns
        test_cases = [
            (813, "HERO_01"),   # Warrior hero
            (637, "HERO_02"),   # Mage hero
            (489, "CS2_023"),   # Regular spell
        ]
        
        for dbf_id, expected_card_id in test_cases:
            result = self.loader.get_card_id_from_dbf_id(dbf_id)
            self.assertEqual(result, expected_card_id, 
                           f"DBF ID {dbf_id} should map to {expected_card_id}, got {result}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of mapping structures."""
        # Check that mappings are created without excessive memory usage
        import sys
        
        # Get size of mapping dictionaries
        dbf_map_size = sys.getsizeof(self.loader.dbf_id_to_card_id_map)
        card_map_size = sys.getsizeof(self.loader.card_id_to_dbf_id_map)
        
        # With our test data, these should be reasonable sizes
        self.assertLess(dbf_map_size, 10000)  # Less than 10KB
        self.assertLess(card_map_size, 10000)  # Less than 10KB
    
    def test_thread_safety(self):
        """Test thread safety of mapping lookups."""
        import threading
        import time
        
        results = []
        errors = []
        
        def lookup_worker():
            try:
                for _ in range(100):
                    result1 = self.loader.get_card_id_from_dbf_id(637)
                    result2 = self.loader.get_class_from_hero_card_id("HERO_02")
                    results.append((result1, result2))
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=lookup_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 500)  # 5 threads * 100 iterations
        
        # All results should be consistent
        for result1, result2 in results:
            self.assertEqual(result1, "HERO_02")
            self.assertEqual(result2, "MAGE")


class TestHeroIdMappingIntegration(unittest.TestCase):
    """Integration tests for hero ID mapping with real data scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a more realistic mock dataset
        self.realistic_cards_data = self._create_realistic_cards_data()
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load:
            mock_load.return_value = self.realistic_cards_data
            self.loader = CardsJsonLoader()
    
    def _create_realistic_cards_data(self):
        """Create realistic cards data for integration testing."""
        # This simulates a more realistic cards.json structure
        cards_data = {}
        
        # Add heroes
        hero_data = [
            ("HERO_01", "Garrosh Hellscream", "WARRIOR", 813),
            ("HERO_02", "Jaina Proudmoore", "MAGE", 637),
            ("HERO_03", "Rexxar", "HUNTER", 31),
            ("HERO_04", "Uther Lightbringer", "PALADIN", 671),
            ("HERO_05", "Anduin Wrynn", "PRIEST", 813),  # Duplicate DBF ID for testing
            ("HERO_06", "Valeera Sanguinar", "ROGUE", 930),
            ("HERO_07", "Thrall", "SHAMAN", 1066),
            ("HERO_08", "Gul'dan", "WARLOCK", 893),
            ("HERO_09", "Malfurion Stormrage", "DRUID", 274),
            ("HERO_10", "Illidan Stormrage", "DEMONHUNTER", 56550),
        ]
        
        for card_id, name, player_class, dbf_id in hero_data:
            cards_data[card_id] = {
                "id": card_id,
                "name": name,
                "playerClass": player_class,
                "type": "HERO",
                "dbfId": dbf_id
            }
        
        # Add some regular cards to simulate realistic environment
        regular_cards = [
            ("CS2_023", "Arcane Intellect", "MAGE", "SPELL", 489, 3),
            ("CS2_102", "Heroic Strike", "WARRIOR", "SPELL", 1003, 2),
            ("CS2_061", "Drain Life", "WARLOCK", "SPELL", 1009, 2),
            ("CS2_089", "Holy Light", "PALADIN", "SPELL", 1060, 2),
            ("CS2_025", "Arcane Shot", "HUNTER", "SPELL", 25, 1),
        ]
        
        for card_id, name, player_class, card_type, dbf_id, cost in regular_cards:
            cards_data[card_id] = {
                "id": card_id,
                "name": name,
                "playerClass": player_class,
                "type": card_type,
                "dbfId": dbf_id,
                "cost": cost
            }
        
        return cards_data
    
    def test_complete_hero_mapping_workflow(self):
        """Test complete workflow from DBF ID to class name."""
        # Test the complete workflow: DBF ID -> Card ID -> Class Name
        test_workflows = [
            (813, "HERO_01", "WARRIOR"),   # Note: might get HERO_05 due to duplicate
            (637, "HERO_02", "MAGE"),
            (31, "HERO_03", "HUNTER"),
            (671, "HERO_04", "PALADIN"),
            (930, "HERO_06", "ROGUE"),
            (1066, "HERO_07", "SHAMAN"),
            (893, "HERO_08", "WARLOCK"),
            (274, "HERO_09", "DRUID"),
            (56550, "HERO_10", "DEMONHUNTER"),
        ]
        
        for dbf_id, expected_card_id, expected_class in test_workflows:
            # Step 1: DBF ID to Card ID
            card_id = self.loader.get_card_id_from_dbf_id(dbf_id)
            self.assertIsNotNone(card_id, f"Failed to get card ID for DBF ID {dbf_id}")
            
            # For duplicate DBF IDs, just check it's one of the valid options
            if dbf_id == 813:  # Duplicate DBF ID case
                self.assertIn(card_id, ["HERO_01", "HERO_05"])
                expected_class = "WARRIOR" if card_id == "HERO_01" else "PRIEST"
            else:
                self.assertEqual(card_id, expected_card_id)
            
            # Step 2: Card ID to Class Name
            class_name = self.loader.get_class_from_hero_card_id(card_id)
            self.assertEqual(class_name, expected_class)
    
    def test_hsreplay_integration_scenario(self):
        """Test typical HSReplay API integration scenario."""
        # Simulate receiving hero data from HSReplay API
        hsreplay_hero_data = [
            {"dbf_id": 637, "name": "Jaina Proudmoore"},
            {"dbf_id": 813, "name": "Garrosh Hellscream"},  
            {"dbf_id": 31, "name": "Rexxar"},
        ]
        
        # Process each hero through our mapping system
        processed_heroes = []
        for hero_data in hsreplay_hero_data:
            dbf_id = hero_data["dbf_id"]
            
            # Get card ID from DBF ID (as we would with HSReplay data)
            card_id = self.loader.get_card_id_from_dbf_id(dbf_id)
            self.assertIsNotNone(card_id, f"Could not map DBF ID {dbf_id}")
            
            # Get class name from card ID
            class_name = self.loader.get_class_from_hero_card_id(card_id)
            self.assertIsNotNone(class_name, f"Could not get class for card {card_id}")
            
            processed_heroes.append({
                "original_data": hero_data,
                "card_id": card_id,
                "class_name": class_name
            })
        
        # Verify we got expected classes
        class_names = [hero["class_name"] for hero in processed_heroes]
        self.assertIn("MAGE", class_names)
        self.assertIn("HUNTER", class_names)
        # Note: WARRIOR or PRIEST possible due to duplicate DBF ID 813
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test handling of invalid DBF IDs (as might come from corrupted API data)
        invalid_dbf_ids = [None, 0, -1, 999999, "invalid", [], {}]
        
        for invalid_id in invalid_dbf_ids:
            result = self.loader.get_card_id_from_dbf_id(invalid_id)
            self.assertIsNone(result, f"Should return None for invalid DBF ID: {invalid_id}")
        
        # Test handling of invalid card IDs
        invalid_card_ids = [None, "", "INVALID_HERO", "not_a_hero", 12345, [], {}]
        
        for invalid_id in invalid_card_ids:
            result = self.loader.get_class_from_hero_card_id(invalid_id)
            self.assertIsNone(result, f"Should return None for invalid card ID: {invalid_id}")
    
    def test_logging_and_statistics_integration(self):
        """Test logging and statistics in integration context."""
        # Get mapping statistics
        stats = self.loader.get_id_mapping_statistics()
        
        # Verify comprehensive statistics
        expected_keys = [
            'total_cards_loaded', 'dbf_id_mappings', 'hero_cards_found',
            'mapping_status', 'duplicate_dbf_ids', 'load_time_ms'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing expected statistic: {key}")
        
        # Verify realistic values
        self.assertGreater(stats['total_cards_loaded'], 10)
        self.assertEqual(stats['hero_cards_found'], 10)
        self.assertGreaterEqual(stats['duplicate_dbf_ids'], 1)  # We have at least one duplicate
        self.assertEqual(stats['mapping_status'], 'complete')


def run_hero_id_mapping_tests():
    """Run all hero ID mapping tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestHeroIdMapping))
    test_suite.addTest(unittest.makeSuite(TestHeroIdMappingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_hero_id_mapping_tests()
    
    if success:
        print("\n✅ All hero ID mapping tests passed!")
    else:
        print("\n❌ Some hero ID mapping tests failed!")
        sys.exit(1)