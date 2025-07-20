"""
Unit tests for ID mapping functionality in CardsJsonLoader.

Tests the critical dbf_id ↔ card_id translation required for HSReplay integration.
"""

import unittest
import logging
from pathlib import Path
from unittest.mock import patch, mock_open
from typing import Dict, Any

# Assuming the project structure
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from arena_bot.data.cards_json_loader import CardsJsonLoader


class TestIDMapping(unittest.TestCase):
    """Test cases for HSReplay ID mapping functionality."""
    
    def setUp(self):
        """Set up test fixtures with sample card data."""
        # Sample cards.json data for testing
        self.sample_cards_data = [
            {
                "id": "TOY_380",
                "name": "Forge-wrought Pal",
                "dbfId": 82615,
                "cardClass": "PALADIN",
                "type": "MINION",
                "collectible": True,
                "cost": 2,
                "attack": 2,
                "health": 3
            },
            {
                "id": "HERO_01",
                "name": "Garrosh Hellscream",
                "dbfId": 7,
                "cardClass": "WARRIOR",
                "type": "HERO",
                "collectible": False
            },
            {
                "id": "EX1_339",
                "name": "Wolfrider",
                "dbfId": 576,
                "cardClass": "NEUTRAL",
                "type": "MINION",
                "collectible": True,
                "cost": 3,
                "attack": 3,
                "health": 1
            },
            {
                "id": "HERO_02",
                "name": "Jaina Proudmoore",
                "dbfId": 637,
                "cardClass": "MAGE",
                "type": "HERO",
                "collectible": False
            }
        ]
        
        # Mock the cards.json file loading
        self.mock_cards_json = mock_open(read_data=str(self.sample_cards_data).replace("'", '"'))
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_dbf_id_mapping_creation(self, mock_json_load, mock_exists, mock_file):
        """Test that dbf_id mappings are created correctly during initialization."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_cards_data
        
        loader = CardsJsonLoader()
        
        # Verify dbf_id to card_id mapping
        self.assertEqual(loader.get_card_id_from_dbf_id(82615), "TOY_380")
        self.assertEqual(loader.get_card_id_from_dbf_id(576), "EX1_339")
        self.assertEqual(loader.get_card_id_from_dbf_id(7), "HERO_01")
        self.assertEqual(loader.get_card_id_from_dbf_id(637), "HERO_02")
        
        # Test non-existent dbf_id
        self.assertIsNone(loader.get_card_id_from_dbf_id(999999))
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_card_id_to_dbf_id_mapping(self, mock_json_load, mock_exists, mock_file):
        """Test reverse mapping from card_id to dbf_id."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_cards_data
        
        loader = CardsJsonLoader()
        
        # Verify card_id to dbf_id mapping
        self.assertEqual(loader.get_dbf_id_from_card_id("TOY_380"), 82615)
        self.assertEqual(loader.get_dbf_id_from_card_id("EX1_339"), 576)
        self.assertEqual(loader.get_dbf_id_from_card_id("HERO_01"), 7)
        self.assertEqual(loader.get_dbf_id_from_card_id("HERO_02"), 637)
        
        # Test non-existent card_id
        self.assertIsNone(loader.get_dbf_id_from_card_id("FAKE_CARD"))
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_hero_class_mapping(self, mock_json_load, mock_exists, mock_file):
        """Test hero card ID to class mapping."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_cards_data
        
        loader = CardsJsonLoader()
        
        # Verify hero to class mapping
        self.assertEqual(loader.get_class_from_hero_card_id("HERO_01"), "WARRIOR")
        self.assertEqual(loader.get_class_from_hero_card_id("HERO_02"), "MAGE")
        
        # Test non-existent hero
        self.assertIsNone(loader.get_class_from_hero_card_id("HERO_FAKE"))
        
        # Test non-hero card (should return None)
        self.assertIsNone(loader.get_class_from_hero_card_id("TOY_380"))
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_mapping_statistics(self, mock_json_load, mock_exists, mock_file):
        """Test mapping statistics functionality."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_cards_data
        
        loader = CardsJsonLoader()
        stats = loader.get_id_mapping_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_cards_loaded'], 4)
        self.assertEqual(stats['dbf_id_mappings'], 4)  # All cards have dbf_id
        self.assertEqual(stats['hero_class_mappings'], 2)  # Only 2 heroes
        self.assertEqual(stats['dbf_mapping_rate'], 100.0)  # 4/4 = 100%
        self.assertEqual(stats['mapping_status'], 'ready')
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_missing_dbf_id_handling(self, mock_json_load, mock_exists, mock_file):
        """Test handling of cards without dbf_id."""
        # Card data with missing dbf_id
        incomplete_data = [
            {
                "id": "VALID_CARD",
                "name": "Valid Card",
                "dbfId": 12345,
                "cardClass": "NEUTRAL",
                "type": "MINION"
            },
            {
                "id": "MISSING_DBF_CARD", 
                "name": "Missing DBF Card",
                # No dbfId field
                "cardClass": "NEUTRAL",
                "type": "MINION"
            }
        ]
        
        mock_exists.return_value = True
        mock_json_load.return_value = incomplete_data
        
        loader = CardsJsonLoader()
        
        # Valid card should work
        self.assertEqual(loader.get_card_id_from_dbf_id(12345), "VALID_CARD")
        self.assertEqual(loader.get_dbf_id_from_card_id("VALID_CARD"), 12345)
        
        # Card without dbf_id should not be in mapping
        self.assertIsNone(loader.get_dbf_id_from_card_id("MISSING_DBF_CARD"))
        
        # Statistics should reflect partial mapping
        stats = loader.get_id_mapping_statistics()
        self.assertEqual(stats['total_cards_loaded'], 2)
        self.assertEqual(stats['dbf_id_mappings'], 1)  # Only 1 valid mapping
        self.assertEqual(stats['dbf_mapping_rate'], 50.0)  # 1/2 = 50%
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_duplicate_dbf_id_handling(self, mock_json_load, mock_exists, mock_file):
        """Test handling of duplicate dbf_ids (should use last occurrence)."""
        duplicate_data = [
            {
                "id": "CARD_A",
                "name": "Card A",
                "dbfId": 1001,
                "cardClass": "NEUTRAL",
                "type": "MINION"
            },
            {
                "id": "CARD_B",
                "name": "Card B", 
                "dbfId": 1001,  # Duplicate dbf_id
                "cardClass": "NEUTRAL",
                "type": "MINION"
            }
        ]
        
        mock_exists.return_value = True
        mock_json_load.return_value = duplicate_data
        
        loader = CardsJsonLoader()
        
        # Should use the last occurrence (CARD_B)
        self.assertEqual(loader.get_card_id_from_dbf_id(1001), "CARD_B")
        
        # Both cards should have valid reverse mappings
        self.assertEqual(loader.get_dbf_id_from_card_id("CARD_A"), 1001)
        self.assertEqual(loader.get_dbf_id_from_card_id("CARD_B"), 1001)
        
    @patch('arena_bot.data.cards_json_loader.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_hsreplay_integration_scenario(self, mock_json_load, mock_exists, mock_file):
        """Test a realistic HSReplay integration scenario."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_cards_data
        
        loader = CardsJsonLoader()
        
        # Simulate HSReplay API returning dbf_ids
        hsreplay_card_data = [
            {"dbf_id": 82615, "win_rate": 58.7},
            {"dbf_id": 576, "win_rate": 52.3},
            {"dbf_id": 999999, "win_rate": 45.0}  # Unknown card
        ]
        
        # Process HSReplay data and translate to card_ids
        translated_data = []
        for card_data in hsreplay_card_data:
            card_id = loader.get_card_id_from_dbf_id(card_data["dbf_id"])
            if card_id:
                translated_data.append({
                    "card_id": card_id,
                    "win_rate": card_data["win_rate"]
                })
        
        # Verify successful translation
        self.assertEqual(len(translated_data), 2)  # Only 2 known cards
        self.assertEqual(translated_data[0]["card_id"], "TOY_380")
        self.assertEqual(translated_data[1]["card_id"], "EX1_339")
        
        # Verify hero selection scenario
        hero_dbf_ids = [7, 637]  # WARRIOR, MAGE
        hero_classes = []
        for dbf_id in hero_dbf_ids:
            card_id = loader.get_card_id_from_dbf_id(dbf_id)
            if card_id:
                hero_class = loader.get_class_from_hero_card_id(card_id)
                if hero_class:
                    hero_classes.append(hero_class)
        
        self.assertEqual(hero_classes, ["WARRIOR", "MAGE"])


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run the tests
    unittest.main()