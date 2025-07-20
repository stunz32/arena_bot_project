"""
Integration Tests for Hero Selection Workflow

Comprehensive integration tests for the complete hero selection workflow,
testing the end-to-end process from log detection to GUI display.
"""

import unittest
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import json
import queue
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.hearthstone_log_monitor import HearthstoneLogMonitor
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.gui.integrated_arena_bot_gui import IntegratedArenaBotGUI
from arena_bot.data.cards_json_loader import CardsJsonLoader


class TestHeroSelectionWorkflow(unittest.TestCase):
    """Integration tests for complete hero selection workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock cards data for consistent testing
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
            }
        }
        
        # Mock hero winrate data
        self.mock_hero_winrates = {
            "WARRIOR": 0.5507,
            "MAGE": 0.5234,
            "HUNTER": 0.4892
        }
        
        # Event queue for testing
        self.event_queue = queue.Queue()
        
        # Mock components
        self.mock_cards_loader = None
        self.mock_hero_advisor = None
        self.mock_gui = None
        self.mock_log_monitor = None
    
    def test_complete_hero_selection_workflow(self):
        """Test complete workflow from log detection to GUI display."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates, \
             patch('tkinter.Tk') as mock_tk:
            
            # Setup mock data
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Step 1: Initialize components
            cards_loader = CardsJsonLoader()
            hero_advisor = HeroSelectionAdvisor()
            
            # Step 2: Simulate log detection
            log_monitor = HearthstoneLogMonitor()
            
            # Mock log parsing result
            mock_log_entries = [
                "D 12:34:56.789 GameState.DebugPrintPower() - BLOCK_START BlockType=TRIGGER Entity=[name=DraftManager] EffectId=GameUtils.CardDrawn",
                "D 12:34:57.123 GameState.DebugPrintPower() - TAG_CHANGE Entity=HERO_01 tag=ZONE value=HAND",
                "D 12:34:57.456 GameState.DebugPrintPower() - TAG_CHANGE Entity=HERO_02 tag=ZONE value=HAND", 
                "D 12:34:57.789 GameState.DebugPrintPower() - TAG_CHANGE Entity=HERO_03 tag=ZONE value=HAND",
                "D 12:34:58.012 GameState.DebugPrintPower() - BLOCK_END"
            ]
            
            # Process log entries
            detected_heroes = []
            for log_entry in mock_log_entries:
                result = log_monitor._parse_hero_choices(log_entry)
                if result:
                    detected_heroes.extend(result)
            
            # Should detect all three heroes
            self.assertEqual(len(detected_heroes), 3)
            self.assertIn("HERO_01", detected_heroes)
            self.assertIn("HERO_02", detected_heroes)
            self.assertIn("HERO_03", detected_heroes)
            
            # Step 3: Translate hero IDs to class names
            hero_classes = []
            for hero_id in detected_heroes:
                class_name = cards_loader.get_class_from_hero_card_id(hero_id)
                if class_name:
                    hero_classes.append(class_name)
            
            self.assertEqual(len(hero_classes), 3)
            self.assertIn("WARRIOR", hero_classes)
            self.assertIn("MAGE", hero_classes)
            self.assertIn("HUNTER", hero_classes)
            
            # Step 4: Get hero recommendations
            recommendations = hero_advisor.recommend_hero(hero_classes)
            
            # Verify recommendations structure
            self.assertIsInstance(recommendations, list)
            self.assertEqual(len(recommendations), 3)
            
            for rec in recommendations:
                self.assertIn('hero_class', rec)
                self.assertIn('winrate', rec)
                self.assertIn('confidence', rec)
                self.assertIn('explanation', rec)
                self.assertIn('rank', rec)
            
            # Step 5: Verify ranking (WARRIOR should be top due to highest winrate)
            recommendations.sort(key=lambda x: x['rank'])
            self.assertEqual(recommendations[0]['hero_class'], "WARRIOR")
            self.assertEqual(recommendations[1]['hero_class'], "MAGE")
            self.assertEqual(recommendations[2]['hero_class'], "HUNTER")
    
    def test_hero_detection_edge_cases(self):
        """Test hero detection with edge cases and malformed data."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards:
            mock_load_cards.return_value = self.mock_cards_data
            
            log_monitor = HearthstoneLogMonitor()
            cards_loader = CardsJsonLoader()
            
            # Test with malformed log entries
            malformed_log_entries = [
                "",  # Empty string
                "Invalid log entry",  # No pattern match
                "D 12:34:56.789 GameState.DebugPrintPower() - TAG_CHANGE Entity=INVALID_HERO tag=ZONE value=HAND",  # Invalid hero
                "D 12:34:56.789 GameState.DebugPrintPower() - TAG_CHANGE Entity=HERO_01 tag=ZONE value=DECK",  # Wrong zone
                "D 12:34:56.789 GameState.DebugPrintPower() - TAG_CHANGE Entity=HERO_01 tag=ZONE value=HAND"  # Valid entry
            ]
            
            detected_heroes = []
            for log_entry in malformed_log_entries:
                result = log_monitor._parse_hero_choices(log_entry)
                if result:
                    detected_heroes.extend(result)
            
            # Should only detect the valid hero
            self.assertEqual(len(detected_heroes), 1)
            self.assertEqual(detected_heroes[0], "HERO_01")
            
            # Verify class translation works
            class_name = cards_loader.get_class_from_hero_card_id(detected_heroes[0])
            self.assertEqual(class_name, "WARRIOR")
    
    def test_workflow_with_missing_hero_data(self):
        """Test workflow when hero winrate data is unavailable."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = {}  # No hero data available
            
            hero_advisor = HeroSelectionAdvisor()
            hero_classes = ["WARRIOR", "MAGE", "HUNTER"]
            
            recommendations = hero_advisor.recommend_hero(hero_classes)
            
            # Should still provide recommendations using fallback logic
            self.assertEqual(len(recommendations), 3)
            
            for rec in recommendations:
                self.assertIn('hero_class', rec)
                self.assertIn('explanation', rec)
                # Should have fallback winrate and lower confidence
                self.assertLess(rec['confidence'], 0.8)
    
    def test_workflow_timing_and_performance(self):
        """Test workflow timing and performance characteristics."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Measure initialization time
            start_time = time.time()
            cards_loader = CardsJsonLoader()
            hero_advisor = HeroSelectionAdvisor()
            init_time = time.time() - start_time
            
            # Initialization should be fast (under 2 seconds)
            self.assertLess(init_time, 2.0)
            
            # Measure hero recommendation time
            hero_classes = ["WARRIOR", "MAGE", "HUNTER", "PALADIN", "PRIEST", 
                          "ROGUE", "SHAMAN", "WARLOCK", "DRUID", "DEMONHUNTER"]
            
            start_time = time.time()
            recommendations = hero_advisor.recommend_hero(hero_classes[:3])
            recommendation_time = time.time() - start_time
            
            # Hero recommendation should be very fast (under 0.5 seconds)
            self.assertLess(recommendation_time, 0.5)
            
            # Verify recommendations are complete
            self.assertEqual(len(recommendations), 3)
    
    def test_concurrent_workflow_execution(self):
        """Test workflow under concurrent execution scenarios."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            results = []
            errors = []
            
            def worker_thread(thread_id):
                try:
                    hero_classes = ["WARRIOR", "MAGE", "HUNTER"]
                    recommendations = hero_advisor.recommend_hero(hero_classes)
                    results.append((thread_id, recommendations))
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all threads completed successfully
            self.assertEqual(len(errors), 0, f"Errors in concurrent execution: {errors}")
            self.assertEqual(len(results), 5)
            
            # Verify all results are consistent
            first_result = results[0][1]
            for thread_id, recommendations in results:
                self.assertEqual(len(recommendations), 3)
                # Results should be deterministic
                self.assertEqual(recommendations[0]['hero_class'], first_result[0]['hero_class'])
    
    def test_workflow_error_recovery(self):
        """Test workflow error recovery and fallback mechanisms."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards:
            mock_load_cards.return_value = self.mock_cards_data
            
            # Test with failing hero advisor
            with patch('arena_bot.ai_v2.hero_selector.HeroSelectionAdvisor.recommend_hero') as mock_recommend:
                mock_recommend.side_effect = Exception("API failure")
                
                try:
                    hero_advisor = HeroSelectionAdvisor()
                    recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    
                    # Should handle error gracefully
                    self.fail("Expected exception was not raised")
                except Exception as e:
                    # Exception should be handled by error recovery system
                    self.assertIn("API failure", str(e))
    
    def test_gui_integration_workflow(self):
        """Test integration with GUI components."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates, \
             patch('tkinter.Tk') as mock_tk:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Mock GUI components
            mock_root = MagicMock()
            mock_tk.return_value = mock_root
            
            # Test event creation and processing
            event_data = {
                'event': 'HERO_CHOICES_READY',
                'hero_classes': ['WARRIOR', 'MAGE', 'HUNTER'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Simulate GUI processing of hero selection event
            gui = IntegratedArenaBotGUI()
            
            # Mock the hero selection display method
            with patch.object(gui, '_display_hero_selection_ui') as mock_display:
                gui._handle_hero_choices_ready(event_data)
                
                # Verify GUI method was called with correct data
                mock_display.assert_called_once()
                call_args = mock_display.call_args[0]
                self.assertEqual(call_args[0], ['WARRIOR', 'MAGE', 'HUNTER'])
    
    def test_data_flow_integrity(self):
        """Test data integrity throughout the workflow."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Step 1: Raw hero IDs from log
            raw_hero_ids = ["HERO_01", "HERO_02", "HERO_03"]
            
            # Step 2: Translate to class names
            cards_loader = CardsJsonLoader()
            hero_classes = []
            for hero_id in raw_hero_ids:
                class_name = cards_loader.get_class_from_hero_card_id(hero_id)
                if class_name:
                    hero_classes.append(class_name)
            
            # Verify translation integrity
            self.assertEqual(len(hero_classes), 3)
            expected_classes = ["WARRIOR", "MAGE", "HUNTER"]
            for expected_class in expected_classes:
                self.assertIn(expected_class, hero_classes)
            
            # Step 3: Get recommendations
            hero_advisor = HeroSelectionAdvisor()
            recommendations = hero_advisor.recommend_hero(hero_classes)
            
            # Verify recommendation integrity
            self.assertEqual(len(recommendations), 3)
            
            # Each original class should have a recommendation
            recommended_classes = [rec['hero_class'] for rec in recommendations]
            for original_class in hero_classes:
                self.assertIn(original_class, recommended_classes)
            
            # Verify winrate data integrity
            for rec in recommendations:
                if rec['hero_class'] in self.mock_hero_winrates:
                    expected_winrate = self.mock_hero_winrates[rec['hero_class']]
                    self.assertAlmostEqual(rec['winrate'], expected_winrate, places=4)
    
    def test_memory_usage_during_workflow(self):
        """Test memory usage characteristics during workflow execution."""
        import psutil
        import os
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Measure initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute workflow multiple times
            for i in range(10):
                cards_loader = CardsJsonLoader()
                hero_advisor = HeroSelectionAdvisor()
                recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                # Verify each iteration produces valid results
                self.assertEqual(len(recommendations), 3)
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 50MB for 10 iterations)
            self.assertLess(memory_increase, 50.0, 
                          f"Memory usage increased by {memory_increase:.2f}MB")


class TestHeroSelectionWorkflowIntegration(unittest.TestCase):
    """Advanced integration tests for hero selection workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create more comprehensive mock data
        self.comprehensive_cards_data = self._create_comprehensive_cards_data()
        self.comprehensive_hero_winrates = self._create_comprehensive_hero_winrates()
    
    def _create_comprehensive_cards_data(self):
        """Create comprehensive cards data for testing."""
        return {
            "HERO_01": {"id": "HERO_01", "name": "Garrosh Hellscream", "playerClass": "WARRIOR", "type": "HERO", "dbfId": 813},
            "HERO_02": {"id": "HERO_02", "name": "Jaina Proudmoore", "playerClass": "MAGE", "type": "HERO", "dbfId": 637},
            "HERO_03": {"id": "HERO_03", "name": "Rexxar", "playerClass": "HUNTER", "type": "HERO", "dbfId": 31},
            "HERO_04": {"id": "HERO_04", "name": "Uther Lightbringer", "playerClass": "PALADIN", "type": "HERO", "dbfId": 671},
            "HERO_05": {"id": "HERO_05", "name": "Anduin Wrynn", "playerClass": "PRIEST", "type": "HERO", "dbfId": 813},
            "HERO_06": {"id": "HERO_06", "name": "Valeera Sanguinar", "playerClass": "ROGUE", "type": "HERO", "dbfId": 930},
            "HERO_07": {"id": "HERO_07", "name": "Thrall", "playerClass": "SHAMAN", "type": "HERO", "dbfId": 1066},
            "HERO_08": {"id": "HERO_08", "name": "Gul'dan", "playerClass": "WARLOCK", "type": "HERO", "dbfId": 893},
            "HERO_09": {"id": "HERO_09", "name": "Malfurion Stormrage", "playerClass": "DRUID", "type": "HERO", "dbfId": 274},
            "HERO_10": {"id": "HERO_10", "name": "Illidan Stormrage", "playerClass": "DEMONHUNTER", "type": "HERO", "dbfId": 56550}
        }
    
    def _create_comprehensive_hero_winrates(self):
        """Create comprehensive hero winrates for testing."""
        return {
            "WARRIOR": 0.5507,
            "MAGE": 0.5234,
            "HUNTER": 0.4892,
            "PALADIN": 0.5123,
            "PRIEST": 0.4967,
            "ROGUE": 0.5089,
            "SHAMAN": 0.4834,
            "WARLOCK": 0.5345,
            "DRUID": 0.4923,
            "DEMONHUNTER": 0.5012
        }
    
    def test_full_system_integration(self):
        """Test complete system integration with all components."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates, \
             patch('tkinter.Tk') as mock_tk:
            
            mock_load_cards.return_value = self.comprehensive_cards_data
            mock_hero_winrates.return_value = self.comprehensive_hero_winrates
            
            # Initialize all system components
            cards_loader = CardsJsonLoader()
            hero_advisor = HeroSelectionAdvisor()
            log_monitor = HearthstoneLogMonitor()
            
            # Simulate complete draft session
            draft_sessions = [
                {
                    'hero_choices': ['HERO_01', 'HERO_02', 'HERO_03'],
                    'expected_classes': ['WARRIOR', 'MAGE', 'HUNTER']
                },
                {
                    'hero_choices': ['HERO_04', 'HERO_05', 'HERO_06'],
                    'expected_classes': ['PALADIN', 'PRIEST', 'ROGUE']
                },
                {
                    'hero_choices': ['HERO_07', 'HERO_08', 'HERO_09'],
                    'expected_classes': ['SHAMAN', 'WARLOCK', 'DRUID']
                }
            ]
            
            for session in draft_sessions:
                # Step 1: Hero detection
                hero_classes = []
                for hero_id in session['hero_choices']:
                    class_name = cards_loader.get_class_from_hero_card_id(hero_id)
                    if class_name:
                        hero_classes.append(class_name)
                
                # Verify detection
                self.assertEqual(set(hero_classes), set(session['expected_classes']))
                
                # Step 2: Get recommendations
                recommendations = hero_advisor.recommend_hero(hero_classes)
                
                # Verify recommendations
                self.assertEqual(len(recommendations), 3)
                
                # Verify all classes are represented
                recommended_classes = [rec['hero_class'] for rec in recommendations]
                self.assertEqual(set(recommended_classes), set(session['expected_classes']))
                
                # Verify ranking is based on winrates
                sorted_recommendations = sorted(recommendations, key=lambda x: x['winrate'], reverse=True)
                expected_order = sorted(session['expected_classes'], 
                                      key=lambda c: self.comprehensive_hero_winrates[c], reverse=True)
                
                actual_order = [rec['hero_class'] for rec in sorted_recommendations]
                self.assertEqual(actual_order, expected_order)
    
    def test_workflow_with_system_stress(self):
        """Test workflow under system stress conditions."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.comprehensive_cards_data
            mock_hero_winrates.return_value = self.comprehensive_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            
            # Test rapid successive calls
            start_time = time.time()
            for i in range(100):
                hero_classes = ["WARRIOR", "MAGE", "HUNTER"]
                recommendations = hero_advisor.recommend_hero(hero_classes)
                self.assertEqual(len(recommendations), 3)
            
            elapsed_time = time.time() - start_time
            
            # Should handle 100 calls in reasonable time (under 10 seconds)
            self.assertLess(elapsed_time, 10.0)
    
    def test_workflow_data_consistency(self):
        """Test data consistency across multiple workflow executions."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.comprehensive_cards_data
            mock_hero_winrates.return_value = self.comprehensive_hero_winrates
            
            hero_advisor = HeroSelectionAdvisor()
            hero_classes = ["WARRIOR", "MAGE", "HUNTER"]
            
            # Execute workflow multiple times
            results = []
            for i in range(10):
                recommendations = hero_advisor.recommend_hero(hero_classes)
                results.append(recommendations)
            
            # Verify all results are identical (deterministic)
            first_result = results[0]
            for result in results[1:]:
                self.assertEqual(len(result), len(first_result))
                for i, rec in enumerate(result):
                    self.assertEqual(rec['hero_class'], first_result[i]['hero_class'])
                    self.assertEqual(rec['winrate'], first_result[i]['winrate'])
                    self.assertEqual(rec['rank'], first_result[i]['rank'])


def run_hero_selection_workflow_tests():
    """Run all hero selection workflow tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestHeroSelectionWorkflow))
    test_suite.addTest(unittest.makeSuite(TestHeroSelectionWorkflowIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_hero_selection_workflow_tests()
    
    if success:
        print("\n✅ All hero selection workflow tests passed!")
    else:
        print("\n❌ Some hero selection workflow tests failed!")
        sys.exit(1)