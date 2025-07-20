"""
Performance Tests for Dual-API Workflow (Heroes + Cards)

Comprehensive performance tests for the complete dual-API workflow,
testing response times, throughput, and resource usage within time targets.
"""

import unittest
import sys
import time
import threading
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.data_sourcing.hsreplay_scraper import HSReplayDataScraper
from arena_bot.data.cards_json_loader import CardsJsonLoader
from arena_bot.ai_v2.data_models import DeckState


class TestPerformanceDualAPI(unittest.TestCase):
    """Performance tests for dual-API workflow."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        # Performance targets (in seconds)
        self.HERO_SELECTION_TARGET = 2.0  # Hero selection should complete within 2 seconds
        self.CARD_EVALUATION_TARGET = 1.0  # Card evaluation should complete within 1 second
        self.COMPLETE_WORKFLOW_TARGET = 5.0  # Complete workflow within 5 seconds
        self.CONCURRENT_DEGRADATION_LIMIT = 2.0  # Performance should not degrade more than 2x under load
        
        # Mock data for consistent testing
        self.mock_cards_data = self._create_performance_cards_data()
        self.mock_hero_winrates = self._create_performance_hero_data()
        self.mock_hsreplay_card_data = self._create_performance_hsreplay_data()
        
        # Test deck state
        self.test_deck_state = DeckState(
            cards_drafted=[],
            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
            archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
            synergy_groups={},
            hero_class="WARRIOR"
        )
    
    def _create_performance_cards_data(self):
        """Create cards data optimized for performance testing."""
        cards_data = {}
        
        # Create hero cards
        hero_classes = ["WARRIOR", "MAGE", "HUNTER", "PALADIN", "PRIEST", 
                       "ROGUE", "SHAMAN", "WARLOCK", "DRUID", "DEMONHUNTER"]
        for i, hero_class in enumerate(hero_classes, 1):
            cards_data[f"HERO_{i:02d}"] = {
                "id": f"HERO_{i:02d}",
                "name": f"{hero_class.title()} Hero",
                "playerClass": hero_class,
                "type": "HERO",
                "dbfId": 800 + i
            }
        
        # Create test cards for evaluation
        for i in range(1, 101):  # 100 test cards
            cards_data[f"TEST_{i:03d}"] = {
                "id": f"TEST_{i:03d}",
                "name": f"Test Card {i}",
                "playerClass": "NEUTRAL" if i % 10 == 0 else hero_classes[i % len(hero_classes)],
                "type": "MINION" if i % 2 == 0 else "SPELL",
                "cost": (i % 7) + 1,
                "dbfId": 1000 + i
            }
            
            # Add minion stats for minions
            if cards_data[f"TEST_{i:03d}"]["type"] == "MINION":
                cards_data[f"TEST_{i:03d}"]["attack"] = (i % 5) + 1
                cards_data[f"TEST_{i:03d}"]["health"] = (i % 6) + 1
        
        return cards_data
    
    def _create_performance_hero_data(self):
        """Create hero winrate data for performance testing."""
        hero_classes = ["WARRIOR", "MAGE", "HUNTER", "PALADIN", "PRIEST", 
                       "ROGUE", "SHAMAN", "WARLOCK", "DRUID", "DEMONHUNTER"]
        
        return {hero_class: 0.45 + (i * 0.01) for i, hero_class in enumerate(hero_classes)}
    
    def _create_performance_hsreplay_data(self):
        """Create HSReplay card data for performance testing."""
        hsreplay_data = {}
        
        for i in range(1, 101):
            card_id = f"TEST_{i:03d}"
            hsreplay_data[card_id] = {
                "overall_winrate": 0.45 + (i % 20) * 0.005,
                "play_rate": 0.10 + (i % 15) * 0.01,
                "pick_rate": 0.05 + (i % 25) * 0.002
            }
        
        return hsreplay_data
    
    def test_hero_selection_performance(self):
        """Test hero selection performance within time targets."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Test single hero selection
            start_time = time.time()
            
            hero_advisor = HeroSelectionAdvisor()
            hero_classes = ["WARRIOR", "MAGE", "HUNTER"]
            recommendations = hero_advisor.recommend_hero(hero_classes)
            
            elapsed_time = time.time() - start_time
            
            # Verify performance target
            self.assertLess(elapsed_time, self.HERO_SELECTION_TARGET,
                          f"Hero selection took {elapsed_time:.3f}s, target is {self.HERO_SELECTION_TARGET}s")
            
            # Verify functionality
            self.assertEqual(len(recommendations), 3)
            self.assertTrue(all('hero_class' in rec for rec in recommendations))
    
    def test_card_evaluation_performance(self):
        """Test card evaluation performance within time targets."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_card_data
            
            # Test single card evaluation
            start_time = time.time()
            
            card_evaluator = CardEvaluationEngine()
            test_card = self.mock_cards_data["TEST_001"]
            evaluation = card_evaluator.evaluate_card(test_card, self.test_deck_state, "WARRIOR")
            
            elapsed_time = time.time() - start_time
            
            # Verify performance target
            self.assertLess(elapsed_time, self.CARD_EVALUATION_TARGET,
                          f"Card evaluation took {elapsed_time:.3f}s, target is {self.CARD_EVALUATION_TARGET}s")
            
            # Verify functionality
            self.assertIsNotNone(evaluation)
            self.assertTrue(hasattr(evaluation, 'total_score'))
    
    def test_complete_workflow_performance(self):
        """Test complete dual-API workflow performance."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_card_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Test complete workflow
            start_time = time.time()
            
            # Step 1: Hero selection
            hero_advisor = HeroSelectionAdvisor()
            hero_classes = ["WARRIOR", "MAGE", "HUNTER"]
            hero_recommendations = hero_advisor.recommend_hero(hero_classes)
            
            # Step 2: Card evaluation with selected hero
            selected_hero = hero_recommendations[0]['hero_class']
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Simulate card choices
            card_choices = [
                self.mock_cards_data["TEST_001"],
                self.mock_cards_data["TEST_002"],
                self.mock_cards_data["TEST_003"]
            ]
            
            # Update deck state with selected hero
            workflow_deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class=selected_hero
            )
            
            card_recommendation = grandmaster_advisor.get_recommendation(
                card_choices, workflow_deck_state, selected_hero
            )
            
            elapsed_time = time.time() - start_time
            
            # Verify performance target
            self.assertLess(elapsed_time, self.COMPLETE_WORKFLOW_TARGET,
                          f"Complete workflow took {elapsed_time:.3f}s, target is {self.COMPLETE_WORKFLOW_TARGET}s")
            
            # Verify functionality
            self.assertEqual(len(hero_recommendations), 3)
            self.assertIsNotNone(card_recommendation)
            self.assertIn('recommended_pick_index', card_recommendation.decision_data)
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_card_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Test single-threaded baseline
            def single_workflow():
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                grandmaster_advisor = GrandmasterAdvisor()
                card_choices = [
                    self.mock_cards_data["TEST_001"],
                    self.mock_cards_data["TEST_002"],
                    self.mock_cards_data["TEST_003"]
                ]
                
                card_recommendation = grandmaster_advisor.get_recommendation(
                    card_choices, self.test_deck_state, "WARRIOR"
                )
                
                return hero_recommendations, card_recommendation
            
            # Measure single-threaded performance
            start_time = time.time()
            single_result = single_workflow()
            single_thread_time = time.time() - start_time
            
            # Test concurrent performance
            def concurrent_worker(worker_id):
                start_time = time.time()
                result = single_workflow()
                elapsed_time = time.time() - start_time
                return worker_id, elapsed_time, result
            
            # Run 5 concurrent workflows
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_worker, i) for i in range(5)]
                concurrent_results = [future.result() for future in as_completed(futures)]
            
            total_concurrent_time = time.time() - start_time
            
            # Analyze results
            individual_times = [result[1] for result in concurrent_results]
            avg_concurrent_time = statistics.mean(individual_times)
            max_concurrent_time = max(individual_times)
            
            # Performance should not degrade significantly under load
            degradation_factor = max_concurrent_time / single_thread_time
            self.assertLess(degradation_factor, self.CONCURRENT_DEGRADATION_LIMIT,
                          f"Performance degraded by {degradation_factor:.2f}x under concurrent load")
            
            # All concurrent operations should complete
            self.assertEqual(len(concurrent_results), 5)
            
            # Verify all results are valid
            for worker_id, elapsed_time, (hero_recs, card_rec) in concurrent_results:
                self.assertEqual(len(hero_recs), 3)
                self.assertIsNotNone(card_rec)
    
    def test_memory_performance(self):
        """Test memory usage during workflow execution."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_card_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Measure initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute multiple workflows
            for i in range(20):
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                grandmaster_advisor = GrandmasterAdvisor()
                card_choices = [
                    self.mock_cards_data[f"TEST_{(i*3+1):03d}"],
                    self.mock_cards_data[f"TEST_{(i*3+2):03d}"],
                    self.mock_cards_data[f"TEST_{(i*3+3):03d}"]
                ]
                
                card_recommendation = grandmaster_advisor.get_recommendation(
                    card_choices, self.test_deck_state, "WARRIOR"
                )
                
                # Verify results
                self.assertEqual(len(hero_recommendations), 3)
                self.assertIsNotNone(card_recommendation)
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for 20 workflows)
            self.assertLess(memory_increase, 100.0,
                          f"Memory usage increased by {memory_increase:.2f}MB, which exceeds limit")
    
    def test_api_response_time_simulation(self):
        """Test performance with simulated API response delays."""
        def slow_api_response(*args, **kwargs):
            """Simulate slow API response."""
            time.sleep(0.2)  # 200ms delay
            return self.mock_hero_winrates
        
        def slow_card_api_response(*args, **kwargs):
            """Simulate slow card API response."""
            time.sleep(0.3)  # 300ms delay
            return self.mock_hsreplay_card_data
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats', side_effect=slow_card_api_response), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates', side_effect=slow_api_response):
            
            mock_load_cards.return_value = self.mock_cards_data
            
            # Test with simulated delays
            start_time = time.time()
            
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            grandmaster_advisor = GrandmasterAdvisor()
            card_choices = [
                self.mock_cards_data["TEST_001"],
                self.mock_cards_data["TEST_002"],
                self.mock_cards_data["TEST_003"]
            ]
            
            card_recommendation = grandmaster_advisor.get_recommendation(
                card_choices, self.test_deck_state, "WARRIOR"
            )
            
            elapsed_time = time.time() - start_time
            
            # Should still complete within reasonable time despite API delays
            # API delays are 200ms + 300ms = 500ms, plus processing time
            max_expected_time = 2.0  # Should complete within 2 seconds
            self.assertLess(elapsed_time, max_expected_time,
                          f"Workflow with API delays took {elapsed_time:.3f}s, expected under {max_expected_time}s")
            
            # Verify functionality still works
            self.assertEqual(len(hero_recommendations), 3)
            self.assertIsNotNone(card_recommendation)
    
    def test_throughput_performance(self):
        """Test throughput performance (requests per second)."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.mock_cards_data
            mock_hsreplay.return_value = self.mock_hsreplay_card_data
            mock_hero_winrates.return_value = self.mock_hero_winrates
            
            # Initialize components once
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Measure throughput over 10 seconds
            test_duration = 10.0  # seconds
            start_time = time.time()
            operations_completed = 0
            
            while (time.time() - start_time) < test_duration:
                # Perform one complete operation
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                card_choices = [
                    self.mock_cards_data["TEST_001"],
                    self.mock_cards_data["TEST_002"],
                    self.mock_cards_data["TEST_003"]
                ]
                
                card_recommendation = grandmaster_advisor.get_recommendation(
                    card_choices, self.test_deck_state, "WARRIOR"
                )
                
                operations_completed += 1
                
                # Verify each operation
                self.assertEqual(len(hero_recommendations), 3)
                self.assertIsNotNone(card_recommendation)
            
            actual_duration = time.time() - start_time
            throughput = operations_completed / actual_duration
            
            # Should achieve reasonable throughput (at least 5 operations per second)
            min_throughput = 5.0
            self.assertGreater(throughput, min_throughput,
                             f"Throughput was {throughput:.2f} ops/sec, minimum target is {min_throughput} ops/sec")
    
    def test_scalability_with_data_size(self):
        """Test performance scalability with increasing data size."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            # Test with different data sizes
            data_sizes = [100, 500, 1000]  # Number of cards
            performance_results = []
            
            for data_size in data_sizes:
                # Create data set of specified size
                large_cards_data = {}
                large_hsreplay_data = {}
                
                for i in range(1, data_size + 1):
                    card_id = f"SCALE_{i:04d}"
                    large_cards_data[card_id] = {
                        "id": card_id,
                        "name": f"Scale Card {i}",
                        "playerClass": "NEUTRAL",
                        "type": "MINION",
                        "cost": (i % 7) + 1,
                        "dbfId": 2000 + i
                    }
                    
                    large_hsreplay_data[card_id] = {
                        "overall_winrate": 0.45 + (i % 20) * 0.005,
                        "play_rate": 0.10 + (i % 15) * 0.01
                    }
                
                # Add base cards
                large_cards_data.update(self.mock_cards_data)
                large_hsreplay_data.update(self.mock_hsreplay_card_data)
                
                mock_load_cards.return_value = large_cards_data
                mock_hsreplay.return_value = large_hsreplay_data
                mock_hero_winrates.return_value = self.mock_hero_winrates
                
                # Measure performance
                start_time = time.time()
                
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                grandmaster_advisor = GrandmasterAdvisor()
                card_choices = [
                    large_cards_data["TEST_001"],
                    large_cards_data["TEST_002"],
                    large_cards_data["TEST_003"]
                ]
                
                card_recommendation = grandmaster_advisor.get_recommendation(
                    card_choices, self.test_deck_state, "WARRIOR"
                )
                
                elapsed_time = time.time() - start_time
                performance_results.append((data_size, elapsed_time))
                
                # Verify functionality
                self.assertEqual(len(hero_recommendations), 3)
                self.assertIsNotNone(card_recommendation)
            
            # Analyze scalability
            # Performance should scale sub-linearly (not proportionally to data size)
            small_size, small_time = performance_results[0]
            large_size, large_time = performance_results[-1]
            
            size_ratio = large_size / small_size
            time_ratio = large_time / small_time
            
            # Time increase should be much less than data size increase
            self.assertLess(time_ratio, size_ratio / 2,
                          f"Performance does not scale well: {time_ratio:.2f}x time for {size_ratio:.2f}x data")


class TestPerformanceDualAPIEdgeCases(unittest.TestCase):
    """Test performance edge cases and error conditions."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.minimal_data = {
            "HERO_01": {
                "id": "HERO_01",
                "name": "Test Hero",
                "playerClass": "WARRIOR",
                "type": "HERO",
                "dbfId": 999
            },
            "TEST_001": {
                "id": "TEST_001",
                "name": "Test Card",
                "playerClass": "NEUTRAL",
                "type": "MINION",
                "cost": 3,
                "dbfId": 1000
            }
        }
    
    def test_performance_with_missing_data(self):
        """Test performance when API data is missing."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.minimal_data
            mock_hsreplay.return_value = {}  # No HSReplay data
            mock_hero_winrates.return_value = {}  # No hero data
            
            # Should still perform reasonably with missing data
            start_time = time.time()
            
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            grandmaster_advisor = GrandmasterAdvisor()
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            card_choices = [self.minimal_data["TEST_001"]] * 3  # Same card 3 times
            card_recommendation = grandmaster_advisor.get_recommendation(
                card_choices, deck_state, "WARRIOR"
            )
            
            elapsed_time = time.time() - start_time
            
            # Should complete quickly even with missing data (fallback should be fast)
            self.assertLess(elapsed_time, 3.0, "Fallback performance should be fast")
            
            # Should still provide results
            self.assertEqual(len(hero_recommendations), 3)
            self.assertIsNotNone(card_recommendation)
    
    def test_performance_with_api_timeouts(self):
        """Test performance when APIs timeout."""
        def timeout_api_call(*args, **kwargs):
            """Simulate API timeout."""
            time.sleep(5.0)  # Long delay to simulate timeout
            raise TimeoutError("API request timed out")
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats', side_effect=timeout_api_call), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates', side_effect=timeout_api_call):
            
            mock_load_cards.return_value = self.minimal_data
            
            # Should handle timeouts gracefully and not hang
            start_time = time.time()
            
            try:
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                grandmaster_advisor = GrandmasterAdvisor()
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class="WARRIOR"
                )
                
                card_choices = [self.minimal_data["TEST_001"]] * 3
                card_recommendation = grandmaster_advisor.get_recommendation(
                    card_choices, deck_state, "WARRIOR"
                )
                
                elapsed_time = time.time() - start_time
                
                # Should not take too long even with timeouts (should use fallback)
                self.assertLess(elapsed_time, 8.0, "Should handle timeouts quickly with fallback")
                
                # Should still provide results
                self.assertEqual(len(hero_recommendations), 3)
                self.assertIsNotNone(card_recommendation)
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                # If it fails, it should fail quickly
                self.assertLess(elapsed_time, 8.0, "Even failures should be fast")


def run_performance_dual_api_tests():
    """Run all dual-API performance tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestPerformanceDualAPI))
    test_suite.addTest(unittest.makeSuite(TestPerformanceDualAPIEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_performance_dual_api_tests()
    
    if success:
        print("\n✅ All dual-API performance tests passed!")
    else:
        print("\n❌ Some dual-API performance tests failed!")
        sys.exit(1)