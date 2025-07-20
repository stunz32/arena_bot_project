"""
Network Resilience Testing with Partial API Availability

Comprehensive tests for network resilience, testing system behavior under
various network conditions and partial API availability scenarios.
"""

import unittest
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, side_effect
import json
from datetime import datetime
import requests
import socket
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.data_sourcing.hsreplay_scraper import HSReplayDataScraper
from arena_bot.data.cards_json_loader import CardsJsonLoader
from arena_bot.ai_v2.data_models import DeckState
from arena_bot.ai_v2.advanced_error_recovery import get_error_recovery


class NetworkFailureException(Exception):
    """Custom exception for network failures."""
    pass


class TestNetworkResilience(unittest.TestCase):
    """Network resilience tests for partial API availability."""
    
    def setUp(self):
        """Set up network resilience test fixtures."""
        # Network failure scenarios
        self.NETWORK_SCENARIOS = {
            'complete_failure': {
                'hero_api_available': False,
                'card_api_available': False,
                'description': 'Complete network failure'
            },
            'hero_api_only': {
                'hero_api_available': True,
                'card_api_available': False,
                'description': 'Only hero API available'
            },
            'card_api_only': {
                'hero_api_available': False,
                'card_api_available': True,
                'description': 'Only card API available'
            },
            'intermittent_failure': {
                'hero_api_available': 'intermittent',
                'card_api_available': 'intermittent',
                'description': 'Intermittent network failures'
            },
            'slow_responses': {
                'hero_api_available': 'slow',
                'card_api_available': 'slow',
                'description': 'Slow API responses'
            },
            'timeout_scenarios': {
                'hero_api_available': 'timeout',
                'card_api_available': 'timeout',
                'description': 'API timeout scenarios'
            }
        }
        
        # Mock data for fallback scenarios
        self.fallback_cards_data = self._create_fallback_cards_data()
        self.fallback_hero_winrates = self._create_fallback_hero_winrates()
        self.fallback_hsreplay_data = self._create_fallback_hsreplay_data()
        
        # Test deck state
        self.test_deck_state = DeckState(
            cards_drafted=[],
            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
            archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
            synergy_groups={},
            hero_class="WARRIOR"
        )
    
    def _create_fallback_cards_data(self):
        """Create fallback cards data for offline scenarios."""
        return {
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
            "TEST_001": {
                "id": "TEST_001",
                "name": "Test Card 1",
                "playerClass": "NEUTRAL",
                "type": "MINION",
                "cost": 3,
                "attack": 3,
                "health": 3,
                "dbfId": 1001
            },
            "TEST_002": {
                "id": "TEST_002",
                "name": "Test Card 2",
                "playerClass": "WARRIOR",
                "type": "SPELL",
                "cost": 2,
                "dbfId": 1002
            },
            "TEST_003": {
                "id": "TEST_003",
                "name": "Test Card 3",
                "playerClass": "MAGE",
                "type": "SPELL",
                "cost": 4,
                "dbfId": 1003
            }
        }
    
    def _create_fallback_hero_winrates(self):
        """Create fallback hero winrates for offline scenarios."""
        return {
            "WARRIOR": 0.52,
            "MAGE": 0.51,
            "HUNTER": 0.50
        }
    
    def _create_fallback_hsreplay_data(self):
        """Create fallback HSReplay data for offline scenarios."""
        return {
            "TEST_001": {
                "overall_winrate": 0.51,
                "play_rate": 0.12
            },
            "TEST_002": {
                "overall_winrate": 0.53,
                "play_rate": 0.15
            },
            "TEST_003": {
                "overall_winrate": 0.49,
                "play_rate": 0.10
            }
        }
    
    def _create_network_failure_side_effect(self, scenario_config, api_type):
        """Create side effect function for simulating network failures."""
        def side_effect_function(*args, **kwargs):
            availability = scenario_config.get(f'{api_type}_api_available', True)
            
            if availability is False:
                # Complete failure
                raise requests.exceptions.ConnectionError("Network unreachable")
            elif availability == 'intermittent':
                # 50% chance of failure
                import random
                if random.random() < 0.5:
                    raise requests.exceptions.ConnectionError("Intermittent network failure")
                return self._get_fallback_data(api_type)
            elif availability == 'slow':
                # Slow response
                time.sleep(2.0)  # 2 second delay
                return self._get_fallback_data(api_type)
            elif availability == 'timeout':
                # Timeout
                raise requests.exceptions.Timeout("Request timed out")
            else:
                # Normal operation
                return self._get_fallback_data(api_type)
        
        return side_effect_function
    
    def _get_fallback_data(self, api_type):
        """Get fallback data for API type."""
        if api_type == 'hero':
            return self.fallback_hero_winrates
        elif api_type == 'card':
            return self.fallback_hsreplay_data
        else:
            return {}
    
    def test_complete_network_failure_resilience(self):
        """Test system resilience under complete network failure."""
        scenario = self.NETWORK_SCENARIOS['complete_failure']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats', 
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Test hero selection with complete network failure
            start_time = time.time()
            
            try:
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                elapsed_time = time.time() - start_time
                
                # Should complete within reasonable time even with network failure
                self.assertLess(elapsed_time, 10.0, "Should handle network failure gracefully")
                
                # Should still provide recommendations (using fallback)
                self.assertEqual(len(hero_recommendations), 3)
                for rec in hero_recommendations:
                    self.assertIn('hero_class', rec)
                    self.assertIn('confidence', rec)
                    # Confidence should be lower due to missing data
                    self.assertLess(rec['confidence'], 0.8)
                
            except Exception as e:
                # If it fails, it should fail gracefully
                self.assertIn("network", str(e).lower(), f"Unexpected error: {e}")
            
            # Test card evaluation with complete network failure
            try:
                grandmaster_advisor = GrandmasterAdvisor()
                card_choices = [
                    self.fallback_cards_data["TEST_001"],
                    self.fallback_cards_data["TEST_002"],
                    self.fallback_cards_data["TEST_003"]
                ]
                
                card_recommendation = grandmaster_advisor.get_recommendation(
                    card_choices, self.test_deck_state, "WARRIOR"
                )
                
                # Should still provide a recommendation
                self.assertIsNotNone(card_recommendation)
                self.assertIn('recommended_pick_index', card_recommendation.decision_data)
                
                # Confidence should be lower
                confidence = card_recommendation.decision_data.get('confidence_level', 1.0)
                self.assertLess(confidence, 0.8)
                
            except Exception as e:
                # Should handle gracefully
                self.assertIn("network", str(e).lower(), f"Unexpected card evaluation error: {e}")
    
    def test_partial_api_availability_hero_only(self):
        """Test system behavior when only hero API is available."""
        scenario = self.NETWORK_SCENARIOS['hero_api_only']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats',
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Hero selection should work well with hero API available
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            # Should get good hero recommendations
            self.assertEqual(len(hero_recommendations), 3)
            for rec in hero_recommendations:
                # Confidence should be higher since hero API is available
                self.assertGreater(rec['confidence'], 0.6)
            
            # Card evaluation should work but with lower confidence
            grandmaster_advisor = GrandmasterAdvisor()
            card_choices = [
                self.fallback_cards_data["TEST_001"],
                self.fallback_cards_data["TEST_002"],
                self.fallback_cards_data["TEST_003"]
            ]
            
            card_recommendation = grandmaster_advisor.get_recommendation(
                card_choices, self.test_deck_state, "WARRIOR"
            )
            
            # Should still provide card recommendation
            self.assertIsNotNone(card_recommendation)
            
            # But confidence should be lower due to missing card API data
            confidence = card_recommendation.decision_data.get('confidence_level', 1.0)
            self.assertLess(confidence, 0.7)
    
    def test_partial_api_availability_card_only(self):
        """Test system behavior when only card API is available."""
        scenario = self.NETWORK_SCENARIOS['card_api_only']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats',
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Hero selection should work but with lower confidence
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            # Should get hero recommendations
            self.assertEqual(len(hero_recommendations), 3)
            for rec in hero_recommendations:
                # Confidence should be lower since hero API is unavailable
                self.assertLess(rec['confidence'], 0.7)
            
            # Card evaluation should work well with card API available
            grandmaster_advisor = GrandmasterAdvisor()
            card_choices = [
                self.fallback_cards_data["TEST_001"],
                self.fallback_cards_data["TEST_002"],
                self.fallback_cards_data["TEST_003"]
            ]
            
            card_recommendation = grandmaster_advisor.get_recommendation(
                card_choices, self.test_deck_state, "WARRIOR"
            )
            
            # Should provide good card recommendation
            self.assertIsNotNone(card_recommendation)
            
            # Confidence should be reasonable since card API is available
            confidence = card_recommendation.decision_data.get('confidence_level', 1.0)
            self.assertGreater(confidence, 0.6)
    
    def test_intermittent_network_failures(self):
        """Test system behavior under intermittent network failures."""
        scenario = self.NETWORK_SCENARIOS['intermittent_failure']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats',
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Test multiple attempts to see resilience to intermittent failures
            success_count = 0
            total_attempts = 10
            
            for i in range(total_attempts):
                try:
                    hero_advisor = HeroSelectionAdvisor()
                    hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    
                    if hero_recommendations and len(hero_recommendations) == 3:
                        success_count += 1
                        
                except Exception:
                    # Some failures are expected with intermittent issues
                    pass
            
            # Should succeed at least 30% of the time with intermittent failures
            success_rate = success_count / total_attempts
            self.assertGreater(success_rate, 0.3, 
                             f"Success rate {success_rate:.2f} too low for intermittent failures")
    
    def test_slow_api_response_handling(self):
        """Test system behavior with slow API responses."""
        scenario = self.NETWORK_SCENARIOS['slow_responses']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats',
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Test with slow responses
            start_time = time.time()
            
            hero_advisor = HeroSelectionAdvisor()
            hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            elapsed_time = time.time() - start_time
            
            # Should still complete, but may take longer
            self.assertLess(elapsed_time, 15.0, "Should handle slow responses within reasonable time")
            
            # Should still provide recommendations
            self.assertEqual(len(hero_recommendations), 3)
    
    def test_api_timeout_handling(self):
        """Test system behavior when APIs timeout."""
        scenario = self.NETWORK_SCENARIOS['timeout_scenarios']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats',
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Test timeout handling
            start_time = time.time()
            
            try:
                hero_advisor = HeroSelectionAdvisor()
                hero_recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                
                elapsed_time = time.time() - start_time
                
                # Should handle timeouts quickly (fallback should be fast)
                self.assertLess(elapsed_time, 8.0, "Timeout handling should be fast")
                
                # Should still provide recommendations using fallback
                self.assertEqual(len(hero_recommendations), 3)
                
                # Confidence should be lower due to timeout
                for rec in hero_recommendations:
                    self.assertLess(rec['confidence'], 0.7)
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                # Even if it fails, should fail quickly
                self.assertLess(elapsed_time, 8.0, "Even timeout failures should be fast")
    
    def test_network_recovery_behavior(self):
        """Test system behavior when network recovers after failure."""
        # Simulate network recovery scenario
        call_count = [0]  # Use list to modify from inner function
        
        def recovery_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First two calls fail
                raise requests.exceptions.ConnectionError("Network down")
            else:
                # Subsequent calls succeed
                return self.fallback_hero_winrates
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=recovery_side_effect):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # First attempt should fail or use fallback
            hero_advisor1 = HeroSelectionAdvisor()
            try:
                recommendations1 = hero_advisor1.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                # If it succeeds, confidence should be low
                for rec in recommendations1:
                    self.assertLess(rec['confidence'], 0.8)
            except Exception:
                # Failure is acceptable on first attempt
                pass
            
            # Third attempt should succeed with better data
            hero_advisor2 = HeroSelectionAdvisor()
            recommendations2 = hero_advisor2.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            
            # Should get recommendations
            self.assertEqual(len(recommendations2), 3)
    
    def test_concurrent_network_failures(self):
        """Test system behavior under concurrent network failures."""
        scenario = self.NETWORK_SCENARIOS['intermittent_failure']
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats',
                   side_effect=self._create_network_failure_side_effect(scenario, 'card')), \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=self._create_network_failure_side_effect(scenario, 'hero')):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            results = []
            errors = []
            
            def concurrent_worker(worker_id):
                try:
                    hero_advisor = HeroSelectionAdvisor()
                    recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    return worker_id, recommendations
                except Exception as e:
                    errors.append((worker_id, e))
                    return worker_id, None
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_worker, i) for i in range(10)]
                results = [future.result() for future in futures]
            
            # Count successful operations
            successful_results = [r for r in results if r[1] is not None]
            
            # Should have some successful results even with network issues
            success_rate = len(successful_results) / len(results)
            self.assertGreater(success_rate, 0.2, 
                             f"Concurrent success rate {success_rate:.2f} too low")
    
    def test_network_error_propagation(self):
        """Test proper error propagation for network failures."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_api:
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Test different types of network errors
            network_errors = [
                requests.exceptions.ConnectionError("Connection refused"),
                requests.exceptions.Timeout("Request timeout"),
                requests.exceptions.HTTPError("HTTP 500 Error"),
                socket.gaierror("Name resolution failed"),
                OSError("Network unreachable")
            ]
            
            for error in network_errors:
                mock_hero_api.side_effect = error
                
                try:
                    hero_advisor = HeroSelectionAdvisor()
                    recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    
                    # If it succeeds, it should be using fallback
                    self.assertEqual(len(recommendations), 3)
                    for rec in recommendations:
                        self.assertLess(rec['confidence'], 0.8, 
                                      f"Confidence should be low for error: {type(error).__name__}")
                        
                except Exception as e:
                    # Should be a meaningful error message
                    self.assertIn("network", str(e).lower(), 
                                f"Error should mention network issue: {e}")
    
    def test_circuit_breaker_network_protection(self):
        """Test circuit breaker protection during network failures."""
        failure_count = [0]
        
        def failing_api_call(*args, **kwargs):
            failure_count[0] += 1
            raise requests.exceptions.ConnectionError(f"Failure #{failure_count[0]}")
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=failing_api_call):
            
            mock_load_cards.return_value = self.fallback_cards_data
            
            # Multiple attempts should trigger circuit breaker
            for i in range(10):
                try:
                    hero_advisor = HeroSelectionAdvisor()
                    recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                    
                    # Should eventually use fallback/circuit breaker
                    if recommendations:
                        self.assertEqual(len(recommendations), 3)
                        
                except Exception:
                    # Some failures expected as circuit breaker engages
                    pass
            
            # Circuit breaker should limit the number of actual API calls
            # (exact behavior depends on circuit breaker implementation)
            self.assertLess(failure_count[0], 20, 
                          "Circuit breaker should limit repeated failures")


class TestNetworkResilienceEdgeCases(unittest.TestCase):
    """Test edge cases in network resilience."""
    
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
    
    def test_dns_resolution_failure(self):
        """Test behavior when DNS resolution fails."""
        def dns_failure(*args, **kwargs):
            raise socket.gaierror("Name or service not known")
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=dns_failure):
            
            mock_load_cards.return_value = self.minimal_data
            
            # Should handle DNS failure gracefully
            try:
                hero_advisor = HeroSelectionAdvisor()
                recommendations = hero_advisor.recommend_hero(["WARRIOR"])
                
                # Should provide fallback recommendations
                self.assertEqual(len(recommendations), 1)
                self.assertLess(recommendations[0]['confidence'], 0.8)
                
            except Exception as e:
                # Should be a meaningful error
                self.assertIn("network", str(e).lower())
    
    def test_ssl_certificate_errors(self):
        """Test behavior when SSL certificate verification fails."""
        def ssl_error(*args, **kwargs):
            raise requests.exceptions.SSLError("Certificate verification failed")
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=ssl_error):
            
            mock_load_cards.return_value = self.minimal_data
            
            # Should handle SSL errors gracefully
            try:
                hero_advisor = HeroSelectionAdvisor()
                recommendations = hero_advisor.recommend_hero(["WARRIOR"])
                
                # Should provide fallback recommendations
                self.assertEqual(len(recommendations), 1)
                
            except Exception as e:
                # Should be a meaningful error
                self.assertIn("ssl", str(e).lower())
    
    def test_malformed_response_handling(self):
        """Test behavior when API returns malformed responses."""
        def malformed_response(*args, **kwargs):
            return "Invalid JSON response"
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates',
                   side_effect=malformed_response):
            
            mock_load_cards.return_value = self.minimal_data
            
            # Should handle malformed responses gracefully
            try:
                hero_advisor = HeroSelectionAdvisor()
                recommendations = hero_advisor.recommend_hero(["WARRIOR"])
                
                # Should provide fallback recommendations
                self.assertEqual(len(recommendations), 1)
                
            except Exception as e:
                # Should be a meaningful error about data format
                error_msg = str(e).lower()
                self.assertTrue("json" in error_msg or "format" in error_msg or "parse" in error_msg)


def run_network_resilience_tests():
    """Run all network resilience tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestNetworkResilience))
    test_suite.addTest(unittest.makeSuite(TestNetworkResilienceEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_network_resilience_tests()
    
    if success:
        print("\n✅ All network resilience tests passed!")
    else:
        print("\n❌ Some network resilience tests failed!")
        sys.exit(1)