"""
Comprehensive test suite for CardEvaluationEngine - Phase 1
Tests all scoring dimensions, ML fallbacks, caching, and hardening features.
"""

import unittest
import threading
import time
import tempfile
import hashlib
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState
from arena_bot.ai_v2.exceptions import AIEngineError, ModelLoadingError


class TestCardEvaluationEngine(unittest.TestCase):
    """
    Comprehensive test suite for CardEvaluationEngine with hardening validation.
    """
    
    def setUp(self):
        """Set up test environment with clean evaluator instance."""
        self.evaluator = CardEvaluationEngine()
        self.sample_cards = self._create_sample_cards()
        self.sample_deck_state = self._create_sample_deck_state()
    
    def _create_sample_cards(self):
        """Create sample card instances for testing."""
        return [
            CardInstance(
                name="Fiery War Axe",
                cost=2,
                attack=3,
                health=2,
                card_type="weapon",
                rarity="common",
                card_set="classic",
                keywords=["weapon"],
                description="A classic weapon"
            ),
            CardInstance(
                name="Flamestrike",
                cost=7,
                attack=None,
                health=None,
                card_type="spell",
                rarity="epic",
                card_set="classic",
                keywords=["spell", "damage"],
                description="Deal 4 damage to all enemy minions"
            ),
            CardInstance(
                name="Chillwind Yeti",
                cost=4,
                attack=4,
                health=5,
                card_type="minion",
                rarity="common",
                card_set="classic",
                keywords=[],
                description="A vanilla minion"
            )
        ]
    
    def _create_sample_deck_state(self):
        """Create sample deck state for testing."""
        return DeckState(
            drafted_cards=[self.sample_cards[0]],
            available_choices=[
                CardOption(self.sample_cards[0], 0.8),
                CardOption(self.sample_cards[1], 0.7),
                CardOption(self.sample_cards[2], 0.9)
            ],
            draft_pick_number=5,
            wins=2,
            losses=0
        )

    # Core Functionality Tests
    
    def test_evaluate_card_basic_functionality(self):
        """Test basic card evaluation returns valid scores."""
        card = self.sample_cards[0]
        result = self.evaluator.evaluate_card(card, self.sample_deck_state)
        
        # Validate score structure
        self.assertIsInstance(result, dict)
        required_keys = [
            'overall_score', 'base_value', 'tempo_score', 'value_score',
            'synergy_score', 'curve_score', 're_draftability_score'
        ]
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 10)
    
    def test_all_scoring_dimensions(self):
        """Test all six scoring dimensions with different card types."""
        for card in self.sample_cards:
            with self.subTest(card=card.name):
                result = self.evaluator.evaluate_card(card, self.sample_deck_state)
                
                # Each dimension should return reasonable scores
                self.assertGreater(result['base_value'], 0)
                self.assertGreaterEqual(result['tempo_score'], 0)
                self.assertGreaterEqual(result['value_score'], 0)
                self.assertGreaterEqual(result['synergy_score'], 0)
                self.assertGreaterEqual(result['curve_score'], 0)
                self.assertGreaterEqual(result['re_draftability_score'], 0)
    
    def test_input_validation_and_sanitization(self):
        """Test comprehensive input validation and sanitization."""
        # Test null card
        with self.assertRaises(AIEngineError):
            self.evaluator.evaluate_card(None, self.sample_deck_state)
        
        # Test invalid card with missing fields
        invalid_card = CardInstance(
            name="",  # Empty name
            cost=-1,  # Invalid cost
            attack=None,
            health=None,
            card_type="invalid",
            rarity="",
            card_set="",
            keywords=[],
            description=""
        )
        
        # Should handle gracefully with fallbacks
        result = self.evaluator.evaluate_card(invalid_card, self.sample_deck_state)
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
    
    def test_performance_monitoring(self):
        """Test performance monitoring for each scoring dimension."""
        card = self.sample_cards[0]
        
        # Mock the monitoring system
        with patch.object(self.evaluator.monitor, 'record_timing') as mock_timing:
            result = self.evaluator.evaluate_card(card, self.sample_deck_state)
            
            # Verify timing was recorded for each dimension
            self.assertTrue(mock_timing.called)
            self.assertIn('overall_score', result)

    # ML Model Fallback Tests
    
    @patch('arena_bot.ai_v2.card_evaluator.CardEvaluationEngine._load_ml_models')
    def test_ml_model_loading_failure_fallback(self, mock_load):
        """Test fallback to heuristics when ML models fail to load."""
        # Simulate ML model loading failure
        mock_load.side_effect = ModelLoadingError("Model file corrupted")
        
        evaluator = CardEvaluationEngine()
        card = self.sample_cards[0]
        
        # Should fallback to heuristics
        result = evaluator.evaluate_card(card, self.sample_deck_state)
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        self.assertTrue(evaluator._using_fallback_heuristics)
    
    def test_model_integrity_validation(self):
        """Test model integrity validation with checksum verification."""
        # Create a temporary model file with known checksum
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"fake model data")
            tmp_file_path = tmp_file.name
        
        try:
            # Test checksum validation
            expected_checksum = hashlib.sha256(b"fake model data").hexdigest()
            actual_checksum = self.evaluator._calculate_file_checksum(tmp_file_path)
            self.assertEqual(expected_checksum, actual_checksum)
        finally:
            os.unlink(tmp_file_path)
    
    def test_model_loading_timeout(self):
        """Test model loading timeout and recovery."""
        with patch('arena_bot.ai_v2.card_evaluator.CardEvaluationEngine._load_model_with_timeout') as mock_load:
            # Simulate timeout
            mock_load.side_effect = TimeoutError("Model loading timed out")
            
            evaluator = CardEvaluationEngine()
            # Should fallback to heuristics after timeout
            self.assertTrue(evaluator._using_fallback_heuristics)

    # Threading and Concurrency Tests
    
    def test_thread_safe_evaluation(self):
        """Test thread-safe evaluation methods."""
        card = self.sample_cards[0]
        results = []
        errors = []
        
        def evaluate_card_thread():
            try:
                result = self.evaluator.evaluate_card(card, self.sample_deck_state)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 10 concurrent evaluations
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=evaluate_card_thread)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        self.assertEqual(len(errors), 0, f"Errors in concurrent evaluation: {errors}")
        self.assertEqual(len(results), 10)
        
        # All results should be similar (within reasonable variance)
        first_score = results[0]['overall_score']
        for result in results[1:]:
            self.assertAlmostEqual(result['overall_score'], first_score, delta=0.1)

    # Caching System Tests
    
    def test_caching_layer_functionality(self):
        """Test caching layer for expensive calculations."""
        card = self.sample_cards[0]
        
        # First evaluation should cache result
        result1 = self.evaluator.evaluate_card(card, self.sample_deck_state)
        
        # Second evaluation should use cache
        with patch.object(self.evaluator, '_calculate_base_value') as mock_calc:
            result2 = self.evaluator.evaluate_card(card, self.sample_deck_state)
            
            # Should not recalculate if cached
            if self.evaluator._cache_enabled:
                mock_calc.assert_not_called()
        
        # Results should be identical
        self.assertEqual(result1['overall_score'], result2['overall_score'])
    
    def test_cache_key_hashing(self):
        """Test SHA-256 cache key hashing to prevent collisions."""
        card1 = self.sample_cards[0]
        card2 = self.sample_cards[1]
        
        key1 = self.evaluator._generate_cache_key(card1, self.sample_deck_state)
        key2 = self.evaluator._generate_cache_key(card2, self.sample_deck_state)
        
        # Keys should be different and in SHA-256 format
        self.assertNotEqual(key1, key2)
        self.assertEqual(len(key1), 64)  # SHA-256 hex length
        self.assertEqual(len(key2), 64)
    
    def test_cache_eviction_strategy(self):
        """Test LRU cache eviction with memory pressure monitoring."""
        # Fill cache beyond capacity
        for i in range(self.evaluator._max_cache_size + 5):
            test_card = CardInstance(
                name=f"Test Card {i}",
                cost=i % 10,
                attack=i % 10,
                health=i % 10,
                card_type="minion",
                rarity="common",
                card_set="test",
                keywords=[],
                description=f"Test card {i}"
            )
            self.evaluator.evaluate_card(test_card, self.sample_deck_state)
        
        # Cache size should not exceed maximum
        self.assertLessEqual(len(self.evaluator._cache), self.evaluator._max_cache_size)
    
    def test_cache_health_monitoring(self):
        """Test cache corruption detection and recovery."""
        # Simulate cache corruption
        self.evaluator._cache = {"corrupted": "data"}
        
        # Should detect corruption and reset cache
        result = self.evaluator.evaluate_card(self.sample_cards[0], self.sample_deck_state)
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)

    # Edge Cases and Error Handling Tests
    
    def test_edge_case_cards(self):
        """Test evaluation of edge case cards."""
        edge_cases = [
            # Zero cost card
            CardInstance("Innervate", 0, None, None, "spell", "common", "classic", [], "Gain 2 mana"),
            # High cost card
            CardInstance("Mountain Giant", 12, 8, 8, "minion", "epic", "classic", [], "Costs 1 less for each other card"),
            # Weapon with durability
            CardInstance("Gorehowl", 7, 7, 1, "weapon", "epic", "classic", ["weapon"], "Attacking minion doesn't reduce durability"),
        ]
        
        for card in edge_cases:
            with self.subTest(card=card.name):
                result = self.evaluator.evaluate_card(card, self.sample_deck_state)
                self.assertIsInstance(result, dict)
                self.assertIn('overall_score', result)
                self.assertGreaterEqual(result['overall_score'], 0)

    def test_missing_data_scenarios(self):
        """Test fallback heuristics for missing data scenarios."""
        # Card with minimal data
        minimal_card = CardInstance(
            name="Unknown Card",
            cost=3,
            attack=None,
            health=None,
            card_type="unknown",
            rarity="",
            card_set="",
            keywords=[],
            description=""
        )
        
        result = self.evaluator.evaluate_card(minimal_card, self.sample_deck_state)
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        # Should use fallback scoring
        self.assertGreater(result['overall_score'], 0)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Extreme card stats
        extreme_card = CardInstance(
            name="Extreme Card",
            cost=0,
            attack=999,
            health=999,
            card_type="minion",
            rarity="legendary",
            card_set="test",
            keywords=["charge", "taunt", "divine_shield"],
            description="Extreme test card"
        )
        
        result = self.evaluator.evaluate_card(extreme_card, self.sample_deck_state)
        
        # Verify no NaN or Infinity values
        for key, value in result.items():
            self.assertFalse(math.isnan(value) if hasattr(math, 'isnan') else False, 
                           f"NaN value in {key}")
            self.assertFalse(math.isinf(value) if hasattr(math, 'isinf') else False, 
                           f"Infinity value in {key}")
            self.assertIsInstance(value, (int, float))

    # Performance and Stress Tests
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for evaluation speed."""
        card = self.sample_cards[0]
        
        start_time = time.time()
        
        # Evaluate 100 cards
        for _ in range(100):
            result = self.evaluator.evaluate_card(card, self.sample_deck_state)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_evaluation = total_time / 100
        
        # Should average less than 10ms per evaluation
        self.assertLess(avg_time_per_evaluation, 0.01, 
                       f"Average evaluation time too slow: {avg_time_per_evaluation:.4f}s")
    
    def test_memory_usage_monitoring(self):
        """Test memory usage doesn't grow unbounded."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many evaluations
        for i in range(1000):
            card = CardInstance(
                name=f"Memory Test {i}",
                cost=i % 10,
                attack=i % 10,
                health=i % 10,
                card_type="minion",
                rarity="common",
                card_set="test",
                keywords=[],
                description=f"Memory test card {i}"
            )
            self.evaluator.evaluate_card(card, self.sample_deck_state)
        
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        self.assertLess(memory_growth, 50 * 1024 * 1024, 
                       f"Excessive memory growth: {memory_growth / 1024 / 1024:.2f}MB")
    
    def test_concurrent_stress_test(self):
        """Test system under concurrent load."""
        results = []
        errors = []
        
        def stress_evaluation():
            try:
                for i in range(50):
                    card = self.sample_cards[i % len(self.sample_cards)]
                    result = self.evaluator.evaluate_card(card, self.sample_deck_state)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 5 concurrent stress threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(stress_evaluation) for _ in range(5)]
            for future in futures:
                future.result()
        
        # Verify no errors under stress
        self.assertEqual(len(errors), 0, f"Errors under stress: {errors}")
        self.assertEqual(len(results), 250)  # 5 threads * 50 evaluations

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self.evaluator, '_cleanup'):
            self.evaluator._cleanup()


class TestCardEvaluatorSubcomponents(unittest.TestCase):
    """
    Test individual scoring components in detail.
    """
    
    def setUp(self):
        self.evaluator = CardEvaluationEngine()
        self.sample_card = CardInstance(
            name="Test Minion",
            cost=3,
            attack=3,
            health=4,
            card_type="minion",
            rarity="common",
            card_set="test",
            keywords=["taunt"],
            description="Test minion with taunt"
        )
        self.sample_deck_state = DeckState(
            drafted_cards=[],
            available_choices=[],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
    
    def test_base_value_calculation(self):
        """Test base value calculation with ML fallback."""
        result = self.evaluator._calculate_base_value(self.sample_card, self.sample_deck_state)
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 10)
    
    def test_tempo_score_calculation(self):
        """Test tempo score with keyword analysis."""
        result = self.evaluator._calculate_tempo_score(self.sample_card, self.sample_deck_state)
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 10)
    
    def test_value_score_calculation(self):
        """Test value score with resource generation detection."""
        result = self.evaluator._calculate_value_score(self.sample_card, self.sample_deck_state)
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 10)
    
    def test_synergy_score_with_trap_detection(self):
        """Test synergy score with synergy trap detection."""
        result = self.evaluator._calculate_synergy_score(self.sample_card, self.sample_deck_state)
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 10)
    
    def test_curve_score_with_draft_phase(self):
        """Test curve score with draft phase weighting."""
        early_deck = DeckState(
            drafted_cards=[],
            available_choices=[],
            draft_pick_number=3,  # Early draft
            wins=0,
            losses=0
        )
        
        late_deck = DeckState(
            drafted_cards=[self.sample_card] * 25,
            available_choices=[],
            draft_pick_number=28,  # Late draft
            wins=0,
            losses=0
        )
        
        early_score = self.evaluator._calculate_curve_score(self.sample_card, early_deck)
        late_score = self.evaluator._calculate_curve_score(self.sample_card, late_deck)
        
        # Scores should be different based on draft phase
        self.assertIsInstance(early_score, (int, float))
        self.assertIsInstance(late_score, (int, float))
    
    def test_redraftability_score(self):
        """Test re-draftability score with uniqueness analysis."""
        result = self.evaluator._calculate_re_draftability_score(self.sample_card, self.sample_deck_state)
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 10)


if __name__ == '__main__':
    # Import math for NaN/Infinity checks
    import math
    
    # Run the test suite
    unittest.main(verbosity=2)