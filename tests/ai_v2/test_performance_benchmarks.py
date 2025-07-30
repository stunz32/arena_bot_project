"""
Performance benchmarking test suite for Phase 1 AI components.
Tests performance requirements, memory usage, and scalability limits.
"""

import unittest
import time
import threading
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev
import sys

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.deck_analyzer import StrategicDeckAnalyzer
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState


class PerformanceBenchmarks(unittest.TestCase):
    """
    Performance benchmarking test suite for AI components.
    """
    
    def setUp(self):
        """Set up performance testing environment."""
        self.process = psutil.Process(os.getpid())
        self.card_evaluator = CardEvaluationEngine()
        self.deck_analyzer = StrategicDeckAnalyzer()
        self.grandmaster_advisor = GrandmasterAdvisor()
        
        # Create test data sets of varying sizes
        self.test_cards = self._create_test_cards(50)
        self.small_deck = self._create_deck_state(5)
        self.medium_deck = self._create_deck_state(15) 
        self.large_deck = self._create_deck_state(29)
        
        # Performance thresholds (from todo_ai_helper.md)
        self.CARD_EVAL_THRESHOLD_MS = 10  # <10ms per evaluation
        self.DECK_ANALYSIS_THRESHOLD_MS = 50  # <50ms per analysis
        self.FULL_ANALYSIS_THRESHOLD_MS = 100  # <100ms per full analysis
        self.MEMORY_GROWTH_THRESHOLD_MB = 50  # <50MB growth over 1000 operations
    
    def _create_test_cards(self, count):
        """Create test cards for benchmarking."""
        cards = []
        for i in range(count):
            card = CardInstance(
                name=f"Benchmark Card {i}",
                cost=(i % 10) + 1,
                attack=(i % 8) + 1,
                health=(i % 8) + 1,
                card_type="minion" if i % 4 != 0 else "spell",
                rarity=["common", "rare", "epic", "legendary"][i % 4],
                card_set=f"set_{i % 5}",
                keywords=["taunt", "charge", "divine_shield"][:(i % 3)],
                description=f"Benchmark card number {i}"
            )
            cards.append(card)
        return cards
    
    def _create_deck_state(self, drafted_count):
        """Create deck state with specified number of drafted cards."""
        drafted_cards = self.test_cards[:drafted_count]
        available_choices = [
            CardOption(self.test_cards[drafted_count], 0.8),
            CardOption(self.test_cards[drafted_count + 1], 0.7),
            CardOption(self.test_cards[drafted_count + 2], 0.9)
        ]
        
        return DeckState(
            drafted_cards=drafted_cards,
            available_choices=available_choices,
            draft_pick_number=drafted_count + 1,
            wins=drafted_count // 5,
            losses=drafted_count // 8
        )
    
    def _measure_memory_usage(self):
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss
    
    def _run_timed_operation(self, operation, *args, **kwargs):
        """Run operation and return execution time in milliseconds."""
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000, result

    # Card Evaluator Performance Tests
    
    def test_card_evaluator_single_performance(self):
        """Test single card evaluation performance."""
        card = self.test_cards[0]
        deck_state = self.small_deck
        
        # Warm up
        for _ in range(10):
            self.card_evaluator.evaluate_card(card, deck_state)
        
        # Benchmark
        times = []
        for _ in range(100):
            exec_time, result = self._run_timed_operation(
                self.card_evaluator.evaluate_card, card, deck_state
            )
            times.append(exec_time)
            self.assertIsInstance(result, dict)
        
        avg_time = mean(times)
        median_time = median(times)
        max_time = max(times)
        
        print(f"Card Evaluation - Avg: {avg_time:.2f}ms, Median: {median_time:.2f}ms, Max: {max_time:.2f}ms")
        
        # Performance assertions
        self.assertLess(avg_time, self.CARD_EVAL_THRESHOLD_MS,
                       f"Average card evaluation time {avg_time:.2f}ms exceeds threshold {self.CARD_EVAL_THRESHOLD_MS}ms")
        self.assertLess(median_time, self.CARD_EVAL_THRESHOLD_MS,
                       f"Median card evaluation time {median_time:.2f}ms exceeds threshold {self.CARD_EVAL_THRESHOLD_MS}ms")
    
    def test_card_evaluator_batch_performance(self):
        """Test batch card evaluation performance."""
        deck_state = self.medium_deck
        
        start_time = time.perf_counter()
        results = []
        
        # Evaluate 1000 cards
        for i in range(1000):
            card = self.test_cards[i % len(self.test_cards)]
            result = self.card_evaluator.evaluate_card(card, deck_state)
            results.append(result)
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        avg_time_per_card = total_time / 1000
        
        print(f"Batch Card Evaluation - Total: {total_time:.2f}ms, Avg per card: {avg_time_per_card:.2f}ms")
        
        # Should maintain performance under batch load
        self.assertLess(avg_time_per_card, self.CARD_EVAL_THRESHOLD_MS * 1.5,
                       f"Batch evaluation average {avg_time_per_card:.2f}ms too slow")
        self.assertEqual(len(results), 1000)
    
    def test_card_evaluator_caching_performance(self):
        """Test caching performance improvement."""
        card = self.test_cards[0]
        deck_state = self.small_deck
        
        # Time without cache (first evaluation)
        self.card_evaluator._cache.clear()
        time_no_cache, _ = self._run_timed_operation(
            self.card_evaluator.evaluate_card, card, deck_state
        )
        
        # Time with cache (second evaluation)
        time_with_cache, _ = self._run_timed_operation(
            self.card_evaluator.evaluate_card, card, deck_state
        )
        
        print(f"Caching - No cache: {time_no_cache:.2f}ms, With cache: {time_with_cache:.2f}ms")
        
        # Cache should provide significant speedup
        cache_speedup = time_no_cache / max(time_with_cache, 0.001)  # Avoid division by zero
        self.assertGreater(cache_speedup, 2.0,
                          f"Cache speedup {cache_speedup:.2f}x is insufficient")

    # Deck Analyzer Performance Tests
    
    def test_deck_analyzer_performance_scaling(self):
        """Test deck analyzer performance across different deck sizes."""
        test_cases = [
            ("Small deck", self.small_deck),
            ("Medium deck", self.medium_deck),
            ("Large deck", self.large_deck)
        ]
        
        for deck_name, deck_state in test_cases:
            with self.subTest(deck=deck_name):
                # Warm up
                for _ in range(5):
                    self.deck_analyzer.analyze_deck(deck_state)
                
                # Benchmark
                times = []
                for _ in range(50):
                    exec_time, result = self._run_timed_operation(
                        self.deck_analyzer.analyze_deck, deck_state
                    )
                    times.append(exec_time)
                    self.assertIsInstance(result, dict)
                
                avg_time = mean(times)
                print(f"Deck Analysis ({deck_name}) - Avg: {avg_time:.2f}ms")
                
                # Performance should scale reasonably with deck size
                self.assertLess(avg_time, self.DECK_ANALYSIS_THRESHOLD_MS,
                               f"{deck_name} analysis time {avg_time:.2f}ms exceeds threshold")
    
    def test_deck_analyzer_archetype_detection_performance(self):
        """Test archetype detection performance across different archetypes."""
        archetype_decks = self._create_archetype_specific_decks()
        
        for archetype_name, deck_state in archetype_decks.items():
            with self.subTest(archetype=archetype_name):
                times = []
                for _ in range(25):
                    exec_time, result = self._run_timed_operation(
                        self.deck_analyzer.analyze_deck, deck_state
                    )
                    times.append(exec_time)
                
                avg_time = mean(times)
                print(f"Archetype Detection ({archetype_name}) - Avg: {avg_time:.2f}ms")
                
                # Should maintain consistent performance across archetypes
                self.assertLess(avg_time, self.DECK_ANALYSIS_THRESHOLD_MS,
                               f"Archetype detection for {archetype_name} too slow: {avg_time:.2f}ms")
    
    def _create_archetype_specific_decks(self):
        """Create deck states representing different archetypes."""
        aggressive_cards = [
            CardInstance(f"Aggressive {i}", min(i, 3), i+1, max(i, 1), 
                        "minion", "common", "test", ["charge"], f"Aggressive card {i}")
            for i in range(1, 11)
        ]
        
        control_cards = [
            CardInstance(f"Control {i}", i+3, i, i+2, 
                        "minion", "rare", "test", ["taunt"], f"Control card {i}")
            for i in range(5, 15)
        ]
        
        return {
            'aggressive': DeckState(aggressive_cards, [], 15, 0, 0),
            'control': DeckState(control_cards, [], 15, 0, 0),
            'mixed': DeckState(aggressive_cards[:5] + control_cards[:5], [], 15, 0, 0)
        }

    # Grandmaster Advisor Performance Tests
    
    def test_grandmaster_advisor_full_analysis_performance(self):
        """Test full analysis performance including all special features."""
        deck_states = [self.small_deck, self.medium_deck, self.large_deck]
        
        for i, deck_state in enumerate(deck_states):
            deck_name = ["Small", "Medium", "Large"][i]
            with self.subTest(deck=deck_name):
                # Warm up
                for _ in range(3):
                    self.grandmaster_advisor.analyze_draft_choice(deck_state)
                
                # Benchmark
                times = []
                for _ in range(25):
                    exec_time, result = self._run_timed_operation(
                        self.grandmaster_advisor.analyze_draft_choice, deck_state
                    )
                    times.append(exec_time)
                    self.assertIsNotNone(result)
                
                avg_time = mean(times)
                median_time = median(times)
                
                print(f"Full Analysis ({deck_name}) - Avg: {avg_time:.2f}ms, Median: {median_time:.2f}ms")
                
                # Should meet full analysis performance threshold
                self.assertLess(avg_time, self.FULL_ANALYSIS_THRESHOLD_MS,
                               f"Full analysis for {deck_name} deck too slow: {avg_time:.2f}ms")
    
    def test_special_features_performance_impact(self):
        """Test performance impact of special features."""
        deck_state = self.medium_deck
        
        # Measure with all features enabled
        times_full = []
        for _ in range(20):
            exec_time, _ = self._run_timed_operation(
                self.grandmaster_advisor.analyze_draft_choice, deck_state
            )
            times_full.append(exec_time)
        
        avg_full = mean(times_full)
        
        # Test individual feature performance
        features_to_test = [
            'pivot_analysis',
            'greed_analysis', 
            'synergy_trap_analysis',
            'comparative_analysis'
        ]
        
        for feature in features_to_test:
            print(f"Feature '{feature}' - Contributing to {avg_full:.2f}ms total time")
            # Individual feature impact should be reasonable
            # (Detailed feature isolation would require more complex mocking)
        
        # Overall performance should still meet threshold
        self.assertLess(avg_full, self.FULL_ANALYSIS_THRESHOLD_MS,
                       f"Full analysis with all features too slow: {avg_full:.2f}ms")

    # Memory Performance Tests
    
    def test_memory_usage_card_evaluator(self):
        """Test card evaluator memory usage over extended operation."""
        initial_memory = self._measure_memory_usage()
        
        # Perform 1000 evaluations
        for i in range(1000):
            card = self.test_cards[i % len(self.test_cards)]
            deck_state = self.medium_deck
            result = self.card_evaluator.evaluate_card(card, deck_state)
        
        gc.collect()  # Force garbage collection
        final_memory = self._measure_memory_usage()
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        print(f"Card Evaluator Memory Growth: {memory_growth:.2f}MB over 1000 evaluations")
        
        self.assertLess(memory_growth, self.MEMORY_GROWTH_THRESHOLD_MB,
                       f"Memory growth {memory_growth:.2f}MB exceeds threshold {self.MEMORY_GROWTH_THRESHOLD_MB}MB")
    
    def test_memory_usage_deck_analyzer(self):
        """Test deck analyzer memory usage over extended operation."""
        initial_memory = self._measure_memory_usage()
        
        # Perform 500 analyses with varying deck sizes
        for i in range(500):
            deck_state = [self.small_deck, self.medium_deck, self.large_deck][i % 3]
            result = self.deck_analyzer.analyze_deck(deck_state)
        
        gc.collect()
        final_memory = self._measure_memory_usage()
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)
        
        print(f"Deck Analyzer Memory Growth: {memory_growth:.2f}MB over 500 analyses")
        
        self.assertLess(memory_growth, self.MEMORY_GROWTH_THRESHOLD_MB,
                       f"Memory growth {memory_growth:.2f}MB exceeds threshold")
    
    def test_memory_usage_grandmaster_advisor(self):
        """Test full system memory usage over extended operation."""
        initial_memory = self._measure_memory_usage()
        
        # Perform 200 full analyses
        for i in range(200):
            deck_state = [self.small_deck, self.medium_deck, self.large_deck][i % 3]
            result = self.grandmaster_advisor.analyze_draft_choice(deck_state)
        
        gc.collect()
        final_memory = self._measure_memory_usage()
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)
        
        print(f"Full System Memory Growth: {memory_growth:.2f}MB over 200 full analyses")
        
        self.assertLess(memory_growth, self.MEMORY_GROWTH_THRESHOLD_MB * 1.5,
                       f"Full system memory growth {memory_growth:.2f}MB too high")

    # Concurrent Performance Tests
    
    def test_concurrent_card_evaluation_performance(self):
        """Test card evaluation performance under concurrent load."""
        def evaluate_cards_batch():
            times = []
            for i in range(50):
                card = self.test_cards[i % len(self.test_cards)]
                start_time = time.perf_counter()
                result = self.card_evaluator.evaluate_card(card, self.medium_deck)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            return times
        
        # Run concurrent evaluations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(evaluate_cards_batch) for _ in range(4)]
            all_times = []
            for future in as_completed(futures):
                all_times.extend(future.result())
        
        avg_concurrent_time = mean(all_times)
        
        print(f"Concurrent Card Evaluation - Avg: {avg_concurrent_time:.2f}ms across 4 threads")
        
        # Performance under concurrency should be reasonable
        self.assertLess(avg_concurrent_time, self.CARD_EVAL_THRESHOLD_MS * 2,
                       f"Concurrent evaluation too slow: {avg_concurrent_time:.2f}ms")
    
    def test_concurrent_full_analysis_performance(self):
        """Test full analysis performance under concurrent load."""
        def analyze_drafts_batch():
            times = []
            for i in range(15):
                deck_state = [self.small_deck, self.medium_deck, self.large_deck][i % 3]
                start_time = time.perf_counter()
                result = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            return times
        
        # Run concurrent analyses
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(analyze_drafts_batch) for _ in range(3)]
            all_times = []
            for future in as_completed(futures):
                all_times.extend(future.result())
        
        avg_concurrent_time = mean(all_times)
        
        print(f"Concurrent Full Analysis - Avg: {avg_concurrent_time:.2f}ms across 3 threads")
        
        # Should handle concurrency reasonably well
        self.assertLess(avg_concurrent_time, self.FULL_ANALYSIS_THRESHOLD_MS * 2,
                       f"Concurrent full analysis too slow: {avg_concurrent_time:.2f}ms")

    # Scalability Tests
    
    def test_cache_performance_scaling(self):
        """Test cache performance as cache size grows."""
        cache_sizes = [10, 50, 100, 500, 1000]
        performance_results = {}
        
        for cache_size in cache_sizes:
            # Clear and populate cache to target size
            self.card_evaluator._cache.clear()
            
            # Populate cache
            for i in range(cache_size):
                card = self.test_cards[i % len(self.test_cards)]
                self.card_evaluator.evaluate_card(card, self.small_deck)
            
            # Measure cache lookup performance
            times = []
            for _ in range(100):
                card = self.test_cards[0]  # Should be in cache
                exec_time, _ = self._run_timed_operation(
                    self.card_evaluator.evaluate_card, card, self.small_deck
                )
                times.append(exec_time)
            
            avg_time = mean(times)
            performance_results[cache_size] = avg_time
            print(f"Cache size {cache_size} - Avg lookup: {avg_time:.3f}ms")
        
        # Cache performance should not degrade significantly with size
        max_acceptable_degradation = 2.0  # 2x slowdown max
        baseline_performance = performance_results[cache_sizes[0]]
        
        for cache_size in cache_sizes[1:]:
            current_performance = performance_results[cache_size]
            degradation_ratio = current_performance / baseline_performance
            
            self.assertLess(degradation_ratio, max_acceptable_degradation,
                           f"Cache performance degraded {degradation_ratio:.2f}x at size {cache_size}")
    
    def test_large_deck_analysis_performance(self):
        """Test performance with very large deck configurations."""
        # Create deck with maximum realistic size
        large_card_pool = self.test_cards * 5  # 250 cards
        max_deck = DeckState(
            drafted_cards=large_card_pool[:29],
            available_choices=[
                CardOption(large_card_pool[29], 0.8),
                CardOption(large_card_pool[30], 0.7),
                CardOption(large_card_pool[31], 0.9)
            ],
            draft_pick_number=30,
            wins=10,
            losses=2
        )
        
        # Test performance doesn't degrade excessively
        times = []
        for _ in range(10):
            exec_time, result = self._run_timed_operation(
                self.grandmaster_advisor.analyze_draft_choice, max_deck
            )
            times.append(exec_time)
        
        avg_time = mean(times)
        max_time = max(times)
        
        print(f"Large Deck Analysis - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        
        # Should handle large decks within reasonable time
        self.assertLess(avg_time, self.FULL_ANALYSIS_THRESHOLD_MS * 3,
                       f"Large deck analysis too slow: {avg_time:.2f}ms")
        self.assertLess(max_time, self.FULL_ANALYSIS_THRESHOLD_MS * 5,
                       f"Large deck analysis max time too slow: {max_time:.2f}ms")

    # Resource Utilization Tests
    
    def test_cpu_utilization_efficiency(self):
        """Test CPU utilization efficiency during analysis."""
        import threading
        import time
        
        cpu_usage_samples = []
        stop_monitoring = threading.Event()
        
        def monitor_cpu():
            while not stop_monitoring.is_set():
                cpu_percent = self.process.cpu_percent()
                cpu_usage_samples.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Perform intensive analysis workload
            for i in range(100):
                deck_state = [self.small_deck, self.medium_deck, self.large_deck][i % 3]
                result = self.grandmaster_advisor.analyze_draft_choice(deck_state)
        finally:
            stop_monitoring.set()
            monitor_thread.join()
        
        if cpu_usage_samples:
            avg_cpu_usage = mean(cpu_usage_samples)
            max_cpu_usage = max(cpu_usage_samples)
            
            print(f"CPU Usage - Avg: {avg_cpu_usage:.1f}%, Max: {max_cpu_usage:.1f}%")
            
            # CPU usage should be reasonable (not excessive)
            self.assertLess(max_cpu_usage, 90.0,
                           f"CPU usage too high: {max_cpu_usage:.1f}%")
    
    def test_thread_safety_performance_impact(self):
        """Test performance impact of thread safety mechanisms."""
        # Single-threaded baseline
        single_threaded_times = []
        for _ in range(50):
            exec_time, _ = self._run_timed_operation(
                self.card_evaluator.evaluate_card, self.test_cards[0], self.medium_deck
            )
            single_threaded_times.append(exec_time)
        
        single_threaded_avg = mean(single_threaded_times)
        
        # Multi-threaded performance
        def concurrent_evaluation():
            times = []
            for _ in range(25):
                exec_time, _ = self._run_timed_operation(
                    self.card_evaluator.evaluate_card, self.test_cards[0], self.medium_deck
                )
                times.append(exec_time)
            return times
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(concurrent_evaluation) for _ in range(2)]
            multi_threaded_times = []
            for future in as_completed(futures):
                multi_threaded_times.extend(future.result())
        
        multi_threaded_avg = mean(multi_threaded_times)
        thread_safety_overhead = (multi_threaded_avg - single_threaded_avg) / single_threaded_avg
        
        print(f"Thread Safety Overhead: {thread_safety_overhead * 100:.1f}%")
        
        # Thread safety overhead should be minimal
        self.assertLess(thread_safety_overhead, 0.5,
                       f"Thread safety overhead too high: {thread_safety_overhead * 100:.1f}%")

    def tearDown(self):
        """Clean up performance test environment."""
        # Clear caches to prevent interference
        if hasattr(self.card_evaluator, '_cache'):
            self.card_evaluator._cache.clear()
        
        # Force garbage collection
        gc.collect()


class PerformanceRegressionTests(unittest.TestCase):
    """
    Performance regression detection tests.
    """
    
    def setUp(self):
        self.advisor = GrandmasterAdvisor()
        self.baseline_times = self._load_baseline_performance()
    
    def _load_baseline_performance(self):
        """Load baseline performance metrics (would be from previous runs)."""
        # In a real implementation, these would be loaded from a file
        return {
            'card_evaluation_ms': 5.0,
            'deck_analysis_ms': 30.0,
            'full_analysis_ms': 80.0
        }
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        # This would compare current performance against baselines
        # and alert if there's significant degradation
        
        # Create test scenario
        card = CardInstance("Test Card", 3, 3, 3, "minion", "common", "test", [], "Test")
        deck_state = DeckState([card], [], 10, 1, 1)
        
        # Measure current performance
        times = []
        for _ in range(20):
            start_time = time.perf_counter()
            result = self.advisor.analyze_draft_choice(deck_state)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        current_avg = mean(times)
        baseline_avg = self.baseline_times['full_analysis_ms']
        
        # Check for regression (>20% slowdown)
        regression_threshold = 1.2
        performance_ratio = current_avg / baseline_avg
        
        print(f"Performance Regression Check - Current: {current_avg:.2f}ms, Baseline: {baseline_avg:.2f}ms, Ratio: {performance_ratio:.2f}")
        
        if performance_ratio > regression_threshold:
            self.fail(f"Performance regression detected: {performance_ratio:.2f}x slower than baseline")


if __name__ == '__main__':
    print("Starting Performance Benchmark Suite...")
    print("=" * 60)
    
    # Run with high verbosity to see all timing results
    unittest.main(verbosity=2, buffer=True)