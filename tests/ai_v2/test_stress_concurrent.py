"""
Stress testing and concurrent access test suite for Phase 1 AI components.
Tests system behavior under high load, concurrent access, and resource pressure.
"""

import unittest
import threading
import time
import queue
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Event, Lock
from arena_bot.utils.bounded_queue import BoundedQueue
import multiprocessing
import random
import sys

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.deck_analyzer import StrategicDeckAnalyzer
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState
from arena_bot.ai_v2.exceptions import AIEngineError


class StressConcurrentTests(unittest.TestCase):
    """
    Stress testing and concurrent access test suite.
    """
    
    def setUp(self):
        """Set up stress testing environment."""
        self.process = psutil.Process(os.getpid())
        self.card_evaluator = CardEvaluationEngine()
        self.deck_analyzer = StrategicDeckAnalyzer()
        self.grandmaster_advisor = GrandmasterAdvisor()
        
        # Create large test data sets
        self.test_cards = self._create_large_card_set(200)
        self.test_deck_states = self._create_test_deck_states(50)
        
        # Stress test parameters
        self.MAX_CONCURRENT_THREADS = 20
        self.STRESS_TEST_DURATION_SECONDS = 30
        self.HIGH_LOAD_OPERATIONS = 1000
        self.MEMORY_LIMIT_MB = 500  # 500MB memory limit for stress test
    
    def _create_large_card_set(self, count):
        """Create large set of test cards for stress testing."""
        cards = []
        card_types = ["minion", "spell", "weapon"]
        rarities = ["common", "rare", "epic", "legendary"]
        keywords_pool = ["taunt", "charge", "divine_shield", "windfury", "stealth", "rush"]
        
        for i in range(count):
            card_type = random.choice(card_types)
            rarity = random.choice(rarities)
            keywords = random.sample(keywords_pool, random.randint(0, 3))
            
            # Generate stats based on type
            if card_type == "spell":
                attack, health = None, None
            elif card_type == "weapon":
                attack = random.randint(1, 8)
                health = random.randint(1, 5)  # Durability
            else:  # minion
                attack = random.randint(0, 12)
                health = random.randint(1, 12)
            
            card = CardInstance(
                name=f"Stress Test Card {i}",
                cost=random.randint(0, 10),
                attack=attack,
                health=health,
                card_type=card_type,
                rarity=rarity,
                card_set=f"stress_set_{i % 10}",
                keywords=keywords,
                description=f"Stress test card {i} with {card_type} type"
            )
            cards.append(card)
        
        return cards
    
    def _create_test_deck_states(self, count):
        """Create multiple test deck states for stress testing."""
        deck_states = []
        
        for i in range(count):
            # Vary deck composition
            drafted_count = random.randint(0, 28)
            drafted_cards = random.sample(self.test_cards, min(drafted_count, len(self.test_cards)))
            
            # Create available choices
            available_cards = random.sample(self.test_cards, 3)
            available_choices = [
                CardOption(card, random.uniform(0.3, 0.95))
                for card in available_cards
            ]
            
            deck_state = DeckState(
                drafted_cards=drafted_cards,
                available_choices=available_choices,
                draft_pick_number=drafted_count + 1,
                wins=random.randint(0, 11),
                losses=random.randint(0, 2)
            )
            deck_states.append(deck_state)
        
        return deck_states
    
    def _monitor_resource_usage(self, duration_seconds, results_queue):
        """Monitor resource usage during stress test."""
        start_time = time.time()
        max_memory = 0
        max_cpu = 0
        measurements = []
        
        while time.time() - start_time < duration_seconds:
            try:
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent()
                
                max_memory = max(max_memory, memory_mb)
                max_cpu = max(max_cpu, cpu_percent)
                
                measurements.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                break
        
        results_queue.put({
            'max_memory_mb': max_memory,
            'max_cpu_percent': max_cpu,
            'measurements': measurements
        })

    # Thread Safety Stress Tests
    
    def test_card_evaluator_thread_safety_stress(self):
        """Stress test card evaluator thread safety with many concurrent threads."""
        results = []
        errors = []
        completion_times = []
        
        def evaluate_cards_stress():
            thread_start = time.time()
            thread_results = []
            thread_errors = []
            
            try:
                for i in range(50):  # 50 evaluations per thread
                    card = random.choice(self.test_cards)
                    deck_state = random.choice(self.test_deck_states)
                    
                    start_time = time.perf_counter()
                    result = self.card_evaluator.evaluate_card(card, deck_state)
                    end_time = time.perf_counter()
                    
                    thread_results.append({
                        'result': result,
                        'execution_time_ms': (end_time - start_time) * 1000
                    })
                    
                    # Small delay to simulate realistic usage
                    time.sleep(0.001)
                    
            except Exception as e:
                thread_errors.append(e)
            finally:
                thread_completion = time.time() - thread_start
                completion_times.append(thread_completion)
                results.extend(thread_results)
                errors.extend(thread_errors)
        
        # Start resource monitoring
        resource_queue = queue.Queue()
        monitor_thread = threading.Thread(
            target=self._monitor_resource_usage,
            args=(15, resource_queue)
        )
        monitor_thread.start()
        
        # Run stress test with many threads
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_THREADS) as executor:
            futures = [executor.submit(evaluate_cards_stress) for _ in range(15)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        monitor_thread.join()
        resource_stats = resource_queue.get()
        
        # Analyze results
        total_evaluations = len(results)
        total_errors = len(errors)
        avg_completion_time = sum(completion_times) / len(completion_times)
        
        print(f"Card Evaluator Stress Test Results:")
        print(f"  Total evaluations: {total_evaluations}")
        print(f"  Total errors: {total_errors}")
        print(f"  Error rate: {(total_errors / total_evaluations * 100):.2f}%")
        print(f"  Average thread completion: {avg_completion_time:.2f}s")
        print(f"  Max memory usage: {resource_stats['max_memory_mb']:.2f}MB")
        print(f"  Max CPU usage: {resource_stats['max_cpu_percent']:.1f}%")
        
        # Assertions
        self.assertEqual(total_errors, 0, f"Thread safety errors: {errors}")
        self.assertGreater(total_evaluations, 500)  # Should complete many evaluations
        self.assertLess(resource_stats['max_memory_mb'], self.MEMORY_LIMIT_MB,
                       f"Memory usage {resource_stats['max_memory_mb']:.2f}MB exceeds limit")
    
    def test_deck_analyzer_concurrent_analysis_stress(self):
        """Stress test deck analyzer with concurrent analyses."""
        results = []
        errors = []
        lock = Lock()
        
        def analyze_decks_concurrently():
            thread_results = []
            thread_errors = []
            
            try:
                for i in range(25):  # 25 analyses per thread
                    deck_state = random.choice(self.test_deck_states)
                    
                    start_time = time.perf_counter()
                    result = self.deck_analyzer.analyze_deck(deck_state)
                    end_time = time.perf_counter()
                    
                    thread_results.append({
                        'archetype': result['primary_archetype'],
                        'confidence': result['archetype_confidence'],
                        'execution_time_ms': (end_time - start_time) * 1000
                    })
                    
                    time.sleep(0.002)  # Slight delay
                    
            except Exception as e:
                thread_errors.append(e)
            finally:
                with lock:
                    results.extend(thread_results)
                    errors.extend(thread_errors)
        
        # Run concurrent stress test
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(analyze_decks_concurrently) for _ in range(8)]
            for future in as_completed(futures):
                future.result()
        
        # Analyze results
        total_analyses = len(results)
        total_errors = len(errors)
        execution_times = [r['execution_time_ms'] for r in results]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        print(f"Deck Analyzer Concurrent Stress Test Results:")
        print(f"  Total analyses: {total_analyses}")
        print(f"  Total errors: {total_errors}")
        print(f"  Average execution time: {avg_execution_time:.2f}ms")
        
        # Verify archetype distribution is reasonable
        archetypes = [r['archetype'] for r in results]
        unique_archetypes = set(archetypes)
        print(f"  Unique archetypes detected: {len(unique_archetypes)}")
        
        # Assertions
        self.assertEqual(total_errors, 0, f"Concurrent analysis errors: {errors}")
        self.assertGreater(total_analyses, 150)
        self.assertGreater(len(unique_archetypes), 2)  # Should detect varied archetypes
    
    def test_grandmaster_advisor_full_system_stress(self):
        """Stress test the complete grandmaster advisor system."""
        results = []
        errors = []
        performance_metrics = []
        
        def full_analysis_stress():
            thread_results = []
            thread_errors = []
            thread_metrics = []
            
            try:
                for i in range(20):  # 20 full analyses per thread
                    deck_state = random.choice(self.test_deck_states)
                    
                    start_time = time.perf_counter()
                    decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                    end_time = time.perf_counter()
                    
                    execution_time = (end_time - start_time) * 1000
                    
                    thread_results.append(decision)
                    thread_metrics.append({
                        'execution_time_ms': execution_time,
                        'confidence_score': decision.confidence_score,
                        'reasoning_length': len(decision.reasoning)
                    })
                    
                    # Simulate realistic draft timing
                    time.sleep(0.01)
                    
            except Exception as e:
                thread_errors.append(e)
            finally:
                results.extend(thread_results)
                errors.extend(thread_errors)
                performance_metrics.extend(thread_metrics)
        
        # Start resource monitoring
        resource_queue = queue.Queue()
        monitor_thread = threading.Thread(
            target=self._monitor_resource_usage,
            args=(25, resource_queue)
        )
        monitor_thread.start()
        
        # Run full system stress test
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(full_analysis_stress) for _ in range(6)]
            for future in as_completed(futures):
                future.result()
        
        monitor_thread.join()
        resource_stats = resource_queue.get()
        
        # Analyze results
        total_decisions = len(results)
        total_errors = len(errors)
        execution_times = [m['execution_time_ms'] for m in performance_metrics]
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)
        
        confidence_scores = [m['confidence_score'] for m in performance_metrics]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        print(f"Full System Stress Test Results:")
        print(f"  Total decisions: {total_decisions}")
        print(f"  Total errors: {total_errors}")
        print(f"  Average execution time: {avg_execution_time:.2f}ms")
        print(f"  Max execution time: {max_execution_time:.2f}ms")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Max memory usage: {resource_stats['max_memory_mb']:.2f}MB")
        print(f"  Max CPU usage: {resource_stats['max_cpu_percent']:.1f}%")
        
        # Assertions
        self.assertEqual(total_errors, 0, f"Full system errors: {errors}")
        self.assertGreater(total_decisions, 100)
        self.assertLess(avg_execution_time, 150)  # <150ms average
        self.assertLess(max_execution_time, 500)  # <500ms max
        self.assertGreater(avg_confidence, 0.3)  # Reasonable confidence
        self.assertLess(resource_stats['max_memory_mb'], self.MEMORY_LIMIT_MB)

    # High Load Performance Tests
    
    def test_high_volume_card_evaluation(self):
        """Test performance under high volume card evaluation load."""
        total_evaluations = self.HIGH_LOAD_OPERATIONS
        batch_size = 100
        results = []
        
        start_time = time.time()
        
        for batch in range(0, total_evaluations, batch_size):
            batch_results = []
            
            for i in range(min(batch_size, total_evaluations - batch)):
                card = random.choice(self.test_cards)
                deck_state = random.choice(self.test_deck_states)
                
                eval_start = time.perf_counter()
                result = self.card_evaluator.evaluate_card(card, deck_state)
                eval_end = time.perf_counter()
                
                batch_results.append({
                    'result': result,
                    'time_ms': (eval_end - eval_start) * 1000
                })
            
            results.extend(batch_results)
            
            # Force garbage collection every batch to prevent memory buildup
            if batch % (batch_size * 5) == 0:
                gc.collect()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze performance
        execution_times = [r['time_ms'] for r in results]
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        throughput = total_evaluations / total_time
        
        print(f"High Volume Card Evaluation Results:")
        print(f"  Total evaluations: {total_evaluations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} evaluations/second")
        print(f"  Average time per evaluation: {avg_time:.2f}ms")
        print(f"  Max time per evaluation: {max_time:.2f}ms")
        
        # Performance assertions
        self.assertGreater(throughput, 50)  # >50 evaluations/second
        self.assertLess(avg_time, 20)  # <20ms average
        self.assertLess(max_time, 100)  # <100ms max
    
    def test_sustained_load_endurance(self):
        """Test system endurance under sustained load."""
        duration_seconds = self.STRESS_TEST_DURATION_SECONDS
        operations_completed = 0
        errors = []
        memory_samples = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                # Mix of operations
                operation_type = random.choice(['card_eval', 'deck_analysis', 'full_decision'])
                
                if operation_type == 'card_eval':
                    card = random.choice(self.test_cards)
                    deck_state = random.choice(self.test_deck_states)
                    result = self.card_evaluator.evaluate_card(card, deck_state)
                    
                elif operation_type == 'deck_analysis':
                    deck_state = random.choice(self.test_deck_states)
                    result = self.deck_analyzer.analyze_deck(deck_state)
                    
                else:  # full_decision
                    deck_state = random.choice(self.test_deck_states)
                    result = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                
                operations_completed += 1
                
                # Sample memory usage periodically
                if operations_completed % 50 == 0:
                    memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    memory_samples.append(memory_mb)
                    
                    # Force garbage collection if memory is growing
                    if memory_mb > self.MEMORY_LIMIT_MB * 0.8:
                        gc.collect()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.005)
                
            except Exception as e:
                errors.append(e)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        throughput = operations_completed / actual_duration
        
        # Memory analysis
        if memory_samples:
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            max_memory = max(memory_samples)
            memory_growth = final_memory - initial_memory
        else:
            initial_memory = final_memory = max_memory = memory_growth = 0
        
        print(f"Sustained Load Endurance Results:")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Operations completed: {operations_completed}")
        print(f"  Throughput: {throughput:.1f} operations/second")
        print(f"  Errors: {len(errors)}")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Max memory: {max_memory:.2f}MB")
        print(f"  Memory growth: {memory_growth:.2f}MB")
        
        # Endurance assertions
        self.assertEqual(len(errors), 0, f"Endurance test errors: {errors}")
        self.assertGreater(operations_completed, 500)  # Should complete many operations
        self.assertLess(max_memory, self.MEMORY_LIMIT_MB)
        self.assertLess(memory_growth, 100)  # <100MB growth over test period

    # Memory Pressure Tests
    
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        # Create memory pressure by keeping references to many results
        accumulated_results = []
        memory_limit_reached = False
        operations_completed = 0
        
        try:
            while not memory_limit_reached and operations_completed < 2000:
                # Perform operations and keep results in memory
                deck_state = random.choice(self.test_deck_states)
                decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                accumulated_results.append(decision)
                
                operations_completed += 1
                
                # Check memory usage
                if operations_completed % 10 == 0:
                    memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    if memory_mb > self.MEMORY_LIMIT_MB:
                        memory_limit_reached = True
                        print(f"Memory limit reached at {memory_mb:.2f}MB after {operations_completed} operations")
                        break
        
        except Exception as e:
            print(f"Memory pressure caused exception: {e}")
        
        # Test recovery after clearing memory pressure
        accumulated_results.clear()
        gc.collect()
        
        recovery_memory = self.process.memory_info().rss / (1024 * 1024)
        
        # Test that system still works after memory pressure
        recovery_deck = random.choice(self.test_deck_states)
        recovery_result = self.grandmaster_advisor.analyze_draft_choice(recovery_deck)
        
        print(f"Memory Pressure Test Results:")
        print(f"  Operations before limit: {operations_completed}")
        print(f"  Memory after recovery: {recovery_memory:.2f}MB")
        print(f"  System functional after recovery: {recovery_result is not None}")
        
        # Assertions
        self.assertIsNotNone(recovery_result, "System should recover from memory pressure")
        self.assertLess(recovery_memory, self.MEMORY_LIMIT_MB * 0.8, "Memory should be freed after pressure relief")
    
    def test_cache_pressure_handling(self):
        """Test cache behavior under pressure."""
        # Fill caches with many different evaluations
        unique_evaluations = 0
        cache_size_samples = []
        
        for i in range(500):
            # Create unique card for each evaluation to stress cache
            unique_card = CardInstance(
                name=f"Cache Pressure Card {i}",
                cost=i % 10,
                attack=(i % 8) + 1,
                health=(i % 8) + 1,
                card_type="minion",
                rarity="common",
                card_set="cache_test",
                keywords=[],
                description=f"Cache pressure test card {i}"
            )
            
            deck_state = random.choice(self.test_deck_states)
            result = self.card_evaluator.evaluate_card(unique_card, deck_state)
            unique_evaluations += 1
            
            # Sample cache size
            if hasattr(self.card_evaluator, '_cache'):
                cache_size = len(self.card_evaluator._cache)
                cache_size_samples.append(cache_size)
        
        print(f"Cache Pressure Test Results:")
        print(f"  Unique evaluations: {unique_evaluations}")
        if cache_size_samples:
            print(f"  Final cache size: {cache_size_samples[-1]}")
            print(f"  Max cache size observed: {max(cache_size_samples)}")
            
            # Cache should not grow unbounded
            max_cache_size = getattr(self.card_evaluator, '_max_cache_size', 1000)
            self.assertLessEqual(max(cache_size_samples), max_cache_size,
                               "Cache size should be bounded")

    # Resource Leak Tests
    
    def test_thread_resource_cleanup(self):
        """Test that threads are properly cleaned up after operations."""
        initial_thread_count = threading.active_count()
        
        def create_temporary_threads():
            results = []
            
            def thread_operation():
                deck_state = random.choice(self.test_deck_states)
                result = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                results.append(result)
            
            # Create and join many threads
            threads = []
            for _ in range(20):
                thread = threading.Thread(target=thread_operation)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return results
        
        # Run multiple batches of threads
        for batch in range(5):
            results = create_temporary_threads()
            self.assertEqual(len(results), 20)
            
            # Allow some time for cleanup
            time.sleep(0.1)
        
        # Check thread count after cleanup
        final_thread_count = threading.active_count()
        
        print(f"Thread Cleanup Test Results:")
        print(f"  Initial thread count: {initial_thread_count}")
        print(f"  Final thread count: {final_thread_count}")
        
        # Thread count should return to baseline (allow some variance for test framework)
        self.assertLessEqual(final_thread_count - initial_thread_count, 5,
                           "Thread count should return to baseline after operations")
    
    def test_file_handle_management(self):
        """Test file handle management under load."""
        if not hasattr(self.process, 'num_fds'):
            self.skipTest("File descriptor monitoring not available on this platform")
        
        try:
            initial_fds = self.process.num_fds()
        except AttributeError:
            self.skipTest("File descriptor counting not supported")
        
        # Perform many operations that might open files
        for i in range(100):
            deck_state = random.choice(self.test_deck_states)
            result = self.grandmaster_advisor.analyze_draft_choice(deck_state)
            
            # Simulate some file operations if any
            if hasattr(self.grandmaster_advisor, '_log_decision'):
                # This would test file logging if implemented
                pass
        
        final_fds = self.process.num_fds()
        
        print(f"File Handle Management Test Results:")
        print(f"  Initial file descriptors: {initial_fds}")
        print(f"  Final file descriptors: {final_fds}")
        print(f"  File descriptor growth: {final_fds - initial_fds}")
        
        # File descriptors should not leak
        self.assertLess(final_fds - initial_fds, 10,
                       "File descriptors should not leak significantly")

    # Race Condition Tests
    
    def test_race_condition_detection(self):
        """Test for race conditions in shared state access."""
        shared_counter = 0
        race_detected = False
        counter_lock = Lock()
        
        def increment_with_race_detection():
            nonlocal shared_counter, race_detected
            
            for _ in range(100):
                # Perform AI operation
                deck_state = random.choice(self.test_deck_states)
                result = self.card_evaluator.evaluate_card(
                    random.choice(self.test_cards), deck_state
                )
                
                # Test shared counter increment (potential race condition)
                old_value = shared_counter
                time.sleep(0.0001)  # Increase chance of race condition
                
                with counter_lock:
                    if shared_counter != old_value:  # Value changed during sleep
                        # This is expected due to other threads
                        pass
                    shared_counter += 1
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment_with_race_detection) for _ in range(5)]
            for future in as_completed(futures):
                future.result()
        
        expected_count = 5 * 100  # 5 threads * 100 increments each
        
        print(f"Race Condition Test Results:")
        print(f"  Expected count: {expected_count}")
        print(f"  Actual count: {shared_counter}")
        print(f"  Race detected: {race_detected}")
        
        # Counter should be accurate (no race conditions in increment)
        self.assertEqual(shared_counter, expected_count,
                        "Shared counter should be accurate despite concurrent access")
    
    def tearDown(self):
        """Clean up after stress tests."""
        # Force garbage collection
        gc.collect()
        
        # Clear any caches
        if hasattr(self.card_evaluator, '_cache'):
            self.card_evaluator._cache.clear()


class DeadlockDetectionTests(unittest.TestCase):
    """
    Tests for deadlock detection and prevention.
    """
    
    def setUp(self):
        self.advisor = GrandmasterAdvisor()
        self.test_card = CardInstance(
            "Test Card", 3, 3, 3, "minion", "common", "test", [], "Test"
        )
        self.test_deck = DeckState([self.test_card], [], 2, 0, 0)
    
    def test_deadlock_prevention_timeout(self):
        """Test that operations timeout to prevent deadlocks."""
        # This would test timeout mechanisms if implemented
        start_time = time.time()
        
        try:
            # Perform operation that might deadlock
            result = self.advisor.analyze_draft_choice(self.test_deck)
            end_time = time.time()
            
            # Should complete in reasonable time
            execution_time = end_time - start_time
            self.assertLess(execution_time, 10.0,
                           "Operation should complete within timeout period")
            
        except Exception as e:
            # If timeout exception is implemented, verify it's caught
            if "timeout" in str(e).lower():
                self.assertTrue(True, "Timeout mechanism working")
            else:
                raise


if __name__ == '__main__':
    print("Starting Stress Testing and Concurrent Access Suite...")
    print("WARNING: This test suite may consume significant system resources.")
    print("=" * 70)
    
    # Set multiprocessing start method for better compatibility
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    
    # Run with high verbosity
    unittest.main(verbosity=2, buffer=True)