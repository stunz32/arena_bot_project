"""
Memory Usage Testing for Complete System with Dual Data Streams

Comprehensive tests for memory usage patterns, optimization, and leak detection
in the complete Arena Bot system with dual HSReplay data streams (heroes + cards).
"""

import unittest
import sys
import gc
import time
import threading
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tracemalloc
from typing import Dict, List, Tuple, Any
from datetime import datetime
import weakref

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.intelligent_cache_manager import get_cache_manager
from arena_bot.ai_v2.resource_manager import get_resource_manager
from arena_bot.data_sourcing.hsreplay_scraper import HSReplayDataScraper
from arena_bot.data.cards_json_loader import CardsJsonLoader
from arena_bot.ai_v2.data_models import DeckState


class MemoryTracker:
    """Utility class for tracking memory usage during tests."""
    
    def __init__(self, name: str = "test"):
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None
        self.tracemalloc_snapshot = None
    
    def start(self):
        """Start memory tracking."""
        gc.collect()  # Clean up before measurement
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
        # Start tracemalloc for detailed tracking
        tracemalloc.start()
        return self
    
    def update_peak(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop(self):
        """Stop memory tracking and return results."""
        gc.collect()  # Clean up before final measurement
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Get tracemalloc snapshot
        self.tracemalloc_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        return {
            'start_mb': self.start_memory,
            'end_mb': self.end_memory,
            'peak_mb': self.peak_memory,
            'net_increase_mb': self.end_memory - self.start_memory,
            'peak_increase_mb': self.peak_memory - self.start_memory,
            'snapshot': self.tracemalloc_snapshot
        }
    
    def get_top_allocations(self, limit: int = 10) -> List[str]:
        """Get top memory allocations from tracemalloc."""
        if not self.tracemalloc_snapshot:
            return []
        
        top_stats = self.tracemalloc_snapshot.statistics('lineno')
        return [str(stat) for stat in top_stats[:limit]]


class TestMemoryUsage(unittest.TestCase):
    """Memory usage tests for dual data stream system."""
    
    def setUp(self):
        """Set up memory usage test fixtures."""
        # Memory usage thresholds (in MB)
        self.INITIAL_LOAD_MEMORY_LIMIT = 100    # Initial system load
        self.SINGLE_OPERATION_MEMORY_LIMIT = 50  # Single operation overhead
        self.SUSTAINED_OPERATION_MEMORY_LIMIT = 200  # Multiple operations
        self.MEMORY_LEAK_THRESHOLD = 10         # Memory not released after operations
        
        # Create comprehensive test data for memory testing
        self.memory_test_cards_data = self._create_memory_test_cards_data()
        self.memory_test_hero_winrates = self._create_memory_test_hero_winrates()
        self.memory_test_hsreplay_data = self._create_memory_test_hsreplay_data()
        
        # Track object references for leak detection
        self.tracked_objects = []
    
    def _create_memory_test_cards_data(self):
        """Create cards data for memory testing with realistic size."""
        cards_data = {}
        
        # Create heroes
        hero_classes = ["WARRIOR", "MAGE", "HUNTER", "PALADIN", "PRIEST", 
                       "ROGUE", "SHAMAN", "WARLOCK", "DRUID", "DEMONHUNTER"]
        for i, hero_class in enumerate(hero_classes, 1):
            cards_data[f"HERO_{i:02d}"] = {
                "id": f"HERO_{i:02d}",
                "name": f"{hero_class.title()} Hero",
                "playerClass": hero_class,
                "type": "HERO",
                "dbfId": 800 + i,
                "description": f"The {hero_class.lower()} hero with class-specific abilities."
            }
        
        # Create many cards to simulate realistic memory usage
        for i in range(1, 1001):  # 1000 cards
            class_index = i % len(hero_classes)
            player_class = "NEUTRAL" if i % 10 == 0 else hero_classes[class_index]
            
            card_data = {
                "id": f"CARD_{i:04d}",
                "name": f"Test Card {i}",
                "playerClass": player_class,
                "type": "MINION" if i % 3 == 0 else "SPELL",
                "cost": (i % 10) + 1,
                "dbfId": 10000 + i,
                "description": f"A test card with ID {i} for memory testing purposes. This description is intentionally long to simulate realistic card data sizes.",
                "rarity": ["COMMON", "RARE", "EPIC", "LEGENDARY"][i % 4],
                "set": f"TEST_SET_{(i // 100) + 1}"
            }
            
            if card_data["type"] == "MINION":
                card_data["attack"] = (i % 12) + 1
                card_data["health"] = (i % 15) + 1
                card_data["mechanics"] = [f"MECHANIC_{j}" for j in range(i % 3)]
            
            cards_data[card_data["id"]] = card_data
        
        return cards_data
    
    def _create_memory_test_hero_winrates(self):
        """Create hero winrates for memory testing."""
        return {
            "WARRIOR": 0.5580,
            "MAGE": 0.5450,
            "HUNTER": 0.4920,
            "PALADIN": 0.5320,
            "PRIEST": 0.4780,
            "ROGUE": 0.5180,
            "SHAMAN": 0.4650,
            "WARLOCK": 0.5580,
            "DRUID": 0.4850,
            "DEMONHUNTER": 0.5050
        }
    
    def _create_memory_test_hsreplay_data(self):
        """Create HSReplay data for memory testing with realistic size."""
        hsreplay_data = {}
        
        # Create data for many cards
        for i in range(1, 1001):
            card_id = f"CARD_{i:04d}"
            hsreplay_data[card_id] = {
                "overall_winrate": 0.45 + (i % 20) * 0.005,
                "play_rate": 0.05 + (i % 30) * 0.01,
                "pick_rate": 0.10 + (i % 25) * 0.02,
                "games_played": 1000 + (i * 50),
                "times_picked": 500 + (i * 25),
                "deck_winrate_data": {
                    f"archetype_{j}": 0.40 + (j * 0.05) for j in range(5)
                },
                "popularity_by_turn": {
                    f"turn_{j}": (i + j) % 100 for j in range(15)
                },
                "meta_stats": {
                    "last_updated": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "confidence_interval": [0.45 + (i % 10) * 0.01, 0.55 + (i % 10) * 0.01],
                    "sample_size": 5000 + (i * 100)
                }
            }
        
        return hsreplay_data
    
    def test_initial_system_memory_usage(self):
        """Test memory usage during initial system loading."""
        tracker = MemoryTracker("initial_load").start()
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            # Initialize all major components
            tracker.update_peak()
            cards_loader = CardsJsonLoader()
            
            tracker.update_peak()
            hero_advisor = HeroSelectionAdvisor()
            
            tracker.update_peak()
            card_evaluator = CardEvaluationEngine()
            
            tracker.update_peak()
            grandmaster_advisor = GrandmasterAdvisor()
            
            tracker.update_peak()
            
            # Store weak references for leak detection
            self.tracked_objects = [
                weakref.ref(cards_loader),
                weakref.ref(hero_advisor),
                weakref.ref(card_evaluator),
                weakref.ref(grandmaster_advisor)
            ]
        
        results = tracker.stop()
        
        # Check memory usage limits
        self.assertLess(results['peak_increase_mb'], self.INITIAL_LOAD_MEMORY_LIMIT,
                       f"Initial load used {results['peak_increase_mb']:.1f}MB, limit is {self.INITIAL_LOAD_MEMORY_LIMIT}MB")
        
        # Log memory usage for analysis
        print(f"\nInitial load memory usage:")
        print(f"  Start: {results['start_mb']:.1f}MB")
        print(f"  Peak: {results['peak_mb']:.1f}MB")
        print(f"  End: {results['end_mb']:.1f}MB")
        print(f"  Net increase: {results['net_increase_mb']:.1f}MB")
    
    def test_single_operation_memory_usage(self):
        """Test memory usage for single operations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            # Pre-initialize components
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Test hero selection memory usage
            tracker = MemoryTracker("hero_selection").start()
            
            recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
            tracker.update_peak()
            
            hero_results = tracker.stop()
            
            # Test card evaluation memory usage
            tracker = MemoryTracker("card_evaluation").start()
            
            deck_state = DeckState(
                cards_drafted=[],
                mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                synergy_groups={},
                hero_class="WARRIOR"
            )
            
            card_choices = [
                self.memory_test_cards_data["CARD_0001"],
                self.memory_test_cards_data["CARD_0002"],
                self.memory_test_cards_data["CARD_0003"]
            ]
            
            card_recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
            tracker.update_peak()
            
            card_results = tracker.stop()
            
            # Check memory usage for single operations
            self.assertLess(hero_results['peak_increase_mb'], self.SINGLE_OPERATION_MEMORY_LIMIT,
                           f"Hero selection used {hero_results['peak_increase_mb']:.1f}MB, limit is {self.SINGLE_OPERATION_MEMORY_LIMIT}MB")
            
            self.assertLess(card_results['peak_increase_mb'], self.SINGLE_OPERATION_MEMORY_LIMIT,
                           f"Card evaluation used {card_results['peak_increase_mb']:.1f}MB, limit is {self.SINGLE_OPERATION_MEMORY_LIMIT}MB")
            
            # Verify operations completed successfully
            self.assertEqual(len(recommendations), 3)
            self.assertIsNotNone(card_recommendation)
    
    def test_sustained_operations_memory_usage(self):
        """Test memory usage during sustained operations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            # Initialize components
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            
            tracker = MemoryTracker("sustained_operations").start()
            
            # Perform many operations to test sustained usage
            for i in range(50):  # 50 operations
                # Hero selection
                hero_choices = [
                    list(self.memory_test_hero_winrates.keys())[i % 3],
                    list(self.memory_test_hero_winrates.keys())[(i + 1) % 3],
                    list(self.memory_test_hero_winrates.keys())[(i + 2) % 3]
                ]
                recommendations = hero_advisor.recommend_hero(hero_choices)
                
                # Card evaluation
                deck_state = DeckState(
                    cards_drafted=[],
                    mana_curve={1: i % 5, 2: (i + 1) % 5, 3: (i + 2) % 5, 4: 0, 5: 0, 6: 0, 7: 0},
                    archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                    synergy_groups={},
                    hero_class=hero_choices[0]
                )
                
                card_choices = [
                    self.memory_test_cards_data[f"CARD_{(i * 3 + 1):04d}"],
                    self.memory_test_cards_data[f"CARD_{(i * 3 + 2):04d}"],
                    self.memory_test_cards_data[f"CARD_{(i * 3 + 3):04d}"]
                ]
                
                card_recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, hero_choices[0])
                
                # Update peak memory tracking
                tracker.update_peak()
                
                # Verify operations work
                self.assertEqual(len(recommendations), 3)
                self.assertIsNotNone(card_recommendation)
                
                # Periodic garbage collection
                if i % 10 == 9:
                    gc.collect()
            
            results = tracker.stop()
            
            # Check sustained operation memory usage
            self.assertLess(results['peak_increase_mb'], self.SUSTAINED_OPERATION_MEMORY_LIMIT,
                           f"Sustained operations used {results['peak_increase_mb']:.1f}MB, limit is {self.SUSTAINED_OPERATION_MEMORY_LIMIT}MB")
            
            print(f"\nSustained operations memory usage:")
            print(f"  Peak increase: {results['peak_increase_mb']:.1f}MB")
            print(f"  Net increase: {results['net_increase_mb']:.1f}MB")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            # Initialize once
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            
            # Measure baseline memory
            gc.collect()
            baseline_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Perform operations in batches and measure memory growth
            batch_size = 20
            memory_measurements = []
            
            for batch in range(5):  # 5 batches of 20 operations each
                for i in range(batch_size):
                    # Create new objects for each operation
                    hero_choices = ["WARRIOR", "MAGE", "HUNTER"]
                    recommendations = hero_advisor.recommend_hero(hero_choices)
                    
                    deck_state = DeckState(
                        cards_drafted=[],
                        mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                        archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                        synergy_groups={},
                        hero_class="WARRIOR"
                    )
                    
                    card_choices = [
                        self.memory_test_cards_data["CARD_0001"],
                        self.memory_test_cards_data["CARD_0002"],
                        self.memory_test_cards_data["CARD_0003"]
                    ]
                    
                    card_recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
                    
                    # Clear local references
                    del recommendations, deck_state, card_choices, card_recommendation
                
                # Force garbage collection
                gc.collect()
                
                # Measure memory after batch
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_increase = current_memory - baseline_memory
                memory_measurements.append(memory_increase)
            
            # Analyze memory growth trend
            if len(memory_measurements) >= 3:
                # Check if memory keeps growing (indicating a leak)
                final_increase = memory_measurements[-1]
                
                self.assertLess(final_increase, self.MEMORY_LEAK_THRESHOLD,
                               f"Potential memory leak detected: {final_increase:.1f}MB increase after {len(memory_measurements) * batch_size} operations")
                
                # Check for consistent growth
                growth_rate = (memory_measurements[-1] - memory_measurements[0]) / len(memory_measurements)
                self.assertLess(abs(growth_rate), 2.0, 
                               f"Memory growth rate {growth_rate:.2f}MB per batch suggests leak")
            
            print(f"\nMemory leak test results:")
            print(f"  Baseline: {baseline_memory:.1f}MB")
            print(f"  Final increase: {memory_measurements[-1]:.1f}MB")
            print(f"  Memory per batch: {memory_measurements}")
    
    def test_cache_memory_management(self):
        """Test memory management of caching systems."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            tracker = MemoryTracker("cache_management").start()
            
            # Initialize cache manager
            cache_manager = get_cache_manager()
            tracker.update_peak()
            
            # Fill cache with data
            for i in range(100):
                cache_key = f"test_key_{i}"
                cache_data = {
                    "hero_data": self.memory_test_hero_winrates,
                    "card_data": {f"CARD_{j:04d}": self.memory_test_hsreplay_data.get(f"CARD_{j:04d}", {}) 
                                for j in range(1, 50)},  # 50 cards worth of data
                    "timestamp": time.time(),
                    "metadata": {"test": True, "iteration": i}
                }
                cache_manager.set(cache_key, cache_data, ttl=300)
                
                if i % 20 == 19:
                    tracker.update_peak()
            
            # Test cache cleanup
            cache_manager.cleanup_expired()
            tracker.update_peak()
            
            # Test cache size limits
            cache_stats = cache_manager.get_cache_stats()
            tracker.update_peak()
            
            results = tracker.stop()
            
            # Cache should not use excessive memory
            self.assertLess(results['peak_increase_mb'], 150,  # Allow reasonable cache memory
                           f"Cache management used {results['peak_increase_mb']:.1f}MB")
            
            # Cache should have reasonable size
            self.assertIn('total_size_mb', cache_stats)
            self.assertLess(cache_stats['total_size_mb'], 100,
                           f"Cache size {cache_stats['total_size_mb']:.1f}MB exceeds limit")
    
    def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent operations."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            tracker = MemoryTracker("concurrent_operations").start()
            
            # Initialize shared components
            hero_advisor = HeroSelectionAdvisor()
            grandmaster_advisor = GrandmasterAdvisor()
            tracker.update_peak()
            
            results = []
            errors = []
            
            def worker_thread(thread_id):
                try:
                    for i in range(10):  # 10 operations per thread
                        # Hero selection
                        hero_choices = ["WARRIOR", "MAGE", "HUNTER"]
                        recommendations = hero_advisor.recommend_hero(hero_choices)
                        
                        # Card evaluation
                        deck_state = DeckState(
                            cards_drafted=[],
                            mana_curve={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                            archetype_leanings={'aggro': 0.33, 'midrange': 0.33, 'control': 0.34},
                            synergy_groups={},
                            hero_class="WARRIOR"
                        )
                        
                        card_choices = [
                            self.memory_test_cards_data["CARD_0001"],
                            self.memory_test_cards_data["CARD_0002"],
                            self.memory_test_cards_data["CARD_0003"]
                        ]
                        
                        card_recommendation = grandmaster_advisor.get_recommendation(card_choices, deck_state, "WARRIOR")
                        
                        results.append((thread_id, i, len(recommendations), card_recommendation is not None))
                        
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Run concurrent threads
            threads = []
            for i in range(5):  # 5 concurrent threads
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Monitor memory during concurrent execution
            for _ in range(10):
                time.sleep(0.1)
                tracker.update_peak()
            
            # Wait for threads to complete
            for thread in threads:
                thread.join()
            
            final_results = tracker.stop()
            
            # Check concurrent memory usage
            self.assertLess(final_results['peak_increase_mb'], 300,  # Allow for concurrent overhead
                           f"Concurrent operations used {final_results['peak_increase_mb']:.1f}MB")
            
            # Verify all operations completed successfully
            self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
            self.assertEqual(len(results), 50)  # 5 threads * 10 operations
            
            print(f"\nConcurrent operations memory usage:")
            print(f"  Peak increase: {final_results['peak_increase_mb']:.1f}MB")
            print(f"  Operations completed: {len(results)}")
    
    def test_large_dataset_memory_scaling(self):
        """Test memory usage scaling with large datasets."""
        # Create progressively larger datasets
        dataset_sizes = [100, 500, 1000]
        memory_usage = []
        
        for size in dataset_sizes:
            # Create dataset of specified size
            large_cards_data = {}
            large_hsreplay_data = {}
            
            for i in range(1, size + 1):
                card_id = f"LARGE_{i:04d}"
                large_cards_data[card_id] = {
                    "id": card_id,
                    "name": f"Large Card {i}",
                    "playerClass": "NEUTRAL",
                    "type": "MINION",
                    "cost": (i % 10) + 1,
                    "attack": (i % 12) + 1,
                    "health": (i % 15) + 1,
                    "dbfId": 20000 + i,
                    "description": f"Large dataset card {i} with extended description for memory testing." * 3
                }
                
                large_hsreplay_data[card_id] = {
                    "overall_winrate": 0.45 + (i % 20) * 0.005,
                    "play_rate": 0.05 + (i % 30) * 0.01,
                    "extensive_stats": {f"stat_{j}": i + j for j in range(50)}  # Many stats
                }
            
            # Add base data
            large_cards_data.update(self.memory_test_cards_data)
            large_hsreplay_data.update(self.memory_test_hsreplay_data)
            
            with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
                 patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
                 patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
                
                mock_load_cards.return_value = large_cards_data
                mock_hsreplay.return_value = large_hsreplay_data
                mock_hero_winrates.return_value = self.memory_test_hero_winrates
                
                tracker = MemoryTracker(f"dataset_size_{size}").start()
                
                # Initialize system with large dataset
                cards_loader = CardsJsonLoader()
                tracker.update_peak()
                
                hero_advisor = HeroSelectionAdvisor()
                tracker.update_peak()
                
                card_evaluator = CardEvaluationEngine()
                tracker.update_peak()
                
                # Perform some operations
                recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                tracker.update_peak()
                
                results = tracker.stop()
                memory_usage.append((size, results['peak_increase_mb']))
                
                # Clean up for next iteration
                del cards_loader, hero_advisor, card_evaluator, recommendations
                gc.collect()
        
        # Analyze memory scaling
        if len(memory_usage) >= 2:
            # Memory should scale sub-linearly with dataset size
            small_size, small_memory = memory_usage[0]
            large_size, large_memory = memory_usage[-1]
            
            size_ratio = large_size / small_size
            memory_ratio = large_memory / small_memory if small_memory > 0 else float('inf')
            
            # Memory growth should be less than proportional to data growth
            self.assertLess(memory_ratio, size_ratio * 1.5,
                           f"Memory scaling poor: {memory_ratio:.2f}x memory for {size_ratio:.2f}x data")
            
            print(f"\nMemory scaling results:")
            for size, memory in memory_usage:
                print(f"  {size} cards: {memory:.1f}MB")
    
    def test_garbage_collection_effectiveness(self):
        """Test garbage collection effectiveness."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = self.memory_test_cards_data
            mock_hsreplay.return_value = self.memory_test_hsreplay_data
            mock_hero_winrates.return_value = self.memory_test_hero_winrates
            
            # Measure memory before operations
            gc.collect()
            baseline_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Create many temporary objects
            temp_objects = []
            for i in range(100):
                hero_advisor = HeroSelectionAdvisor()
                recommendations = hero_advisor.recommend_hero(["WARRIOR", "MAGE", "HUNTER"])
                temp_objects.append((hero_advisor, recommendations))
            
            # Measure memory with objects
            memory_with_objects = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Clear references
            del temp_objects
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory after cleanup
            memory_after_gc = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Calculate memory recovery
            memory_increase = memory_with_objects - baseline_memory
            memory_recovered = memory_with_objects - memory_after_gc
            recovery_rate = memory_recovered / memory_increase if memory_increase > 0 else 1.0
            
            # Should recover at least 70% of allocated memory
            self.assertGreater(recovery_rate, 0.70,
                             f"Garbage collection only recovered {recovery_rate:.1%} of allocated memory")
            
            print(f"\nGarbage collection effectiveness:")
            print(f"  Baseline: {baseline_memory:.1f}MB")
            print(f"  With objects: {memory_with_objects:.1f}MB")
            print(f"  After GC: {memory_after_gc:.1f}MB")
            print(f"  Recovery rate: {recovery_rate:.1%}")


class TestMemoryUsageEdgeCases(unittest.TestCase):
    """Test edge cases in memory usage."""
    
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
    
    def test_memory_usage_with_empty_data(self):
        """Test memory usage when data sets are empty."""
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_underground_arena_stats') as mock_hsreplay, \
             patch('arena_bot.data_sourcing.hsreplay_scraper.HSReplayDataScraper.get_hero_winrates') as mock_hero_winrates:
            
            mock_load_cards.return_value = {}  # Empty data
            mock_hsreplay.return_value = {}
            mock_hero_winrates.return_value = {}
            
            tracker = MemoryTracker("empty_data").start()
            
            # Should handle empty data gracefully without excessive memory usage
            try:
                hero_advisor = HeroSelectionAdvisor()
                recommendations = hero_advisor.recommend_hero(["WARRIOR"])
                tracker.update_peak()
            except Exception:
                # Failure is acceptable, but should not leak memory
                pass
            
            results = tracker.stop()
            
            # Even with empty data, should not use excessive memory
            self.assertLess(results['peak_increase_mb'], 50,
                           f"Empty data handling used {results['peak_increase_mb']:.1f}MB")
    
    def test_memory_usage_with_malformed_data(self):
        """Test memory usage when data is malformed."""
        malformed_data = {
            "INVALID_CARD": {
                "id": "INVALID_CARD",
                # Missing required fields
                "malformed_field": "invalid_value"
            }
        }
        
        with patch('arena_bot.data.cards_json_loader.CardsJsonLoader._load_cards_data') as mock_load_cards:
            mock_load_cards.return_value = malformed_data
            
            tracker = MemoryTracker("malformed_data").start()
            
            # Should handle malformed data without memory issues
            try:
                cards_loader = CardsJsonLoader()
                tracker.update_peak()
            except Exception:
                # Failure is acceptable
                pass
            
            results = tracker.stop()
            
            # Should not use excessive memory even with malformed data
            self.assertLess(results['peak_increase_mb'], 30,
                           f"Malformed data handling used {results['peak_increase_mb']:.1f}MB")


def run_memory_usage_tests():
    """Run all memory usage tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestMemoryUsage))
    test_suite.addTest(unittest.makeSuite(TestMemoryUsageEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_memory_usage_tests()
    
    if success:
        print("\n✅ All memory usage tests passed!")
    else:
        print("\n❌ Some memory usage tests failed!")
        sys.exit(1)