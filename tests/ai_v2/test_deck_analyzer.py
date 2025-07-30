"""
Comprehensive test suite for StrategicDeckAnalyzer - Phase 1
Tests archetype validation, strategic gap analysis, immutable state architecture, and hardening features.
"""

import unittest
import threading
import time
import copy
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.deck_analyzer import StrategicDeckAnalyzer
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState, DraftChoice
from arena_bot.ai_v2.exceptions import AIEngineError, AnalysisError


class TestStrategicDeckAnalyzer(unittest.TestCase):
    """
    Comprehensive test suite for StrategicDeckAnalyzer with hardening validation.
    """
    
    def setUp(self):
        """Set up test environment with clean analyzer instance."""
        self.analyzer = StrategicDeckAnalyzer()
        self.sample_cards = self._create_sample_cards()
        self.sample_deck_states = self._create_sample_deck_states()
    
    def _create_sample_cards(self):
        """Create sample card instances for different archetypes."""
        return {
            'aggressive': [
                CardInstance("Leper Gnome", 1, 2, 1, "minion", "common", "classic", ["charge"], "Aggressive 1-drop"),
                CardInstance("Knife Juggler", 2, 3, 2, "minion", "rare", "classic", [], "Aggressive 2-drop"),
                CardInstance("Wolfrider", 3, 3, 1, "minion", "common", "classic", ["charge"], "Aggressive charge minion")
            ],
            'control': [
                CardInstance("Flamestrike", 7, None, None, "spell", "epic", "classic", ["spell", "damage"], "Board clear"),
                CardInstance("Water Elemental", 4, 3, 6, "minion", "common", "classic", ["freeze"], "Control minion"),
                CardInstance("Archmage Antonidas", 7, 5, 7, "minion", "legendary", "classic", [], "Control win condition")
            ],
            'value': [
                CardInstance("Azure Drake", 5, 4, 4, "minion", "rare", "classic", ["spell_damage", "card_draw"], "Value minion"),
                CardInstance("Harvest Golem", 3, 2, 3, "minion", "common", "classic", ["deathrattle"], "Sticky value"),
                CardInstance("Cairne Bloodhoof", 6, 4, 5, "minion", "legendary", "classic", ["deathrattle"], "High value legendary")
            ],
            'tempo': [
                CardInstance("Youthful Brewmaster", 2, 3, 2, "minion", "common", "classic", ["battlecry"], "Tempo play"),
                CardInstance("Dark Iron Dwarf", 4, 4, 4, "minion", "common", "classic", ["battlecry"], "Tempo buff"),
                CardInstance("Defender of Argus", 4, 2, 3, "minion", "rare", "classic", ["battlecry", "taunt"], "Tempo/defensive")
            ]
        }
    
    def _create_sample_deck_states(self):
        """Create sample deck states for different scenarios."""
        return {
            'early_aggressive': DeckState(
                drafted_cards=self.sample_cards['aggressive'][:2],
                available_choices=[
                    CardOption(self.sample_cards['aggressive'][2], 0.9),
                    CardOption(self.sample_cards['control'][0], 0.6),
                    CardOption(self.sample_cards['value'][0], 0.7)
                ],
                draft_pick_number=5,
                wins=0,
                losses=0
            ),
            'mid_control': DeckState(
                drafted_cards=self.sample_cards['control'][:1] + self.sample_cards['value'][:1],
                available_choices=[
                    CardOption(self.sample_cards['control'][1], 0.8),
                    CardOption(self.sample_cards['aggressive'][0], 0.5),
                    CardOption(self.sample_cards['tempo'][0], 0.6)
                ],
                draft_pick_number=15,
                wins=2,
                losses=1
            ),
            'late_mixed': DeckState(
                drafted_cards=(self.sample_cards['aggressive'] + 
                             self.sample_cards['control'][:1] + 
                             self.sample_cards['value'][:2]),
                available_choices=[
                    CardOption(self.sample_cards['tempo'][0], 0.7),
                    CardOption(self.sample_cards['control'][2], 0.9),
                    CardOption(self.sample_cards['value'][2], 0.8)
                ],
                draft_pick_number=25,
                wins=1,
                losses=2
            )
        }

    # Core Functionality Tests
    
    def test_analyze_deck_basic_functionality(self):
        """Test basic deck analysis returns valid results."""
        deck_state = self.sample_deck_states['early_aggressive']
        result = self.analyzer.analyze_deck(deck_state)
        
        # Validate analysis structure
        self.assertIsInstance(result, dict)
        required_keys = [
            'primary_archetype', 'archetype_confidence', 'strategic_gaps',
            'cut_candidates', 'curve_analysis', 'synergy_potential'
        ]
        for key in required_keys:
            self.assertIn(key, result)
        
        # Validate archetype detection
        self.assertIn(result['primary_archetype'], ['aggressive', 'control', 'tempo', 'value', 'balanced'])
        self.assertIsInstance(result['archetype_confidence'], (int, float))
        self.assertGreaterEqual(result['archetype_confidence'], 0)
        self.assertLessEqual(result['archetype_confidence'], 1)
    
    def test_archetype_validation_and_scoring(self):
        """Test archetype validation and confidence scoring."""
        for archetype_name, deck_state in self.sample_deck_states.items():
            with self.subTest(archetype=archetype_name):
                result = self.analyzer.analyze_deck(deck_state)
                
                # Should detect reasonable archetype
                self.assertIsInstance(result['primary_archetype'], str)
                self.assertGreater(len(result['primary_archetype']), 0)
                
                # Confidence should be reasonable
                self.assertGreaterEqual(result['archetype_confidence'], 0.3)
                self.assertLessEqual(result['archetype_confidence'], 1.0)
    
    def test_strategic_gap_analysis(self):
        """Test strategic gap analysis with priority weighting."""
        deck_state = self.sample_deck_states['mid_control']
        result = self.analyzer.analyze_deck(deck_state)
        
        # Should identify strategic gaps
        self.assertIsInstance(result['strategic_gaps'], list)
        for gap in result['strategic_gaps']:
            self.assertIsInstance(gap, dict)
            self.assertIn('gap_type', gap)
            self.assertIn('priority', gap)
            self.assertIn('description', gap)
            
            # Priority should be weighted
            self.assertIsInstance(gap['priority'], (int, float))
            self.assertGreaterEqual(gap['priority'], 0)
            self.assertLessEqual(gap['priority'], 1)
    
    def test_cut_candidate_logic(self):
        """Test cut candidate logic with explanatory reasoning."""
        deck_state = self.sample_deck_states['late_mixed']
        result = self.analyzer.analyze_deck(deck_state)
        
        # Should suggest cut candidates for full deck
        self.assertIsInstance(result['cut_candidates'], list)
        for candidate in result['cut_candidates']:
            self.assertIsInstance(candidate, dict)
            self.assertIn('card_name', candidate)
            self.assertIn('cut_priority', candidate)
            self.assertIn('reasoning', candidate)
            
            # Should have explanatory reasoning
            self.assertIsInstance(candidate['reasoning'], str)
            self.assertGreater(len(candidate['reasoning']), 10)
    
    def test_draft_phase_awareness(self):
        """Test draft phase awareness with dynamic thresholds."""
        early_deck = self.sample_deck_states['early_aggressive']
        mid_deck = self.sample_deck_states['mid_control']
        late_deck = self.sample_deck_states['late_mixed']
        
        early_result = self.analyzer.analyze_deck(early_deck)
        mid_result = self.analyzer.analyze_deck(mid_deck)
        late_result = self.analyzer.analyze_deck(late_deck)
        
        # Analysis should adapt to draft phase
        # Early draft should focus on value, late on synergy
        self.assertIsInstance(early_result['curve_analysis'], dict)
        self.assertIsInstance(mid_result['curve_analysis'], dict)
        self.assertIsInstance(late_result['curve_analysis'], dict)
        
        # Strategic priorities should differ by phase
        early_gaps = {gap['gap_type'] for gap in early_result['strategic_gaps']}
        late_gaps = {gap['gap_type'] for gap in late_result['strategic_gaps']}
        # Should have different focus areas
        self.assertTrue(len(early_gaps.symmetric_difference(late_gaps)) > 0)

    # Immutable State Architecture Tests
    
    def test_immutable_deck_state_architecture(self):
        """Test copy-on-write deck modifications don't mutate original."""
        original_deck = self.sample_deck_states['early_aggressive']
        original_cards_count = len(original_deck.drafted_cards)
        
        # Analyze deck
        result = self.analyzer.analyze_deck(original_deck)
        
        # Original deck should be unchanged
        self.assertEqual(len(original_deck.drafted_cards), original_cards_count)
        
        # Internal modifications should not affect original
        test_card = self.sample_cards['aggressive'][0]
        modified_deck = self.analyzer._add_card_to_deck_state(original_deck, test_card)
        
        # Original should still be unchanged
        self.assertEqual(len(original_deck.drafted_cards), original_cards_count)
        self.assertEqual(len(modified_deck.drafted_cards), original_cards_count + 1)
    
    def test_thread_safe_deck_analysis(self):
        """Test thread-safe deck state analysis."""
        deck_state = self.sample_deck_states['mid_control']
        results = []
        errors = []
        
        def analyze_deck_thread():
            try:
                result = self.analyzer.analyze_deck(deck_state)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 10 concurrent analyses
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=analyze_deck_thread)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        self.assertEqual(len(errors), 0, f"Errors in concurrent analysis: {errors}")
        self.assertEqual(len(results), 10)
        
        # All results should be identical (deterministic analysis)
        first_archetype = results[0]['primary_archetype']
        for result in results[1:]:
            self.assertEqual(result['primary_archetype'], first_archetype)

    # Fuzzy Archetype Matching Tests
    
    def test_fuzzy_archetype_matching(self):
        """Test probabilistic archetype scoring for edge cases."""
        # Create mixed archetype deck
        mixed_cards = (self.sample_cards['aggressive'][:1] + 
                      self.sample_cards['control'][:1] + 
                      self.sample_cards['value'][:1])
        
        mixed_deck = DeckState(
            drafted_cards=mixed_cards,
            available_choices=[],
            draft_pick_number=10,
            wins=1,
            losses=0
        )
        
        result = self.analyzer.analyze_deck(mixed_deck)
        
        # Should handle mixed archetypes gracefully
        self.assertIsInstance(result['primary_archetype'], str)
        
        # Should have lower confidence for mixed archetypes
        self.assertLess(result['archetype_confidence'], 0.8)
        
        # Should provide archetype probabilities
        if 'archetype_probabilities' in result:
            self.assertIsInstance(result['archetype_probabilities'], dict)
            total_prob = sum(result['archetype_probabilities'].values())
            self.assertAlmostEqual(total_prob, 1.0, places=2)
    
    def test_recommendation_consistency_validation(self):
        """Test detection and resolution of contradictory recommendations."""
        deck_state = self.sample_deck_states['late_mixed']
        result = self.analyzer.analyze_deck(deck_state)
        
        # Strategic gaps and archetype should be consistent
        primary_archetype = result['primary_archetype']
        strategic_gaps = result['strategic_gaps']
        
        # Gaps should align with archetype needs
        if primary_archetype == 'aggressive':
            gap_types = {gap['gap_type'] for gap in strategic_gaps}
            # Aggressive decks should focus on early game
            self.assertTrue('early_game' in gap_types or 'curve' in gap_types or len(gap_types) == 0)
        
        # Cut candidates should not contradict archetype
        cut_candidates = result['cut_candidates']
        for candidate in cut_candidates:
            # Should have valid reasoning
            self.assertIsInstance(candidate['reasoning'], str)
            self.assertGreater(len(candidate['reasoning']), 5)
    
    def test_incremental_analysis_processing(self):
        """Test incremental processing to prevent memory spikes."""
        # Create large deck state
        large_deck_cards = []
        for archetype_cards in self.sample_cards.values():
            large_deck_cards.extend(archetype_cards)
        
        large_deck = DeckState(
            drafted_cards=large_deck_cards,
            available_choices=[
                CardOption(card, 0.5) for card in large_deck_cards[:3]
            ],
            draft_pick_number=28,
            wins=5,
            losses=3
        )
        
        # Should process without memory issues
        result = self.analyzer.analyze_deck(large_deck)
        self.assertIsInstance(result, dict)
        self.assertIn('primary_archetype', result)

    # Advanced Analysis Tests
    
    def test_curve_analysis_detailed(self):
        """Test detailed mana curve analysis."""
        deck_state = self.sample_deck_states['mid_control']
        result = self.analyzer.analyze_deck(deck_state)
        
        curve_analysis = result['curve_analysis']
        self.assertIsInstance(curve_analysis, dict)
        
        # Should analyze mana distribution
        self.assertIn('curve_quality', curve_analysis)
        self.assertIn('early_game_strength', curve_analysis)
        self.assertIn('late_game_strength', curve_analysis)
        
        # Values should be in reasonable ranges
        self.assertGreaterEqual(curve_analysis['curve_quality'], 0)
        self.assertLessEqual(curve_analysis['curve_quality'], 1)
    
    def test_synergy_potential_analysis(self):
        """Test synergy potential analysis between cards."""
        deck_state = self.sample_deck_states['late_mixed']
        result = self.analyzer.analyze_deck(deck_state)
        
        synergy_potential = result['synergy_potential']
        self.assertIsInstance(synergy_potential, dict)
        
        # Should identify synergy opportunities
        self.assertIn('total_synergy_score', synergy_potential)
        self.assertIn('synergy_opportunities', synergy_potential)
        
        # Synergy score should be reasonable
        self.assertIsInstance(synergy_potential['total_synergy_score'], (int, float))
        self.assertGreaterEqual(synergy_potential['total_synergy_score'], 0)
    
    def test_win_rate_consideration(self):
        """Test analysis consideration of current win/loss record."""
        # Create deck states with different records
        losing_deck = copy.deepcopy(self.sample_deck_states['mid_control'])
        losing_deck.wins = 0
        losing_deck.losses = 2
        
        winning_deck = copy.deepcopy(self.sample_deck_states['mid_control'])
        winning_deck.wins = 3
        winning_deck.losses = 0
        
        losing_result = self.analyzer.analyze_deck(losing_deck)
        winning_result = self.analyzer.analyze_deck(winning_deck)
        
        # Analysis should adapt to current performance
        losing_gaps = losing_result['strategic_gaps']
        winning_gaps = winning_result['strategic_gaps']
        
        # Different records should lead to different strategic priorities
        self.assertIsInstance(losing_gaps, list)
        self.assertIsInstance(winning_gaps, list)

    # Error Handling and Edge Cases
    
    def test_empty_deck_analysis(self):
        """Test analysis of empty deck state."""
        empty_deck = DeckState(
            drafted_cards=[],
            available_choices=[
                CardOption(self.sample_cards['aggressive'][0], 0.8),
                CardOption(self.sample_cards['control'][0], 0.7),
                CardOption(self.sample_cards['value'][0], 0.9)
            ],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        result = self.analyzer.analyze_deck(empty_deck)
        
        # Should handle empty deck gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('primary_archetype', result)
        # Should default to balanced or based on available choices
        self.assertTrue(len(result['primary_archetype']) > 0)
    
    def test_invalid_deck_state_handling(self):
        """Test handling of invalid deck states."""
        # Test null deck state
        with self.assertRaises(AIEngineError):
            self.analyzer.analyze_deck(None)
        
        # Test deck state with invalid data
        invalid_deck = DeckState(
            drafted_cards=[None],  # Invalid card
            available_choices=[],
            draft_pick_number=-1,  # Invalid pick number
            wins=-1,  # Invalid wins
            losses=-1   # Invalid losses
        )
        
        # Should handle gracefully with sanitization
        result = self.analyzer.analyze_deck(invalid_deck)
        self.assertIsInstance(result, dict)
        self.assertIn('primary_archetype', result)
    
    def test_extreme_deck_compositions(self):
        """Test analysis of extreme deck compositions."""
        # All aggressive cards
        all_aggressive = DeckState(
            drafted_cards=self.sample_cards['aggressive'] * 10,  # 30 aggressive cards
            available_choices=[],
            draft_pick_number=30,
            wins=0,
            losses=0
        )
        
        result = self.analyzer.analyze_deck(all_aggressive)
        
        # Should correctly identify aggressive archetype
        self.assertEqual(result['primary_archetype'], 'aggressive')
        self.assertGreater(result['archetype_confidence'], 0.8)
        
        # Should identify potential issues
        strategic_gaps = result['strategic_gaps']
        gap_types = {gap['gap_type'] for gap in strategic_gaps}
        # Should identify lack of variety
        self.assertTrue(len(gap_types) > 0)

    # Performance Tests
    
    def test_analysis_performance_benchmarks(self):
        """Test analysis performance meets benchmarks."""
        deck_state = self.sample_deck_states['mid_control']
        
        start_time = time.time()
        
        # Analyze 50 decks
        for _ in range(50):
            result = self.analyzer.analyze_deck(deck_state)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_analysis = total_time / 50
        
        # Should average less than 50ms per analysis
        self.assertLess(avg_time_per_analysis, 0.05, 
                       f"Average analysis time too slow: {avg_time_per_analysis:.4f}s")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of analysis operations."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many analyses
        for i in range(500):
            # Create variety of deck states
            test_cards = list(self.sample_cards['aggressive'])
            if i % 2:
                test_cards.extend(self.sample_cards['control'])
            if i % 3:
                test_cards.extend(self.sample_cards['value'])
            
            test_deck = DeckState(
                drafted_cards=test_cards[:min(len(test_cards), 20)],
                available_choices=[],
                draft_pick_number=min(i + 1, 30),
                wins=i % 12,
                losses=i % 5
            )
            
            result = self.analyzer.analyze_deck(test_deck)
        
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 30MB)
        self.assertLess(memory_growth, 30 * 1024 * 1024, 
                       f"Excessive memory growth: {memory_growth / 1024 / 1024:.2f}MB")
    
    def test_concurrent_analysis_stress(self):
        """Test system under concurrent analysis load."""
        results = []
        errors = []
        
        def stress_analysis():
            try:
                for i in range(25):
                    deck_name = list(self.sample_deck_states.keys())[i % 3]
                    deck_state = self.sample_deck_states[deck_name]
                    result = self.analyzer.analyze_deck(deck_state)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 8 concurrent stress threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(stress_analysis) for _ in range(8)]
            for future in futures:
                future.result()
        
        # Verify no errors under stress
        self.assertEqual(len(errors), 0, f"Errors under stress: {errors}")
        self.assertEqual(len(results), 200)  # 8 threads * 25 analyses
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self.analyzer, '_cleanup'):
            self.analyzer._cleanup()


class TestArchetypeDetection(unittest.TestCase):
    """
    Detailed tests for archetype detection algorithms.
    """
    
    def setUp(self):
        self.analyzer = StrategicDeckAnalyzer()
    
    def test_aggressive_archetype_detection(self):
        """Test detection of aggressive archetype patterns."""
        aggressive_cards = [
            CardInstance(f"Aggressive Card {i}", min(i, 3), i+1, max(i, 1), 
                        "minion", "common", "test", [], f"Aggressive {i}")
            for i in range(1, 6)
        ]
        
        aggressive_deck = DeckState(
            drafted_cards=aggressive_cards,
            available_choices=[],
            draft_pick_number=15,
            wins=0,
            losses=0
        )
        
        result = self.analyzer.analyze_deck(aggressive_deck)
        
        # Should detect aggressive archetype
        self.assertEqual(result['primary_archetype'], 'aggressive')
        self.assertGreater(result['archetype_confidence'], 0.6)
    
    def test_control_archetype_detection(self):
        """Test detection of control archetype patterns."""
        control_cards = [
            CardInstance(f"Control Card {i}", i+3, i, i+2, 
                        "minion", "rare", "test", [], f"Control {i}")
            for i in range(5, 9)
        ]
        
        control_deck = DeckState(
            drafted_cards=control_cards,
            available_choices=[],
            draft_pick_number=15,
            wins=0,
            losses=0
        )
        
        result = self.analyzer.analyze_deck(control_deck)
        
        # Should detect control archetype
        self.assertEqual(result['primary_archetype'], 'control')
        self.assertGreater(result['archetype_confidence'], 0.6)
    
    def test_balanced_archetype_fallback(self):
        """Test fallback to balanced archetype for unclear decks."""
        mixed_cards = [
            CardInstance(f"Mixed Card {i}", (i % 7) + 1, i % 5 + 1, i % 4 + 1, 
                        "minion", "common", "test", [], f"Mixed {i}")
            for i in range(10)
        ]
        
        mixed_deck = DeckState(
            drafted_cards=mixed_cards,
            available_choices=[],
            draft_pick_number=20,
            wins=0,
            losses=0
        )
        
        result = self.analyzer.analyze_deck(mixed_deck)
        
        # Should detect balanced or have low confidence
        self.assertTrue(
            result['primary_archetype'] == 'balanced' or 
            result['archetype_confidence'] < 0.7
        )


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)