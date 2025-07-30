"""
Comprehensive test suite for GrandmasterAdvisor - Phase 1
Tests integration orchestration, confidence scoring, special features, and hardening.
"""

import unittest
import threading
import time
import json
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState, AIDecision
from arena_bot.ai_v2.exceptions import AIEngineError, AnalysisError, ConfidenceError


class TestGrandmasterAdvisor(unittest.TestCase):
    """
    Comprehensive test suite for GrandmasterAdvisor with integration testing.
    """
    
    def setUp(self):
        """Set up test environment with clean advisor instance."""
        self.advisor = GrandmasterAdvisor()
        self.sample_cards = self._create_sample_cards()
        self.sample_deck_states = self._create_sample_deck_states()
    
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
            ),
            CardInstance(
                name="Boulderfist Ogre",
                cost=6,
                attack=6,
                health=7,
                card_type="minion",
                rarity="common",
                card_set="classic",
                keywords=[],
                description="High value minion"
            )
        ]
    
    def _create_sample_deck_states(self):
        """Create sample deck states for different scenarios."""
        return {
            'early_draft': DeckState(
                drafted_cards=self.sample_cards[:2],
                available_choices=[
                    CardOption(self.sample_cards[2], 0.9),
                    CardOption(self.sample_cards[3], 0.7),
                    CardOption(self.sample_cards[0], 0.8)
                ],
                draft_pick_number=5,
                wins=0,
                losses=0
            ),
            'mid_draft': DeckState(
                drafted_cards=self.sample_cards,
                available_choices=[
                    CardOption(self.sample_cards[1], 0.8),
                    CardOption(self.sample_cards[2], 0.6),
                    CardOption(self.sample_cards[3], 0.9)
                ],
                draft_pick_number=15,
                wins=2,
                losses=1
            ),
            'late_draft': DeckState(
                drafted_cards=self.sample_cards * 7,  # 28 cards
                available_choices=[
                    CardOption(self.sample_cards[0], 0.7),
                    CardOption(self.sample_cards[1], 0.5),
                    CardOption(self.sample_cards[2], 0.8)
                ],
                draft_pick_number=29,
                wins=5,
                losses=2
            )
        }

    # Core Integration Tests
    
    def test_analyze_draft_choice_basic_functionality(self):
        """Test basic draft choice analysis returns valid AIDecision."""
        deck_state = self.sample_deck_states['mid_draft']
        result = self.advisor.analyze_draft_choice(deck_state)
        
        # Validate AIDecision structure
        self.assertIsInstance(result, AIDecision)
        self.assertIsInstance(result.recommended_card, str)
        self.assertIsInstance(result.confidence_score, (int, float))
        self.assertIsInstance(result.reasoning, str)
        self.assertIsInstance(result.strategic_context, dict)
        self.assertIsInstance(result.card_evaluations, dict)
        
        # Validate confidence score
        self.assertGreaterEqual(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1)
        
        # Validate reasoning is meaningful
        self.assertGreater(len(result.reasoning), 20)
        
        # Validate card evaluations
        for card_option in deck_state.available_choices:
            self.assertIn(card_option.card.name, result.card_evaluations)
    
    def test_comprehensive_error_handling(self):
        """Test comprehensive error handling for all AI failures."""
        # Test null deck state
        with self.assertRaises(AIEngineError):
            self.advisor.analyze_draft_choice(None)
        
        # Test empty available choices
        empty_deck = DeckState(
            drafted_cards=[],
            available_choices=[],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        with self.assertRaises(AIEngineError):
            self.advisor.analyze_draft_choice(empty_deck)
        
        # Test malformed card data
        malformed_deck = DeckState(
            drafted_cards=[],
            available_choices=[
                CardOption(None, 0.5)  # Null card
            ],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        # Should handle gracefully with fallbacks
        with self.assertRaises(AIEngineError):
            self.advisor.analyze_draft_choice(malformed_deck)
    
    def test_decision_confidence_scoring(self):
        """Test decision confidence scoring and uncertainty handling."""
        deck_state = self.sample_deck_states['mid_draft']
        result = self.advisor.analyze_draft_choice(deck_state)
        
        # Confidence should be reasonable
        self.assertGreaterEqual(result.confidence_score, 0.3)
        self.assertLessEqual(result.confidence_score, 1.0)
        
        # Should provide confidence breakdown
        self.assertIn('confidence_factors', result.strategic_context)
        confidence_factors = result.strategic_context['confidence_factors']
        self.assertIsInstance(confidence_factors, dict)
        
        # Test uncertainty handling for close decisions
        # Create deck with very similar card options
        similar_cards = [
            CardInstance("Similar Card 1", 3, 3, 3, "minion", "common", "test", [], "Similar stats"),
            CardInstance("Similar Card 2", 3, 3, 3, "minion", "common", "test", [], "Similar stats"),
            CardInstance("Similar Card 3", 3, 3, 3, "minion", "common", "test", [], "Similar stats")
        ]
        
        similar_deck = DeckState(
            drafted_cards=[],
            available_choices=[CardOption(card, 0.7) for card in similar_cards],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        similar_result = self.advisor.analyze_draft_choice(similar_deck)
        
        # Should have lower confidence for similar options
        self.assertLess(similar_result.confidence_score, result.confidence_score)
    
    def test_detailed_audit_trail(self):
        """Test detailed audit trail for all AI decisions."""
        deck_state = self.sample_deck_states['early_draft']
        result = self.advisor.analyze_draft_choice(deck_state)
        
        # Should include audit trail
        self.assertIn('audit_trail', result.strategic_context)
        audit_trail = result.strategic_context['audit_trail']
        self.assertIsInstance(audit_trail, list)
        
        # Audit trail should have detailed steps
        self.assertGreater(len(audit_trail), 3)
        
        for step in audit_trail:
            self.assertIsInstance(step, dict)
            self.assertIn('timestamp', step)
            self.assertIn('component', step)
            self.assertIn('action', step)
            self.assertIn('duration_ms', step)
    
    def test_performance_timing(self):
        """Test performance timing for each analysis stage."""
        deck_state = self.sample_deck_states['mid_draft']
        
        start_time = time.time()
        result = self.advisor.analyze_draft_choice(deck_state)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Should complete analysis within reasonable time
        self.assertLess(total_time, 2.0, f"Analysis took too long: {total_time:.3f}s")
        
        # Should include timing breakdown
        self.assertIn('performance_metrics', result.strategic_context)
        perf_metrics = result.strategic_context['performance_metrics']
        self.assertIsInstance(perf_metrics, dict)
        self.assertIn('total_analysis_time_ms', perf_metrics)
        self.assertIn('component_timings', perf_metrics)
    
    def test_atomic_operations(self):
        """Test atomic operations for recommendation generation."""
        deck_state = self.sample_deck_states['late_draft']
        
        # Multiple concurrent analyses should be atomic
        results = []
        errors = []
        
        def analyze_atomically():
            try:
                result = self.advisor.analyze_draft_choice(deck_state)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 5 concurrent analyses
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=analyze_atomically)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All analyses should complete successfully
        self.assertEqual(len(errors), 0, f"Errors in atomic operations: {errors}")
        self.assertEqual(len(results), 5)
        
        # Results should be consistent (same recommendation)
        first_recommendation = results[0].recommended_card
        for result in results[1:]:
            self.assertEqual(result.recommended_card, first_recommendation)

    # Confidence Scoring Hardening Tests
    
    def test_numerical_stability_validation(self):
        """Test NaN/Infinity detection with fallbacks."""
        # Create extreme card values that might cause numerical issues
        extreme_card = CardInstance(
            name="Extreme Card",
            cost=0,
            attack=999999,
            health=999999,
            card_type="minion",
            rarity="legendary",
            card_set="test",
            keywords=["divine_shield", "taunt", "charge", "windfury"],
            description="Extreme test card"
        )
        
        extreme_deck = DeckState(
            drafted_cards=[],
            available_choices=[CardOption(extreme_card, 0.9)],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        result = self.advisor.analyze_draft_choice(extreme_deck)
        
        # Should handle extreme values gracefully
        self.assertIsInstance(result.confidence_score, (int, float))
        self.assertFalse(math.isnan(result.confidence_score) if hasattr(math, 'isnan') else False)
        self.assertFalse(math.isinf(result.confidence_score) if hasattr(math, 'isinf') else False)
        self.assertGreaterEqual(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1)
    
    def test_format_aware_confidence_calibration(self):
        """Test separate thresholds per game format."""
        # Test different draft scenarios (simulating different formats)
        arena_deck = self.sample_deck_states['mid_draft']
        arena_result = self.advisor.analyze_draft_choice(arena_deck)
        
        # Should adjust confidence based on format characteristics
        self.assertIsInstance(arena_result.confidence_score, (int, float))
        
        # Should include format-specific calibration info
        if 'format_calibration' in arena_result.strategic_context:
            format_cal = arena_result.strategic_context['format_calibration']
            self.assertIsInstance(format_cal, dict)
    
    def test_adversarial_input_detection(self):
        """Test detection and handling of unusual card combinations."""
        # Create unusual card combination
        unusual_cards = [
            CardInstance("Weird Card 1", -1, 0, 0, "unknown", "", "", [], "Weird card"),
            CardInstance("", 1000, -50, 1000, "spell", "legendary", "unknown", ["invalid"], "Invalid card"),
            CardInstance("Normal Card", 3, 3, 3, "minion", "common", "classic", [], "Normal card")
        ]
        
        unusual_deck = DeckState(
            drafted_cards=[],
            available_choices=[CardOption(card, 0.5) for card in unusual_cards],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        # Should detect and handle unusual inputs
        try:
            result = self.advisor.analyze_draft_choice(unusual_deck)
            # If it succeeds, should have reasonable confidence
            self.assertLessEqual(result.confidence_score, 0.8)  # Lower confidence for unusual inputs
        except AIEngineError:
            # Acceptable to reject highly unusual inputs
            pass
    
    def test_robust_confidence_aggregation(self):
        """Test median-based aggregation resistant to outliers."""
        deck_state = self.sample_deck_states['mid_draft']
        
        # Mock individual component confidence scores with outliers
        with patch.object(self.advisor.card_evaluator, 'evaluate_card') as mock_eval:
            mock_eval.return_value = {
                'overall_score': 7.5,
                'confidence': 0.9,
                'base_value': 7.0,
                'tempo_score': 8.0,
                'value_score': 7.5,
                'synergy_score': 6.0,
                'curve_score': 8.5,
                're_draftability_score': 7.0
            }
            
            result = self.advisor.analyze_draft_choice(deck_state)
            
            # Should use robust aggregation
            self.assertIsInstance(result.confidence_score, (int, float))
            self.assertGreaterEqual(result.confidence_score, 0)
            self.assertLessEqual(result.confidence_score, 1)

    # Special Features Tests
    
    def test_dynamic_pivot_advisor(self):
        """Test Dynamic Pivot Advisor with confidence thresholds."""
        # Create scenario where pivot might be recommended
        struggling_deck = DeckState(
            drafted_cards=self.sample_cards[:10],  # Mediocre cards
            available_choices=[
                CardOption(self.sample_cards[0], 0.4),  # Low confidence options
                CardOption(self.sample_cards[1], 0.3),
                CardOption(self.sample_cards[2], 0.5)
            ],
            draft_pick_number=15,
            wins=0,
            losses=2
        )
        
        result = self.advisor.analyze_draft_choice(struggling_deck)
        
        # Should include pivot analysis
        self.assertIn('pivot_analysis', result.strategic_context)
        pivot_analysis = result.strategic_context['pivot_analysis']
        self.assertIsInstance(pivot_analysis, dict)
        self.assertIn('should_pivot', pivot_analysis)
        self.assertIn('pivot_reasoning', pivot_analysis)
        
        # Pivot reasoning should be meaningful
        if pivot_analysis['should_pivot']:
            self.assertIsInstance(pivot_analysis['pivot_reasoning'], str)
            self.assertGreater(len(pivot_analysis['pivot_reasoning']), 10)
    
    def test_greed_meter_risk_assessment(self):
        """Test Greed Meter with risk assessment."""
        # Create high-risk, high-reward scenario
        risky_deck = DeckState(
            drafted_cards=self.sample_cards[:5],
            available_choices=[
                CardOption(self.sample_cards[0], 0.9),  # Safe choice
                CardOption(  # Risky legendary
                    CardInstance("Risky Legendary", 9, 4, 4, "minion", "legendary", "test", [], "High risk/reward"),
                    0.7
                ),
                CardOption(self.sample_cards[2], 0.8)  # Balanced choice
            ],
            draft_pick_number=8,
            wins=2,
            losses=0
        )
        
        result = self.advisor.analyze_draft_choice(risky_deck)
        
        # Should include greed analysis
        self.assertIn('greed_analysis', result.strategic_context)
        greed_analysis = result.strategic_context['greed_analysis']
        self.assertIsInstance(greed_analysis, dict)
        self.assertIn('greed_level', greed_analysis)
        self.assertIn('risk_assessment', greed_analysis)
        
        # Greed level should be reasonable
        self.assertIsInstance(greed_analysis['greed_level'], (int, float))
        self.assertGreaterEqual(greed_analysis['greed_level'], 0)
        self.assertLessEqual(greed_analysis['greed_level'], 1)
    
    def test_synergy_trap_detector(self):
        """Test Synergy Trap Detector logic."""
        # Create potential synergy trap scenario
        synergy_cards = [
            CardInstance("Synergy Card 1", 2, 1, 1, "minion", "common", "test", ["murloc"], "Murloc synergy"),
            CardInstance("Synergy Card 2", 3, 2, 2, "minion", "common", "test", ["murloc"], "Murloc synergy"),
            CardInstance("Powerful Generic", 4, 4, 5, "minion", "rare", "test", [], "Strong standalone")
        ]
        
        synergy_deck = DeckState(
            drafted_cards=[synergy_cards[0]],  # Already have one synergy card
            available_choices=[
                CardOption(synergy_cards[1], 0.6),  # More synergy (potential trap)
                CardOption(synergy_cards[2], 0.8)   # Better standalone card
            ],
            draft_pick_number=10,
            wins=1,
            losses=0
        )
        
        result = self.advisor.analyze_draft_choice(synergy_deck)
        
        # Should include synergy trap analysis
        self.assertIn('synergy_trap_analysis', result.strategic_context)
        trap_analysis = result.strategic_context['synergy_trap_analysis']
        self.assertIsInstance(trap_analysis, dict)
        self.assertIn('trap_detected', trap_analysis)
        self.assertIn('trap_warning', trap_analysis)
        
        # If trap detected, should have warning
        if trap_analysis['trap_detected']:
            self.assertIsInstance(trap_analysis['trap_warning'], str)
            self.assertGreater(len(trap_analysis['trap_warning']), 15)
    
    def test_comparative_explanation_generation(self):
        """Test comparative explanation generation with fallback templates."""
        deck_state = self.sample_deck_states['mid_draft']
        result = self.advisor.analyze_draft_choice(deck_state)
        
        # Should provide comparative analysis
        self.assertIn('comparative_analysis', result.strategic_context)
        comparative = result.strategic_context['comparative_analysis']
        self.assertIsInstance(comparative, dict)
        
        # Should compare all available options
        available_cards = [option.card.name for option in deck_state.available_choices]
        for card_name in available_cards:
            self.assertIn(card_name, comparative)
            
            card_comparison = comparative[card_name]
            self.assertIsInstance(card_comparison, dict)
            self.assertIn('pros', card_comparison)
            self.assertIn('cons', card_comparison)
            self.assertIn('situational_value', card_comparison)
    
    def test_decision_validation_archetype_constraints(self):
        """Test decision validation against archetype constraints."""
        # Create deck with clear archetype
        aggressive_cards = [
            CardInstance(f"Aggressive {i}", min(i, 3), i+1, max(i, 1), 
                        "minion", "common", "test", [], f"Aggressive minion {i}")
            for i in range(1, 6)
        ]
        
        aggressive_deck = DeckState(
            drafted_cards=aggressive_cards,
            available_choices=[
                CardOption(
                    CardInstance("Expensive Control Card", 8, 4, 8, "minion", "epic", "test", [], "Control card"),
                    0.7
                ),
                CardOption(
                    CardInstance("Aggressive Card", 2, 3, 1, "minion", "common", "test", ["charge"], "Aggressive card"),
                    0.8
                )
            ],
            draft_pick_number=12,
            wins=0,
            losses=0
        )
        
        result = self.advisor.analyze_draft_choice(aggressive_deck)
        
        # Should validate against archetype
        self.assertIn('archetype_validation', result.strategic_context)
        validation = result.strategic_context['archetype_validation']
        self.assertIsInstance(validation, dict)
        self.assertIn('archetype_consistency', validation)
        self.assertIn('recommendation_alignment', validation)

    # Performance and Integration Tests
    
    def test_full_integration_workflow(self):
        """Test complete integration workflow with all components."""
        deck_state = self.sample_deck_states['mid_draft']
        
        # Should integrate card evaluator and deck analyzer
        with patch.object(self.advisor.card_evaluator, 'evaluate_card') as mock_eval, \
             patch.object(self.advisor.deck_analyzer, 'analyze_deck') as mock_analyze:
            
            # Mock realistic returns
            mock_eval.return_value = {
                'overall_score': 7.5,
                'base_value': 7.0,
                'tempo_score': 8.0,
                'value_score': 7.5,
                'synergy_score': 6.0,
                'curve_score': 8.5,
                're_draftability_score': 7.0
            }
            
            mock_analyze.return_value = {
                'primary_archetype': 'tempo',
                'archetype_confidence': 0.8,
                'strategic_gaps': [{'gap_type': 'early_game', 'priority': 0.7}],
                'cut_candidates': [],
                'curve_analysis': {'curve_quality': 0.7}
            }
            
            result = self.advisor.analyze_draft_choice(deck_state)
            
            # Should call both components
            self.assertTrue(mock_eval.called)
            self.assertTrue(mock_analyze.called)
            
            # Should integrate results
            self.assertIsInstance(result, AIDecision)
            self.assertIn('deck_analysis', result.strategic_context)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for complete analysis."""
        deck_state = self.sample_deck_states['early_draft']
        
        start_time = time.time()
        
        # Analyze 20 draft choices
        for _ in range(20):
            result = self.advisor.analyze_draft_choice(deck_state)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_analysis = total_time / 20
        
        # Should average less than 100ms per full analysis
        self.assertLess(avg_time_per_analysis, 0.1, 
                       f"Average analysis time too slow: {avg_time_per_analysis:.4f}s")
    
    def test_memory_management_integration(self):
        """Test memory management across all integrated components."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many integrated analyses
        for i in range(200):
            # Vary the deck states
            deck_name = list(self.sample_deck_states.keys())[i % 3]
            deck_state = self.sample_deck_states[deck_name]
            
            result = self.advisor.analyze_draft_choice(deck_state)
        
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        self.assertLess(memory_growth, 100 * 1024 * 1024, 
                       f"Excessive memory growth: {memory_growth / 1024 / 1024:.2f}MB")
    
    def test_concurrent_integration_stress(self):
        """Test concurrent access to integrated system."""
        results = []
        errors = []
        
        def stress_integration():
            try:
                for i in range(15):
                    deck_name = list(self.sample_deck_states.keys())[i % 3]
                    deck_state = self.sample_deck_states[deck_name]
                    result = self.advisor.analyze_draft_choice(deck_state)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 6 concurrent stress threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(stress_integration) for _ in range(6)]
            for future in futures:
                future.result()
        
        # Verify no errors under stress
        self.assertEqual(len(errors), 0, f"Errors under integration stress: {errors}")
        self.assertEqual(len(results), 90)  # 6 threads * 15 analyses

    # Edge Cases and Error Recovery
    
    def test_component_failure_recovery(self):
        """Test recovery when individual components fail."""
        deck_state = self.sample_deck_states['mid_draft']
        
        # Test card evaluator failure recovery
        with patch.object(self.advisor.card_evaluator, 'evaluate_card') as mock_eval:
            mock_eval.side_effect = Exception("Card evaluator failed")
            
            # Should recover gracefully with fallbacks
            result = self.advisor.analyze_draft_choice(deck_state)
            self.assertIsInstance(result, AIDecision)
            self.assertIn('fallback_used', result.strategic_context)
        
        # Test deck analyzer failure recovery
        with patch.object(self.advisor.deck_analyzer, 'analyze_deck') as mock_analyze:
            mock_analyze.side_effect = Exception("Deck analyzer failed")
            
            # Should recover gracefully
            result = self.advisor.analyze_draft_choice(deck_state)
            self.assertIsInstance(result, AIDecision)
            self.assertIn('fallback_used', result.strategic_context)
    
    def test_extreme_draft_scenarios(self):
        """Test extreme draft scenarios for robustness."""
        # Very early draft (pick 1)
        very_early = DeckState(
            drafted_cards=[],
            available_choices=[CardOption(card, 0.8) for card in self.sample_cards[:3]],
            draft_pick_number=1,
            wins=0,
            losses=0
        )
        
        early_result = self.advisor.analyze_draft_choice(very_early)
        self.assertIsInstance(early_result, AIDecision)
        
        # Very late draft (pick 30)
        very_late = DeckState(
            drafted_cards=self.sample_cards * 7 + self.sample_cards[:1],  # 29 cards
            available_choices=[CardOption(card, 0.6) for card in self.sample_cards[:3]],
            draft_pick_number=30,
            wins=11,
            losses=1
        )
        
        late_result = self.advisor.analyze_draft_choice(very_late)
        self.assertIsInstance(late_result, AIDecision)
        
        # Should have different analysis approaches
        self.assertNotEqual(early_result.reasoning, late_result.reasoning)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self.advisor, '_cleanup'):
            self.advisor._cleanup()


class TestSpecialFeatures(unittest.TestCase):
    """
    Detailed tests for special AI features.
    """
    
    def setUp(self):
        self.advisor = GrandmasterAdvisor()
    
    def test_dynamic_pivot_advisor_thresholds(self):
        """Test Dynamic Pivot Advisor confidence thresholds."""
        # Create low-confidence scenario
        low_confidence_deck = DeckState(
            drafted_cards=[],
            available_choices=[
                CardOption(
                    CardInstance("Mediocre Card 1", 3, 2, 3, "minion", "common", "test", [], "Mediocre"),
                    0.4
                ),
                CardOption(
                    CardInstance("Mediocre Card 2", 3, 3, 2, "minion", "common", "test", [], "Also mediocre"),
                    0.3
                )
            ],
            draft_pick_number=8,
            wins=0,
            losses=2
        )
        
        result = self.advisor.analyze_draft_choice(low_confidence_deck)
        pivot_analysis = result.strategic_context.get('pivot_analysis', {})
        
        # Should suggest pivot when confidence is low and record is poor
        if pivot_analysis.get('should_pivot'):
            self.assertIn('pivot_reasoning', pivot_analysis)
            self.assertIn('alternative_strategy', pivot_analysis)
    
    def test_greed_meter_calibration(self):
        """Test Greed Meter calibration across different scenarios."""
        scenarios = [
            # Conservative scenario (early, losing)
            (DeckState([], [], 3, 0, 1), 'conservative'),
            # Balanced scenario (mid, even)
            (DeckState([], [], 15, 3, 3), 'balanced'),
            # Greedy scenario (late, winning)
            (DeckState([], [], 25, 8, 2), 'greedy')
        ]
        
        for deck_state, expected_tendency in scenarios:
            deck_state.available_choices = [
                CardOption(
                    CardInstance("Safe Card", 3, 3, 3, "minion", "common", "test", [], "Safe choice"),
                    0.8
                ),
                CardOption(
                    CardInstance("Risky Card", 6, 1, 1, "minion", "legendary", "test", [], "High risk/reward"),
                    0.6
                )
            ]
            
            with self.subTest(scenario=expected_tendency):
                result = self.advisor.analyze_draft_choice(deck_state)
                greed_analysis = result.strategic_context.get('greed_analysis', {})
                
                self.assertIn('greed_level', greed_analysis)
                greed_level = greed_analysis['greed_level']
                
                # Validate greed level matches scenario expectations
                if expected_tendency == 'conservative':
                    self.assertLess(greed_level, 0.4)
                elif expected_tendency == 'greedy':
                    self.assertGreater(greed_level, 0.6)


if __name__ == '__main__':
    # Import math for NaN/Infinity checks
    import math
    
    # Run the test suite
    unittest.main(verbosity=2)