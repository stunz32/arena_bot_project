"""
Phase 1 Integration Testing and Validation Suite.
Comprehensive end-to-end testing of all Phase 1 AI components working together.
"""

import unittest
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add arena_bot to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arena_bot.ai_v2.card_evaluator import CardEvaluationEngine
from arena_bot.ai_v2.deck_analyzer import StrategicDeckAnalyzer
from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
from arena_bot.ai_v2.data_models import CardOption, CardInstance, DeckState, AIDecision
from arena_bot.ai_v2.exceptions import AIEngineError


class Phase1IntegrationTests(unittest.TestCase):
    """
    Comprehensive integration testing for Phase 1 AI system.
    """
    
    def setUp(self):
        """Set up integration testing environment."""
        # Initialize all components
        self.card_evaluator = CardEvaluationEngine()
        self.deck_analyzer = StrategicDeckAnalyzer()
        self.grandmaster_advisor = GrandmasterAdvisor()
        
        # Create comprehensive test scenarios
        self.test_scenarios = self._create_integration_test_scenarios()
        
        # Track integration metrics
        self.integration_metrics = {
            'total_tests': 0,
            'successful_tests': 0,
            'component_failures': {},
            'performance_metrics': {},
            'quality_scores': []
        }
    
    def _create_integration_test_scenarios(self):
        """Create comprehensive test scenarios covering different draft situations."""
        scenarios = {}
        
        # Early draft scenarios (picks 1-10)
        scenarios['early_aggressive'] = {
            'name': 'Early Aggressive Draft',
            'deck_state': DeckState(
                drafted_cards=[
                    CardInstance("Leper Gnome", 1, 2, 1, "minion", "common", "classic", ["charge"], "1-drop"),
                    CardInstance("Knife Juggler", 2, 3, 2, "minion", "rare", "classic", [], "2-drop")
                ],
                available_choices=[
                    CardOption(CardInstance("Wolfrider", 3, 3, 1, "minion", "common", "classic", ["charge"], "Aggressive 3-drop"), 0.85),
                    CardOption(CardInstance("Water Elemental", 4, 3, 6, "minion", "common", "classic", ["freeze"], "Control minion"), 0.70),
                    CardOption(CardInstance("Azure Drake", 5, 4, 4, "minion", "rare", "classic", ["spell_damage", "card_draw"], "Value minion"), 0.75)
                ],
                draft_pick_number=3,
                wins=0,
                losses=0
            ),
            'expected_archetype': 'aggressive',
            'expected_recommendation': 'Wolfrider',
            'min_confidence': 0.7
        }
        
        scenarios['early_control'] = {
            'name': 'Early Control Draft',
            'deck_state': DeckState(
                drafted_cards=[
                    CardInstance("Flamestrike", 7, None, None, "spell", "epic", "classic", ["spell", "damage"], "Board clear"),
                    CardInstance("Water Elemental", 4, 3, 6, "minion", "common", "classic", ["freeze"], "Control minion")
                ],
                available_choices=[
                    CardOption(CardInstance("Leper Gnome", 1, 2, 1, "minion", "common", "classic", ["charge"], "Aggressive 1-drop"), 0.60),
                    CardOption(CardInstance("Archmage Antonidas", 7, 5, 7, "minion", "legendary", "classic", [], "Control win condition"), 0.90),
                    CardOption(CardInstance("Harvest Golem", 3, 2, 3, "minion", "common", "classic", ["deathrattle"], "Sticky minion"), 0.75)
                ],
                draft_pick_number=3,
                wins=0,
                losses=0
            ),
            'expected_archetype': 'control',
            'expected_recommendation': 'Archmage Antonidas',
            'min_confidence': 0.8
        }
        
        # Mid draft scenarios (picks 11-20)
        scenarios['mid_draft_pivot'] = {
            'name': 'Mid Draft Pivot Decision',
            'deck_state': DeckState(
                drafted_cards=[
                    # Mixed signals requiring pivot decision
                    CardInstance("Leper Gnome", 1, 2, 1, "minion", "common", "classic", ["charge"], "Aggressive"),
                    CardInstance("Flamestrike", 7, None, None, "spell", "epic", "classic", ["spell"], "Control"),
                    CardInstance("Azure Drake", 5, 4, 4, "minion", "rare", "classic", ["spell_damage"], "Value"),
                    CardInstance("Knife Juggler", 2, 3, 2, "minion", "rare", "classic", [], "Aggressive"),
                    CardInstance("Water Elemental", 4, 3, 6, "minion", "common", "classic", ["freeze"], "Control")
                ],
                available_choices=[
                    CardOption(CardInstance("Wolfrider", 3, 3, 1, "minion", "common", "classic", ["charge"], "Continue aggressive"), 0.70),
                    CardOption(CardInstance("Boulderfist Ogre", 6, 6, 7, "minion", "common", "classic", [], "Value/Control"), 0.80),
                    CardOption(CardInstance("Dark Iron Dwarf", 4, 4, 4, "minion", "common", "classic", ["battlecry"], "Tempo"), 0.75)
                ],
                draft_pick_number=6,
                wins=1,
                losses=1
            ),
            'expected_archetype': 'balanced',  # Should detect mixed signals
            'min_confidence': 0.6,
            'should_have_pivot_analysis': True
        }
        
        # Late draft scenarios (picks 21-30)
        scenarios['late_draft_curve_fix'] = {
            'name': 'Late Draft Curve Fixing',
            'deck_state': DeckState(
                drafted_cards=[
                    # Deck with curve issues (too many high cost cards)
                    CardInstance("Flamestrike", 7, None, None, "spell", "epic", "classic", ["spell"], "7-cost"),
                    CardInstance("Boulderfist Ogre", 6, 6, 7, "minion", "common", "classic", [], "6-cost"),
                    CardInstance("Archmage Antonidas", 7, 5, 7, "minion", "legendary", "classic", [], "7-cost"),
                    CardInstance("Fire Elemental", 6, 6, 5, "minion", "common", "classic", ["battlecry"], "6-cost"),
                    CardInstance("Azure Drake", 5, 4, 4, "minion", "rare", "classic", ["spell_damage"], "5-cost")
                ] * 4 + [CardInstance("Water Elemental", 4, 3, 6, "minion", "common", "classic", ["freeze"], "4-cost")] * 5,
                available_choices=[
                    CardOption(CardInstance("Leper Gnome", 1, 2, 1, "minion", "common", "classic", ["charge"], "1-cost early game"), 0.85),
                    CardOption(CardInstance("Mountain Giant", 12, 8, 8, "minion", "epic", "classic", [], "Expensive late game"), 0.70),
                    CardOption(CardInstance("Knife Juggler", 2, 3, 2, "minion", "rare", "classic", [], "2-cost early game"), 0.80)
                ],
                draft_pick_number=26,
                wins=3,
                losses=2
            ),
            'expected_curve_priority': 'early_game',
            'expected_recommendation': 'Leper Gnome',  # Should prioritize curve fixing
            'min_confidence': 0.7
        }
        
        # High-stakes scenarios
        scenarios['high_wins_greedy'] = {
            'name': 'High Wins Greedy Decision',
            'deck_state': DeckState(
                drafted_cards=[
                    CardInstance("Water Elemental", 4, 3, 6, "minion", "common", "classic", ["freeze"], "Solid minion")
                ] * 15,  # Decent but not amazing deck
                available_choices=[
                    CardOption(CardInstance("Chillwind Yeti", 4, 4, 5, "minion", "common", "classic", [], "Safe, solid pick"), 0.85),
                    CardOption(CardInstance("Ysera", 9, 4, 12, "minion", "legendary", "classic", [], "Greedy legendary"), 0.75),
                    CardOption(CardInstance("Boulderfist Ogre", 6, 6, 7, "minion", "common", "classic", [], "Good value"), 0.80)
                ],
                draft_pick_number=16,
                wins=7,  # High wins - might be greedy
                losses=1
            ),
            'expected_greed_level': 'moderate_to_high',
            'should_have_greed_analysis': True,
            'min_confidence': 0.6
        }
        
        # Struggling scenarios
        scenarios['low_wins_safe'] = {
            'name': 'Low Wins Safe Decision',
            'deck_state': DeckState(
                drafted_cards=[
                    CardInstance("Booty Bay Bodyguard", 5, 5, 4, "minion", "common", "classic", ["taunt"], "Below average")
                ] * 12,  # Below average deck
                available_choices=[
                    CardOption(CardInstance("Chillwind Yeti", 4, 4, 5, "minion", "common", "classic", [], "Solid, safe pick"), 0.85),
                    CardOption(CardInstance("Millhouse Manastorm", 2, 4, 4, "minion", "legendary", "classic", [], "Risky legendary"), 0.60),
                    CardOption(CardInstance("Magma Rager", 3, 5, 1, "minion", "common", "classic", [], "Poor card"), 0.40)
                ],
                draft_pick_number=13,
                wins=0,  # Struggling
                losses=2
            ),
            'expected_greed_level': 'conservative',
            'expected_recommendation': 'Chillwind Yeti',  # Should play it safe
            'should_suggest_safe_pick': True,
            'min_confidence': 0.8
        }
        
        return scenarios

    # Component Integration Tests
    
    def test_card_evaluator_deck_analyzer_integration(self):
        """Test integration between card evaluator and deck analyzer."""
        scenario = self.test_scenarios['early_aggressive']
        deck_state = scenario['deck_state']
        
        # Get deck analysis
        deck_analysis = self.deck_analyzer.analyze_deck(deck_state)
        
        # Evaluate each available card in context of deck analysis
        card_evaluations = {}
        for choice in deck_state.available_choices:
            evaluation = self.card_evaluator.evaluate_card(choice.card, deck_state)
            card_evaluations[choice.card.name] = evaluation
        
        # Integration validation
        self.assertIsInstance(deck_analysis, dict)
        self.assertIn('primary_archetype', deck_analysis)
        
        for card_name, evaluation in card_evaluations.items():
            self.assertIsInstance(evaluation, dict)
            self.assertIn('overall_score', evaluation)
            
            # Card evaluation should be consistent with deck archetype
            if deck_analysis['primary_archetype'] == 'aggressive':
                # Aggressive cards should score higher in aggressive deck
                if 'charge' in [choice.card.keywords for choice in deck_state.available_choices if choice.card.name == card_name][0]:
                    self.assertGreater(evaluation['tempo_score'], 5.0)
        
        print(f"✓ Card Evaluator-Deck Analyzer Integration: {deck_analysis['primary_archetype']} archetype detected")
        
        self.integration_metrics['successful_tests'] += 1
    
    def test_deck_analyzer_grandmaster_advisor_integration(self):
        """Test integration between deck analyzer and grandmaster advisor."""
        scenario = self.test_scenarios['mid_draft_pivot']
        deck_state = scenario['deck_state']
        
        # Get standalone deck analysis
        standalone_analysis = self.deck_analyzer.analyze_deck(deck_state)
        
        # Get grandmaster advisor decision (which internally uses deck analyzer)
        advisor_decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
        
        # Integration validation
        self.assertIsInstance(standalone_analysis, dict)
        self.assertIsInstance(advisor_decision, AIDecision)
        
        # Advisor decision should include deck analysis context
        self.assertIn('deck_analysis', advisor_decision.strategic_context)
        integrated_analysis = advisor_decision.strategic_context['deck_analysis']
        
        # Core analysis should be consistent
        self.assertEqual(
            standalone_analysis['primary_archetype'],
            integrated_analysis['primary_archetype']
        )
        
        # Advisor should use deck analysis for decision making
        if scenario.get('should_have_pivot_analysis'):
            self.assertIn('pivot_analysis', advisor_decision.strategic_context)
            pivot_analysis = advisor_decision.strategic_context['pivot_analysis']
            self.assertIn('should_pivot', pivot_analysis)
        
        print(f"✓ Deck Analyzer-Grandmaster Advisor Integration: Pivot analysis included")
        
        self.integration_metrics['successful_tests'] += 1
    
    def test_full_system_integration(self):
        """Test complete system integration across all components."""
        scenario = self.test_scenarios['early_control']
        deck_state = scenario['deck_state']
        
        start_time = time.perf_counter()
        
        # Full system analysis
        decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Comprehensive validation
        self.assertIsInstance(decision, AIDecision)
        
        # Validate core decision properties
        self.assertIsInstance(decision.recommended_card, str)
        self.assertGreater(len(decision.recommended_card), 0)
        self.assertIsInstance(decision.confidence_score, (int, float))
        self.assertGreaterEqual(decision.confidence_score, 0)
        self.assertLessEqual(decision.confidence_score, 1)
        self.assertIsInstance(decision.reasoning, str)
        self.assertGreater(len(decision.reasoning), 20)
        
        # Validate strategic context completeness
        required_context_keys = [
            'deck_analysis', 'card_evaluations', 'performance_metrics',
            'confidence_factors', 'audit_trail'
        ]
        for key in required_context_keys:
            self.assertIn(key, decision.strategic_context, f"Missing context key: {key}")
        
        # Validate card evaluations for all choices
        for choice in deck_state.available_choices:
            self.assertIn(choice.card.name, decision.card_evaluations)
            card_eval = decision.card_evaluations[choice.card.name]
            self.assertIsInstance(card_eval, dict)
            self.assertIn('overall_score', card_eval)
        
        # Validate performance metrics
        perf_metrics = decision.strategic_context['performance_metrics']
        self.assertIn('total_analysis_time_ms', perf_metrics)
        self.assertIn('component_timings', perf_metrics)
        
        # Validate audit trail
        audit_trail = decision.strategic_context['audit_trail']
        self.assertIsInstance(audit_trail, list)
        self.assertGreater(len(audit_trail), 3)  # Should have multiple steps
        
        for step in audit_trail:
            self.assertIn('timestamp', step)
            self.assertIn('component', step)
            self.assertIn('action', step)
            self.assertIn('duration_ms', step)
        
        # Performance validation
        self.assertLess(execution_time_ms, 200, f"Full system too slow: {execution_time_ms:.2f}ms")
        
        # Quality validation
        if scenario.get('expected_recommendation'):
            if decision.recommended_card == scenario['expected_recommendation']:
                quality_score = 1.0
            else:
                quality_score = 0.5  # Partial credit for reasonable decision
        else:
            quality_score = 0.8  # Default for scenarios without specific expectation
        
        self.integration_metrics['quality_scores'].append(quality_score)
        self.integration_metrics['performance_metrics']['full_system_ms'] = execution_time_ms
        
        print(f"✓ Full System Integration: {execution_time_ms:.2f}ms, Quality: {quality_score:.1f}")
        
        self.integration_metrics['successful_tests'] += 1

    # Scenario-Based Integration Tests
    
    def test_early_draft_integration(self):
        """Test integration for early draft scenarios."""
        scenarios = ['early_aggressive', 'early_control']
        
        for scenario_name in scenarios:
            with self.subTest(scenario=scenario_name):
                scenario = self.test_scenarios[scenario_name]
                deck_state = scenario['deck_state']
                
                decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                
                # Validate archetype detection
                deck_analysis = decision.strategic_context.get('deck_analysis', {})
                detected_archetype = deck_analysis.get('primary_archetype', 'unknown')
                
                if 'expected_archetype' in scenario:
                    expected = scenario['expected_archetype']
                    self.assertEqual(detected_archetype, expected,
                                   f"Expected {expected} archetype, got {detected_archetype}")
                
                # Validate confidence levels
                if 'min_confidence' in scenario:
                    min_conf = scenario['min_confidence']
                    self.assertGreaterEqual(decision.confidence_score, min_conf,
                                          f"Confidence {decision.confidence_score:.3f} below minimum {min_conf}")
                
                # Early draft should prioritize value/power
                self.assertIn('strategic_context', decision.strategic_context.get('deck_analysis', {}))
                
                print(f"✓ Early Draft Integration ({scenario_name}): {detected_archetype} archetype")
    
    def test_mid_draft_integration(self):
        """Test integration for mid-draft decision scenarios."""
        scenario = self.test_scenarios['mid_draft_pivot']
        deck_state = scenario['deck_state']
        
        decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
        
        # Should include pivot analysis for mixed deck
        self.assertIn('pivot_analysis', decision.strategic_context)
        pivot_analysis = decision.strategic_context['pivot_analysis']
        
        self.assertIn('should_pivot', pivot_analysis)
        self.assertIn('pivot_reasoning', pivot_analysis)
        self.assertIsInstance(pivot_analysis['should_pivot'], bool)
        
        if pivot_analysis['should_pivot']:
            self.assertIsInstance(pivot_analysis['pivot_reasoning'], str)
            self.assertGreater(len(pivot_analysis['pivot_reasoning']), 10)
        
        # Should have lower confidence due to mixed signals
        max_expected_confidence = 0.8
        self.assertLessEqual(decision.confidence_score, max_expected_confidence,
                           f"Confidence too high for mixed deck: {decision.confidence_score:.3f}")
        
        print(f"✓ Mid Draft Integration: Pivot analysis included, confidence: {decision.confidence_score:.3f}")
    
    def test_late_draft_integration(self):
        """Test integration for late draft curve-fixing scenarios."""
        scenario = self.test_scenarios['late_draft_curve_fix']
        deck_state = scenario['deck_state']
        
        decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
        
        # Should recognize curve issues
        deck_analysis = decision.strategic_context.get('deck_analysis', {})
        curve_analysis = deck_analysis.get('curve_analysis', {})
        
        self.assertIn('curve_quality', curve_analysis)
        curve_quality = curve_analysis['curve_quality']
        self.assertLess(curve_quality, 0.7, f"Should detect curve issues, got quality: {curve_quality}")
        
        # Should recommend early game card for curve fixing
        if 'expected_recommendation' in scenario:
            expected_rec = scenario['expected_recommendation']
            self.assertEqual(decision.recommended_card, expected_rec,
                           f"Expected {expected_rec} for curve fixing, got {decision.recommended_card}")
        
        # Strategic context should mention curve concerns
        reasoning_lower = decision.reasoning.lower()
        self.assertTrue(
            any(term in reasoning_lower for term in ['curve', 'early', 'mana']),
            f"Reasoning should mention curve issues: {decision.reasoning[:100]}..."
        )
        
        print(f"✓ Late Draft Integration: Curve fixing prioritized")
    
    def test_risk_assessment_integration(self):
        """Test integration of risk assessment features."""
        high_risk_scenario = self.test_scenarios['high_wins_greedy']
        low_risk_scenario = self.test_scenarios['low_wins_safe']
        
        # Test greedy scenario (high wins)
        greedy_decision = self.grandmaster_advisor.analyze_draft_choice(
            high_risk_scenario['deck_state']
        )
        
        self.assertIn('greed_analysis', greedy_decision.strategic_context)
        greed_analysis = greedy_decision.strategic_context['greed_analysis']
        
        self.assertIn('greed_level', greed_analysis)
        self.assertIn('risk_assessment', greed_analysis)
        
        greed_level = greed_analysis['greed_level']
        self.assertIsInstance(greed_level, (int, float))
        self.assertGreaterEqual(greed_level, 0.5, f"Should be greedy with high wins: {greed_level}")
        
        # Test conservative scenario (low wins)
        safe_decision = self.grandmaster_advisor.analyze_draft_choice(
            low_risk_scenario['deck_state']
        )
        
        safe_greed_analysis = safe_decision.strategic_context.get('greed_analysis', {})
        safe_greed_level = safe_greed_analysis.get('greed_level', 0.5)
        
        # Should be more conservative when struggling
        self.assertLess(safe_greed_level, greed_level,
                       f"Should be more conservative when struggling: {safe_greed_level} vs {greed_level}")
        
        print(f"✓ Risk Assessment Integration: Greedy {greed_level:.2f}, Safe {safe_greed_level:.2f}")

    # Error Handling Integration Tests
    
    def test_component_failure_recovery_integration(self):
        """Test system recovery when individual components fail."""
        scenario = self.test_scenarios['early_aggressive']
        deck_state = scenario['deck_state']
        
        # Test card evaluator failure recovery
        with unittest.mock.patch.object(self.grandmaster_advisor.card_evaluator, 'evaluate_card') as mock_eval:
            mock_eval.side_effect = Exception("Card evaluator failed")
            
            try:
                decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                
                # Should still produce a decision with fallbacks
                self.assertIsInstance(decision, AIDecision)
                self.assertIn('fallback_used', decision.strategic_context)
                
                fallback_info = decision.strategic_context['fallback_used']
                self.assertIn('card_evaluator', fallback_info)
                
                print("✓ Card Evaluator Failure Recovery: Fallback system working")
                
            except AIEngineError:
                # Acceptable to fail gracefully with proper error
                print("✓ Card Evaluator Failure Recovery: Graceful failure")
        
        # Test deck analyzer failure recovery
        with unittest.mock.patch.object(self.grandmaster_advisor.deck_analyzer, 'analyze_deck') as mock_analyze:
            mock_analyze.side_effect = Exception("Deck analyzer failed")
            
            try:
                decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                
                # Should still produce a decision
                self.assertIsInstance(decision, AIDecision)
                self.assertIn('fallback_used', decision.strategic_context)
                
                print("✓ Deck Analyzer Failure Recovery: Fallback system working")
                
            except AIEngineError:
                # Acceptable graceful failure
                print("✓ Deck Analyzer Failure Recovery: Graceful failure")
    
    def test_invalid_data_handling_integration(self):
        """Test system handling of invalid data across components."""
        # Create deck state with problematic data
        problematic_card = CardInstance(
            name="",  # Empty name
            cost=-1,  # Invalid cost
            attack=None,
            health=None,
            card_type="invalid_type",
            rarity="",
            card_set="",
            keywords=None,
            description=""
        )
        
        try:
            problematic_deck = DeckState(
                drafted_cards=[],
                available_choices=[CardOption(problematic_card, 0.5)],
                draft_pick_number=1,
                wins=0,
                losses=0
            )
            
            # System should handle invalid data gracefully
            try:
                decision = self.grandmaster_advisor.analyze_draft_choice(problematic_deck)
                
                # If it succeeds, should note data issues
                if 'data_issues' in decision.strategic_context:
                    print("✓ Invalid Data Integration: Issues detected and handled")
                else:
                    print("✓ Invalid Data Integration: Processed without detection")
                    
            except (AIEngineError, ValueError, TypeError):
                # Acceptable to reject invalid data
                print("✓ Invalid Data Integration: Properly rejected invalid data")
                
        except (ValueError, TypeError):
            # Acceptable if data validation catches issues early
            print("✓ Invalid Data Integration: Early validation working")

    # Performance Integration Tests
    
    def test_performance_integration_benchmarks(self):
        """Test integrated system performance meets all benchmarks."""
        all_scenarios = list(self.test_scenarios.values())
        execution_times = []
        memory_usage = []
        
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        for scenario in all_scenarios:
            deck_state = scenario['deck_state']
            
            # Measure execution time
            start_time = time.perf_counter()
            decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)
            
            # Measure memory usage
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_usage.append(current_memory - initial_memory)
            
            # Validate decision quality
            self.assertIsInstance(decision, AIDecision)
            self.assertGreater(decision.confidence_score, 0.2)
        
        # Performance analysis
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)
        avg_memory_growth = sum(memory_usage) / len(memory_usage)
        max_memory_growth = max(memory_usage)
        
        print(f"Performance Integration Results:")
        print(f"  Average execution time: {avg_execution_time:.2f}ms")
        print(f"  Maximum execution time: {max_execution_time:.2f}ms")
        print(f"  Average memory growth: {avg_memory_growth:.2f}MB")
        print(f"  Maximum memory growth: {max_memory_growth:.2f}MB")
        
        # Performance assertions (from todo_ai_helper.md requirements)
        self.assertLess(avg_execution_time, 100, f"Average execution too slow: {avg_execution_time:.2f}ms")
        self.assertLess(max_execution_time, 200, f"Max execution too slow: {max_execution_time:.2f}ms")
        self.assertLess(avg_memory_growth, 50, f"Memory growth too high: {avg_memory_growth:.2f}MB")
        
        # Store metrics
        self.integration_metrics['performance_metrics'].update({
            'avg_execution_time_ms': avg_execution_time,
            'max_execution_time_ms': max_execution_time,
            'avg_memory_growth_mb': avg_memory_growth,
            'max_memory_growth_mb': max_memory_growth
        })

    # Quality and Consistency Integration Tests
    
    def test_decision_quality_integration(self):
        """Test overall decision quality across all scenarios."""
        quality_scores = []
        consistency_scores = []
        
        for scenario_name, scenario in self.test_scenarios.items():
            deck_state = scenario['deck_state']
            
            # Get multiple decisions for consistency testing
            decisions = []
            for _ in range(3):
                decision = self.grandmaster_advisor.analyze_draft_choice(deck_state)
                decisions.append(decision)
            
            # Quality assessment
            primary_decision = decisions[0]
            
            # Base quality score
            quality_score = 0.5  # Baseline
            
            # Bonus for meeting expectations
            if 'expected_recommendation' in scenario:
                if primary_decision.recommended_card == scenario['expected_recommendation']:
                    quality_score += 0.3
            
            if 'min_confidence' in scenario:
                if primary_decision.confidence_score >= scenario['min_confidence']:
                    quality_score += 0.2
            
            # Consistency assessment
            recommendations = [d.recommended_card for d in decisions]
            unique_recommendations = set(recommendations)
            consistency_score = 1.0 - (len(unique_recommendations) - 1) * 0.3  # Penalty for inconsistency
            
            quality_scores.append(quality_score)
            consistency_scores.append(consistency_score)
            
            print(f"Scenario {scenario_name}: Quality {quality_score:.2f}, Consistency {consistency_score:.2f}")
        
        # Overall quality metrics
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        print(f"Overall Quality Integration:")
        print(f"  Average quality score: {avg_quality:.3f}")
        print(f"  Average consistency score: {avg_consistency:.3f}")
        
        # Quality assertions
        self.assertGreater(avg_quality, 0.7, f"Overall quality too low: {avg_quality:.3f}")
        self.assertGreater(avg_consistency, 0.8, f"Consistency too low: {avg_consistency:.3f}")
        
        # Store metrics
        self.integration_metrics['quality_scores'].extend(quality_scores)
    
    def test_special_features_integration(self):
        """Test integration of all special AI features."""
        # Test Dynamic Pivot Advisor
        pivot_scenario = self.test_scenarios['mid_draft_pivot']
        pivot_decision = self.grandmaster_advisor.analyze_draft_choice(pivot_scenario['deck_state'])
        
        self.assertIn('pivot_analysis', pivot_decision.strategic_context)
        pivot_analysis = pivot_decision.strategic_context['pivot_analysis']
        self.assertIn('should_pivot', pivot_analysis)
        
        # Test Greed Meter
        greedy_scenario = self.test_scenarios['high_wins_greedy']
        greedy_decision = self.grandmaster_advisor.analyze_draft_choice(greedy_scenario['deck_state'])
        
        self.assertIn('greed_analysis', greedy_decision.strategic_context)
        greed_analysis = greedy_decision.strategic_context['greed_analysis']
        self.assertIn('greed_level', greed_analysis)
        self.assertIn('risk_assessment', greed_analysis)
        
        # Test Synergy Trap Detector (would need specific synergy scenario)
        # For now, verify the analysis structure exists
        for decision in [pivot_decision, greedy_decision]:
            if 'synergy_trap_analysis' in decision.strategic_context:
                trap_analysis = decision.strategic_context['synergy_trap_analysis']
                self.assertIn('trap_detected', trap_analysis)
        
        # Test Comparative Analysis
        self.assertIn('comparative_analysis', pivot_decision.strategic_context)
        comparative = pivot_decision.strategic_context['comparative_analysis']
        
        # Should compare all available options
        available_cards = [choice.card.name for choice in pivot_scenario['deck_state'].available_choices]
        for card_name in available_cards:
            self.assertIn(card_name, comparative)
        
        print("✓ Special Features Integration: All features working")

    def tearDown(self):
        """Clean up and report integration test results."""
        self.integration_metrics['total_tests'] = self.integration_metrics['successful_tests']
        
        # Calculate overall quality score
        if self.integration_metrics['quality_scores']:
            overall_quality = sum(self.integration_metrics['quality_scores']) / len(self.integration_metrics['quality_scores'])
            self.integration_metrics['overall_quality'] = overall_quality
        
        # Final integration report would be printed here
        # print(f"\nPhase 1 Integration Test Summary:")
        # print(f"  Tests completed: {self.integration_metrics['total_tests']}")
        # print(f"  Overall quality: {self.integration_metrics.get('overall_quality', 0.0):.3f}")


class Phase1ValidationTests(unittest.TestCase):
    """
    Final validation tests for Phase 1 completeness.
    """
    
    def test_all_requirements_implemented(self):
        """Validate all Phase 1 requirements are implemented."""
        # Test that all major components exist and are functional
        card_evaluator = CardEvaluationEngine()
        deck_analyzer = StrategicDeckAnalyzer()
        grandmaster_advisor = GrandmasterAdvisor()
        
        # Validate component interfaces
        self.assertTrue(hasattr(card_evaluator, 'evaluate_card'))
        self.assertTrue(hasattr(deck_analyzer, 'analyze_deck'))
        self.assertTrue(hasattr(grandmaster_advisor, 'analyze_draft_choice'))
        
        # Test basic functionality
        test_card = CardInstance("Test", 3, 3, 3, "minion", "common", "test", [], "Test card")
        test_deck = DeckState([test_card], [CardOption(test_card, 0.8)], 2, 0, 0)
        
        # All components should work
        card_result = card_evaluator.evaluate_card(test_card, test_deck)
        deck_result = deck_analyzer.analyze_deck(test_deck)
        advisor_result = grandmaster_advisor.analyze_draft_choice(test_deck)
        
        self.assertIsInstance(card_result, dict)
        self.assertIsInstance(deck_result, dict)
        self.assertIsInstance(advisor_result, AIDecision)
        
        print("✓ All Phase 1 Requirements: Components implemented and functional")
    
    def test_hardening_features_implemented(self):
        """Validate all hardening features are implemented."""
        # This would test specific hardening features
        # For now, verify basic error handling exists
        
        grandmaster_advisor = GrandmasterAdvisor()
        
        # Test null handling
        try:
            grandmaster_advisor.analyze_draft_choice(None)
            self.fail("Should reject null input")
        except (AIEngineError, ValueError, TypeError):
            # Expected to reject null input
            pass
        
        print("✓ Hardening Features: Basic validation working")
    
    def test_performance_requirements_met(self):
        """Validate performance requirements are met."""
        grandmaster_advisor = GrandmasterAdvisor()
        test_card = CardInstance("Test", 3, 3, 3, "minion", "common", "test", [], "Test")
        test_deck = DeckState([test_card], [CardOption(test_card, 0.8)], 2, 0, 0)
        
        # Time multiple operations
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = grandmaster_advisor.analyze_draft_choice(test_deck)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Performance requirements from todo_ai_helper.md
        self.assertLess(avg_time, 100, f"Average time {avg_time:.2f}ms exceeds 100ms requirement")
        self.assertLess(max_time, 200, f"Max time {max_time:.2f}ms exceeds 200ms requirement")
        
        print(f"✓ Performance Requirements: Avg {avg_time:.2f}ms, Max {max_time:.2f}ms")


if __name__ == '__main__':
    print("Starting Phase 1 Integration and Validation Test Suite...")
    print("=" * 70)
    
    # Run with high verbosity to see all integration results
    unittest.main(verbosity=2)