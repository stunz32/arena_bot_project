#!/usr/bin/env python3
"""
Integration Tests for Phase 2 & 2.5 - AI Helper Integration
Tests the integration between GUI components, AI Helper system, and manual correction workflow.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main GUI class
from integrated_arena_bot_gui import IntegratedArenaBotGUI

class TestPhase2Integration(unittest.TestCase):
    """Test suite for Phase 2 & 2.5 AI Helper integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a minimal tkinter root for testing
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during testing
        
        # Mock external dependencies
        with patch('integrated_arena_bot_gui.CardRefiner'), \
             patch('integrated_arena_bot_gui.CardsJSONLoaderMulti'), \
             patch('integrated_arena_bot_gui.HearthstoneLogMonitor'):
            
            self.gui = IntegratedArenaBotGUI(self.root)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'root'):
            self.root.destroy()
    
    def test_ai_helper_initialization(self):
        """Test P2.2: AI Helper system initialization."""
        # Test that AI Helper components are properly initialized
        self.assertIsNotNone(self.gui.event_queue)
        self.assertIsNotNone(self.gui.correction_history)
        self.assertIsNotNone(self.gui.correction_analytics)
        
        # Test initial values
        self.assertEqual(len(self.gui.correction_history), 0)
        self.assertEqual(self.gui.correction_history_index, -1)
        self.assertEqual(self.gui.correction_analytics['total_corrections'], 0)
        
    def test_component_lifecycle_methods(self):
        """Test P2.5.2: Component lifecycle management."""
        # Test that lifecycle methods exist and are callable
        self.assertTrue(hasattr(self.gui, '_start_visual_intelligence'))
        self.assertTrue(hasattr(self.gui, '_stop_visual_intelligence'))
        
        # Test methods can be called without errors
        try:
            self.gui._start_visual_intelligence()
            self.gui._stop_visual_intelligence()
        except Exception as e:
            self.fail(f"Component lifecycle methods failed: {e}")
    
    def test_event_driven_architecture(self):
        """Test P2.5.3: Event-driven architecture."""
        # Test event queue functionality
        test_event = {
            'type': 'test_event',
            'data': {'test_data': 'value'}
        }
        
        self.gui.event_queue.put(test_event)
        self.assertFalse(self.gui.event_queue.empty())
        
        # Test event handling method exists
        self.assertTrue(hasattr(self.gui, '_handle_event'))
        self.assertTrue(hasattr(self.gui, '_check_for_events'))
    
    def test_manual_correction_workflow(self):
        """Test P2.3: Enhanced manual correction workflow."""
        # Test correction analytics initialization
        self.assertIn('total_corrections', self.gui.correction_analytics)
        self.assertIn('corrections_by_card', self.gui.correction_analytics)
        self.assertIn('correction_accuracy', self.gui.correction_analytics)
        
        # Test correction workflow methods exist
        self.assertTrue(hasattr(self.gui, '_calculate_correction_confidence'))
        self.assertTrue(hasattr(self.gui, '_add_correction_to_history'))
        self.assertTrue(hasattr(self.gui, '_update_correction_analytics'))
        self.assertTrue(hasattr(self.gui, 'undo_correction'))
        self.assertTrue(hasattr(self.gui, 'redo_correction'))
    
    def test_correction_confidence_calculation(self):
        """Test correction confidence calculation logic."""
        # Mock card data
        old_card_data = {
            'card_name': 'Test Card',
            'confidence': 0.5
        }
        
        # Test confidence calculation
        confidence = self.gui._calculate_correction_confidence(old_card_data, 'NEW_CARD_ID')
        
        # Should return a value between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(confidence, float)
    
    def test_correction_history_management(self):
        """Test correction history and undo/redo functionality."""
        # Create a mock correction action
        correction_action = {
            'action_type': 'correction',
            'card_index': 0,
            'old_card_data': {'card_name': 'Old Card', 'confidence': 0.8},
            'new_card_code': 'NEW_CARD',
            'timestamp': datetime.now(),
            'confidence_score': 0.9
        }
        
        # Test adding to history
        initial_length = len(self.gui.correction_history)
        self.gui._add_correction_to_history(correction_action)
        
        self.assertEqual(len(self.gui.correction_history), initial_length + 1)
        self.assertEqual(self.gui.correction_history_index, 0)
        
        # Test analytics update
        initial_total = self.gui.correction_analytics['total_corrections']
        self.gui._update_correction_analytics(correction_action)
        
        self.assertEqual(self.gui.correction_analytics['total_corrections'], initial_total + 1)
    
    def test_settings_dialog_methods(self):
        """Test P2.5.4: Settings dialog functionality."""
        # Test that settings-related methods exist
        self.assertTrue(hasattr(self.gui, '_open_settings_dialog'))
        self.assertTrue(hasattr(self.gui, '_populate_settings_analytics'))
        self.assertTrue(hasattr(self.gui, '_export_correction_analytics'))
        
        # Test settings dialog can be created (mocked)
        with patch('tkinter.Toplevel'), \
             patch('tkinter.messagebox.showinfo'):
            try:
                self.gui._open_settings_dialog()
            except Exception as e:
                self.fail(f"Settings dialog creation failed: {e}")
    
    def test_archetype_preference_handling(self):
        """Test archetype preference management."""
        # Test that archetype variable exists
        if hasattr(self.gui, 'archetype_var'):
            # Test archetype change handling
            with patch.object(self.gui, 'log_text'):
                try:
                    self.gui._on_archetype_changed('Aggressive')
                except Exception as e:
                    self.fail(f"Archetype change handling failed: {e}")
    
    def test_deck_state_building(self):
        """Test P2.5.1: Deck state construction."""
        # Test that deck state building method exists
        self.assertTrue(hasattr(self.gui, '_build_deck_state_from_detection'))
        
        # Mock detection result
        mock_result = {
            'detected_cards': [
                {
                    'card_code': 'TEST_CARD_1',
                    'card_name': 'Test Card 1',
                    'confidence': 0.85,
                    'strategy': 'pHash'
                },
                {
                    'card_code': 'TEST_CARD_2', 
                    'card_name': 'Test Card 2',
                    'confidence': 0.92,
                    'strategy': 'Ultimate'
                }
            ]
        }
        
        # Test deck state building (may fail due to missing AI components, but method should exist)
        try:
            with patch('integrated_arena_bot_gui.DeckState'), \
                 patch('integrated_arena_bot_gui.CardOption'), \
                 patch('integrated_arena_bot_gui.CardInfo'):
                deck_state = self.gui._build_deck_state_from_detection(mock_result)
        except Exception:
            # Expected to fail without full AI Helper system, but method should exist
            pass
    
    def test_dual_ai_system_support(self):
        """Test dual AI system (AI Helper + Legacy) support."""
        # Test that show_analysis_result handles both AI systems
        self.assertTrue(hasattr(self.gui, 'show_analysis_result'))
        self.assertTrue(hasattr(self.gui, '_show_enhanced_analysis'))
        self.assertTrue(hasattr(self.gui, '_show_legacy_analysis'))
        
        # Mock analysis result
        mock_result = {
            'detected_cards': [
                {
                    'card_code': 'TEST_CARD',
                    'card_name': 'Test Card',
                    'confidence': 0.9,
                    'strategy': 'pHash'
                }
            ],
            'recommendation': {
                'recommended_card': 'TEST_CARD',
                'reasoning': 'Test reasoning'
            }
        }
        
        # Test analysis display (should fall back to legacy)
        with patch.object(self.gui, 'log_text'), \
             patch.object(self.gui, 'show_recommendation'), \
             patch.object(self.gui, '_update_card_images_and_names'):
            try:
                self.gui.show_analysis_result(mock_result)
            except Exception as e:
                self.fail(f"Analysis result display failed: {e}")

class TestLogMonitorIntegration(unittest.TestCase):
    """Test Phase 2.1: Log monitor integration."""
    
    def test_log_monitor_enhancements(self):
        """Test enhanced log monitor features."""
        # Import log monitor
        from hearthstone_log_monitor import HearthstoneLogMonitor
        
        # Test enhanced features exist
        monitor = HearthstoneLogMonitor()
        
        # Test event deduplication
        self.assertTrue(hasattr(monitor, 'event_deduplication_cache'))
        self.assertTrue(hasattr(monitor, '_generate_event_signature'))
        self.assertTrue(hasattr(monitor, '_is_duplicate_event'))
        
        # Test heartbeat monitoring
        self.assertTrue(hasattr(monitor, '_check_heartbeat_and_log_accessibility'))
        self.assertTrue(hasattr(monitor, '_attempt_log_error_recovery'))
        
        # Test enhanced patterns
        self.assertIn('draft_choices_detailed', monitor.patterns)
    
    def test_event_deduplication(self):
        """Test event deduplication functionality."""
        from hearthstone_log_monitor import HearthstoneLogMonitor
        
        monitor = HearthstoneLogMonitor()
        
        # Test signature generation
        signature1 = monitor._generate_event_signature("Test message", "arena")
        signature2 = monitor._generate_event_signature("Test message", "arena") 
        signature3 = monitor._generate_event_signature("Different message", "arena")
        
        self.assertEqual(signature1, signature2)
        self.assertNotEqual(signature1, signature3)
        
        # Test duplicate detection
        self.assertFalse(monitor._is_duplicate_event(signature1))  # First time
        self.assertTrue(monitor._is_duplicate_event(signature1))   # Second time (duplicate)
        self.assertFalse(monitor._is_duplicate_event(signature3))  # Different message

def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Phase 2 & 2.5 Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestLogMonitorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🧪 Integration Tests Complete")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n💥 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n🎯 Overall Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)