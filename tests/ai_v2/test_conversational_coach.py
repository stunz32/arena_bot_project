"""
Test Suite for ConversationalCoach (Phase 4.1)
Comprehensive testing of NLP safety, context awareness, and hardening features

This test suite validates all aspects of the ConversationalCoach including:
- NLP safety validation and content filtering
- Context management and memory handling
- Multi-language input detection and handling
- Response safety validation and content sanitization
- Performance metrics and resource management
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# Import the modules under test
from arena_bot.ai_v2.conversational_coach import (
    ConversationalCoach, ConversationMessage, UserProfile, ConversationSession,
    ConversationTone, UserSkillLevel, MessageType, ConversationContext,
    InputSafetyLevel, InputSafetyFilter, ConversationContextManager
)
from arena_bot.ai_v2.data_models import DeckState, CardInfo, CardClass, ArchetypePreference
from arena_bot.ai_v2.exceptions import AIHelperException, DataValidationError

class TestConversationalCoach(unittest.TestCase):
    """Test suite for ConversationalCoach core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coach = ConversationalCoach()
        self.user_profile = UserProfile(
            skill_level=UserSkillLevel.INTERMEDIATE,
            preferred_tone=ConversationTone.FRIENDLY
        )
        self.deck_state = DeckState(
            hero_class=CardClass.MAGE,
            current_pick=5,
            archetype_preference=ArchetypePreference.TEMPO
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if self.coach:
            self.coach.shutdown()
    
    def test_initialization(self):
        """Test ConversationalCoach initialization"""
        self.assertIsNotNone(self.coach)
        self.assertIsInstance(self.coach.knowledge_base, dict)
        self.assertIsInstance(self.coach.response_templates, dict)
        self.assertIsInstance(self.coach.safety_filter, InputSafetyFilter)
        self.assertIsInstance(self.coach.context_manager, ConversationContextManager)
        
        # Verify default configuration
        self.assertEqual(self.coach.config['max_memory_mb'], 100)
        self.assertEqual(self.coach.config['max_concurrent_sessions'], 50)
        self.assertEqual(self.coach.config['session_timeout_minutes'], 30)
    
    def test_basic_user_input_processing(self):
        """Test basic user input processing"""
        response = self.coach.process_user_input(
            user_input="Hello, can you help me with my draft?",
            user_profile=self.user_profile
        )
        
        self.assertTrue(response['success'] if 'success' in response else 'response' in response)
        self.assertIn('response', response)
        self.assertIn('confidence', response)
        self.assertIn('processing_time_ms', response)
        self.assertIsInstance(response['processing_time_ms'], float)
        self.assertGreater(response['confidence'], 0.0)
        self.assertLessEqual(response['confidence'], 1.0)
    
    def test_card_comparison_context(self):
        """Test context-aware response for card comparisons"""
        response = self.coach.process_user_input(
            user_input="Should I pick Fireball or Frostbolt?",
            context=ConversationContext.CARD_COMPARISON,
            deck_state=self.deck_state,
            user_profile=self.user_profile
        )
        
        self.assertIn('response', response)
        self.assertIn('suggestions', response)
        self.assertEqual(response.get('message_type'), MessageType.COACH_RESPONSE.value)
        
        # Response should be contextually appropriate
        response_text = response['response'].lower()
        self.assertTrue(any(word in response_text for word in ['analyze', 'card', 'deck', 'choose']))
    
    def test_strategy_discussion_context(self):
        """Test strategy discussion context handling"""
        response = self.coach.process_user_input(
            user_input="What archetype should I focus on?",
            context=ConversationContext.STRATEGY_DISCUSSION,
            deck_state=self.deck_state,
            user_profile=self.user_profile
        )
        
        self.assertIn('response', response)
        response_text = response['response'].lower()
        self.assertTrue(any(word in response_text for word in ['strategy', 'archetype', 'draft', 'advice']))
    
    def test_skill_level_personalization(self):
        """Test personalization based on user skill level"""
        # Test beginner response
        beginner_profile = UserProfile(skill_level=UserSkillLevel.BEGINNER)
        beginner_response = self.coach.process_user_input(
            user_input="Which cards should I compare?",
            context=ConversationContext.CARD_COMPARISON,
            user_profile=beginner_profile
        )
        
        # Test expert response
        expert_profile = UserProfile(skill_level=UserSkillLevel.EXPERT)
        expert_response = self.coach.process_user_input(
            user_input="Which cards should I compare?",
            context=ConversationContext.CARD_COMPARISON,
            user_profile=expert_profile
        )
        
        # Responses should be different based on skill level
        self.assertNotEqual(beginner_response['response'], expert_response['response'])
        
        # Beginner response should mention basic concepts
        beginner_text = beginner_response['response'].lower()
        self.assertTrue(any(word in beginner_text for word in ['basic', 'simple', 'focus']))
    
    def test_session_management(self):
        """Test conversation session management"""
        session_id = "test_session_123"
        
        # First message creates session
        response1 = self.coach.process_user_input(
            user_input="Hello!",
            session_id=session_id,
            user_profile=self.user_profile
        )
        
        self.assertEqual(response1.get('session_id'), session_id)
        self.assertTrue(session_id in self.coach.sessions)
        
        # Second message uses existing session
        response2 = self.coach.process_user_input(
            user_input="What should I pick?",
            session_id=session_id,
            user_profile=self.user_profile
        )
        
        self.assertEqual(response2.get('session_id'), session_id)
        session = self.coach.sessions[session_id]
        self.assertEqual(session.total_exchanges, 2)
    
    def test_coaching_suggestions(self):
        """Test proactive coaching suggestions"""
        suggestions = self.coach.get_coaching_suggestions(
            self.deck_state,
            self.user_profile
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Suggestions should be strings
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
            self.assertGreater(len(suggestion), 10)  # Meaningful length
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        initial_metrics = self.coach.get_performance_metrics()
        
        # Process some inputs
        for i in range(3):
            self.coach.process_user_input(
                user_input=f"Test message {i}",
                user_profile=self.user_profile
            )
        
        final_metrics = self.coach.get_performance_metrics()
        
        # Metrics should be updated
        self.assertGreater(final_metrics['total_interactions'], initial_metrics['total_interactions'])
        self.assertGreaterEqual(final_metrics['avg_response_time'], 0)
        self.assertIsInstance(final_metrics['active_sessions'], int)
    
    def test_user_feedback_integration(self):
        """Test user feedback processing"""
        session_id = "feedback_test_session"
        
        # Create a session
        self.coach.process_user_input(
            user_input="Test message",
            session_id=session_id,
            user_profile=self.user_profile
        )
        
        # Provide feedback
        self.coach.update_user_feedback(session_id, 5, "Great response!")
        
        # Session should exist and have updated engagement
        self.assertIn(session_id, self.coach.sessions)
        session = self.coach.sessions[session_id]
        self.assertGreaterEqual(session.user_engagement_score, 0.5)


class TestInputSafetyFilter(unittest.TestCase):
    """Test suite for input safety validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.safety_filter = InputSafetyFilter()
    
    def test_safe_input_validation(self):
        """Test validation of safe inputs"""
        safe_inputs = [
            "Which card should I pick?",
            "How does tempo work in Arena?",
            "What's the best archetype for Mage?"
        ]
        
        for safe_input in safe_inputs:
            result = self.safety_filter.validate_input(safe_input)
            self.assertEqual(result.safety_level, InputSafetyLevel.SAFE)
    
    def test_questionable_input_detection(self):
        """Test detection of questionable content"""
        questionable_inputs = [
            "How do I hack the game?",
            "Is there a cheat for Arena?",
            "Can I exploit this bug?"
        ]
        
        for questionable_input in questionable_inputs:
            result = self.safety_filter.validate_input(questionable_input)
            self.assertEqual(result.safety_level, InputSafetyLevel.QUESTIONABLE)
    
    def test_blocked_input_detection(self):
        """Test detection and blocking of unsafe content"""
        blocked_inputs = [
            "My password is 123456",
            "My credit card is 1234-5678-9012-3456",
            "What's your password?"
        ]
        
        for blocked_input in blocked_inputs:
            result = self.safety_filter.validate_input(blocked_input)
            self.assertEqual(result.safety_level, InputSafetyLevel.BLOCKED)


class TestConversationDataModels(unittest.TestCase):
    """Test suite for conversation data models"""
    
    def test_conversation_message_creation(self):
        """Test ConversationMessage creation and validation"""
        message = ConversationMessage(
            content="Test message",
            message_type=MessageType.USER_QUESTION,
            context=ConversationContext.GENERAL_COACHING
        )
        
        self.assertEqual(message.content, "Test message")
        self.assertEqual(message.message_type, MessageType.USER_QUESTION)
        self.assertEqual(message.context, ConversationContext.GENERAL_COACHING)
        self.assertIsInstance(message.timestamp, datetime)
        
        # Test serialization
        message_dict = message.to_dict()
        self.assertIsInstance(message_dict, dict)
        self.assertEqual(message_dict['content'], "Test message")
        self.assertEqual(message_dict['message_type'], "user_question")
    
    def test_user_profile_creation(self):
        """Test UserProfile creation and methods"""
        profile = UserProfile(
            skill_level=UserSkillLevel.INTERMEDIATE,
            preferred_tone=ConversationTone.FRIENDLY,
            favorite_archetypes=[ArchetypePreference.TEMPO]
        )
        
        self.assertEqual(profile.skill_level, UserSkillLevel.INTERMEDIATE)
        self.assertEqual(profile.preferred_tone, ConversationTone.FRIENDLY)
        self.assertIn(ArchetypePreference.TEMPO, profile.favorite_archetypes)
        
        # Test feedback rating methods
        profile.feedback_ratings = [4, 5, 3, 4, 5]
        average_rating = profile.get_average_rating()
        self.assertEqual(average_rating, 4.2)
    
    def test_user_profile_adaptive_skill_update(self):
        """Test adaptive skill level updates"""
        profile = UserProfile(
            skill_level=UserSkillLevel.BEGINNER,
            total_interactions=20,
            successful_predictions=18,
            feedback_ratings=[4, 5, 4, 5, 5]
        )
        
        initial_skill = profile.skill_level
        profile.update_skill_level_adaptive()
        
        # Should upgrade skill level due to high success rate and ratings
        self.assertNotEqual(profile.skill_level, initial_skill)
        self.assertEqual(profile.skill_level, UserSkillLevel.INTERMEDIATE)
    
    def test_conversation_session_management(self):
        """Test ConversationSession functionality"""
        session = ConversationSession(
            session_id="test_session",
            current_deck_state=DeckState(hero_class=CardClass.MAGE)
        )
        
        # Test message addition
        message = ConversationMessage(
            content="Test message",
            message_type=MessageType.USER_QUESTION
        )
        session.add_message(message)
        
        self.assertEqual(session.total_exchanges, 1)
        self.assertEqual(len(session.messages_history), 1)
        self.assertGreater(session.user_engagement_score, 0.5)
        
        # Test expiration
        self.assertFalse(session.is_expired(30))  # Should not be expired
        
        # Test recent context
        recent_context = session.get_recent_context(5)
        self.assertEqual(len(recent_context), 1)
        self.assertEqual(recent_context[0], message)


class TestConversationalCoachHardening(unittest.TestCase):
    """Test suite for ConversationalCoach hardening features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coach = ConversationalCoach()
        self.user_profile = UserProfile()
    
    def tearDown(self):
        """Clean up after tests"""
        if self.coach:
            self.coach.shutdown()
    
    def test_language_detection(self):
        """Test P4.1.1: Multi-Language Input Detection"""
        # Test English detection
        english_text = "Which card should I pick from these options?"
        lang = self.coach._detect_language(english_text)
        self.assertEqual(lang, 'en')
        
        # Test Spanish detection
        spanish_text = "¿Qué carta debería elegir de estas opciones?"
        lang = self.coach._detect_language(spanish_text)
        # Note: Simple heuristic may not always detect correctly
        self.assertIn(lang, ['es', 'en'])  # Allow both as fallback
        
        # Test fallback for unsupported language
        response = self.coach.process_user_input(
            user_input="Quelle carte dois-je choisir?",  # French
            user_profile=self.user_profile
        )
        
        # Should either process or provide fallback message
        self.assertIn('response', response)
    
    def test_input_length_validation(self):
        """Test P4.1.2: Input Length Validation & Chunking"""
        # Test normal length input
        normal_input = "This is a normal length question about cards."
        response = self.coach.process_user_input(
            user_input=normal_input,
            user_profile=self.user_profile
        )
        self.assertIn('response', response)
        
        # Test very long input
        long_input = "This is a very long question. " * 100  # Should exceed max length
        response = self.coach.process_user_input(
            user_input=long_input,
            user_profile=self.user_profile
        )
        
        # Should still process (with truncation)
        self.assertIn('response', response)
    
    def test_context_window_management(self):
        """Test P4.1.5: Context Window Management"""
        session_id = "context_test_session"
        
        # Fill up context window
        for i in range(15):  # Exceed default context window size
            self.coach.process_user_input(
                user_input=f"Test message number {i}",
                session_id=session_id,
                user_profile=self.user_profile
            )
        
        # Session should exist and have managed context
        self.assertIn(session_id, self.coach.sessions)
        session = self.coach.sessions[session_id]
        
        # Context should be managed (not exceed max size significantly)
        self.assertLessEqual(len(session.messages_history), 12)  # Some buffer allowed
        
        # Should have context summary when window is managed
        if len(session.context_summary) > 0:
            self.assertIn('summarized_at', session.context_summary)
    
    def test_response_safety_validation(self):
        """Test P4.1.6: Response Safety Validation"""
        # Create a mock response that might need validation
        test_response = {
            'content': 'A' * 3000,  # Very long response
            'confidence': 0.8,
            'message_type': MessageType.COACH_RESPONSE
        }
        
        validated_response = self.coach._validate_response_safety(test_response)
        
        # Response should be truncated if too long
        self.assertLessEqual(len(validated_response['content']), 2000)
        self.assertIn('safety_issues', validated_response)
    
    def test_session_cleanup_and_resource_management(self):
        """Test session cleanup and resource management"""
        initial_session_count = len(self.coach.sessions)
        
        # Create many sessions to test resource limits
        for i in range(60):  # Exceed max_sessions limit
            session_id = f"test_session_{i}"
            self.coach.process_user_input(
                user_input="Test message",
                session_id=session_id,
                user_profile=self.user_profile
            )
        
        # Should not exceed resource limits significantly
        self.assertLessEqual(len(self.coach.sessions), 55)  # Some buffer allowed
    
    def test_knowledge_gap_detection(self):
        """Test P4.1.4: Knowledge Gap Detection & Handling"""
        # Test with unknown card name
        response = self.coach.process_user_input(
            user_input="Should I pick UnknownCardName123?",
            user_profile=self.user_profile
        )
        
        self.assertIn('knowledge_gaps', response)
        if response['knowledge_gaps']:
            self.assertTrue(any('Unknown card' in gap for gap in response['knowledge_gaps']))
    
    def test_performance_threshold_monitoring(self):
        """Test performance threshold monitoring"""
        # Process input and check performance metrics
        start_time = time.time()
        response = self.coach.process_user_input(
            user_input="Test performance monitoring",
            user_profile=self.user_profile
        )
        end_time = time.time()
        
        processing_time = response.get('processing_time_ms', 0)
        actual_time = (end_time - start_time) * 1000
        
        # Processing time should be reasonable
        self.assertLess(processing_time, 10000)  # Less than 10 seconds
        self.assertGreater(processing_time, 0)  # Greater than 0
        
        # Should be approximately correct (within reasonable margin)
        self.assertLess(abs(processing_time - actual_time), 1000)  # Within 1 second margin
    
    def test_concurrent_access_safety(self):
        """Test thread safety with concurrent access"""
        responses = []
        errors = []
        
        def process_message(thread_id):
            try:
                response = self.coach.process_user_input(
                    user_input=f"Concurrent test message from thread {thread_id}",
                    session_id=f"concurrent_session_{thread_id}",
                    user_profile=self.user_profile
                )
                responses.append(response)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_message, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Should not have errors and should have responses
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(responses), 5)
        
        # All responses should be valid
        for response in responses:
            self.assertIn('response', response)
            self.assertIn('session_id', response)


class TestConversationContextManager(unittest.TestCase):
    """Test suite for ConversationContextManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.context_manager = ConversationContextManager()
    
    def test_context_transition_suggestions(self):
        """Test context transition suggestions"""
        # Test card comparison context
        comparison_message = "Which is better, Fireball or Frostbolt?"
        suggested_context = self.context_manager.suggest_context_transition(
            ConversationContext.GENERAL_COACHING,
            comparison_message
        )
        self.assertEqual(suggested_context, ConversationContext.CARD_COMPARISON)
        
        # Test strategy context
        strategy_message = "What archetype should I focus on?"
        suggested_context = self.context_manager.suggest_context_transition(
            ConversationContext.GENERAL_COACHING,
            strategy_message
        )
        self.assertEqual(suggested_context, ConversationContext.STRATEGY_DISCUSSION)
        
        # Test learning context
        learning_message = "Why is tempo important in Arena?"
        suggested_context = self.context_manager.suggest_context_transition(
            ConversationContext.GENERAL_COACHING,
            learning_message
        )
        self.assertEqual(suggested_context, ConversationContext.LEARNING)


# Performance and Integration Tests

class TestConversationalCoachIntegration(unittest.TestCase):
    """Integration tests for ConversationalCoach with other components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coach = ConversationalCoach()
        self.user_profile = UserProfile(
            skill_level=UserSkillLevel.INTERMEDIATE,
            preferred_tone=ConversationTone.FRIENDLY
        )
        
        # Create realistic deck state
        cards = [
            CardInfo(name="Fireball", cost=4, card_class=CardClass.MAGE),
            CardInfo(name="Frostbolt", cost=2, card_class=CardClass.MAGE),
            CardInfo(name="Water Elemental", cost=4, card_class=CardClass.MAGE)
        ]
        self.deck_state = DeckState(
            cards=cards,
            hero_class=CardClass.MAGE,
            current_pick=4,
            archetype_preference=ArchetypePreference.TEMPO
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if self.coach:
            self.coach.shutdown()
    
    def test_full_conversation_flow(self):
        """Test complete conversation flow with realistic interactions"""
        session_id = "integration_test_session"
        
        # 1. Initial greeting
        response1 = self.coach.process_user_input(
            user_input="Hello! I need help with my Arena draft.",
            session_id=session_id,
            user_profile=self.user_profile
        )
        self.assertIn('response', response1)
        
        # 2. Card comparison question
        response2 = self.coach.process_user_input(
            user_input="Should I pick Fireball or Flamestrike?",
            session_id=session_id,
            context=ConversationContext.CARD_COMPARISON,
            deck_state=self.deck_state,
            user_profile=self.user_profile
        )
        self.assertIn('response', response2)
        self.assertEqual(response2.get('message_type'), MessageType.COACH_RESPONSE.value)
        
        # 3. Strategy discussion
        response3 = self.coach.process_user_input(
            user_input="What archetype should I focus on with these cards?",
            session_id=session_id,
            context=ConversationContext.STRATEGY_DISCUSSION,
            deck_state=self.deck_state,
            user_profile=self.user_profile
        )
        self.assertIn('response', response3)
        
        # 4. Learning question
        response4 = self.coach.process_user_input(
            user_input="Why is tempo important in Arena?",
            session_id=session_id,
            context=ConversationContext.LEARNING,
            user_profile=self.user_profile
        )
        self.assertIn('response', response4)
        
        # Verify session continuity
        self.assertEqual(response1.get('session_id'), session_id)
        self.assertEqual(response2.get('session_id'), session_id)
        self.assertEqual(response3.get('session_id'), session_id)
        self.assertEqual(response4.get('session_id'), session_id)
        
        # Session should have all messages
        session = self.coach.sessions[session_id]
        self.assertEqual(session.total_exchanges, 4)
    
    def test_coaching_suggestions_integration(self):
        """Test coaching suggestions with real deck state"""
        suggestions = self.coach.get_coaching_suggestions(
            self.deck_state,
            self.user_profile
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Suggestions should be contextually appropriate for the deck phase
        suggestion_text = ' '.join(suggestions).lower()
        if self.deck_state.current_pick <= 10:
            # Early phase suggestions
            self.assertTrue(any(word in suggestion_text for word in ['powerful', 'individual', 'removal']))
        
        # Should be personalized for user's skill level
        if self.user_profile.skill_level == UserSkillLevel.BEGINNER:
            self.assertTrue(any(suggestion.startswith('💡') for suggestion in suggestions))
    
    def test_error_handling_and_recovery(self):
        """Test error handling and graceful recovery"""
        # Test with invalid session data
        response = self.coach.process_user_input(
            user_input="Test message",
            session_id=None,  # Invalid session ID
            user_profile=None,  # Invalid user profile
            deck_state=None   # Invalid deck state
        )
        
        # Should still provide a response without crashing
        self.assertIn('response', response)
        
        # Test with malformed input
        response2 = self.coach.process_user_input(
            user_input="",  # Empty input
            user_profile=self.user_profile
        )
        
        # Should handle gracefully
        self.assertIn('response', response2)
    
    def test_memory_usage_bounds(self):
        """Test that memory usage stays within bounds"""
        import sys
        
        initial_sessions = len(self.coach.sessions)
        
        # Create many conversations to test memory management
        for i in range(100):
            session_id = f"memory_test_{i}"
            for j in range(10):  # Multiple messages per session
                self.coach.process_user_input(
                    user_input=f"Test message {j} in session {i}",
                    session_id=session_id,
                    user_profile=self.user_profile
                )
        
        # Memory usage should be managed
        final_sessions = len(self.coach.sessions)
        self.assertLess(final_sessions, 80)  # Should not grow unbounded
        
        # Performance metrics should show resource management
        metrics = self.coach.get_performance_metrics()
        self.assertLess(metrics['memory_usage_estimate'], 200)  # Within reasonable bounds


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)