"""
AI Helper v2 - Conversational Coach (NLP-Safe & Context-Resilient)
Intelligent conversational AI for Hearthstone Arena draft coaching

This module implements the ConversationalCoach with comprehensive hardening
features for natural language processing safety, context management, and
resilient conversation handling.

Key Features:
- Context-aware question generation with draft phase intelligence
- Multi-language input detection with graceful fallback
- Conversation memory management with intelligent summarization
- Response safety validation with content filtering
- Personalization based on user skill level and preferences
- Knowledge gap detection and alternative suggestions
- Session boundary management with clean context transitions
"""

import re
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import uuid

from .data_models import (
    DeckState, CardInfo, AIDecision, ArchetypePreference, 
    DraftPhase, ConfidenceLevel, BaseDataModel
)
from .exceptions import (
    AIHelperException, DataValidationError, ConfigurationError,
    ResourceExhaustionError, PerformanceError
)

logger = logging.getLogger(__name__)

# === Conversation Types and Enums ===

class ConversationTone(Enum):
    """Conversation tone settings"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    ENCOURAGING = "encouraging"

class UserSkillLevel(Enum):
    """User skill level classification"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"

class MessageType(Enum):
    """Types of conversation messages"""
    USER_QUESTION = "user_question"
    COACH_RESPONSE = "coach_response"
    SYSTEM_NOTIFICATION = "system_notification"
    SUGGESTION = "suggestion"
    EXPLANATION = "explanation"

class ConversationContext(Enum):
    """Conversation context types"""
    DRAFT_ANALYSIS = "draft_analysis"
    CARD_COMPARISON = "card_comparison"
    STRATEGY_DISCUSSION = "strategy_discussion"
    GENERAL_COACHING = "general_coaching"
    LEARNING = "learning"

class InputSafetyLevel(Enum):
    """Input safety validation levels"""
    SAFE = "safe"
    QUESTIONABLE = "questionable"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

# === Data Models ===

@dataclass
class ConversationMessage(BaseDataModel):
    """
    Individual conversation message with metadata
    
    Usage Example:
        message = ConversationMessage(
            content="Why is Fireball better than Flamestrike here?",
            message_type=MessageType.USER_QUESTION,
            context=ConversationContext.CARD_COMPARISON,
            deck_state=current_deck_state
        )
    """
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    context: ConversationContext = ConversationContext.GENERAL_COACHING
    user_skill_level: UserSkillLevel = UserSkillLevel.INTERMEDIATE
    deck_state: Optional[DeckState] = None
    referenced_cards: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    safety_level: InputSafetyLevel = InputSafetyLevel.SAFE
    language_detected: str = "en"
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = super().to_dict()
        result['message_type'] = self.message_type.value
        result['context'] = self.context.value
        result['user_skill_level'] = self.user_skill_level.value
        result['safety_level'] = self.safety_level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class UserProfile(BaseDataModel):
    """
    User profile for personalization
    
    Usage Example:
        profile = UserProfile(
            skill_level=UserSkillLevel.INTERMEDIATE,
            preferred_tone=ConversationTone.FRIENDLY,
            favorite_archetypes=[ArchetypePreference.TEMPO],
            learning_goals=["Better curve understanding", "Synergy evaluation"]
        )
    """
    skill_level: UserSkillLevel = UserSkillLevel.INTERMEDIATE
    preferred_tone: ConversationTone = ConversationTone.FRIENDLY
    favorite_archetypes: List[ArchetypePreference] = field(default_factory=list)
    learning_goals: List[str] = field(default_factory=list)
    conversation_history_size: int = 50
    language_preference: str = "en"
    coaching_frequency: str = "balanced"  # minimal, balanced, detailed
    show_advanced_explanations: bool = True
    last_interaction: Optional[datetime] = None
    total_interactions: int = 0
    successful_predictions: int = 0
    feedback_ratings: List[int] = field(default_factory=list)
    
    def get_average_rating(self) -> float:
        """Get average feedback rating"""
        if not self.feedback_ratings:
            return 0.0
        return sum(self.feedback_ratings) / len(self.feedback_ratings)
    
    def update_skill_level_adaptive(self):
        """Adaptively update skill level based on interaction patterns"""
        if self.total_interactions < 10:
            return  # Not enough data
            
        success_rate = self.successful_predictions / max(1, self.total_interactions)
        avg_rating = self.get_average_rating()
        
        # Simple adaptive logic
        if success_rate > 0.8 and avg_rating > 4.0:
            if self.skill_level == UserSkillLevel.BEGINNER:
                self.skill_level = UserSkillLevel.INTERMEDIATE
            elif self.skill_level == UserSkillLevel.INTERMEDIATE:
                self.skill_level = UserSkillLevel.ADVANCED
        elif success_rate < 0.4 and avg_rating < 2.0:
            if self.skill_level == UserSkillLevel.ADVANCED:
                self.skill_level = UserSkillLevel.INTERMEDIATE
            elif self.skill_level == UserSkillLevel.INTERMEDIATE:
                self.skill_level = UserSkillLevel.BEGINNER

@dataclass
class ConversationSession(BaseDataModel):
    """
    Conversation session with context management
    
    Usage Example:
        session = ConversationSession(
            session_id="draft_session_123",
            current_deck_state=deck_state,
            messages_history=deque(maxlen=50)
        )
    """
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    current_deck_state: Optional[DeckState] = None
    messages_history: deque = field(default_factory=lambda: deque(maxlen=50))
    context_summary: Dict[str, Any] = field(default_factory=dict)
    active_topics: Set[str] = field(default_factory=set)
    last_activity: datetime = field(default_factory=datetime.now)
    total_exchanges: int = 0
    user_engagement_score: float = 0.5
    
    def add_message(self, message: ConversationMessage):
        """Add message to session history"""
        self.messages_history.append(message)
        self.last_activity = datetime.now()
        self.total_exchanges += 1
        
        # Update engagement score based on message type and content length
        if message.message_type == MessageType.USER_QUESTION:
            engagement_boost = min(0.1, len(message.content) / 1000)
            self.user_engagement_score = min(1.0, self.user_engagement_score + engagement_boost)
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def get_recent_context(self, num_messages: int = 5) -> List[ConversationMessage]:
        """Get recent conversation context"""
        return list(self.messages_history)[-num_messages:]

# === Core ConversationalCoach Implementation ===

class ConversationalCoach:
    """
    Intelligent conversational coach for Hearthstone Arena drafting
    
    Features comprehensive NLP safety, context management, and personalized coaching
    with multiple hardening layers for production-grade reliability.
    
    Usage Example:
        coach = ConversationalCoach()
        response = coach.process_user_input(
            user_input="Should I pick Fireball or Flamestrike?",
            deck_state=current_deck,
            user_profile=user_profile
        )
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Conversational Coach
        
        Args:
            config: Configuration dictionary with coach settings
        """
        self.config = config or self._get_default_config()
        self.logger = logger
        
        # Core components
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.knowledge_base = self._initialize_knowledge_base()
        self.response_templates = self._load_response_templates()
        self.safety_filter = InputSafetyFilter()
        self.context_manager = ConversationContextManager()
        
        # Performance tracking
        self.performance_metrics = {
            'total_interactions': 0,
            'avg_response_time': 0.0,
            'safety_blocks': 0,
            'context_resets': 0,
            'fallback_responses': 0
        }
        
        # Resource monitoring
        self.max_memory_usage = self.config.get('max_memory_mb', 100)
        self.max_sessions = self.config.get('max_concurrent_sessions', 50)
        self.session_timeout = self.config.get('session_timeout_minutes', 30)
        
        logger.info("ConversationalCoach initialized with safety and context management")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_memory_mb': 100,
            'max_concurrent_sessions': 50,
            'session_timeout_minutes': 30,
            'max_input_length': 1000,
            'max_response_length': 2000,
            'enable_learning': True,
            'safety_level': 'medium',
            'context_window_size': 10,
            'response_timeout_seconds': 5.0,
            'enable_multi_language': False,
            'default_language': 'en'
        }
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize the coaching knowledge base"""
        return {
            'card_archetypes': {
                'removal': ['Fireball', 'Flamestrike', 'Polymorph', 'Hex'],
                'early_game': ['Cogmaster', 'Zombie Chow', 'Undertaker'],
                'late_game': ['Ragnaros', 'Ysera', 'Deathwing'],
                'card_draw': ['Arcane Intellect', 'Nourish', 'Sprint'],
                'healing': ['Antique Healbot', 'Earthen Ring Farseer']
            },
            'draft_advice': {
                'early_picks': [
                    "Focus on powerful individual cards",
                    "Don't commit to synergies too early",
                    "Premium removal is always valuable"
                ],
                'mid_picks': [
                    "Start considering curve and synergies",
                    "Look for your deck's identity",
                    "Fill gaps in your mana curve"
                ],
                'late_picks': [
                    "Complete your curve",
                    "Take situational cards if needed",
                    "Consider tech choices for meta"
                ]
            },
            'common_mistakes': [
                "Overvaluing synergy in early picks",
                "Ignoring mana curve balance",
                "Not adapting to offered cards",
                "Undervaluing removal spells"
            ]
        }
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different contexts"""
        return {
            'greeting': [
                "Hello! I'm here to help you draft a great Arena deck. What would you like to know?",
                "Welcome! Ready to build an amazing Arena deck together?",
                "Hi there! I'm your Arena coach. How can I help you today?"
            ],
            'card_comparison': [
                "Let me analyze these cards for your current deck...",
                "Great question! Here's how I'd compare these options:",
                "Looking at your deck state, here's my analysis:"
            ],
            'strategy_advice': [
                "Based on your current deck, here's what I'd recommend:",
                "Your deck is shaping up nicely. Here's my strategic advice:",
                "Let me give you some strategy tips for this stage of your draft:"
            ],
            'encouragement': [
                "You're making great progress with this draft!",
                "Nice choice! That card fits well in your strategy.",
                "Excellent thinking! You're getting the hang of this."
            ],
            'clarification': [
                "Could you clarify what you'd like to know about?",
                "I want to make sure I understand your question correctly.",
                "Let me make sure I'm addressing what you're asking about."
            ],
            'fallback': [
                "That's an interesting question! Let me think about it in the context of your current draft.",
                "I want to give you the best advice possible. Could you provide more details?",
                "Let me help you with that. Here's what I think about your situation:"
            ]
        }
    
    # === P4.1 Core Implementation ===
    
    def process_user_input(
        self, 
        user_input: str,
        session_id: str = None,
        deck_state: DeckState = None,
        user_profile: UserProfile = None,
        context: ConversationContext = ConversationContext.GENERAL_COACHING
    ) -> Dict[str, Any]:
        """
        Process user input and generate appropriate coaching response
        
        Args:
            user_input: The user's question or comment
            session_id: Session identifier for context management
            deck_state: Current draft deck state
            user_profile: User profile for personalization
            context: Conversation context for appropriate responses
            
        Returns:
            Dict containing response, confidence, metadata, and suggestions
        """
        start_time = time.time()
        
        try:
            # P4.1.1: Multi-Language Input Detection with Graceful Fallback
            language_detected = self._detect_language(user_input)
            if language_detected != 'en' and not self.config.get('enable_multi_language', False):
                return self._create_fallback_response(
                    "I currently support English only. Please rephrase your question in English.",
                    MessageType.SYSTEM_NOTIFICATION
                )
            
            # P4.1.2: Input Length Validation & Chunking
            if len(user_input) > self.config.get('max_input_length', 1000):
                user_input = self._chunk_long_input(user_input)
            
            # P4.1.3: Smart Content Filtering
            safety_result = self.safety_filter.validate_input(user_input)
            if safety_result.safety_level == InputSafetyLevel.BLOCKED:
                self.performance_metrics['safety_blocks'] += 1
                return self._create_safety_response(safety_result)
            
            # Session management
            session = self._get_or_create_session(session_id, deck_state)
            user_profile = user_profile or self._get_default_user_profile()
            
            # P4.1.4: Knowledge Gap Detection & Handling
            knowledge_gaps = self._detect_knowledge_gaps(user_input)
            
            # Create conversation message
            message = ConversationMessage(
                content=user_input,
                message_type=MessageType.USER_QUESTION,
                context=context,
                user_skill_level=user_profile.skill_level,
                deck_state=deck_state,
                language_detected=language_detected,
                safety_level=safety_result.safety_level
            )
            
            # P4.1.5: Context Window Management
            self._manage_context_window(session, message)
            
            # Generate response based on context and user profile
            response_data = self._generate_contextual_response(
                message, session, user_profile, knowledge_gaps
            )
            
            # P4.1.6: Response Safety Validation
            validated_response = self._validate_response_safety(response_data)
            
            # Update session and metrics
            session.add_message(message)
            processing_time = (time.time() - start_time) * 1000
            message.processing_time_ms = processing_time
            
            self._update_performance_metrics(processing_time)
            
            return {
                'response': validated_response['content'],
                'confidence': validated_response['confidence'],
                'message_type': validated_response['message_type'],
                'suggestions': validated_response.get('suggestions', []),
                'context_updated': True,
                'processing_time_ms': processing_time,
                'session_id': session.session_id,
                'user_skill_level': user_profile.skill_level.value,
                'knowledge_gaps': knowledge_gaps,
                'safety_level': safety_result.safety_level.value
            }
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            self.performance_metrics['fallback_responses'] += 1
            return self._create_error_response(str(e))
    
    # === P4.1 Hardening Implementation ===
    
    def _detect_language(self, text: str) -> str:
        """
        P4.1.1: Multi-Language Input Detection
        Simple language detection with fallback to English
        """
        # Simple heuristic-based language detection
        english_indicators = ['the', 'and', 'or', 'is', 'are', 'this', 'that', 'card', 'deck']
        spanish_indicators = ['el', 'la', 'y', 'o', 'es', 'son', 'esta', 'carta']
        french_indicators = ['le', 'la', 'et', 'ou', 'est', 'sont', 'cette', 'carte']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        english_score = sum(1 for word in words if word in english_indicators)
        spanish_score = sum(1 for word in words if word in spanish_indicators)
        french_score = sum(1 for word in words if word in french_indicators)
        
        if spanish_score > english_score and spanish_score > french_score:
            return 'es'
        elif french_score > english_score and french_score > spanish_score:
            return 'fr'
        else:
            return 'en'
    
    def _chunk_long_input(self, text: str) -> str:
        """
        P4.1.2: Input Length Validation & Chunking
        Safely truncate long inputs while preserving meaning
        """
        max_length = self.config.get('max_input_length', 1000)
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundaries
        sentences = text.split('.')
        result = ""
        for sentence in sentences:
            if len(result + sentence + ".") <= max_length:
                result += sentence + "."
            else:
                break
        
        if not result:
            # Fallback to simple truncation
            result = text[:max_length-3] + "..."
        
        logger.warning(f"Input truncated from {len(text)} to {len(result)} characters")
        return result
    
    def _detect_knowledge_gaps(self, user_input: str) -> List[str]:
        """
        P4.1.4: Knowledge Gap Detection & Handling
        Identify unknown cards or concepts and suggest alternatives
        """
        knowledge_gaps = []
        
        # Extract potential card names (capitalized words)
        potential_cards = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
        
        for card_name in potential_cards:
            if not self._is_known_card(card_name):
                knowledge_gaps.append(f"Unknown card: {card_name}")
        
        # Check for complex strategy terms
        complex_terms = ['synergy', 'tempo', 'value', 'curve', 'archetype']
        mentioned_terms = [term for term in complex_terms if term.lower() in user_input.lower()]
        
        if mentioned_terms:
            knowledge_gaps.extend([f"Complex concept: {term}" for term in mentioned_terms])
        
        return knowledge_gaps
    
    def _is_known_card(self, card_name: str) -> bool:
        """Check if a card is in our knowledge base"""
        all_cards = []
        for archetype_cards in self.knowledge_base.get('card_archetypes', {}).values():
            all_cards.extend(archetype_cards)
        return card_name in all_cards
    
    def _manage_context_window(self, session: ConversationSession, new_message: ConversationMessage):
        """
        P4.1.5: Context Window Management
        Intelligent summarization when approaching limits
        """
        max_context = self.config.get('context_window_size', 10)
        
        if len(session.messages_history) >= max_context:
            # Summarize older messages
            old_messages = list(session.messages_history)[:max_context//2]
            summary = self._summarize_conversation_context(old_messages)
            session.context_summary.update(summary)
            
            # Clear old messages and reset deque
            session.messages_history = deque(
                list(session.messages_history)[max_context//2:], 
                maxlen=max_context
            )
            
            logger.info(f"Context window managed for session {session.session_id}")
    
    def _summarize_conversation_context(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """Create intelligent summary of conversation context"""
        topics = defaultdict(int)
        card_mentions = defaultdict(int)
        total_messages = len(messages)
        
        for msg in messages:
            # Extract topics
            if msg.context:
                topics[msg.context.value] += 1
            
            # Extract card references
            for card in msg.referenced_cards:
                card_mentions[card] += 1
        
        return {
            'summarized_at': datetime.now().isoformat(),
            'message_count': total_messages,
            'main_topics': dict(topics),
            'frequently_mentioned_cards': dict(card_mentions),
            'engagement_level': sum(len(msg.content) for msg in messages) / max(1, total_messages)
        }
    
    def _validate_response_safety(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        P4.1.6: Response Safety Validation
        Multi-layer response validation before display
        """
        content = response_data.get('content', '')
        
        # Check for potentially problematic content
        safety_issues = []
        
        # Check length
        if len(content) > self.config.get('max_response_length', 2000):
            content = content[:self.config.get('max_response_length', 2000)-3] + "..."
            safety_issues.append("Response truncated for length")
        
        # Check for sensitive information (placeholder)
        sensitive_patterns = [r'\b\d{4}-\d{4}-\d{4}-\d{4}\b']  # Credit card pattern
        for pattern in sensitive_patterns:
            if re.search(pattern, content):
                safety_issues.append("Sensitive information detected")
                content = re.sub(pattern, "[REDACTED]", content)
        
        # Validate structure
        if not content.strip():
            content = "I apologize, but I couldn't generate a proper response. Could you rephrase your question?"
            safety_issues.append("Empty response corrected")
        
        if safety_issues:
            logger.warning(f"Response safety issues resolved: {safety_issues}")
        
        response_data['content'] = content
        response_data['safety_issues'] = safety_issues
        return response_data
    
    # === P4.1 Additional Hardening ===
    
    def _get_or_create_session(self, session_id: str = None, deck_state: DeckState = None) -> ConversationSession:
        """Get existing session or create new one with resource management"""
        # P4.1.8: Session Boundary Detection
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        # Check resource limits
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_session_id = min(self.sessions.keys(), 
                                  key=lambda x: self.sessions[x].last_activity)
            del self.sessions[oldest_session_id]
            logger.warning(f"Removed oldest session {oldest_session_id} due to resource limits")
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                current_deck_state=deck_state
            )
            logger.info(f"Created new conversation session: {session_id}")
        else:
            # P4.1.8: Clean context transitions between drafts
            existing_session = self.sessions[session_id]
            if deck_state and existing_session.current_deck_state:
                if deck_state._id != existing_session.current_deck_state._id:
                    # New draft detected, reset context
                    existing_session.context_summary = {}
                    existing_session.active_topics = set()
                    existing_session.current_deck_state = deck_state
                    self.performance_metrics['context_resets'] += 1
                    logger.info(f"Context reset for new draft in session {session_id}")
        
        return self.sessions[session_id]
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions to manage resources"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def _get_default_user_profile(self) -> UserProfile:
        """Get default user profile"""
        return UserProfile(
            skill_level=UserSkillLevel.INTERMEDIATE,
            preferred_tone=ConversationTone.FRIENDLY
        )
    
    def _generate_contextual_response(
        self, 
        message: ConversationMessage,
        session: ConversationSession,
        user_profile: UserProfile,
        knowledge_gaps: List[str]
    ) -> Dict[str, Any]:
        """Generate contextual response based on all available information"""
        
        # Determine response strategy based on context
        if message.context == ConversationContext.CARD_COMPARISON:
            return self._generate_card_comparison_response(message, session, user_profile)
        elif message.context == ConversationContext.STRATEGY_DISCUSSION:
            return self._generate_strategy_response(message, session, user_profile)
        elif knowledge_gaps:
            return self._generate_knowledge_gap_response(message, knowledge_gaps, user_profile)
        else:
            return self._generate_general_response(message, session, user_profile)
    
    def _generate_card_comparison_response(
        self, 
        message: ConversationMessage, 
        session: ConversationSession,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Generate response for card comparison questions"""
        templates = self.response_templates.get('card_comparison', [])
        base_response = templates[0] if templates else "Let me analyze these cards for you."
        
        # Add personalized analysis based on skill level
        if user_profile.skill_level == UserSkillLevel.BEGINNER:
            analysis = "I'll focus on the basic strengths of each card and how they fit your deck."
        elif user_profile.skill_level == UserSkillLevel.ADVANCED:
            analysis = "I'll provide detailed analysis including synergies, meta considerations, and situational value."
        else:
            analysis = "I'll give you a balanced analysis of both cards' strengths and how they fit your current draft."
        
        content = f"{base_response} {analysis}"
        
        return {
            'content': content,
            'confidence': 0.8,
            'message_type': MessageType.COACH_RESPONSE,
            'suggestions': [
                "Would you like me to explain the mana curve impact?",
                "Should I analyze the synergies with your current cards?"
            ]
        }
    
    def _generate_strategy_response(
        self, 
        message: ConversationMessage,
        session: ConversationSession, 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Generate response for strategy discussions"""
        current_pick = 1
        if session.current_deck_state:
            current_pick = session.current_deck_state.current_pick
        
        # Get appropriate advice based on draft phase
        if current_pick <= 10:
            advice_category = 'early_picks'
        elif current_pick <= 20:
            advice_category = 'mid_picks'  
        else:
            advice_category = 'late_picks'
        
        advice_options = self.knowledge_base['draft_advice'].get(advice_category, [])
        advice = advice_options[0] if advice_options else "Focus on strong individual cards."
        
        content = f"Based on where you are in the draft (pick {current_pick}), here's my strategic advice: {advice}"
        
        return {
            'content': content,
            'confidence': 0.85,
            'message_type': MessageType.COACH_RESPONSE,
            'suggestions': [
                "Would you like specific archetype advice?",
                "Should I analyze your mana curve?"
            ]
        }
    
    def _generate_knowledge_gap_response(
        self, 
        message: ConversationMessage,
        knowledge_gaps: List[str],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Generate response when knowledge gaps are detected"""
        gap_explanations = []
        
        for gap in knowledge_gaps:
            if "Unknown card" in gap:
                gap_explanations.append("I don't have information about that specific card, but I can help you evaluate it based on general principles.")
            elif "Complex concept" in gap:
                gap_explanations.append("That's an advanced concept - let me explain it in simpler terms.")
        
        explanation = " ".join(gap_explanations)
        content = f"I want to make sure I give you the best advice. {explanation} Could you provide more context about what you'd like to know?"
        
        return {
            'content': content,
            'confidence': 0.6,
            'message_type': MessageType.CLARIFICATION,
            'suggestions': [
                "Could you describe the card's stats and abilities?",
                "What specific aspect would you like me to focus on?"
            ]
        }
    
    def _generate_general_response(
        self, 
        message: ConversationMessage,
        session: ConversationSession,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Generate general coaching response"""
        templates = self.response_templates.get('fallback', [])
        content = templates[0] if templates else "I'm here to help with your Arena draft. What would you like to know?"
        
        # Add encouragement based on user engagement
        if session.user_engagement_score > 0.7:
            encouragement = self.response_templates.get('encouragement', ["You're doing great!"])[0]
            content = f"{encouragement} {content}"
        
        return {
            'content': content,
            'confidence': 0.7,
            'message_type': MessageType.COACH_RESPONSE,
            'suggestions': [
                "Ask me about card comparisons",
                "Get strategy advice for your draft",
                "Learn about deck archetypes"
            ]
        }
    
    def _create_fallback_response(self, content: str, message_type: MessageType) -> Dict[str, Any]:
        """Create standardized fallback response"""
        return {
            'response': content,
            'confidence': 0.3,
            'message_type': message_type.value,
            'suggestions': [],
            'context_updated': False,
            'processing_time_ms': 0.0,
            'fallback_used': True
        }
    
    def _create_safety_response(self, safety_result) -> Dict[str, Any]:
        """Create response for safety-blocked input"""
        return {
            'response': "I'm sorry, but I can't process that input. Please rephrase your question about Arena drafting.",
            'confidence': 0.0,
            'message_type': MessageType.SYSTEM_NOTIFICATION.value,
            'suggestions': ["Ask about card comparisons", "Get drafting strategy advice"],
            'context_updated': False,
            'processing_time_ms': 0.0,
            'safety_blocked': True,
            'safety_level': safety_result.safety_level.value
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'response': "I apologize, but I encountered an issue processing your request. Please try again.",
            'confidence': 0.0,
            'message_type': MessageType.SYSTEM_NOTIFICATION.value,
            'suggestions': ["Try rephrasing your question", "Ask about a specific card or strategy"],
            'context_updated': False,
            'processing_time_ms': 0.0,
            'error': True,
            'error_message': error_message
        }
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_interactions'] += 1
        
        # Update running average of response time
        current_avg = self.performance_metrics['avg_response_time']
        total_interactions = self.performance_metrics['total_interactions']
        
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total_interactions - 1) + processing_time) / total_interactions
        )
        
        # Check performance thresholds
        max_response_time = self.config.get('response_timeout_seconds', 5.0) * 1000
        if processing_time > max_response_time:
            logger.warning(f"Response time {processing_time:.2f}ms exceeded threshold {max_response_time}ms")
    
    # === Public API Methods ===
    
    def get_coaching_suggestions(self, deck_state: DeckState, user_profile: UserProfile = None) -> List[str]:
        """
        Get proactive coaching suggestions based on deck state
        
        Args:
            deck_state: Current draft deck state
            user_profile: User profile for personalization
            
        Returns:
            List of coaching suggestions
        """
        suggestions = []
        user_profile = user_profile or self._get_default_user_profile()
        
        # Analyze deck state for suggestions
        if deck_state.draft_phase == DraftPhase.EARLY:
            suggestions.extend([
                "Focus on powerful individual cards rather than synergies",
                "Premium removal spells are always valuable",
                "Don't commit to an archetype too early"
            ])
        elif deck_state.draft_phase == DraftPhase.MID:
            suggestions.extend([
                "Start considering your deck's curve and synergies",
                "Look for cards that define your archetype",
                "Fill gaps in your mana curve"
            ])
        else:  # Late phase
            suggestions.extend([
                "Complete your mana curve with remaining picks",
                "Consider tech cards for common matchups",
                "Don't be afraid of situational cards if needed"
            ])
        
        # Personalize based on skill level
        if user_profile.skill_level == UserSkillLevel.BEGINNER:
            suggestions = [f"💡 {s}" for s in suggestions[:2]]  # Limit suggestions for beginners
        
        return suggestions
    
    def update_user_feedback(self, session_id: str, rating: int, feedback: str = None):
        """
        Update user feedback for learning and improvement
        
        Args:
            session_id: Session identifier
            rating: Rating from 1-5
            feedback: Optional text feedback
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Update engagement based on feedback
            if rating >= 4:
                session.user_engagement_score = min(1.0, session.user_engagement_score + 0.1)
            elif rating <= 2:
                session.user_engagement_score = max(0.0, session.user_engagement_score - 0.1)
            
            logger.info(f"Received feedback for session {session_id}: rating={rating}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            'active_sessions': len(self.sessions),
            'memory_usage_estimate': len(self.sessions) * 50,  # Rough estimate
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def shutdown(self):
        """Clean shutdown of the conversational coach"""
        # Save any important state
        logger.info(f"ConversationalCoach shutting down. Final metrics: {self.get_performance_metrics()}")
        
        # Clear sessions
        self.sessions.clear()
        self.user_profiles.clear()


# === Supporting Classes ===

class InputSafetyFilter:
    """Input safety validation with graduated filtering"""
    
    def __init__(self):
        self.blocked_patterns = [
            r'\b(password|credit card|ssn)\b',
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card pattern
        ]
        self.questionable_patterns = [
            r'\b(hack|cheat|exploit)\b'
        ]
    
    def validate_input(self, text: str) -> 'SafetyResult':
        """Validate input safety with graduated response"""
        text_lower = text.lower()
        
        # Check for blocked content
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                return SafetyResult(InputSafetyLevel.BLOCKED, f"Blocked pattern: {pattern}")
        
        # Check for questionable content
        for pattern in self.questionable_patterns:
            if re.search(pattern, text_lower):
                return SafetyResult(InputSafetyLevel.QUESTIONABLE, f"Questionable pattern: {pattern}")
        
        return SafetyResult(InputSafetyLevel.SAFE, "Input validated as safe")

@dataclass
class SafetyResult:
    """Result of safety validation"""
    safety_level: InputSafetyLevel
    reason: str

class ConversationContextManager:
    """Manage conversation context and transitions"""
    
    def __init__(self):
        self.context_transitions = {
            ConversationContext.GENERAL_COACHING: [
                ConversationContext.CARD_COMPARISON,
                ConversationContext.STRATEGY_DISCUSSION
            ],
            ConversationContext.CARD_COMPARISON: [
                ConversationContext.STRATEGY_DISCUSSION,
                ConversationContext.LEARNING
            ]
        }
    
    def suggest_context_transition(self, current_context: ConversationContext, message: str) -> Optional[ConversationContext]:
        """Suggest appropriate context transition based on message content"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['compare', 'better', 'choose', 'pick']):
            return ConversationContext.CARD_COMPARISON
        elif any(word in message_lower for word in ['strategy', 'plan', 'archetype', 'gameplan']):
            return ConversationContext.STRATEGY_DISCUSSION
        elif any(word in message_lower for word in ['learn', 'explain', 'understand', 'why']):
            return ConversationContext.LEARNING
        
        return None

# Export main components
__all__ = [
    'ConversationalCoach', 'ConversationMessage', 'UserProfile', 'ConversationSession',
    'ConversationTone', 'UserSkillLevel', 'MessageType', 'ConversationContext',
    'InputSafetyLevel', 'InputSafetyFilter', 'ConversationContextManager'
]