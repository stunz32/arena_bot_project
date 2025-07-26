"""
Conversational Coach - Enhanced Hero-Aware Socratic Teaching AI

Provides interactive coaching through natural language understanding
and hero-specific templated response generation for educational draft guidance.
Enhanced with comprehensive hero-aware questioning and strategic advice.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from .data_models import AIDecision, DeckState

# Import for accessing other AI v2 components - using relative imports to avoid circular dependencies
from .grandmaster_advisor import GrandmasterAdvisor
from .hero_selector import HeroSelectionAdvisor


class ConversationalCoach:
    """
    Enhanced Interactive coaching system with hero-aware analysis.
    
    Provides Socratic questioning and educational explanations tailored to
    specific heroes, helping users understand hero-specific draft strategy
    and improve their decision-making with personalized guidance.
    """
    
    def __init__(self):
        """Initialize the enhanced conversational coach."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI v2 components for enhanced analysis
        self.grandmaster_advisor = None  # Will be injected when available
        self.hero_selector = None  # Will be injected when available
        
        # Rule-based NLU patterns (enhanced)
        self.intent_patterns = self._build_enhanced_intent_patterns()
        
        # Response templates (hero-aware)
        self.response_templates = self._build_hero_aware_response_templates()
        
        # Hero-specific coaching profiles
        self.hero_coaching_profiles = self._build_hero_coaching_profiles()
        
        # Socratic questioning patterns
        self.socratic_patterns = self._build_socratic_questioning_patterns()
        
        # Conversation context tracking
        self.conversation_history = []
        self.current_hero_class = None
        self.user_skill_level = "intermediate"  # Can be set based on user performance
        
        # Educational goals tracking
        self.educational_objectives = {
            'curve_understanding': False,
            'archetype_recognition': False,
            'synergy_awareness': False,
            'meta_knowledge': False,
            'hero_strengths': False
        }
        
        self.logger.info("Enhanced ConversationalCoach initialized with hero-aware capabilities")
    
    def process_user_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Process natural language query and generate hero-aware educational response.
        
        Args:
            query: User's question or statement
            context: Current draft context (AIDecision, DeckState, hero_class, etc.)
            
        Returns:
            Educational response with hero-specific Socratic questioning
        """
        try:
            # Update current context
            self._update_context(context)
            
            # Record conversation
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_query': query,
                'context': context.get('draft_stage', 'unknown')
            })
            
            # Classify intent with enhanced patterns
            intent = self._classify_enhanced_intent(query, context)
            
            # Generate hero-aware response
            response = self._generate_hero_aware_response(intent, query, context)
            
            # Add follow-up Socratic question if appropriate
            follow_up = self._generate_contextual_follow_up(intent, context)
            if follow_up:
                response += f" {follow_up}"
            
            # Update educational objectives based on interaction
            self._update_educational_objectives(intent, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing user query: {e}")
            return self._generate_fallback_response(query, context)
    
    def generate_socratic_question(self, hovered_card_index: int, 
                                 ai_decision: AIDecision, 
                                 deck_state: Optional[DeckState] = None) -> str:
        """
        Generate hero-specific Socratic question when user hovers over a card.
        
        Designed to make users think about the strategic implications
        specific to their hero choice rather than just telling them what to pick.
        """
        try:
            if hovered_card_index >= len(ai_decision.all_offered_cards_analysis):
                return "What factors are most important when evaluating this card?"
            
            card_analysis = ai_decision.all_offered_cards_analysis[hovered_card_index]
            card_id = card_analysis.get("card_id", "Unknown")
            
            # Get hero-specific context
            hero_class = deck_state.hero_class if deck_state else self.current_hero_class
            
            # Generate hero-aware question
            if hero_class:
                hero_question = self._generate_hero_specific_question(
                    card_id, hero_class, card_analysis, deck_state
                )
                if hero_question:
                    return hero_question
            
            # Get recommended pick context
            recommended_index = ai_decision.recommended_pick_index
            is_recommended = (hovered_card_index == recommended_index)
            
            # Generate contextual question based on card position
            if is_recommended:
                return self._generate_recommended_card_question(card_id, card_analysis, hero_class)
            else:
                return self._generate_alternative_card_question(
                    card_id, card_analysis, recommended_index, ai_decision, hero_class
                )
                
        except Exception as e:
            self.logger.warning(f"Error generating Socratic question: {e}")
            return f"What strategic value does this card bring to your {self.current_hero_class or 'deck'}?"
    
    # === ENHANCED HERO-AWARE METHODS ===
    
    def set_ai_components(self, grandmaster_advisor: GrandmasterAdvisor, 
                         hero_selector: HeroSelectionAdvisor) -> None:
        """Inject AI v2 components for enhanced analysis."""
        self.grandmaster_advisor = grandmaster_advisor
        self.hero_selector = hero_selector
        self.logger.info("AI v2 components injected into ConversationalCoach")
    
    def set_user_skill_level(self, skill_level: str) -> None:
        """Set user skill level for personalized coaching."""
        self.user_skill_level = skill_level
        self.logger.info(f"User skill level set to: {skill_level}")
    
    def _update_context(self, context: Dict[str, Any]) -> None:
        """Update current conversation context."""
        try:
            if 'hero_class' in context:
                self.current_hero_class = context['hero_class']
            
            if 'deck_state' in context and context['deck_state']:
                self.current_hero_class = context['deck_state'].hero_class
                
        except Exception as e:
            self.logger.warning(f"Error updating context: {e}")
    
    def _classify_enhanced_intent(self, query: str, context: Dict[str, Any]) -> str:
        """Enhanced intent classification with hero-aware patterns."""
        query_lower = query.lower()
        
        # Check for hero-specific patterns first
        hero_class = context.get('hero_class') or self.current_hero_class
        if hero_class:
            hero_patterns = self.intent_patterns.get(f"hero_{hero_class.lower()}", {})
            for intent, patterns in hero_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        return f"hero_specific_{intent}"
        
        # Check general patterns
        for intent, patterns in self.intent_patterns.items():
            if isinstance(patterns, list):  # Skip hero-specific pattern dicts
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        return intent
        
        return "general_question"
    
    def _generate_hero_aware_response(self, intent: str, query: str, context: Dict[str, Any]) -> str:
        """Generate hero-aware response using enhanced templates."""
        try:
            hero_class = context.get('hero_class') or self.current_hero_class
            deck_state = context.get('deck_state')
            
            # Get hero profile for context
            hero_profile = self.hero_coaching_profiles.get(hero_class, {}) if hero_class else {}
            
            # Get appropriate template
            template_key = intent
            if hero_class and f"{intent}_hero" in self.response_templates:
                template_key = f"{intent}_hero"
            
            template = self.response_templates.get(template_key, 
                                                 self.response_templates.get("general_question", 
                                                                           "Let's think about this strategically."))
            
            # Fill template with context
            response = self._fill_template_with_context(template, query, hero_class, hero_profile, context)
            
            return response
            
        except Exception as e:
            self.logger.warning(f"Error generating hero-aware response: {e}")
            return f"That's an interesting question about {query}. Let's consider how this applies to your {self.current_hero_class or 'current strategy'}."
    
    def _fill_template_with_context(self, template: str, query: str, hero_class: Optional[str],
                                   hero_profile: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Fill response template with contextual information."""
        try:
            # Prepare template variables
            template_vars = {
                'query': query,
                'hero_class': hero_class or 'your hero',
                'hero_strength': hero_profile.get('primary_strength', 'versatility'),
                'hero_weakness': hero_profile.get('primary_weakness', 'consistency'),
                'archetype': hero_profile.get('best_archetype', 'Balanced'),
                'draft_stage': context.get('draft_stage', 'current stage'),
                'user_skill': self.user_skill_level
            }
            
            # Add deck-specific context if available
            deck_state = context.get('deck_state')
            if deck_state and hasattr(deck_state, 'drafted_cards'):
                cards_drafted = len(deck_state.drafted_cards)
                template_vars['cards_drafted'] = cards_drafted
                template_vars['stage_description'] = self._get_stage_description(cards_drafted)
            else:
                template_vars['cards_drafted'] = 0
                template_vars['stage_description'] = 'early draft'
            
            return template.format(**template_vars)
            
        except KeyError as e:
            self.logger.warning(f"Missing template variable {e}: {template}")
            # Return template with basic substitution
            return template.replace('{query}', query).replace('{hero_class}', hero_class or 'your hero')
        except Exception as e:
            self.logger.warning(f"Error filling template: {e}")
            return template
    
    def _generate_contextual_follow_up(self, intent: str, context: Dict[str, Any]) -> Optional[str]:
        """Generate contextual follow-up Socratic questions."""
        try:
            hero_class = context.get('hero_class') or self.current_hero_class
            if not hero_class:
                return None
            
            follow_up_patterns = self.socratic_patterns.get(intent, {})
            hero_follow_ups = follow_up_patterns.get(hero_class, [])
            
            if hero_follow_ups:
                import random
                return random.choice(hero_follow_ups)
            
            # General follow-ups
            general_follow_ups = follow_up_patterns.get('general', [])
            if general_follow_ups:
                import random
                return random.choice(general_follow_ups)
            
            return None
            
        except Exception:
            return None
    
    def _update_educational_objectives(self, intent: str, context: Dict[str, Any]) -> None:
        """Update educational objectives based on user interaction."""
        try:
            # Map intents to educational objectives
            intent_mapping = {
                'curve_question': 'curve_understanding',
                'archetype_question': 'archetype_recognition', 
                'synergy_question': 'synergy_awareness',
                'meta_question': 'meta_knowledge',
                'hero_specific': 'hero_strengths'
            }
            
            objective = intent_mapping.get(intent)
            if objective:
                self.educational_objectives[objective] = True
                
        except Exception:
            pass  # Non-critical functionality
    
    def _generate_fallback_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate fallback response when main system fails."""
        hero_class = context.get('hero_class') or self.current_hero_class
        if hero_class:
            return f"That's an interesting question about {query}. How do you think this relates to {hero_class}'s strengths in Arena?"
        else:
            return f"That's a thoughtful question about {query}. What factors do you think are most important to consider?"
    
    # === HERO-SPECIFIC SOCRATIC QUESTION GENERATORS ===
    
    def _generate_hero_specific_question(self, card_id: str, hero_class: str, 
                                       card_analysis: Dict[str, Any], 
                                       deck_state: Optional[DeckState]) -> Optional[str]:
        """Generate hero-specific Socratic questions."""
        try:
            hero_profile = self.hero_coaching_profiles.get(hero_class, {})
            
            # Get hero-specific questioning patterns
            if hero_class == 'MAGE':
                return self._generate_mage_question(card_id, card_analysis, deck_state)
            elif hero_class == 'PALADIN':
                return self._generate_paladin_question(card_id, card_analysis, deck_state) 
            elif hero_class == 'HUNTER':
                return self._generate_hunter_question(card_id, card_analysis, deck_state)
            elif hero_class == 'WARRIOR':
                return self._generate_warrior_question(card_id, card_analysis, deck_state)
            elif hero_class == 'PRIEST':
                return self._generate_priest_question(card_id, card_analysis, deck_state)
            elif hero_class == 'WARLOCK':
                return self._generate_warlock_question(card_id, card_analysis, deck_state)
            elif hero_class == 'ROGUE':
                return self._generate_rogue_question(card_id, card_analysis, deck_state)
            elif hero_class == 'SHAMAN':
                return self._generate_shaman_question(card_id, card_analysis, deck_state)
            elif hero_class == 'DRUID':
                return self._generate_druid_question(card_id, card_analysis, deck_state)
            elif hero_class == 'DEMONHUNTER':
                return self._generate_demonhunter_question(card_id, card_analysis, deck_state)
            
            return None
            
        except Exception:
            return None
    
    def _generate_mage_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Mage-specific questions focusing on spells and tempo."""
        questions = [
            f"How does {card_id} help Mage's spell-based strategy?",
            f"Would {card_id} give you the tempo Mage needs to control the board?",
            f"Does {card_id} synergize with Mage's hero power for late game value?",
            f"How does {card_id} fit into Mage's flexible playstyle?",
            f"Would {card_id} help against Mage's weakness to early aggression?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_paladin_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Paladin-specific questions focusing on minions and buffs."""
        questions = [
            f"How does {card_id} support Paladin's minion-based strategy?",
            f"Would {card_id} benefit from Paladin's buffing capabilities?",
            f"Does {card_id} help establish the board presence Paladin needs?",
            f"How does {card_id} work with Paladin's Divine Shield synergies?",
            f"Would {card_id} help Paladin maintain early game pressure?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_hunter_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Hunter-specific questions focusing on aggression and beasts."""
        questions = [
            f"Does {card_id} support Hunter's aggressive game plan?",
            f"How does {card_id} work with Hunter's beast synergies?",
            f"Would {card_id} help Hunter maintain early game pressure?",
            f"Does {card_id} provide the direct damage Hunter needs?",
            f"How does {card_id} help Hunter close out games quickly?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_warrior_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Warrior-specific questions focusing on weapons and control."""
        questions = [
            f"How does {card_id} support Warrior's weapon-based strategy?",
            f"Would {card_id} help Warrior survive to the late game?",
            f"Does {card_id} work with Warrior's armor generation?",
            f"How does {card_id} fit Warrior's control playstyle?",
            f"Would {card_id} give Warrior the board clears it needs?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_priest_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Priest-specific questions focusing on healing and value."""
        questions = [
            f"How does {card_id} support Priest's value-oriented strategy?",
            f"Would {card_id} benefit from Priest's healing capabilities?",
            f"Does {card_id} help Priest's late game win condition?",
            f"How does {card_id} work with Priest's high-health minions?",
            f"Would {card_id} help Priest survive early aggression?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_warlock_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Warlock-specific questions focusing on life management and flexibility."""
        questions = [
            f"Is {card_id} worth the life cost for Warlock's strategy?",
            f"How does {card_id} work with Warlock's card draw advantage?",
            f"Would {card_id} fit Warlock's flexible archetype approach?",
            f"Does {card_id} help manage Warlock's life total effectively?",
            f"How does {card_id} support Warlock's demon synergies?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_rogue_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Rogue-specific questions focusing on tempo and combo."""
        questions = [
            f"How does {card_id} support Rogue's tempo-based strategy?",
            f"Would {card_id} work with Rogue's weapon synergies?",
            f"Does {card_id} fit into Rogue's combo gameplay?",
            f"How does {card_id} help Rogue's efficient minion trading?",
            f"Would {card_id} give Rogue the board control it needs?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_shaman_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Shaman-specific questions focusing on elementals and overload."""
        questions = [
            f"How does {card_id} work with Shaman's elemental synergies?",
            f"Would {card_id} be worth the overload cost for Shaman?",
            f"Does {card_id} support Shaman's midrange strategy?",
            f"How does {card_id} work with Shaman's totem generation?",
            f"Would {card_id} help Shaman's burst potential?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_druid_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Druid-specific questions focusing on ramp and choose cards."""
        questions = [
            f"How does {card_id} benefit from Druid's mana acceleration?",
            f"Would {card_id} support Druid's ramp strategy?",
            f"Does {card_id} work with Druid's choose card flexibility?",
            f"How does {card_id} fit Druid's big minion strategy?",
            f"Would {card_id} help Druid's beast synergies?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_demonhunter_question(self, card_id: str, card_analysis: Dict, deck_state: Optional[DeckState]) -> str:
        """Generate Demon Hunter-specific questions focusing on aggression and outcast."""
        questions = [
            f"How does {card_id} support Demon Hunter's aggressive strategy?",
            f"Would {card_id} benefit from Demon Hunter's outcast mechanics?",
            f"Does {card_id} help Demon Hunter's early game pressure?",
            f"How does {card_id} work with Demon Hunter's weapons?",
            f"Would {card_id} help Demon Hunter close out games quickly?"
        ]
        import random
        return random.choice(questions)
    
    def _generate_recommended_card_question(self, card_id: str, card_analysis: Dict, hero_class: Optional[str]) -> str:
        """Generate question about the AI's recommended pick."""
        questions = [
            f"Why might {card_id} be the best choice here?",
            f"What makes {card_id} stand out from the other options?",
            f"How would {card_id} improve your deck's strategy?",
            f"What specific problem does {card_id} solve for your deck?"
        ]
        
        if hero_class:
            questions.extend([
                f"How does {card_id} leverage {hero_class}'s strengths?",
                f"What makes {card_id} particularly good for {hero_class}?"
            ])
        
        import random
        return random.choice(questions)
    
    def _generate_alternative_card_question(self, card_id: str, card_analysis: Dict, 
                                          recommended_index: int, ai_decision: AIDecision,
                                          hero_class: Optional[str]) -> str:
        """Generate question about non-recommended cards."""
        try:
            recommended_card = ai_decision.all_offered_cards_analysis[recommended_index].get("card_id", "the recommended card")
        except:
            recommended_card = "the recommended card"
        
        questions = [
            f"What would {card_id} offer that {recommended_card} doesn't?",
            f"In what situation would {card_id} be better than {recommended_card}?",
            f"What are the trade-offs of picking {card_id} over {recommended_card}?",
            f"How would choosing {card_id} change your deck's direction?"
        ]
        
        import random
        return random.choice(questions)
    
    def _get_stage_description(self, cards_drafted: int) -> str:
        """Get description of current draft stage."""
        if cards_drafted <= 10:
            return "early draft"
        elif cards_drafted <= 20:
            return "mid draft"
        else:
            return "late draft"
    
    # === ENHANCED PATTERN AND TEMPLATE BUILDERS ===
    
    def _build_enhanced_intent_patterns(self) -> Dict[str, Any]:
        """Build enhanced regex patterns for intent classification."""
        return {
            "card_comparison": [
                r"why.*pick.*over",
                r"better.*than",
                r"compare.*cards",
                r"which.*should.*choose",
                r"versus|vs"
            ],
            "archetype_question": [
                r"what.*archetype",
                r"should.*play.*aggro|control|tempo",
                r"deck.*strategy",
                r"playstyle",
                r"game.*plan",
                r"win.*condition"
            ],
            "curve_question": [
                r"mana.*curve",
                r"cost.*distribution", 
                r"early.*game",
                r"late.*game",
                r"curve.*out",
                r"mana.*cost"
            ],
            "synergy_question": [
                r"synergy|synergies",
                r"work.*together",
                r"combo",
                r"tribal",
                r"package",
                r"theme"
            ],
            "meta_question": [
                r"meta",
                r"winrate",
                r"statistics",
                r"data",
                r"tier.*list",
                r"best.*cards"
            ],
            "hero_question": [
                r"hero.*power",
                r"class.*specific",
                r"(mage|paladin|hunter|warrior|priest|warlock|rogue|shaman|druid|demonhunter).*strategy",
                r"hero.*strength",
                r"class.*synergy"
            ],
            "draft_advice": [
                r"what.*should.*draft",
                r"help.*me.*pick",
                r"advice",
                r"recommendation",
                r"guidance"
            ]
        }
    
    def _build_hero_aware_response_templates(self) -> Dict[str, str]:
        """Build hero-aware response templates for natural language generation."""
        return {
            # Enhanced general templates
            "card_comparison": "Consider how each card fits {hero_class}'s strategy. {hero_class} excels at {hero_strength} - which card better leverages this advantage? What does your {archetype} deck need more right now?",
            
            "card_comparison_hero": "When playing {hero_class}, think about {hero_strength} versus {hero_weakness}. Which card addresses your current needs in this {stage_description}?",
            
            "archetype_question": "Your archetype should complement {hero_class}'s natural strengths. Given {hero_class}'s focus on {hero_strength}, what archetype emerges from your picks so far?",
            
            "archetype_question_hero": "{hero_class} traditionally performs well with {archetype} strategies. How do your {cards_drafted} cards support this direction?",
            
            "curve_question": "Curve is crucial for {hero_class}'s {archetype} strategy. {hero_class} typically wants strong plays in the early-to-mid game. Where do you see gaps in your current curve?",
            
            "curve_question_hero": "For {hero_class}, curve planning should consider {hero_strength}. How does your current mana distribution support {hero_class}'s game plan?",
            
            "synergy_question": "{hero_class} has natural synergies with certain card types. How many pieces of this synergy do you have, and is it worth building around?",
            
            "synergy_question_hero": "Synergy can be powerful for {hero_class}, especially {hero_strength}-based packages. But avoid synergy traps - how consistent would this be?",
            
            "meta_question": "The current meta shows {hero_class} performing well with certain strategies. Based on the data, what approach do you think suits {hero_class} best right now?",
            
            "meta_question_hero": "Meta statistics suggest {hero_class} works well with {archetype} builds. How does this align with your current draft direction?",
            
            "hero_question": "That's an excellent question about {hero_class}'s capabilities. {hero_class}'s main strength is {hero_strength}, but watch out for {hero_weakness}. How does this affect your decision?",
            
            "draft_advice": "In this {stage_description}, {hero_class} should focus on {hero_strength}. What specific role does your deck need filled most urgently?",
            
            "general_question": "That's a thoughtful question about '{query}' for {hero_class}. Consider how this impacts {hero_class}'s {archetype} strategy. What factors matter most here?",
            
            # Skill-level specific templates
            "beginner_advice": "As a {user_skill} player with {hero_class}, focus on {hero_strength} and solid fundamentals. What seems like the safest, most powerful choice?",
            
            "advanced_advice": "Given your {user_skill} level with {hero_class}, consider the subtle interactions with {hero_strength}. How might this impact your late-game positioning?"
        }
    
    def _build_hero_coaching_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Build hero-specific coaching profiles for personalized guidance."""
        return {
            'MAGE': {
                'primary_strength': 'spell flexibility and tempo control',
                'primary_weakness': 'early game board presence',
                'best_archetype': 'Tempo',
                'key_concepts': ['spell synergy', 'tempo swings', 'card generation'],
                'coaching_focus': 'Value timing and spell efficiency'
            },
            'PALADIN': {
                'primary_strength': 'minion synergies and board buffs',
                'primary_weakness': 'card draw and reach',
                'best_archetype': 'Aggro',
                'key_concepts': ['divine shield', 'minion buffs', 'board presence'],
                'coaching_focus': 'Maintaining board control'
            },
            'HUNTER': {
                'primary_strength': 'early aggression and beast synergy',
                'primary_weakness': 'late game value',
                'best_archetype': 'Aggro',
                'key_concepts': ['face damage', 'beast tribal', 'tempo'],
                'coaching_focus': 'Closing games efficiently'
            },
            'WARRIOR': {
                'primary_strength': 'weapon synergy and late game',
                'primary_weakness': 'early game tempo',
                'best_archetype': 'Control',
                'key_concepts': ['weapon value', 'armor gain', 'removal'],
                'coaching_focus': 'Surviving to win conditions'
            },
            'PRIEST': {
                'primary_strength': 'healing and value generation',
                'primary_weakness': 'tempo and early pressure',
                'best_archetype': 'Control',
                'key_concepts': ['health synergy', 'value trades', 'late game'],
                'coaching_focus': 'Maximizing card advantage'
            },
            'WARLOCK': {
                'primary_strength': 'card draw and flexibility',
                'primary_weakness': 'life management',
                'best_archetype': 'Flexible',
                'key_concepts': ['life as resource', 'card advantage', 'demon synergy'],
                'coaching_focus': 'Resource management'
            },
            'ROGUE': {
                'primary_strength': 'efficient trading and tempo',
                'primary_weakness': 'health management',
                'best_archetype': 'Tempo',
                'key_concepts': ['weapon usage', 'combo mechanics', 'efficient removal'],
                'coaching_focus': 'Maximizing efficiency'
            },
            'SHAMAN': {
                'primary_strength': 'elemental synergy and burst',
                'primary_weakness': 'overload management',
                'best_archetype': 'Elemental',
                'key_concepts': ['elemental chains', 'overload planning', 'burst damage'],
                'coaching_focus': 'Managing resources and timing'
            },
            'DRUID': {
                'primary_strength': 'mana acceleration and big threats',
                'primary_weakness': 'early game consistency',
                'best_archetype': 'Ramp',
                'key_concepts': ['mana ramp', 'choose flexibility', 'big minions'],
                'coaching_focus': 'Scaling to late game'
            },
            'DEMONHUNTER': {
                'primary_strength': 'early aggression and weapons',
                'primary_weakness': 'card draw and late game',
                'best_archetype': 'Aggro',
                'key_concepts': ['outcast mechanics', 'weapon synergy', 'face damage'],
                'coaching_focus': 'Aggressive tempo plays'
            }
        }
    
    def _build_socratic_questioning_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Build Socratic questioning patterns for different contexts."""
        return {
            'card_comparison': {
                'general': [
                    "What would change if you picked the other card instead?",
                    "Which card fits better with your existing synergies?",
                    "How do these cards compare in your current meta?"
                ],
                'MAGE': [
                    "Which card gives Mage better spell synergy?",
                    "How do these cards help Mage's tempo game plan?"
                ],
                'PALADIN': [
                    "Which card benefits more from Paladin's buffing potential?",
                    "How do these cards support board presence?"
                ]
            },
            'archetype_question': {
                'general': [
                    "What archetype signals do you see in your draft so far?",
                    "How committed are you to your current direction?"
                ],
                'HUNTER': [
                    "Is your Hunter draft leaning aggressive or value-oriented?",
                    "How many beast synergies have you picked up?"
                ]
            },
            'meta_question': {
                'general': [
                    "How might the current meta influence this decision?",
                    "What does the data tell us about this strategy?"
                ]
            }
        }
    
    # === EDUCATIONAL PROGRESS TRACKING ===
    
    def get_educational_progress(self) -> Dict[str, Any]:
        """Get user's educational progress and learning insights."""
        return {
            'objectives_completed': sum(1 for completed in self.educational_objectives.values() if completed),
            'total_objectives': len(self.educational_objectives),
            'completion_percentage': (
                sum(1 for completed in self.educational_objectives.values() if completed) / 
                len(self.educational_objectives) * 100
            ),
            'detailed_progress': self.educational_objectives,
            'conversation_count': len(self.conversation_history),
            'current_hero_focus': self.current_hero_class,
            'skill_level': self.user_skill_level,
            'areas_for_improvement': [
                objective for objective, completed in self.educational_objectives.items() 
                if not completed
            ]
        }
    
    def generate_learning_summary(self) -> str:
        """Generate a summary of the user's learning progress."""
        progress = self.get_educational_progress()
        
        if progress['completion_percentage'] >= 80:
            return f"Excellent progress! You've mastered {progress['objectives_completed']}/{progress['total_objectives']} key concepts. You're showing strong understanding of draft strategy."
        elif progress['completion_percentage'] >= 60:
            return f"Good progress on draft concepts ({progress['objectives_completed']}/{progress['total_objectives']} completed). Keep exploring different strategic aspects."
        elif progress['completion_percentage'] >= 40:
            return f"You're building solid foundations ({progress['objectives_completed']}/{progress['total_objectives']} concepts). Continue asking questions to deepen your understanding."
        else:
            return f"Great that you're asking questions! Focus on understanding basic concepts like curve, synergy, and archetype identification."
    
    def suggest_learning_focus(self) -> Optional[str]:
        """Suggest what the user should focus on learning next."""
        incomplete_objectives = [
            obj for obj, completed in self.educational_objectives.items() if not completed
        ]
        
        if not incomplete_objectives:
            return "You've covered the fundamentals! Focus on advanced concepts like meta positioning and matchup considerations."
        
        # Prioritize learning objectives
        priority_order = ['curve_understanding', 'archetype_recognition', 'hero_strengths', 'synergy_awareness', 'meta_knowledge']
        
        for priority_obj in priority_order:
            if priority_obj in incomplete_objectives:
                suggestions = {
                    'curve_understanding': "Try asking about mana curve and early/late game balance",
                    'archetype_recognition': "Explore different deck archetypes and how they work",
                    'hero_strengths': "Learn about your hero's specific strengths and weaknesses",
                    'synergy_awareness': "Investigate how cards work together in combos",
                    'meta_knowledge': "Ask about current meta trends and statistics"
                }
                return suggestions.get(priority_obj)
        
        return None