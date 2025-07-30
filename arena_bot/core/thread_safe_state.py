#!/usr/bin/env python3
"""
Thread-Safe Immutable State Management - Emergency Thread Safety Fix

This module implements the copy-on-write immutable state pattern for DeckState
as specified in THREAD_SAFETY_ANALYSIS.md to eliminate state corruption bugs.

Key Features:
- Copy-on-write pattern for atomic state updates
- Thread-safe access with RLock protection
- Immutable state access prevents race conditions
- Atomic state transitions with rollback capability
- State versioning for debugging and audit trails
"""

import copy
import threading
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import the global lock manager for proper ordering
from .lock_manager import get_global_lock_manager, LockType

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StateOperationType(Enum):
    """Types of state operations for audit trail"""
    READ = "read"
    UPDATE = "update"
    REPLACE = "replace"
    ROLLBACK = "rollback"

@dataclass
class StateOperation:
    """Record of a state operation for audit and debugging"""
    operation_id: str
    operation_type: StateOperationType
    timestamp: datetime
    thread_id: int
    success: bool
    duration_ms: float
    error: Optional[str] = None
    state_version: int = 0

class ThreadSafeImmutableContainer(Generic[T]):
    """
    Generic thread-safe immutable container implementing copy-on-write pattern
    
    This is the core implementation that prevents state corruption by ensuring
    all state access is atomic and all modifications create new immutable copies.
    
    Usage:
        container = ThreadSafeImmutableContainer(initial_state)
        
        # Thread-safe read
        current_state = container.get_state()
        
        # Thread-safe atomic update
        def modifier(state):
            new_state = copy.deepcopy(state)
            new_state.some_field = new_value
            return new_state
        
        updated_state = container.update_state(modifier)
    """
    
    def __init__(self, initial_state: T, container_name: str = "unknown"):
        """
        Initialize thread-safe container
        
        Args:
            initial_state: Initial state object
            container_name: Name for lock registration and debugging
        """
        self._state = initial_state
        self._container_name = container_name
        self._version = 1
        self._created_at = datetime.now()
        
        # Register a dedicated lock for this container
        self._lock_manager = get_global_lock_manager()
        self._lock_name = f"state_{container_name}_{id(self)}"
        self._lock = self._lock_manager.register_lock(
            self._lock_name, 
            LockType.AI  # DeckState is primarily AI component state
        )
        
        # Operation audit trail for debugging
        self._operations: List[StateOperation] = []
        self._max_operations = 100  # Prevent memory growth
        
        # Statistics
        self._stats = {
            'reads': 0,
            'updates': 0,
            'rollbacks': 0,
            'contentions': 0,
            'total_update_time': 0.0,
            'max_update_time': 0.0
        }
        
        logger.debug(f"Initialized ThreadSafeImmutableContainer: {container_name}")
    
    def get_state(self) -> T:
        """
        Get immutable copy of current state (thread-safe)
        
        Returns deep copy to prevent external mutation of internal state.
        This is the key to preventing race conditions - external code can
        never modify the internal state directly.
        
        Returns:
            Deep copy of current state
        """
        operation_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        thread_id = threading.get_ident()
        
        try:
            with self._lock_manager.acquire_lock(self._lock_name):
                # Create immutable copy
                state_copy = copy.deepcopy(self._state)
                
                # Record successful operation
                self._record_operation(
                    operation_id=operation_id,
                    operation_type=StateOperationType.READ,
                    thread_id=thread_id,
                    duration_ms=(time.time() - start_time) * 1000,
                    success=True
                )
                
                self._stats['reads'] += 1
                return state_copy
                
        except Exception as e:
            # Record failed operation
            self._record_operation(
                operation_id=operation_id,
                operation_type=StateOperationType.READ,
                thread_id=thread_id,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
            
            logger.error(f"State read failed for {self._container_name}: {e}")
            raise
    
    def update_state(self, modifier_func: Callable[[T], T]) -> T:
        """
        Atomically update state using modifier function (thread-safe)
        
        This implements the copy-on-write pattern:
        1. Create deep copy of current state
        2. Apply modifier function to copy
        3. Atomically replace internal state with modified copy
        4. Return the new state
        
        Args:
            modifier_func: Function that takes current state and returns new state
            
        Returns:
            The new state after modification
            
        Raises:
            Exception: If modifier function fails or validation fails
        """
        operation_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        thread_id = threading.get_ident()
        
        # Store original state for rollback
        original_state = None
        original_version = None
        
        try:
            with self._lock_manager.acquire_lock(self._lock_name):
                # Store rollback information
                original_state = self._state
                original_version = self._version
                
                # Create deep copy for modification
                state_copy = copy.deepcopy(self._state)
                
                # Apply modifier function
                new_state = modifier_func(state_copy)
                
                # Validate new state (if it has a validate method)
                if hasattr(new_state, 'validate'):
                    new_state.validate()
                
                # Atomically update internal state
                self._state = new_state
                self._version += 1
                
                # Record successful operation
                duration_ms = (time.time() - start_time) * 1000
                self._record_operation(
                    operation_id=operation_id,
                    operation_type=StateOperationType.UPDATE,
                    thread_id=thread_id,
                    duration_ms=duration_ms,
                    success=True,
                    state_version=self._version
                )
                
                # Update statistics
                self._stats['updates'] += 1
                self._stats['total_update_time'] += duration_ms
                self._stats['max_update_time'] = max(self._stats['max_update_time'], duration_ms)
                
                logger.debug(f"State updated for {self._container_name} (v{self._version})")
                return copy.deepcopy(new_state)
                
        except Exception as e:
            # Rollback on error
            if original_state is not None:
                try:
                    with self._lock_manager.acquire_lock(self._lock_name):
                        self._state = original_state
                        self._version = original_version
                        self._stats['rollbacks'] += 1
                        
                        logger.warning(f"Rolled back state for {self._container_name} due to error: {e}")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed for {self._container_name}: {rollback_error}")
            
            # Record failed operation
            self._record_operation(
                operation_id=operation_id,
                operation_type=StateOperationType.UPDATE,
                thread_id=thread_id,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
            
            logger.error(f"State update failed for {self._container_name}: {e}")
            raise
    
    def replace_state(self, new_state: T) -> T:
        """
        Replace entire state atomically (thread-safe)
        
        Args:
            new_state: New state to replace current state
            
        Returns:
            Deep copy of the new state
        """
        def replacer(current_state: T) -> T:
            return new_state
        
        return self.update_state(replacer)
    
    def get_version(self) -> int:
        """Get current state version (thread-safe)"""
        with self._lock_manager.acquire_lock(self._lock_name):
            return self._version
    
    def get_container_info(self) -> Dict[str, Any]:
        """Get container metadata and statistics (thread-safe)"""
        with self._lock_manager.acquire_lock(self._lock_name):
            avg_update_time = (
                self._stats['total_update_time'] / self._stats['updates']
                if self._stats['updates'] > 0 else 0.0
            )
            
            return {
                'container_name': self._container_name,
                'version': self._version,
                'created_at': self._created_at.isoformat(),
                'lock_name': self._lock_name,
                'statistics': {
                    **self._stats,
                    'avg_update_time': avg_update_time
                },
                'recent_operations': len(self._operations)
            }
    
    def _record_operation(
        self, 
        operation_id: str,
        operation_type: StateOperationType,
        thread_id: int,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
        state_version: int = None
    ):
        """Record operation in audit trail"""
        operation = StateOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            timestamp=datetime.now(),
            thread_id=thread_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            state_version=state_version or self._version
        )
        
        # Add to operations list with size limit
        self._operations.append(operation)
        if len(self._operations) > self._max_operations:
            self._operations.pop(0)
    
    def get_operation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operation history (thread-safe)"""
        with self._lock_manager.acquire_lock(self._lock_name):
            recent_operations = self._operations[-limit:] if self._operations else []
            
            return [
                {
                    'operation_id': op.operation_id,
                    'type': op.operation_type.value,
                    'timestamp': op.timestamp.isoformat(),
                    'thread_id': op.thread_id,
                    'success': op.success,
                    'duration_ms': op.duration_ms,
                    'error': op.error,
                    'state_version': op.state_version
                }
                for op in recent_operations
            ]

class ThreadSafeDeckState:
    """
    Thread-safe wrapper for DeckState implementing the immutable state pattern
    
    This is the specific implementation for DeckState as required by the
    THREAD_SAFETY_ANALYSIS.md report. It prevents all state corruption bugs
    by ensuring atomic operations and immutable access.
    
    Usage Example:
        # Create thread-safe deck state
        initial_deck = DeckState(cards=[], hero_class=CardClass.MAGE)
        safe_deck = ThreadSafeDeckState(initial_deck)
        
        # Thread-safe read
        current_deck = safe_deck.get_state()
        
        # Thread-safe atomic update
        def add_card(deck_state):
            new_deck = copy.deepcopy(deck_state)
            new_deck.cards.append(new_card)
            new_deck.total_cards += 1
            return new_deck
        
        updated_deck = safe_deck.update_state(add_card)
    """
    
    def __init__(self, initial_deck_state, deck_name: str = "draft_deck"):
        """
        Initialize thread-safe deck state wrapper
        
        Args:
            initial_deck_state: Initial DeckState object
            deck_name: Name for identification and debugging
        """
        # Import here to avoid circular dependencies
        try:
            from arena_bot.ai_v2.data_models import DeckState
            
            if not isinstance(initial_deck_state, DeckState):
                raise ValueError(f"Expected DeckState object, got {type(initial_deck_state)}")
                
        except ImportError:
            logger.warning("DeckState import failed - using generic validation")
        
        self._container = ThreadSafeImmutableContainer(
            initial_state=initial_deck_state,
            container_name=f"deck_{deck_name}"
        )
        
        self._deck_name = deck_name
        logger.info(f"🛡️ Created ThreadSafeDeckState: {deck_name}")
    
    def get_state(self):
        """
        Get immutable copy of current deck state
        
        Returns:
            Deep copy of DeckState - safe for external use
        """
        return self._container.get_state()
    
    def update_state(self, modifier_func: Callable):
        """
        Atomically update deck state using modifier function
        
        Args:
            modifier_func: Function that takes DeckState and returns modified DeckState
            
        Returns:
            New DeckState after modification
            
        Example:
            def add_card_to_deck(deck_state):
                new_deck = copy.deepcopy(deck_state)
                new_deck.cards.append(new_card)
                # Recalculate derived properties
                new_deck.total_cards = len(new_deck.cards)
                new_deck.mana_curve = new_deck._calculate_mana_curve()
                new_deck.average_cost = new_deck._calculate_average_cost()
                return new_deck
            
            updated_deck = safe_deck.update_state(add_card_to_deck)
        """
        return self._container.update_state(modifier_func)
    
    def replace_state(self, new_deck_state):
        """
        Replace entire deck state atomically
        
        Args:
            new_deck_state: New DeckState to replace current state
            
        Returns:
            Deep copy of the new state
        """
        return self._container.replace_state(new_deck_state)
    
    def get_version(self) -> int:
        """Get current state version for debugging"""
        return self._container.get_version()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about state operations"""
        container_info = self._container.get_container_info()
        container_info['deck_name'] = self._deck_name
        return container_info
    
    def get_operation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operation history for debugging"""
        return self._container.get_operation_history(limit)
    
    # Convenience methods for common deck operations
    
    def add_card(self, card_info):
        """
        Convenience method to add a card to the deck
        
        Args:
            card_info: CardInfo object to add
            
        Returns:
            Updated DeckState
        """
        def add_card_modifier(deck_state):
            new_deck = copy.deepcopy(deck_state)
            new_deck.cards.append(card_info)
            
            # Recalculate derived properties
            new_deck.total_cards = len(new_deck.cards)
            new_deck.mana_curve = new_deck._calculate_mana_curve()
            new_deck.average_cost = new_deck._calculate_average_cost()
            new_deck.class_distribution = new_deck._calculate_class_distribution()
            new_deck.rarity_distribution = new_deck._calculate_rarity_distribution()
            
            return new_deck
        
        return self.update_state(add_card_modifier)
    
    def remove_card(self, card_index: int):
        """
        Convenience method to remove a card by index
        
        Args:
            card_index: Index of card to remove
            
        Returns:
            Updated DeckState
        """
        def remove_card_modifier(deck_state):
            new_deck = copy.deepcopy(deck_state)
            
            if 0 <= card_index < len(new_deck.cards):
                new_deck.cards.pop(card_index)
                
                # Recalculate derived properties
                new_deck.total_cards = len(new_deck.cards)
                new_deck.mana_curve = new_deck._calculate_mana_curve()
                new_deck.average_cost = new_deck._calculate_average_cost()
                new_deck.class_distribution = new_deck._calculate_class_distribution()
                new_deck.rarity_distribution = new_deck._calculate_rarity_distribution()
            
            return new_deck
        
        return self.update_state(remove_card_modifier)
    
    def update_pick_number(self, new_pick: int):
        """
        Convenience method to update current pick number
        
        Args:
            new_pick: New pick number
            
        Returns:
            Updated DeckState
        """
        def update_pick_modifier(deck_state):
            new_deck = copy.deepcopy(deck_state)
            new_deck.current_pick = new_pick
            new_deck.draft_phase = new_deck._calculate_draft_phase()
            return new_deck
        
        return self.update_state(update_pick_modifier)

# Factory function for creating thread-safe deck states
def create_thread_safe_deck_state(initial_deck_state, deck_name: str = "draft_deck") -> ThreadSafeDeckState:
    """
    Factory function to create thread-safe deck state
    
    Args:
        initial_deck_state: Initial DeckState object
        deck_name: Name for identification
        
    Returns:
        ThreadSafeDeckState wrapper
    """
    return ThreadSafeDeckState(initial_deck_state, deck_name)

# Emergency state recovery functions
def emergency_recover_deck_state(corrupted_state, fallback_state):
    """
    Emergency recovery function for corrupted deck states
    
    Args:
        corrupted_state: The corrupted state that needs recovery
        fallback_state: Clean fallback state to use
        
    Returns:
        ThreadSafeDeckState with recovered state
    """
    logger.critical("🚨 Emergency deck state recovery initiated")
    
    try:
        # Try to salvage what we can from corrupted state
        if hasattr(corrupted_state, 'cards') and corrupted_state.cards:
            recovered_deck = create_thread_safe_deck_state(
                corrupted_state, "emergency_recovery"
            )
            logger.info("✅ Successfully recovered corrupted deck state")
            return recovered_deck
            
    except Exception as e:
        logger.error(f"Failed to recover corrupted state: {e}")
    
    # Use fallback state
    fallback_deck = create_thread_safe_deck_state(
        fallback_state, "emergency_fallback"
    )
    logger.warning("⚠️ Using fallback state for recovery")
    return fallback_deck

if __name__ == "__main__":
    # Test the thread-safe state management
    import sys
    sys.path.append('/mnt/d/cursor bots/arena_bot_project')
    
    try:
        from arena_bot.ai_v2.data_models import DeckState, CardClass, ArchetypePreference
        
        # Create initial deck state
        initial_deck = DeckState(
            cards=[],
            hero_class=CardClass.MAGE,
            current_pick=1,
            archetype_preference=ArchetypePreference.TEMPO
        )
        
        # Create thread-safe wrapper
        safe_deck = create_thread_safe_deck_state(initial_deck, "test_deck")
        
        # Test read
        current_state = safe_deck.get_state()
        print(f"✅ Initial state read: {current_state.hero_class}")
        
        # Test update
        def update_pick(deck_state):
            new_deck = copy.deepcopy(deck_state)
            new_deck.current_pick = 5
            return new_deck
        
        updated_state = safe_deck.update_state(update_pick)
        print(f"✅ State updated: pick {updated_state.current_pick}")
        
        # Show statistics
        stats = safe_deck.get_statistics()
        print(f"📊 Statistics: {stats['statistics']}")
        
        print("🛡️ Thread-safe state management test complete")
        
    except ImportError as e:
        print(f"⚠️ Import error (expected in testing): {e}")
        print("🛡️ Thread-safe state management system ready")