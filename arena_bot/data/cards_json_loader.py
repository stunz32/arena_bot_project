"""
Hearthstone Cards JSON Loader - Arena Tracker Style
Loads and manages the official Hearthstone cards.json database
Enhanced with fuzzy matching for HearthArena card name mapping
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass

try:
    from rapidfuzz import fuzz, process
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_MATCHING_AVAILABLE = True
    except ImportError:
        FUZZY_MATCHING_AVAILABLE = False


@dataclass
class CardMatch:
    """Container for fuzzy card matching results."""
    card_id: str
    card_name: str
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'normalized'

class CardsJsonLoader:
    """
    Loads and manages Hearthstone cards.json like Arena Tracker.
    Provides card ID to name translation and other card attributes.
    """
    
    def __init__(self, cards_json_path: Optional[Path] = None):
        """Initialize the cards JSON loader."""
        self.logger = logging.getLogger(__name__)
        
        if cards_json_path is None:
            cards_json_path = Path(__file__).parent.parent.parent / "assets" / "cards.json"
        
        self.cards_json_path = Path(cards_json_path)
        self.cards_data: Dict[str, Dict[str, Any]] = {}
        self.cards_by_name: Dict[str, str] = {}  # name -> card_id mapping
        self.normalized_names: Dict[str, str] = {}  # normalized_name -> card_id mapping
        
        # Fuzzy matching configuration
        self.fuzzy_threshold = 80  # Minimum similarity score for fuzzy matches
        self.exact_threshold = 95   # Threshold for "near exact" matches
        
        if not FUZZY_MATCHING_AVAILABLE:
            self.logger.warning("Fuzzy matching libraries not available. Install with: pip install rapidfuzz")
        
        self.load_cards_json()
    
    def load_cards_json(self):
        """Load the cards.json file into memory like Arena Tracker."""
        try:
            if not self.cards_json_path.exists():
                self.logger.error(f"Cards JSON not found: {self.cards_json_path}")
                return
            
            with open(self.cards_json_path, 'r', encoding='utf-8') as f:
                cards_list = json.load(f)
            
            # Convert list to dict indexed by card ID (like Arena Tracker)
            for card in cards_list:
                card_id = card.get('id', '')
                if card_id:
                    self.cards_data[card_id] = card
                    
                    # Also index by name for reverse lookups
                    name = card.get('name', '')
                    if name:
                        self.cards_by_name[name.lower()] = card_id
                        
                        # Create normalized name index for fuzzy matching
                        normalized_name = self._normalize_card_name(name)
                        self.normalized_names[normalized_name] = card_id
            
            self.logger.info(f"Loaded {len(self.cards_data)} cards from JSON database")
            
        except Exception as e:
            self.logger.error(f"Failed to load cards JSON: {e}")
    
    def get_card_name(self, card_id: str) -> str:
        """
        Get card name from card ID - like Arena Tracker's cardEnNameFromCode().
        
        Args:
            card_id: Card ID (e.g., "EX1_339")
            
        Returns:
            Card name or "Unknown Card" if not found
        """
        card_data = self.cards_data.get(card_id)
        if card_data:
            return card_data.get('name', f'Unknown ({card_id})')
        return f'Unknown ({card_id})'
    
    def get_card_attribute(self, card_id: str, attribute: str) -> Any:
        """
        Get any card attribute - like Arena Tracker's getCardAttribute().
        
        Args:
            card_id: Card ID
            attribute: Attribute name (e.g., 'cost', 'attack', 'health', 'rarity')
            
        Returns:
            Attribute value or None if not found
        """
        card_data = self.cards_data.get(card_id)
        if card_data:
            return card_data.get(attribute)
        return None
    
    def get_card_id_from_name(self, card_name: str) -> Optional[str]:
        """
        Get card ID from name - like Arena Tracker's cardLocalCodeFromName().
        
        Args:
            card_name: Card name
            
        Returns:
            Card ID or None if not found
        """
        return self.cards_by_name.get(card_name.lower())
    
    def is_collectible(self, card_id: str) -> bool:
        """Check if card is collectible (draftable)."""
        return self.get_card_attribute(card_id, 'collectible') == True
    
    def get_card_set(self, card_id: str) -> Optional[str]:
        """Get card set name."""
        return self.get_card_attribute(card_id, 'set')
    
    def get_card_cost(self, card_id: str) -> Optional[int]:
        """Get card mana cost."""
        return self.get_card_attribute(card_id, 'cost')
    
    def get_card_rarity(self, card_id: str) -> Optional[str]:
        """Get card rarity."""
        return self.get_card_attribute(card_id, 'rarity')
    
    def get_card_class(self, card_id: str) -> Optional[str]:
        """Get card class."""
        return self.get_card_attribute(card_id, 'cardClass')
    
    def get_card_type(self, card_id: str) -> Optional[str]:
        """Get card type."""
        return self.get_card_attribute(card_id, 'type')
    
    def _normalize_card_name(self, name: str) -> str:
        """
        Normalize card name for fuzzy matching.
        
        Removes punctuation, extra spaces, and standardizes format.
        
        Args:
            name: Original card name
            
        Returns:
            Normalized card name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove common punctuation and special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def get_card_id_fuzzy(self, card_name: str, threshold: Optional[float] = None) -> Optional[CardMatch]:
        """
        Get card ID from name using fuzzy matching.
        
        Tries multiple matching strategies:
        1. Exact match (case insensitive)
        2. Normalized match (no punctuation)
        3. Fuzzy matching with similarity scoring
        
        Args:
            card_name: Card name to match
            threshold: Minimum similarity threshold (uses default if None)
            
        Returns:
            CardMatch with best match or None if no good match found
        """
        if not card_name or not card_name.strip():
            return None
        
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        # Strategy 1: Exact match (case insensitive)
        exact_match = self.get_card_id_from_name(card_name)
        if exact_match:
            return CardMatch(
                card_id=exact_match,
                card_name=self.get_card_name(exact_match),
                confidence=100.0,
                match_type='exact'
            )
        
        # Strategy 2: Normalized match
        normalized_query = self._normalize_card_name(card_name)
        normalized_match = self.normalized_names.get(normalized_query)
        if normalized_match:
            return CardMatch(
                card_id=normalized_match,
                card_name=self.get_card_name(normalized_match),
                confidence=95.0,
                match_type='normalized'
            )
        
        # Strategy 3: Fuzzy matching
        if FUZZY_MATCHING_AVAILABLE:
            return self._fuzzy_match_card_name(card_name, threshold)
        else:
            self.logger.warning("Fuzzy matching not available - install rapidfuzz or fuzzywuzzy")
            return None
    
    def _fuzzy_match_card_name(self, card_name: str, threshold: float) -> Optional[CardMatch]:
        """
        Perform fuzzy matching against all card names.
        
        Args:
            card_name: Query card name
            threshold: Minimum similarity threshold
            
        Returns:
            Best CardMatch or None
        """
        if not FUZZY_MATCHING_AVAILABLE:
            return None
        
        try:
            # Get all card names for fuzzy matching
            all_card_names = list(self.cards_by_name.keys())
            
            if not all_card_names:
                return None
            
            # Find best match using fuzzy matching
            result = process.extractOne(
                card_name.lower(),
                all_card_names,
                scorer=fuzz.ratio
            )
            
            if result and result[1] >= threshold:
                matched_name, confidence = result[0], result[1]
                card_id = self.cards_by_name[matched_name]
                
                return CardMatch(
                    card_id=card_id,
                    card_name=self.get_card_name(card_id),
                    confidence=confidence,
                    match_type='fuzzy'
                )
            
        except Exception as e:
            self.logger.error(f"Fuzzy matching failed: {e}")
        
        return None
    
    def get_multiple_card_matches(self, card_name: str, max_results: int = 5, 
                                 threshold: Optional[float] = None) -> List[CardMatch]:
        """
        Get multiple potential card matches sorted by confidence.
        
        Args:
            card_name: Card name to match
            max_results: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of CardMatch objects sorted by confidence (highest first)
        """
        if not card_name or not card_name.strip():
            return []
        
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        matches = []
        
        # First try exact match
        exact_match = self.get_card_id_from_name(card_name)
        if exact_match:
            matches.append(CardMatch(
                card_id=exact_match,
                card_name=self.get_card_name(exact_match),
                confidence=100.0,
                match_type='exact'
            ))
        
        # Then try fuzzy matching if available
        if FUZZY_MATCHING_AVAILABLE:
            try:
                all_card_names = list(self.cards_by_name.keys())
                
                # Get multiple fuzzy matches
                fuzzy_results = process.extract(
                    card_name.lower(),
                    all_card_names,
                    scorer=fuzz.ratio,
                    limit=max_results * 2  # Get extra to filter
                )
                
                for matched_name, confidence in fuzzy_results:
                    if confidence >= threshold:
                        card_id = self.cards_by_name[matched_name]
                        
                        # Skip if we already have this card (from exact match)
                        if any(match.card_id == card_id for match in matches):
                            continue
                        
                        matches.append(CardMatch(
                            card_id=card_id,
                            card_name=self.get_card_name(card_id),
                            confidence=confidence,
                            match_type='fuzzy'
                        ))
                
            except Exception as e:
                self.logger.error(f"Multiple fuzzy matching failed: {e}")
        
        # Sort by confidence (highest first) and limit results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]
    
    def validate_card_mapping(self, heartharena_name: str, card_id: str) -> bool:
        """
        Validate that a HearthArena card name maps correctly to a card ID.
        
        Args:
            heartharena_name: Card name from HearthArena
            card_id: Proposed card ID mapping
            
        Returns:
            True if mapping seems valid
        """
        if not heartharena_name or not card_id:
            return False
        
        # Check if card ID exists
        if card_id not in self.cards_data:
            return False
        
        # Get the official card name
        official_name = self.get_card_name(card_id)
        
        # Calculate similarity between names
        if FUZZY_MATCHING_AVAILABLE:
            similarity = fuzz.ratio(heartharena_name.lower(), official_name.lower())
            return similarity >= 70  # 70% similarity required for validation
        else:
            # Fallback to simple string comparison
            return heartharena_name.lower() in official_name.lower() or official_name.lower() in heartharena_name.lower()
    
    def get_mapping_statistics(self, heartharena_names: List[str]) -> Dict[str, Any]:
        """
        Get statistics about card name mapping success.
        
        Args:
            heartharena_names: List of card names from HearthArena
            
        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            'total_names': len(heartharena_names),
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'normalized_matches': 0,
            'no_matches': 0,
            'success_rate': 0.0,
            'failed_names': []
        }
        
        for name in heartharena_names:
            match = self.get_card_id_fuzzy(name)
            if match:
                if match.match_type == 'exact':
                    stats['exact_matches'] += 1
                elif match.match_type == 'fuzzy':
                    stats['fuzzy_matches'] += 1
                elif match.match_type == 'normalized':
                    stats['normalized_matches'] += 1
            else:
                stats['no_matches'] += 1
                stats['failed_names'].append(name)
        
        successful_matches = stats['exact_matches'] + stats['fuzzy_matches'] + stats['normalized_matches']
        if stats['total_names'] > 0:
            stats['success_rate'] = successful_matches / stats['total_names'] * 100
        
        return stats
    
    def batch_map_heartharena_names(self, heartharena_names: List[str]) -> Dict[str, Optional[str]]:
        """
        Map multiple HearthArena card names to card IDs in batch.
        
        Args:
            heartharena_names: List of card names from HearthArena
            
        Returns:
            Dictionary mapping HearthArena names to card IDs (None for failed mappings)
        """
        mappings = {}
        
        for name in heartharena_names:
            match = self.get_card_id_fuzzy(name)
            mappings[name] = match.card_id if match else None
        
        return mappings
    
    def get_all_collectible_cards(self) -> List[Dict[str, Any]]:
        """Returns a list of all collectible card data dictionaries."""
        return [card for card in self.cards_data.values() if card.get('collectible')]

# Global instance like Arena Tracker
_cards_json_loader = None

def get_cards_json_loader() -> CardsJsonLoader:
    """Get the global cards JSON loader instance."""
    global _cards_json_loader
    if _cards_json_loader is None:
        _cards_json_loader = CardsJsonLoader()
    return _cards_json_loader

def get_card_name(card_id: str) -> str:
    """Convenience function to get card name from ID."""
    return get_cards_json_loader().get_card_name(card_id)