"""
Arena Card Database

Manages current arena-eligible cards from HearthArena with automatic updates,
caching, and efficient access methods. Provides authoritative source for
which cards are currently draftable in Hearthstone Arena.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import time
import hashlib

try:
    from .arena_version_manager import get_arena_version_manager
    from .cards_json_loader import get_cards_json_loader, CardMatch
    from .heartharena_tier_manager import get_heartharena_tier_manager, TierData
    from .tier_cache_manager import get_tier_cache_manager
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent))
    from arena_version_manager import get_arena_version_manager
    from cards_json_loader import get_cards_json_loader, CardMatch
    from heartharena_tier_manager import get_heartharena_tier_manager, TierData
    from tier_cache_manager import get_tier_cache_manager


@dataclass
class ArenaCardData:
    """Container for arena card database with tier information."""
    last_updated: str
    source: str
    version: str
    classes: Dict[str, List[str]]  # class_name -> [card_ids]
    metadata: Dict[str, Any]
    raw_heartharena_data: Dict[str, List[str]]  # Raw card names from HearthArena
    mapping_stats: Dict[str, Any]
    tier_data: Optional[Dict[str, Dict[str, TierData]]] = None  # class_name -> {card_id: tier_data}
    
    def get_total_cards(self) -> int:
        """Get total number of unique arena cards."""
        all_cards = set()
        for cards in self.classes.values():
            all_cards.update(cards)
        return len(all_cards)
    
    def get_cards_for_class(self, class_name: str) -> List[str]:
        """Get arena cards for specific class including neutrals."""
        class_cards = self.classes.get(class_name.lower(), [])
        neutral_cards = self.classes.get('neutral', [])
        return class_cards + neutral_cards
    
    def is_arena_eligible(self, card_id: str, class_name: str = None) -> bool:
        """Check if card is arena-eligible for given class."""
        if class_name:
            return card_id in self.get_cards_for_class(class_name)
        else:
            # Check if card exists in any class
            all_cards = set()
            for cards in self.classes.values():
                all_cards.update(cards)
            return card_id in all_cards
    
    def get_card_tier(self, card_id: str, class_name: str) -> Optional[TierData]:
        """Get tier information for a specific card in a class."""
        if not self.tier_data or class_name not in self.tier_data:
            return None
        return self.tier_data[class_name].get(card_id)
    
    def get_cards_with_tiers(self, class_name: str) -> Dict[str, Optional[TierData]]:
        """Get all arena cards for a class with their tier information."""
        class_cards = self.get_cards_for_class(class_name)
        result = {}
        
        for card_id in class_cards:
            tier_info = self.get_card_tier(card_id, class_name) if self.tier_data else None
            result[card_id] = tier_info
        
        return result
    
    def get_tier_statistics(self) -> Dict[str, Any]:
        """Get statistics about tier data."""
        if not self.tier_data:
            return {'has_tier_data': False}
        
        stats = {
            'has_tier_data': True,
            'classes_with_tiers': len(self.tier_data),
            'total_cards_with_tiers': 0,
            'tier_distribution': {}
        }
        
        tier_counts = {}
        for class_tiers in self.tier_data.values():
            stats['total_cards_with_tiers'] += len(class_tiers)
            for tier_data in class_tiers.values():
                tier = tier_data.tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        stats['tier_distribution'] = tier_counts
        return stats


class ArenaCardDatabase:
    """
    Manages arena card database with HearthArena integration.
    
    Provides caching, automatic updates, and efficient access to current
    arena-eligible cards. Uses HearthArena as authoritative source.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize arena card database.
        
        Args:
            cache_dir: Directory for cache files (uses default if None)
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "arena"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.arena_data_file = self.cache_dir / "arena_card_data.json"
        self.heartharena_raw_file = self.cache_dir / "heartharena_raw.json"
        self.mapping_cache_file = self.cache_dir / "name_mappings.json"
        self.update_log_file = self.cache_dir / "update_history.json"
        
        # Configuration
        self.max_cache_age_days = 7  # Auto-update after 7 days
        self.min_success_rate = 80   # Minimum mapping success rate
        self.database_version = "1.0"
        
        # Runtime data
        self.arena_data: Optional[ArenaCardData] = None
        self.cards_loader = get_cards_json_loader()
        self.version_manager = get_arena_version_manager()
        self.tier_manager = get_heartharena_tier_manager()
        self.tier_cache = get_tier_cache_manager()
        
        # Load existing data
        self.load_cached_data()
        
        self.logger.info("ArenaCardDatabase initialized with tier support")
    
    def load_cached_data(self) -> bool:
        """
        Load arena card data from cache.
        
        Returns:
            True if cached data loaded successfully
        """
        try:
            if not self.arena_data_file.exists():
                self.logger.info("No cached arena data found")
                return False
            
            with open(self.arena_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert tier_data back to TierData objects if present
            tier_data = None
            if 'tier_data' in data and data['tier_data']:
                tier_data = {}
                for class_name, class_tiers in data['tier_data'].items():
                    tier_data[class_name] = {}
                    for card_id, tier_info in class_tiers.items():
                        tier_data[class_name][card_id] = TierData(
                            tier=tier_info['tier'],
                            tier_index=tier_info['tier_index'],
                            confidence=tier_info['confidence']
                        )
            
            # Create ArenaCardData with converted tier data
            data['tier_data'] = tier_data
            self.arena_data = ArenaCardData(**data)
            
            cache_age = self.get_cache_age_days()
            self.logger.info(f"✅ Loaded cached arena data ({cache_age:.1f} days old)")
            self.logger.info(f"   {self.arena_data.get_total_cards()} total arena cards")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load cached data: {e}")
            self.arena_data = None
            return False
    
    def save_arena_data(self, arena_data: ArenaCardData) -> bool:
        """
        Save arena card data to cache.
        
        Args:
            arena_data: Arena card data to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Create backup of existing data
            if self.arena_data_file.exists():
                backup_file = self.arena_data_file.with_suffix('.json.backup')
                self.arena_data_file.rename(backup_file)
            
            # Convert TierData objects to serializable format
            data_dict = asdict(arena_data)
            if data_dict['tier_data']:
                serializable_tier_data = {}
                for class_name, class_tiers in data_dict['tier_data'].items():
                    serializable_tier_data[class_name] = {}
                    for card_id, tier_data in class_tiers.items():
                        serializable_tier_data[class_name][card_id] = {
                            'tier': tier_data['tier'],
                            'tier_index': tier_data['tier_index'],
                            'confidence': tier_data['confidence']
                        }
                data_dict['tier_data'] = serializable_tier_data
            
            # Save new data
            with open(self.arena_data_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)
            
            self.arena_data = arena_data
            self.logger.info(f"✅ Saved arena data to cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save arena data: {e}")
            return False
    
    def get_cache_age_days(self) -> float:
        """Get age of cached data in days."""
        if not self.arena_data or not self.arena_data.last_updated:
            return float('inf')
        
        try:
            last_update = datetime.fromisoformat(self.arena_data.last_updated)
            age = datetime.now() - last_update
            return age.total_seconds() / (24 * 3600)  # Convert to days
        except Exception:
            return float('inf')
    
    def is_cache_stale(self) -> bool:
        """Check if cache needs updating."""
        return self.get_cache_age_days() > self.max_cache_age_days
    
    def update_with_tier_data(self, force: bool = False) -> bool:
        """
        Update arena database with HearthArena tier information.
        
        Args:
            force: Force tier data update
            
        Returns:
            True if tier integration successful
        """
        try:
            self.logger.info("🎯 Integrating HearthArena tier data...")
            
            # Update tier cache (which handles tier manager updates)
            cache_success = self.tier_cache.update_tier_cache(force=force)
            if not cache_success:
                self.logger.warning("⚠️ Tier cache update failed, proceeding without tiers")
                return True  # Don't fail the whole process
            
            # Get tier statistics from cache
            cache_stats = self.tier_cache.get_cache_statistics()
            if cache_stats['status'] != 'loaded':
                self.logger.warning("⚠️ No tier data available in cache")
                return True  # Don't fail the whole process
            
            # Integrate tier data with arena cards
            if self.arena_data:
                tier_data = {}
                
                for class_name in self.arena_data.classes.keys():
                    class_tiers = self.tier_cache.get_class_tiers(class_name)
                    
                    if class_tiers:
                        tier_data[class_name] = {}
                        
                        # Map tier data to arena cards using fuzzy matching
                        for arena_card_id in self.arena_data.classes[class_name]:
                            card_name = self.cards_loader.get_card_name(arena_card_id)
                            
                            # Try exact match first
                            if card_name in class_tiers:
                                tier_data[class_name][arena_card_id] = class_tiers[card_name]
                            else:
                                # Try fuzzy matching for name variations
                                best_match = None
                                best_score = 0.0
                                
                                for tier_card_name, tier_info in class_tiers.items():
                                    # Simple similarity check
                                    if card_name and tier_card_name:
                                        # Basic similarity - can be enhanced with rapidfuzz
                                        score = len(set(card_name.lower().split()) & set(tier_card_name.lower().split()))
                                        if score > best_score and score > 0:
                                            best_score = score
                                            best_match = tier_info
                                
                                if best_match and best_score > 0:
                                    tier_data[class_name][arena_card_id] = best_match
                
                # Update arena data with tier information
                self.arena_data.tier_data = tier_data
                
                # Update metadata with cache information
                self.arena_data.metadata['tier_integration'] = {
                    'tier_data_available': True,
                    'tier_classes': len(tier_data),
                    'total_tier_mappings': sum(len(class_tiers) for class_tiers in tier_data.values()),
                    'tier_update_time': datetime.now().isoformat(),
                    'cache_size_bytes': cache_stats.get('cache_size_bytes', 0),
                    'cache_compression_ratio': cache_stats.get('compression_ratio', 1.0)
                }
                
                self.logger.info(f"✅ Tier integration completed:")
                self.logger.info(f"   Classes with tiers: {len(tier_data)}")
                self.logger.info(f"   Total tier mappings: {sum(len(class_tiers) for class_tiers in tier_data.values())}")
                
                return True
            else:
                self.logger.warning("⚠️ No arena data available for tier integration")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Tier integration failed: {e}")
            return False
    
    def update_from_arena_version(self, force: bool = False) -> bool:
        """
        Update arena cards using Arena Tracker's approach with version manager.
        
        Args:
            force: Force update even if cache is fresh
            
        Returns:
            True if update successful
        """
        if not force and not self.is_cache_stale():
            self.logger.info("Cache is fresh, skipping arena update")
            return True
        
        self.logger.info("🚀 Starting arena version update (Arena Tracker method)...")
        
        try:
            # Update arena version data
            version_success = self.version_manager.update_arena_version(force=force)
            if not version_success:
                self.logger.error("❌ Arena version update failed")
                return False
            
            # Get version info
            version_info = self.version_manager.get_version_info()
            self.logger.info(f"✅ Arena version data updated:")
            self.logger.info(f"   Version hash: {version_info['version_hash']}")
            self.logger.info(f"   Arena sets: {version_info['arena_set_count']}")
            self.logger.info(f"   Eligible cards: {version_info['eligible_card_count']}")
            
            # Get eligible cards by class
            classes_data = {}
            all_eligible = self.version_manager.get_all_eligible_cards()
            
            # Group cards by class
            for card_id in all_eligible:
                card_data = self.cards_loader.cards_data.get(card_id, {})
                card_class = card_data.get('cardClass', 'NEUTRAL').lower()
                
                if card_class not in classes_data:
                    classes_data[card_class] = []
                classes_data[card_class].append(card_id)
            
            # Create arena card data
            arena_data = ArenaCardData(
                last_updated=datetime.now().isoformat(),
                source="arena_tracker_method",
                version=self.database_version,
                classes=classes_data,
                metadata={
                    'total_cards': len(all_eligible),
                    'version_hash': version_info['version_hash'],
                    'arena_sets': version_info['arena_sets'],
                    'arena_set_count': version_info['arena_set_count'],
                    'source_url': version_info['source_url'],
                    'method': 'arena_tracker_filtering'
                },
                raw_heartharena_data={},  # Not applicable for this method
                mapping_stats={
                    'total_input_cards': len(all_eligible),
                    'total_mapped_cards': len(all_eligible),
                    'exact_matches': len(all_eligible),
                    'fuzzy_matches': 0,
                    'normalized_matches': 0,
                    'failed_mappings': 0,
                    'success_rate': 100.0,
                    'failed_names': []
                }
            )
            
            # Validate data quality
            if not self._validate_arena_data(arena_data):
                self.logger.error("❌ Arena data validation failed")
                return False
            
            # Save to cache
            if self.save_arena_data(arena_data):
                # Integrate tier data after successful arena data update
                self.logger.info("🎯 Integrating HearthArena tier data...")
                self.update_with_tier_data(force=False)
                
                self._log_arena_version_update(version_info)
                self.logger.info("🎯 Arena Tracker update completed successfully")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Arena version update failed: {e}")
            return False
    
    # Keep old method as fallback
    def update_from_heartharena(self, force: bool = False) -> bool:
        """
        Fallback method using web scraping (deprecated).
        
        Args:
            force: Force update even if cache is fresh
            
        Returns:
            True if update successful
        """
        self.logger.warning("Using deprecated HearthArena scraping method as fallback")
        self.logger.info("Attempting Arena Tracker method instead...")
        return self.update_from_arena_version(force=force)
    
    def _map_heartharena_data(self, heartharena_cards: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Map HearthArena card names to card IDs.
        
        Args:
            heartharena_cards: Raw HearthArena data (card names)
            
        Returns:
            Dictionary with mapped classes and statistics
        """
        self.logger.info("🔗 Mapping HearthArena card names to card IDs...")
        
        mapped_classes = {}
        overall_stats = {
            'total_input_cards': 0,
            'total_mapped_cards': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'normalized_matches': 0,
            'failed_mappings': 0,
            'success_rate': 0.0,
            'failed_names': []
        }
        
        for class_name, card_names in heartharena_cards.items():
            self.logger.info(f"   Mapping {class_name}: {len(card_names)} cards")
            
            mapped_card_ids = []
            class_stats = self.cards_loader.get_mapping_statistics(card_names)
            
            # Map each card name
            for name in card_names:
                match = self.cards_loader.get_card_id_fuzzy(name)
                if match:
                    mapped_card_ids.append(match.card_id)
                else:
                    overall_stats['failed_names'].append(f"{class_name}:{name}")
            
            mapped_classes[class_name.lower()] = mapped_card_ids
            
            # Aggregate statistics
            overall_stats['total_input_cards'] += class_stats['total_names']
            overall_stats['total_mapped_cards'] += len(mapped_card_ids)
            overall_stats['exact_matches'] += class_stats['exact_matches']
            overall_stats['fuzzy_matches'] += class_stats['fuzzy_matches']
            overall_stats['normalized_matches'] += class_stats['normalized_matches']
            overall_stats['failed_mappings'] += class_stats['no_matches']
            
            self.logger.info(f"   ✅ {class_name}: {len(mapped_card_ids)}/{len(card_names)} mapped "
                           f"({len(mapped_card_ids)/len(card_names)*100:.1f}%)")
        
        # Calculate overall success rate
        if overall_stats['total_input_cards'] > 0:
            overall_stats['success_rate'] = (overall_stats['total_mapped_cards'] / 
                                           overall_stats['total_input_cards'] * 100)
        
        self.logger.info(f"🎯 Overall mapping: {overall_stats['total_mapped_cards']}/{overall_stats['total_input_cards']} "
                        f"({overall_stats['success_rate']:.1f}% success rate)")
        
        return {
            'classes': mapped_classes,
            'stats': overall_stats
        }
    
    def _validate_arena_data(self, arena_data: ArenaCardData) -> bool:
        """
        Validate arena card data quality.
        
        Args:
            arena_data: Arena data to validate
            
        Returns:
            True if data passes validation
        """
        try:
            # Check total card count
            total_cards = arena_data.get_total_cards()
            if total_cards < 1000:  # Expect at least 1000 arena cards
                self.logger.error(f"❌ Too few cards: {total_cards} (expected > 1000)")
                return False
            
            if total_cards > 3000:  # Sanity check upper bound
                self.logger.error(f"❌ Too many cards: {total_cards} (expected < 3000)")
                return False
            
            # Check class distribution
            for class_name, cards in arena_data.classes.items():
                if class_name == 'neutral':
                    if len(cards) < 200:  # Expect at least 200 neutral cards
                        self.logger.error(f"❌ Too few neutral cards: {len(cards)}")
                        return False
                else:
                    if len(cards) < 50:  # Expect at least 50 cards per class
                        self.logger.error(f"❌ Too few {class_name} cards: {len(cards)}")
                        return False
            
            # Check mapping success rate
            success_rate = arena_data.mapping_stats.get('success_rate', 0)
            if success_rate < self.min_success_rate:
                self.logger.error(f"❌ Low mapping success rate: {success_rate:.1f}% "
                                f"(minimum: {self.min_success_rate}%)")
                return False
            
            self.logger.info(f"✅ Arena data validation passed")
            self.logger.info(f"   Total cards: {total_cards}")
            self.logger.info(f"   Mapping success: {success_rate:.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation error: {e}")
            return False
    
    def _log_update(self, scraping_result: Any, mapping_stats: Dict[str, Any]):
        """Log update history for debugging and monitoring."""
        try:
            update_entry = {
                'timestamp': datetime.now().isoformat(),
                'scraping_success': getattr(scraping_result, 'success', True),
                'total_scraped_cards': getattr(scraping_result, 'total_cards', 0),
                'scraping_time': getattr(scraping_result, 'scraping_time', 0),
                'scraping_errors': getattr(scraping_result, 'errors', []),
                'mapping_success_rate': mapping_stats['success_rate'],
                'total_mapped_cards': mapping_stats['total_mapped_cards'],
                'failed_mappings': len(mapping_stats['failed_names'])
            }
            
            # Load existing history
            history = []
            if self.update_log_file.exists():
                with open(self.update_log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # Add new entry and keep last 50 updates
            history.append(update_entry)
            history = history[-50:]
            
            # Save history
            with open(self.update_log_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to log update: {e}")
    
    def _log_arena_version_update(self, version_info: Dict[str, Any]):
        """Log arena version update history for debugging and monitoring."""
        try:
            update_entry = {
                'timestamp': datetime.now().isoformat(),
                'method': 'arena_tracker_filtering',
                'version_hash': version_info['version_hash'],
                'arena_set_count': version_info['arena_set_count'],
                'eligible_card_count': version_info['eligible_card_count'],
                'source_url': version_info['source_url'],
                'multiclass_enabled': version_info.get('multiclass_enabled', False),
                'special_event': version_info.get('special_event')
            }
            
            # Load existing history
            history = []
            if self.update_log_file.exists():
                with open(self.update_log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # Add new entry and keep last 50 updates
            history.append(update_entry)
            history = history[-50:]
            
            # Save history
            with open(self.update_log_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to log arena version update: {e}")
    
    def get_arena_cards_for_class(self, class_name: str) -> List[str]:
        """
        Get arena-eligible cards for specific class including neutrals.
        
        Args:
            class_name: Class name (e.g., 'mage', 'warrior')
            
        Returns:
            List of card IDs eligible for the class
        """
        if not self.arena_data:
            self.logger.warning("No arena data available")
            return []
        
        return self.arena_data.get_cards_for_class(class_name)
    
    def get_all_arena_cards(self) -> List[str]:
        """Get all arena-eligible cards across all classes."""
        if not self.arena_data:
            self.logger.warning("No arena_data available - returning empty list")
            return []
        
        all_cards = set()
        for class_name, cards in self.arena_data.classes.items():
            all_cards.update(cards)
            if len(all_cards) < 10:  # Log first few cards for debugging
                self.logger.debug(f"Arena cards for {class_name}: {len(cards)} cards")
        
        card_list = list(all_cards)
        self.logger.info(f"Total arena-eligible cards: {len(card_list)}")
        
        return card_list
    
    def is_card_arena_eligible(self, card_id: str, class_name: str = None) -> bool:
        """
        Check if card is arena-eligible.
        
        Args:
            card_id: Card ID to check
            class_name: Specific class to check (None for any class)
            
        Returns:
            True if card is arena-eligible
        """
        if not self.arena_data:
            return False
        
        return self.arena_data.is_arena_eligible(card_id, class_name)
    
    def get_arena_card_counts(self) -> Dict[str, int]:
        """Get card counts by class."""
        if not self.arena_data:
            return {}
        
        return {class_name: len(cards) 
                for class_name, cards in self.arena_data.classes.items()}
    
    def get_arena_cards_with_tiers(self, class_name: str) -> Dict[str, Optional[TierData]]:
        """
        Get arena cards for a class with their tier information.
        
        Args:
            class_name: Class name (e.g., 'mage', 'warrior')
            
        Returns:
            Dictionary mapping card IDs to tier data (None if no tier available)
        """
        if not self.arena_data:
            return {}
        
        return self.arena_data.get_cards_with_tiers(class_name)
    
    def get_card_tier_info(self, card_id: str, class_name: str) -> Optional[TierData]:
        """
        Get tier information for a specific card.
        
        Args:
            card_id: Card ID to check
            class_name: Class name for tier lookup
            
        Returns:
            TierData if available, None otherwise
        """
        if not self.arena_data:
            return None
        
        return self.arena_data.get_card_tier(card_id, class_name)
    
    def get_card_tier_fast(self, card_name: str, class_name: str) -> Optional[TierData]:
        """
        Get tier information directly from tier cache (faster lookup).
        
        Args:
            card_name: Card name (not ID)
            class_name: Class name for tier lookup
            
        Returns:
            TierData if available, None otherwise
        """
        return self.tier_cache.get_card_tier(card_name, class_name)
    
    def get_arena_histograms(self, class_name: str = None) -> Dict[str, Any]:
        """
        Get histogram data for arena-eligible cards (for GUI Arena Priority logic).
        
        This method provides the focused arena database that the GUI needs for 
        creating arena-priority histogram matchers.
        
        Args:
            class_name: Specific class to filter by (None for all arena cards)
            
        Returns:
            Dictionary mapping card codes to histogram data for arena-eligible cards
        """
        try:
            if not self.arena_data:
                self.logger.warning("No arena data available for histogram creation")
                return {}
            
            # Get arena-eligible card IDs
            if class_name:
                eligible_cards = self.get_arena_cards_for_class(class_name)
                self.logger.info(f"Creating arena histograms for {class_name}: {len(eligible_cards)} cards")
            else:
                eligible_cards = self.get_all_arena_cards()
                self.logger.info(f"Creating arena histograms for all classes: {len(eligible_cards)} cards")
            
            # Load asset loader for histogram computation
            try:
                from ..utils.asset_loader import get_asset_loader
                asset_loader = get_asset_loader()
            except ImportError:
                self.logger.error("Cannot import asset_loader for histogram computation")
                return {}
            
            # Create histogram dictionary for arena-eligible cards only
            arena_histograms = {}
            
            success_count = 0
            for i, card_id in enumerate(eligible_cards):
                try:
                    # Add debug logging for first few cards
                    if i < 5:
                        self.logger.info(f"Attempting to load card {i+1}: {card_id}")
                    
                    # Load card image (normal version)
                    card_image = asset_loader.load_card_image(card_id, premium=False)
                    if card_image is not None:
                        # Import histogram calculation function
                        from ..detection.histogram_matcher import calculate_histogram
                        histogram = calculate_histogram(card_image)
                        arena_histograms[card_id] = histogram
                        success_count += 1
                    elif i < 5:
                        self.logger.warning(f"Failed to load normal image for {card_id}")
                    
                    # Also load premium version if available
                    premium_image = asset_loader.load_card_image(card_id, premium=True)
                    if premium_image is not None:
                        from ..detection.histogram_matcher import calculate_histogram
                        premium_histogram = calculate_histogram(premium_image)
                        arena_histograms[f"{card_id}_premium"] = premium_histogram
                        success_count += 1
                    elif i < 5:
                        self.logger.debug(f"No premium image for {card_id}")
                        
                except Exception as e:
                    if i < 5:
                        self.logger.error(f"Exception loading histogram for {card_id}: {e}")
                    else:
                        self.logger.debug(f"Failed to load histogram for {card_id}: {e}")
                    continue
            
            self.logger.info(f"✅ Created {len(arena_histograms)} arena histograms from {len(eligible_cards)} eligible cards (success rate: {success_count}/{len(eligible_cards)*2} = {success_count/(len(eligible_cards)*2)*100:.1f}%)")
            return arena_histograms
            
        except Exception as e:
            self.logger.error(f"Failed to create arena histograms: {e}")
            return {}
    
    def get_tier_cache_info(self) -> Dict[str, Any]:
        """Get information about the tier cache performance."""
        return self.tier_cache.get_cache_statistics()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information."""
        if not self.arena_data:
            return {
                'status': 'no_data',
                'cache_age_days': float('inf'),
                'needs_update': True
            }
        
        cache_age = self.get_cache_age_days()
        
        return {
            'status': 'loaded',
            'last_updated': self.arena_data.last_updated,
            'cache_age_days': cache_age,
            'needs_update': self.is_cache_stale(),
            'source': self.arena_data.source,
            'version': self.arena_data.version,
            'total_cards': self.arena_data.get_total_cards(),
            'card_counts': self.get_arena_card_counts(),
            'mapping_stats': self.arena_data.mapping_stats,
            'raw_data_available': bool(self.arena_data.raw_heartharena_data),
            'tier_stats': self.arena_data.get_tier_statistics(),
            'tier_cache_info': self.get_tier_cache_info()
        }
    
    def get_failed_mappings(self) -> List[str]:
        """Get list of card names that failed to map."""
        if not self.arena_data or not self.arena_data.mapping_stats:
            return []
        
        return self.arena_data.mapping_stats.get('failed_names', [])
    
    def has_data(self) -> bool:
        """Check if arena data is available."""
        return self.arena_data is not None
    
    def check_for_updates(self) -> Tuple[bool, str]:
        """
        Check if arena database needs updating.
        
        Returns:
            Tuple of (needs_update, reason)
        """
        if not self.has_data():
            return True, "No arena data available"
        
        if self.is_cache_stale():
            cache_age = self.get_cache_age_days()
            return True, f"Cache is {cache_age:.1f} days old (max: {self.max_cache_age_days})"
        
        return False, "Cache is fresh"


# Global instance
_arena_card_database = None


def get_arena_card_database() -> ArenaCardDatabase:
    """
    Get the global arena card database instance.
    
    Returns:
        ArenaCardDatabase instance
    """
    global _arena_card_database
    if _arena_card_database is None:
        _arena_card_database = ArenaCardDatabase()
    return _arena_card_database


if __name__ == "__main__":
    # Test the arena card database
    logging.basicConfig(level=logging.INFO)
    
    db = get_arena_card_database()
    
    print("Arena Card Database Test")
    print("=" * 40)
    
    # Show current status
    info = db.get_database_info()
    print(f"Status: {info['status']}")
    
    if info['status'] == 'loaded':
        print(f"Last updated: {info['last_updated']}")
        print(f"Cache age: {info['cache_age_days']:.1f} days")
        print(f"Total cards: {info['total_cards']}")
        print("\nCard counts by class:")
        for class_name, count in info['card_counts'].items():
            print(f"  {class_name}: {count}")
        
        # Show tier statistics if available
        tier_stats = info.get('tier_stats', {})
        if tier_stats.get('has_tier_data'):
            print(f"\nTier data:")
            print(f"  Classes with tiers: {tier_stats['classes_with_tiers']}")
            print(f"  Total tier mappings: {tier_stats['total_cards_with_tiers']}")
            print(f"  Tier distribution:")
            for tier, count in tier_stats.get('tier_distribution', {}).items():
                print(f"    {tier}: {count} cards")
        else:
            print("\nNo tier data available")
        
        # Show tier cache performance
        cache_info = info.get('tier_cache_info', {})
        if cache_info.get('status') == 'loaded':
            print(f"\nTier cache performance:")
            print(f"  Cache size: {cache_info['cache_size_bytes']:,} bytes")
            print(f"  Compression: {cache_info.get('compression_ratio', 1.0):.1f}x")
            if 'performance' in cache_info:
                perf = cache_info['performance']
                print(f"  Save time: {perf['save_time_ms']:.1f}ms")
                print(f"  Efficiency: {perf['compression_efficiency']:.1f}%")
    
    # Check if update needed
    needs_update, reason = db.check_for_updates()
    print(f"\nUpdate needed: {needs_update}")
    if needs_update:
        print(f"Reason: {reason}")
        
        # Ask user if they want to update
        response = input("\nUpdate arena database? (y/N): ").lower()
        if response == 'y':
            print("\nUpdating arena database...")
            success = db.update_from_arena_version(force=True)
            if success:
                print("✅ Update completed successfully!")
                
                # Show updated info
                info = db.get_database_info()
                print(f"New total: {info['total_cards']} cards")
                print(f"Mapping success: {info['mapping_stats']['success_rate']:.1f}%")
                
                # Show tier integration results
                tier_stats = info.get('tier_stats', {})
                if tier_stats.get('has_tier_data'):
                    print(f"Tier integration: {tier_stats['total_cards_with_tiers']} cards with tiers")
                else:
                    print("Tier integration: No tier data integrated")
            else:
                print("❌ Update failed!")