"""
Arena Version Manager

Implements Arena Tracker's arena eligibility filtering system.
Downloads current arena rotation data and manages card set filtering.
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import hashlib
import time

try:
    from .cards_json_loader import get_cards_json_loader
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent))
    from cards_json_loader import get_cards_json_loader


@dataclass
class ArenaVersionData:
    """Container for arena version information."""
    arena_sets: List[str]
    banned_cards: List[str]
    multiclass_enabled: bool
    rarity_restrictions: Dict[str, Any]
    special_event: Optional[str]
    version_hash: str
    last_updated: str
    source_url: str


@dataclass
class EligibilityStats:
    """Statistics for arena eligibility filtering."""
    total_cards: int
    after_set_filtering: int
    after_class_filtering: int
    after_ban_filtering: int
    after_rarity_filtering: int
    final_eligible: int
    filtering_time_ms: float


class ArenaVersionManager:
    """
    Manages arena version data and card eligibility filtering.
    
    Implements Arena Tracker's approach with downloadable JSON files
    for current arena rotation information.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize arena version manager.
        
        Args:
            cache_dir: Directory for cached data (uses default if None)
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "arena_version"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.version_file = self.cache_dir / "arena_version.json"
        self.eligible_cards_file = self.cache_dir / "eligible_cards.json"
        self.stats_file = self.cache_dir / "filtering_stats.json"
        
        # Configuration - Updated working URLs
        self.arena_urls = [
            "https://api.hearthstonejson.com/v1/latest/enUS/metadata.json",  # HearthstoneJSON metadata
            "https://hearthstonejson.com/docs/arena_sets.json",  # Custom arena sets
            "https://raw.githubusercontent.com/HearthSim/hsdata/master/arena_sets.json",  # HSData
        ]
        
        self.max_cache_age_hours = 6  # Update every 6 hours
        self.timeout_seconds = 30
        
        # Runtime data
        self.arena_version: Optional[ArenaVersionData] = None
        self.eligible_cards: Set[str] = set()
        self.cards_loader = get_cards_json_loader()
        
        # Default arena sets (more conservative fallback with recent sets)
        self.default_arena_sets = [
            "CORE", "EXPERT1", "TITANS", "WONDERS", "WHIZBANGS_WORKSHOP", 
            "PATH_OF_ARTHAS", "REVENDRETH", "SUNKEN_CITY", "ALTERAC_VALLEY",
            "STORMWIND", "THE_BARRENS", "DARKMOON_FAIRE"
        ]
        
        # Static ban list (problematic cards)
        self.static_banned_cards = [
            "HERO_01",  # Jaina
            "HERO_02",  # Rexxar
            "HERO_03",  # Uther
            "HERO_04",  # Malfurion
            "HERO_05",  # Anduin
            "HERO_06",  # Valeera
            "HERO_07",  # Thrall
            "HERO_08",  # Guldan
            "HERO_09",  # Garrosh
            "HERO_10",  # Illidan
        ]
        
        # Load existing data
        self.load_cached_data()
        
        self.logger.info("ArenaVersionManager initialized")
    
    def load_cached_data(self) -> bool:
        """
        Load arena version data from cache.
        
        Returns:
            True if cached data loaded successfully
        """
        try:
            if not self.version_file.exists():
                self.logger.info("No cached arena version data found")
                return False
            
            with open(self.version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.arena_version = ArenaVersionData(**data)
            
            # Load eligible cards
            if self.eligible_cards_file.exists():
                with open(self.eligible_cards_file, 'r', encoding='utf-8') as f:
                    cards_data = json.load(f)
                    self.eligible_cards = set(cards_data.get('eligible_cards', []))
            
            cache_age = self.get_cache_age_hours()
            self.logger.info(f"✅ Loaded cached arena version data ({cache_age:.1f} hours old)")
            self.logger.info(f"   Arena sets: {len(self.arena_version.arena_sets)}")
            self.logger.info(f"   Banned cards: {len(self.arena_version.banned_cards)}")
            self.logger.info(f"   Eligible cards: {len(self.eligible_cards)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load cached arena version data: {e}")
            self.arena_version = None
            self.eligible_cards = set()
            return False
    
    def save_arena_version_data(self, version_data: ArenaVersionData) -> bool:
        """
        Save arena version data to cache.
        
        Args:
            version_data: Arena version data to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Create backup of existing data
            if self.version_file.exists():
                backup_file = self.version_file.with_suffix('.json.backup')
                self.version_file.rename(backup_file)
            
            # Save new data
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(version_data), f, indent=2, ensure_ascii=False)
            
            self.arena_version = version_data
            self.logger.info("✅ Saved arena version data to cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save arena version data: {e}")
            return False
    
    def get_cache_age_hours(self) -> float:
        """Get age of cached data in hours."""
        if not self.arena_version or not self.arena_version.last_updated:
            return float('inf')
        
        try:
            last_update = datetime.fromisoformat(self.arena_version.last_updated)
            age = datetime.now() - last_update
            return age.total_seconds() / 3600  # Convert to hours
        except Exception:
            return float('inf')
    
    def is_cache_stale(self) -> bool:
        """Check if cache needs updating."""
        return self.get_cache_age_hours() > self.max_cache_age_hours
    
    def download_arena_version(self) -> Optional[ArenaVersionData]:
        """
        Download current arena version data from multiple sources.
        
        Returns:
            ArenaVersionData if successful, None otherwise
        """
        self.logger.info("🌐 Downloading arena version data...")
        
        for url in self.arena_urls:
            try:
                self.logger.info(f"   Trying: {url}")
                
                response = requests.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                
                data = response.json()
                
                # Parse different URL formats
                if "arenaSets.json" in url:
                    # Arena Tracker format
                    arena_sets = data.get("arenaSets", [])
                    banned_cards = data.get("bannedCards", [])
                    multiclass_enabled = data.get("multiclassEnabled", False)
                    rarity_restrictions = data.get("rarityRestrictions", {})
                    special_event = data.get("specialEvent")
                    
                elif "cards.json" in url:
                    # HearthstoneJSON format - extract sets
                    all_sets = set()
                    for card in data:
                        if card.get("set"):
                            all_sets.add(card["set"])
                    
                    arena_sets = list(all_sets)
                    banned_cards = []
                    multiclass_enabled = False
                    rarity_restrictions = {}
                    special_event = None
                    
                else:
                    # Generic format
                    arena_sets = data.get("sets", self.default_arena_sets)
                    banned_cards = data.get("banned", [])
                    multiclass_enabled = data.get("multiclass", False)
                    rarity_restrictions = data.get("rarity", {})
                    special_event = data.get("event")
                
                # Create version hash
                version_content = json.dumps(sorted(arena_sets) + sorted(banned_cards))
                version_hash = hashlib.md5(version_content.encode()).hexdigest()[:8]
                
                version_data = ArenaVersionData(
                    arena_sets=arena_sets,
                    banned_cards=banned_cards,
                    multiclass_enabled=multiclass_enabled,
                    rarity_restrictions=rarity_restrictions,
                    special_event=special_event,
                    version_hash=version_hash,
                    last_updated=datetime.now().isoformat(),
                    source_url=url
                )
                
                self.logger.info(f"✅ Downloaded arena version data from {url}")
                self.logger.info(f"   Arena sets: {len(arena_sets)}")
                self.logger.info(f"   Banned cards: {len(banned_cards)}")
                self.logger.info(f"   Version hash: {version_hash}")
                
                return version_data
                
            except Exception as e:
                self.logger.warning(f"   Failed to download from {url}: {e}")
                continue
        
        self.logger.error("❌ Failed to download arena version data from any source")
        return None
    
    def update_arena_version(self, force: bool = False) -> bool:
        """
        Update arena version data.
        
        Args:
            force: Force update even if cache is fresh
            
        Returns:
            True if update successful
        """
        if not force and not self.is_cache_stale():
            self.logger.info("Arena version cache is fresh, skipping update")
            return True
        
        self.logger.info("🚀 Starting arena version update...")
        
        # Download new version data
        new_version = self.download_arena_version()
        if not new_version:
            # Fallback to default if download fails
            self.logger.warning("Using default arena sets as fallback")
            new_version = ArenaVersionData(
                arena_sets=self.default_arena_sets,
                banned_cards=self.static_banned_cards,
                multiclass_enabled=False,
                rarity_restrictions={},
                special_event=None,
                version_hash="default",
                last_updated=datetime.now().isoformat(),
                source_url="fallback"
            )
        
        # Save to cache
        if self.save_arena_version_data(new_version):
            # Update eligible cards
            self.update_eligible_cards()
            self.logger.info("🎯 Arena version update completed successfully")
            return True
        else:
            return False
    
    def update_eligible_cards(self) -> bool:
        """
        Update the list of arena-eligible cards based on current version.
        
        Returns:
            True if update successful
        """
        if not self.arena_version:
            self.logger.error("No arena version data available")
            return False
        
        start_time = time.time()
        self.logger.info("🔍 Filtering cards for arena eligibility...")
        
        # Get all cards from loader
        all_cards = self.cards_loader.cards_data
        
        # Initialize stats
        stats = EligibilityStats(
            total_cards=len(all_cards),
            after_set_filtering=0,
            after_class_filtering=0,
            after_ban_filtering=0,
            after_rarity_filtering=0,
            final_eligible=0,
            filtering_time_ms=0
        )
        
        eligible_cards = set()
        
        # Stage 1: Set filtering
        for card_id, card_data in all_cards.items():
            card_set = card_data.get('set', '')
            if card_set in self.arena_version.arena_sets:
                eligible_cards.add(card_id)
        
        stats.after_set_filtering = len(eligible_cards)
        self.logger.info(f"   After set filtering: {stats.after_set_filtering} cards")
        
        # Stage 2: Class filtering (keep all for now - will be filtered per-draft)
        stats.after_class_filtering = len(eligible_cards)
        
        # Stage 3: Ban filtering
        banned_set = set(self.arena_version.banned_cards + self.static_banned_cards)
        eligible_cards -= banned_set
        stats.after_ban_filtering = len(eligible_cards)
        self.logger.info(f"   After ban filtering: {stats.after_ban_filtering} cards")
        
        # Stage 4: Rarity filtering (if restrictions exist)
        if self.arena_version.rarity_restrictions:
            allowed_rarities = set(self.arena_version.rarity_restrictions.get('allowed', []))
            if allowed_rarities:
                filtered_cards = set()
                for card_id in eligible_cards:
                    card_data = all_cards.get(card_id, {})
                    rarity = card_data.get('rarity', '').upper()
                    if rarity in allowed_rarities:
                        filtered_cards.add(card_id)
                eligible_cards = filtered_cards
        
        stats.after_rarity_filtering = len(eligible_cards)
        stats.final_eligible = len(eligible_cards)
        
        # Calculate timing
        stats.filtering_time_ms = (time.time() - start_time) * 1000
        
        # Save results
        self.eligible_cards = eligible_cards
        
        # Save to cache
        eligible_data = {
            'eligible_cards': list(eligible_cards),
            'stats': asdict(stats),
            'version_hash': self.arena_version.version_hash,
            'updated_at': datetime.now().isoformat()
        }
        
        try:
            with open(self.eligible_cards_file, 'w', encoding='utf-8') as f:
                json.dump(eligible_data, f, indent=2)
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(stats), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save eligible cards cache: {e}")
        
        # Log final statistics
        total_reduction = (1 - stats.final_eligible / stats.total_cards) * 100
        self.logger.info(f"✅ Arena eligibility filtering completed:")
        self.logger.info(f"   Total cards: {stats.total_cards}")
        self.logger.info(f"   Final eligible: {stats.final_eligible}")
        self.logger.info(f"   Reduction: {total_reduction:.1f}%")
        self.logger.info(f"   Processing time: {stats.filtering_time_ms:.1f}ms")
        
        return True
    
    def get_eligible_cards_for_class(self, hero_class: str, partner_class: Optional[str] = None) -> Set[str]:
        """
        Get arena-eligible cards for a specific class.
        
        Args:
            hero_class: Primary hero class
            partner_class: Partner class for multiclass arena
            
        Returns:
            Set of eligible card IDs
        """
        if not self.eligible_cards:
            self.logger.warning("No eligible cards loaded")
            return set()
        
        class_filtered = set()
        
        for card_id in self.eligible_cards:
            card_data = self.cards_loader.cards_data.get(card_id, {})
            card_class = card_data.get('cardClass', '').upper()
            
            # Always include neutral cards
            if card_class == 'NEUTRAL':
                class_filtered.add(card_id)
                continue
            
            # Include hero class cards
            if card_class == hero_class.upper():
                class_filtered.add(card_id)
                continue
            
            # Include partner class cards (multiclass arena)
            if partner_class and card_class == partner_class.upper():
                class_filtered.add(card_id)
                continue
        
        return class_filtered
    
    def get_all_eligible_cards(self) -> Set[str]:
        """Get all arena-eligible cards regardless of class."""
        return self.eligible_cards.copy()
    
    def is_card_eligible(self, card_id: str, hero_class: Optional[str] = None) -> bool:
        """
        Check if a card is arena-eligible.
        
        Args:
            card_id: Card ID to check
            hero_class: Hero class for class-specific check
            
        Returns:
            True if card is eligible
        """
        if hero_class:
            eligible_for_class = self.get_eligible_cards_for_class(hero_class)
            return card_id in eligible_for_class
        else:
            return card_id in self.eligible_cards
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information."""
        if not self.arena_version:
            return {
                'status': 'no_data',
                'cache_age_hours': float('inf'),
                'needs_update': True
            }
        
        cache_age = self.get_cache_age_hours()
        
        return {
            'status': 'loaded',
            'version_hash': self.arena_version.version_hash,
            'last_updated': self.arena_version.last_updated,
            'cache_age_hours': cache_age,
            'needs_update': self.is_cache_stale(),
            'source_url': self.arena_version.source_url,
            'arena_sets': self.arena_version.arena_sets,
            'arena_set_count': len(self.arena_version.arena_sets),
            'banned_card_count': len(self.arena_version.banned_cards),
            'eligible_card_count': len(self.eligible_cards),
            'multiclass_enabled': self.arena_version.multiclass_enabled,
            'special_event': self.arena_version.special_event
        }
    
    def has_data(self) -> bool:
        """Check if arena version data is available."""
        return self.arena_version is not None and len(self.eligible_cards) > 0


# Global instance
_arena_version_manager = None


def get_arena_version_manager() -> ArenaVersionManager:
    """
    Get the global arena version manager instance.
    
    Returns:
        ArenaVersionManager instance
    """
    global _arena_version_manager
    if _arena_version_manager is None:
        _arena_version_manager = ArenaVersionManager()
    return _arena_version_manager


if __name__ == "__main__":
    # Test the arena version manager
    logging.basicConfig(level=logging.INFO)
    
    manager = get_arena_version_manager()
    
    print("Arena Version Manager Test")
    print("=" * 40)
    
    # Show current status
    info = manager.get_version_info()
    print(f"Status: {info['status']}")
    
    if info['status'] == 'loaded':
        print(f"Version hash: {info['version_hash']}")
        print(f"Last updated: {info['last_updated']}")
        print(f"Cache age: {info['cache_age_hours']:.1f} hours")
        print(f"Arena sets: {info['arena_set_count']}")
        print(f"Eligible cards: {info['eligible_card_count']}")
    
    # Check if update needed
    if info.get('needs_update', True):
        print(f"\nUpdating arena version data...")
        success = manager.update_arena_version(force=True)
        if success:
            print("✅ Update completed successfully!")
            
            # Show updated info
            info = manager.get_version_info()
            print(f"New eligible cards: {info['eligible_card_count']}")
        else:
            print("❌ Update failed!")
    
    # Test class filtering
    print(f"\nTesting class filtering...")
    mage_cards = manager.get_eligible_cards_for_class("MAGE")
    print(f"Mage eligible cards: {len(mage_cards)}")
    
    warrior_cards = manager.get_eligible_cards_for_class("WARRIOR")
    print(f"Warrior eligible cards: {len(warrior_cards)}")