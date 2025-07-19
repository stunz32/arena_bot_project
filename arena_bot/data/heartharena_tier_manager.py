"""
HearthArena Tier Manager

Implements EzArena's proven approach for scraping HearthArena tier lists.
Uses Beautiful Soup HTML parsing for fast, reliable tier data extraction.
"""

import logging
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import json

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from .cards_json_loader import get_cards_json_loader
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent))
    from cards_json_loader import get_cards_json_loader


@dataclass
class TierData:
    """Container for tier information."""
    tier: str
    tier_index: int  # 0 (best) to 7 (worst)
    confidence: float


@dataclass
class HearthArenaTierResult:
    """Container for HearthArena tier scraping results."""
    success: bool
    classes: Dict[str, Dict[str, TierData]]  # {class: {card_name: tier_data}}
    total_cards: int
    scraping_time: float
    errors: List[str]
    timestamp: str
    source_url: str


class HearthArenaTierManager:
    """
    Manages HearthArena tier data using EzArena's proven scraping approach.
    
    Uses Beautiful Soup HTML parsing for fast, reliable extraction of
    HearthArena's tier list data with proper caching and error handling.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize HearthArena tier manager.
        
        Args:
            cache_dir: Directory for cached tier data
        """
        self.logger = logging.getLogger(__name__)
        
        if not BS4_AVAILABLE:
            self.logger.error("Beautiful Soup not available. Install with: pip install beautifulsoup4")
            raise ImportError("Beautiful Soup is required for HearthArena tier scraping")
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache" / "heartharena_tiers"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.tier_data_file = self.cache_dir / "tier_data.json"
        self.tier_cache_file = self.cache_dir / "tier_cache.json"
        self.update_log_file = self.cache_dir / "update_history.json"
        
        # HearthArena configuration
        self.base_url = "https://www.heartharena.com/tierlist"
        self.timeout_seconds = 30
        self.max_cache_age_hours = 24  # Update daily
        
        # EzArena's exact tier system (ordered from best to worst)
        self.TIERS = [
            'beyond-great',   # 0 - Best tier
            'great',          # 1
            'good',           # 2
            'above-average',  # 3
            'average',        # 4
            'below-average',  # 5
            'bad',            # 6
            'terrible'        # 7 - Worst tier
        ]
        
        # HearthArena class names
        self.class_names = [
            'mage', 'warrior', 'hunter', 'priest', 'warlock',
            'rogue', 'shaman', 'paladin', 'druid', 'demon-hunter'
        ]
        
        # Runtime data
        self.tier_data: Optional[Dict[str, Dict[str, TierData]]] = None
        self.cards_loader = get_cards_json_loader()
        
        # Load existing data
        self.load_cached_data()
        
        self.logger.info("HearthArenaTierManager initialized")
    
    def load_cached_data(self) -> bool:
        """
        Load tier data from cache.
        
        Returns:
            True if cached data loaded successfully
        """
        try:
            if not self.tier_data_file.exists():
                self.logger.info("No cached tier data found")
                return False
            
            with open(self.tier_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to TierData objects
            self.tier_data = {}
            for class_name, class_tiers in data.get('classes', {}).items():
                self.tier_data[class_name] = {}
                for card_name, tier_info in class_tiers.items():
                    self.tier_data[class_name][card_name] = TierData(
                        tier=tier_info['tier'],
                        tier_index=tier_info['tier_index'],
                        confidence=tier_info['confidence']
                    )
            
            cache_age = self.get_cache_age_hours()
            self.logger.info(f"✅ Loaded cached tier data ({cache_age:.1f} hours old)")
            self.logger.info(f"   Classes: {len(self.tier_data)}")
            
            total_cards = sum(len(class_tiers) for class_tiers in self.tier_data.values())
            self.logger.info(f"   Total tier entries: {total_cards}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load cached tier data: {e}")
            self.tier_data = None
            return False
    
    def get_cache_age_hours(self) -> float:
        """Get age of cached data in hours."""
        if not self.tier_data_file.exists():
            return float('inf')
        
        try:
            mtime = self.tier_data_file.stat().st_mtime
            age = time.time() - mtime
            return age / 3600  # Convert to hours
        except Exception:
            return float('inf')
    
    def is_cache_stale(self) -> bool:
        """Check if cache needs updating."""
        return self.get_cache_age_hours() > self.max_cache_age_hours
    
    def scrape_heartharena_tierlist(self) -> HearthArenaTierResult:
        """
        Scrape HearthArena tier list using EzArena's proven approach.
        
        Returns:
            HearthArenaTierResult with complete tier data
        """
        start_time = time.time()
        errors = []
        classes_data = {}
        
        try:
            self.logger.info("🌐 Scraping HearthArena tier list (EzArena method)...")
            
            # Download HTML page (EzArena's approach)
            response = requests.get(self.base_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            
            # Parse with Beautiful Soup
            soup = BeautifulSoup(response.text, 'html.parser')
            self.logger.info(f"✅ Downloaded and parsed HearthArena page")
            
            # Extract tier data for each class (EzArena's method)
            for class_name in self.class_names:
                try:
                    self.logger.info(f"🎯 Extracting tiers for {class_name}...")
                    
                    # Find the section for this class (id=class_name)
                    tierlist = soup.find(id=class_name)
                    
                    if not tierlist:
                        self.logger.warning(f"⚠️ No section found for class: {class_name}")
                        errors.append(f"No section found for {class_name}")
                        classes_data[class_name] = {}
                        continue
                    
                    class_cards = {}
                    
                    # Extract cards from each tier (EzArena's exact approach)
                    for tier_index, tier in enumerate(self.TIERS):
                        try:
                            # Find all tier blocks for this tier level
                            tier_blocks = tierlist.find_all(class_=f"tier {tier}")
                            
                            for tier_block in tier_blocks:
                                # Find the ordered list within this tier block
                                ol = tier_block.find('ol')
                                if not ol:
                                    continue
                                
                                # Extract card names from <dt> tags
                                card_elements = ol.find_all('dt')
                                
                                for card_element in card_elements:
                                    card_text = card_element.get_text()
                                    
                                    # Skip blank placeholders (EzArena's check)
                                    if card_text == '\xa0':
                                        break
                                    
                                    # Remove trailing score/colon (EzArena's approach)
                                    card_name = card_text.strip()
                                    if card_name.endswith(':'):
                                        card_name = card_name[:-1]
                                    
                                    # Remove trailing digits (score) more robustly
                                    while card_name and card_name[-1].isdigit():
                                        card_name = card_name[:-1]
                                    
                                    card_name = card_name.strip()
                                    
                                    if card_name and len(card_name) > 1:
                                        class_cards[card_name] = TierData(
                                            tier=tier,
                                            tier_index=tier_index,
                                            confidence=1.0  # Full confidence for HearthArena data
                                        )
                        
                        except Exception as e:
                            self.logger.debug(f"Error processing tier {tier} for {class_name}: {e}")
                            continue
                    
                    classes_data[class_name] = class_cards
                    self.logger.info(f"✅ {class_name}: {len(class_cards)} cards across all tiers")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing class {class_name}: {e}")
                    errors.append(f"Error processing {class_name}: {str(e)}")
                    classes_data[class_name] = {}
            
            # Calculate totals
            total_cards = sum(len(class_tiers) for class_tiers in classes_data.values())
            scraping_time = time.time() - start_time
            
            success = total_cards > 0 and len(errors) < len(self.class_names)
            
            self.logger.info(f"📊 HearthArena scraping complete:")
            self.logger.info(f"   Total cards: {total_cards}")
            self.logger.info(f"   Classes processed: {len(classes_data)}")
            self.logger.info(f"   Scraping time: {scraping_time:.1f}s")
            self.logger.info(f"   Errors: {len(errors)}")
            
            return HearthArenaTierResult(
                success=success,
                classes=classes_data,
                total_cards=total_cards,
                scraping_time=scraping_time,
                errors=errors,
                timestamp=datetime.now().isoformat(),
                source_url=self.base_url
            )
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error during HearthArena scraping: {e}")
            return HearthArenaTierResult(
                success=False,
                classes={},
                total_cards=0,
                scraping_time=time.time() - start_time,
                errors=[f"Fatal error: {str(e)}"],
                timestamp=datetime.now().isoformat(),
                source_url=self.base_url
            )
    
    def save_tier_data(self, tier_result: HearthArenaTierResult) -> bool:
        """
        Save tier data to cache.
        
        Args:
            tier_result: Tier data to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Create backup of existing data
            if self.tier_data_file.exists():
                backup_file = self.tier_data_file.with_suffix('.json.backup')
                self.tier_data_file.rename(backup_file)
            
            # Convert TierData objects to serializable format
            serializable_data = {
                'classes': {},
                'metadata': {
                    'total_cards': tier_result.total_cards,
                    'scraping_time': tier_result.scraping_time,
                    'timestamp': tier_result.timestamp,
                    'source_url': tier_result.source_url,
                    'success': tier_result.success,
                    'errors': tier_result.errors
                }
            }
            
            for class_name, class_tiers in tier_result.classes.items():
                serializable_data['classes'][class_name] = {}
                for card_name, tier_data in class_tiers.items():
                    serializable_data['classes'][class_name][card_name] = {
                        'tier': tier_data.tier,
                        'tier_index': tier_data.tier_index,
                        'confidence': tier_data.confidence
                    }
            
            # Save new data
            with open(self.tier_data_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            self.tier_data = tier_result.classes
            self.logger.info("✅ Saved tier data to cache")
            
            # Log update history
            self._log_update(tier_result)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save tier data: {e}")
            return False
    
    def update_tier_data(self, force: bool = False) -> bool:
        """
        Update tier data from HearthArena.
        
        Args:
            force: Force update even if cache is fresh
            
        Returns:
            True if update successful
        """
        if not force and not self.is_cache_stale():
            self.logger.info("Tier data cache is fresh, skipping update")
            return True
        
        self.logger.info("🚀 Starting HearthArena tier data update...")
        
        # Scrape new tier data
        result = self.scrape_heartharena_tierlist()
        
        if result.success:
            # Save to cache
            if self.save_tier_data(result):
                self.logger.info("🎯 HearthArena tier update completed successfully")
                return True
            else:
                self.logger.error("❌ Failed to save tier data")
                return False
        else:
            self.logger.error("❌ HearthArena scraping failed")
            return False
    
    def get_card_tier(self, card_name: str, class_name: str) -> Optional[TierData]:
        """
        Get tier information for a specific card.
        
        Args:
            card_name: Name of the card
            class_name: Hero class name
            
        Returns:
            TierData if found, None otherwise
        """
        if not self.tier_data or class_name not in self.tier_data:
            return None
        
        return self.tier_data[class_name].get(card_name)
    
    def get_class_tiers(self, class_name: str) -> Dict[str, TierData]:
        """
        Get all tier data for a specific class.
        
        Args:
            class_name: Hero class name
            
        Returns:
            Dictionary mapping card names to tier data
        """
        if not self.tier_data:
            return {}
        
        return self.tier_data.get(class_name, {})
    
    def get_tier_cards(self, class_name: str, tier: str) -> List[str]:
        """
        Get all cards in a specific tier for a class.
        
        Args:
            class_name: Hero class name
            tier: Tier name (e.g., 'great', 'good')
            
        Returns:
            List of card names in the tier
        """
        class_tiers = self.get_class_tiers(class_name)
        return [card_name for card_name, tier_data in class_tiers.items() 
                if tier_data.tier == tier]
    
    def get_tier_statistics(self) -> Dict[str, any]:
        """Get comprehensive tier statistics."""
        if not self.tier_data:
            return {'status': 'no_data'}
        
        stats = {
            'status': 'loaded',
            'cache_age_hours': self.get_cache_age_hours(),
            'needs_update': self.is_cache_stale(),
            'classes': len(self.tier_data),
            'total_cards': sum(len(class_tiers) for class_tiers in self.tier_data.values()),
            'tier_distribution': {},
            'class_counts': {}
        }
        
        # Calculate tier distribution
        tier_counts = {tier: 0 for tier in self.TIERS}
        for class_tiers in self.tier_data.values():
            for tier_data in class_tiers.values():
                tier_counts[tier_data.tier] += 1
        
        stats['tier_distribution'] = tier_counts
        
        # Calculate class counts
        for class_name, class_tiers in self.tier_data.items():
            stats['class_counts'][class_name] = len(class_tiers)
        
        return stats
    
    def _log_update(self, tier_result: HearthArenaTierResult):
        """Log update history for debugging and monitoring."""
        try:
            update_entry = {
                'timestamp': tier_result.timestamp,
                'success': tier_result.success,
                'total_cards': tier_result.total_cards,
                'scraping_time': tier_result.scraping_time,
                'errors': tier_result.errors,
                'classes_processed': len(tier_result.classes)
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
            self.logger.warning(f"Failed to log tier update: {e}")
    
    def has_data(self) -> bool:
        """Check if tier data is available."""
        return self.tier_data is not None and len(self.tier_data) > 0


# Global instance
_heartharena_tier_manager = None


def get_heartharena_tier_manager() -> HearthArenaTierManager:
    """
    Get the global HearthArena tier manager instance.
    
    Returns:
        HearthArenaTierManager instance
    """
    global _heartharena_tier_manager
    if _heartharena_tier_manager is None:
        _heartharena_tier_manager = HearthArenaTierManager()
    return _heartharena_tier_manager


if __name__ == "__main__":
    # Test the tier manager
    logging.basicConfig(level=logging.INFO)
    
    manager = get_heartharena_tier_manager()
    
    print("HearthArena Tier Manager Test")
    print("=" * 40)
    
    # Show current status
    stats = manager.get_tier_statistics()
    print(f"Status: {stats['status']}")
    
    if stats['status'] == 'loaded':
        print(f"Cache age: {stats['cache_age_hours']:.1f} hours")
        print(f"Total cards: {stats['total_cards']}")
        print(f"Classes: {stats['classes']}")
        print("\nTier distribution:")
        for tier, count in stats['tier_distribution'].items():
            print(f"  {tier}: {count} cards")
    
    # Check if update needed
    if stats.get('needs_update', True):
        print(f"\nUpdating tier data from HearthArena...")
        success = manager.update_tier_data(force=True)
        if success:
            print("✅ Update completed successfully!")
            
            # Show updated stats
            stats = manager.get_tier_statistics()
            print(f"New total: {stats['total_cards']} cards")
        else:
            print("❌ Update failed!")
    
    # Test tier lookup
    print(f"\nTesting tier lookup...")
    mage_cards = manager.get_class_tiers('mage')
    if mage_cards:
        print(f"Mage cards with tiers: {len(mage_cards)}")
        # Show first few as examples
        for i, (card_name, tier_data) in enumerate(mage_cards.items()):
            if i >= 5:
                break
            print(f"  {card_name}: {tier_data.tier}")