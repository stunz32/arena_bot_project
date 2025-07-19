"""
HearthArena Web Scraper (Enhanced with EzArena Method)

Fast, reliable scraping of HearthArena.com tier lists using BeautifulSoup.
Provides authoritative tier data and arena card information.
"""

import logging
import time
import requests
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from .heartharena_tier_manager import get_heartharena_tier_manager, HearthArenaTierResult
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent))
    from heartharena_tier_manager import get_heartharena_tier_manager, HearthArenaTierResult


@dataclass
class HearthArenaScrapingResult:
    """Container for HearthArena scraping results."""
    success: bool
    cards_by_class: Dict[str, List[str]]
    total_cards: int
    scraping_time: float
    errors: List[str]
    timestamp: str


class HearthArenaScraper:
    """
    Enhanced HearthArena scraper using EzArena's proven BeautifulSoup approach.
    
    Fast, reliable scraping of HearthArena tier lists with automatic
    tier data extraction and arena card identification.
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """
        Initialize HearthArena scraper.
        
        Args:
            headless: Unused (maintained for compatibility)
            timeout: Request timeout for HTTP operations
        """
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        
        # HearthArena configuration
        self.base_url = "https://www.heartharena.com/tierlist"
        self.class_names = [
            'mage', 'warrior', 'hunter', 'priest', 'warlock',
            'rogue', 'shaman', 'paladin', 'druid', 'demon-hunter'
        ]
        
        if not BS4_AVAILABLE:
            self.logger.error("Beautiful Soup not available. Install with: pip install beautifulsoup4")
            raise ImportError("Beautiful Soup is required for HearthArena scraping")
        
        # Get tier manager for enhanced functionality
        self.tier_manager = get_heartharena_tier_manager()
        
        self.logger.info("HearthArenaScraper initialized with EzArena method")
    
    def scrape_all_arena_cards(self) -> HearthArenaScrapingResult:
        """
        Scrape all arena cards using the enhanced tier manager.
        
        Returns:
            HearthArenaScrapingResult with tier-based arena card data
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting enhanced HearthArena scraping (EzArena method)...")
            
            # Update tier data using our tier manager
            tier_update_success = self.tier_manager.update_tier_data(force=False)
            
            if not tier_update_success:
                self.logger.warning("⚠️ Tier data update failed, using cached data")
            
            # Get tier statistics
            stats = self.tier_manager.get_tier_statistics()
            
            if stats['status'] != 'loaded':
                return HearthArenaScrapingResult(
                    success=False,
                    cards_by_class={},
                    total_cards=0,
                    scraping_time=time.time() - start_time,
                    errors=["No tier data available"],
                    timestamp=datetime.now().isoformat()
                )
            
            # Convert tier data to HearthArenaScrapingResult format
            cards_by_class = {}
            
            for class_name in self.class_names:
                class_tiers = self.tier_manager.get_class_tiers(class_name)
                # Extract just the card names (removing tier information for compatibility)
                cards_by_class[class_name] = list(class_tiers.keys())
            
            total_cards = sum(len(cards) for cards in cards_by_class.values())
            scraping_time = time.time() - start_time
            
            self.logger.info(f"✅ Enhanced scraping completed:")
            self.logger.info(f"   Total cards: {total_cards}")
            self.logger.info(f"   Classes: {len(cards_by_class)}")
            self.logger.info(f"   Time: {scraping_time:.1f}s")
            
            return HearthArenaScrapingResult(
                success=True,
                cards_by_class=cards_by_class,
                total_cards=total_cards,
                scraping_time=scraping_time,
                errors=[],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced scraping failed: {e}")
            return HearthArenaScrapingResult(
                success=False,
                cards_by_class={},
                total_cards=0,
                scraping_time=time.time() - start_time,
                errors=[f"Error: {str(e)}"],
                timestamp=datetime.now().isoformat()
            )
    
    def get_tier_data(self) -> Optional[HearthArenaTierResult]:
        """
        Get comprehensive tier data from HearthArena.
        
        Returns:
            HearthArenaTierResult with complete tier information
        """
        try:
            # Update tier data
            success = self.tier_manager.update_tier_data(force=False)
            
            if not success:
                self.logger.warning("Failed to update tier data")
                return None
            
            # Create tier result from manager data
            classes_data = {}
            for class_name in self.class_names:
                classes_data[class_name] = self.tier_manager.get_class_tiers(class_name)
            
            total_cards = sum(len(class_tiers) for class_tiers in classes_data.values())
            
            return HearthArenaTierResult(
                success=True,
                classes=classes_data,
                total_cards=total_cards,
                scraping_time=0.0,  # Already cached
                errors=[],
                timestamp=datetime.now().isoformat(),
                source_url=self.tier_manager.base_url
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get tier data: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test connection to HearthArena without full scraping.
        
        Returns:
            True if HearthArena is accessible
        """
        try:
            self.logger.info(f"🌐 Testing connection to {self.base_url}")
            response = requests.get(self.base_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check if we got valid HTML
            if "heartharena" in response.text.lower():
                self.logger.info("✅ Connection to HearthArena successful")
                return True
            else:
                self.logger.error("❌ Unexpected response from HearthArena")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources (compatibility method)."""
        # No cleanup needed for HTTP-based approach
        self.logger.info("✅ Cleanup completed (no resources to clean)")


# Global instance
_heartharena_scraper = None


def get_heartharena_scraper(headless: bool = True) -> HearthArenaScraper:
    """
    Get the global HearthArena scraper instance.
    
    Args:
        headless: Unused (maintained for compatibility)
        
    Returns:
        HearthArenaScraper instance
    """
    global _heartharena_scraper
    if _heartharena_scraper is None:
        _heartharena_scraper = HearthArenaScraper(headless=headless)
    return _heartharena_scraper


if __name__ == "__main__":
    # Test the enhanced scraper
    logging.basicConfig(level=logging.INFO)
    
    scraper = get_heartharena_scraper(headless=True)
    
    print("Enhanced HearthArena Scraper Test (EzArena Method)")
    print("=" * 60)
    
    # Test connection
    print("Testing connection to HearthArena...")
    if scraper.test_connection():
        print("✅ Connection successful!")
        
        print("\nTesting tier data retrieval...")
        tier_data = scraper.get_tier_data()
        
        if tier_data and tier_data.success:
            print(f"✅ Tier data retrieved successfully!")
            print(f"📊 Total cards with tiers: {tier_data.total_cards}")
            
            # Show sample tier data
            for class_name in ['mage', 'warrior']:
                class_tiers = tier_data.classes.get(class_name, {})
                if class_tiers:
                    print(f"\n{class_name.title()} tier sample:")
                    for i, (card_name, tier_data_obj) in enumerate(class_tiers.items()):
                        if i >= 3:  # Show first 3
                            break
                        print(f"  {card_name}: {tier_data_obj.tier}")
        else:
            print("❌ Failed to retrieve tier data!")
        
        print("\nTesting arena card scraping...")
        result = scraper.scrape_all_arena_cards()
        
        if result.success:
            print(f"✅ Scraping successful!")
            print(f"📊 Found {result.total_cards} total arena cards")
            print(f"⏱️ Scraping time: {result.scraping_time:.1f}s")
            for class_name, cards in result.cards_by_class.items():
                print(f"   {class_name}: {len(cards)} cards")
        else:
            print("❌ Scraping failed!")
            for error in result.errors:
                print(f"   Error: {error}")
    else:
        print("❌ Connection failed!")
    
    print("\n🎯 EzArena method integration complete!")
    print("Fast, reliable tier list scraping without Selenium complexity.")