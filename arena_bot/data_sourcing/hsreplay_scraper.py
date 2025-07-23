"""
HSReplay API Integration

Handles data acquisition from HSReplay API endpoints for both card and hero statistics.
Implements robust caching, error handling, and fallback strategies.
"""

import json
import logging
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import the ID mapping functionality
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.cards_json_loader import get_cards_json_loader


@dataclass
class HSReplayConfig:
    """Configuration for HSReplay API access."""
    CARD_STATS_URL = "https://hsreplay.net/api/v1/analytics/card_stats/"
    HERO_STATS_URL = "https://hsreplay.net/api/v1/analytics/hero_stats/"
    CARD_PARAMS = {"gameType": "ARENA", "timeRange": "CURRENT_PATCH", "rankRange": "ALL"}
    CARD_CACHE_HOURS = 24
    HERO_CACHE_HOURS = 12
    REQUEST_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }


class HSReplayScraper:
    """
    HSReplay API client with comprehensive error handling and caching.
    
    Manages separate data streams for card statistics and hero performance.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize HSReplay scraper with caching and session management."""
        self.logger = logging.getLogger(__name__)
        
        # Cache directory setup
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.card_cache_file = self.cache_dir / "hsreplay_data.json"
        self.hero_cache_file = self.cache_dir / "hsreplay_hero_data.json"
        
        # Session management
        self.session = requests.Session()
        self.session.headers.update(HSReplayConfig.HEADERS)
        
        # Cards JSON loader for ID translation
        self.cards_loader = get_cards_json_loader()
        
        # Performance tracking
        self.last_card_fetch = None
        self.last_hero_fetch = None
        self.api_call_count = 0
        
        # Status tracking for graceful UI integration
        self.card_data_status = 'unknown'  # 'online', 'cached', 'offline', 'error'
        self.hero_data_status = 'unknown'
        self.last_error_message = None
        self.card_cache_age_hours = 0
        self.hero_cache_age_hours = 0
        
        self.logger.info("HSReplayScraper initialized with caching enabled")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current HSReplay data source status for UI display.
        
        Returns:
            Status dictionary with information for both card and hero data
        """
        # Calculate cache ages
        self._update_cache_ages()
        
        # Determine overall status
        if self.card_data_status == 'online' and self.hero_data_status == 'online':
            overall_status = 'online'
        elif self.card_data_status in ['cached', 'online'] and self.hero_data_status in ['cached', 'online']:
            overall_status = 'cached'
        elif self.card_data_status == 'error' or self.hero_data_status == 'error':
            overall_status = 'error'
        else:
            overall_status = 'offline'
        
        return {
            'status': overall_status,
            'card_status': self.card_data_status,
            'hero_status': self.hero_data_status,
            'card_cache_age_hours': self.card_cache_age_hours,
            'hero_cache_age_hours': self.hero_cache_age_hours,
            'last_error': self.last_error_message,
            'api_calls_today': self.api_call_count
        }
    
    def _update_cache_ages(self):
        """Update cache age information in hours."""
        # Card cache age
        if self.card_cache_file.exists():
            card_age = time.time() - self.card_cache_file.stat().st_mtime
            self.card_cache_age_hours = card_age / 3600
        else:
            self.card_cache_age_hours = 999  # No cache
            
        # Hero cache age  
        if self.hero_cache_file.exists():
            hero_age = time.time() - self.hero_cache_file.stat().st_mtime
            self.hero_cache_age_hours = hero_age / 3600
        else:
            self.hero_cache_age_hours = 999  # No cache
    
    def get_underground_arena_stats(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get Underground Arena card statistics with ID translation.
        
        Args:
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            Dictionary mapping card_id to stats: {card_id: {"win_rate": float, "play_rate": float, ...}}
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if not force_refresh:
                cached_data = self._load_card_cache()
                if cached_data:
                    self.card_data_status = 'cached'
                    self.logger.info(f"Using cached card data ({len(cached_data)} cards)")
                    return cached_data
            
            # Fetch fresh data from API
            self.logger.info("Fetching fresh card data from HSReplay API...")
            raw_data = self._fetch_card_stats_from_api()
            
            if not raw_data:
                self.logger.warning("No card data received from API, using cached fallback")
                cached_fallback = self._load_card_cache()
                if cached_fallback:
                    self.card_data_status = 'cached'
                    return cached_fallback
                else:
                    self.card_data_status = 'offline'
                    return {}
            
            # Translate dbf_ids to card_ids
            translated_data = self._translate_card_data(raw_data)
            
            # Cache the translated data
            self._save_card_cache(translated_data)
            
            # Success - API data retrieved
            self.card_data_status = 'online'
            self.last_error_message = None
            
            fetch_time = time.time() - start_time
            self.logger.info(f"Card data fetched and cached: {len(translated_data)} cards in {fetch_time:.2f}s")
            
            return translated_data
            
        except Exception as e:
            error_msg = f"Error fetching card stats: {e}"
            self.logger.error(error_msg)
            self.last_error_message = str(e)
            
            # Fallback to cache
            cached_fallback = self._load_card_cache()
            if cached_fallback:
                self.card_data_status = 'cached'
                self.logger.warning("Using stale cached data as fallback")
                return cached_fallback
            else:
                self.card_data_status = 'error'
                return {}
    
    def get_hero_winrates(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Get hero class winrates for Underground Arena.
        
        Args:
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            Dictionary mapping class names to winrates: {"WARRIOR": 55.07, "MAGE": 52.34, ...}
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if not force_refresh:
                cached_data = self._load_hero_cache()
                if cached_data:
                    self.hero_data_status = 'cached'
                    self.logger.info(f"Using cached hero data ({len(cached_data)} classes)")
                    return cached_data
            
            # Fetch fresh data from API
            self.logger.info("Fetching fresh hero data from HSReplay API...")
            raw_data = self._fetch_hero_stats_from_api()
            
            if not raw_data:
                self.logger.warning("No hero data received from API, using cached fallback")
                cached_fallback = self._load_hero_cache()
                if cached_fallback:
                    self.hero_data_status = 'cached'
                    return cached_fallback
                else:
                    self.hero_data_status = 'offline'
                    return {}
            
            # Extract winrates by class
            hero_winrates = self._extract_hero_winrates(raw_data)
            
            # Cache the data
            self._save_hero_cache(hero_winrates)
            
            # Success - API data retrieved
            self.hero_data_status = 'online'
            self.last_error_message = None
            
            fetch_time = time.time() - start_time
            self.logger.info(f"Hero data fetched and cached: {len(hero_winrates)} classes in {fetch_time:.2f}s")
            
            return hero_winrates
            
        except Exception as e:
            error_msg = f"Error fetching hero stats: {e}"
            self.logger.error(error_msg)
            self.last_error_message = str(e)
            
            # Fallback to cache
            cached_fallback = self._load_hero_cache()
            if cached_fallback:
                self.hero_data_status = 'cached'
                self.logger.warning("Using stale cached hero data as fallback")
                return cached_fallback
            else:
                self.hero_data_status = 'error'
                return {}
    
    def _fetch_card_stats_from_api(self) -> Optional[List[Dict]]:
        """Fetch raw card statistics from HSReplay API."""
        try:
            self.api_call_count += 1
            self.last_card_fetch = datetime.now()
            
            response = self._make_api_request(
                HSReplayConfig.CARD_STATS_URL,
                params=HSReplayConfig.CARD_PARAMS
            )
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if isinstance(data, list) and len(data) > 0:
                    self.logger.info(f"Successfully fetched {len(data)} card records from API")
                    return data
                else:
                    self.logger.warning("API returned empty or invalid card data structure")
                    return None
            else:
                status = response.status_code if response else "No response"
                self.logger.error(f"Card API request failed with status: {status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception during card API fetch: {e}")
            return None
    
    def _fetch_hero_stats_from_api(self) -> Optional[Dict]:
        """Fetch raw hero statistics from HSReplay API."""
        try:
            self.api_call_count += 1
            self.last_hero_fetch = datetime.now()
            
            response = self._make_api_request(HSReplayConfig.HERO_STATS_URL)
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Look for BGT_ARENA section (Underground Arena mode)
                if isinstance(data, dict) and "BGT_ARENA" in data:
                    bgt_data = data["BGT_ARENA"]
                    self.logger.info(f"Successfully fetched hero data from BGT_ARENA section")
                    return bgt_data
                else:
                    self.logger.warning("API response missing BGT_ARENA section")
                    return None
            else:
                status = response.status_code if response else "No response"
                self.logger.error(f"Hero API request failed with status: {status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception during hero API fetch: {e}")
            return None
    
    def _make_api_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make API request with retry logic and error handling."""
        for attempt in range(HSReplayConfig.RETRY_ATTEMPTS):
            try:
                self.logger.debug(f"API request attempt {attempt + 1}/{HSReplayConfig.RETRY_ATTEMPTS}: {url}")
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=HSReplayConfig.REQUEST_TIMEOUT
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', HSReplayConfig.RETRY_DELAY))
                    self.logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                    time.sleep(retry_after)
                    continue
                
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                    time.sleep(HSReplayConfig.RETRY_DELAY)
                    continue
                else:
                    self.logger.error("All API request attempts failed")
                    return None
        
        return None
    
    def _translate_card_data(self, raw_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Translate HSReplay dbf_ids to card_ids using CardsJsonLoader."""
        translated_data = {}
        translation_failures = 0
        
        for card_entry in raw_data:
            dbf_id = card_entry.get("dbf_id")
            if not dbf_id:
                continue
            
            # Translate dbf_id to card_id
            card_id = self.cards_loader.get_card_id_from_dbf_id(dbf_id)
            if card_id:
                # Extract relevant statistics
                stats = {
                    "win_rate": card_entry.get("win_rate", 0.0),
                    "play_rate": card_entry.get("play_rate", 0.0),
                    "deck_win_rate": card_entry.get("deck_win_rate", 0.0),
                    "times_played": card_entry.get("times_played", 0),
                    "dbf_id": dbf_id  # Keep for reference
                }
                translated_data[card_id] = stats
            else:
                translation_failures += 1
        
        if translation_failures > 0:
            self.logger.warning(f"Failed to translate {translation_failures} cards (unknown dbf_ids)")
        
        success_rate = len(translated_data) / len(raw_data) * 100 if raw_data else 0
        self.logger.info(f"Card translation: {len(translated_data)}/{len(raw_data)} ({success_rate:.1f}% success)")
        
        return translated_data
    
    def _extract_hero_winrates(self, hero_data: Dict) -> Dict[str, float]:
        """Extract hero class winrates from BGT_ARENA data."""
        winrates = {}
        
        try:
            # The exact structure may vary, but typically contains class performance data
            if "class_performance" in hero_data:
                class_data = hero_data["class_performance"]
                
                for class_name, stats in class_data.items():
                    if isinstance(stats, dict) and "win_rate" in stats:
                        winrates[class_name.upper()] = float(stats["win_rate"])
            
            # Alternative structure: direct class mapping
            elif isinstance(hero_data, dict):
                for key, value in hero_data.items():
                    if isinstance(value, dict) and "win_rate" in value:
                        # Try to extract class name from key
                        class_name = key.upper()
                        if class_name in ["WARRIOR", "MAGE", "HUNTER", "PRIEST", "WARLOCK", 
                                        "ROGUE", "SHAMAN", "PALADIN", "DRUID", "DEMONHUNTER"]:
                            winrates[class_name] = float(value["win_rate"])
            
            self.logger.info(f"Extracted hero winrates for {len(winrates)} classes")
            
        except Exception as e:
            self.logger.error(f"Error extracting hero winrates: {e}")
        
        return winrates
    
    def _load_card_cache(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load card data from cache if valid."""
        try:
            if not self.card_cache_file.exists():
                return None
            
            # Check cache age
            file_age = datetime.now() - datetime.fromtimestamp(self.card_cache_file.stat().st_mtime)
            if file_age > timedelta(hours=HSReplayConfig.CARD_CACHE_HOURS):
                self.logger.info("Card cache expired")
                return None
            
            with open(self.card_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and len(data) > 0:
                return data
            
        except Exception as e:
            self.logger.warning(f"Failed to load card cache: {e}")
        
        return None
    
    def _save_card_cache(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Save card data to cache."""
        try:
            with open(self.card_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Card cache saved with {len(data)} entries")
        except Exception as e:
            self.logger.error(f"Failed to save card cache: {e}")
    
    def _load_hero_cache(self) -> Optional[Dict[str, float]]:
        """Load hero data from cache if valid."""
        try:
            if not self.hero_cache_file.exists():
                return None
            
            # Check cache age
            file_age = datetime.now() - datetime.fromtimestamp(self.hero_cache_file.stat().st_mtime)
            if file_age > timedelta(hours=HSReplayConfig.HERO_CACHE_HOURS):
                self.logger.info("Hero cache expired")
                return None
            
            with open(self.hero_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and len(data) > 0:
                return data
            
        except Exception as e:
            self.logger.warning(f"Failed to load hero cache: {e}")
        
        return None
    
    def _save_hero_cache(self, data: Dict[str, float]) -> None:
        """Save hero data to cache."""
        try:
            with open(self.hero_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Hero cache saved with {len(data)} entries")
        except Exception as e:
            self.logger.error(f"Failed to save hero cache: {e}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status and health information."""
        return {
            "session_active": bool(self.session),
            "api_calls_made": self.api_call_count,
            "last_card_fetch": self.last_card_fetch.isoformat() if self.last_card_fetch else None,
            "last_hero_fetch": self.last_hero_fetch.isoformat() if self.last_hero_fetch else None,
            "card_cache_exists": self.card_cache_file.exists(),
            "hero_cache_exists": self.hero_cache_file.exists(),
            "card_cache_age_hours": self._get_cache_age_hours(self.card_cache_file),
            "hero_cache_age_hours": self._get_cache_age_hours(self.hero_cache_file),
            "cache_directory": str(self.cache_dir)
        }
    
    def _get_cache_age_hours(self, cache_file: Path) -> Optional[float]:
        """Get cache file age in hours."""
        try:
            if cache_file.exists():
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                return file_age.total_seconds() / 3600
        except Exception:
            pass
        return None
    
    def clear_cache(self, card_cache: bool = True, hero_cache: bool = True) -> None:
        """Clear cached data files."""
        try:
            if card_cache and self.card_cache_file.exists():
                self.card_cache_file.unlink()
                self.logger.info("Card cache cleared")
            
            if hero_cache and self.hero_cache_file.exists():
                self.hero_cache_file.unlink()
                self.logger.info("Hero cache cleared")
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def __del__(self):
        """Cleanup session on destruction."""
        if hasattr(self, 'session') and self.session:
            self.session.close()


# Global instance for easy access
_hsreplay_scraper = None

def get_hsreplay_scraper() -> HSReplayScraper:
    """Get the global HSReplay scraper instance."""
    global _hsreplay_scraper
    if _hsreplay_scraper is None:
        _hsreplay_scraper = HSReplayScraper()
    return _hsreplay_scraper