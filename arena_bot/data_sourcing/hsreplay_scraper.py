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

# Import Enhanced Observability System
from ai_v2.logging_utils import (
    get_structured_logger, RequestContext, 
    log_performance, time_operation, log_complex_error,
    LogCategory
)


@dataclass
class HSReplayConfig:
    """Configuration for HSReplay API access."""
    # Updated URLs based on current HSReplay API structure
    CARD_STATS_URL = "https://hsreplay.net/analytics/query/card_list_free/?GameType=UNDERGROUND_ARENA&TimeRange=CURRENT_PATCH&LeagueRankRange=BRONZE_THROUGH_GOLD"
    HERO_STATS_URL = "https://hsreplay.net/analytics/query/player_class_performance_summary_v2/"
    # CARD_PARAMS removed - parameters now embedded in URL
    CARD_CACHE_HOURS = 24
    HERO_CACHE_HOURS = 12
    # Production-grade timeout: (connect_timeout, read_timeout)
    REQUEST_TIMEOUT = (5, 30)  # 5s connect, 30s read
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2
    # Enhanced browser-like headers for reliability
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://hsreplay.net/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    }


class HSReplayScraper:
    """
    HSReplay API client with comprehensive error handling and caching.
    
    Manages separate data streams for card statistics and hero performance.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize HSReplay scraper with caching and session management."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize Enhanced Observability System  
        self.structured_logger = get_structured_logger("HSReplayScraper")
        self.structured_logger.info(
            "HSReplay scraper initialized",
            category=LogCategory.INITIALIZATION,
            cache_dir=str(cache_dir) if cache_dir else "default"
        )
        
        # Cache directory setup
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "assets" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.card_cache_file = self.cache_dir / "hsreplay_data.json"
        self.hero_cache_file = self.cache_dir / "hsreplay_hero_data.json"
        
        # Production-grade session management with retry strategy
        self.session = self._create_robust_session()
        
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
    
    def _create_robust_session(self) -> requests.Session:
        """Create production-grade session with retry strategy and connection pooling."""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        
        # Set enhanced headers
        session.headers.update(HSReplayConfig.HEADERS)
        
        # Configure retry strategy for resilient API calls
        retry_strategy = Retry(
            total=3,  # Total number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            backoff_factor=0.3,  # Backoff factor for exponential delay
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Only retry safe methods
            respect_retry_after_header=True  # Honor server's Retry-After header
        )
        
        # Create HTTP adapter with retry configuration and connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=20,      # Maximum connections in each pool
            pool_block=False      # Don't block when pool is full
        )
        
        # Mount adapters for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        self.logger.debug("Created robust session with retry strategy and connection pooling")
        return session
    
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
            
            if not raw_data:  # Empty list means API failed
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
            
            if not raw_data:  # Empty dict means API failed
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
    
    def _fetch_card_stats_from_api(self) -> List[Dict]:
        """Fetch raw card statistics from HSReplay API. Returns empty list on failure."""
        try:
            self.api_call_count += 1
            self.last_card_fetch = datetime.now()
            
            # New URL has parameters embedded, no separate params needed
            response = self._make_api_request(HSReplayConfig.CARD_STATS_URL)
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Handle HSReplay API response structure: {'series': {'data': {'ALL': [...]}}}
                    if isinstance(data, dict) and 'series' in data:
                        series_data = data['series']
                        if isinstance(series_data, dict) and 'data' in series_data:
                            # HSReplay returns data organized by class, use 'ALL' for all cards
                            class_data = series_data['data']
                            if isinstance(class_data, dict) and 'ALL' in class_data:
                                card_data = class_data['ALL']
                                if isinstance(card_data, list) and len(card_data) > 0:
                                    self.logger.info(f"Successfully fetched {len(card_data)} card records from HSReplay")
                                    return card_data
                            
                            # Fallback: try to use any available class data
                            for class_name, class_card_data in class_data.items():
                                if isinstance(class_card_data, list) and len(class_card_data) > 0:
                                    self.logger.info(f"Using {class_name} class data: {len(class_card_data)} cards")
                                    return class_card_data
                    
                    # Fallback for legacy or other structures
                    elif isinstance(data, list) and len(data) > 0:
                        self.logger.info(f"Successfully fetched {len(data)} card records from API (direct list)")
                        return data
                    
                    self.logger.warning("API returned empty or unrecognized card data structure")
                    self.logger.debug(f"Response structure keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    return []
                        
                except (ValueError, TypeError) as json_error:
                    self.logger.error(f"Failed to parse API response as JSON: {json_error}")
                    return []  # Return empty list instead of None
                    
            else:
                status = response.status_code if response else "No response"
                self.logger.warning(f"Card API request failed with HTTP {status}, proceeding with empty data")
                return []  # Return empty list instead of None
                
        except Exception as e:
            self.logger.warning(f"Exception during card API fetch: {e}, proceeding with empty data")
            return []  # Return empty list instead of None
    
    def _fetch_hero_stats_from_api(self) -> Dict:
        """Fetch raw hero statistics from HSReplay API. Returns empty dict on failure."""
        try:
            self.api_call_count += 1
            self.last_hero_fetch = datetime.now()
            
            response = self._make_api_request(HSReplayConfig.HERO_STATS_URL)
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Handle HSReplay hero API structure: {'series': {'data': {'BGT_ARENA': {...}}}}
                    if isinstance(data, dict) and 'series' in data:
                        series_data = data['series']
                        if isinstance(series_data, dict) and 'data' in series_data:
                            # HSReplay returns data by game type, use BGT_ARENA for Underground Arena
                            game_data = series_data['data']
                            if isinstance(game_data, dict) and 'BGT_ARENA' in game_data:
                                bgt_data = game_data['BGT_ARENA']
                                self.logger.info(f"Successfully fetched hero data from BGT_ARENA section")
                                return bgt_data
                            
                            # Fallback: try any available game mode data
                            for game_mode, mode_data in game_data.items():
                                if isinstance(mode_data, dict) and len(mode_data) > 0:
                                    self.logger.info(f"Using {game_mode} hero data as fallback")
                                    return mode_data
                    
                    # Legacy direct BGT_ARENA structure
                    elif isinstance(data, dict) and "BGT_ARENA" in data:
                        bgt_data = data["BGT_ARENA"]
                        self.logger.info(f"Successfully fetched hero data from direct BGT_ARENA section")
                        return bgt_data
                    
                    self.logger.warning("API response has unrecognized hero data structure")
                    self.logger.debug(f"Response structure keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    return {}
                        
                except (ValueError, TypeError) as json_error:
                    self.logger.error(f"Failed to parse hero API response as JSON: {json_error}")
                    return {}  # Return empty dict instead of None
                    
            else:
                status = response.status_code if response else "No response"
                self.logger.warning(f"Hero API request failed with HTTP {status}, proceeding with empty data")
                return {}  # Return empty dict instead of None
                
        except Exception as e:
            self.logger.warning(f"Exception during hero API fetch: {e}, proceeding with empty data")
            return {}  # Return empty dict instead of None
    
    @log_performance(threshold_ms=5000.0)  # 5 second threshold for API calls
    def _make_api_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make production-grade API request with comprehensive error handling and debugging."""
        
        # Create request context for traceability
        with RequestContext(operation_name="hsreplay_api_request") as ctx:
            self.structured_logger.info(
                "Starting HSReplay API request",
                category=LogCategory.API_CALL,
                url=url,
                params=params,
                max_attempts=HSReplayConfig.RETRY_ATTEMPTS
            )
            
            for attempt in range(HSReplayConfig.RETRY_ATTEMPTS):
                try:
                    # Enhanced debug logging
                    self.logger.info(
                        f"API request attempt {attempt + 1}/{HSReplayConfig.RETRY_ATTEMPTS}: {url}"
                        f"{f' with params: {params}' if params else ''}"
                    )
                    
                    self.structured_logger.debug(
                        f"API request attempt {attempt + 1}",
                        category=LogCategory.API_CALL,
                        attempt_number=attempt + 1,
                        url=url,
                        params=params
                    )
                
                    # Production-grade request with comprehensive configuration
                    with time_operation(f"http_request_attempt_{attempt+1}", self.structured_logger):
                        response = self.session.get(
                            url,
                            params=params,
                            timeout=HSReplayConfig.REQUEST_TIMEOUT,  # Now tuple (connect, read)
                            verify=True,  # Explicitly enable SSL verification
                            allow_redirects=True,
                            stream=False
                        )
                        
                        # Debug response information
                        self.logger.debug(
                            f"Response received: status={response.status_code}, "
                            f"size={len(response.content)} bytes, "
                            f"time={response.elapsed.total_seconds():.2f}s, "
                            f"url={response.url}"
                        )
                        
                        # Structured response logging
                        self.structured_logger.info(
                            f"HTTP response received for attempt {attempt + 1}",
                            category=LogCategory.API_CALL,
                            status_code=response.status_code,
                            response_size_bytes=len(response.content),
                            response_time_ms=response.elapsed.total_seconds() * 1000,
                            final_url=str(response.url),
                            attempt_number=attempt + 1
                        )
                    
                        # Check for rate limiting (429)
                        if response.status_code == 429:
                            retry_after = int(response.headers.get('Retry-After', HSReplayConfig.RETRY_DELAY * 2))
                            self.logger.warning(f"Rate limited (429), waiting {retry_after}s before retry")
                            
                            self.structured_logger.warning(
                                "API rate limited - implementing backoff",
                                category=LogCategory.API_CALL,
                                status_code=429,
                                retry_after_seconds=retry_after,
                                attempt_number=attempt + 1,
                                backoff_strategy="server_directed"
                            )
                            
                            time.sleep(retry_after)
                            continue
                        
                        # Check for server errors (5xx) that warrant retries
                        if response.status_code in [500, 502, 503, 504]:
                            if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                                delay = HSReplayConfig.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                                self.logger.warning(
                                    f"Server error {response.status_code}, retrying in {delay}s (attempt {attempt + 1})"
                                )
                                
                                self.structured_logger.warning(
                                    "Server error encountered - retrying with exponential backoff",
                                    category=LogCategory.API_CALL,
                                    status_code=response.status_code,
                                    delay_seconds=delay,
                                    attempt_number=attempt + 1,
                                    backoff_strategy="exponential"
                                )
                                
                                time.sleep(delay)
                                continue
                            else:
                                self.logger.error(f"Server error {response.status_code} persisted after all retries")
                                
                                self.structured_logger.error(
                                    "Server error persisted after all retry attempts",
                                    category=LogCategory.API_CALL,
                                    status_code=response.status_code,
                                    total_attempts=HSReplayConfig.RETRY_ATTEMPTS,
                                    final_attempt=True
                                )
                                
                                return None
                        
                        # Success case - return response for all other status codes
                        self.structured_logger.info(
                            "API request completed successfully",
                            category=LogCategory.API_CALL,
                            status_code=response.status_code,
                            attempts_used=attempt + 1,
                            success=True
                        )
                        
                        return response
                    
                except requests.exceptions.ConnectTimeout as e:
                    self.logger.warning(
                        f"Connect timeout on attempt {attempt + 1} "
                        f"(timeout={HSReplayConfig.REQUEST_TIMEOUT[0]}s): {e}"
                    )
                    
                    log_complex_error(
                        self.structured_logger,
                        "HTTP connect timeout during API request",
                        e,
                        operation_data={
                            "url": url,
                            "attempt": attempt + 1,
                            "timeout_seconds": HSReplayConfig.REQUEST_TIMEOUT[0]
                        }
                    )
                    
                    if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                        time.sleep(HSReplayConfig.RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        self.logger.error("Connect timeout persisted after all retries - server unreachable")
                        
                        self.structured_logger.error(
                            "Connect timeout persisted after all retries",
                            category=LogCategory.API_CALL,
                            error_type="connect_timeout",
                            total_attempts=HSReplayConfig.RETRY_ATTEMPTS,
                            url=url
                        )
                        
                        return None
                        
                except requests.exceptions.ReadTimeout as e:
                    self.logger.warning(
                        f"Read timeout on attempt {attempt + 1} "
                        f"(timeout={HSReplayConfig.REQUEST_TIMEOUT[1]}s): {e}"
                    )
                    
                    log_complex_error(
                        self.structured_logger,
                        "HTTP read timeout during API request",
                        e,
                        operation_data={
                            "url": url,
                            "attempt": attempt + 1,
                            "timeout_seconds": HSReplayConfig.REQUEST_TIMEOUT[1]
                        }
                    )
                    
                    if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                        time.sleep(HSReplayConfig.RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        self.logger.error("Read timeout persisted after all retries - server too slow")
                        
                        self.structured_logger.error(
                            "Read timeout persisted after all retries",
                            category=LogCategory.API_CALL,
                            error_type="read_timeout",
                            total_attempts=HSReplayConfig.RETRY_ATTEMPTS,
                            url=url
                        )
                        
                        return None
                    
                except requests.exceptions.ConnectionError as e:
                    self.logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                    
                    log_complex_error(
                        self.structured_logger,
                        "HTTP connection error during API request",
                        e,
                        operation_data={
                            "url": url,
                            "attempt": attempt + 1
                        }
                    )
                    
                    if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                        time.sleep(HSReplayConfig.RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        self.logger.error("Connection error persisted after all retries - check network")
                        
                        self.structured_logger.error(
                            "Connection error persisted after all retries",
                            category=LogCategory.API_CALL,
                            error_type="connection_error",
                            total_attempts=HSReplayConfig.RETRY_ATTEMPTS,
                            url=url
                        )
                        
                        return None
                        
                except requests.exceptions.SSLError as e:
                    self.logger.error(f"SSL error on attempt {attempt + 1}: {e}")
                    
                    log_complex_error(
                        self.structured_logger,
                        "SSL error during API request",
                        e,
                        operation_data={
                            "url": url,
                            "attempt": attempt + 1,
                            "ssl_verification": True
                        }
                    )
                    
                    # SSL errors usually don't warrant retries
                    return None
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Request exception on attempt {attempt + 1}: {e}")
                    if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                        time.sleep(HSReplayConfig.RETRY_DELAY)
                        continue
                    else:
                        self.logger.error(f"Request exception persisted after all retries: {e}")
                        return None
                            
                except Exception as e:
                    self.logger.error(f"Unexpected error in API request attempt {attempt + 1}: {e}")
                    if attempt < HSReplayConfig.RETRY_ATTEMPTS - 1:
                        time.sleep(HSReplayConfig.RETRY_DELAY)
                        continue
                    else:
                        self.logger.error(f"Unexpected error persisted after all retries: {e}")
                        return None
        
        # Should never reach here, but defensive programming
        self.logger.error("API request failed - exhausted all retry attempts")
        return None
    
    def _translate_card_data(self, raw_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Translate HSReplay card data to standardized format using CardsJsonLoader."""
        translated_data = {}
        translation_failures = 0
        
        for card_entry in raw_data:
            # Try multiple possible ID field names
            dbf_id = None
            card_id_direct = None
            
            # Check for various ID formats in new API
            if "dbf_id" in card_entry:
                dbf_id = card_entry["dbf_id"]
            elif "dbfId" in card_entry:
                dbf_id = card_entry["dbfId"]
            elif "card_id" in card_entry:
                card_id_direct = card_entry["card_id"]
            elif "cardId" in card_entry:
                card_id_direct = card_entry["cardId"]
            elif "id" in card_entry:
                # Could be either dbf_id or card_id
                id_value = card_entry["id"]
                if isinstance(id_value, int):
                    dbf_id = id_value
                else:
                    card_id_direct = id_value
            
            # Skip if no ID found
            if not dbf_id and not card_id_direct:
                continue
            
            # Translate dbf_id to card_id if needed
            if dbf_id and not card_id_direct:
                card_id_direct = self.cards_loader.get_card_id_from_dbf_id(dbf_id)
            
            if card_id_direct:
                # Extract relevant statistics with HSReplay field names
                stats = {
                    "win_rate": self._extract_stat(card_entry, ["included_winrate", "winrate_when_played", "win_rate", "winRate"]) / 100.0,  # Convert percentage
                    "play_rate": self._extract_stat(card_entry, ["included_popularity", "play_rate", "playRate", "popularity"]) / 100.0,  # Convert percentage
                    "deck_win_rate": self._extract_stat(card_entry, ["included_winrate", "deck_win_rate", "deckWinRate"]) / 100.0,  # Convert percentage
                    "times_played": self._extract_stat(card_entry, ["times_played", "timesPlayed", "games", "count"], default=0),
                    "dbf_id": dbf_id  # Keep for reference if available
                }
                translated_data[card_id_direct] = stats
            else:
                translation_failures += 1
        
        if translation_failures > 0:
            self.logger.warning(f"Failed to translate {translation_failures} cards (unknown IDs)")
        
        success_rate = len(translated_data) / len(raw_data) * 100 if raw_data else 0
        self.logger.info(f"Card translation: {len(translated_data)}/{len(raw_data)} ({success_rate:.1f}% success)")
        
        return translated_data
    
    def _extract_stat(self, data: Dict, field_names: List[str], default: float = 0.0) -> float:
        """Extract a statistic from data dict using multiple possible field names."""
        for field_name in field_names:
            if field_name in data:
                value = data[field_name]
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
        return default
    
    def _extract_hero_winrates(self, hero_data: Dict) -> Dict[str, float]:
        """Extract hero class winrates from player class performance data."""
        winrates = {}
        valid_classes = {"WARRIOR", "MAGE", "HUNTER", "PRIEST", "WARLOCK", 
                        "ROGUE", "SHAMAN", "PALADIN", "DRUID", "DEMONHUNTER"}
        
        try:
            # Handle multiple possible data structures
            
            # Structure 1: New API with class performance summary
            if "class_performance" in hero_data:
                class_data = hero_data["class_performance"]
                for class_name, stats in class_data.items():
                    if isinstance(stats, dict):
                        win_rate = self._extract_stat(stats, ["win_rate", "winRate", "winrate", "wr"])
                        if win_rate > 0:
                            winrates[class_name.upper()] = win_rate
            
            # Structure 2: Direct class mapping with stats
            elif isinstance(hero_data, dict):
                for key, value in hero_data.items():
                    if isinstance(value, dict):
                        # Check if key is a valid class name or contains one
                        class_name = self._extract_class_name(key)
                        if class_name and class_name in valid_classes:
                            win_rate = self._extract_stat(value, ["win_rate", "winRate", "winrate", "wr", "win_rate_percent"])
                            if win_rate > 0:
                                # Convert percentage to decimal if needed
                                if win_rate > 1:
                                    win_rate = win_rate / 100.0
                                winrates[class_name] = win_rate
            
            # Structure 3: List format with class identifiers
            if not winrates and isinstance(hero_data, list):
                for entry in hero_data:
                    if isinstance(entry, dict):
                        class_name = self._extract_class_name(entry.get("class_name", entry.get("className", entry.get("hero", ""))))
                        if class_name and class_name in valid_classes:
                            win_rate = self._extract_stat(entry, ["win_rate", "winRate", "winrate", "wr"])
                            if win_rate > 0:
                                if win_rate > 1:
                                    win_rate = win_rate / 100.0
                                winrates[class_name] = win_rate
            
            # Log successful extraction
            if winrates:
                self.logger.info(f"Extracted hero winrates for {len(winrates)} classes: {list(winrates.keys())}")
            else:
                self.logger.warning("No hero winrates extracted from API response")
                self.logger.debug(f"Hero data structure: {list(hero_data.keys()) if isinstance(hero_data, dict) else type(hero_data)}")
            
        except Exception as e:
            self.logger.error(f"Error extracting hero winrates: {e}")
        
        return winrates
    
    def _extract_class_name(self, name_input: str) -> Optional[str]:
        """Extract standardized class name from various input formats."""
        if not name_input or not isinstance(name_input, str):
            return None
        
        # Normalize and check for valid class names
        name_upper = name_input.upper().strip()
        valid_classes = {"WARRIOR", "MAGE", "HUNTER", "PRIEST", "WARLOCK", 
                        "ROGUE", "SHAMAN", "PALADIN", "DRUID", "DEMONHUNTER"}
        
        # Direct match
        if name_upper in valid_classes:
            return name_upper
        
        # Handle variations and partial matches
        class_mappings = {
            "DH": "DEMONHUNTER",
            "DEMON_HUNTER": "DEMONHUNTER",
            "DEMON HUNTER": "DEMONHUNTER",
        }
        
        if name_upper in class_mappings:
            return class_mappings[name_upper]
        
        # Check if any valid class name is contained in the input
        for class_name in valid_classes:
            if class_name in name_upper:
                return class_name
        
        return None
    
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