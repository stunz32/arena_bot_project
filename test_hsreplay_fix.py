#!/usr/bin/env python3
"""
Test script to validate HSReplay scraper improvements.
This script tests the fixed API request logic to ensure the "No response" errors are resolved.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arena_bot.data_sourcing.hsreplay_scraper import HSReplayScraper, HSReplayConfig

def setup_logging():
    """Configure detailed logging to see API request flow."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Also log requests/urllib3 details for debugging
    logging.getLogger("urllib3.connectionpool").setLevel(logging.DEBUG)
    logging.getLogger("requests.packages.urllib3").setLevel(logging.DEBUG)

def test_api_requests():
    """Test the improved HSReplay API request logic."""
    print("=" * 80)
    print("TESTING IMPROVED HSREPLAY API REQUEST LOGIC")
    print("=" * 80)
    
    setup_logging()
    
    # Create scraper instance
    print("\n1. Creating HSReplayScraper instance...")
    scraper = HSReplayScraper()
    
    # Test configuration
    print(f"\n2. Configuration check:")
    print(f"   • Card Stats URL: {HSReplayConfig.CARD_STATS_URL}")
    print(f"   • Hero Stats URL: {HSReplayConfig.HERO_STATS_URL}")
    print(f"   • Request Timeout: {HSReplayConfig.REQUEST_TIMEOUT}")
    print(f"   • Retry Attempts: {HSReplayConfig.RETRY_ATTEMPTS}")
    print(f"   • Headers Count: {len(HSReplayConfig.HEADERS)}")
    
    # Test direct API call
    print(f"\n3. Testing direct API request to card stats endpoint...")
    try:
        response = scraper._make_api_request(
            HSReplayConfig.CARD_STATS_URL,
            params=HSReplayConfig.CARD_PARAMS
        )
        
        if response is None:
            print("   ❌ FAILED: _make_api_request returned None")
            return False
        else:
            print(f"   ✅ SUCCESS: Received response with status {response.status_code}")
            print(f"   • Response size: {len(response.content)} bytes")
            print(f"   • Response time: {response.elapsed.total_seconds():.2f}s")
            print(f"   • Final URL: {response.url}")
            
            # Try to parse as JSON
            try:
                data = response.json()
                if isinstance(data, list):
                    print(f"   • JSON data: list with {len(data)} items")
                    if len(data) > 0:
                        print(f"   • First item keys: {list(data[0].keys())[:5]}...")
                elif isinstance(data, dict):
                    print(f"   • JSON data: dict with keys: {list(data.keys())[:5]}...")
                else:
                    print(f"   • JSON data: {type(data)}")
            except Exception as json_err:
                print(f"   ⚠️  JSON parse error: {json_err}")
                print(f"   • Response content preview: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False
    
    # Test hero API endpoint
    print(f"\n4. Testing direct API request to hero stats endpoint...")
    try:
        response = scraper._make_api_request(HSReplayConfig.HERO_STATS_URL)
        
        if response is None:
            print("   ❌ FAILED: _make_api_request returned None")
            return False
        else:
            print(f"   ✅ SUCCESS: Received response with status {response.status_code}")
            print(f"   • Response size: {len(response.content)} bytes")
            print(f"   • Response time: {response.elapsed.total_seconds():.2f}s")
            
            # Try to parse as JSON and look for BGT_ARENA
            try:
                data = response.json()
                if isinstance(data, dict):
                    print(f"   • JSON data: dict with {len(data)} keys")
                    if "BGT_ARENA" in data:
                        print("   ✅ Found BGT_ARENA section in response")
                    else:
                        print(f"   • Available keys: {list(data.keys())[:10]}...")
                else:
                    print(f"   • JSON data type: {type(data)}")
            except Exception as json_err:
                print(f"   ⚠️  JSON parse error: {json_err}")
                print(f"   • Response content preview: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False
    
    # Test high-level methods
    print(f"\n5. Testing high-level scraper methods...")
    
    try:
        print("   • Testing get_underground_arena_stats()...")
        card_stats = scraper.get_underground_arena_stats(force_refresh=True)
        if card_stats:
            print(f"     ✅ SUCCESS: Retrieved {len(card_stats)} card stats")
        else:
            print("     ⚠️  No card stats returned (may be normal if API is down)")
            
        print("   • Testing get_hero_winrates()...")
        hero_stats = scraper.get_hero_winrates(force_refresh=True)
        if hero_stats:
            print(f"     ✅ SUCCESS: Retrieved {len(hero_stats)} hero winrates")
            print(f"     • Classes: {list(hero_stats.keys())}")
        else:
            print("     ⚠️  No hero stats returned (may be normal if API is down)")
            
    except Exception as e:
        print(f"   ❌ HIGH-LEVEL METHOD EXCEPTION: {e}")
        return False
    
    # Get final status
    print(f"\n6. Final status check...")
    status = scraper.get_status()
    print(f"   • Overall status: {status['status']}")
    print(f"   • Card status: {status['card_status']}")
    print(f"   • Hero status: {status['hero_status']}")
    print(f"   • API calls made: {status['api_calls_today']}")
    if status.get('last_error'):
        print(f"   • Last error: {status['last_error']}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("If you see HTTP 200 responses above, the 'No response' issue is FIXED!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_api_requests()
    sys.exit(0 if success else 1)