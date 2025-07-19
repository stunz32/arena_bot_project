#!/usr/bin/env python3
"""
Test import availability for debugging
"""

print("Testing imports...")

# Test rapidfuzz
try:
    from rapidfuzz import fuzz, process
    print("✅ rapidfuzz imports work")
    FUZZY_MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"❌ rapidfuzz failed: {e}")
    FUZZY_MATCHING_AVAILABLE = False

# Test selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, WebDriverException
    print("✅ selenium imports work")
    SELENIUM_AVAILABLE = True
except ImportError as e:
    print(f"❌ selenium failed: {e}")
    SELENIUM_AVAILABLE = False

# Test webdriver-manager
try:
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    print("✅ webdriver-manager imports work")
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"❌ webdriver-manager failed: {e}")
    WEBDRIVER_MANAGER_AVAILABLE = False

print(f"\nFinal status:")
print(f"FUZZY_MATCHING_AVAILABLE: {FUZZY_MATCHING_AVAILABLE}")
print(f"SELENIUM_AVAILABLE: {SELENIUM_AVAILABLE}")
print(f"WEBDRIVER_MANAGER_AVAILABLE: {WEBDRIVER_MANAGER_AVAILABLE}")

if FUZZY_MATCHING_AVAILABLE and SELENIUM_AVAILABLE:
    print("\n✅ All imports working - arena database setup should work!")
else:
    print("\n❌ Some imports failed - this explains the errors")