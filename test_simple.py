#!/usr/bin/env python3
"""
Simple step-by-step import test
"""

print("Starting import tests...")

print("1. Testing rapidfuzz...")
try:
    import rapidfuzz
    print("   ✅ rapidfuzz base import works")
except Exception as e:
    print(f"   ❌ rapidfuzz base import failed: {e}")
    exit(1)

print("2. Testing rapidfuzz.fuzz...")
try:
    from rapidfuzz import fuzz
    print("   ✅ rapidfuzz.fuzz works")
except Exception as e:
    print(f"   ❌ rapidfuzz.fuzz failed: {e}")

print("3. Testing selenium base...")
try:
    import selenium
    print("   ✅ selenium base import works")
except Exception as e:
    print(f"   ❌ selenium base import failed: {e}")
    exit(1)

print("4. Testing selenium webdriver...")
try:
    from selenium import webdriver
    print("   ✅ selenium webdriver works")
except Exception as e:
    print(f"   ❌ selenium webdriver failed: {e}")

print("All basic imports completed without crash!")
input("Press Enter to continue...")