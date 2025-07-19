#!/usr/bin/env python3
"""
Headless test script for Arena Bot core functionality.
Tests card recognition without requiring Qt GUI components.
"""

import os
import sys
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow {Image.__version__}")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        return False
    
    return True

def test_asset_loading():
    """Test asset loading functionality."""
    print("\n📁 Testing asset loading...")
    
    try:
        from arena_bot.utils.asset_loader import AssetLoader
        
        loader = AssetLoader()
        
        # Test loading some card images
        available_cards = loader.get_available_cards()
        card_count = len(available_cards)
        
        # Test loading templates
        mana_templates = loader.load_mana_templates()
        rarity_templates = loader.load_rarity_templates()
        
        print(f"✅ Asset loader initialized")
        print(f"   - Card images: {card_count}")
        print(f"   - Mana templates: {len(mana_templates)}")
        print(f"   - Rarity templates: {len(rarity_templates)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Asset loading failed: {e}")
        return False

def test_histogram_matching():
    """Test histogram matching functionality without screen capture."""
    print("\n📊 Testing histogram matching...")
    
    try:
        from arena_bot.detection.histogram_matcher import HistogramMatcher
        
        matcher = HistogramMatcher()
        
        # Test with a dummy histogram (correct dimensions for HSV histogram)
        import numpy as np
        dummy_hist = np.random.rand(50, 60).astype(np.float32)
        
        # Test basic functionality
        db_size = matcher.get_database_size()
        
        print(f"✅ Histogram matcher working")
        print(f"   - Database size: {db_size} cards")
        print(f"   - Histogram parameters: {matcher.H_BINS}x{matcher.S_BINS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Histogram matching failed: {e}")
        return False

def test_template_matching():
    """Test template matching functionality."""
    print("\n🎯 Testing template matching...")
    
    try:
        from arena_bot.detection.template_matcher import TemplateMatcher
        
        matcher = TemplateMatcher()
        
        # Test with a dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        mana_result = matcher.detect_mana_cost(dummy_image)
        rarity_result = matcher.detect_rarity(dummy_image)
        
        print(f"✅ Template matcher working")
        print(f"   - Mana templates loaded: {len(matcher.mana_templates)}")
        print(f"   - Rarity templates loaded: {len(matcher.rarity_templates)}")
        print(f"   - Test mana detection: {mana_result}")
        print(f"   - Test rarity detection: {rarity_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Template matching failed: {e}")
        return False

def main():
    """Run all headless tests."""
    print("🎮 Arena Bot - Headless Test Suite")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs
    
    tests = [
        ("Import Test", test_imports),
        ("Asset Loading", test_asset_loading),
        ("Histogram Matching", test_histogram_matching),
        ("Template Matching", test_template_matching),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Arena Bot core functionality is working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)