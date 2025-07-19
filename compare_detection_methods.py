#!/usr/bin/env python3
"""
Compare different detection methods: Manual vs Template-based vs Auto-detection.
Shows the effectiveness of the new window detection system.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def compare_detection_methods(screenshot_path: str):
    """Compare manual, template-based, and auto-detection methods."""
    print("🔍 DETECTION METHOD COMPARISON")
    print("=" * 80)
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print("❌ Failed to load screenshot")
            return False
        
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import get_histogram_matcher
        from arena_bot.core.window_detector import get_window_detector
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        window_detector = get_window_detector()
        window_detector.initialize()
        
        # Load card database
        available_cards = asset_loader.get_available_cards()
        test_cards = available_cards[:500]
        card_images = {code: img for code in test_cards if (img := asset_loader.load_card_image(code)) is not None}
        histogram_matcher.load_card_database(card_images)
        
        print(f"📚 Loaded {histogram_matcher.get_database_size()} cards for testing")
        print()
        
        # Method 1: Manual positioning (our original approach)
        print("🔧 METHOD 1: MANUAL POSITIONING")
        print("-" * 50)
        
        if width > 3000:  # Ultrawide
            manual_regions = [
                (int(width * 0.25 - 100), int(height * 0.35), 200, 280),
                (int(width * 0.50 - 100), int(height * 0.35), 200, 280),
                (int(width * 0.75 - 100), int(height * 0.35), 200, 280)
            ]
        else:  # Standard
            scale_x, scale_y = width / 1920.0, height / 1080.0
            manual_regions = [
                (int(480 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y)),
                (int(860 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y)),
                (int(1240 * scale_x), int(300 * scale_y), int(200 * scale_x), int(300 * scale_y))
            ]
        
        manual_results = test_regions(screenshot, manual_regions, histogram_matcher, "manual")
        print(f"Manual positioning: {len(manual_results)}/3 cards detected")
        for i, result in enumerate(manual_results):
            print(f"  Card {i+1}: {result['card_code']} (conf: {result['confidence']:.3f})")
        print()
        
        # Method 2: UI Template-based detection
        print("🎯 METHOD 2: UI TEMPLATE-BASED DETECTION")
        print("-" * 50)
        
        ui_elements = window_detector.detect_arena_ui(screenshot)
        if ui_elements and ui_elements.confidence > 0.3:
            template_results = test_regions(screenshot, ui_elements.card_regions, histogram_matcher, "template")
            print(f"Template-based detection: {len(template_results)}/3 cards detected")
            print(f"UI confidence: {ui_elements.confidence:.3f}")
            for i, result in enumerate(template_results):
                print(f"  Card {i+1}: {result['card_code']} (conf: {result['confidence']:.3f})")
        else:
            template_results = []
            print("Template-based detection: Failed to detect UI")
        print()
        
        # Method 3: Auto-detection (hybrid approach)
        print("🤖 METHOD 3: AUTO-DETECTION (HYBRID)")
        print("-" * 50)
        
        auto_ui_elements = window_detector.auto_detect_arena_cards(screenshot)
        if auto_ui_elements:
            auto_results = test_regions(screenshot, auto_ui_elements.card_regions, histogram_matcher, "auto")
            detection_type = "Template-based" if auto_ui_elements.confidence > 0.5 else "Manual fallback"
            print(f"Auto-detection: {len(auto_results)}/3 cards detected")
            print(f"Detection type: {detection_type}")
            print(f"Confidence: {auto_ui_elements.confidence:.3f}")
            for i, result in enumerate(auto_results):
                print(f"  Card {i+1}: {result['card_code']} (conf: {result['confidence']:.3f})")
        else:
            auto_results = []
            print("Auto-detection: Failed")
        print()
        
        # Comparison Summary
        print("📊 COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Method':<25} {'Cards Detected':<15} {'Avg Confidence':<15} {'Best Match'}")
        print("-" * 80)
        
        methods = [
            ("Manual Positioning", manual_results),
            ("Template-based", template_results),
            ("Auto-detection", auto_results)
        ]
        
        for method_name, results in methods:
            if results:
                avg_conf = sum(r['confidence'] for r in results) / len(results)
                best_match = max(results, key=lambda x: x['confidence'])
                print(f"{method_name:<25} {len(results)}/3{'':<11} {avg_conf:.3f}{'':<11} {best_match['card_code']} ({best_match['confidence']:.3f})")
            else:
                print(f"{method_name:<25} {'0/3':<15} {'N/A':<15} {'N/A'}")
        
        print()
        print("💡 OBSERVATIONS:")
        
        # Compare region positions
        if ui_elements and ui_elements.confidence > 0.3:
            print("   🎯 Template-based detection found different card positions than manual:")
            for i, (manual_region, template_region) in enumerate(zip(manual_regions, ui_elements.card_regions)):
                manual_pos = f"({manual_region[0]}, {manual_region[1]})"
                template_pos = f"({template_region[0]}, {template_region[1]})"
                print(f"      Card {i+1}: Manual {manual_pos} vs Template {template_pos}")
        
        if len(auto_results) > len(manual_results):
            print("   ✅ Auto-detection performed better than manual positioning")
        elif len(auto_results) == len(manual_results):
            print("   ⚖️  Auto-detection performed equally to manual positioning")
        else:
            print("   ⚠️  Manual positioning performed better (template needs tuning)")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regions(screenshot, regions, histogram_matcher, prefix):
    """Test card detection on given regions."""
    results = []
    
    for i, (x, y, w, h) in enumerate(regions):
        # Bounds checking
        height, width = screenshot.shape[:2]
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w > 50 and h > 50:
            card_image = screenshot[y:y+h, x:x+w]
            
            # Save region for debugging
            cv2.imwrite(f"{prefix}_region_{i+1}.png", card_image)
            
            # Test histogram matching
            match = histogram_matcher.match_card(card_image, confidence_threshold=0.8)
            if match:
                results.append({
                    'position': i+1,
                    'card_code': match.card_code,
                    'confidence': match.confidence,
                    'region': (x, y, w, h)
                })
    
    return results

def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    return compare_detection_methods(screenshot_path)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)