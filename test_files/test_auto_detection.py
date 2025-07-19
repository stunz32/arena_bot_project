#!/usr/bin/env python3
"""
Test the complete auto detection system.
"""

import cv2
import logging
from arena_bot.core.auto_detector import get_auto_detector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    """Test automatic detection on our screenshot."""
    print("=== Testing Auto Detection System ===")
    
    # Initialize auto detector
    auto_detector = get_auto_detector()
    
    # Test with our screenshot
    screenshot_path = "screenshot.png"
    
    print(f"\nProcessing screenshot: {screenshot_path}")
    result = auto_detector.detect_single_screenshot(screenshot_path)
    
    if result and result['success']:
        print(f"\n✓ Detection successful!")
        print(f"Interface detected at: {result['interface_rect']}")
        print(f"Cards detected: {len(result['detected_cards'])}/3")
        print(f"Overall accuracy: {result['accuracy']:.1%}")
        
        print("\nDetected cards:")
        for i, card in enumerate(result['detected_cards']):
            print(f"  Card {card['position']}: {card['card_code']} "
                  f"(confidence: {card['confidence']:.3f}, "
                  f"{'premium' if card['is_premium'] else 'normal'})")
    else:
        print("\n✗ Detection failed")
        if result:
            print(f"Accuracy: {result.get('accuracy', 0):.1%}")

if __name__ == "__main__":
    main()