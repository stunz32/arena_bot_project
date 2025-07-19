#!/usr/bin/env python3
"""
Test Ultimate Detector Performance
Tests the ultimate detector with debug images to measure actual accuracy.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project modules
sys.path.insert(0, str(Path(__file__).parent))

def test_ultimate_detector():
    """Test ultimate detector with debug images."""
    print("🎯 TESTING ULTIMATE DETECTOR PERFORMANCE")
    print("=" * 60)
    
    try:
        from ultimate_card_detector_clean import UltimateCardDetector
        
        # Load a comprehensive set of cards from different expansions
        from arena_bot.utils.asset_loader import AssetLoader
        asset_loader = AssetLoader()
        
        # Get all available card codes by scanning the cards directory
        cards_dir = asset_loader.assets_dir / "cards"
        all_card_files = list(cards_dir.glob("*.png"))
        
        # Extract unique card codes (remove _premium suffix)
        test_cards = []
        for card_file in all_card_files[:200]:  # Test with first 200 cards for speed
            card_code = card_file.stem.replace("_premium", "")
            if card_code not in test_cards and not card_code.endswith("t"):  # Skip tokens
                test_cards.append(card_code)
        
        print(f"📚 Loaded {len(test_cards)} cards from database for comprehensive testing")
        
        detector = UltimateCardDetector(target_cards=test_cards)
        print(f"✅ Ultimate detector initialized with {len(test_cards)} target cards")
        
        # Test with debug images
        debug_images = ["debug_card_1.png", "debug_card_2.png", "debug_card_3.png"]
        
        for i, image_name in enumerate(debug_images, 1):
            image_path = Path(__file__).parent / image_name
            
            if not image_path.exists():
                print(f"⚠️ Card {i}: {image_name} not found")
                continue
                
            print(f"\n🔍 Testing Card {i}: {image_name}")
            
            # Load and create mock screenshot
            card_image = cv2.imread(str(image_path))
            if card_image is None:
                print(f"❌ Failed to load {image_name}")
                continue
            
            # Create realistic arena screenshot
            screenshot = np.zeros((1440, 3440, 3), dtype=np.uint8)
            
            # Arena interface coordinates
            interface_x, interface_y = 1122, 233
            interface_w, interface_h = 1197, 704
            
            # Add background
            screenshot[interface_y:interface_y+interface_h, interface_x:interface_x+interface_w] = [45, 25, 15]
            
            # Place scaled card
            card_w, card_h = 447, 493
            card_resized = cv2.resize(card_image, (card_w, card_h))
            
            # Place 3 cards at proper positions
            card_positions = [
                (interface_x + 80, interface_y + 100),
                (interface_x + 375, interface_y + 100), 
                (interface_x + 670, interface_y + 100)
            ]
            
            for pos_x, pos_y in card_positions:
                end_x = pos_x + card_w
                end_y = pos_y + card_h
                screenshot[pos_y:end_y, pos_x:end_x] = card_resized
            
            print(f"   Created arena screenshot with 3 cards")
            
            # Test detection
            try:
                result = detector.detect_cards_with_targets(screenshot, test_cards)
                
                if result.get('success'):
                    detected_cards = result.get('detected_cards', [])
                    print(f"✅ Detection successful: {len(detected_cards)} cards")
                    
                    for card in detected_cards:
                        print(f"   📋 Card {card.get('position', '?')}: {card.get('card_name', 'Unknown')} "
                              f"(conf: {card.get('confidence', 0):.3f})")
                else:
                    error = result.get('error', 'unknown')
                    print(f"❌ Detection failed: {error}")
                    
            except Exception as e:
                print(f"❌ Error during detection: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎯 Ultimate detector test complete!")
        
    except Exception as e:
        print(f"❌ Failed to initialize ultimate detector: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ultimate_detector()