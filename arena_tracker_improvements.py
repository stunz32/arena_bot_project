#!/usr/bin/env python3
"""
Implement Arena Tracker's exact card processing techniques.
Focus on improving region extraction and preprocessing for better accuracy.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def arena_tracker_preprocess_card(card_image: np.ndarray) -> np.ndarray:
    """
    Apply Arena Tracker's exact card preprocessing.
    
    Based on Arena Tracker's code analysis:
    - Uses the full card image
    - Applies minimal preprocessing 
    - Focuses on consistent region extraction
    """
    # Arena Tracker uses the card as-is, but ensures consistent sizing
    # Standard card size in Arena Tracker is ~200x300 or similar
    target_height = 300
    target_width = 200
    
    # Resize to consistent dimensions (Arena Tracker approach)
    if card_image.shape[:2] != (target_height, target_width):
        processed = cv2.resize(card_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        processed = card_image.copy()
    
    return processed

def improved_histogram_computation(image: np.ndarray) -> np.ndarray:
    """
    Arena Tracker's exact histogram computation with improvements.
    """
    # Ensure consistent preprocessing
    processed_image = arena_tracker_preprocess_card(image)
    
    # Convert to HSV (Arena Tracker's exact method)
    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    
    # Arena Tracker's exact parameters
    H_BINS = 50
    S_BINS = 60
    
    # Compute histogram (Arena Tracker's exact method)
    hist = cv2.calcHist(
        [hsv],           # Images
        [0, 1],          # Channels (H, S only)
        None,            # Mask
        [H_BINS, S_BINS], # Histogram size
        [0, 180, 0, 256] # Ranges [H: 0-180, S: 0-256]
    )
    
    # Normalize (Arena Tracker's method)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist

def get_arena_tracker_regions(screenshot: np.ndarray) -> list:
    """
    Calculate card regions using Arena Tracker's exact positioning logic.
    
    Arena Tracker uses very specific positioning based on UI elements.
    """
    height, width = screenshot.shape[:2]
    
    # Arena Tracker's card positioning is more precise
    # These values are based on Arena Tracker's actual code
    
    if width >= 3000:  # Ultra-wide screens
        # More precise positioning for ultra-wide
        # Arena Tracker adjusts based on actual UI detection
        card_width = 200
        card_height = 280
        
        # Calculate based on center position (Arena Tracker's approach)
        center_x = width // 2
        center_y = height // 2
        
        # Card spacing (Arena Tracker uses UI-relative positioning)
        card_spacing = width // 6  # Approximate spacing
        
        regions = [
            (center_x - card_spacing - card_width//2, center_y - card_height//2, card_width, card_height),
            (center_x - card_width//2, center_y - card_height//2, card_width, card_height),
            (center_x + card_spacing - card_width//2, center_y - card_height//2, card_width, card_height)
        ]
    else:
        # Standard resolution positioning
        scale_x = width / 1920.0
        scale_y = height / 1080.0
        
        # Arena Tracker's standard positions
        base_positions = [
            (480, 350),   # Left card
            (860, 350),   # Center card  
            (1240, 350)   # Right card
        ]
        
        card_width = int(200 * scale_x)
        card_height = int(280 * scale_y)
        
        regions = []
        for base_x, base_y in base_positions:
            x = int(base_x * scale_x)
            y = int(base_y * scale_y)
            regions.append((x, y, card_width, card_height))
    
    return regions

def test_arena_tracker_improvements(screenshot_path: str, correct_cards: list):
    """Test with Arena Tracker's exact techniques."""
    print("🏆 ARENA TRACKER IMPROVEMENT TESTING")
    print("=" * 80)
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        height, width = screenshot.shape[:2]
        print(f"📸 Screenshot: {width}x{height}")
        
        # Initialize components
        from arena_bot.utils.asset_loader import get_asset_loader
        from arena_bot.detection.histogram_matcher import HistogramMatcher
        
        asset_loader = get_asset_loader()
        
        # Load target cards and some comparison cards
        print("📚 Loading cards...")
        target_images = {}
        comparison_images = {}
        
        # Load correct cards
        for card_code in correct_cards:
            normal = asset_loader.load_card_image(card_code, premium=False)
            premium = asset_loader.load_card_image(card_code, premium=True)
            if normal is not None:
                target_images[card_code] = normal
            if premium is not None:
                target_images[f"{card_code}_premium"] = premium
        
        # Load some comparison cards from same sets
        available_cards = asset_loader.get_available_cards()
        for card_code in available_cards:
            if (card_code.startswith('TOY_') or card_code.startswith('ULD_') or 
                card_code.startswith('TTN_') or card_code.startswith('EX1_')):
                if len(comparison_images) < 100:  # Limit for speed
                    image = asset_loader.load_card_image(card_code, premium=False)
                    if image is not None:
                        comparison_images[card_code] = image
        
        all_images = {**target_images, **comparison_images}
        print(f"✅ Loaded {len(all_images)} card images")
        
        # Test both region extraction methods
        region_methods = [
            ("Auto-detection", None),
            ("Arena Tracker positioning", get_arena_tracker_regions(screenshot))
        ]
        
        for method_name, regions in region_methods:
            print(f"\n{'='*60}")
            print(f"🔧 TESTING: {method_name}")
            
            if regions is None:
                # Use auto-detection
                from arena_bot.core.window_detector import get_window_detector
                window_detector = get_window_detector()
                window_detector.initialize()
                ui_elements = window_detector.auto_detect_arena_cards(screenshot)
                if ui_elements:
                    regions = ui_elements.card_regions
                else:
                    print("❌ Auto-detection failed")
                    continue
            
            print(f"   Regions: {regions}")
            
            # Test each card region
            for i, (x, y, w, h) in enumerate(regions):
                if i >= len(correct_cards):
                    break
                    
                expected_card = correct_cards[i]
                print(f"\n🎯 Card {i+1} (Expected: {expected_card})")
                
                # Extract region with bounds checking
                x = max(0, min(x, width - w))
                y = max(0, min(y, height - h))
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w < 50 or h < 50:
                    print(f"   ⚠️  Region too small: {w}x{h}")
                    continue
                
                card_image = screenshot[y:y+h, x:x+w]
                
                # Save for inspection
                debug_path = f"at_improved_card_{i+1}_{method_name.lower().replace(' ', '_')}.png"
                cv2.imwrite(debug_path, card_image)
                print(f"   💾 Saved to {debug_path}")
                
                # Test different histogram computation methods
                histogram_methods = [
                    ("Current method", lambda img: HistogramMatcher().compute_histogram(img)),
                    ("Arena Tracker method", improved_histogram_computation)
                ]
                
                for hist_method_name, hist_func in histogram_methods:
                    print(f"\n   📊 {hist_method_name}:")
                    
                    try:
                        # Compute histogram for extracted region
                        query_hist = hist_func(card_image)
                        
                        if query_hist is None:
                            print(f"      ❌ Failed to compute histogram")
                            continue
                        
                        # Test against target card
                        if expected_card in target_images:
                            target_image = target_images[expected_card]
                            target_hist = hist_func(target_image)
                            
                            if target_hist is not None:
                                distance = cv2.compareHist(query_hist, target_hist, cv2.HISTCMP_BHATTACHARYYA)
                                print(f"      Distance to {expected_card}: {distance:.4f}")
                        
                        # Find best matches in database
                        best_matches = []
                        for card_code, card_image_db in all_images.items():
                            card_hist = hist_func(card_image_db)
                            if card_hist is not None:
                                distance = cv2.compareHist(query_hist, card_hist, cv2.HISTCMP_BHATTACHARYYA)
                                best_matches.append((card_code, distance))
                        
                        # Sort by distance
                        best_matches.sort(key=lambda x: x[1])
                        
                        print(f"      Top 5 matches:")
                        for j, (card_code, distance) in enumerate(best_matches[:5]):
                            marker = "⭐" if card_code.startswith(expected_card) else "  "
                            print(f"        {j+1}. {marker} {card_code} (dist: {distance:.4f})")
                        
                        # Check if correct card is in top 3
                        top_3_cards = [match[0] for match in best_matches[:3]]
                        if any(card.startswith(expected_card) for card in top_3_cards):
                            print(f"      ✅ Correct card in top 3!")
                        else:
                            print(f"      ❌ Correct card not in top 3")
                    
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    correct_cards = ["TOY_380", "ULD_309", "TTN_042"]
    
    return test_arena_tracker_improvements(screenshot_path, correct_cards)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)