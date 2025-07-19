#!/usr/bin/env python3
"""
Test with full card database (4000+ cards).
Compare detection results with limited vs full database.
"""

import os
import sys
import cv2
import time
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_full_database(screenshot_path: str):
    """Test card detection with the full database."""
    print("🎮 Arena Bot - FULL DATABASE TESTING")
    print("=" * 70)
    print("🔄 Loading ALL 4,019+ cards - this may take a few minutes...")
    
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
        from arena_bot.detection.template_matcher import get_template_matcher
        from arena_bot.core.window_detector import get_window_detector
        
        asset_loader = get_asset_loader()
        histogram_matcher = get_histogram_matcher()
        template_matcher = get_template_matcher()
        window_detector = get_window_detector()
        
        # Initialize components
        print("🔧 Initializing detection components...")
        template_matcher.initialize()
        window_detector.initialize()
        
        # Get all available cards
        print("📚 Discovering all available cards...")
        available_cards = asset_loader.get_available_cards()
        total_cards = len(available_cards)
        print(f"   Found {total_cards} cards in database")
        
        # Load ALL cards (this will take time)
        print("⏳ Loading full card database - progress every 500 cards...")
        start_time = time.time()
        
        card_images = {}
        loaded_count = 0
        failed_count = 0
        
        for i, card_code in enumerate(available_cards):
            # Load normal version
            normal_image = asset_loader.load_card_image(card_code, premium=False)
            if normal_image is not None:
                card_images[card_code] = normal_image
                loaded_count += 1
            else:
                failed_count += 1
            
            # Load premium version if available
            premium_image = asset_loader.load_card_image(card_code, premium=True) 
            if premium_image is not None:
                card_images[f"{card_code}_premium"] = premium_image
                loaded_count += 1
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"   Loaded {i + 1}/{total_cards} cards ({loaded_count} images) - {elapsed:.1f}s")
        
        final_elapsed = time.time() - start_time
        print(f"✅ Card loading complete!")
        print(f"   📊 Total cards: {total_cards}")
        print(f"   📊 Images loaded: {loaded_count}")
        print(f"   📊 Failed to load: {failed_count}")
        print(f"   ⏱️  Loading time: {final_elapsed:.1f} seconds")
        
        # Load into histogram matcher
        print("🔄 Computing histograms for all cards...")
        hist_start_time = time.time()
        
        histogram_matcher.load_card_database(card_images)
        final_db_size = histogram_matcher.get_database_size()
        
        hist_elapsed = time.time() - hist_start_time
        print(f"✅ Histogram computation complete!")
        print(f"   📊 Database size: {final_db_size} histograms")
        print(f"   ⏱️  Histogram time: {hist_elapsed:.1f} seconds")
        print(f"   ⏱️  Total setup time: {final_elapsed + hist_elapsed:.1f} seconds")
        print()
        
        # Auto-detect arena regions
        print("🎯 AUTO-DETECTING arena interface...")
        ui_elements = window_detector.auto_detect_arena_cards(screenshot)
        
        if ui_elements is None:
            print("❌ Failed to detect arena interface")
            return False
        
        detection_type = "Template-based" if ui_elements.confidence > 0.5 else "Manual fallback"
        print(f"✅ Arena interface detected")
        print(f"   Method: {detection_type}")
        print(f"   Confidence: {ui_elements.confidence:.3f}")
        print(f"   Regions: {len(ui_elements.card_regions)} cards")
        print()
        
        # Test card detection with FULL database
        print("🔍 TESTING CARD DETECTION WITH FULL DATABASE")
        print("-" * 70)
        
        results = []
        detection_start = time.time()
        
        for i, (x, y, w, h) in enumerate(ui_elements.card_regions):
            print(f"  Testing card {i+1} at region ({x}, {y}, {w}, {h})")
            
            # Extract card region with bounds checking
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w > 50 and h > 50:
                card_image = screenshot[y:y+h, x:x+w]
                
                # Save region for debugging
                debug_path = f"full_db_card_{i+1}.png"
                cv2.imwrite(debug_path, card_image)
                print(f"    💾 Saved region to {debug_path}")
                
                # Test histogram matching (measure time)
                match_start = time.time()
                hist = histogram_matcher.compute_histogram(card_image)
                
                if hist is not None:
                    # Get top matches for analysis
                    best_matches = histogram_matcher.find_best_matches(hist, max_candidates=5)
                    match_elapsed = time.time() - match_start
                    
                    if best_matches:
                        print(f"    🔍 Top matches ({match_elapsed:.3f}s):")
                        for j, match in enumerate(best_matches[:3]):
                            print(f"      {j+1}. {match.card_code} (distance: {match.distance:.3f}, conf: {match.confidence:.3f})")
                    
                    # Get best match with threshold
                    match = histogram_matcher.match_card(card_image, confidence_threshold=0.8)
                    
                    if match:
                        print(f"    ✅ DETECTED: {match.card_code}")
                        print(f"       Confidence: {match.confidence:.3f}")
                        print(f"       Premium: {match.is_premium}")
                        
                        # Test template matching
                        card_h, card_w = card_image.shape[:2]
                        mana_region = card_image[0:int(card_h*0.3), 0:int(card_w*0.3)]
                        rarity_region = card_image[int(card_h*0.7):card_h, int(card_w*0.4):int(card_w*0.6)]
                        
                        mana_cost = template_matcher.detect_mana_cost(mana_region)
                        rarity = template_matcher.detect_rarity(rarity_region)
                        
                        print(f"       Mana: {mana_cost}, Rarity: {rarity}")
                        
                        results.append({
                            'position': i+1,
                            'card_code': match.card_code,
                            'confidence': match.confidence,
                            'mana_cost': mana_cost,
                            'rarity': rarity,
                            'is_premium': match.is_premium,
                            'detection_time': match_elapsed
                        })
                    else:
                        print(f"    ❌ No confident match found")
                        if best_matches:
                            print(f"       Best was: {best_matches[0].card_code} (conf: {best_matches[0].confidence:.3f})")
                else:
                    print(f"    ❌ Failed to compute histogram")
                    
                print()  # Space between cards
            else:
                print(f"    ⚠️  Region too small or out of bounds")
        
        detection_elapsed = time.time() - detection_start
        
        # Results summary
        print("📊 FULL DATABASE RESULTS")
        print("=" * 70)
        print(f"Database size: {final_db_size:,} histograms")
        print(f"Setup time: {final_elapsed + hist_elapsed:.1f} seconds")
        print(f"Detection time: {detection_elapsed:.3f} seconds")
        print(f"Cards detected: {len(results)}/3")
        
        if results:
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            avg_time = sum(r['detection_time'] for r in results) / len(results)
            print(f"Average confidence: {avg_conf:.3f}")
            print(f"Average detection time: {avg_time:.3f}s per card")
            print()
            
            print("🎯 DETECTED CARDS:")
            for result in results:
                premium_text = " (Premium)" if result['is_premium'] else ""
                print(f"   Card {result['position']}: {result['card_code']}{premium_text}")
                print(f"      Confidence: {result['confidence']:.3f}")
                print(f"      Mana: {result['mana_cost']}, Rarity: {result['rarity']}")
                print(f"      Detection time: {result['detection_time']:.3f}s")
                print()
        
        # Performance analysis
        print("⚡ PERFORMANCE ANALYSIS:")
        if len(results) > 0:
            print(f"   ✅ Success rate: {len(results)}/3 ({len(results)/3*100:.1f}%)")
        else:
            print(f"   ❌ Success rate: 0/3 (0.0%)")
        
        print(f"   📈 Database scale: {final_db_size/500:.1f}x larger than test database")
        print(f"   ⚡ Detection speed: {avg_time*1000:.0f}ms per card" if results else "   ⚡ Detection speed: N/A")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Full database testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Setup logging (reduce noise during large operations)
    logging.basicConfig(level=logging.WARNING)
    
    screenshot_path = "screenshot.png"
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return False
    
    success = test_full_database(screenshot_path)
    
    if success:
        print("\n🎉 Full database testing completed successfully!")
        print("💡 Check full_db_card_*.png files for extracted regions")
    else:
        print("\n⚠️  Full database testing completed with issues")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)