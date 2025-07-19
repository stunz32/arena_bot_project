#!/usr/bin/env python3
"""
Debug histogram matching by directly comparing extracted cards vs reference cards.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the arena_bot package to path
sys.path.insert(0, str(Path(__file__).parent))

def compute_arena_tracker_histogram(image: np.ndarray) -> np.ndarray:
    """Arena Tracker's exact histogram method."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    ranges = [0, 180, 0, 256]
    channels = [0, 1]
    
    hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def extract_arena_tracker_region(card_image: np.ndarray, is_premium: bool = False) -> np.ndarray:
    """Extract Arena Tracker's 80x80 region."""
    if is_premium:
        x, y, w, h = 57, 71, 80, 80
    else:
        x, y, w, h = 60, 71, 80, 80
    
    if (card_image.shape[1] < x + w) or (card_image.shape[0] < y + h):
        return None
    
    return card_image[y:y+h, x:x+w]

def debug_histogram_matching():
    """Debug why histogram matching isn't finding the correct cards."""
    print("🔬 DEBUGGING HISTOGRAM MATCHING")
    print("=" * 80)
    
    try:
        from arena_bot.utils.asset_loader import get_asset_loader
        asset_loader = get_asset_loader()
        
        target_cards = ["TOY_380", "ULD_309", "TTN_042"]
        extracted_card_files = [
            "correct_coords_card_1.png",  # TOY_380
            "correct_coords_card_2.png",  # ULD_309
            "correct_coords_card_3.png",  # TTN_042
        ]
        
        for i, (card_code, extracted_file) in enumerate(zip(target_cards, extracted_card_files)):
            print(f"\n{'='*60}")
            print(f"🔍 DEBUGGING CARD {i+1}: {card_code}")
            print(f"{'='*60}")
            
            # Load reference card from database
            reference_card = asset_loader.load_card_image(card_code, premium=False)
            if reference_card is None:
                print(f"❌ Reference card {card_code} not found")
                continue
            
            # Load extracted card from screenshot
            extracted_card = cv2.imread(extracted_file)
            if extracted_card is None:
                print(f"❌ Extracted card file {extracted_file} not found")
                continue
            
            print(f"📄 Reference card shape: {reference_card.shape}")
            print(f"📸 Extracted card shape: {extracted_card.shape}")
            
            # Method 1: Compare reference Arena Tracker region vs extracted card directly
            print(f"\n🔬 Method 1: Reference AT region vs Extracted (resized to 80x80)")
            
            # Get Arena Tracker region from reference
            reference_at_region = extract_arena_tracker_region(reference_card, is_premium=False)
            if reference_at_region is not None:
                print(f"   Reference AT region: {reference_at_region.shape}")
                cv2.imwrite(f"debug_ref_{card_code}_at_region.png", reference_at_region)
                
                # Resize extracted card to 80x80 for comparison
                extracted_resized = cv2.resize(extracted_card, (80, 80), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"debug_ext_{card_code}_resized.png", extracted_resized)
                
                # Compute histograms
                ref_hist = compute_arena_tracker_histogram(reference_at_region)
                ext_hist = compute_arena_tracker_histogram(extracted_resized)
                
                # Compare directly
                distance = cv2.compareHist(ref_hist, ext_hist, cv2.HISTCMP_BHATTACHARYYA)
                print(f"   Direct comparison distance: {distance:.4f}")
                
                if distance < 0.5:
                    print(f"   ✅ GOOD MATCH! (distance < 0.5)")
                else:
                    print(f"   ❌ Poor match (distance >= 0.5)")
            
            # Method 2: Test different regions of extracted card
            print(f"\n🔬 Method 2: Testing different regions of extracted card")
            
            strategies = [
                ("Full card resized", cv2.resize(extracted_card, (80, 80), interpolation=cv2.INTER_AREA)),
                ("Upper 70% resized", cv2.resize(extracted_card[0:int(extracted_card.shape[0]*0.7), :], (80, 80), interpolation=cv2.INTER_AREA)),
                ("Card art region", cv2.resize(extracted_card[20:150, 20:extracted_card.shape[1]-20], (80, 80), interpolation=cv2.INTER_AREA) if extracted_card.shape[0] >= 170 else None),
                ("Center crop", cv2.resize(extracted_card[30:extracted_card.shape[0]-30, 30:extracted_card.shape[1]-30], (80, 80), interpolation=cv2.INTER_AREA) if extracted_card.shape[0] >= 60 else None),
            ]
            
            best_distance = float('inf')
            best_strategy = None
            
            if reference_at_region is not None:
                ref_hist = compute_arena_tracker_histogram(reference_at_region)
                
                for strategy_name, processed_region in strategies:
                    if processed_region is None:
                        continue
                    
                    # Save processed region for inspection
                    cv2.imwrite(f"debug_ext_{card_code}_{strategy_name.replace(' ', '_').lower()}.png", processed_region)
                    
                    # Compute histogram and compare
                    ext_hist = compute_arena_tracker_histogram(processed_region)
                    distance = cv2.compareHist(ref_hist, ext_hist, cv2.HISTCMP_BHATTACHARYYA)
                    
                    print(f"   {strategy_name}: distance = {distance:.4f}")
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_strategy = strategy_name
                
                print(f"   🏆 Best strategy: {best_strategy} (distance: {best_distance:.4f})")
            
            # Method 3: Compare against full database to see ranking
            print(f"\n🔬 Method 3: Database ranking test")
            
            # Load a small subset of database for ranking test
            print("   Loading database subset...")
            available_cards = asset_loader.get_available_cards()
            card_hists = {}
            
            # Load target card and some others for comparison
            test_cards = [card_code] + [c for c in available_cards[:100] if c != card_code][:20]
            
            for test_card in test_cards:
                for is_premium in [False, True]:
                    test_image = asset_loader.load_card_image(test_card, premium=is_premium)
                    if test_image is not None:
                        at_region = extract_arena_tracker_region(test_image, is_premium=is_premium)
                        if at_region is not None:
                            hist = compute_arena_tracker_histogram(at_region)
                            hist_key = f"{test_card}{'_premium' if is_premium else ''}"
                            card_hists[hist_key] = hist
            
            print(f"   Loaded {len(card_hists)} test histograms")
            
            # Test with best extracted region
            if best_strategy and best_distance < float('inf'):
                # Get the best processed region
                for strategy_name, processed_region in strategies:
                    if strategy_name == best_strategy and processed_region is not None:
                        ext_hist = compute_arena_tracker_histogram(processed_region)
                        
                        # Compare with database
                        matches = []
                        for hist_key, card_hist in card_hists.items():
                            distance = cv2.compareHist(ext_hist, card_hist, cv2.HISTCMP_BHATTACHARYYA)
                            matches.append((distance, hist_key))
                        
                        matches.sort(key=lambda x: x[0])
                        
                        print(f"   📋 Top 10 matches:")
                        target_rank = None
                        for rank, (distance, hist_key) in enumerate(matches[:10]):
                            base_code = hist_key.replace('_premium', '')
                            is_target = base_code == card_code
                            marker = "🎯" if is_target else "  "
                            print(f"      {rank+1:2d}. {marker} {hist_key:20s} (dist: {distance:.4f})")
                            
                            if is_target and target_rank is None:
                                target_rank = rank + 1
                        
                        if target_rank:
                            print(f"   ✅ Target found at rank {target_rank}!")
                        else:
                            print(f"   ❌ Target not in top 10")
                        break
        
        print(f"\n{'='*80}")
        print("🔬 DEBUGGING SUMMARY")
        print(f"{'='*80}")
        print("Check the debug_* images to see:")
        print("1. Reference Arena Tracker regions (debug_ref_*_at_region.png)")
        print("2. Extracted card processing strategies (debug_ext_*_*.png)")
        print("3. Direct distance comparisons between reference and extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    logging.basicConfig(level=logging.WARNING)
    return debug_histogram_matching()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)