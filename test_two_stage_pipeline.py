#!/usr/bin/env python3

import cv2
import numpy as np
import logging
from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector
from arena_bot.core.card_refiner import CardRefiner


def test_two_stage_pipeline():
    """Test the complete two-stage pipeline with refinement."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load the screenshot
    screenshot_path = "/mnt/d/cursor bots/arena_bot_project/debug_frames/Hearthstone Screenshot 07-11-25 17.33.10.png"
    screenshot = cv2.imread(screenshot_path)
    
    if screenshot is None:
        print(f"Could not load screenshot from {screenshot_path}")
        return
    
    print(f"Testing two-stage pipeline on {screenshot.shape[1]}x{screenshot.shape[0]} screenshot")
    
    # Initialize detector
    detector = SmartCoordinateDetector()
    
    # STAGE 1: Get coarse ROI from SmartCoordinateDetector
    coarse_result = detector.detect_cards_automatically(screenshot)
    
    if not coarse_result or not coarse_result['success']:
        print("Stage 1 failed - could not detect coarse card positions")
        return
    
    coarse_positions = coarse_result['card_positions']
    height, width = screenshot.shape[:2]
    
    # STAGE 2: Apply CardRefiner to each coarse position
    refined_positions = []
    
    for i, (x, y, w, h) in enumerate(coarse_positions):
        print(f"\n=== REFINING CARD {i+1} ===")
        print(f"Coarse position: ({x}, {y}, {w}, {h})")
        
        # Extract ROI
        roi_image = screenshot[y:y+h, x:x+w]
        
        # Apply CardRefiner
        refined_x, refined_y, refined_w, refined_h = CardRefiner.refine_card_region(roi_image)
        
        # Convert back to screenshot coordinates
        final_x = x + refined_x
        final_y = y + refined_y
        final_w = refined_w
        final_h = refined_h
        
        refined_positions.append((final_x, final_y, final_w, final_h))
        
        print(f"Refined position: ({final_x}, {final_y}, {final_w}, {final_h})")
        
        # Calculate area change
        coarse_area = w * h
        refined_area = final_w * final_h
        area_change = ((refined_area - coarse_area) / coarse_area) * 100
        print(f"Area change: {area_change:+.1f}%")
    
    # Create result dict
    result = {
        'coarse_positions': coarse_positions,
        'refined_positions': refined_positions,
        'success': len(refined_positions) >= 2,
        'confidence': coarse_result.get('confidence', 0.0)
    }
    
    print("\n=== TWO-STAGE PIPELINE RESULTS ===")
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Detection method: two_stage_pipeline_refined")
    print(f"Pipeline stage: refined")
    print(f"Refinement applied: True")
    
    # Compare coarse vs refined positions
    coarse_positions = result['coarse_positions']
    refined_positions = result['refined_positions']
    
    print(f"\n=== COARSE vs REFINED COMPARISON ===")
    for i in range(min(len(coarse_positions), len(refined_positions))):
        coarse = coarse_positions[i]
        refined = refined_positions[i]
        
        print(f"Card {i+1}:")
        print(f"  Coarse:  ({coarse[0]}, {coarse[1]}, {coarse[2]}, {coarse[3]})")
        print(f"  Refined: ({refined[0]}, {refined[1]}, {refined[2]}, {refined[3]})")
        
        # Calculate area change
        coarse_area = coarse[2] * coarse[3]
        refined_area = refined[2] * refined[3]
        area_change = (refined_area - coarse_area) / coarse_area * 100
        print(f"  Area change: {area_change:+.1f}%")
    
    # Save refined card cutouts
    print(f"\n=== SAVING REFINED CUTOUTS ===")
    for i, (x, y, w, h) in enumerate(refined_positions):
        refined_card = screenshot[y:y+h, x:x+w]
        output_path = f"/mnt/d/cursor bots/arena_bot_project/debug_frames/TWO_STAGE_REFINED_Card{i+1}.png"
        cv2.imwrite(output_path, refined_card)
        print(f"Saved refined Card {i+1} to: {output_path}")
        print(f"  Size: {w}x{h}, Aspect ratio: {w/h:.3f}")
    
    # Check method recommendations from coarse result
    method_recs = coarse_result.get('method_recommendations', [])
    print(f"\n=== METHOD RECOMMENDATIONS ===")
    for i, (method, confidence) in enumerate(method_recs):
        print(f"Card {i+1}: {method} (confidence: {confidence:.3f})")
    
    # Show optimization regions available from coarse result
    optimized_regions = coarse_result.get('optimized_regions', {})
    if optimized_regions:
        print(f"\n=== OPTIMIZATION REGIONS AVAILABLE ===")
        for card_key, optimizations in optimized_regions.items():
            print(f"{card_key}:")
            for method, region in optimizations.items():
                print(f"  {method}: {region}")
    
    return result


if __name__ == "__main__":
    test_two_stage_pipeline()