#!/usr/bin/env python3

import cv2
import numpy as np
from arena_bot.core.card_refiner import CardRefiner


def test_refiner_on_existing_cutouts():
    """Test the CardRefiner on our existing coarse cutouts."""
    
    cutout_files = [
        "/mnt/d/cursor bots/arena_bot_project/debug_frames/SMARTDETECTOR_TEST_Card1.png",
        "/mnt/d/cursor bots/arena_bot_project/debug_frames/SMARTDETECTOR_TEST_Card2.png", 
        "/mnt/d/cursor bots/arena_bot_project/debug_frames/SMARTDETECTOR_TEST_Card3.png"
    ]
    
    for i, cutout_path in enumerate(cutout_files, 1):
        print(f"\n=== Testing Card {i} Refinement ===")
        
        # Load the coarse cutout from Stage 1
        roi_image = cv2.imread(cutout_path)
        if roi_image is None:
            print(f"Could not load {cutout_path}")
            continue
            
        print(f"Original ROI size: {roi_image.shape[1]}x{roi_image.shape[0]}")
        
        # Test the refinement with debug output (shape-finder approach)
        debug_base = f"/mnt/d/cursor bots/arena_bot_project/debug_frames/SHAPE_DEBUG_Card{i}"
        refined_box = CardRefiner.debug_refinement(roi_image, debug_base)
        
        x, y, w, h = refined_box
        print(f"Refined box: x={x}, y={y}, w={w}, h={h}")
        print(f"Refined aspect ratio: {w/h:.3f}")
        
        # Extract the refined card region
        refined_card = roi_image[y:y+h, x:x+w]
        
        # Save the refined cutout
        refined_path = f"/mnt/d/cursor bots/arena_bot_project/debug_frames/SHAPE_REFINED_CARD_{i}.png"
        cv2.imwrite(refined_path, refined_card)
        print(f"Saved shape-refined card to: {refined_path}")


if __name__ == "__main__":
    test_refiner_on_existing_cutouts()