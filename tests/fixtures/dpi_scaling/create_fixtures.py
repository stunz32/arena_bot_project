#!/usr/bin/env python3
"""
Create synthetic DPI test fixtures with known card positions.

This script generates test images at different DPI scaling factors with
precisely positioned card regions for coordinate detection testing.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any

def create_synthetic_draft_image(width: int, height: int, 
                                card_positions: List[Tuple[int, int, int, int]],
                                output_path: str, metadata_path: str) -> bool:
    """
    Create a synthetic draft image with card regions at specified positions.
    
    Args:
        width: Image width
        height: Image height
        card_positions: List of (x, y, width, height) for each card
        output_path: Path to save the image
        metadata_path: Path to save position metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import cv2
        
        # Create base image (dark background like Hearthstone)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (20, 25, 35)  # Dark blue-gray background
        
        metadata = {
            'image_dimensions': [width, height],
            'card_positions': [],
            'creation_info': 'Synthetic DPI test fixture'
        }
        
        # Draw card regions
        for i, (x, y, w, h) in enumerate(card_positions):
            # Card background (brownish like Hearthstone cards)
            cv2.rectangle(image, (x, y), (x + w, y + h), (139, 101, 73), -1)
            
            # Card border
            cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), 2)
            
            # Mana crystal (blue circle in top-left)
            mana_center = (x + 25, y + 25)
            cv2.circle(image, mana_center, 20, (0, 100, 255), -1)
            cv2.circle(image, mana_center, 20, (255, 255, 255), 2)
            
            # Card art region (darker rectangle)
            art_x, art_y = x + 15, y + 50
            art_w, art_h = w - 30, h - 120
            cv2.rectangle(image, (art_x, art_y), (art_x + art_w, art_y + art_h), 
                         (80, 60, 40), -1)
            
            # Card name region (light rectangle at bottom)
            name_x, name_y = x + 10, y + h - 60
            name_w, name_h = w - 20, 40
            cv2.rectangle(image, (name_x, name_y), (name_x + name_w, name_y + name_h), 
                         (200, 180, 140), -1)
            
            # Add position to metadata
            metadata['card_positions'].append({
                'index': i,
                'position': [x, y, w, h],
                'mana_center': list(mana_center),
                'art_region': [art_x, art_y, art_w, art_h],
                'name_region': [name_x, name_y, name_w, name_h]
            })
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        
        # Save metadata
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Created fixture: {output_path}")
        print(f"Saved metadata: {metadata_path}")
        return True
        
    except ImportError:
        print("ERROR: OpenCV not available for fixture creation")
        return False
    except Exception as e:
        print(f"ERROR creating fixture: {e}")
        return False


def create_dpi_fixture_set():
    """Create a complete set of DPI test fixtures."""
    
    # Base positions for 1920x1080 (100% scaling)
    # These are realistic Arena draft card positions
    base_positions = [
        (735, 233, 450, 500),   # Left card
        (1235, 233, 450, 500),  # Center card  
        (1735, 233, 450, 500),  # Right card
    ]
    
    scaling_factors = [
        (1.0, 1920, 1080),    # 100% scaling
        (1.25, 1536, 864),    # 125% scaling
        (1.5, 1280, 720),     # 150% scaling
    ]
    
    fixtures_dir = os.path.dirname(__file__)
    
    success_count = 0
    total_count = len(scaling_factors)
    
    for scale, width, height in scaling_factors:
        # Scale down the positions for higher DPI
        scaled_positions = []
        for x, y, w, h in base_positions:
            scaled_x = int(x / scale)
            scaled_y = int(y / scale)
            scaled_w = int(w / scale)
            scaled_h = int(h / scale)
            scaled_positions.append((scaled_x, scaled_y, scaled_w, scaled_h))
        
        # Create file names
        scale_pct = int(scale * 100)
        image_name = f"draft_test_{scale_pct}pct.png"
        metadata_name = f"draft_test_{scale_pct}pct.json"
        
        image_path = os.path.join(fixtures_dir, image_name)
        metadata_path = os.path.join(fixtures_dir, metadata_name)
        
        # Create the fixture
        if create_synthetic_draft_image(width, height, scaled_positions, 
                                      image_path, metadata_path):
            success_count += 1
    
    print(f"\nDPI fixture creation complete: {success_count}/{total_count} successful")
    return success_count == total_count


if __name__ == "__main__":
    create_dpi_fixture_set()