#!/usr/bin/env python3
"""
Create synthetic card detection test fixtures.

This script generates cropped card images with known identities for testing
histogram and template matching accuracy.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

def create_synthetic_card(width: int, height: int, card_info: Dict[str, Any], 
                         output_path: str) -> bool:
    """
    Create a synthetic card image with distinctive features.
    
    Args:
        width: Card width
        height: Card height
        card_info: Card metadata (name, mana_cost, colors, etc.)
        output_path: Path to save the image
        
    Returns:
        True if successful
    """
    try:
        import cv2
        
        # Create base card image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Card background color based on rarity
        rarity_colors = {
            'common': (139, 101, 73),     # Brown
            'rare': (0, 112, 221),        # Blue  
            'epic': (163, 53, 238),       # Purple
            'legendary': (255, 128, 0)    # Orange
        }
        
        bg_color = rarity_colors.get(card_info.get('rarity', 'common'), (139, 101, 73))
        image[:] = bg_color
        
        # Card border
        cv2.rectangle(image, (0, 0), (width-1, height-1), (200, 200, 200), 2)
        
        # Mana crystal (top-left corner)
        mana_cost = card_info.get('mana_cost', 1)
        mana_center = (25, 25)
        mana_radius = 20
        
        # Mana crystal color based on cost
        if mana_cost <= 2:
            mana_color = (255, 255, 255)  # White for low cost
        elif mana_cost <= 4:
            mana_color = (0, 255, 255)    # Cyan for medium cost
        else:
            mana_color = (255, 0, 255)    # Magenta for high cost
            
        cv2.circle(image, mana_center, mana_radius, mana_color, -1)
        cv2.circle(image, mana_center, mana_radius, (255, 255, 255), 2)
        
        # Mana cost text
        cv2.putText(image, str(mana_cost), (mana_center[0]-8, mana_center[1]+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Card art region (distinctive pattern based on card name)
        art_x, art_y = 15, 50
        art_w, art_h = width - 30, height - 120
        
        # Create a simple pattern based on card name hash
        name_hash = hash(card_info.get('name', 'unknown')) % 1000
        pattern_color = (
            (name_hash * 123) % 256,
            (name_hash * 456) % 256, 
            (name_hash * 789) % 256
        )
        
        cv2.rectangle(image, (art_x, art_y), (art_x + art_w, art_y + art_h), 
                     pattern_color, -1)
        
        # Add some distinctive shapes for histogram matching
        if name_hash % 3 == 0:
            # Add circles
            for i in range(3):
                cx = art_x + (i + 1) * art_w // 4
                cy = art_y + art_h // 2
                cv2.circle(image, (cx, cy), 15, (255, 255, 255), -1)
        elif name_hash % 3 == 1:
            # Add rectangles  
            for i in range(2):
                rx = art_x + (i + 1) * art_w // 3
                ry = art_y + art_h // 3
                cv2.rectangle(image, (rx-10, ry-10), (rx+10, ry+10), (255, 255, 255), -1)
        else:
            # Add lines
            cv2.line(image, (art_x, art_y), (art_x + art_w, art_y + art_h), (255, 255, 255), 3)
            cv2.line(image, (art_x + art_w, art_y), (art_x, art_y + art_h), (255, 255, 255), 3)
        
        # Card name region
        name_x, name_y = 10, height - 60
        name_w, name_h = width - 20, 40
        cv2.rectangle(image, (name_x, name_y), (name_x + name_w, name_y + name_h), 
                     (200, 180, 140), -1)
        
        # Card name text (truncated to fit)
        name = card_info.get('name', 'Test Card')[:12]
        cv2.putText(image, name, (name_x + 5, name_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        
        print(f"Created card fixture: {output_path}")
        return True
        
    except ImportError:
        print("ERROR: OpenCV not available for fixture creation")
        return False
    except Exception as e:
        print(f"ERROR creating card fixture: {e}")
        return False


def create_negative_control(width: int, height: int, control_type: str, 
                           output_path: str) -> bool:
    """
    Create negative control images that should NOT match any cards.
    
    Args:
        width: Image width
        height: Image height
        control_type: Type of negative control
        output_path: Path to save the image
        
    Returns:
        True if successful
    """
    try:
        import cv2
        
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if control_type == "random_noise":
            # Random noise pattern
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
        elif control_type == "solid_color":
            # Solid color that's not card-like
            image[:] = (128, 64, 192)  # Purple
            
        elif control_type == "geometric_pattern":
            # Geometric pattern unlike cards
            image[:] = (50, 50, 50)  # Dark gray base
            
            # Draw grid pattern
            for i in range(0, width, 20):
                cv2.line(image, (i, 0), (i, height), (255, 255, 255), 1)
            for j in range(0, height, 20):
                cv2.line(image, (0, j), (width, j), (255, 255, 255), 1)
                
        elif control_type == "mana_gem_fake":
            # Similar to mana gem but wrong colors/position
            image[:] = (100, 50, 150)  # Purple background
            
            # Fake mana gem in wrong position (center)
            center = (width // 2, height // 2)
            cv2.circle(image, center, 25, (255, 0, 0), -1)  # Red instead of blue
            cv2.circle(image, center, 25, (255, 255, 255), 2)
            
        else:
            # Default: gradient pattern
            for y in range(height):
                color_val = int(255 * y / height)
                image[y, :] = (color_val, 255 - color_val, 128)
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        
        print(f"Created negative control: {output_path}")
        return True
        
    except ImportError:
        print("ERROR: OpenCV not available for fixture creation")
        return False
    except Exception as e:
        print(f"ERROR creating negative control: {e}")
        return False


def create_detection_fixture_set():
    """Create a complete set of detection test fixtures."""
    
    fixtures_dir = os.path.dirname(__file__)
    
    # Test cards with known properties
    test_cards = [
        {
            'id': 'fireball_test',
            'name': 'Fireball',
            'mana_cost': 4,
            'rarity': 'common',
            'card_type': 'spell'
        },
        {
            'id': 'polymorph_test', 
            'name': 'Polymorph',
            'mana_cost': 4,
            'rarity': 'common',
            'card_type': 'spell'
        },
        {
            'id': 'flamestrike_test',
            'name': 'Flamestrike', 
            'mana_cost': 7,
            'rarity': 'common',
            'card_type': 'spell'
        },
        {
            'id': 'arcane_intellect_test',
            'name': 'Arcane Intellect',
            'mana_cost': 3,
            'rarity': 'common', 
            'card_type': 'spell'
        },
        {
            'id': 'legenday_minion_test',
            'name': 'Test Legendary',
            'mana_cost': 8,
            'rarity': 'legendary',
            'card_type': 'minion'
        }
    ]
    
    # Negative controls
    negative_controls = [
        'random_noise',
        'solid_color', 
        'geometric_pattern',
        'mana_gem_fake'
    ]
    
    success_count = 0
    total_count = len(test_cards) + len(negative_controls)
    
    # Create positive test cards
    for card in test_cards:
        image_path = os.path.join(fixtures_dir, f"{card['id']}.png")
        metadata_path = os.path.join(fixtures_dir, f"{card['id']}.json")
        
        # Create card image (typical Arena card crop size)
        if create_synthetic_card(200, 280, card, image_path):
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    **card,
                    'fixture_type': 'positive_control',
                    'expected_match': True,
                    'image_dimensions': [200, 280]
                }, f, indent=2)
            success_count += 1
    
    # Create negative controls
    for control_type in negative_controls:
        image_path = os.path.join(fixtures_dir, f"negative_{control_type}.png")
        metadata_path = os.path.join(fixtures_dir, f"negative_{control_type}.json")
        
        if create_negative_control(200, 280, control_type, image_path):
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'id': f'negative_{control_type}',
                    'name': f'Negative Control: {control_type.title()}',
                    'fixture_type': 'negative_control',
                    'expected_match': False,
                    'control_type': control_type,
                    'image_dimensions': [200, 280]
                }, f, indent=2)
            success_count += 1
    
    print(f"\nDetection fixture creation complete: {success_count}/{total_count} successful")
    return success_count == total_count


if __name__ == "__main__":
    create_detection_fixture_set()