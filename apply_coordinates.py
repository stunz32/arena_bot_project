#!/usr/bin/env python3
"""
Coordinate Integration Script for Arena Bot
Applies captured coordinates from visual picker to the main bot
"""

import json
import os
import shutil
from datetime import datetime

def apply_coordinates_to_bot():
    """Apply captured coordinates to the main Arena Bot"""
    
    # Paths
    coord_file = "captured_coordinates.json"
    settings_file = "coordinate_settings.json" 
    main_bot_file = "integrated_arena_bot_gui.py"
    
    # Check if coordinates were captured
    if not os.path.exists(coord_file):
        print("❌ No captured coordinates found!")
        print("Please run visual_coordinate_picker.py first to capture coordinates.")
        return False
    
    # Load coordinates
    try:
        with open(coord_file, 'r') as f:
            coord_data = json.load(f)
        
        print("✅ Coordinates loaded successfully!")
        print(f"Screen resolution: {coord_data['screen_resolution']}")
        print(f"Captured: {coord_data['timestamp']}")
        print(f"Cards captured: {len(coord_data['card_coordinates'])}")
        
        # Display coordinates
        for card in coord_data['card_coordinates']:
            print(f"  Card {card['card_number']}: x={card['x']}, y={card['y']}, w={card['width']}, h={card['height']}")
        
    except Exception as e:
        print(f"❌ Failed to load coordinates: {e}")
        return False
    
    # Backup main bot file
    try:
        backup_path = main_bot_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(main_bot_file, backup_path)
        print(f"✅ Main bot backed up to: {backup_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not backup main bot file: {e}")
    
    # Read main bot file
    try:
        with open(main_bot_file, 'r') as f:
            bot_content = f.read()
    except Exception as e:
        print(f"❌ Failed to read main bot file: {e}")
        return False
    
    # Prepare new coordinates code
    coordinates_list = [card['coordinates_list'] for card in coord_data['card_coordinates']]
    new_coordinates_code = f"""        # Coordinates captured from visual picker - {coord_data['timestamp']}
        # Screen resolution: {coord_data['screen_resolution']}
        card_coordinates = {coordinates_list}"""
    
    # Find and replace coordinates in the main bot
    try:
        # Look for existing coordinate definitions to replace
        import re
        
        # Pattern to find card coordinates assignment
        coord_pattern = r'card_coordinates\s*=\s*\[.*?\]'
        
        if re.search(coord_pattern, bot_content, re.DOTALL):
            # Replace existing coordinates
            bot_content = re.sub(coord_pattern, f"card_coordinates = {coordinates_list}", bot_content, flags=re.DOTALL)
            print("✅ Updated existing coordinates in main bot")
        else:
            # Look for a place to inject coordinates (e.g., in a method)
            # This is a fallback - we'll add coordinates at the beginning of analyze_screenshot method
            analyze_pattern = r'(def analyze_screenshot.*?\n)'
            if re.search(analyze_pattern, bot_content):
                replacement = f"\\1        # Auto-injected coordinates from visual picker\n        card_coordinates = {coordinates_list}\n"
                bot_content = re.sub(analyze_pattern, replacement, bot_content)
                print("✅ Injected coordinates into analyze_screenshot method")
            else:
                print("⚠️ Could not find suitable location to inject coordinates")
                print("You may need to manually add coordinates to the bot")
                return False
        
        # Write updated bot file
        with open(main_bot_file, 'w') as f:
            f.write(bot_content)
        
        print("✅ Main bot updated with new coordinates!")
        
    except Exception as e:
        print(f"❌ Failed to update main bot: {e}")
        return False
    
    # Create a simple test script to verify coordinates
    test_script = f"""#!/usr/bin/env python3
# Quick test of captured coordinates
import json
from PIL import ImageGrab

# Load coordinates
with open('captured_coordinates.json', 'r') as f:
    data = json.load(f)

print("Testing captured coordinates...")
print(f"Screen resolution: {{data['screen_resolution']}}")

# Take screenshot and test regions
screenshot = ImageGrab.grab()
for card in data['card_coordinates']:
    x, y, w, h = card['coordinates_list']
    region = screenshot.crop((x, y, x + w, y + h))
    test_path = f"test_region_card_{{card['card_number']}}.png"
    region.save(test_path)
    print(f"Saved test region {{card['card_number']}}: {{test_path}}")

print("Test regions saved! Check the files to verify coordinates are correct.")
"""
    
    test_file = "test_coordinates.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"✅ Test script created: {test_file}")
    print("")
    print("🎯 Integration Complete!")
    print("=======================")
    print("Your Arena Bot has been updated with the captured coordinates.")
    print("")
    print("Next steps:")
    print("1. Test the coordinates: python test_coordinates.py")
    print("2. Run the main bot: python integrated_arena_bot_gui.py")
    print("3. If coordinates need adjustment, run visual_coordinate_picker.py again")
    
    return True

if __name__ == "__main__":
    apply_coordinates_to_bot()