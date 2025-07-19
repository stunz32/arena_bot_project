#!/usr/bin/env python3
"""
Card name database for user-friendly display.
Maps card codes to actual card names.
"""

# Card code to name mapping
CARD_NAMES = {
    # Test cards we're using
    'TOY_380': 'Toy Captain Tarim',
    'ULD_309': 'Dragonqueen Alexstrasza', 
    'TTN_042': 'Thassarian',
    
    # Common Arena cards (expanded database)
    'AT_001': 'Flame Lance',
    'AT_002': 'Effigy',
    'AT_003': 'Fallen Hero',
    'AT_004': 'Arcane Blast',
    'AT_005': 'Polymorph: Boar',
    
    # Legendary examples
    'EX1_001': 'Lightwarden',
    'EX1_002': 'The Black Knight',
    'EX1_046': 'Dark Iron Dwarf',
    
    # More examples for variety
    'CS2_234': 'Shadow Word: Pain',
    'CS2_235': 'Northshire Cleric',
    'CS2_236': 'Divine Spirit',
    
    # Neutral cards
    'CS2_142': 'Kobold Geomancer',
    'CS2_147': 'Gnomish Inventor',
    'CS2_151': 'Silver Hand Recruit',
    
    # Spell examples
    'CS2_025': 'Arcane Intellect',
    'CS2_029': 'Fireball',
    'CS2_032': 'Flamestrike',
}

def get_card_name(card_code: str) -> str:
    """
    Get the user-friendly name for a card code.
    
    Args:
        card_code: Hearthstone card code (e.g., 'TOY_380')
        
    Returns:
        User-friendly card name
    """
    # Remove premium suffix if present
    clean_code = card_code.replace('_premium', '')
    
    # Return name if found, otherwise return a cleaned-up version of the code
    if clean_code in CARD_NAMES:
        return CARD_NAMES[clean_code]
    else:
        # Make unknown cards more readable
        # Convert something like "UNK_123" to "Unknown Card (UNK_123)"
        return f"Unknown Card ({clean_code})"

def is_premium_card(card_code: str) -> bool:
    """Check if a card code represents a premium (golden) version."""
    return '_premium' in card_code.lower()

def format_card_display(card_code: str) -> str:
    """
    Format a card for display with name and premium indicator.
    
    Args:
        card_code: Card code to format
        
    Returns:
        Formatted string like "Fireball ✨" or "Fireball"
    """
    name = get_card_name(card_code)
    if is_premium_card(card_code):
        return f"{name} ✨"  # Golden star for premium cards
    return name