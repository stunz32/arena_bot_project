"""
Archetype Configuration System

Defines the "personality" of the AI for each playstyle.
Each archetype has different weights for the dimensional scores,
creating distinct strategic preferences and recommendations.
"""

from typing import Dict, Any


# Core archetype scoring weights - the "DNA" of each playstyle
ARCHETYPE_WEIGHTS = {
    "Aggro": {
        "base_value": 0.8,           # Slightly lower emphasis on raw power
        "tempo_score": 1.5,          # HIGH - immediate board impact crucial
        "value_score": 0.3,          # LOW - don't care about long-term value
        "synergy_score": 0.7,        # MEDIUM - some tribal synergies matter
        "curve_score": 1.2,          # HIGH - smooth curve essential
        "re_draftability_score": 1.0, # MEDIUM - some flexibility needed
        "greed_score": 0.4,          # LOW - avoid risky/specialized cards
    },
    
    "Tempo": {
        "base_value": 1.0,           # Balanced - good cards matter
        "tempo_score": 1.3,          # HIGH - board control priority
        "value_score": 0.8,          # MEDIUM - some card advantage useful
        "synergy_score": 0.9,        # MEDIUM-HIGH - synergies create tempo
        "curve_score": 1.1,          # HIGH - curve still important
        "re_draftability_score": 0.9, # MEDIUM - some flexibility
        "greed_score": 0.6,          # MEDIUM-LOW - moderate risk tolerance
    },
    
    "Control": {
        "base_value": 1.2,           # HIGH - card quality crucial
        "tempo_score": 0.6,          # LOW - early board less important
        "value_score": 1.4,          # VERY HIGH - card advantage key
        "synergy_score": 0.8,        # MEDIUM - some synergies useful
        "curve_score": 0.7,          # LOW - can afford gaps in curve
        "re_draftability_score": 0.7, # MEDIUM-LOW - more committed picks
        "greed_score": 1.1,          # HIGH - willing to take powerful/risky cards
    },
    
    "Synergy": {
        "base_value": 0.9,           # MEDIUM - raw power less important
        "tempo_score": 0.8,          # MEDIUM - some board presence needed
        "value_score": 1.0,          # MEDIUM - balanced approach
        "synergy_score": 1.6,        # VERY HIGH - synergies define the deck
        "curve_score": 0.9,          # MEDIUM - curve secondary to synergies
        "re_draftability_score": 0.5, # LOW - committed to synergy packages
        "greed_score": 1.3,          # HIGH - willing to risk for synergy payoffs
    },
    
    "Attrition": {
        "base_value": 1.1,           # HIGH - efficient cards important
        "tempo_score": 0.7,          # MEDIUM-LOW - slower start acceptable
        "value_score": 1.3,          # VERY HIGH - maximize card advantage
        "synergy_score": 0.6,        # MEDIUM-LOW - focus on individual power
        "curve_score": 0.8,          # MEDIUM - can afford some gaps
        "re_draftability_score": 0.8, # MEDIUM - some flexibility useful
        "greed_score": 0.9,          # MEDIUM - calculated risks only
    },
    
    "Balanced": {
        "base_value": 1.0,           # MEDIUM - baseline importance
        "tempo_score": 1.0,          # MEDIUM - balanced board presence
        "value_score": 1.0,          # MEDIUM - balanced card advantage
        "synergy_score": 1.0,        # MEDIUM - opportunistic synergies
        "curve_score": 1.0,          # MEDIUM - solid curve preferred
        "re_draftability_score": 1.0, # MEDIUM - maximum flexibility
        "greed_score": 0.8,          # MEDIUM-LOW - conservative approach
    }
}


# Archetype descriptions for user interface
ARCHETYPE_DESCRIPTIONS = {
    "Aggro": {
        "name": "Aggro",
        "subtitle": "Fast & Furious",
        "description": "Win quickly with efficient minions and direct damage. Prioritizes low-cost cards and immediate board impact.",
        "playstyle": "Aggressive",
        "complexity": "Low",
        "win_condition": "Deal 30 damage before turn 8",
        "key_stats": ["Low mana curve", "High minion count", "Direct damage"],
        "hero_preferences": ["Hunter", "Demon Hunter", "Warlock"],
        "example_cards": ["Low-cost minions", "Charge creatures", "Direct damage spells"]
    },
    
    "Tempo": {
        "name": "Tempo", 
        "subtitle": "Board Control",
        "description": "Maintain board control through efficient trades and pressure. Balances aggression with value.",
        "playstyle": "Proactive",
        "complexity": "Medium",
        "win_condition": "Sustained board pressure",
        "key_stats": ["Efficient minions", "Removal spells", "Card synergies"],
        "hero_preferences": ["Mage", "Rogue", "Paladin"],
        "example_cards": ["Efficient minions", "Cheap removal", "Tempo swings"]
    },
    
    "Control": {
        "name": "Control",
        "subtitle": "Late Game Power", 
        "description": "Survive early pressure and win with powerful late-game cards. Values card quality over speed.",
        "playstyle": "Reactive",
        "complexity": "High",
        "win_condition": "Outlast opponent with superior cards",
        "key_stats": ["High-value cards", "Board clears", "Card draw"],
        "hero_preferences": ["Warrior", "Priest", "Mage"],
        "example_cards": ["Board clears", "Card draw", "Win conditions"]
    },
    
    "Synergy": {
        "name": "Synergy",
        "subtitle": "Combo Power",
        "description": "Build around specific card interactions and tribal synergies for explosive turns.",
        "playstyle": "Synergistic", 
        "complexity": "High",
        "win_condition": "Synergy package payoffs",
        "key_stats": ["Tribal synergies", "Combo pieces", "Package density"],
        "hero_preferences": ["Depends on available synergies"],
        "example_cards": ["Tribal cards", "Combo enablers", "Synergy payoffs"]
    },
    
    "Attrition": {
        "name": "Attrition",
        "subtitle": "Resource Wars",
        "description": "Win through superior resource management and card advantage. Outlasts opponents.",
        "playstyle": "Defensive",
        "complexity": "Medium-High", 
        "win_condition": "Resource advantage in late game",
        "key_stats": ["Card advantage", "Efficient removal", "Sustainability"],
        "hero_preferences": ["Priest", "Warlock", "Warrior"],
        "example_cards": ["Card draw", "Efficient removal", "Value engines"]
    },
    
    "Balanced": {
        "name": "Balanced",
        "subtitle": "Adaptive Strategy",
        "description": "Flexible approach that adapts to draft offers. Good for beginners or uncertain metas.",
        "playstyle": "Adaptive",
        "complexity": "Low-Medium",
        "win_condition": "Situational adaptation",
        "key_stats": ["Solid curve", "Good cards", "Flexibility"],
        "hero_preferences": ["Any"],
        "example_cards": ["Generally strong cards", "Flexible options", "Solid minions"]
    }
}


# Ideal deck metrics for each archetype (used for conformance scoring)
ARCHETYPE_IDEALS = {
    "Aggro": {
        "average_mana_cost": 2.8,    # Very low curve
        "minion_count": 22,          # Creature-heavy
        "spell_count": 8,            # Mostly direct damage/buffs
        "curve_distribution": {       # Ideal mana curve
            1: 4, 2: 7, 3: 6, 4: 4, 5: 3, 6: 2, 7: 1, 8: 1, 9: 0, 10: 0
        },
        "required_keywords": ["Charge", "Haste", "Rush"],
        "avoid_keywords": ["Taunt", "Lifesteal"] 
    },
    
    "Tempo": {
        "average_mana_cost": 3.2,
        "minion_count": 18,
        "spell_count": 12,
        "curve_distribution": {
            1: 2, 2: 5, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 1, 10: 0
        },
        "required_keywords": ["Battlecry", "Rush"],
        "avoid_keywords": []
    },
    
    "Control": {
        "average_mana_cost": 4.1,
        "minion_count": 14,
        "spell_count": 16,
        "curve_distribution": {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4, 7: 4, 8: 3, 9: 2, 10: 1
        },
        "required_keywords": ["Taunt", "Lifesteal", "Card Draw"],
        "avoid_keywords": ["Charge"]
    },
    
    "Synergy": {
        "average_mana_cost": 3.5,
        "minion_count": 20,
        "spell_count": 10,
        "curve_distribution": {
            1: 3, 2: 4, 3: 5, 4: 6, 5: 4, 6: 3, 7: 2, 8: 2, 9: 1, 10: 0
        },
        "required_keywords": ["Tribal", "Synergy"],
        "avoid_keywords": []
    },
    
    "Attrition": {
        "average_mana_cost": 3.8,
        "minion_count": 16,
        "spell_count": 14,
        "curve_distribution": {
            1: 2, 2: 3, 3: 4, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 2, 10: 1
        },
        "required_keywords": ["Card Draw", "Lifesteal", "Value"],
        "avoid_keywords": ["Charge"]
    },
    
    "Balanced": {
        "average_mana_cost": 3.4,
        "minion_count": 18,
        "spell_count": 12,
        "curve_distribution": {
            1: 2, 2: 4, 3: 5, 4: 5, 5: 4, 6: 3, 7: 3, 8: 2, 9: 1, 10: 1
        },
        "required_keywords": [],
        "avoid_keywords": []
    }
}


# Hero class preferences for each archetype
HERO_ARCHETYPE_AFFINITY = {
    "WARRIOR": {
        "Aggro": 0.7, "Tempo": 0.8, "Control": 0.9, 
        "Synergy": 0.6, "Attrition": 0.8, "Balanced": 0.8
    },
    "MAGE": {
        "Aggro": 0.6, "Tempo": 0.9, "Control": 0.9,
        "Synergy": 0.8, "Attrition": 0.7, "Balanced": 0.8
    },
    "HUNTER": {
        "Aggro": 0.9, "Tempo": 0.8, "Control": 0.4,
        "Synergy": 0.7, "Attrition": 0.5, "Balanced": 0.7
    },
    "PRIEST": {
        "Aggro": 0.3, "Tempo": 0.6, "Control": 0.9,
        "Synergy": 0.7, "Attrition": 0.9, "Balanced": 0.7
    },
    "WARLOCK": {
        "Aggro": 0.8, "Tempo": 0.7, "Control": 0.6,
        "Synergy": 0.8, "Attrition": 0.8, "Balanced": 0.8
    },
    "ROGUE": {
        "Aggro": 0.8, "Tempo": 0.9, "Control": 0.5,
        "Synergy": 0.8, "Attrition": 0.6, "Balanced": 0.7
    },
    "SHAMAN": {
        "Aggro": 0.7, "Tempo": 0.8, "Control": 0.7,
        "Synergy": 0.9, "Attrition": 0.7, "Balanced": 0.8
    },
    "PALADIN": {
        "Aggro": 0.8, "Tempo": 0.9, "Control": 0.7,
        "Synergy": 0.8, "Attrition": 0.7, "Balanced": 0.8
    },
    "DRUID": {
        "Aggro": 0.6, "Tempo": 0.7, "Control": 0.8,
        "Synergy": 0.8, "Attrition": 0.8, "Balanced": 0.8
    },
    "DEMONHUNTER": {
        "Aggro": 0.9, "Tempo": 0.8, "Control": 0.4,
        "Synergy": 0.7, "Attrition": 0.5, "Balanced": 0.7
    }
}


def get_archetype_weights(archetype: str) -> Dict[str, float]:
    """Get scoring weights for a specific archetype."""
    return ARCHETYPE_WEIGHTS.get(archetype, ARCHETYPE_WEIGHTS["Balanced"])


def get_archetype_description(archetype: str) -> Dict[str, Any]:
    """Get user-friendly description for an archetype.""" 
    return ARCHETYPE_DESCRIPTIONS.get(archetype, ARCHETYPE_DESCRIPTIONS["Balanced"])


def get_archetype_ideals(archetype: str) -> Dict[str, Any]:
    """Get ideal deck metrics for conformance scoring."""
    return ARCHETYPE_IDEALS.get(archetype, ARCHETYPE_IDEALS["Balanced"])


def get_hero_archetype_affinity(hero: str, archetype: str) -> float:
    """Get how well a hero class fits an archetype (0.0 to 1.0)."""
    hero_affinities = HERO_ARCHETYPE_AFFINITY.get(hero.upper(), {})
    return hero_affinities.get(archetype, 0.7)  # Default to 0.7 if unknown


def suggest_archetype_for_hero(hero: str) -> str:
    """Suggest the best archetype for a given hero class."""
    hero_affinities = HERO_ARCHETYPE_AFFINITY.get(hero.upper(), {})
    if not hero_affinities:
        return "Balanced"
    
    best_archetype = max(hero_affinities.items(), key=lambda x: x[1])
    return best_archetype[0]