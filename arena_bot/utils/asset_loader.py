"""
Simple asset loading utilities for Arena Bot.

Handles loading card images, templates, and data files.
Following CLAUDE.md principles - minimal and focused.
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class AssetLoader:
    """Simple asset loading and caching system."""
    
    def __init__(self, assets_dir: Path = None):
        """
        Initialize asset loader.
        
        Args:
            assets_dir: Path to assets directory
        """
        if assets_dir is None:
            assets_dir = Path(__file__).parent.parent.parent / "assets"
        
        self.assets_dir = Path(assets_dir)
        self.logger = logging.getLogger(__name__)
        
        # Simple caches
        self._card_cache: Dict[str, np.ndarray] = {}
        self._template_cache: Dict[str, np.ndarray] = {}
        
        self.logger.info(f"AssetLoader initialized with assets_dir: {self.assets_dir}")
    
    def load_card_image(self, card_code: str, premium: bool = False) -> Optional[np.ndarray]:
        """
        Load a card image by card code.
        
        Args:
            card_code: Hearthstone card code (e.g., "AT_001")
            premium: Whether to load premium (golden) version
            
        Returns:
            OpenCV image array or None if not found
        """
        suffix = "_premium" if premium else ""
        filename = f"{card_code}{suffix}.png"
        cache_key = f"{card_code}_{suffix}"
        
        # Check cache first
        if cache_key in self._card_cache:
            return self._card_cache[cache_key]
        
        # Load from file
        card_path = self.assets_dir / "cards" / filename
        if not card_path.exists():
            self.logger.warning(f"Card image not found: {card_path}")
            return None
        
        try:
            image = cv2.imread(str(card_path))
            if image is not None:
                self._card_cache[cache_key] = image
                return image
            else:
                self.logger.warning(f"Failed to load card image: {card_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading card image {card_path}: {e}")
            return None
    
    def load_template(self, template_type: str, template_id: str) -> Optional[np.ndarray]:
        """
        Load a template image.
        
        Args:
            template_type: Type of template ("mana", "rarity", "ui")
            template_id: Template identifier (e.g., "mana0", "rarity1")
            
        Returns:
            OpenCV image array or None if not found
        """
        cache_key = f"{template_type}_{template_id}"
        
        # Check cache first
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # Load from file
        template_path = self.assets_dir / "templates" / template_type / f"{template_id}.png"
        if not template_path.exists():
            self.logger.warning(f"Template not found: {template_path}")
            return None
        
        try:
            image = cv2.imread(str(template_path))
            if image is not None:
                self._template_cache[cache_key] = image
                return image
            else:
                self.logger.warning(f"Failed to load template: {template_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading template {template_path}: {e}")
            return None
    
    def load_mana_templates(self) -> Dict[int, np.ndarray]:
        """
        Load all mana cost templates (0-9).
        
        Returns:
            Dictionary mapping mana cost to template image
        """
        mana_templates = {}
        
        for mana_cost in range(10):
            template = self.load_template("mana", f"mana{mana_cost}")
            if template is not None:
                mana_templates[mana_cost] = template
        
        self.logger.info(f"Loaded {len(mana_templates)} mana templates")
        return mana_templates
    
    def load_rarity_templates(self) -> Dict[int, np.ndarray]:
        """
        Load all rarity templates (0-3).
        
        Returns:
            Dictionary mapping rarity to template image
        """
        rarity_templates = {}
        
        for rarity in range(4):
            template = self.load_template("rarity", f"rarity{rarity}")
            if template is not None:
                rarity_templates[rarity] = template
        
        self.logger.info(f"Loaded {len(rarity_templates)} rarity templates")
        return rarity_templates
    
    def load_ui_template(self, template_name: str) -> Optional[np.ndarray]:
        """
        Load a UI template.
        
        Args:
            template_name: Name of the template (without .png extension)
            
        Returns:
            OpenCV image array or None if not found
        """
        return self.load_template("ui", template_name)
    
    def load_json_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON data file.
        
        Args:
            filename: Name of the JSON file
            
        Returns:
            Parsed JSON data or None if not found
        """
        data_path = self.assets_dir / "data" / filename
        
        if not data_path.exists():
            self.logger.warning(f"JSON data file not found: {data_path}")
            return None
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded JSON data from {data_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading JSON data {data_path}: {e}")
            return None
    
    def get_available_cards(self) -> List[str]:
        """
        Get list of available card codes.
        
        Returns:
            List of card codes
        """
        cards_dir = self.assets_dir / "cards"
        card_codes = []
        
        if cards_dir.exists():
            for card_file in cards_dir.glob("*.png"):
                if not card_file.stem.endswith("_premium"):
                    card_codes.append(card_file.stem)
        
        self.logger.info(f"Found {len(card_codes)} available card codes")
        return sorted(card_codes)
    
    def load_all_cards(self, max_cards: Optional[int] = None, 
                       exclude_prefixes: Optional[List[str]] = None,
                       include_premium: bool = True) -> Dict[str, np.ndarray]:
        """
        Load all available card images for batch processing.
        
        Args:
            max_cards: Maximum number of cards to load (None for all)
            exclude_prefixes: List of card code prefixes to exclude (e.g., ['HERO_', 'BG_'])
            include_premium: Whether to include premium (golden) card variants
            
        Returns:
            Dictionary mapping card codes to OpenCV image arrays
        """
        cards_dir = self.assets_dir / "cards"
        if not cards_dir.exists():
            self.logger.warning(f"Cards directory not found: {cards_dir}")
            return {}
        
        # Default exclusions for non-playable cards
        if exclude_prefixes is None:
            exclude_prefixes = ['HERO_', 'BG_', 'TB_', 'KARA_', 'CHEAT_']
        
        card_images = {}
        card_count = 0
        
        # Get all PNG files in cards directory
        all_card_files = list(cards_dir.glob("*.png"))
        self.logger.info(f"Found {len(all_card_files)} card image files")
        
        for card_file in all_card_files:
            # Check if we've reached the limit
            if max_cards and card_count >= max_cards:
                break
            
            card_code = card_file.stem
            
            # Skip excluded prefixes
            if any(card_code.startswith(prefix) for prefix in exclude_prefixes):
                continue
            
            # Skip premium cards if not requested
            if not include_premium and card_code.endswith('_premium'):
                continue
            
            try:
                # Try to load from cache first
                if card_code in self._card_cache:
                    image = self._card_cache[card_code]
                else:
                    # Load image from disk
                    image = cv2.imread(str(card_file))
                    if image is not None:
                        # Cache for future use
                        self._card_cache[card_code] = image
                
                if image is not None:
                    card_images[card_code] = image
                    card_count += 1
                    
                    # Progress reporting for large batches
                    if card_count % 1000 == 0:
                        self.logger.info(f"Loaded {card_count} cards...")
                else:
                    self.logger.warning(f"Failed to load card image: {card_file}")
                    
            except Exception as e:
                self.logger.error(f"Error loading card {card_file}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(card_images)} card images")
        return card_images
    
    def clear_cache(self):
        """Clear all cached assets."""
        self._card_cache.clear()
        self._template_cache.clear()
        self.logger.info("Asset cache cleared")


# Global asset loader instance
_asset_loader = None


def get_asset_loader() -> AssetLoader:
    """
    Get the global asset loader instance.
    
    Returns:
        AssetLoader instance
    """
    global _asset_loader
    if _asset_loader is None:
        _asset_loader = AssetLoader()
    return _asset_loader