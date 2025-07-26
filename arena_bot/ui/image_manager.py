#!/usr/bin/env python3
"""
PhotoImage Reference Manager for Arena Bot GUI.

Solves the classic Tkinter garbage collection issue by properly managing
PhotoImage object references and providing utilities for asset integration.
"""

import tkinter as tk
from tkinter import ImageTk
from PIL import Image
import cv2
import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple
from pathlib import Path


class PhotoImageManager:
    """
    Centralized manager for PhotoImage objects to prevent garbage collection.
    
    This class solves the 'pyimage1 doesn't exist' error by maintaining
    proper references to all PhotoImage objects and providing utilities
    for converting between different image formats.
    """
    
    def __init__(self):
        """Initialize the PhotoImage manager."""
        self.logger = logging.getLogger(__name__)
        
        # Main storage for PhotoImage references (prevents garbage collection)
        self._image_refs: Dict[str, ImageTk.PhotoImage] = {}
        
        # Counter for generating unique keys
        self._counter = 0
        
        self.logger.info("PhotoImageManager initialized")
    
    def create_photo_image(self, 
                          source: Union[np.ndarray, Image.Image, str, Path],
                          key: Optional[str] = None,
                          size: Optional[Tuple[int, int]] = None) -> Tuple[ImageTk.PhotoImage, str]:
        """
        Create a PhotoImage from various source types with proper reference management.
        
        Args:
            source: Image source (OpenCV array, PIL Image, file path)
            key: Optional key for storing reference (auto-generated if None)
            size: Optional resize dimensions (width, height)
            
        Returns:
            Tuple of (PhotoImage object, reference key)
        """
        try:
            # Convert source to PIL Image
            pil_image = self._convert_to_pil(source)
            
            # Resize if requested
            if size is not None:
                pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            
            # Create PhotoImage
            photo_image = ImageTk.PhotoImage(pil_image)
            
            # Generate key if not provided
            if key is None:
                key = f"image_{self._counter}"
                self._counter += 1
            
            # Store reference to prevent garbage collection
            self._image_refs[key] = photo_image
            
            self.logger.debug(f"Created PhotoImage with key: {key}")
            return photo_image, key
            
        except Exception as e:
            self.logger.error(f"Failed to create PhotoImage from {type(source)}: {e}")
            raise
    
    def _convert_to_pil(self, source: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """
        Convert various image sources to PIL Image format.
        
        Args:
            source: Image source to convert
            
        Returns:
            PIL Image object
        """
        if isinstance(source, Image.Image):
            return source
            
        elif isinstance(source, np.ndarray):
            # OpenCV image (BGR) -> PIL Image (RGB)
            if len(source.shape) == 3 and source.shape[2] == 3:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_image)
            else:
                # Grayscale or other format
                return Image.fromarray(source)
                
        elif isinstance(source, (str, Path)):
            # File path
            return Image.open(source)
            
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")
    
    def update_widget_image(self, 
                           widget: tk.Widget,
                           source: Union[np.ndarray, Image.Image, str, Path],
                           key: Optional[str] = None,
                           size: Optional[Tuple[int, int]] = None,
                           clear_text: bool = True) -> str:
        """
        Update a Tkinter widget's image with proper reference management.
        
        Args:
            widget: Tkinter widget to update (Label, Button, etc.)
            source: Image source
            key: Optional reference key
            size: Optional resize dimensions
            clear_text: Whether to clear widget text when setting image
            
        Returns:
            Reference key for the created image
        """
        try:
            # Create PhotoImage with reference management
            photo_image, ref_key = self.create_photo_image(source, key, size)
            
            # Update widget
            widget.config(image=photo_image)
            if clear_text:
                widget.config(text="")
            
            # CRITICAL: Set widget's image attribute to prevent garbage collection
            widget.image = photo_image
            
            return ref_key
            
        except Exception as e:
            self.logger.error(f"Failed to update widget image: {e}")
            # Set error state
            widget.config(image="", text="Image Error")
            if hasattr(widget, 'image'):
                widget.image = None
            raise
    
    def clear_widget_image(self, widget: tk.Widget, key: Optional[str] = None):
        """
        Clear a widget's image and clean up references.
        
        Args:
            widget: Widget to clear
            key: Optional reference key to remove
        """
        try:
            # Clear widget
            widget.config(image="", text="No Image")
            if hasattr(widget, 'image'):
                widget.image = None
            
            # Remove from reference storage if key provided
            if key and key in self._image_refs:
                del self._image_refs[key]
                self.logger.debug(f"Removed image reference: {key}")
                
        except Exception as e:
            self.logger.error(f"Failed to clear widget image: {e}")
    
    def get_image(self, key: str) -> Optional[ImageTk.PhotoImage]:
        """
        Get a stored PhotoImage by key.
        
        Args:
            key: Reference key
            
        Returns:
            PhotoImage object or None if not found
        """
        return self._image_refs.get(key)
    
    def remove_image(self, key: str) -> bool:
        """
        Remove a stored PhotoImage reference.
        
        Args:
            key: Reference key to remove
            
        Returns:
            True if removed, False if not found
        """
        if key in self._image_refs:
            del self._image_refs[key]
            self.logger.debug(f"Removed image reference: {key}")
            return True
        return False
    
    def clear_all(self):
        """Clear all stored image references."""
        count = len(self._image_refs)
        self._image_refs.clear()
        self.logger.info(f"Cleared {count} image references")
    
    def get_reference_count(self) -> int:
        """Get the number of stored image references."""
        return len(self._image_refs)
    
    def get_reference_keys(self) -> list:
        """Get all stored reference keys."""
        return list(self._image_refs.keys())


class AssetImageBridge:
    """
    Bridge between AssetLoader (OpenCV images) and GUI (PhotoImage objects).
    
    This class provides utilities for loading card images from the asset system
    and converting them to GUI-compatible formats with proper reference management.
    """
    
    def __init__(self, asset_loader, image_manager: PhotoImageManager):
        """
        Initialize the asset-image bridge.
        
        Args:
            asset_loader: AssetLoader instance
            image_manager: PhotoImageManager instance
        """
        self.asset_loader = asset_loader
        self.image_manager = image_manager
        self.logger = logging.getLogger(__name__)
    
    def load_card_image_for_gui(self, 
                               card_code: str,
                               premium: bool = False,
                               size: Optional[Tuple[int, int]] = None) -> Optional[Tuple[ImageTk.PhotoImage, str]]:
        """
        Load a card image from assets and convert to PhotoImage for GUI use.
        
        Args:
            card_code: Card code to load
            premium: Whether to load premium version
            size: Optional resize dimensions
            
        Returns:
            Tuple of (PhotoImage, reference_key) or None if failed
        """
        try:
            # Load OpenCV image from asset loader
            cv_image = self.asset_loader.load_card_image(card_code, premium)
            if cv_image is None:
                self.logger.warning(f"Could not load card image: {card_code}")
                return None
            
            # Create unique key for this card
            suffix = "_premium" if premium else ""
            key = f"card_{card_code}{suffix}"
            
            # Convert to PhotoImage with reference management
            photo_image, ref_key = self.image_manager.create_photo_image(
                cv_image, key, size
            )
            
            return photo_image, ref_key
            
        except Exception as e:
            self.logger.error(f"Failed to load card image for GUI {card_code}: {e}")
            return None
    
    def update_card_widget(self,
                          widget: tk.Widget,
                          card_code: str,
                          premium: bool = False,
                          size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Update a widget with a card image from assets.
        
        Args:
            widget: Widget to update
            card_code: Card code to load
            premium: Whether to load premium version
            size: Optional resize dimensions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.load_card_image_for_gui(card_code, premium, size)
            if result is None:
                # Clear widget and show error
                self.image_manager.clear_widget_image(widget)
                widget.config(text="Image Not Found")
                return False
            
            photo_image, ref_key = result
            
            # Update widget with image
            widget.config(image=photo_image, text="")
            widget.image = photo_image  # Prevent garbage collection
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update card widget {card_code}: {e}")
            self.image_manager.clear_widget_image(widget)
            widget.config(text="Image Error")
            return False


# Global instances for singleton pattern
_global_image_manager: Optional[PhotoImageManager] = None
_global_asset_bridge: Optional[AssetImageBridge] = None


def get_image_manager() -> PhotoImageManager:
    """Get the global PhotoImageManager instance."""
    global _global_image_manager
    if _global_image_manager is None:
        _global_image_manager = PhotoImageManager()
    return _global_image_manager


def get_asset_bridge(asset_loader) -> AssetImageBridge:
    """Get the global AssetImageBridge instance."""
    global _global_asset_bridge
    if _global_asset_bridge is None:
        _global_asset_bridge = AssetImageBridge(asset_loader, get_image_manager())
    return _global_asset_bridge


def create_safe_photo_image(source: Union[np.ndarray, Image.Image, str, Path],
                           size: Optional[Tuple[int, int]] = None) -> Tuple[ImageTk.PhotoImage, str]:
    """
    Convenience function to create a PhotoImage with proper reference management.
    
    Args:
        source: Image source
        size: Optional resize dimensions
        
    Returns:
        Tuple of (PhotoImage, reference_key)
    """
    return get_image_manager().create_photo_image(source, size=size)


def update_widget_with_safe_image(widget: tk.Widget,
                                 source: Union[np.ndarray, Image.Image, str, Path],
                                 size: Optional[Tuple[int, int]] = None) -> str:
    """
    Convenience function to update a widget with proper image reference management.
    
    Args:
        widget: Widget to update
        source: Image source
        size: Optional resize dimensions
        
    Returns:
        Reference key for the created image
    """
    return get_image_manager().update_widget_image(widget, source, size=size)