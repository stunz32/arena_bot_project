#!/usr/bin/env python3
"""
INTEGRATED ARENA BOT - GUI VERSION
Complete system with GUI interface for Windows
Combines log monitoring + visual detection + AI recommendations
"""

import sys
import time
import threading
import cv2
import numpy as np
import os
import json
import copy
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
from queue import Queue

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

# Import debug modules
from debug_config import get_debug_config, is_debug_enabled, enable_debug, disable_debug
from visual_debugger import VisualDebugger, create_debug_visualization, save_debug_image
from metrics_logger import MetricsLogger, log_detection_metrics, generate_performance_report

# Import CardRefiner for perfect coordinate refinement
from arena_bot.core.card_refiner import CardRefiner

class ManualCorrectionDialog(tk.Toplevel):
    """
    Dialog window for manual card correction with auto-complete search.
    Allows users to override incorrect card detections.
    """
    
    def __init__(self, parent_bot, callback):
        """
        Initialize the manual correction dialog.
        
        Args:
            parent_bot: Reference to the main IntegratedArenaBotGUI instance
            callback: Function to call when correction is complete
        """
        super().__init__(parent_bot.root)
        
        self.parent_bot = parent_bot
        self.callback = callback
        self.selected_card_code = None
        self.current_suggestions = []  # Store (name, card_id) tuples
        
        self.setup_dialog()
        self._update_suggestions()  # Initialize with empty search
    
    def setup_dialog(self):
        """Setup the dialog window and UI components."""
        self.title("Manual Card Correction")
        self.geometry("500x400")
        self.configure(bg='#2C3E50')
        
        # Make dialog modal
        self.transient(self.parent_bot.root)
        self.grab_set()
        
        # Center the dialog
        self.center_window()
        
        # Main frame
        main_frame = tk.Frame(self, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🔧 Manual Card Correction",
            font=('Arial', 14, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions_label = tk.Label(
            main_frame,
            text="Type a card name to search, or select from the suggestions below:",
            font=('Arial', 10),
            fg='#BDC3C7',
            bg='#2C3E50'
        )
        instructions_label.pack(pady=(0, 10))
        
        # Search entry
        self.search_entry = tk.Entry(
            main_frame,
            font=('Arial', 12),
            width=50,
            bg='#34495E',
            fg='#ECF0F1',
            insertbackground='#ECF0F1'
        )
        self.search_entry.pack(pady=(0, 10), fill='x')
        self.search_entry.bind('<KeyRelease>', self._on_key_release)
        
        # Suggestions listbox with scrollbar
        listbox_frame = tk.Frame(main_frame, bg='#2C3E50')
        listbox_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Scrollbar
        scrollbar = tk.Scrollbar(listbox_frame, bg='#34495E')
        scrollbar.pack(side='right', fill='y')
        
        # Listbox
        self.suggestions_listbox = tk.Listbox(
            listbox_frame,
            font=('Arial', 10),
            bg='#34495E',
            fg='#ECF0F1',
            selectbackground='#3498DB',
            selectforeground='#ECF0F1',
            height=12,
            yscrollcommand=scrollbar.set
        )
        self.suggestions_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.suggestions_listbox.yview)
        
        # Bind double-click to select
        self.suggestions_listbox.bind('<Double-Button-1>', self._on_select)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg='#2C3E50')
        buttons_frame.pack(fill='x')
        
        # OK button
        self.ok_button = tk.Button(
            buttons_frame,
            text="OK",
            font=('Arial', 10, 'bold'),
            bg='#27AE60',
            fg='white',
            command=self._on_select,
            width=15
        )
        self.ok_button.pack(side='left', padx=(0, 10))
        
        # Cancel button
        cancel_button = tk.Button(
            buttons_frame,
            text="Cancel",
            font=('Arial', 10),
            bg='#E74C3C',
            fg='white',
            command=self._on_cancel,
            width=15
        )
        cancel_button.pack(side='left')
        
        # Focus on search entry
        self.search_entry.focus()
    
    def center_window(self):
        """Center the dialog window on screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    
    def _on_key_release(self, event):
        """Handle key release in search entry for live search."""
        self._update_suggestions()
    
    def _update_suggestions(self, event=None):
        """Update the suggestions listbox based on search term."""
        search_term = self.search_entry.get().lower().strip()
        self.suggestions_listbox.delete(0, tk.END)
        
        # If the search box is empty, show an empty list (per plan requirement)
        if not search_term:
            return
        
        # Perform a live search against the FULL card database
        try:
            # Get ALL collectible cards from the master JSON loader
            all_collectible_cards = self.parent_bot.cards_json_loader.get_all_collectible_cards()
            
            # Filter names that start with the search term
            matching_cards = [
                (card_data.get('name', ''), card_data.get('id', '')) 
                for card_data in all_collectible_cards
                if card_data.get('name', '').lower().startswith(search_term)
            ]
            
            # Sort the results alphabetically by name
            matching_cards.sort(key=lambda x: x[0])

            # Store the (name, id) tuples for later use
            self.current_suggestions = matching_cards
            
            # Display only the names in the listbox
            for name, card_id in matching_cards[:50]:  # Show up to 50 results
                self.suggestions_listbox.insert(tk.END, name)
                
        except Exception as e:
            self.suggestions_listbox.insert(tk.END, f"Error: {e}")
            print(f"Error updating suggestions: {e}")
    
    def _on_select(self, event=None):
        """Handle selection of a card from the listbox."""
        selection_indices = self.suggestions_listbox.curselection()
        if not selection_indices:
            return

        selection_index = selection_indices[0]
        
        # Use the index to get the (name, card_id) tuple from our stored list
        if selection_index < len(self.current_suggestions):
            selected_name, selected_card_id = self.current_suggestions[selection_index]
            
            # Call the callback with the DIRECT CARD ID. No lookup needed!
            self.callback(selected_card_id)
            self.destroy()
        else:
            messagebox.showerror("Error", "Selection index out of range. Please try again.")
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.destroy()
    

class IntegratedArenaBotGUI:
    """
    Complete Arena Bot with GUI interface:
    - Log monitoring (Arena Tracker style) 
    - Visual screenshot analysis
    - AI recommendations (draft advisor)
    - Full GUI interface for Windows
    """
    
    def __init__(self):
        """Initialize the integrated arena bot with GUI."""
        print("🚀 INTEGRATED ARENA BOT - GUI VERSION")
        print("=" * 80)
        print("✅ Full functionality with graphical interface:")
        print("   • Log monitoring (Arena Tracker methodology)")
        print("   • Visual screenshot analysis")
        print("   • AI recommendations (draft advisor)")
        print("   • Complete GUI interface")
        print("=" * 80)
        
        # Initialize subsystems
        self.init_log_monitoring()
        self.init_ai_advisor()
        self.init_card_detection()
        
        # Initialize debug systems
        self.debug_config = get_debug_config()
        self.visual_debugger = VisualDebugger()
        self.metrics_logger = MetricsLogger()
        self.ground_truth_data = self.load_ground_truth_data()
        
        # State management
        self.running = False
        self.last_full_analysis_result = None
        self.in_draft = False
        self.current_hero = None
        self.draft_picks_count = 0
        self.custom_coordinates = None  # Store user-selected coordinates
        self.last_known_good_coords = None  # Cache successful coordinates for stability
        self._enable_custom_mode_on_startup = False  # Flag for auto-enabling custom mode
        self.last_analysis_candidates = [[], [], []]  # Store top candidates for manual correction
        self.last_detection_result = None  # Store last analysis result for correction callbacks
        self.last_full_analysis_result = None  # Store complete analysis result with candidate_lists
        
        # Load saved coordinates if available
        self.load_saved_coordinates()
        
        # Card name database (Arena Tracker style)
        self.cards_json_loader = self.init_cards_json()
        
        # Arena database for filtering arena-eligible cards
        self.arena_database = None
        try:
            from arena_bot.data.arena_card_database import ArenaCardDatabase
            self.arena_database = ArenaCardDatabase()
            print("✅ Arena database loaded for intelligent card search")
        except Exception as e:
            print(f"⚠️ Arena database not available: {e}")
            print("   Manual correction will use full card database")
        
        # Threading setup for non-blocking analysis
        self.result_queue = Queue()
        self.analysis_in_progress = False
        
        # Background cache building
        self.cache_build_in_progress = False
        self._start_background_cache_builder()
        
        # GUI setup
        self.setup_gui()
        
        print("🎯 Integrated Arena Bot GUI ready!")
    
    def _run_detection_with_timeout(self, detection_func, timeout_seconds: float, method_name: str):
        """
        Run a detection function with timeout protection to prevent freezing.
        
        Args:
            detection_func: Function to run with timeout
            timeout_seconds: Maximum time to wait
            method_name: Name for logging
            
        Returns:
            Result from detection_func or None if timeout
        """
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def worker():
            try:
                result = detection_func()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        # Start the detection in a separate thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        # Wait for result with timeout
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            self.log_text(f"   ⚠️ {method_name} detection timeout ({timeout_seconds:.1f}s), falling back")
            return None
        
        # Check for exceptions
        if not exception_queue.empty():
            exception = exception_queue.get()
            self.log_text(f"   ⚠️ {method_name} detection error: {exception}")
            return None
        
        # Get result
        if not result_queue.empty():
            return result_queue.get()
        
        return None
    
    def _validate_cached_coordinates(self, screenshot: np.ndarray) -> bool:
        """
        Validate that cached coordinates are still valid for the current screenshot.
        
        Args:
            screenshot: Current screenshot to validate against
            
        Returns:
            True if cached coordinates are still valid
        """
        if not self.last_known_good_coords:
            return False
        
        try:
            height, width = screenshot.shape[:2]
            
            # Basic sanity checks - coordinates should still be within bounds
            for i, (x, y, w, h) in enumerate(self.last_known_good_coords):
                if x + w > width or y + h > height or x < 0 or y < 0:
                    self.log_text(f"   ⚠️ Cached coordinate {i+1} out of bounds: ({x},{y},{w},{h})")
                    return False
            
            # Check if regions contain reasonable card-like content
            valid_regions = 0
            for i, (x, y, w, h) in enumerate(self.last_known_good_coords):
                region = screenshot[y:y+h, x:x+w]
                
                # Basic quality checks
                if region.size == 0:
                    continue
                
                # Check for reasonable brightness (not all black/white)
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                mean_brightness = np.mean(gray)
                brightness_variance = np.var(gray)
                
                # Should have some content (not empty) and some variation
                if 20 < mean_brightness < 235 and brightness_variance > 100:
                    valid_regions += 1
            
            # Need at least 2 valid regions to consider cache valid
            if valid_regions >= 2:
                self.log_text(f"   ✅ Cache validation: {valid_regions}/{len(self.last_known_good_coords)} regions valid")
                return True
            else:
                self.log_text(f"   ❌ Cache validation failed: only {valid_regions}/{len(self.last_known_good_coords)} regions valid")
                return False
                
        except Exception as e:
            self.log_text(f"   ❌ Cache validation error: {e}")
            return False
    
    def _resolve_ambiguity_with_features(self, query_region: np.ndarray, candidates: list):
        """
        Uses feature matching to disambiguate between several close histogram matches.
        
        Args:
            query_region: The card region image to match
            candidates: List of candidate matches from histogram matching
            
        Returns:
            The best match from feature matching, or the original best match if feature matching fails
        """
        if not self.ultimate_detector:
            self.log_text("      ⚠️ Ultimate Detection Engine not available for fallback.")
            return candidates[0] if candidates else None
        
        candidate_card_codes = [c.card_code for c in candidates]
        self.log_text(f"      Verifying candidates: {', '.join([self.get_card_name(c) for c in candidate_card_codes])}")
        
        # Create a temporary, focused database with only the candidates
        candidate_images = {}
        for code in candidate_card_codes:
            # Handle premium cards
            base_code = code.replace('_premium', '')
            is_premium = '_premium' in code
            
            # Load the card image
            if self.asset_loader:
                image = self.asset_loader.load_card_image(base_code, premium=is_premium)
                if image is not None:
                    candidate_images[code] = image
                else:
                    self.log_text(f"      ⚠️ Could not load image for {code}")
        
        if not candidate_images:
            self.log_text("      ⚠️ No candidate images loaded for feature matching")
            return candidates[0] if candidates else None
        
        try:
            # Initialize the Ultimate Detection Engine database for just these candidates
            # This is fast because it's only for a few cards
            self.ultimate_detector.load_card_database(candidate_images)
            
            # Run the ultimate detection
            ultimate_result = self.ultimate_detector.detect_card_ultimate(query_region)
            
            if ultimate_result and hasattr(ultimate_result, 'card_code'):
                best_feature_match = ultimate_result
                
                # Find the corresponding original histogram match object to return
                for original_match in candidates:
                    if original_match.card_code == best_feature_match.card_code:
                        algorithm_name = getattr(best_feature_match, 'algorithm', 'Feature Ensemble')
                        self.log_text(f"      Feature algorithm '{algorithm_name}' selected {self.get_card_name(original_match.card_code)}.")
                        return original_match
                
                # If we can't find the original match, create a new one with the feature result
                self.log_text(f"      Feature matching found {self.get_card_name(best_feature_match.card_code)} but no original match found.")
                return best_feature_match
            
        except Exception as e:
            self.log_text(f"      ⚠️ Feature matching error: {e}")
        
        # If feature matching fails, stick with the original best histogram match
        self.log_text("      ⚠️ Feature matching did not find a confident result. Using original top match.")
        return candidates[0] if candidates else None
    
    def _open_correction_dialog(self, card_index):
        """
        Open the manual correction dialog for a specific card slot.
        
        Args:
            card_index: Index of the card to correct (0, 1, or 2)
        """
        # Check our new persistent state variable
        if not self.last_full_analysis_result:
            messagebox.showinfo("Info", "Please analyze a screenshot first.")
            return

        # Pass the callback and the parent bot instance to the dialog
        def on_correction_complete(new_card_code):
            self._on_card_corrected(card_index, new_card_code)

        # Create and run the dialog
        dialog = ManualCorrectionDialog(self, on_correction_complete)
        
        # Log the correction attempt
        self.log_text(f"🔧 Opening manual correction dialog for Card {card_index + 1}")
    
    def _on_card_corrected(self, card_index, new_card_code):
        """
        Handles a manual card correction by updating the card list, re-running the
        full analysis pipeline (including the AI advisor), and then refreshing the UI.
        """
        try:
            card_name = self.get_card_name(new_card_code)
            self.log_text(f"✅ Manual correction for Card #{card_index + 1}: {card_name}")

            if not self.last_full_analysis_result or 'detected_cards' not in self.last_full_analysis_result:
                messagebox.showerror("Error", "No analysis data to correct.")
                return

            # --- Step 1: Create an updated list of detected cards ---
            # Start with a deep copy of the last known detected cards
            updated_detected_cards = copy.deepcopy(self.last_full_analysis_result['detected_cards'])

            if card_index < len(updated_detected_cards):
                # Update the specific card with the corrected information
                updated_detected_cards[card_index]['card_code'] = new_card_code
                updated_detected_cards[card_index]['card_name'] = self.get_card_name(new_card_code)
                updated_detected_cards[card_index]['confidence'] = 1.0  # Manual override is 100% confident
                updated_detected_cards[card_index]['strategy'] = 'Manual Correction'
            else:
                self.log_text(f"❌ Error: Card index {card_index} is out of bounds.")
                return

            # --- Step 2: Re-run the AI Recommendation with the new card list ---
            self.log_text("🤖 Re-running AI advisor with corrected card list...")
            new_recommendation = None
            if self.advisor:
                card_codes_for_ai = [card['card_code'] for card in updated_detected_cards if card.get('card_code') != 'Unknown']
                if len(card_codes_for_ai) >= 3:
                    try:
                        new_choice = self.advisor.analyze_draft_choice(card_codes_for_ai[:3], self.current_hero or 'unknown')
                        new_recommendation = {
                            'recommended_pick': new_choice.recommended_pick + 1,
                            'recommended_card': new_choice.cards[new_choice.recommended_pick].card_code,
                            'reasoning': new_choice.reasoning,
                            'card_details': [vars(card) for card in new_choice.cards]
                        }
                        self.log_text(f"   ✅ New recommendation generated: {self.get_card_name(new_recommendation['recommended_card'])}")
                    except Exception as e:
                        self.log_text(f"   ⚠️ AI recommendation failed during re-run: {e}")
            
            # --- Step 3: Create a new, complete analysis object ---
            # This is the new source of truth for the UI
            new_analysis_result = {
                'detected_cards': updated_detected_cards,
                'recommendation': new_recommendation,
                'candidate_lists': self.last_full_analysis_result.get('candidate_lists', []),
                'success': True
            }

            # --- Step 4: Update the application state and refresh the UI ---
            self.last_full_analysis_result = new_analysis_result
            self.show_analysis_result(self.last_full_analysis_result)

        except Exception as e:
            self.log_text(f"❌ Error processing card correction: {e}")
            messagebox.showerror("Error", f"Failed to apply correction: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_display_with_corrected_data(self, detection_result):
        """
        Update the display and recommendations with corrected card data.
        
        Args:
            detection_result: The detection result with corrected card data
        """
        try:
            # Update the card display
            detected_cards = detection_result.get('detected_cards', [])
            
            # Update card name labels and images
            for i, card_data in enumerate(detected_cards[:3]):  # Only process first 3 cards
                if i < len(self.card_name_labels):
                    card_name = card_data.get('card_name', 'Unknown')
                    confidence = card_data.get('confidence', 0.0)
                    detection_method = card_data.get('detection_method', 'Unknown')
                    
                    # Update name label
                    display_text = f"Card {i+1}: {card_name}"
                    if detection_method == 'Manual Correction':
                        display_text += " [CORRECTED]"
                    else:
                        display_text += f" ({confidence:.2f})"
                    
                    self.card_name_labels[i].config(text=display_text)
                    
                    # Update card image if possible
                    if hasattr(self, 'asset_loader') and self.asset_loader:
                        try:
                            card_code = card_data.get('card_code', '')
                            base_code = card_code.replace('_premium', '')
                            is_premium = '_premium' in card_code
                            
                            card_image = self.asset_loader.load_card_image(base_code, premium=is_premium)
                            if card_image is not None:
                                # Convert to PIL and resize for display
                                from PIL import Image, ImageTk
                                pil_image = Image.fromarray(cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB))
                                pil_image = pil_image.resize((200, 280), Image.Resampling.LANCZOS)
                                photo = ImageTk.PhotoImage(pil_image)
                                
                                self.card_image_labels[i].config(image=photo)
                                self.card_image_labels[i].image = photo  # Keep reference
                        except Exception as e:
                            self.log_text(f"⚠️ Could not update image for card {i+1}: {e}")
            
            # Re-run AI recommendations if available
            if hasattr(self, 'draft_advisor') and self.draft_advisor:
                try:
                    # Extract card codes for AI analysis
                    card_codes = [card.get('card_code', '') for card in detected_cards[:3]]
                    
                    # Get AI recommendation
                    recommendation = self.draft_advisor.get_pick_recommendation(card_codes)
                    
                    if recommendation:
                        # Update recommendation display
                        self.show_recommendation(
                            f"AI Recommendation (Updated): {recommendation.get('recommended_card', 'Unknown')}", 
                            recommendation.get('explanation', 'No explanation available')
                        )
                        
                        self.log_text(f"🤖 AI recommendation updated with corrected data")
                    else:
                        self.log_text("⚠️ Could not get updated AI recommendation")
                        
                except Exception as e:
                    self.log_text(f"⚠️ Could not update AI recommendation: {e}")
            
            self.log_text("✅ Display updated with corrected card data")
            
        except Exception as e:
            self.log_text(f"❌ Error updating display: {e}")
            messagebox.showerror("Error", f"Failed to update display: {e}")
    
    def get_card_name(self, card_code):
        """
        Get the display name for a card code.
        
        Args:
            card_code: The card code to get the name for
            
        Returns:
            The card name, or the card code if name not found
        """
        if not card_code:
            return "Unknown Card"
        
        try:
            # Try to get name from cards_json_loader
            if hasattr(self, 'cards_json_loader') and self.cards_json_loader:
                card_data = self.cards_json_loader.get_card_data(card_code)
                if card_data and 'name' in card_data:
                    return card_data['name']
            
            # Fallback: return the card code itself
            return card_code
        except Exception as e:
            return card_code
    
    def _validate_region_for_refinement(self, region: np.ndarray, card_number: int) -> bool:
        """
        Validate that a region is suitable for CardRefiner processing.
        
        Prevents CardRefiner from processing garbage regions that would result
        in tiny, useless crops like 65x103 pixels.
        
        Args:
            region: Image region to validate
            card_number: Card number for logging
            
        Returns:
            True if region is suitable for refinement
        """
        try:
            if region.size == 0:
                self.log_text(f"   ❌ Card {card_number} region is empty")
                return False
            
            h, w = region.shape[:2]
            area = w * h
            
            # Minimum size requirements for refinement
            min_width, min_height = 200, 300
            min_area = 50000  # Should be larger than refined result
            
            if w < min_width or h < min_height:
                self.log_text(f"   ❌ Card {card_number} region too small for refinement: {w}x{h}")
                return False
            
            if area < min_area:
                self.log_text(f"   ❌ Card {card_number} area too small for refinement: {area}")
                return False
            
            # Check for reasonable content (not just background)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # Brightness check (should not be all black or all white)
            mean_brightness = np.mean(gray)
            if mean_brightness < 10 or mean_brightness > 245:
                self.log_text(f"   ❌ Card {card_number} brightness invalid for refinement: {mean_brightness:.1f}")
                return False
            
            # Variance check (should have some content variation)
            brightness_variance = np.var(gray)
            if brightness_variance < 50:
                self.log_text(f"   ❌ Card {card_number} too uniform for refinement: variance {brightness_variance:.1f}")
                return False
            
            # Edge density check (should have some structure)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density < 0.02:  # Very few edges
                self.log_text(f"   ❌ Card {card_number} insufficient edges for refinement: {edge_density:.3f}")
                return False
            
            self.log_text(f"   ✅ Card {card_number} region suitable for refinement: {w}x{h}, variance: {brightness_variance:.1f}")
            return True
            
        except Exception as e:
            self.log_text(f"   ❌ Error validating region for refinement: {e}")
            return False
    
    def load_ground_truth_data(self):
        """Load ground truth data for validation testing."""
        try:
            ground_truth_file = self.debug_config.GROUND_TRUTH_FILE
            if ground_truth_file.exists():
                with open(ground_truth_file, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded ground truth data: {ground_truth_file}")
                return data
            else:
                print(f"⚠️ Ground truth file not found: {ground_truth_file}")
                return None
        except Exception as e:
            print(f"⚠️ Failed to load ground truth data: {e}")
            return None
    
    def get_ground_truth_boxes(self, resolution: str = None) -> list:
        """Get ground truth card coordinates for current resolution."""
        if not self.ground_truth_data:
            return []
        
        if not resolution:
            # Auto-detect resolution from current screenshot
            resolution = "3440x1440"  # Default to user's resolution
        
        try:
            resolutions = self.ground_truth_data.get('resolutions', {})
            if resolution in resolutions:
                positions = resolutions[resolution].get('card_positions', [])
                # Convert to (x, y, w, h) format
                return [(pos['x'], pos['y'], pos['width'], pos['height']) for pos in positions]
        except Exception as e:
            print(f"⚠️ Error getting ground truth boxes: {e}")
        
        return []
    
    def init_cards_json(self):
        """Initialize cards JSON database like Arena Tracker."""
        try:
            from arena_bot.data.cards_json_loader import get_cards_json_loader
            cards_loader = get_cards_json_loader()
            print(f"✅ Loaded Hearthstone cards.json database")
            return cards_loader
        except Exception as e:
            print(f"⚠️ Cards JSON not available: {e}")
            return None
    
    def get_card_name(self, card_code):
        """Get user-friendly card name using Arena Tracker method."""
        clean_code = card_code.replace('_premium', '')
        if self.cards_json_loader:
            name = self.cards_json_loader.get_card_name(clean_code)
            if '_premium' in card_code and name != f"Unknown ({clean_code})":
                return f"{name} ✨"
            return name
        return f"Unknown Card ({clean_code})"
    
    def enhance_card_image(self, image, aggressive=False):
        """
        Enhanced image preprocessing for better card detection.
        
        Applies multiple enhancement techniques to improve detection accuracy
        for cards with difficult lighting, angles, or quality issues.
        
        Args:
            image: Input card image
            aggressive: If True, applies more aggressive enhancement for poor quality images
        """
        try:
            if image is None or image.size == 0:
                return image
            
            # Convert to different color spaces for processing
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Enhancement pipeline
            enhanced = image.copy()
            
            # 1. Adaptive histogram equalization (CLAHE) on LAB L channel
            # Improves contrast while preserving colors
            l_channel = lab[:,:,0]
            if aggressive:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))  # More aggressive
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Standard
            l_enhanced = clahe.apply(l_channel)
            lab[:,:,0] = l_enhanced
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 2. Gamma correction for brightness adjustment
            # Helps with over/under-exposed cards
            gamma = self.calculate_optimal_gamma(enhanced)
            if gamma != 1.0:
                gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, gamma_table)
            
            # 3. Unsharp mask for detail enhancement
            # Improves edge definition for better histogram matching
            if aggressive:
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)  # Stronger blur
                enhanced = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)  # More aggressive
            else:
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
                enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            # 4. Noise reduction while preserving edges
            # Reduces noise that can interfere with histogram matching
            if aggressive:
                enhanced = cv2.bilateralFilter(enhanced, 11, 100, 100)  # Stronger noise reduction
            else:
                enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 5. Color balance correction
            # Ensures consistent color representation
            enhanced = self.auto_color_balance(enhanced)
            
            return enhanced
            
        except Exception as e:
            # If enhancement fails, return original image
            self.log_text(f"      ⚠️ Image enhancement error: {e}")
            return image
    
    def calculate_optimal_gamma(self, image):
        """Calculate optimal gamma correction based on image brightness."""
        try:
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Adjust gamma based on brightness
            if mean_brightness < 80:  # Dark image
                return 0.7  # Brighten
            elif mean_brightness > 180:  # Bright image
                return 1.3  # Darken
            else:
                return 1.0  # No adjustment needed
        except:
            return 1.0
    
    def auto_color_balance(self, image):
        """Apply automatic color balance correction."""
        try:
            # Simple white balance using gray world assumption
            result = image.copy().astype(np.float32)
            
            # Calculate channel means
            b_mean, g_mean, r_mean = cv2.mean(result)[:3]
            
            # Calculate scaling factors
            gray_mean = (b_mean + g_mean + r_mean) / 3
            b_scale = gray_mean / b_mean if b_mean > 0 else 1.0
            g_scale = gray_mean / g_mean if g_mean > 0 else 1.0  
            r_scale = gray_mean / r_mean if r_mean > 0 else 1.0
            
            # Apply scaling
            result[:,:,0] *= b_scale
            result[:,:,1] *= g_scale
            result[:,:,2] *= r_scale
            
            # Clamp and convert back
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
            
        except:
            return image
    
    def multi_scale_detection(self, image, session_id):
        """
        Multi-scale detection using different resize strategies.
        
        Tries multiple image sizes and strategies to find the best match,
        similar to how Arena Tracker handles different card qualities.
        """
        try:
            if image is None or image.size == 0:
                return None
            
            # Different resize strategies to try
            strategies = [
                ("original", image),  # Use original size
                ("80x80_area", cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)),
                ("100x100_area", cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)),
                ("80x80_cubic", cv2.resize(image, (80, 80), interpolation=cv2.INTER_CUBIC)),
                ("64x64_area", cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)),
                ("120x120_area", cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA))
            ]
            
            best_overall_match = None
            best_overall_confidence = 0.0
            strategy_results = []
            
            for strategy_name, processed_image in strategies:
                try:
                    # Try detection with this strategy
                    match = self.histogram_matcher.match_card(
                        processed_image,
                        confidence_threshold=None,  # Use adaptive threshold
                        attempt_count=0,
                        session_id=f"{session_id}_{strategy_name}"
                    )
                    
                    if match:
                        strategy_results.append((strategy_name, match, match.confidence))
                        
                        # Track the best overall match
                        if match.confidence > best_overall_confidence:
                            best_overall_confidence = match.confidence
                            best_overall_match = match
                            best_overall_match.detection_strategy = strategy_name
                        
                        self.log_text(f"      📏 {strategy_name}: {self.get_card_name(match.card_code)} (conf: {match.confidence:.3f})")
                    else:
                        self.log_text(f"      📏 {strategy_name}: No match")
                        
                except Exception as e:
                    self.log_text(f"      ⚠️ {strategy_name} failed: {e}")
                    continue
            
            # Analyze results for consistency
            if len(strategy_results) >= 2:
                # Check if multiple strategies agree
                card_codes = [result[1].card_code for result in strategy_results]
                most_common = max(set(card_codes), key=card_codes.count) if card_codes else None
                agreement_count = card_codes.count(most_common) if most_common else 0
                
                if agreement_count >= 2:
                    # Multiple strategies agree - boost confidence
                    if best_overall_match and best_overall_match.card_code == most_common:
                        best_overall_match.confidence = min(1.0, best_overall_match.confidence * 1.2)
                        best_overall_match.stability_score = min(1.0, best_overall_match.stability_score + 0.2)
                        self.log_text(f"      ✅ Multi-strategy agreement detected - confidence boosted")
                
                self.log_text(f"      📊 Strategy consensus: {agreement_count}/{len(strategy_results)} agree on {self.get_card_name(most_common) if most_common else 'None'}")
            
            return best_overall_match
            
        except Exception as e:
            self.log_text(f"      ❌ Multi-scale detection failed: {e}")
            # Fallback to single strategy
            try:
                return self.histogram_matcher.match_card(
                    image,
                    confidence_threshold=None,
                    attempt_count=0,
                    session_id=session_id
                )
            except:
                return None
    
    def assess_region_quality(self, image):
        """
        Assess the quality of a captured card region.
        
        Returns a quality score (0.0-1.0) and list of identified issues.
        Helps identify regions that may cause detection problems.
        """
        try:
            if image is None or image.size == 0:
                return 0.0, ["Empty or invalid image"]
            
            h, w = image.shape[:2]
            quality_score = 1.0
            issues = []
            
            # 1. Size validation
            if w < 100 or h < 100:
                quality_score -= 0.3
                issues.append(f"Too small ({w}x{h})")
            elif w < 150 or h < 150:
                quality_score -= 0.1
                issues.append(f"Small size ({w}x{h})")
            
            # 2. Aspect ratio check (cards should be roughly portrait)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 1.2:  # Too wide
                quality_score -= 0.2
                issues.append(f"Too wide (ratio: {aspect_ratio:.2f})")
            elif aspect_ratio < 0.4:  # Too narrow
                quality_score -= 0.2
                issues.append(f"Too narrow (ratio: {aspect_ratio:.2f})")
            
            # 3. Brightness analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            if mean_brightness < 30:
                quality_score -= 0.3
                issues.append("Too dark")
            elif mean_brightness > 220:
                quality_score -= 0.2
                issues.append("Too bright")
            
            # 4. Contrast analysis
            if std_brightness < 20:
                quality_score -= 0.2
                issues.append("Low contrast")
            
            # 5. Color analysis - check if it looks like a card
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Cards should have reasonable color variety
            h_channel = hsv[:,:,0]
            s_channel = hsv[:,:,1]
            
            # Check for color variety (not just grayscale)
            color_variance = np.std(h_channel)
            if color_variance < 10:
                quality_score -= 0.1
                issues.append("Low color variety")
            
            # Check saturation (cards usually have some colored elements)
            mean_saturation = np.mean(s_channel)
            if mean_saturation < 30:
                quality_score -= 0.1
                issues.append("Low saturation")
            
            # 6. Edge detection - cards should have clear boundaries
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            if edge_density < 0.05:  # Very few edges
                quality_score -= 0.2
                issues.append("Few edges detected")
            elif edge_density > 0.4:  # Too many edges (noisy)
                quality_score -= 0.1
                issues.append("Too many edges (noisy)")
            
            # 7. Uniform color regions (might be capturing background)
            # Check if image is mostly one color
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            max_hist_value = np.max(hist)
            if max_hist_value > (w * h * 0.6):  # More than 60% same color
                quality_score -= 0.3
                issues.append("Mostly uniform color")
            
            # 8. Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                quality_score -= 0.2
                issues.append("Blurry image")
            
            # Clamp quality score
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score, issues
            
        except Exception as e:
            return 0.0, [f"Quality assessment error: {e}"]
    
    def init_log_monitoring(self):
        """Initialize the log monitoring system."""
        try:
            from hearthstone_log_monitor import HearthstoneLogMonitor
            
            self.log_monitor = HearthstoneLogMonitor()
            self.setup_log_callbacks()
            
            print("✅ Log monitoring system loaded")
        except Exception as e:
            print(f"⚠️ Log monitoring not available: {e}")
            self.log_monitor = None
    
    def init_ai_advisor(self):
        """Initialize the AI draft advisor."""
        try:
            from arena_bot.ai.draft_advisor import get_draft_advisor
            
            self.advisor = get_draft_advisor()
            print("✅ AI draft advisor loaded")
        except Exception as e:
            print(f"⚠️ AI advisor not available: {e}")
            self.advisor = None
    
    def init_card_detection(self):
        """Initialize card detection system with Ultimate Detection Engine option."""
        try:
            # Import detection components - BASIC proven detection system
            from arena_bot.detection.histogram_matcher import get_histogram_matcher
            from arena_bot.detection.template_matcher import get_template_matcher
            from arena_bot.detection.validation_engine import get_validation_engine
            from arena_bot.utils.asset_loader import get_asset_loader
            from arena_bot.core.smart_coordinate_detector import get_smart_coordinate_detector
            from arena_bot.data.arena_card_database import get_arena_card_database
            
            # Initialize basic detection system (PROVEN TO WORK - always available as fallback)
            self.histogram_matcher = get_histogram_matcher()
            self.asset_loader = get_asset_loader()
            self.smart_detector = get_smart_coordinate_detector()  # 100% accuracy detector
            self.arena_database = get_arena_card_database()  # Arena card database for priority filtering
            
            # Keep template validation for mana cost and rarity
            self.template_matcher = get_template_matcher()
            self.validation_engine = get_validation_engine()
            
            # Initialize template matcher with available templates
            if self.template_matcher:
                template_init_success = self.template_matcher.initialize()
                if template_init_success:
                    print("✅ Template matching enabled (mana cost & rarity validation)")
                else:
                    print("⚠️ Template matching disabled (initialization failed)")
                    self.template_matcher = None
            
            # NEW: Initialize Ultimate Detection Engine
            self.ultimate_detector = None
            try:
                from arena_bot.detection.ultimate_detector import get_ultimate_detector, DetectionMode
                self.ultimate_detector = get_ultimate_detector(DetectionMode.ULTIMATE)
                print("🚀 Ultimate Detection Engine loaded (95-99% accuracy enhancement)")
                print("   • SafeImagePreprocessor: CLAHE, bilateral filtering, unsharp masking")
                print("   • FreeAlgorithmEnsemble: ORB, BRISK, AKAZE, SIFT (patent-free)")
                print("   • AdvancedTemplateValidator: Smart template validation & filtering")
                print("   • IntelligentVoting: Multi-algorithm consensus with confidence boosting")
            except Exception as e:
                print(f"⚠️ Ultimate Detection Engine not available: {e}")
                print("   Using basic detection only")
            
            # NEW: Initialize pHash Matcher (Ultra-fast pre-filtering)
            self.phash_matcher = None
            try:
                from arena_bot.detection.phash_matcher import get_perceptual_hash_matcher
                self.phash_matcher = get_perceptual_hash_matcher(
                    use_cache=True,
                    hamming_threshold=10  # Conservative threshold for high accuracy
                )
                print("⚡ Perceptual Hash Matcher loaded (100-1000x faster detection)")
                print("   • Ultra-fast pHash pre-filtering for clear card images")
                print("   • Hamming distance matching with 64-bit DCT hashes")
                print("   • Binary cache for sub-millisecond database loading")
                print("   • Graceful fallback to histogram matching")
            except Exception as e:
                print(f"⚠️ Perceptual Hash Matcher not available: {e}")
                print("   Install with: pip install imagehash")
                print("   Using histogram detection only")
            
            # Load card database for all detection systems
            self._load_card_database()
            
            print("✅ BASIC card detection system loaded (Arena Tracker proven algorithm)")
            print("✅ Smart coordinate detector loaded (100% accuracy)")
            if self.ultimate_detector:
                print("🎯 Ultimate Detection available - toggle in GUI for enhanced accuracy")
            else:
                print("🎯 Using basic detection + template validation (proven working system)")
        except Exception as e:
            print(f"⚠️ Card detection not available: {e}")
            self.histogram_matcher = None
            self.template_matcher = None
            self.validation_engine = None
            self.asset_loader = None
            self.smart_detector = None
            self.ultimate_detector = None
    
    def _load_card_database(self):
        """Load card images into detection systems."""
        if not self.asset_loader:
            return
            
        # Load card images from assets directory
        cards_dir = self.asset_loader.assets_dir / "cards"
        if not cards_dir.exists():
            print(f"⚠️ Cards directory not found: {cards_dir}")
            return
            
        card_images = {}
        card_count = 0
        
        # Load all available card images (full database for maximum detection accuracy)
        for card_file in cards_dir.glob("*.png"):
            try:
                import cv2
                image = cv2.imread(str(card_file))
                if image is not None:
                    card_code = card_file.stem  # Remove .png extension
                    
                    # Filter out non-draftable cards (HERO, BG, etc.)
                    if not any(card_code.startswith(prefix) for prefix in ['HERO_', 'BG_', 'TB_', 'KARA_']):
                        card_images[card_code] = image
                        card_count += 1
            except Exception as e:
                continue
        
        if card_images:
            # Load into basic histogram matcher (always available)
            if self.histogram_matcher:
                self.histogram_matcher.load_card_database(card_images)
                print(f"✅ Loaded {card_count} card images for basic detection")
            
            # Load into pHash matcher (ultra-fast detection)
            if self.phash_matcher:
                self._load_phash_database(card_images)
            
            # Ultimate Detection Engine database loading is deferred for performance
            if self.ultimate_detector:
                print(f"🚀 Ultimate Detection Engine ready (database loading deferred for performance)")
                print("   Database will be loaded automatically when Ultimate Detection is first used")
        else:
            print("⚠️ No card images found")
    
    def _load_ultimate_database(self):
        """Load card database into Ultimate Detection Engine (cache-aware lazy loading)."""
        if not self.ultimate_detector:
            return
            
        try:
            # Use the engine's built-in efficient loading (uses caching automatically)
            self.ultimate_detector.load_card_database()  # No card_images parameter = use internal loading
            self._ultimate_db_loaded = True
            
            # Get status to verify loading and show cache performance
            status = self.ultimate_detector.get_status()
            components = status.get('components', {})
            active_components = [name for name, active in components.items() if active]
            self.log_text(f"   ✅ Ultimate Detection Engine loaded with cache optimization")
            self.log_text(f"   Active components: {', '.join(active_components)}")
            
            # Show database stats if available
            if 'feature_database_stats' in status:
                stats = status['feature_database_stats']
                self.log_text(f"   Database stats: {stats}")
                
        except Exception as e:
            self.log_text(f"   ❌ Failed to load Ultimate Detection database: {e}")
    
    def _load_phash_database(self, card_images):
        """Load card database into pHash matcher with caching."""
        if not self.phash_matcher:
            return
        
        try:
            from arena_bot.detection.phash_cache_manager import get_phash_cache_manager
            
            # Get cache manager
            cache_manager = get_phash_cache_manager()
            
            # Try to load from cache first
            cached_phashes = cache_manager.load_phashes(hash_size=8, hamming_threshold=10)
            
            if cached_phashes:
                # Load from cache
                self.phash_matcher.phash_database = {v: k for k, v in cached_phashes.items()}
                self.phash_matcher.card_phashes = cached_phashes
                print(f"⚡ Loaded {len(cached_phashes)} pHashes from cache in {cache_manager.stats.load_time_ms:.1f}ms")
            else:
                # Compute pHashes for the first time
                print("⚡ Computing pHashes for card database (first time, ~30-60 seconds)...")
                
                def progress_callback(processed, total):
                    if processed % 500 == 0:
                        progress_percent = (processed / total) * 100
                        print(f"   Progress: {processed}/{total} cards ({progress_percent:.1f}%)")
                
                # Load pHashes with progress reporting
                self.phash_matcher.load_card_database(card_images, progress_callback=progress_callback)
                
                # Save to cache for future use
                phash_data = self.phash_matcher.card_phashes.copy()
                cache_success = cache_manager.save_phashes(phash_data, hash_size=8, hamming_threshold=10)
                
                if cache_success:
                    print(f"⚡ pHash database ready: {len(phash_data)} cards, cache saved for fast future loading")
                else:
                    print(f"⚡ pHash database ready: {len(phash_data)} cards (cache save failed)")
            
        except Exception as e:
            print(f"⚠️ Failed to load pHash database: {e}")
            print("   pHash detection will be unavailable")
            self.phash_matcher = None
    
    def setup_log_callbacks(self):
        """Setup callbacks for log monitoring."""
        if not self.log_monitor:
            return
        
        def on_draft_start():
            self.log_text(f"\n{'🎯' * 50}")
            self.log_text("🎯 ARENA DRAFT STARTED!")
            self.log_text("🎯 Ready to analyze screenshots!")
            self.log_text(f"{'🎯' * 50}")
            self.in_draft = True
            self.draft_picks_count = 0
            self.update_status("Arena Draft Active")
        
        def on_draft_complete(picks):
            self.log_text(f"\n{'🏆' * 50}")
            self.log_text("🏆 ARENA DRAFT COMPLETED!")
            self.log_text(f"🏆 Total picks: {len(picks)}")
            self.log_text(f"{'🏆' * 50}")
            self.in_draft = False
            self.update_status("Draft Complete")
        
        def on_game_state_change(old_state, new_state):
            self.log_text(f"\n🎮 GAME STATE: {old_state.value} → {new_state.value}")
            self.in_draft = (new_state.value == "Arena Draft")
            self.update_status(f"Game State: {new_state.value}")
        
        def on_draft_pick(pick):
            self.draft_picks_count += 1
            card_name = self.get_card_name(pick.card_code)
            
            self.log_text(f"\n📋 PICK #{self.draft_picks_count}: {card_name}")
            if pick.is_premium:
                self.log_text("   ✨ GOLDEN CARD!")
        
        self.log_monitor.on_draft_start = on_draft_start
        self.log_monitor.on_draft_complete = on_draft_complete
        self.log_monitor.on_game_state_change = on_game_state_change
        self.log_monitor.on_draft_pick = on_draft_pick
    
    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title("🎯 Integrated Arena Bot - Complete GUI")
        self.root.geometry("1800x1200")  # Even larger size for proper card visibility
        self.root.configure(bg='#2C3E50')
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#34495E', relief='raised', bd=2)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(
            title_frame,
            text="🎯 INTEGRATED ARENA BOT - COMPLETE GUI",
            font=('Arial', 16, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        )
        title_label.pack(pady=10)
        
        # Status area
        status_frame = tk.Frame(self.root, bg='#2C3E50')
        status_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            status_frame,
            text="🔍 STATUS:",
            font=('Arial', 10, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        ).pack(side='left')
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready - Monitoring for Arena Drafts",
            font=('Arial', 10),
            fg='#3498DB',
            bg='#2C3E50'
        )
        self.status_label.pack(side='left', padx=10)
        
        # Coordinate status indicator
        self.coord_status_label = tk.Label(
            status_frame,
            text="Auto-Detect Mode",
            font=('Arial', 9),
            fg='#E67E22',
            bg='#2C3E50'
        )
        self.coord_status_label.pack(side='right', padx=10)
        
        # Progress indicator frame (initially hidden)
        self.progress_frame = tk.Frame(self.root, bg='#2C3E50')
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="🔄 Processing...",
            font=('Arial', 10),
            fg='#F39C12',
            bg='#2C3E50'
        )
        self.progress_label.pack(side='left', padx=10)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=300
        )
        self.progress_bar.pack(side='left', padx=10)
        
        # Initially hide the progress frame
        self.progress_frame.pack_forget()
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg='#2C3E50')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶️ START MONITORING",
            command=self.toggle_monitoring,
            bg='#27AE60',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=3
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.screenshot_btn = tk.Button(
            control_frame,
            text="📸 ANALYZE SCREENSHOT",
            command=self.manual_screenshot,
            bg='#3498DB',
            fg='white',
            font=('Arial', 10),
            relief='raised',
            bd=2
        )
        self.screenshot_btn.pack(side='left', padx=5)
        
        # Detection method selector
        self.detection_method = tk.StringVar(value="simple_working")
        detection_methods = [
            ("✅ Simple Working", "simple_working"),
            ("🔄 Hybrid Cascade", "hybrid_cascade"),
            ("🎯 Enhanced Auto", "enhanced_auto"),
            ("📐 Static Scaling", "static_scaling"),
            ("🔍 Contour Detection", "contour_detection"),
            ("⚓ Anchor Detection", "anchor_detection")
        ]
        
        method_menu = tk.OptionMenu(control_frame, self.detection_method, *[v for _, v in detection_methods])
        method_menu.config(bg='#9B59B6', fg='white', font=('Arial', 9))
        method_menu['menu'].config(bg='#8E44AD', fg='white')
        method_menu.pack(side='left', padx=5)
        
        # Debug controls
        self.debug_enabled = tk.BooleanVar(value=is_debug_enabled())
        debug_check = tk.Checkbutton(
            control_frame,
            text="🐛 DEBUG",
            variable=self.debug_enabled,
            command=self.toggle_debug_mode,
            bg='#2C3E50',
            fg='#ECF0F1',
            selectcolor='#E74C3C',
            font=('Arial', 9)
        )
        debug_check.pack(side='left', padx=5)
        
        # Performance report button
        self.perf_report_btn = tk.Button(
            control_frame,
            text="📊 REPORT",
            command=self.show_performance_report,
            bg='#8E44AD',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        self.perf_report_btn.pack(side='left', padx=5)
        
        # Visual coordinate selection button
        self.coord_select_btn = tk.Button(
            control_frame,
            text="🎯 SELECT CARD REGIONS",
            command=self.open_coordinate_selector,
            bg='#9B59B6',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        self.coord_select_btn.pack(side='left', padx=5)
        
        # Coordinate mode toggle
        self.use_custom_coords = tk.BooleanVar(value=False)
        self.coord_mode_btn = tk.Checkbutton(
            control_frame,
            text="Use Custom Coords",
            variable=self.use_custom_coords,
            command=self.toggle_coordinate_mode,
            bg='#2C3E50',
            fg='#ECF0F1',
            selectcolor='#34495E',
            font=('Arial', 8),
            relief='flat'
        )
        self.coord_mode_btn.pack(side='left', padx=5)
        
        # Ultimate Detection toggle (NEW!)
        self.use_ultimate_detection = tk.BooleanVar(value=False)
        self.ultimate_detection_btn = tk.Checkbutton(
            control_frame,
            text="🚀 Ultimate Detection",
            variable=self.use_ultimate_detection,
            command=self.toggle_ultimate_detection,
            bg='#2C3E50',
            fg='#E74C3C',  # Red color to indicate advanced feature
            selectcolor='#34495E',
            font=('Arial', 8, 'bold'),
            relief='flat'
        )
        # Only show if Ultimate Detection Engine is available
        if self.ultimate_detector:
            self.ultimate_detection_btn.pack(side='left', padx=5)
        
        # Arena Priority toggle for enhanced arena draft detection
        self.use_arena_priority = tk.BooleanVar(value=True)  # Default enabled for arena drafts
        self.arena_priority_btn = tk.Checkbutton(
            control_frame,
            text="🎯 Arena Priority",
            variable=self.use_arena_priority,
            command=self.toggle_arena_priority,
            bg='#2C3E50',
            fg='#F39C12',  # Orange color to indicate arena-specific feature
            selectcolor='#34495E',
            font=('Arial', 8, 'bold'),
            relief='flat'
        )
        # Show if histogram matcher is available and arena database exists
        if (self.histogram_matcher and 
            hasattr(self.histogram_matcher, 'arena_database') and 
            self.histogram_matcher.arena_database):
            self.arena_priority_btn.pack(side='left', padx=5)
        
        # pHash Detection toggle for ultra-fast detection
        self.use_phash_detection = tk.BooleanVar(value=False)  # Default disabled for safety
        self.phash_detection_btn = tk.Checkbutton(
            control_frame,
            text="⚡ pHash Detection",
            variable=self.use_phash_detection,
            command=self.toggle_phash_detection,
            bg='#2C3E50',
            fg='#E67E22',  # Electric orange color to indicate speed
            selectcolor='#34495E',
            font=('Arial', 8, 'bold'),
            relief='flat'
        )
        # Only show if pHash matcher is available
        if self.phash_matcher:
            self.phash_detection_btn.pack(side='left', padx=5)
        
        # Enable custom mode if coordinates were loaded during startup
        if self._enable_custom_mode_on_startup:
            self.use_custom_coords.set(True)
            print("🎯 Custom coordinate mode enabled (coordinates loaded from previous session)")
        
        # Update coordinate status display
        self.update_coordinate_status()
        
        # Log area - reduced height to make room for larger card images
        log_frame = tk.LabelFrame(
            self.root,
            text="📋 LOG OUTPUT",
            font=('Arial', 10, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text_widget = scrolledtext.ScrolledText(
            log_frame,
            height=10,  # Reduced from 15 to make room for larger card images
            bg='#1C1C1C',
            fg='#ECF0F1',
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        self.log_text_widget.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Card images area - made much larger to accommodate bigger images
        card_frame = tk.LabelFrame(
            self.root,
            text="🃏 DETECTED CARDS",
            font=('Arial', 10, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        card_frame.pack(fill='both', expand=False, padx=10, pady=5)
        
        # Create card image display
        self.card_images_frame = tk.Frame(card_frame, bg='#2C3E50')
        self.card_images_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize card image labels with much larger size
        self.card_image_labels = []
        self.card_name_labels = []
        self.card_correct_buttons = []  # Store references to correct buttons
        for i in range(3):
            card_container = tk.Frame(self.card_images_frame, bg='#34495E', relief='raised', bd=3)
            card_container.pack(side='left', padx=15, pady=15, fill='both', expand=True)
            
            # Card name label (larger text)
            name_label = tk.Label(
                card_container,
                text=f"Card {i+1}: Waiting...",
                font=('Arial', 11, 'bold'),
                fg='#ECF0F1',
                bg='#34495E',
                wraplength=200,
                height=2
            )
            name_label.pack(pady=5)
            self.card_name_labels.append(name_label)
            
            # Card image label (much larger for actual card visibility)
            img_label = tk.Label(
                card_container,
                text="No Image",
                bg='#2C3E50',
                fg='#BDC3C7',
                width=40,  # Much larger width for better visibility
                height=30, # Much larger height for better visibility
                relief='sunken',
                bd=2
            )
            img_label.pack(pady=10, padx=10, fill='both', expand=True)
            self.card_image_labels.append(img_label)
            
            # Correct button for manual override
            correct_btn = tk.Button(
                card_container,
                text="Correct...",
                font=('Arial', 8),
                bg='#95A5A6',
                fg='white',
                command=lambda idx=i: self._open_correction_dialog(idx)  # Use lambda to pass the index
            )
            correct_btn.pack(pady=(0, 5))
            self.card_correct_buttons.append(correct_btn)
        
        # Recommendation area
        rec_frame = tk.LabelFrame(
            self.root,
            text="🎯 RECOMMENDATIONS",
            font=('Arial', 10, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        rec_frame.pack(fill='x', padx=10, pady=5)
        
        self.recommendation_text = tk.Text(
            rec_frame,
            height=4,
            bg='#34495E',
            fg='#ECF0F1',
            font=('Arial', 9),
            wrap=tk.WORD
        )
        self.recommendation_text.pack(fill='x', padx=5, pady=5)
        
        # Initial log message
        self.log_text("🎯 Integrated Arena Bot GUI Initialized!")
        self.log_text("✅ Log monitoring system ready")
        self.log_text("✅ Visual detection system ready") 
        self.log_text("✅ AI draft advisor ready")
        self.log_text("\n📋 Instructions:")
        self.log_text("1. Click 'START MONITORING' to begin")
        self.log_text("2. Open Hearthstone and start an Arena draft")
        self.log_text("3. The bot will automatically detect and provide recommendations")
        self.log_text("4. Use 'ANALYZE SCREENSHOT' for manual analysis")
        
        # Show initial recommendations
        self.show_recommendation("Ready for Arena Draft", "Start monitoring and open Hearthstone Arena mode to begin receiving AI recommendations.")
    
    def toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        if self.debug_enabled.get():
            enable_debug()
            self.log_text("🐛 DEBUG MODE ENABLED - Visual debugging and metrics logging active")
        else:
            disable_debug()
            self.log_text("📊 DEBUG MODE DISABLED - Normal operation mode")
    
    def show_performance_report(self):
        """Show performance report window."""
        try:
            report = generate_performance_report()
            self._show_report_window(report)
        except Exception as e:
            self.log_text(f"❌ Failed to generate performance report: {e}")
    
    def _show_report_window(self, report):
        """Display performance report in new window."""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Performance Report")
        report_window.geometry("800x600")
        report_window.configure(bg='#2C3E50')
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            report_window,
            bg='#1C1C1C',
            fg='#ECF0F1',
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Format and display report
        report_text = self._format_performance_report(report)
        text_widget.insert(tk.END, report_text)
        text_widget.configure(state='disabled')
    
    def _format_performance_report(self, report):
        """Format performance report for display."""
        if 'error' in report:
            return f"Error: {report['error']}"
        
        text = "📊 ARENA BOT PERFORMANCE REPORT\n"
        text += "=" * 50 + "\n\n"
        
        # Session summary
        if 'session_summary' in report:
            summary = report['session_summary']
            text += "📈 SESSION SUMMARY:\n"
            text += f"   Total Tests: {summary.get('total_tests', 0)}\n"
            text += f"   Average IoU: {summary.get('average_iou', 0):.3f}\n"
            text += f"   Average Confidence: {summary.get('average_confidence', 0):.3f}\n"
            text += f"   Average Timing: {summary.get('average_timing', 0):.1f}ms\n\n"
        
        # Method comparison
        if 'method_comparison' in report:
            text += "🔬 METHOD COMPARISON:\n"
            for method, stats in report['method_comparison'].items():
                text += f"   {method}:\n"
                text += f"      Tests: {stats['tests']}\n"
                text += f"      Avg IoU: {stats['avg_iou']:.3f}\n"
                text += f"      Avg Confidence: {stats['avg_confidence']:.3f}\n"
                text += f"      Avg Time: {stats['avg_time_ms']:.1f}ms\n"
                text += f"      Grades: {stats['grade_distribution']}\n\n"
        
        # Grade distribution
        if 'grade_distribution' in report:
            text += "🎯 OVERALL GRADE DISTRIBUTION:\n"
            for grade, count in report['grade_distribution'].items():
                text += f"   {grade}: {count} tests\n"
        
        return text

    def log_text(self, text):
        """Add text to the log widget."""
        if hasattr(self, 'log_text_widget'):
            self.log_text_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
            self.log_text_widget.see(tk.END)
        print(text)  # Also print to console
    
    def update_status(self, status):
        """Update the status label."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=status)
    
    def show_recommendation(self, title, recommendation):
        """Show recommendation in the GUI."""
        if hasattr(self, 'recommendation_text'):
            self.recommendation_text.delete('1.0', tk.END)
            self.recommendation_text.insert('1.0', f"{title}\n\n{recommendation}")
    
    def toggle_monitoring(self):
        """Start or stop monitoring."""
        if not self.running:
            self.running = True
            self.start_btn.config(text="⏸️ STOP MONITORING", bg='#E74C3C')
            self.update_status("Monitoring Active")
            
            # Start log monitoring if available
            if self.log_monitor:
                self.log_monitor.start_monitoring()
                self.log_text("✅ Started log monitoring")
            
            self.log_text("🎯 Monitoring started - waiting for Arena drafts...")
        else:
            self.running = False
            self.start_btn.config(text="▶️ START MONITORING", bg='#27AE60')
            self.update_status("Monitoring Stopped")
            
            # Stop log monitoring if available
            if self.log_monitor:
                self.log_monitor.stop_monitoring()
                self.log_text("⏸️ Stopped log monitoring")
    
    def open_coordinate_selector(self):
        """Open visual coordinate selection interface."""
        self.log_text("🎯 Opening visual coordinate selector...")
        
        try:
            # Create coordinate selector window
            coord_window = CoordinateSelector(self)
            coord_window.run()
            
        except Exception as e:
            self.log_text(f"❌ Coordinate selector failed: {e}")
    
    def load_saved_coordinates(self):
        """Load previously saved coordinates from JSON file."""
        try:
            import json
            import os
            
            coord_file = "captured_coordinates.json"
            if os.path.exists(coord_file):
                with open(coord_file, 'r') as f:
                    coord_data = json.load(f)
                
                # Extract coordinates in the format expected by the bot
                if 'card_coordinates' in coord_data:
                    self.custom_coordinates = []
                    for card in coord_data['card_coordinates']:
                        # Convert to (x, y, width, height) format
                        coord_tuple = (card['x'], card['y'], card['width'], card['height'])
                        self.custom_coordinates.append(coord_tuple)
                    
                    print(f"✅ Loaded {len(self.custom_coordinates)} saved coordinates from previous session")
                    print(f"   Screen resolution: {coord_data.get('screen_resolution', 'Unknown')}")
                    for i, coord in enumerate(self.custom_coordinates):
                        print(f"   Card {i+1}: x={coord[0]}, y={coord[1]}, w={coord[2]}, h={coord[3]}")
                    
                    # Auto-enable custom coordinate mode (will be set after GUI init)
                    self._enable_custom_mode_on_startup = True
                    print("🎯 Will auto-enable custom coordinate mode after GUI initialization")
                else:
                    print("⚠️ Invalid coordinate file format")
            else:
                print("ℹ️ No saved coordinates found - use 'SELECT CARD REGIONS' to create custom coordinates")
                
        except Exception as e:
            print(f"⚠️ Failed to load saved coordinates: {e}")
    
    def toggle_coordinate_mode(self):
        """Toggle between automatic and custom coordinate detection."""
        if self.use_custom_coords.get():
            if self.custom_coordinates:
                self.log_text("🎯 Switched to custom coordinate mode")
                self.log_text(f"   Using {len(self.custom_coordinates)} custom regions")
            else:
                self.log_text("⚠️ Custom coordinate mode enabled but no custom coordinates set")
                self.log_text("   Use 'SELECT CARD REGIONS' to define custom coordinates")
        else:
            self.log_text("🔄 Switched to automatic coordinate detection")
            self.log_text("   Will use smart detection + resolution-based fallback")
        
        # Update visual status
        self.update_coordinate_status()
    
    def update_coordinate_status(self):
        """Update the coordinate status indicator."""
        if hasattr(self, 'coord_status_label'):
            if self.use_custom_coords.get() and self.custom_coordinates:
                self.coord_status_label.config(
                    text=f"Custom Mode ({len(self.custom_coordinates)} regions)",
                    fg='#27AE60'  # Green
                )
            elif self.use_custom_coords.get() and not self.custom_coordinates:
                self.coord_status_label.config(
                    text="Custom Mode (No Regions)",
                    fg='#E74C3C'  # Red
                )
            else:
                self.coord_status_label.config(
                    text="Auto-Detect Mode",
                    fg='#E67E22'  # Orange
                )
    
    def apply_new_coordinates(self, coordinates):
        """Apply new coordinates captured from visual selector."""
        try:
            self.log_text("✅ Applying new coordinates to detection system...")
            
            # Validate coordinate regions
            issues, recommendations = self.validate_coordinate_regions(coordinates)
            
            if issues:
                self.log_text("⚠️ Coordinate validation found some issues:")
                for issue in issues:
                    self.log_text(f"   • {issue}")
                self.log_text("💡 Recommendations:")
                for rec in recommendations:
                    self.log_text(f"   • {rec}")
                self.log_text("   Consider re-selecting regions for better detection accuracy.")
            else:
                self.log_text("✅ Coordinate validation passed - regions look good!")
            
            # Store coordinates for use in analysis
            self.custom_coordinates = coordinates
            
            # Enable custom coordinate mode
            self.use_custom_coords.set(True)
            
            # Log the new coordinates
            for i, (x, y, w, h) in enumerate(coordinates):
                self.log_text(f"   Card {i+1}: x={x}, y={y}, width={w}, height={h}")
            
            # Test the new coordinates immediately
            self.test_new_coordinates(coordinates)
            
            self.log_text("✅ Coordinates applied successfully! Custom mode enabled.")
            self.log_text("   Use 'ANALYZE SCREENSHOT' to test or uncheck 'Use Custom Coords' for auto-detect.")
            
            # Update visual status
            self.update_coordinate_status()
            
        except Exception as e:
            self.log_text(f"❌ Failed to apply coordinates: {e}")
    
    def test_new_coordinates(self, coordinates):
        """Test the new coordinates by capturing regions."""
        try:
            from PIL import ImageGrab
            screenshot_pil = ImageGrab.grab()
            screenshot = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
            
            for i, (x, y, w, h) in enumerate(coordinates):
                # Extract card region
                card_region = screenshot[y:y+h, x:x+w]
                test_path = f"test_card_region_{i+1}.png"
                cv2.imwrite(test_path, card_region)
                self.log_text(f"   💾 Test capture saved: {test_path}")
                
                # Update card display in GUI with test region
                test_card_data = {
                    'card_name': f"Test Region {i+1}",
                    'confidence': 1.0,
                    'image_path': test_path
                }
                # Update individual card
                if i < len(self.card_name_labels):
                    self.card_name_labels[i].config(text=f"Card {i+1}: Test Region {i+1}")
                    if i < len(self.card_image_labels):
                        try:
                            img = Image.fromarray(cv2.cvtColor(card_region, cv2.COLOR_BGR2RGB))
                            img = img.resize((400, 280), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            self.card_image_labels[i].config(image=photo, text="")
                            self.card_image_labels[i].image = photo  # Keep a reference
                        except Exception as e:
                            self.parent_bot.log_text(f"⚠️ Image display failed: {e}")
            
        except Exception as e:
            self.log_text(f"⚠️ Test capture failed: {e}")
    
    def validate_coordinate_regions(self, coordinates):
        """Validate if coordinate regions are suitable for card detection."""
        issues = []
        recommendations = []
        
        for i, (x, y, w, h) in enumerate(coordinates):
            card_num = i + 1
            
            # Check region size
            if w < 150 or h < 180:
                issues.append(f"Card {card_num} region too small ({w}×{h}) - recommended minimum 150×180")
                recommendations.append(f"Make Card {card_num} region larger to capture more card detail")
            
            # Check aspect ratio (Hearthstone cards are roughly 2:3 ratio)
            aspect_ratio = w / h if h > 0 else 0
            expected_ratio = 0.67  # ~2:3 ratio
            if abs(aspect_ratio - expected_ratio) > 0.3:
                issues.append(f"Card {card_num} unusual aspect ratio ({aspect_ratio:.2f}) - cards are usually ~0.67")
                recommendations.append(f"Adjust Card {card_num} to capture more card-like proportions")
            
            # Check for very different sizes (inconsistent regions)
            if i > 0:
                prev_area = coordinates[i-1][2] * coordinates[i-1][3]
                current_area = w * h
                size_diff = abs(current_area - prev_area) / max(current_area, prev_area)
                if size_diff > 0.5:  # 50% difference
                    issues.append(f"Card {card_num} size very different from Card {i} - may cause detection issues")
                    recommendations.append(f"Try to make Card {card_num} similar size to other cards")
        
        return issues, recommendations
    
    def manual_screenshot(self):
        """Take and analyze a manual screenshot with non-blocking threading."""
        # Prevent multiple simultaneous analyses
        if self.analysis_in_progress:
            self.log_text("⚠️ Analysis already in progress, please wait...")
            return
            
        self.log_text("📸 Taking screenshot for analysis...")
        
        # Update UI to show analysis is starting
        self.analysis_in_progress = True
        self.screenshot_btn.config(state=tk.DISABLED)
        self.update_status("Analyzing... (First Ultimate Detection run may take several minutes)")
        
        # Show progress indicator
        self.progress_frame.pack(fill='x', padx=10, pady=5)
        self.progress_bar.start(10)  # Start animation with 10ms intervals
        
        # Start the analysis in a background thread
        threading.Thread(target=self._run_analysis_in_thread, daemon=True).start()
        
        # Start checking the queue for results
        self.root.after(100, self._check_for_result)
    
    def _run_analysis_in_thread(self):
        """Worker function that runs the analysis in a background thread."""
        try:
            # Try multiple screenshot methods
            screenshot = None
            
            # Method 1: PIL ImageGrab (Windows native)
            try:
                from PIL import ImageGrab
                screenshot_pil = ImageGrab.grab()
                screenshot = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
                # Note: We can't call log_text from worker thread, so we'll include this in the result
            except Exception as e:
                self.result_queue.put({
                    'success': False, 
                    'error': f'PIL ImageGrab failed: {e}',
                    'log_message': f"⚠️ PIL ImageGrab failed: {e}"
                })
                return
            
            if screenshot is not None:
                # This is the potentially slow part - analyze the screenshot
                result = self.analyze_screenshot_data(screenshot)
                
                # Put the result in the queue for the main thread
                self.result_queue.put({
                    'success': True,
                    'result': result,
                    'log_message': "✅ Screenshot captured with PIL ImageGrab"
                })
            else:
                self.result_queue.put({
                    'success': False,
                    'error': 'Could not take screenshot',
                    'log_message': "❌ Could not take screenshot"
                })
                
        except Exception as e:
            # Put any errors in the queue
            self.result_queue.put({
                'success': False,
                'error': str(e),
                'log_message': f"❌ Screenshot analysis failed: {e}"
            })
    
    def _check_for_result(self):
        """Check the result queue and update the UI when analysis is complete."""
        try:
            # Check if there's a result in the queue (non-blocking)
            queue_result = self.result_queue.get_nowait()
            
            # We have a result! Update the UI on the main thread
            if 'log_message' in queue_result:
                self.log_text(queue_result['log_message'])
            
            if queue_result['success']:
                result = queue_result.get('result')
                if result:
                    self.last_full_analysis_result = result  # Store the entire result
                    self.show_analysis_result(result)
                else:
                    self.log_text("❌ Could not analyze screenshot - no arena interface found")
            else:
                error = queue_result.get('error', 'Unknown error')
                self.log_text(f"❌ Analysis failed: {error}")
            
            # Re-enable the button and reset status
            self.analysis_in_progress = False
            self.screenshot_btn.config(state=tk.NORMAL)
            self.update_status("Analysis complete. Ready for next screenshot.")
            
            # Hide progress indicator
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            
        except:
            # Queue is empty, result not ready yet - check again in 100ms
            self.root.after(100, self._check_for_result)
    
    def _start_background_cache_builder(self):
        """Start background cache building if cache is incomplete."""
        if self.cache_build_in_progress:
            return
            
        try:
            # Check if feature cache exists and is complete
            from arena_bot.detection.feature_cache_manager import FeatureCacheManager
            cache_manager = FeatureCacheManager()
            
            stats = cache_manager.get_cache_stats()
            total_cached = stats['total_cached_cards']
            
            # If cache is mostly empty, start background building
            if total_cached < 100:  # Threshold for "empty enough" cache
                self.log_text(f"🔄 Starting background feature cache build (current: {total_cached} cards)...")
                self.log_text("   This will eliminate Ultimate Detection delays after completion.")
                
                self.cache_build_in_progress = True
                threading.Thread(target=self._build_cache_in_background, daemon=True).start()
            else:
                self.log_text(f"✅ Feature cache ready ({total_cached} cards cached)")
                
        except Exception as e:
            self.log_text(f"⚠️ Background cache check failed: {e}")
    
    def _build_cache_in_background(self):
        """Build feature cache in background thread."""
        try:
            # Import required modules
            from arena_bot.utils.asset_loader import AssetLoader
            from arena_bot.detection.feature_cache_manager import FeatureCacheManager
            from arena_bot.detection.feature_ensemble import PatentFreeFeatureDetector
            
            # Initialize components
            asset_loader = AssetLoader()
            cache_manager = FeatureCacheManager()
            
            # Get available cards (no limit for complete cache)
            available_cards = asset_loader.get_available_cards()  # Full cache for complete Ultimate Detection
            
            # Build cache for ORB first (fastest algorithm)
            detector = PatentFreeFeatureDetector('ORB', use_cache=True)
            
            cached_count = 0
            for i, card_code in enumerate(available_cards):
                try:
                    # Check if already cached
                    if cache_manager.is_cached(card_code, 'ORB'):
                        continue
                    
                    # Load and process card
                    card_image = asset_loader.load_card_image(card_code)
                    if card_image is not None:
                        success = detector.add_card_features(card_code, card_image)
                        if success:
                            cached_count += 1
                    
                    # Progress update every 50 cards
                    if (i + 1) % 50 == 0:
                        progress = (i + 1) / len(available_cards) * 100
                        # Schedule UI update on main thread
                        self.root.after(0, lambda: self.log_text(
                            f"🔄 Background cache: {cached_count} cards cached ({progress:.1f}%)"
                        ))
                    
                except Exception as e:
                    # Continue on individual card failures
                    continue
            
            # Final update
            self.root.after(0, lambda: self.log_text(
                f"✅ Background cache build complete: {cached_count} cards cached"
            ))
            self.root.after(0, lambda: self.log_text(
                "⚡ Ultimate Detection will now load instantly!"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: self.log_text(
                f"❌ Background cache build failed: {e}"
            ))
        finally:
            self.cache_build_in_progress = False
    
    def analyze_screenshot_data(self, screenshot):
        """Analyze screenshot data for draft cards."""
        if screenshot is None:
            return None
        
        height, width = screenshot.shape[:2]
        resolution_str = f"{width}x{height}"
        self.log_text(f"🔍 Analyzing screenshot: {resolution_str}")
        
        # Get ground truth data for validation if debug mode enabled
        ground_truth_boxes = []
        if is_debug_enabled():
            ground_truth_boxes = self.get_ground_truth_boxes(resolution_str)
            if ground_truth_boxes:
                self.log_text(f"🎯 Loaded {len(ground_truth_boxes)} ground truth boxes for validation")
        
        # Debug coordinate mode status
        checkbox_state = self.use_custom_coords.get() if hasattr(self, 'use_custom_coords') else False
        has_coords = self.custom_coordinates is not None and len(self.custom_coordinates) > 0
        self.log_text(f"🔧 Debug: Custom coords checkbox={checkbox_state}, has_coords={has_coords}")
        
        card_regions = None
        detection_method_used = "unknown"
        detection_timing = None
        detection_start_time = time.time()
        
        # Check if custom coordinates are enabled and available first
        if checkbox_state and has_coords:
            card_regions = self.custom_coordinates
            self.log_text("🎯 Using custom coordinates from visual selector:")
            for i, coord in enumerate(card_regions):
                self.log_text(f"   Card {i+1}: x={coord[0]}, y={coord[1]}, w={coord[2]}, h={coord[3]}")
            self.log_text(f"   Total custom regions: {len(card_regions)}")
        
        # Try enhanced smart coordinate detection if custom coords not enabled
        elif self.smart_detector:
            try:
                # Use selected detection method
                detection_method = getattr(self, 'detection_method', None)
                method_value = detection_method.get() if detection_method else "hybrid_cascade"
                
                smart_result = None
                
                # Try last known good coordinates first for speed and stability
                if self.last_known_good_coords and method_value in ["hybrid_cascade", "enhanced_auto"]:
                    self.log_text("   🔄 Trying last known good coordinates first...")
                    if self._validate_cached_coordinates(screenshot):
                        self.log_text("   ✅ Last known good coordinates still valid")
                        smart_result = {
                            'card_positions': self.last_known_good_coords,
                            'detection_method': 'cached_coordinates',
                            'success': True,
                            'confidence': 1.0,
                            'cascade_stage': 'cached'
                        }
                
                # If cached coordinates failed or not available, run detection
                if not smart_result:
                    if method_value == "simple_working":
                        smart_result = self.smart_detector.detect_cards_simple_working(screenshot)
                    elif method_value == "hybrid_cascade":
                        smart_result = self.smart_detector.detect_cards_with_hybrid_cascade(screenshot)
                    elif method_value == "static_scaling":
                        smart_result = self.smart_detector.detect_cards_via_static_scaling(screenshot)
                    elif method_value == "contour_detection":
                        smart_result = self.smart_detector.detect_cards_via_contours(screenshot)
                    elif method_value == "anchor_detection":
                        smart_result = self.smart_detector.detect_cards_via_anchors(screenshot)
                    else:  # enhanced_auto or fallback
                        smart_result = self.smart_detector.detect_cards_automatically(screenshot)
                
                # If selected method fails, try hybrid cascade as fallback
                if not smart_result or not smart_result.get('success'):
                    if method_value != "hybrid_cascade":
                        self.log_text(f"⚠️ {method_value} failed, trying hybrid cascade...")
                        smart_result = self.smart_detector.detect_cards_with_hybrid_cascade(screenshot)
                    # Final fallback to enhanced auto
                    if not smart_result or not smart_result.get('success'):
                        smart_result = self.smart_detector.detect_cards_automatically(screenshot)
                
                if smart_result and smart_result.get('success'):
                    card_positions = smart_result.get('card_positions', [])
                    if card_positions:
                        # Cache successful coordinates for future use
                        if smart_result.get('cascade_stage') != 'cached':  # Don't cache cached coordinates
                            self.last_known_good_coords = card_positions.copy()
                            self.log_text(f"   💾 Cached {len(card_positions)} successful coordinates")
                        
                        card_regions = [(x, y, w, h) for x, y, w, h in card_positions]
                        
                        # Enhanced logging with optimization info
                        detection_method = smart_result.get('detection_method', 'smart_detector')
                        overall_confidence = smart_result.get('confidence', 0.0)
                        card_size_used = smart_result.get('card_size_used', (0, 0))
                        cascade_stage = smart_result.get('cascade_stage', 'unknown')
                        cascade_confidence = smart_result.get('cascade_confidence', 0.0)
                        
                        # Show cascade information if available
                        if cascade_stage != 'unknown':
                            self.log_text(f"🔄 Hybrid Cascade Detection: {len(card_regions)} cards detected")
                            self.log_text(f"   🎯 CASCADE STAGE: {cascade_stage.upper()} (confidence: {cascade_confidence:.3f})")
                            self.log_text(f"   Method: {detection_method}")
                        else:
                            self.log_text(f"🎯 Enhanced Smart Detector: {len(card_regions)} cards detected")
                            self.log_text(f"   Method: {detection_method}")
                        
                        self.log_text(f"   Overall confidence: {overall_confidence:.3f}")
                        self.log_text(f"   Dynamic card size: {card_size_used[0]}×{card_size_used[1]} pixels")
                        
                        # Show method recommendations and optimizations if available
                        if smart_result.get('optimization_available'):
                            method_recs = smart_result.get('method_recommendations', [])
                            stats = smart_result.get('stats', {})
                            
                            self.log_text(f"   Recommended methods: {stats.get('recommended_methods', [])}")
                            self.log_text(f"   pHash-ready regions: {stats.get('phash_ready_regions', 0)}/3")
                            self.log_text(f"   Method confidence: {smart_result.get('method_confidence', 0.0):.3f}")
                            
                            # Store optimization info for detection algorithms to use
                            self.smart_detector_optimizations = smart_result.get('optimized_regions', {})
                            
                            # Record detection timing and method
                            detection_timing = (time.time() - detection_start_time) * 1000  # Convert to ms
                            detection_method_used = smart_result.get('detection_method', method_value)
                            
                            # Create debug visualization if enabled
                            if is_debug_enabled():
                                debug_img = create_debug_visualization(
                                    screenshot,
                                    card_regions,
                                    ground_truth_boxes,
                                    detection_method_used,
                                    timing_ms=detection_timing
                                )
                                
                                # Save debug image
                                debug_path = save_debug_image(debug_img, "detection_analysis", detection_method_used)
                                if debug_path:
                                    self.log_text(f"🐛 Debug image saved: {debug_path}")
                                
                                # Log metrics for analysis
                                metrics_data = log_detection_metrics(
                                    screenshot_file="current_analysis",
                                    resolution=(width, height),
                                    detection_method=detection_method_used,
                                    detected_boxes=card_regions,
                                    ground_truth_boxes=ground_truth_boxes,
                                    detection_time_ms=detection_timing
                                )
                                
                                if metrics_data and 'overall_grade' in metrics_data:
                                    grade = metrics_data['overall_grade']
                                    mean_iou = metrics_data.get('mean_iou', 0.0)
                                    self.log_text(f"📊 Detection Grade: {grade} (IoU: {mean_iou:.3f})")
                        
                    else:
                        self.log_text("⚠️ Smart detector succeeded but no card positions found")
                else:
                    confidence = smart_result.get('confidence', 0.0) if smart_result else 0.0
                    self.log_text(f"⚠️ Smart detector failed (confidence: {confidence:.3f}), falling back to manual coordinates")
            except Exception as e:
                self.log_text(f"⚠️ Smart detector error: {e}")
        
        # Fallback to resolution-based coordinates if no other method worked
        if not card_regions:
            height, width = screenshot.shape[:2]
            self.log_text(f"🔍 Screen resolution: {width}x{height}")
            self.log_text("📐 Using resolution-based coordinate fallback")
            
            if width >= 3440:  # Ultrawide 3440x1440
                # Based on your screenshot, the cards are positioned in the center-right area
                # The arena interface appears to start around x=1000 and cards are roughly:
                # Left card: ~1150, Middle: ~1400, Right: ~1650
                # Cards appear to be around y=75 and roughly 250x350 in size
                card_regions = [
                    (1100, 75, 250, 350),   # Left card (corrected coordinates)
                    (1375, 75, 250, 350),   # Middle card  
                    (1650, 75, 250, 350),   # Right card
                ]
                self.log_text("📐 Using corrected ultrawide (3440x1440) coordinates")
            elif width >= 2560:  # 2K resolution
                card_regions = [
                    (640, 160, 350, 300),   # Left card
                    (1105, 160, 350, 300),  # Middle card  
                    (1570, 160, 350, 300),  # Right card
                ]
                self.log_text("📐 Using 2K (2560x1440) coordinates")
            else:  # Standard 1920x1080
                card_regions = [
                    (410, 120, 300, 250),   # Left card
                    (855, 120, 300, 250),   # Middle card
                    (1300, 120, 300, 250),  # Right card
                ]
                self.log_text("📐 Using standard (1920x1080) coordinates")
        
        detected_cards = []
        all_slots_candidates = []  # Initialize candidates list for all slots
        
        if self.histogram_matcher and self.asset_loader:
            # Try to detect each card using histogram matching
            for i, (x, y, w, h) in enumerate(card_regions):
                self.log_text(f"   Analyzing card {i+1}...")
                self.log_text(f"   Region bounds: x={x}, y={y}, w={w}, h={h}")
                
                # Extract card region
                if (y + h <= screenshot.shape[0] and x + w <= screenshot.shape[1] and 
                    x >= 0 and y >= 0):
                    
                    # Stage 1: Extract coarse region from SmartCoordinateDetector
                    coarse_region = screenshot[y:y+h, x:x+w]
                    self.log_text(f"   Coarse region size: {coarse_region.shape[1]}x{coarse_region.shape[0]} pixels")
                    
                    # Stage 2: Apply CardRefiner for pixel-perfect cropping (removes UI contamination)
                    # First validate that the region is suitable for refinement
                    if self._validate_region_for_refinement(coarse_region, i+1):
                        self.log_text(f"   🔧 Refining card {i+1} with Color-Guided Adaptive Crop...")
                        try:
                            refined_x, refined_y, refined_w, refined_h = CardRefiner.refine_card_region(coarse_region)
                            
                            # Validate refined dimensions before using
                            if refined_w > 50 and refined_h > 50 and refined_w * refined_h > 10000:
                                # Extract the final, clean card region
                                card_region = coarse_region[refined_y:refined_y+refined_h, refined_x:refined_x+refined_w]
                                self.log_text(f"   ✅ Refined region size: {card_region.shape[1]}x{card_region.shape[0]} pixels")
                                self.log_text(f"   📐 Refinement crop: ({refined_x}, {refined_y}) -> {refined_w}×{refined_h}")
                            else:
                                self.log_text(f"   ⚠️ CardRefiner produced tiny region ({refined_w}x{refined_h}), using coarse region")
                                card_region = coarse_region
                        except Exception as e:
                            self.log_text(f"   ⚠️ CardRefiner failed: {e}, using coarse region")
                            card_region = coarse_region
                    else:
                        self.log_text(f"   ⚠️ Coarse region not suitable for refinement, using as-is")
                        card_region = coarse_region
                    
                    # Save card image for visual feedback with full path
                    card_image_path = os.path.abspath(f"debug_card_{i+1}.png")
                    success = cv2.imwrite(card_image_path, card_region)
                    self.log_text(f"   💾 Saved card image: {card_image_path} (success: {success})")
                    
                    # ENHANCED detection using pHash pre-filtering -> Ultimate Detection Engine -> histogram matching
                    session_id = f"gui_detection_{int(time.time())}"
                    
                    # Detection method selection based on toggles
                    use_phash = (hasattr(self, 'use_phash_detection') and 
                               self.use_phash_detection.get() and 
                               self.phash_matcher is not None)
                    
                    use_ultimate = (hasattr(self, 'use_ultimate_detection') and 
                                  self.use_ultimate_detection.get() and 
                                  self.ultimate_detector is not None)
                    
                    best_match = None
                    all_matches = []
                    detection_method = "Unknown"
                    detection_icon = "🔍"
                    
                    # STAGE 1: pHash Pre-filtering (Ultra-fast for clear cards)
                    if use_phash:
                        self.log_text(f"   ⚡ Attempting pHash detection...")
                        
                        # Use optimized region if available from SmartCoordinateDetector
                        detection_region = card_region
                        region_optimized = False
                        
                        if hasattr(self, 'smart_detector_optimizations') and self.smart_detector_optimizations:
                            card_key = f"card_{i+1}"
                            if card_key in self.smart_detector_optimizations:
                                optimizations = self.smart_detector_optimizations[card_key]
                                if "phash" in optimizations:
                                    # Extract optimized region for pHash
                                    opt_x, opt_y, opt_w, opt_h = optimizations["phash"]
                                    detection_region = screenshot[opt_y:opt_y+opt_h, opt_x:opt_x+opt_w]
                                    region_optimized = True
                                    self.log_text(f"      🎯 Using pHash-optimized region: {opt_w}×{opt_h} pixels")
                        
                        # Performance safeguards
                        phash_start_time = time.time()
                        phash_timeout = 2.0 if region_optimized else 1.0  # More time for optimized regions
                        
                        try:
                            # Enhanced validation for pHash-optimized regions
                            if detection_region is None or detection_region.size == 0:
                                self.log_text(f"      ⚠️ pHash: Invalid card region, skipping")
                            elif detection_region.shape[0] < 50 or detection_region.shape[1] < 50:
                                self.log_text(f"      ⚠️ pHash: Card region too small ({detection_region.shape[1]}x{detection_region.shape[0]}), skipping")
                            else:
                                # Attempt pHash detection with timeout protection
                                phash_result = self._run_detection_with_timeout(
                                    lambda: self.phash_matcher.find_best_phash_match(
                                        detection_region, 
                                        confidence_threshold=0.6  # Balanced threshold for pHash reliability
                                    ),
                                    timeout_seconds=2.0,
                                    method_name="pHash"
                                )
                                
                                detection_time = time.time() - phash_start_time
                                
                                # Check for timeout (shouldn't happen with pHash, but safety first)
                                if detection_time > phash_timeout:
                                    self.log_text(f"      ⚠️ pHash detection timeout ({detection_time:.3f}s), falling back")
                                    phash_result = None
                                
                                if phash_result:
                                    # Convert PHashMatch to CardMatch-like format for compatibility
                                    class PHashCardMatch:
                                        def __init__(self, phash_match):
                                            self.card_code = phash_match.card_code
                                            self.confidence = phash_match.confidence
                                            self.distance = phash_match.hamming_distance
                                            self.is_premium = phash_match.is_premium
                                            self.processing_time = phash_match.processing_time
                                    
                                    best_match = PHashCardMatch(phash_result)
                                    all_matches = [best_match]
                                    all_slots_candidates.append(all_matches)  # Store candidates for return value
                                    detection_method = "pHash Pre-filter"
                                    detection_icon = "⚡"
                                    
                                    # Validate result quality
                                    if phash_result.hamming_distance <= 15:  # Good match
                                        self.log_text(f"      ⚡ pHash match found! ({detection_time*1000:.1f}ms)")
                                        self.log_text(f"      📊 Hamming distance: {phash_result.hamming_distance}/64")
                                        self.log_text(f"      🎯 Confidence: {phash_result.confidence:.3f}")
                                    else:
                                        # Questionable match, allow fallback
                                        self.log_text(f"      ⚠️ pHash match uncertain (distance: {phash_result.hamming_distance}), allowing fallback")
                                        if phash_result.confidence < 0.8:
                                            best_match = None  # Force fallback for low confidence
                                else:
                                    self.log_text(f"      ⚡ pHash: No confident match found ({detection_time*1000:.1f}ms), falling back...")
                        
                        except ImportError as e:
                            self.log_text(f"      ⚠️ pHash detection unavailable: {e}")
                            self.log_text(f"      Install with: pip install imagehash")
                            # Disable pHash detection to prevent repeated errors
                            self.use_phash_detection.set(False)
                            self.phash_matcher = None
                        
                        except MemoryError as e:
                            self.log_text(f"      ⚠️ pHash detection memory error: {e}")
                            self.log_text(f"      Card region size: {card_region.shape if card_region is not None else 'None'}")
                            # Temporarily disable pHash to prevent memory issues
                            self.use_phash_detection.set(False)
                        
                        except Exception as e:
                            self.log_text(f"      ⚠️ pHash detection error: {e}")
                            detection_time = time.time() - phash_start_time
                            if detection_time > phash_timeout:
                                self.log_text(f"      ⚠️ Error occurred after {detection_time:.3f}s - possible timeout")
                            
                            # Log additional debug info for troubleshooting
                            if hasattr(e, '__class__'):
                                self.log_text(f"      🔧 Error type: {e.__class__.__name__}")
                            
                            # Don't disable pHash for single errors, but log for monitoring
                    
                    # STAGE 2: Ultimate Detection (if pHash failed and enabled)
                    if not best_match and use_ultimate:
                        self.log_text(f"   🚀 Using Ultimate Detection Engine...")
                        
                        
                        # Use Ultimate Detection Engine with timeout protection
                        ultimate_result = self._run_detection_with_timeout(
                            lambda: self.ultimate_detector.detect_card_ultimate(card_region),
                            timeout_seconds=5.0,
                            method_name="Ultimate Detection"
                        )
                        
                        if ultimate_result and ultimate_result.confidence > 0.3:
                            # Convert UltimateDetectionResult to CardMatch-like format
                            class UltimateMatch:
                                def __init__(self, result):
                                    self.card_code = result.card_code
                                    self.confidence = result.confidence
                                    self.distance = result.distance
                                    self.is_premium = result.card_code.endswith('_premium')
                                    # Store ultimate-specific data
                                    self.algorithm_used = result.algorithm_used
                                    self.preprocessing_applied = result.preprocessing_applied
                                    self.template_validated = result.template_validated
                                    self.consensus_level = result.consensus_level
                                    self.processing_time = result.processing_time
                            
                            best_match = UltimateMatch(ultimate_result)
                            
                            # Log detailed ultimate detection info
                            self.log_text(f"      🎯 Algorithm: {ultimate_result.algorithm_used}")
                            self.log_text(f"      🔧 Preprocessing: {ultimate_result.preprocessing_applied}")
                            self.log_text(f"      ✅ Template validated: {ultimate_result.template_validated}")
                            self.log_text(f"      👥 Consensus level: {ultimate_result.consensus_level}")
                            self.log_text(f"      ⏱️ Processing time: {ultimate_result.processing_time:.3f}s")
                            
                            # Create mock all_matches for compatibility
                            all_matches = [best_match]
                            all_slots_candidates.append(all_matches)  # Store candidates for return value
                            detection_method = "Ultimate Detection"
                            detection_icon = "🚀"
                        else:
                            self.log_text(f"      ⚠️ Ultimate detection confidence too low: {ultimate_result.confidence:.3f}")
                    
                    # STAGE 3: Histogram Matching (if pHash and Ultimate both failed)
                    if not best_match:
                        # Determine which histogram matcher to use
                        prefer_arena = self.in_draft and hasattr(self, 'use_arena_priority') and self.use_arena_priority.get()
                        active_histogram_matcher = self.histogram_matcher  # Default to the main one
                        
                        if prefer_arena:
                            self.log_text(f"   🎯 Arena Priority enabled. Creating focused database...")
                            try:
                                from arena_bot.detection.histogram_matcher import HistogramMatcher
                                from arena_bot.data.arena_card_database import get_arena_card_database
                                
                                # 1. Get arena histograms directly from arena database (FIXED ARCHITECTURE)
                                arena_database = get_arena_card_database()
                                arena_histograms = arena_database.get_arena_histograms()
                                
                                if arena_histograms:
                                    # 2. Create a NEW, temporary matcher with just the arena-eligible histograms
                                    focused_matcher = HistogramMatcher()
                                    focused_matcher.load_card_database_from_histograms(arena_histograms)
                                    
                                    # 3. Use this temporary matcher for the analysis
                                    active_histogram_matcher = focused_matcher
                                    self.log_text(f"   ✅ Focused matcher created with {len(arena_histograms)} arena histograms.")
                                else:
                                    self.log_text(f"   ⚠️ No arena histograms available, using basic matcher.")
                                
                            except Exception as e:
                                self.log_text(f"   ⚠️ Failed to create focused arena database: {e}")
                                self.log_text("   Falling back to basic histogram matcher.")
                        else:
                            self.log_text(f"   📊 Using basic histogram matching...")
                        
                        # Use the selected matcher (focused or basic)
                        query_hist = active_histogram_matcher.compute_histogram(card_region)
                        if query_hist is not None:
                            all_matches = active_histogram_matcher.find_best_matches(query_hist, max_candidates=10)  # Get top 10 for manual correction
                            all_slots_candidates.append(all_matches)  # Store candidates for return value
                            best_match = all_matches[0] if all_matches else None
                            
                            # Store top candidates for manual correction
                            if all_matches:
                                self.last_analysis_candidates[i] = all_matches[:10]  # Store top 10 candidates
                            else:
                                self.last_analysis_candidates[i] = []
                            if best_match:
                                detection_method = "Arena-Priority Histogram" if prefer_arena else "Basic Histogram"
                                detection_icon = "🎯" if prefer_arena else "📊"
                                
                                # Check for ambiguous matches that need disambiguation
                                is_ambiguous = False
                                if best_match and len(all_matches) > 1:
                                    # Condition 1: Low confidence in the top match
                                    if best_match.confidence < 0.6:
                                        is_ambiguous = True
                                        self.log_text("   ⚠️ Low confidence match detected. Triggering advanced verification.")
                                    
                                    # Condition 2: Top candidates are too close in score (the Artanis vs. Stegodon case)
                                    second_match = all_matches[1]
                                    score_difference = abs(best_match.distance - second_match.distance)
                                    if score_difference < 0.05:  # If scores are within 5% of each other
                                        is_ambiguous = True
                                        self.log_text(f"   ⚠️ Ambiguous match found! Top 2 scores are very close (diff: {score_difference:.4f}).")
                                        self.log_text(f"      1. {self.get_card_name(best_match.card_code)} ({best_match.confidence:.3f})")
                                        self.log_text(f"      2. {self.get_card_name(second_match.card_code)} ({second_match.confidence:.3f})")
                                
                                if is_ambiguous:
                                    # If the match is ambiguous, call the new feature matching fallback
                                    self.log_text("   🔬 Using Feature Matching to resolve ambiguity...")
                                    disambiguated_match = self._resolve_ambiguity_with_features(card_region, all_matches[:5])  # Verify top 5 candidates
                                    if disambiguated_match:
                                        best_match = disambiguated_match
                                        self.log_text(f"   ✅ Feature Matching selected: {self.get_card_name(best_match.card_code)}")
                                        detection_method = "Feature Matching Fallback"
                                        detection_icon = "🔬"
                        else:
                            best_match = None
                            all_matches = []
                    
                    if best_match:
                        card_name = self.get_card_name(best_match.card_code)
                        
                        # Detection method and icon already set in each detection stage
                        
                        # Check arena eligibility for display
                        arena_eligible = False
                        if (self.histogram_matcher and 
                            hasattr(self.histogram_matcher, 'arena_database') and 
                            self.histogram_matcher.arena_database):
                            arena_eligible = self.histogram_matcher.arena_database.is_card_arena_eligible(best_match.card_code)
                        
                        # Format card name with arena indicator
                        arena_indicator = " 🏟️" if arena_eligible else ""
                        formatted_card_name = f"{card_name}{arena_indicator}"
                        
                        self.log_text(f"   {detection_icon} {detection_method} result for card {i+1}:")
                        self.log_text(f"      ✅ Best: {formatted_card_name} (conf: {best_match.confidence:.3f})")
                        
                        # Show top 3 for comparison
                        if len(all_matches) > 1:
                            self.log_text(f"   🔍 Top alternatives:")
                            for j, match in enumerate(all_matches[:3]):
                                match_name = self.get_card_name(match.card_code)
                                
                                # Check arena eligibility for alternatives
                                alt_arena_eligible = False
                                if (self.histogram_matcher and 
                                    hasattr(self.histogram_matcher, 'arena_database') and 
                                    self.histogram_matcher.arena_database):
                                    alt_arena_eligible = self.histogram_matcher.arena_database.is_card_arena_eligible(match.card_code)
                                
                                alt_arena_indicator = " 🏟️" if alt_arena_eligible else ""
                                formatted_alt_name = f"{match_name}{alt_arena_indicator}"
                                
                                self.log_text(f"      {j+1}. {formatted_alt_name} ({match.confidence:.3f})")
                        
                        # Keep template validation for mana cost and rarity
                        final_confidence = best_match.confidence
                        
                        if self.validation_engine and self.template_matcher:
                            self.log_text(f"      🔍 Running template validation...")
                            try:
                                # Extract mana cost region (top-left corner)
                                mana_h, mana_w = min(30, h//4), min(30, w//4)
                                mana_region = card_region[0:mana_h, 0:mana_w] if mana_h > 0 and mana_w > 0 else None
                                
                                # Get expected card data for validation (skip - method doesn't exist)
                                card_data = None
                                expected_mana = card_data.get('cost') if card_data else None
                                expected_rarity = card_data.get('rarity') if card_data else None
                                
                                # Pass OpenCV images directly to validation engine
                                validation_result = self.validation_engine.validate_card_detection(
                                    best_match, 
                                    mana_region=mana_region,
                                    expected_mana=expected_mana,
                                    expected_rarity=expected_rarity
                                )
                                
                                if validation_result.is_valid:
                                    final_confidence = validation_result.confidence
                                    self.log_text(f"      ✅ Validation passed (conf: {validation_result.confidence:.3f})")
                                    if validation_result.mana_cost is not None:
                                        self.log_text(f"      💎 Detected mana: {validation_result.mana_cost}")
                                else:
                                    final_confidence = validation_result.confidence * 0.8  # Reduce confidence for failed validation
                                    self.log_text(f"      ⚠️ Validation failed (conf: {validation_result.confidence:.3f})")
                                    
                            except Exception as e:
                                self.log_text(f"      ⚠️ Validation error: {e}")
                        else:
                            if not self.template_matcher:
                                self.log_text(f"      ℹ️ Template validation skipped (no templates)")
                            else:
                                self.log_text(f"      ℹ️ Template validation skipped (no validation engine)")
                        
                        # NEW: Convert the custom match object (UltimateMatch, PHashCardMatch, etc.)
                        # into a standard dictionary before it goes anywhere else.
                        standardized_match_data = {
                            'position': i + 1,
                            'card_code': best_match.card_code,
                            'card_name': card_name,
                            'confidence': final_confidence,
                            'distance': getattr(best_match, 'distance', 1.0 - best_match.confidence), # Handle different match types
                            'is_premium': getattr(best_match, 'is_premium', '_premium' in best_match.card_code),
                            'region': (x, y, w, h),
                            'image_path': card_image_path,
                            'enhanced_path': None,
                            'detection_method': detection_method, # The final method used
                            'basic_metrics': {
                                'distance': getattr(best_match, 'distance', 1.0 - best_match.confidence),
                            },
                            'validation_result': None
                        }
                        detected_cards.append(standardized_match_data)
                        
                        # Simple logging
                        self.log_text(f"   🃏 Card {i+1}: {card_name}")
                        self.log_text(f"      📊 Final confidence: {final_confidence:.3f} | Distance: {best_match.distance:.3f}")
                    else:
                        # No match found
                        all_slots_candidates.append([])  # Store empty list for no matches
                        self.log_text(f"   ⚠️ No confident match found for card {i+1}")
                        detected_cards.append({
                            'position': i + 1,
                            'card_code': 'Unknown',
                            'card_name': '❌ Detection Failed',
                            'confidence': 0.0,
                            'region': (x, y, w, h),
                            'image_path': card_image_path
                        })
        
        if detected_cards:
            # Get AI recommendation if advisor is available
            recommendation = None
            if self.advisor and len(detected_cards) >= 3:
                try:
                    card_codes = [card['card_code'] for card in detected_cards if card['card_code'] != 'Unknown']
                    if len(card_codes) >= 3:
                        choice = self.advisor.analyze_draft_choice(card_codes[:3], 'warrior')  # Default class
                        recommendation = {
                            'recommended_pick': choice.recommended_pick + 1,
                            'recommended_card': choice.cards[choice.recommended_pick].card_code,
                            'reasoning': choice.reasoning,
                            'card_details': [
                                {
                                    'card_code': card.card_code,
                                    'tier': card.tier_letter,
                                    'tier_score': card.tier_score,
                                    'win_rate': card.win_rate,
                                    'notes': card.notes
                                }
                                for card in choice.cards
                            ]
                        }
                except Exception as e:
                    self.log_text(f"⚠️ AI recommendation failed: {e}")
            
            return {
                'detected_cards': detected_cards,
                'recommendation': recommendation,
                'candidate_lists': all_slots_candidates  # Store candidate lists for manual correction
            }
        
        return {
            'detected_cards': [],
            'recommendation': None,
            'candidate_lists': []
        }
    
    def update_card_images(self, detected_cards):
        """Update the card images in the GUI."""
        for i in range(3):
            if i < len(detected_cards):
                card = detected_cards[i]
                
                # Update card name
                confidence_text = f" ({card['confidence']:.3f})" if card['confidence'] > 0 else ""
                self.card_name_labels[i].config(text=f"Card {i+1}: {card['card_name']}{confidence_text}")
                
                # Update card image with confidence threshold checks
                confidence_threshold = 0.15  # Minimum confidence for displaying card images
                
                if (card['confidence'] >= confidence_threshold and 
                    'image_path' in card and os.path.exists(card['image_path']) and 
                    card['card_name'] != "Unknown"):
                    try:
                        # Load and resize image to much larger size for better visibility
                        img = Image.open(card['image_path'])
                        # Make image significantly larger - actual card-like size
                        img = img.resize((400, 280), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        
                        # Update image label
                        self.card_image_labels[i].config(image=photo, text="")
                        self.card_image_labels[i].image = photo  # Keep a reference
                        self.log_text(f"   📷 Loaded card image: {card['image_path']}")
                    except Exception as e:
                        error_msg = f"Image Error:\n{str(e)[:50]}"
                        self.card_image_labels[i].config(image="", text=error_msg)
                        self.log_text(f"   ❌ Image load error: {e}")
                else:
                    # Display "Detection Failed" placeholder for low confidence results
                    if card['confidence'] < confidence_threshold:
                        placeholder_text = f"Detection Failed\nLow Confidence\n({card['confidence']:.3f})"
                        self.card_image_labels[i].config(image="", text=placeholder_text)
                        self.log_text(f"   ⚠️ Card {i+1} confidence too low for display: {card['confidence']:.3f}")
                    elif card['card_name'] == "Unknown":
                        placeholder_text = "Detection Failed\nUnknown Card"
                        self.card_image_labels[i].config(image="", text=placeholder_text)
                        self.log_text(f"   ⚠️ Card {i+1} unknown card, not displaying")
                    else:
                        self.card_image_labels[i].config(image="", text="No Image\nFound")
                        if 'image_path' in card:
                            self.log_text(f"   ⚠️ Image file missing: {card.get('image_path', 'No path')}")
                        else:
                            self.log_text(f"   ⚠️ No image path in card data")
            else:
                # Clear unused card slots
                self.card_name_labels[i].config(text=f"Card {i+1}: Waiting...")
                self.card_image_labels[i].config(image="", text="No Image")
    
    def show_analysis_result(self, result):
        """Show analysis result in GUI."""
        detected_cards = result['detected_cards']
        recommendation = result['recommendation']
        
        # Sort detected cards by x-coordinate to ensure correct UI positioning (left to right)
        detected_cards_sorted = sorted(detected_cards, key=lambda c: c['region'][0] if 'region' in c else 0)
        self.log_text(f"   🔄 Sorted {len(detected_cards_sorted)} cards by screen position (left to right)")
        
        # Update card images in GUI with sorted cards
        self.update_card_images(detected_cards_sorted)
        
        self.log_text(f"\n✅ Detected {len(detected_cards_sorted)} cards with enhanced analysis:")
        for card in detected_cards_sorted:
            # Enhanced display with quality and strategy info
            strategy = card.get('enhanced_metrics', {}).get('detection_strategy', 'unknown')
            quality = card.get('quality_assessment', {}).get('quality_score', 0.0)
            composite = card.get('enhanced_metrics', {}).get('composite_score', 0.0)
            
            self.log_text(f"   {card['position']}. {card['card_name']} (conf: {card['confidence']:.3f})")
            self.log_text(f"      🎯 Strategy: {strategy} | Quality: {quality:.2f} | Composite: {composite:.3f}")
            
            # Show quality issues if any
            quality_issues = card.get('quality_assessment', {}).get('quality_issues', [])
            if quality_issues:
                self.log_text(f"      ⚠️ Issues: {', '.join(quality_issues[:3])}")  # Show first 3 issues
        
        if recommendation:
            rec_card_name = self.get_card_name(recommendation['recommended_card'])
            rec_text = f"🎯 RECOMMENDED PICK: {rec_card_name}\n\n"
            rec_text += f"📊 Position: #{recommendation['recommended_pick']}\n\n"
            rec_text += f"💭 Reasoning: {recommendation['reasoning']}\n\n"
            rec_text += "📋 All Cards:\n"
            
            for i, card_detail in enumerate(recommendation['card_details']):
                card_name = self.get_card_name(card_detail['card_code'])
                marker = "👑" if i == recommendation['recommended_pick'] - 1 else "📋"
                # Use 'tier_letter' and 'win_rate' from the card_detail dictionary
                rec_text += f"{marker} {i+1}. {card_name} (Tier {card_detail['tier_letter']}, {card_detail['win_rate']:.0%} WR)\n"
            
            self.show_recommendation("AI Draft Recommendation", rec_text)
            self.log_text(f"\n🎯 AI RECOMMENDATION: Pick #{recommendation['recommended_pick']} - {rec_card_name}")
        else:
            self.show_recommendation("Cards Detected", f"Found {len(detected_cards)} cards but no AI recommendation available.")
    
    def run(self):
        """Start the GUI application."""
        if hasattr(self, 'root'):
            try:
                self.root.mainloop()
            except KeyboardInterrupt:
                self.stop()
        else:
            self.log_text("❌ GUI not available, running in command-line mode")
            self.run_command_line()
    
    def stop(self):
        """Stop the bot."""
        self.running = False
        if self.log_monitor:
            self.log_monitor.stop_monitoring()
        if hasattr(self, 'root'):
            self.root.quit()
        print("❌ Arena Bot stopped")
    
    def toggle_ultimate_detection(self):
        """Toggle Ultimate Detection Engine on/off."""
        if not self.ultimate_detector:
            self.log_text("⚠️ Ultimate Detection Engine not available")
            self.use_ultimate_detection.set(False)
            return
        
        if self.use_ultimate_detection.get():
            self.log_text("🚀 Ultimate Detection Engine ENABLED")
            self.log_text("   Enhancement features active:")
            self.log_text("   • Advanced image preprocessing (CLAHE, bilateral filtering, unsharp masking)")
            self.log_text("   • Multi-algorithm ensemble (ORB, BRISK, AKAZE, SIFT)")
            self.log_text("   • Template-enhanced validation with smart filtering")
            self.log_text("   • Intelligent voting with consensus boosting")
            self.log_text("   Expected accuracy: 95-99% (vs 65-70% basic)")
        else:
            self.log_text("📉 Ultimate Detection Engine DISABLED")
            self.log_text("   Using basic histogram matching + template validation")
            self.log_text("   Expected accuracy: 65-70% (proven working system)")
    
    def toggle_arena_priority(self):
        """Toggle Arena Priority detection on/off."""
        if not (self.arena_database and hasattr(self.arena_database, 'get_all_arena_cards')):
            self.log_text("⚠️ Arena Priority not available (arena database missing)")
            self.use_arena_priority.set(False)
            return
        
        if self.use_arena_priority.get():
            self.log_text("🎯 Arena Priority ENABLED")
            self.log_text("   Arena draft optimization active:")
            self.log_text("   • Prioritizes arena-eligible cards in detection results")
            self.log_text("   • Uses HearthArena.com authoritative card data")
            self.log_text("   • Creates focused detection database for faster performance")
            self.log_text("   • Enhanced accuracy for current arena rotation")
        else:
            self.log_text("📉 Arena Priority DISABLED")
            self.log_text("   Using standard card detection without arena prioritization")
    
    def toggle_phash_detection(self):
        """Toggle pHash Detection on/off."""
        if not self.phash_matcher:
            self.log_text("⚠️ pHash Detection not available")
            self.log_text("   Install with: pip install imagehash")
            self.use_phash_detection.set(False)
            return
        
        if self.use_phash_detection.get():
            self.log_text("⚡ pHash Detection ENABLED")
            self.log_text("   Ultra-fast pre-filtering active:")
            self.log_text("   • 100-1000x faster detection for clear card images")
            self.log_text("   • 64-bit DCT perceptual hashes with Hamming distance matching")
            self.log_text("   • Sub-millisecond detection time for matching cards")
            self.log_text("   • Graceful fallback to histogram matching for unclear cards")
            
            # Show pHash statistics if available
            try:
                stats = self.phash_matcher.get_statistics()
                if stats['total_cards'] > 0:
                    self.log_text(f"   ✅ pHash database ready: {stats['total_cards']} cards loaded")
                    if stats['total_lookups'] > 0:
                        self.log_text(f"   📊 Performance: {stats['avg_lookup_time_ms']:.1f}ms avg, "
                                    f"{stats['success_rate']*100:.1f}% success rate")
                else:
                    self.log_text("   ⚠️ pHash database not loaded - will load on first use")
            except Exception as e:
                self.log_text(f"   ℹ️ pHash statistics unavailable: {e}")
        else:
            self.log_text("📉 pHash Detection DISABLED")
            self.log_text("   Using histogram matching as primary detection method")
            self.log_text("   Expected detection time: 50-500ms per card")
    
    def run_command_line(self):
        """Fallback command-line mode."""
        print("\n🔧 Running in command-line mode")
        print("Commands:")
        print("  - 'start': Start monitoring")
        print("  - 'stop': Stop monitoring") 
        print("  - 'screenshot': Analyze current screenshot")
        print("  - 'quit': Exit bot")
        
        try:
            while True:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'start':
                    self.toggle_monitoring()
                elif cmd == 'stop':
                    if self.running:
                        self.toggle_monitoring()
                elif cmd == 'screenshot':
                    self.manual_screenshot()
                elif cmd in ['quit', 'exit']:
                    self.stop()
                    break
                else:
                    print("Unknown command. Use 'start', 'stop', 'screenshot', or 'quit'")
                    
        except KeyboardInterrupt:
            self.stop()


class CoordinateSelector:
    """Visual coordinate selection interface integrated into main bot."""
    
    def __init__(self, parent_bot):
        """Initialize coordinate selector with reference to parent bot."""
        self.parent_bot = parent_bot
        self.window = None
        self.canvas = None
        self.screenshot = None
        self.screenshot_tk = None
        self.rectangles = []
        self.current_rect = None
        self.start_x = None
        self.start_y = None
        self.drawing = False
        self.display_scale = 1.0
    
    def run(self):
        """Run the coordinate selector interface."""
        self.create_window()
        self.take_screenshot()
        
    def create_window(self):
        """Create the coordinate selector window."""
        self.window = tk.Toplevel(self.parent_bot.root)
        self.window.title("🎯 Visual Card Region Selector")
        self.window.geometry("1100x800")
        self.window.configure(bg='#2b2b2b')
        
        # Make it modal
        self.window.transient(self.parent_bot.root)
        self.window.grab_set()
        
        # Title
        title_label = tk.Label(self.window, text="🎯 Select Card Regions", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.window, 
                               text="Draw rectangles around the 3 arena draft cards. Click and drag to select each region.",
                               font=('Arial', 11), fg='#cccccc', bg='#2b2b2b')
        instructions.pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.window, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, pady=10)
        
        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear All", 
                                  command=self.clear_rectangles, font=('Arial', 11),
                                  bg='#FF9800', fg='white', padx=15, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.test_btn = tk.Button(button_frame, text="🧪 Test Regions", 
                                 command=self.test_regions, font=('Arial', 11),
                                 bg='#9C27B0', fg='white', padx=15, pady=5)
        self.test_btn.pack(side=tk.LEFT, padx=10)
        
        self.apply_btn = tk.Button(button_frame, text="✅ Apply Coordinates", 
                                  command=self.apply_coordinates, font=('Arial', 12, 'bold'),
                                  bg='#4CAF50', fg='white', padx=20, pady=5)
        self.apply_btn.pack(side=tk.LEFT, padx=10)
        
        self.cancel_btn = tk.Button(button_frame, text="❌ Cancel", 
                                   command=self.close_window, font=('Arial', 11),
                                   bg='#F44336', fg='white', padx=15, pady=5)
        self.cancel_btn.pack(side=tk.RIGHT, padx=10)
        
        # Canvas frame with scrollbars
        canvas_frame = tk.Frame(self.window, bg='#2b2b2b')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', highlightthickness=0)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for rectangle drawing
        self.canvas.bind("<Button-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_rectangle)
        
        # Status label
        self.status_label = tk.Label(self.window, text="Taking screenshot...", 
                                    font=('Arial', 10), fg='#4CAF50', bg='#2b2b2b')
        self.status_label.pack(pady=5)
    
    def take_screenshot(self):
        """Capture and display screenshot for region selection."""
        try:
            self.window.withdraw()  # Hide window during screenshot
            self.window.after(500, self._capture_and_display)
        except Exception as e:
            self.parent_bot.log_text(f"❌ Screenshot failed: {e}")
            self.close_window()
    
    def _capture_and_display(self):
        """Actually capture and display the screenshot."""
        try:
            from PIL import ImageGrab
            
            # Capture full screen
            self.screenshot = ImageGrab.grab()
            
            # Calculate display size for canvas
            screen_width, screen_height = self.screenshot.size
            max_display_width = 1000
            max_display_height = 600
            
            scale_x = max_display_width / screen_width if screen_width > max_display_width else 1
            scale_y = max_display_height / screen_height if screen_height > max_display_height else 1
            self.display_scale = min(scale_x, scale_y)
            
            # Create display version
            display_width = int(screen_width * self.display_scale)
            display_height = int(screen_height * self.display_scale)
            
            display_screenshot = self.screenshot.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.screenshot_tk = ImageTk.PhotoImage(display_screenshot)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_tk)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            self.status_label.config(text=f"Screenshot ready! Draw rectangles around cards. Resolution: {screen_width}x{screen_height}")
            self.window.deiconify()  # Show window again
            
        except Exception as e:
            self.parent_bot.log_text(f"❌ Screenshot capture failed: {e}")
            self.close_window()
    
    def start_rectangle(self, event):
        """Start drawing a rectangle."""
        if self.screenshot is None:
            return
            
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.drawing = True
    
    def draw_rectangle(self, event):
        """Draw rectangle as user drags."""
        if not self.drawing:
            return
            
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        # Delete current rectangle if it exists
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Draw new rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline='#FF5722', width=3, fill='', stipple='gray50'
        )
    
    def end_rectangle(self, event):
        """Finish drawing rectangle."""
        if not self.drawing:
            return
            
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Ensure rectangle has minimum size
        if abs(end_x - self.start_x) < 10 or abs(end_y - self.start_y) < 10:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.drawing = False
            return
        
        # Calculate actual screen coordinates (scale back up)
        actual_x1 = int(min(self.start_x, end_x) / self.display_scale)
        actual_y1 = int(min(self.start_y, end_y) / self.display_scale)
        actual_x2 = int(max(self.start_x, end_x) / self.display_scale)
        actual_y2 = int(max(self.start_y, end_y) / self.display_scale)
        
        # Store rectangle info
        rect_info = {
            'canvas_rect': self.current_rect,
            'coordinates': (actual_x1, actual_y1, actual_x2 - actual_x1, actual_y2 - actual_y1),
            'display_coords': (self.start_x, self.start_y, end_x, end_y)
        }
        
        self.rectangles.append(rect_info)
        
        # Add rectangle number label
        center_x = (self.start_x + end_x) / 2
        center_y = (self.start_y + end_y) / 2
        text_id = self.canvas.create_text(center_x, center_y, text=str(len(self.rectangles)), 
                                         font=('Arial', 14, 'bold'), fill='#FF5722')
        rect_info['text_id'] = text_id
        
        self.drawing = False
        self.current_rect = None
        
        # Update status
        if len(self.rectangles) == 3:
            self.status_label.config(text="Perfect! 3 card regions selected. Click 'Apply Coordinates' to use them.", fg='#4CAF50')
        else:
            self.status_label.config(text=f"Regions selected: {len(self.rectangles)}/3. Continue drawing around cards.")
    
    def clear_rectangles(self):
        """Clear all drawn rectangles."""
        for rect_info in self.rectangles:
            self.canvas.delete(rect_info['canvas_rect'])
            if 'text_id' in rect_info:
                self.canvas.delete(rect_info['text_id'])
        
        self.rectangles = []
        self.status_label.config(text="Rectangles cleared. Draw new rectangles around cards.")
    
    def test_regions(self):
        """Test the selected regions by showing captured areas."""
        if not self.rectangles:
            messagebox.showwarning("Warning", "Please draw rectangles first!")
            return
        
        try:
            test_coordinates = [rect['coordinates'] for rect in self.rectangles]
            self.parent_bot.test_new_coordinates(test_coordinates)
            self.status_label.config(text="Test captures created! Check the main bot log for results.")
            
        except Exception as e:
            self.parent_bot.log_text(f"❌ Test regions failed: {e}")
    
    def apply_coordinates(self):
        """Apply the selected coordinates to the main bot."""
        if not self.rectangles:
            messagebox.showwarning("Warning", "Please draw rectangles first!")
            return
        
        if len(self.rectangles) != 3:
            result = messagebox.askyesno("Confirm", 
                                       f"You have {len(self.rectangles)} rectangles, but 3 are recommended. Continue anyway?")
            if not result:
                return
        
        try:
            # Extract coordinates
            coordinates = [rect['coordinates'] for rect in self.rectangles]
            
            # Apply to main bot
            self.parent_bot.apply_new_coordinates(coordinates)
            
            # Show success message
            coord_text = "\n".join([f"Card {i+1}: {coord}" for i, coord in enumerate(coordinates)])
            messagebox.showinfo("Success", f"Coordinates applied successfully!\n\n{coord_text}")
            
            self.close_window()
            
        except Exception as e:
            self.parent_bot.log_text(f"❌ Failed to apply coordinates: {e}")
            messagebox.showerror("Error", f"Failed to apply coordinates: {e}")
    
    def close_window(self):
        """Close the coordinate selector window."""
        if self.window:
            self.window.grab_release()
            self.window.destroy()
    
    def toggle_coordinate_mode(self):
        """Toggle coordinate mode between auto-detect and custom."""
        if hasattr(self, 'coord_status_label'):
            self.update_coordinate_status()
    
    def update_coordinate_status(self):
        """Update the coordinate status display."""
        if not hasattr(self, 'coord_status_label'):
            return
            
        checkbox_state = self.use_custom_coords.get() if hasattr(self, 'use_custom_coords') else False
        has_coords = self.custom_coordinates is not None and len(self.custom_coordinates) > 0
        
        if checkbox_state and has_coords:
            self.coord_status_label.config(
                text="🎯 Custom Coordinates Active",
                fg='#27AE60'  # Green for active
            )
        elif has_coords:
            self.coord_status_label.config(
                text="⚙️ Custom Coordinates Available",
                fg='#E67E22'  # Orange for available
            )
        else:
            self.coord_status_label.config(
                text="🔍 Auto-Detect Mode",
                fg='#E67E22'  # Orange for auto
            )


def main():
    """Start the integrated arena bot with GUI."""
    print("🚀 Initializing Integrated Arena Bot GUI...")
    
    try:
        bot = IntegratedArenaBotGUI()
        bot.run()
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()